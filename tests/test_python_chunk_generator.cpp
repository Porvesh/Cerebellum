#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cerebellum/python_chunk_generator.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

namespace {

std::shared_ptr<const ObservationSnapshot> observation(std::uint8_t first_pixel = 1) {
    auto value = std::make_shared<ObservationSnapshot>();
    value->sequence = 7;
    value->capture_time = now() - std::chrono::milliseconds(2);
    value->state = {1.25F, -2.5F, 3.75F, 4.0F, 5.0F, 6.0F};
    value->task = "pick up the blue block";
    CameraImage image;
    image.feature_name = "observation.images.camera1";
    image.channels = 3;
    image.height = 2;
    image.width = 3;
    image.pixels = {first_pixel, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18};
    value->images.push_back(std::move(image));
    value->validate();
    return value;
}

class StaticObservationSource final : public ObservationSource {
public:
    explicit StaticObservationSource(std::shared_ptr<const ObservationSnapshot> value)
        : value_(std::move(value)) {}

    std::shared_ptr<const ObservationSnapshot> latest() noexcept override { return value_; }
    void publish(std::shared_ptr<const ObservationSnapshot> value) { value_ = std::move(value); }

private:
    std::shared_ptr<const ObservationSnapshot> value_;
};

PythonChunkGeneratorOptions options() {
    PythonChunkGeneratorOptions value;
    value.python_executable = CEREBELLUM_TEST_PYTHON;
    value.python_package_path = CEREBELLUM_TEST_PYTHONPATH;
    value.startup_timeout = std::chrono::seconds(5);
    value.inference_timeout = std::chrono::seconds(5);
    value.seed = 9;
    return value;
}

void test_round_trip_preserves_both_action_spaces() {
    RuntimeConfig config;
    StaticObservationSource source(observation());
    PythonChunkGenerator generator(config, source, options());
    CHECK(generator.healthy());

    Chunk first;
    const InferenceRequest request{0, -1, config.chunk_size, Stitching::Discard};
    CHECK(generator.generate(request, first));
    CHECK(first.count == config.chunk_size);
    CHECK(first.stamps.obs_seq == 7);
    CHECK(first.stamps.t_obs_capture <= now());

    bool padded_tail_has_model_data = false;
    for (int i = 0; i < first.count; ++i) {
        for (int d = 0; d < config.action_dim; ++d) {
            CHECK(first.actions[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] ==
                  first.model_actions[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)]);
        }
        for (int d = config.action_dim; d < config.padded_action_dim; ++d) {
            CHECK(first.actions[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] == 0.0F);
            padded_tail_has_model_data = padded_tail_has_model_data ||
                first.model_actions[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] != 0.0F;
        }
    }
    CHECK(padded_tail_has_model_data);

    Chunk second;
    const InferenceRequest next{50, 49, config.chunk_size, Stitching::Discard};
    CHECK(generator.generate(next, second));
    CHECK(second.stamps.obs_seq == 7);
    CHECK(second.model_actions == first.model_actions);  // fixed seed, independent timeline

    source.publish(observation(/*first_pixel=*/99));
    Chunk changed;
    const InferenceRequest third{100, 99, config.chunk_size, Stitching::Discard};
    CHECK(generator.generate(third, changed));
    CHECK(changed.model_actions != first.model_actions);  // Python consumed changed image bytes
}

void test_missing_worker_fails_at_startup() {
    RuntimeConfig config;
    StaticObservationSource source(observation());
    auto bad_options = options();
    bad_options.python_executable = "/definitely/not/a/python";
    bad_options.startup_timeout = std::chrono::milliseconds(500);
    bool threw = false;
    try {
        PythonChunkGenerator generator(config, source, bad_options);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_unimplemented_rtc_is_explicit_and_recoverable() {
    RuntimeConfig config;
    StaticObservationSource source(observation());
    PythonChunkGenerator generator(config, source, options());
    Chunk chunk;
    CHECK(!generator.generate(
        InferenceRequest{0, -1, config.chunk_size, Stitching::Rtc}, chunk));
    CHECK(generator.healthy());
    CHECK(generator.last_error().find("RTC conditioning") != std::string::npos);
    CHECK(generator.generate(
        InferenceRequest{0, -1, config.chunk_size, Stitching::Discard}, chunk));
}

class RecordingSink final : public ActionSink {
public:
    explicit RecordingSink(std::size_t capacity) { emissions.reserve(capacity); }
    void emit(const ActionEmission& emission) noexcept override { emissions.push_back(emission); }
    std::vector<ActionEmission> emissions;
};

void test_runtime_publishes_python_chunk_to_control() {
    RuntimeConfig config;
    StaticObservationSource source(observation());
    PythonChunkGenerator generator(config, source, options());
    RecordingSink sink(100);
    RuntimeLoop loop(config, generator, sink, 100, std::chrono::microseconds(50));

    loop.run_for(100, std::chrono::milliseconds(1));

    CHECK(loop.inference_stats().generated > 0);
    CHECK(loop.queue().consumer_stats().chunks_accepted > 0);
    bool emitted_python_action = false;
    for (const auto& emission : sink.emissions) {
        emitted_python_action = emitted_python_action || !emission.fallback;
    }
    CHECK(emitted_python_action);
}

void test_missing_observation_does_not_break_worker() {
    RuntimeConfig config;
    StaticObservationSource source(nullptr);
    PythonChunkGenerator generator(config, source, options());
    Chunk chunk;
    const InferenceRequest request{0, -1, config.chunk_size, Stitching::Discard};
    CHECK(!generator.generate(request, chunk));
    CHECK(generator.healthy());
    CHECK(generator.last_error().find("no observation") != std::string::npos);
    source.publish(observation());
    CHECK(generator.generate(request, chunk));
}

std::shared_ptr<const ObservationSnapshot> observation_for(
    const WorkerObservationSchema& schema) {
    auto value = std::make_shared<ObservationSnapshot>();
    value->sequence = 101;
    value->capture_time = now();
    value->state.assign(schema.state_dim, 0.0F);
    value->task = "move the object to the target";
    for (const WorkerImageSpec& expected : schema.images) {
        CameraImage image;
        image.feature_name = expected.feature_name;
        image.channels = expected.channels;
        image.height = expected.height;
        image.width = expected.width;
        image.pixels.resize(static_cast<std::size_t>(image.channels) * image.height * image.width);
        value->images.push_back(std::move(image));
    }
    value->validate();
    return value;
}

void test_real_smolvla_bridge_when_requested() {
    const char* enabled = std::getenv("CEREBELLUM_RUN_SMOLVLA_BRIDGE");
    if (!enabled || std::string(enabled) != "1") return;

    RuntimeConfig config;
    StaticObservationSource source(nullptr);
    auto real_options = options();
    real_options.runner = PythonRunner::SmolVla;
    real_options.device = std::getenv("CEREBELLUM_SMOLVLA_DEVICE")
                              ? std::getenv("CEREBELLUM_SMOLVLA_DEVICE")
                              : "cpu";
    real_options.local_files_only = true;
    real_options.startup_timeout = std::chrono::minutes(15);
    real_options.inference_timeout = std::chrono::minutes(2);

    PythonChunkGenerator generator(config, source, real_options);
    CHECK(generator.observation_schema().state_dim == 6);
    CHECK(generator.observation_schema().images.size() == 3);
    for (const WorkerImageSpec& image : generator.observation_schema().images) {
        CHECK(image.channels == 3);
        CHECK(image.height == 256);
        CHECK(image.width == 256);
    }
    Chunk chunk;
    source.publish(observation());
    CHECK(!generator.generate(
        InferenceRequest{0, -1, config.chunk_size, Stitching::Discard}, chunk));
    CHECK(generator.healthy());
    CHECK(generator.last_error().find("schema") != std::string::npos);

    source.publish(observation_for(generator.observation_schema()));
    CHECK(generator.generate(
        InferenceRequest{0, -1, config.chunk_size, Stitching::Discard}, chunk));
    CHECK(chunk.count == 50);
    CHECK(chunk.stamps.obs_seq == 101);
}

}  // namespace

int main() {
    test_round_trip_preserves_both_action_spaces();
    test_missing_worker_fails_at_startup();
    test_unimplemented_rtc_is_explicit_and_recoverable();
    test_runtime_publishes_python_chunk_to_control();
    test_missing_observation_does_not_break_worker();
    test_real_smolvla_bridge_when_requested();

    if (g_failures == 0) {
        std::printf("test_python_chunk_generator: all checks passed\n");
    } else {
        std::printf("test_python_chunk_generator: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
