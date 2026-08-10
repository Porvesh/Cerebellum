#include <chrono>
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

PythonChunkGeneratorOptions options() {
    PythonChunkGeneratorOptions value;
    value.python_executable = CEREBELLUM_TEST_PYTHON;
    value.python_package_path = CEREBELLUM_TEST_PYTHONPATH;
    value.timeout = std::chrono::seconds(5);
    value.seed = 9;
    return value;
}

void test_round_trip_preserves_both_action_spaces() {
    RuntimeConfig config;
    PythonChunkGenerator generator(config, options());
    CHECK(generator.healthy());

    Chunk first;
    const InferenceRequest request{0, -1, config.chunk_size, Stitching::Discard};
    CHECK(generator.generate(request, first));
    CHECK(first.count == config.chunk_size);
    CHECK(first.stamps.obs_seq == 0);
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
    CHECK(second.stamps.obs_seq == 1);
    CHECK(second.model_actions == first.model_actions);  // fixed seed, independent timeline
}

void test_missing_worker_fails_at_startup() {
    RuntimeConfig config;
    auto bad_options = options();
    bad_options.python_executable = "/definitely/not/a/python";
    bad_options.timeout = std::chrono::milliseconds(500);
    bool threw = false;
    try {
        PythonChunkGenerator generator(config, bad_options);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_unimplemented_rtc_is_explicit_and_recoverable() {
    RuntimeConfig config;
    PythonChunkGenerator generator(config, options());
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
    PythonChunkGenerator generator(config, options());
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

}  // namespace

int main() {
    test_round_trip_preserves_both_action_spaces();
    test_missing_worker_fails_at_startup();
    test_unimplemented_rtc_is_explicit_and_recoverable();
    test_runtime_publishes_python_chunk_to_control();

    if (g_failures == 0) {
        std::printf("test_python_chunk_generator: all checks passed\n");
    } else {
        std::printf("test_python_chunk_generator: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
