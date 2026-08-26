#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cerebellum/replay.hpp"
#include "cerebellum/replay_episode.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_THROWS(statement)                                                   \
    do {                                                                          \
        bool threw = false;                                                        \
        try {                                                                      \
            statement;                                                            \
        } catch (const std::invalid_argument &) {                                  \
            threw = true;                                                          \
        }                                                                          \
        CHECK(threw);                                                              \
    } while (0)

namespace {

ReplayObservationFrame frame(std::uint64_t sequence, Nanos offset,
                             std::uint8_t pixel = 0) {
    ObservationSnapshot observation;
    observation.sequence = sequence;
    observation.state = {static_cast<float>(sequence), 0.0F};
    observation.task = "replay the recorded task";
    observation.images.push_back(CameraImage{"camera", 3, 1, 1, {pixel, pixel, pixel}});
    return ReplayObservationFrame{offset, std::move(observation)};
}

void test_replay_selects_newest_frame_and_rebases_time() {
    ReplayObservationSource source({
        frame(1, Nanos::zero()),
        frame(2, std::chrono::milliseconds(2)),
        frame(3, std::chrono::milliseconds(5)),
    });
    CHECK(!source.latest());
    const TimePoint origin = now();
    source.start(origin);

    CHECK(source.latest_at(origin)->sequence == 1);
    CHECK(source.latest_at(origin + std::chrono::milliseconds(4))->sequence == 2);
    CHECK(source.latest_at(origin + std::chrono::milliseconds(6))->sequence == 3);
    CHECK(source.latest_at(origin + std::chrono::milliseconds(1))->sequence == 3);
    CHECK(source.current_index() == 2);
    CHECK(source.finished(origin + std::chrono::milliseconds(5)));
    CHECK(source.latest_at(origin)->capture_time == origin + std::chrono::milliseconds(5));
}

void test_replay_rejects_an_invalid_timeline() {
    CHECK_THROWS(ReplayObservationSource({frame(1, std::chrono::milliseconds(1))}));
    CHECK_THROWS(ReplayObservationSource(
        {frame(1, Nanos::zero()), frame(2, Nanos::zero())}));
    CHECK_THROWS(ReplayObservationSource(
        {frame(2, Nanos::zero()), frame(1, std::chrono::milliseconds(1))}));
}

class ReplayGenerator final : public ChunkGenerator {
   public:
    ReplayGenerator(ReplayObservationSource &source, int chunk_size)
        : source_(source), chunk_size_(chunk_size) {}

    bool generate(const InferenceRequest &, Chunk &out) noexcept override {
        const auto observation = source_.latest();
        if (!observation) return false;
        seen_sequences.push_back(observation->sequence);
        std::this_thread::sleep_for(std::chrono::microseconds(600));
        out.count = chunk_size_;
        out.stamps.obs_seq = observation->sequence;
        out.stamps.t_obs_capture = observation->capture_time;
        for (int i = 0; i < out.count; ++i) {
            const float value = static_cast<float>(observation->sequence);
            out.actions[static_cast<std::size_t>(i)].fill(value);
            out.model_actions[static_cast<std::size_t>(i)].fill(value);
        }
        return true;
    }

    std::vector<std::uint64_t> seen_sequences;

   private:
    ReplayObservationSource &source_;
    int chunk_size_;
};

void test_replay_runs_through_the_two_loop_runtime() {
    ReplayObservationSource source({
        frame(10, Nanos::zero(), 10),
        frame(11, std::chrono::milliseconds(2), 11),
        frame(12, std::chrono::milliseconds(4), 12),
        frame(13, std::chrono::milliseconds(6), 13),
    });
    RuntimeConfig config;
    config.control_period = std::chrono::milliseconds(1);
    config.inference_budget_ms = 1.0;
    config.chunk_size = 12;
    config.refresh_trigger = 6;
    config.queue_capacity = 18;
    config.refresh_policy = RefreshPolicy::Continuous;
    config.validate();

    ReplayGenerator generator(source, config.chunk_size);
    ReplayActionSink sink(12);
    RuntimeLoop loop(config, generator, sink, 12, std::chrono::microseconds(50));
    source.start();
    loop.run_for(12);

    CHECK(sink.records().size() == 12);
    CHECK(sink.dropped_records() == 0);
    CHECK(generator.seen_sequences.size() >= 4);
    CHECK(generator.seen_sequences.front() >= 10);
    CHECK(generator.seen_sequences.front() <= 13);
    CHECK(generator.seen_sequences.back() == 13);
    for (std::size_t i = 1; i < generator.seen_sequences.size(); ++i) {
        CHECK(generator.seen_sequences[i] >= generator.seen_sequences[i - 1]);
    }
    CHECK(loop.metrics().ticks == 12);

    std::ostringstream csv;
    write_replay_actions_csv(csv, sink.records(), config.action_dim);
    CHECK(csv.str().find("step,chunk_id,chunk_index,chunk_boundary,deadline_offset_ns,"
                         "emit_offset_ns,fallback,observation_age_ns,safety_flags,"
                         "safety_rejected,action_0") == 0);
    CHECK(csv.str().find("\n11,") != std::string::npos);
}

void test_action_smoothness_uses_real_consecutive_actions() {
    std::vector<ReplayActionRecord> records;
    for (int step = 0; step < 4; ++step) {
        ActionEmission emission;
        emission.step = step;
        emission.chunk_id = step < 2 ? 1 : 2;
        emission.chunk_index = step < 2 ? step : step - 2;
        constexpr float values[]{0.0F, 1.0F, 3.0F, 6.0F};
        emission.action[0] = values[step];
        records.push_back(ReplayActionRecord{emission, now()});
    }

    const ActionSmoothnessSummary summary = summarize_action_smoothness(records, 1);
    CHECK(summary.first_difference_linf.count == 3);
    CHECK(summary.first_difference_linf.p50_ns == 2 * ActionSmoothnessSummary::kScale);
    CHECK(summary.second_difference_linf.count == 2);
    CHECK(summary.second_difference_linf.p50_ns == ActionSmoothnessSummary::kScale);
    CHECK(summary.third_difference_linf.count == 1);
    CHECK(summary.third_difference_linf.max_ns == 0);

    std::ostringstream csv;
    write_replay_actions_csv(csv, records, 1);
    CHECK(csv.str().find("\n0,1,0,1,") != std::string::npos);
    CHECK(csv.str().find("\n1,1,1,0,") != std::string::npos);
    CHECK(csv.str().find("\n2,2,0,1,") != std::string::npos);

    records[2].emission.fallback = true;
    const ActionSmoothnessSummary without_fallback = summarize_action_smoothness(records, 1);
    CHECK(without_fallback.first_difference_linf.count == 1);
    CHECK(without_fallback.second_difference_linf.count == 0);
    CHECK(without_fallback.third_difference_linf.count == 0);
}

void test_replay_sink_reports_capacity_exhaustion() {
    ReplayActionSink sink(1);
    sink.emit(ActionEmission{});
    sink.emit(ActionEmission{});
    CHECK(sink.records().size() == 1);
    CHECK(sink.dropped_records() == 1);
}

ReplayEpisode episode() {
    ReplayEpisode value;
    value.task = "replay the recorded task";
    value.state_dim = 2;
    value.action_dim = 2;
    value.frames = {
        frame(20, Nanos::zero(), 20),
        frame(21, std::chrono::milliseconds(33), 21),
        frame(22, std::chrono::milliseconds(66), 22),
    };
    for (int i = 0; i < 3; ++i) {
        Action action{};
        action[0] = static_cast<float>(i);
        action[1] = static_cast<float>(i) + 0.5F;
        value.reference_actions.push_back(action);
    }
    return value;
}

void test_episode_binary_round_trip() {
    const ReplayEpisode original = episode();
    std::stringstream encoded(std::ios::in | std::ios::out | std::ios::binary);
    write_replay_episode(encoded, original);
    encoded.seekg(0);
    const ReplayEpisode decoded = read_replay_episode(encoded);

    CHECK(decoded.task == original.task);
    CHECK(decoded.state_dim == 2);
    CHECK(decoded.action_dim == 2);
    CHECK(decoded.frames.size() == 3);
    CHECK(decoded.frames[1].offset == std::chrono::milliseconds(33));
    CHECK(decoded.frames[2].observation.sequence == 22);
    CHECK(decoded.frames[2].observation.images[0].pixels[0] == 22);
    CHECK(decoded.reference_actions[2][0] == 2.0F);
    CHECK(decoded.reference_actions[2][1] == 2.5F);
}

void test_episode_rejects_corruption_and_schema_changes() {
    std::stringstream bad_magic(std::ios::in | std::ios::out | std::ios::binary);
    bad_magic << "not a replay";
    bad_magic.seekg(0);
    CHECK_THROWS(read_replay_episode(bad_magic));

    ReplayEpisode changed = episode();
    changed.frames[1].observation.images[0].width = 2;
    CHECK_THROWS(changed.validate());

    ReplayEpisode missing_reference = episode();
    missing_reference.reference_actions.pop_back();
    CHECK_THROWS(missing_reference.validate());
}

void test_quality_aligns_absolute_steps_and_excludes_fallbacks() {
    const ReplayEpisode value = episode();
    std::vector<ReplayActionRecord> records;

    ActionEmission predicted;
    predicted.step = 0;
    predicted.action[0] = 1.0F;
    predicted.action[1] = 2.5F;
    records.push_back(ReplayActionRecord{predicted, now()});

    ActionEmission fallback;
    fallback.step = 1;
    fallback.fallback = true;
    records.push_back(ReplayActionRecord{fallback, now()});

    ActionEmission outside;
    outside.step = 9;
    records.push_back(ReplayActionRecord{outside, now()});

    const ReplayQualitySummary quality =
        evaluate_replay_actions(records, value.reference_actions, value.action_dim);
    CHECK(quality.compared_actions == 1);
    CHECK(quality.fallback_actions == 1);
    CHECK(quality.missing_reference_actions == 1);
    CHECK(std::abs(quality.mean_absolute_error - 1.5) < 1e-9);
    CHECK(std::abs(quality.root_mean_square_error - std::sqrt(2.5)) < 1e-9);
    CHECK(std::abs(quality.max_linf_error - 2.0) < 1e-9);
}

}  // namespace

int main() {
    test_replay_selects_newest_frame_and_rebases_time();
    test_replay_rejects_an_invalid_timeline();
    test_replay_runs_through_the_two_loop_runtime();
    test_replay_sink_reports_capacity_exhaustion();
    test_action_smoothness_uses_real_consecutive_actions();
    test_episode_binary_round_trip();
    test_episode_rejects_corruption_and_schema_changes();
    test_quality_aligns_absolute_steps_and_excludes_fallbacks();

    if (g_failures == 0) {
        std::printf("test_replay: all checks passed\n");
    } else {
        std::printf("test_replay: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
