#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cerebellum/replay.hpp"

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
    config.chunk_size = 12;
    config.refresh_trigger = 6;
    config.queue_capacity = 18;
    config.refresh_policy = RefreshPolicy::Continuous;
    config.validate();

    ReplayGenerator generator(source, config.chunk_size);
    ReplayActionSink sink(12);
    RuntimeLoop loop(config, generator, sink, 12, std::chrono::microseconds(50));
    source.start();
    loop.run_for(12, std::chrono::milliseconds(1));

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
    CHECK(csv.str().find("step,deadline_offset_ns,emit_offset_ns,fallback,action_0") == 0);
    CHECK(csv.str().find("\n11,") != std::string::npos);
}

void test_replay_sink_reports_capacity_exhaustion() {
    ReplayActionSink sink(1);
    sink.emit(ActionEmission{});
    sink.emit(ActionEmission{});
    CHECK(sink.records().size() == 1);
    CHECK(sink.dropped_records() == 1);
}

}  // namespace

int main() {
    test_replay_selects_newest_frame_and_rebases_time();
    test_replay_rejects_an_invalid_timeline();
    test_replay_runs_through_the_two_loop_runtime();
    test_replay_sink_reports_capacity_exhaustion();

    if (g_failures == 0) {
        std::printf("test_replay: all checks passed\n");
    } else {
        std::printf("test_replay: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
