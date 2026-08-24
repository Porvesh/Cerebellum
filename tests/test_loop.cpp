// The complete phase-1 skeleton: one fixed-grid control loop and one irregular
// inference loop, joined only by ActionChunkQueue.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "cerebellum/loop.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

namespace {

class RecordingSink final : public ActionSink {
   public:
    explicit RecordingSink(std::size_t capacity, Nanos delay = Nanos::zero()) : delay_(delay) {
        emissions.reserve(capacity);
    }

    void emit(const ActionEmission &emission) noexcept override {
        emissions.push_back(emission);  // capacity is reserved before control starts
        if (delay_ > Nanos::zero()) std::this_thread::sleep_for(delay_);
    }

    std::vector<ActionEmission> emissions;

   private:
    Nanos delay_;
};

class SyntheticGenerator final : public ChunkGenerator {
   public:
    SyntheticGenerator(int chunk_size, Nanos delay) : chunk_size_(chunk_size), delay_(delay) {}

    bool generate(const InferenceRequest &request, Chunk &out) noexcept override {
        const TimePoint capture = now();
        const std::uint64_t observation = ++calls;
        std::this_thread::sleep_for(delay_);

        out.count = chunk_size_;
        out.stamps.obs_seq = observation;
        out.stamps.t_obs_capture = capture;
        for (int i = 0; i < chunk_size_; ++i) {
            const float value = static_cast<float>(request.first_step + i);
            for (std::size_t d = 0; d < kPaddedActionDim; ++d) {
                out.actions[static_cast<std::size_t>(i)][d] = value;
            }
        }
        return true;
    }

    std::uint64_t calls = 0;

   private:
    int chunk_size_;
    Nanos delay_;
};

class FailingGenerator final : public ChunkGenerator {
   public:
    bool generate(const InferenceRequest &, Chunk &) noexcept override {
        ++calls;
        return false;
    }
    std::uint64_t calls = 0;
};

struct SeenRtcRequest {
    std::int64_t first_step = 0;
    std::int64_t prefix_first_step = 0;
    int prefix_count = 0;
    float first_prefix_value = -1.0F;
};

class RtcRecordingGenerator final : public ChunkGenerator {
   public:
    bool generate(const InferenceRequest &request, Chunk &out) noexcept override {
        seen.push_back(SeenRtcRequest{
            request.first_step,
            request.rtc.prefix_first_step,
            request.rtc.prefix_count,
            request.rtc.present() ? request.rtc.prefix[0][0] : -1.0F,
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        out.count = request.action_count;
        out.stamps.obs_seq = seen.size();
        out.stamps.t_obs_capture = now();
        for (int i = 0; i < out.count; ++i) {
            const float value = static_cast<float>(request.first_step + i);
            out.actions[static_cast<std::size_t>(i)].fill(value);
            out.model_actions[static_cast<std::size_t>(i)].fill(value);
        }
        return true;
    }

    std::vector<SeenRtcRequest> seen;
};

RuntimeConfig small_config(UnderrunPolicy policy = UnderrunPolicy::HoldLast) {
    RuntimeConfig cfg{};
    cfg.control_period = std::chrono::microseconds(500);
    cfg.inference_budget_ms = 0.5;
    cfg.chunk_size = 12;
    cfg.refresh_trigger = 6;
    cfg.queue_capacity = 18;
    cfg.rtc.inference_delay = 1;
    cfg.rtc.execution_horizon = 1;
    cfg.underrun = policy;
    if (policy == UnderrunPolicy::Zeros) {
        cfg.action_space = ActionSpace::JointVelocity;
    }
    cfg.validate();
    return cfg;
}

void seed(RuntimeLoop &loop, int count, float first, float delta) {
    Chunk *chunk = loop.queue().acquire();
    CHECK(chunk != nullptr);
    if (!chunk) return;
    chunk->first_step = 0;
    chunk->count = count;
    chunk->stamps.chunk_id = 999;
    chunk->stamps.obs_seq = 999;
    chunk->stamps.t_obs_capture = now();
    chunk->stamps.t_infer_start = chunk->stamps.t_obs_capture;
    chunk->stamps.t_infer_end = chunk->stamps.t_obs_capture;
    for (int i = 0; i < count; ++i) {
        chunk->actions[static_cast<std::size_t>(i)][0] = first + delta * static_cast<float>(i);
    }
    CHECK(loop.queue().publish(chunk));
}

void test_two_loops_run_concurrently() {
    const RuntimeConfig cfg = small_config();
    SyntheticGenerator generator(cfg.chunk_size, std::chrono::milliseconds(2));
    RecordingSink sink(100);
    RuntimeLoop loop(cfg, generator, sink, 100, std::chrono::microseconds(50));

    loop.run_for(100);

    CHECK(sink.emissions.size() == 100);
    CHECK(sink.emissions[1].deadline - sink.emissions[0].deadline == cfg.control_period);
    CHECK(loop.metrics().ticks == 100);
    CHECK(loop.inference_stats().generated > 3);
    CHECK(loop.inference_stats().generation_wall_ns > 0);
    CHECK(loop.queue().consumer_stats().chunks_accepted > 3);
    CHECK(generator.calls == loop.inference_stats().requests);
    CHECK(loop.inference_stats().invalid_chunks == 0);

    std::size_t real_actions = 0;
    for (const auto &emission : sink.emissions) {
        if (!emission.fallback) ++real_actions;
    }
    CHECK(real_actions > 50);
}

void test_rtc_delay_estimator_keeps_recent_worst_case() {
    RtcDelayEstimator estimator(/*configured_floor=*/5);
    CHECK(estimator.conservative_steps() == 5);
    estimator.observe(3);
    CHECK(estimator.conservative_steps() == 5);
    estimator.observe(8);
    CHECK(estimator.conservative_steps() == 8);

    // Ten newer four-step samples evict the old eight-step outlier, but the
    // configured five-step safety floor still applies.
    for (std::size_t i = 0; i < kRtcDelayWindow; ++i) estimator.observe(4);
    CHECK(estimator.conservative_steps() == 5);
}

void test_rtc_retains_and_aligns_model_prefix_on_worker() {
    RuntimeConfig cfg = small_config();
    cfg.stitching = Stitching::Rtc;
    cfg.refresh_policy = RefreshPolicy::Horizon;
    cfg.validate();
    RtcRecordingGenerator generator;
    RecordingSink sink(80);
    RuntimeLoop loop(cfg, generator, sink, 80, std::chrono::microseconds(50));

    loop.run_for(80);

    CHECK(generator.seen.size() >= 2);
    if (generator.seen.size() >= 2) {
        const SeenRtcRequest &conditioned = generator.seen[1];
        CHECK(conditioned.prefix_count > 0);
        CHECK(conditioned.prefix_first_step == conditioned.first_step);
        CHECK(conditioned.first_prefix_value == static_cast<float>(conditioned.first_step));
    }
}

std::vector<ActionEmission> run_fallback(UnderrunPolicy policy) {
    RuntimeConfig cfg = small_config(policy);
    FailingGenerator generator;
    RecordingSink sink(4);
    RuntimeLoop loop(cfg, generator, sink, 4, std::chrono::microseconds(25));
    seed(loop, /*count=*/2, /*first=*/1.0f, /*delta=*/2.0f);
    loop.run_for(4);
    CHECK(loop.metrics().underruns == 2);
    CHECK(generator.calls > 0);
    return sink.emissions;
}

void test_underrun_policies() {
    const auto held = run_fallback(UnderrunPolicy::HoldLast);
    CHECK(held.size() == 4);
    CHECK(held[0].action[0] == 1.0f);
    CHECK(held[1].action[0] == 3.0f);
    CHECK(held[2].fallback && held[2].action[0] == 3.0f);
    CHECK(held[3].fallback && held[3].action[0] == 3.0f);

    const auto zeroed = run_fallback(UnderrunPolicy::Zeros);
    CHECK(zeroed[2].fallback && zeroed[2].action[0] == 0.0f);
    CHECK(zeroed[3].fallback && zeroed[3].action[0] == 0.0f);

    const auto extrapolated = run_fallback(UnderrunPolicy::Extrapolate);
    CHECK(extrapolated[2].fallback && extrapolated[2].action[0] == 5.0f);
    CHECK(extrapolated[3].fallback && extrapolated[3].action[0] == 7.0f);
}

void test_slow_control_skips_expired_steps() {
    RuntimeConfig cfg = small_config();
    FailingGenerator generator;
    RecordingSink sink(5, std::chrono::milliseconds(2));
    RuntimeLoop loop(cfg, generator, sink, 5, std::chrono::microseconds(25));
    seed(loop, /*count=*/12, /*first=*/0.0f, /*delta=*/1.0f);

    loop.run_for(5);

    CHECK(sink.emissions.size() == 5);
    CHECK(sink.emissions.back().step > 4);  // grid advanced farther than call count
    CHECK(loop.queue().consumer_stats().steps_skipped > 0);
    for (const auto &emission : sink.emissions) {
        if (!emission.fallback) {
            CHECK(emission.action[0] == static_cast<float>(emission.step));
        }
    }
}

void test_synchronous_control_preserves_every_step() {
    RuntimeConfig cfg = small_config();
    FailingGenerator generator;
    RecordingSink sink(5, std::chrono::milliseconds(2));
    RuntimeLoop loop(cfg, generator, sink, 5, std::chrono::microseconds(25));
    seed(loop, /*count=*/12, /*first=*/0.0f, /*delta=*/1.0f);

    loop.run_for(5, ControlPacing::SynchronousEvaluation);

    CHECK(sink.emissions.size() == 5);
    CHECK(loop.queue().consumer_stats().steps_skipped == 0);
    for (std::size_t i = 0; i < sink.emissions.size(); ++i) {
        CHECK(sink.emissions[i].step == static_cast<std::int64_t>(i));
        CHECK(sink.emissions[i].action[0] == static_cast<float>(i));
    }
}

void test_safety_runs_before_emit_and_fallback_history() {
    RuntimeConfig cfg = small_config();
    FailingGenerator generator;
    RecordingSink sink(2);
    SafetyConfig safety_config(cfg.action_dim);
    safety_config.min_action[0] = -1.0F;
    safety_config.max_action[0] = 1.0F;
    ActionSafetyFilter safety(safety_config);
    RuntimeLoop loop(cfg, generator, sink, 2, std::chrono::microseconds(25), &safety);
    seed(loop, /*count=*/1, /*first=*/10.0F, /*delta=*/0.0F);

    loop.run_for(2);

    CHECK(sink.emissions.size() == 2);
    CHECK(sink.emissions[0].action[0] == 1.0F);
    CHECK((sink.emissions[0].safety_flags & safety_flag(SafetyFlag::ActionClamped)) != 0);
    CHECK(!sink.emissions[0].safety_rejected);
    CHECK(sink.emissions[1].fallback);
    CHECK(sink.emissions[1].action[0] == 1.0F);
}

void test_external_stop_joins_both_loops() {
    const RuntimeConfig cfg = small_config();
    SyntheticGenerator generator(cfg.chunk_size, std::chrono::milliseconds(2));
    RecordingSink sink(100);
    RuntimeLoop loop(cfg, generator, sink, 100, std::chrono::microseconds(25));

    std::thread control(
        [&] { loop.run_for(/*max_ticks=*/1'000'000); });
    while (!loop.running()) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    loop.request_stop();
    control.join();

    CHECK(!loop.running());
    CHECK(loop.metrics().ticks > 0);
    CHECK(loop.metrics().ticks < 1'000'000);
    CHECK(sink.emissions.size() == loop.metrics().ticks);
}

}  // namespace

int main() {
    test_two_loops_run_concurrently();
    test_rtc_delay_estimator_keeps_recent_worst_case();
    test_rtc_retains_and_aligns_model_prefix_on_worker();
    test_underrun_policies();
    test_slow_control_skips_expired_steps();
    test_synchronous_control_preserves_every_step();
    test_safety_runs_before_emit_and_fallback_history();
    test_external_stop_joins_both_loops();

    if (g_failures == 0) {
        std::printf("test_loop: all checks passed\n");
    } else {
        std::printf("test_loop: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
