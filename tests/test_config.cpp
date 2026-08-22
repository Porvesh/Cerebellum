// The invariant, as arithmetic (spec.md §1, §15).
//
// This test is written BEFORE config.hpp exists, so the build is red until you
// write it. That is deliberate: the test is the spec made executable, and it is
// the contract you are implementing against.
//
// It also commits to an API. If you want a different shape — validate() in a
// constructor instead of a method, a typed ConfigError like Axon's instead of
// std::invalid_argument, chrono::duration<double,milli> instead of a plain
// double — change it and move this test with it. The assertions about NUMBERS
// are the part that must survive; the spelling is yours.

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "cerebellum/config.hpp"
#include "cerebellum/validate.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Expect `expr` to throw. Used for the validators — a config that cannot work
// must refuse to be used, not limp along and produce a plausible-looking p99.
#define CHECK_THROWS(expr)                                                                \
    do {                                                                                  \
        bool threw = false;                                                               \
        try {                                                                             \
            (void)(expr);                                                                 \
        } catch (const std::exception &) {                                                \
            threw = true;                                                                 \
        }                                                                                 \
        if (!threw) {                                                                     \
            std::printf("  FAIL %s:%d  expected throw: %s\n", __FILE__, __LINE__, #expr); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

// --- the two numbers everything is judged against ---------------------------

static void test_invariant() {
    CHECK(kControlHz == 30);
    CHECK(kMaxStalenessMs == 350.0);

    // Derived from kControlHz, never written as a literal. 1e9/30 is not an
    // integer; 33'333'333 ns drops a third of a nanosecond per tick, which is
    // 36 us over an hour and irrelevant against a 33 ms period. Decided, not
    // discovered.
    CHECK(kControlPeriod == std::chrono::nanoseconds{1'000'000'000 / kControlHz});
    CHECK(kControlPeriod.count() == 33'333'333);

    RuntimeConfig cfg;
    CHECK(cfg.refresh_policy == RefreshPolicy::Tail);
}

// --- facts read out of the checkpoint ---------------------------------------

static void test_checkpoint_facts() {
    CHECK(kChunkSize == 50);        // chunk_size / n_action_steps
    CHECK(kActionDim == 6);         // output_features["action"].shape
    CHECK(kDenoiseSteps == 10);     // num_steps
    CHECK(kPaddedActionDim == 32);  // max_action_dim
    // The model denoises in 32 dims and slices to 6, so RTC's committed prefix
    // must live in the padded space (spec.md §4.5).
    CHECK(kActionDim < kPaddedActionDim);
}

// --- the §15 arithmetic -----------------------------------------------------

static void test_steps_for() {
    // One function serving both the refresh trigger and RTC's inference_delay,
    // because §15.5's point is that they are the same quantity from opposite
    // sides. Rounded UP: a forward spanning 3.5 ticks occupies 4.
    CHECK(steps_for(118.0) == 4);            // §4.4's worked example
    CHECK(steps_for(kBudgetTargetMs) == 5);  // §15.2's correction
    CHECK(steps_for(0.0) == 0);
    CHECK(steps_for(33.0) == 1);
    CHECK(steps_for(34.0) == 2);
    CHECK(steps_for(146.0, std::chrono::milliseconds(100)) == 2);
    CHECK(steps_for_duration(std::chrono::milliseconds(150),
                             std::chrono::milliseconds(100)) == 2);
    CHECK(steps_for_duration(std::chrono::milliseconds(150), kControlPeriod) == 5);
}

static void test_staleness_bound_conflicts_with_chunk_size() {
    // spec.md §15.1, pinned so the contradiction cannot be quietly forgotten.
    //
    // staleness(i) = budget + i*period, so under the §6 budget only seven actions
    // of a chunk fit inside the measured 350 ms bound — against a checkpoint
    // that really does emit 50. Continuous refresh is still required.
    CHECK(max_actions_within_staleness() == 7);
    CHECK(kChunkSize > max_actions_within_staleness());

    // A budget that already exceeds the bound leaves nothing usable at all.
    CHECK(max_actions_within_staleness(351.0) == 0);
}

static void test_horizon_chain() {
    // delay <= horizon <= chunk size.
    check_horizons(3, 5, 50);  // must not throw

    // Below the delay: committed actions left unconstrained, so the seam returns
    // in the one place RTC exists to remove it.
    CHECK_THROWS(check_horizons(5, 3, 50));

    // A committed prefix cannot extend beyond the model chunk.
    CHECK_THROWS(check_horizons(3, 51, 50));
}

// --- validation -------------------------------------------------------------

static void test_defaults_are_valid() {
    RuntimeConfig cfg{};
    cfg.validate();
    CHECK(cfg.chunk_size == 50);
    CHECK(cfg.action_dim == 6);
    CHECK(cfg.stitching == Stitching::Discard);
    CHECK(cfg.control_period_ms() > 33.3 && cfg.control_period_ms() < 33.4);
    CHECK(cfg.effective_refresh_trigger() == 6);
    CHECK(cfg.effective_rtc_inference_delay() == 5);
    CHECK(cfg.effective_rtc_execution_horizon() == 6);
    // A 50-action chunk at 30 Hz takes ~1.67 s to drain — the staleness its tail
    // inherits, and the other half of §15.1.
    CHECK(cfg.chunk_duration_ms() > 1666.0 && cfg.chunk_duration_ms() < 1667.0);
}

static void test_ten_hz_derives_all_step_counts_from_one_period() {
    RuntimeConfig cfg{};
    cfg.control_period = std::chrono::milliseconds(100);
    cfg.validate();

    CHECK(cfg.control_period_ms() == 100.0);
    CHECK(cfg.chunk_duration_ms() == 5000.0);
    CHECK(cfg.steps_for_ms(cfg.inference_budget_ms) == 2);
    CHECK(cfg.effective_refresh_trigger() == 3);
    CHECK(cfg.effective_rtc_inference_delay() == 2);
    CHECK(cfg.effective_rtc_execution_horizon() == 2);
    CHECK(max_actions_within_staleness(cfg.inference_budget_ms, cfg.control_period) == 3);
}

static void test_validation_rejects_unusable_configs() {
    {
        RuntimeConfig cfg{};
        cfg.stitching = Stitching::Rtc;
        cfg.refresh_policy = RefreshPolicy::Horizon;
        cfg.rtc.denoise_steps = 0;
        CHECK_THROWS(cfg.validate());
    }
    {  // R >= H re-infers before the previous chunk produced anything.
        RuntimeConfig cfg{};
        cfg.chunk_size = 10;
        cfg.refresh_trigger = 10;
        CHECK_THROWS(cfg.validate());
    }
    {  // A chunk pushed while R remain has to fit.
        RuntimeConfig cfg{};
        cfg.queue_capacity = 50;  // == H, no room for the R still queued
        CHECK_THROWS(cfg.validate());
    }
    {  // The RTC defaults are now the measured five-tick inference delay plus
        // one executed action: d=5, s=6, R=6.
        RuntimeConfig cfg{};
        cfg.stitching = Stitching::Rtc;
        cfg.refresh_policy = RefreshPolicy::Horizon;
        cfg.validate();

        cfg.rtc.execution_horizon = 4;
        CHECK_THROWS(cfg.validate());
    }
}

int main() {
    test_invariant();
    test_checkpoint_facts();
    test_steps_for();
    test_staleness_bound_conflicts_with_chunk_size();
    test_horizon_chain();
    test_defaults_are_valid();
    test_ten_hz_derives_all_step_counts_from_one_period();
    test_validation_rejects_unusable_configs();

    if (g_failures == 0) {
        std::printf("test_config: all checks passed\n");
    } else {
        std::printf("test_config: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
