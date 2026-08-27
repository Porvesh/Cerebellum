// The numbers the runtime is judged against (spec.md §1, §14).
// Data only — anything that can reject a config lives in validate.hpp.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>

namespace cerebellum {

// --- deployment defaults ---------------------------------------------------
// RuntimeConfig may select another policy-native period; 30 Hz remains the
// physical deployment default and the basis of the existing benchmark corpus.

inline constexpr int kControlHz = 30;
// Measured deployment target: two back-to-back ~149 ms SmolVLA windows, one
// control-tick alignment, and a small scheduling margin. Discard still exceeds
// this; continuous refresh/RTC is expected to be judged against it.
inline constexpr double kMaxStalenessMs = 350.0;

// Derived, never a literal: 1e9/30 truncates to 33'333'333 ns, which drifts
// 36 us/hour and doesn't matter.
inline constexpr std::chrono::nanoseconds kControlPeriod{1'000'000'000 / kControlHz};
inline constexpr double kControlPeriodMs = static_cast<double>(kControlPeriod.count()) / 1e6;

// --- §6 budget --------------------------------------------------------------

// A target, not a measurement. Replace when phase 1 step 3 measures it (§15.2).
inline constexpr double kBudgetTargetMs = 146.0;

// Ticks spanned by `ms`, rounded up. Serves both the refresh trigger and RTC's
// inference_delay (§15.5). Hand-rolled: std::ceil isn't constexpr until C++23.
inline constexpr int steps_for(double ms, std::chrono::nanoseconds period = kControlPeriod) {
    if (ms <= 0.0) return 0;
    const double period_ms = static_cast<double>(period.count()) / 1e6;
    if (period_ms <= 0.0) return 0;
    const double ticks = ms / period_ms;
    const int whole = static_cast<int>(ticks);
    return static_cast<double>(whole) < ticks ? whole + 1 : whole;
}

inline constexpr int steps_for_duration(std::chrono::nanoseconds duration,
                                        std::chrono::nanoseconds period = kControlPeriod) {
    if (duration <= std::chrono::nanoseconds::zero() ||
        period <= std::chrono::nanoseconds::zero()) {
        return 0;
    }
    const auto whole = duration.count() / period.count();
    const auto remainder = duration.count() % period.count();
    return static_cast<int>(whole + (remainder != 0 ? 1 : 0));
}

// --- checkpoint shape -------------------------------------------------------

// Compile-time because the steady-state loop allocates nothing (invariant #3);
// reconcile() checks it against what actually loaded.
struct PolicyShape {
    int chunk_size;         // chunk_size / n_action_steps
    int action_dim;         // output_features["action"].shape
    int denoise_steps;      // num_steps
    int padded_action_dim;  // max_action_dim
};

// lerobot/smolvla_base config.json, read 2026-08-04. Pi-0 differs (§15.3).
inline constexpr PolicyShape kSmolVlaBase{
    .chunk_size = 50,
    .action_dim = 6,
    .denoise_steps = 10,
    .padded_action_dim = 32,
};

inline constexpr int kChunkSize = kSmolVlaBase.chunk_size;
inline constexpr int kActionDim = kSmolVlaBase.action_dim;
inline constexpr int kDenoiseSteps = kSmolVlaBase.denoise_steps;
inline constexpr int kPaddedActionDim = kSmolVlaBase.padded_action_dim;
inline constexpr std::size_t kRtcDelayWindow = 10;

// --- policies ---------------------------------------------------------------

// Picks the safe underrun fallback. Zeros mean "stop" only in a relative space;
// against absolute positions they denormalise to the dataset mean pose (§15.4).
enum class ActionSpace { AbsolutePosition, JointVelocity, Delta };

enum class UnderrunPolicy { HoldLast, Zeros, Extrapolate };

enum class Stitching { Discard, Ensemble, Rtc };

// Absolute preserves the model chunk's original control timeline and skips a
// prefix that completed after its intended ticks. FreshStart rebases a newly
// accepted plan to the current tick, matching policies evaluated by executing
// action zero after every replan (for example SmolVLA LIBERO n_action_steps=1).
enum class ChunkAlignment { Absolute, FreshStart };

// Tail conserves inference work by draining most of a chunk. Continuous starts
// the next request as soon as the previous result is accepted, trading model
// duty cycle for fresher observations.
enum class RefreshPolicy { Tail, Continuous, Horizon };

struct RtcConfig {
    // Zero means derive the step count from RuntimeConfig's control period.
    // Positive values are explicit benchmark/deployment overrides.
    int execution_horizon = 0;
    int inference_delay = 0;
    int denoise_steps = kDenoiseSteps;
};

struct RuntimeConfig {
    // Single source of truth for every duration-to-action-step conversion.
    std::chrono::nanoseconds control_period = kControlPeriod;
    double inference_budget_ms = kBudgetTargetMs;
    // Deployment/profile requirement. The 30 Hz base profile retains the
    // original 350 ms contract; slower checkpoints may set an evidence-backed
    // bound without weakening every runtime configuration.
    double max_staleness_ms = kMaxStalenessMs;

    int chunk_size = kChunkSize;
    int action_dim = kActionDim;
    int padded_action_dim = kPaddedActionDim;
    int denoise_steps = kDenoiseSteps;

    // Zero derives ceil(inference_budget / period) plus one scheduling tick.
    int refresh_trigger = 0;
    int queue_capacity = 64;  // >= chunk_size + effective_refresh_trigger()

    Stitching stitching = Stitching::Discard;
    ChunkAlignment chunk_alignment = ChunkAlignment::Absolute;
    RefreshPolicy refresh_policy = RefreshPolicy::Tail;
    RtcConfig rtc{};

    ActionSpace action_space = ActionSpace::AbsolutePosition;  // LeRobot convention
    UnderrunPolicy underrun = UnderrunPolicy::HoldLast;

    double control_period_ms() const {
        return static_cast<double>(control_period.count()) / 1e6;
    }

    int steps_for_ms(double ms) const { return cerebellum::steps_for(ms, control_period); }

    int effective_refresh_trigger() const {
        return refresh_trigger > 0 ? refresh_trigger : steps_for_ms(inference_budget_ms) + 1;
    }

    int effective_rtc_inference_delay() const {
        return rtc.inference_delay > 0 ? rtc.inference_delay : steps_for_ms(inference_budget_ms);
    }

    int effective_rtc_execution_horizon() const {
        if (rtc.execution_horizon > 0) return rtc.execution_horizon;
        // Preserve the default six-action/200 ms horizon at 30 Hz, while a
        // 10 Hz policy derives two actions from the same wall-clock interval.
        constexpr double kDefaultHorizonMs = 6.0 * kControlPeriodMs;
        return std::max(effective_rtc_inference_delay(), steps_for_ms(kDefaultHorizonMs));
    }

    // ~1.67 s at the default 30 Hz; 5 s for a 10 Hz LIBERO policy.
    double chunk_duration_ms() const { return chunk_size * control_period_ms(); }

    void validate() const;  // validate.hpp
};

}  // namespace cerebellum
