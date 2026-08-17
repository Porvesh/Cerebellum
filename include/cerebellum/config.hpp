// The numbers the runtime is judged against (spec.md §1, §14).
// Data only — anything that can reject a config lives in validate.hpp.

#pragma once

#include <chrono>
#include <cstddef>

namespace cerebellum {

// --- the contract -----------------------------------------------------------
// Hardcoded so a measurement can't erode them.

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
inline constexpr int steps_for(double ms) {
    if (ms <= 0.0) return 0;
    const double ticks = ms / kControlPeriodMs;
    const int whole = static_cast<int>(ticks);
    return static_cast<double>(whole) < ticks ? whole + 1 : whole;
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

// Tail conserves inference work by draining most of a chunk. Continuous starts
// the next request as soon as the previous result is accepted, trading model
// duty cycle for fresher observations.
enum class RefreshPolicy { Tail, Continuous, Horizon };

struct RtcConfig {
    // Measured SmolVLA takes five 30 Hz ticks. Starting the next inference after
    // one emitted action leaves six committed actions, matching s >= d.
    int execution_horizon = 6;
    int inference_delay = steps_for(kBudgetTargetMs);
    int denoise_steps = kDenoiseSteps;
};

struct RuntimeConfig {
    int chunk_size = kChunkSize;
    int action_dim = kActionDim;
    int padded_action_dim = kPaddedActionDim;
    int denoise_steps = kDenoiseSteps;

    int refresh_trigger = 6;  // >= steps_for(kBudgetTargetMs), < chunk_size
    int queue_capacity = 64;  // >= chunk_size + refresh_trigger

    Stitching stitching = Stitching::Discard;
    RefreshPolicy refresh_policy = RefreshPolicy::Tail;
    RtcConfig rtc{};

    ActionSpace action_space = ActionSpace::AbsolutePosition;  // LeRobot convention
    UnderrunPolicy underrun = UnderrunPolicy::HoldLast;

    double control_period_ms() const { return kControlPeriodMs; }

    // ~1.67 s to drain at H=50 — the staleness its tail inherits (§15.1).
    double chunk_duration_ms() const { return chunk_size * kControlPeriodMs; }

    void validate() const;  // validate.hpp
};

}  // namespace cerebellum
