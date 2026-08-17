// Everything that can reject a config, and the §15 arithmetic it rejects on.

#pragma once

#include <stdexcept>
#include <string>

#include "cerebellum/config.hpp"

namespace cerebellum {

// Actions of a chunk inside the staleness bound, where
// staleness(i) = budget + i * period. Returns 7 under the §6 budget against a
// checkpoint that emits 50 — continuous refresh is required to hold the bound
// (§15.1).
inline constexpr int max_actions_within_staleness(double budget_ms = kBudgetTargetMs) {
    if (budget_ms > kMaxStalenessMs) return 0;
    return static_cast<int>((kMaxStalenessMs - budget_ms) / kControlPeriodMs) + 1;
}

// Without this the kChunkSize constants are a tautology — nothing else in the
// build reads config.json (§15.3).
inline void reconcile(const PolicyShape &expected, const PolicyShape &actual) {
    auto field = [](const char *name, int e, int a) {
        if (e != a) {
            throw std::invalid_argument(std::string("policy shape mismatch: ") + name +
                                        " compiled as " + std::to_string(e) + ", checkpoint says " +
                                        std::to_string(a));
        }
    };
    field("chunk_size", expected.chunk_size, actual.chunk_size);
    field("action_dim", expected.action_dim, actual.action_dim);
    field("denoise_steps", expected.denoise_steps, actual.denoise_steps);
    field("padded_action_dim", expected.padded_action_dim, actual.padded_action_dim);
}

// RTC needs enough committed actions to cover inference, while the horizon must
// fit inside the model chunk. Horizon scheduling starts inference after s-d
// actions; the tail refresh trigger is not part of this inequality.
inline void check_horizons(int inference_delay, int execution_horizon, int chunk_size) {
    if (execution_horizon < inference_delay) {
        throw std::invalid_argument("execution_horizon (" + std::to_string(execution_horizon) +
                                    ") < inference_delay (" + std::to_string(inference_delay) +
                                    "): committed actions left unconstrained, so the seam returns");
    }
    if (execution_horizon > chunk_size) {
        throw std::invalid_argument("execution_horizon (" + std::to_string(execution_horizon) +
                                    ") > chunk_size (" + std::to_string(chunk_size) +
                                    "): committed prefix does not fit in a model chunk");
    }
}

// Not checked here: chunk_duration_ms() against kMaxStalenessMs. Every default
// config violates it, so enforcing it would mean a runtime that can't start or
// a quietly edited bound. Asserted in the test, settled by the §11.2 sweep
// (§15.1).
inline void RuntimeConfig::validate() const {
    if (chunk_size <= 0) throw std::invalid_argument("chunk_size must be positive");
    if (action_dim <= 0) throw std::invalid_argument("action_dim must be positive");
    if (denoise_steps <= 0) throw std::invalid_argument("denoise_steps must be positive");
    if (refresh_trigger < 0) {
        throw std::invalid_argument("refresh_trigger must be non-negative");
    }

    // The model denoises padded and slices down, so RTC's prefix lives there
    // (§4.5).
    if (action_dim > padded_action_dim) {
        throw std::invalid_argument("action_dim (" + std::to_string(action_dim) +
                                    ") > padded_action_dim (" + std::to_string(padded_action_dim) +
                                    ")");
    }

    // R >= H re-infers before the previous chunk produced anything.
    if (refresh_trigger >= chunk_size) {
        throw std::invalid_argument("refresh_trigger (" + std::to_string(refresh_trigger) +
                                    ") >= chunk_size (" + std::to_string(chunk_size) + ")");
    }

    // R must cover observation-to-chunk-ready, not just the forward pass (§15.2).
    const int floor_r = steps_for(kBudgetTargetMs);
    if (refresh_trigger < floor_r) {
        throw std::invalid_argument("refresh_trigger (" + std::to_string(refresh_trigger) +
                                    ") below ceil(budget/period) = " + std::to_string(floor_r) +
                                    ": guarantees an underrun");
    }

    // A chunk pushed while R remain queued has to fit.
    if (queue_capacity < chunk_size + refresh_trigger) {
        throw std::invalid_argument("queue_capacity (" + std::to_string(queue_capacity) +
                                    ") < chunk_size + refresh_trigger (" +
                                    std::to_string(chunk_size + refresh_trigger) + ")");
    }

    // Zeros against absolute positions command motion to the dataset mean
    // (§15.4).
    if (underrun == UnderrunPolicy::Zeros && action_space == ActionSpace::AbsolutePosition) {
        throw std::invalid_argument(
            "UnderrunPolicy::Zeros with ActionSpace::AbsolutePosition: an underrun "
            "would command motion to the dataset mean pose");
    }

    if (stitching == Stitching::Rtc) {
        if (rtc.denoise_steps <= 0) {
            throw std::invalid_argument("RTC denoise_steps must be positive");
        }
        if (refresh_policy != RefreshPolicy::Horizon) {
            throw std::invalid_argument(
                "Stitching::Rtc requires RefreshPolicy::Horizon so inference starts "
                "at s-d");
        }
        check_horizons(rtc.inference_delay, rtc.execution_horizon, chunk_size);
    }
}

}  // namespace cerebellum
