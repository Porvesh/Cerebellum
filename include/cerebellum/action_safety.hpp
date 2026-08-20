// Allocation-free software guardrails for actions leaving the control loop.
// Physical limits are supplied by the robot adapter; this file only implements
// robot-independent validation and limiting.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cerebellum/chunk_queue.hpp"
#include "cerebellum/timing.hpp"

namespace cerebellum {

enum class SafetyFlag : std::uint32_t {
    None = 0,
    ActionClamped = 1U << 0,
    ActionRateLimited = 1U << 1,
    ActionAccelerationLimited = 1U << 2,
    NonFiniteRejected = 1U << 3,
    StaleRejected = 1U << 4,
};

constexpr std::uint32_t safety_flag(SafetyFlag flag) noexcept {
    return static_cast<std::uint32_t>(flag);
}

struct SafetyConfig {
    int action_dim = 0;
    std::array<float, kPaddedActionDim> min_action{};
    std::array<float, kPaddedActionDim> max_action{};
    std::array<float, kPaddedActionDim> max_action_rate{};          // action units / second
    std::array<float, kPaddedActionDim> max_action_acceleration{};  // action units / second^2
    Action replacement_action{};
    Nanos max_observation_age = Nanos::max();

    explicit SafetyConfig(int dimensions) : action_dim(dimensions) {
        min_action.fill(-std::numeric_limits<float>::infinity());
        max_action.fill(std::numeric_limits<float>::infinity());
        max_action_rate.fill(std::numeric_limits<float>::infinity());
        max_action_acceleration.fill(std::numeric_limits<float>::infinity());
    }

    void validate() const {
        if (action_dim <= 0 || action_dim > kPaddedActionDim) {
            throw std::invalid_argument("safety action_dim is invalid");
        }
        if (max_observation_age <= Nanos::zero()) {
            throw std::invalid_argument("max_observation_age must be positive");
        }
        for (int d = 0; d < action_dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(d);
            if (std::isnan(min_action[i]) || std::isnan(max_action[i]) ||
                min_action[i] > max_action[i]) {
                throw std::invalid_argument("invalid safety action bounds");
            }
            if (!(max_action_rate[i] > 0.0F) || !(max_action_acceleration[i] > 0.0F)) {
                throw std::invalid_argument("safety rate limits must be positive");
            }
            if (!std::isfinite(replacement_action[i]) ||
                replacement_action[i] < min_action[i] || replacement_action[i] > max_action[i]) {
                throw std::invalid_argument("replacement action violates safety bounds");
            }
        }
    }
};

struct SafetyResult {
    Action action{};
    std::uint32_t flags = 0;
    bool rejected = false;

    bool modified() const noexcept { return flags != 0; }
    bool has(SafetyFlag flag) const noexcept { return (flags & safety_flag(flag)) != 0; }
};

struct SafetyStats {
    std::uint64_t checked = 0;
    std::uint64_t modified = 0;
    std::uint64_t rejected = 0;
    std::uint64_t action_clamped = 0;
    std::uint64_t action_rate_limited = 0;
    std::uint64_t action_acceleration_limited = 0;
    std::uint64_t nonfinite_rejected = 0;
    std::uint64_t stale_rejected = 0;
};

class ActionSafetyFilter {
   public:
    explicit ActionSafetyFilter(SafetyConfig config) : config_(config) { config_.validate(); }

    // Robot adapters should seed the filter with the measured starting command
    // or pose. Without a seed, the configured replacement action is the first
    // safe anchor.
    void reset(const Action &safe_action) {
        validate_seed(safe_action);
        last_action_ = safe_action;
        for (std::size_t i = static_cast<std::size_t>(config_.action_dim);
             i < last_action_.size(); ++i) {
            last_action_[i] = 0.0F;
        }
        previous_velocity_.fill(0.0F);
        have_last_ = true;
        have_velocity_ = false;
    }

    SafetyResult apply(const Action &requested, Nanos period, bool has_observation,
                       Nanos observation_age) noexcept {
        SafetyResult result{};
        ++stats_.checked;

        bool finite = true;
        for (int d = 0; d < config_.action_dim; ++d) {
            finite = finite && std::isfinite(requested[static_cast<std::size_t>(d)]);
        }
        if (!finite) {
            result.flags |= safety_flag(SafetyFlag::NonFiniteRejected);
            ++stats_.nonfinite_rejected;
        }
        if (has_observation && observation_age > config_.max_observation_age) {
            result.flags |= safety_flag(SafetyFlag::StaleRejected);
            ++stats_.stale_rejected;
        }

        result.rejected = !finite || (has_observation && observation_age > config_.max_observation_age);
        if (result.rejected) {
            result.action = have_last_ ? last_action_ : config_.replacement_action;
            for (std::size_t i = static_cast<std::size_t>(config_.action_dim);
                 i < result.action.size(); ++i) {
                result.action[i] = 0.0F;
            }
            ++stats_.rejected;
            ++stats_.modified;
            remember(result.action, period);
            return result;
        }

        result.action = requested;
        for (std::size_t i = static_cast<std::size_t>(config_.action_dim);
             i < result.action.size(); ++i) {
            result.action[i] = 0.0F;
        }

        const float seconds = std::chrono::duration<float>(period).count();
        for (int d = 0; d < config_.action_dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(d);
            const float clamped =
                std::clamp(result.action[i], config_.min_action[i], config_.max_action[i]);
            if (clamped != result.action[i]) {
                result.action[i] = clamped;
                result.flags |= safety_flag(SafetyFlag::ActionClamped);
            }

            if (!have_last_ || !(seconds > 0.0F)) continue;
            const float max_delta = config_.max_action_rate[i] * seconds;
            const float velocity_limited =
                std::clamp(result.action[i], last_action_[i] - max_delta, last_action_[i] + max_delta);
            if (velocity_limited != result.action[i]) {
                result.action[i] = velocity_limited;
                result.flags |= safety_flag(SafetyFlag::ActionRateLimited);
            }

            if (!have_velocity_) continue;
            const float requested_velocity = (result.action[i] - last_action_[i]) / seconds;
            const float max_velocity_delta = config_.max_action_acceleration[i] * seconds;
            const float limited_velocity =
                std::clamp(requested_velocity, previous_velocity_[i] - max_velocity_delta,
                           previous_velocity_[i] + max_velocity_delta);
            const float acceleration_limited = last_action_[i] + limited_velocity * seconds;
            if (acceleration_limited != result.action[i]) {
                result.action[i] = acceleration_limited;
                result.flags |= safety_flag(SafetyFlag::ActionAccelerationLimited);
            }

            // Absolute command bounds have priority when rate and acceleration
            // constraints cannot all be satisfied near a boundary.
            const float bounded =
                std::clamp(result.action[i], config_.min_action[i], config_.max_action[i]);
            if (bounded != result.action[i]) {
                result.action[i] = bounded;
                result.flags |= safety_flag(SafetyFlag::ActionClamped);
            }
        }

        if (result.has(SafetyFlag::ActionClamped)) ++stats_.action_clamped;
        if (result.has(SafetyFlag::ActionRateLimited)) ++stats_.action_rate_limited;
        if (result.has(SafetyFlag::ActionAccelerationLimited)) {
            ++stats_.action_acceleration_limited;
        }
        if (result.modified()) ++stats_.modified;
        remember(result.action, period);
        return result;
    }

    const SafetyConfig &config() const noexcept { return config_; }
    const SafetyStats &stats() const noexcept { return stats_; }

   private:
    void validate_seed(const Action &action) const {
        for (int d = 0; d < config_.action_dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(d);
            if (!std::isfinite(action[i]) || action[i] < config_.min_action[i] ||
                action[i] > config_.max_action[i]) {
                throw std::invalid_argument("safety seed violates configured bounds");
            }
        }
    }

    void remember(const Action &action, Nanos period) noexcept {
        const float seconds = std::chrono::duration<float>(period).count();
        if (have_last_ && seconds > 0.0F) {
            for (int d = 0; d < config_.action_dim; ++d) {
                const std::size_t i = static_cast<std::size_t>(d);
                previous_velocity_[i] = (action[i] - last_action_[i]) / seconds;
            }
            have_velocity_ = true;
        }
        last_action_ = action;
        have_last_ = true;
    }

    const SafetyConfig config_;
    Action last_action_{};
    Action previous_velocity_{};
    bool have_last_ = false;
    bool have_velocity_ = false;
    SafetyStats stats_{};
};

}  // namespace cerebellum
