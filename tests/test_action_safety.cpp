#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "cerebellum/action_safety.hpp"

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

SafetyConfig one_dimension() {
    SafetyConfig config(1);
    config.min_action[0] = -2.0F;
    config.max_action[0] = 2.0F;
    config.replacement_action[0] = 0.25F;
    return config;
}

void test_position_bounds_and_padded_tail() {
    ActionSafetyFilter filter(one_dimension());
    Action requested{};
    requested[0] = 9.0F;
    requested[4] = 99.0F;

    const SafetyResult result =
        filter.apply(requested, std::chrono::seconds(1), false, Nanos::zero());

    CHECK(result.action[0] == 2.0F);
    CHECK(result.action[4] == 0.0F);
    CHECK(result.has(SafetyFlag::ActionClamped));
    CHECK(!result.rejected);
    CHECK(filter.stats().action_clamped == 1);
    CHECK(filter.stats().requested_below_min[0] == 0);
    CHECK(filter.stats().requested_above_max[0] == 1);
    CHECK(filter.stats().max_bound_excess[0] == 7.0F);
}

void test_velocity_limit_uses_control_period() {
    SafetyConfig config = one_dimension();
    config.max_action_rate[0] = 2.0F;
    ActionSafetyFilter filter(config);
    Action initial{};
    filter.reset(initial);
    Action requested{};
    requested[0] = 2.0F;

    const SafetyResult result =
        filter.apply(requested, std::chrono::milliseconds(500), false, Nanos::zero());

    CHECK(result.action[0] == 1.0F);
    CHECK(result.has(SafetyFlag::ActionRateLimited));
}

void test_acceleration_limit_uses_previous_safe_velocity() {
    SafetyConfig config = one_dimension();
    config.max_action_acceleration[0] = 1.0F;
    ActionSafetyFilter filter(config);
    Action zero{};
    filter.reset(zero);
    (void)filter.apply(zero, std::chrono::seconds(1), false, Nanos::zero());

    Action requested{};
    requested[0] = 2.0F;
    const SafetyResult result =
        filter.apply(requested, std::chrono::seconds(1), false, Nanos::zero());

    CHECK(result.action[0] == 1.0F);
    CHECK(result.has(SafetyFlag::ActionAccelerationLimited));
}

void test_nonfinite_and_stale_actions_hold_last_safe() {
    SafetyConfig config = one_dimension();
    config.max_observation_age = std::chrono::milliseconds(100);
    ActionSafetyFilter filter(config);
    Action safe{};
    safe[0] = 0.5F;
    filter.reset(safe);

    Action invalid{};
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    SafetyResult result =
        filter.apply(invalid, std::chrono::milliseconds(33), true, std::chrono::milliseconds(1));
    CHECK(result.rejected);
    CHECK(result.has(SafetyFlag::NonFiniteRejected));
    CHECK(result.action[0] == 0.5F);

    Action stale{};
    stale[0] = 1.0F;
    result = filter.apply(stale, std::chrono::milliseconds(33), true,
                          std::chrono::milliseconds(101));
    CHECK(result.rejected);
    CHECK(result.has(SafetyFlag::StaleRejected));
    CHECK(result.action[0] == 0.5F);
    CHECK(filter.stats().rejected == 2);
}

void test_default_unlimited_dynamics_do_not_modify_action() {
    ActionSafetyFilter filter(one_dimension());
    Action initial{};
    initial[0] = -0.1234567F;
    filter.reset(initial);
    (void)filter.apply(initial, kControlPeriod, false, Nanos::zero());

    Action requested{};
    requested[0] = 0.7654321F;
    const SafetyResult result = filter.apply(requested, kControlPeriod, false, Nanos::zero());

    CHECK(result.action[0] == requested[0]);
    CHECK(!result.modified());
    CHECK(filter.stats().action_rate_limited == 0);
    CHECK(filter.stats().action_acceleration_limited == 0);
}

void test_invalid_configuration_and_seed_are_rejected() {
    SafetyConfig config = one_dimension();
    config.max_action_rate[0] = 0.0F;
    bool threw = false;
    try {
        ActionSafetyFilter filter(config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);

    ActionSafetyFilter filter(one_dimension());
    Action seed{};
    seed[0] = 3.0F;
    threw = false;
    try {
        filter.reset(seed);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_position_bounds_and_padded_tail();
    test_velocity_limit_uses_control_period();
    test_acceleration_limit_uses_previous_safe_velocity();
    test_nonfinite_and_stale_actions_hold_last_safe();
    test_default_unlimited_dynamics_do_not_modify_action();
    test_invalid_configuration_and_seed_are_rejected();

    if (g_failures == 0) {
        std::printf("test_action_safety: all checks passed\n");
    } else {
        std::printf("test_action_safety: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
