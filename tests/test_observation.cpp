#include <cstdio>
#include <limits>
#include <stdexcept>

#include "cerebellum/observation.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_THROWS(statement)                                                \
    do {                                                                       \
        bool threw = false;                                                    \
        try { statement; } catch (const std::invalid_argument&) { threw = true; } \
        CHECK(threw);                                                          \
    } while (0)

namespace {

ObservationSnapshot valid_observation() {
    ObservationSnapshot observation;
    observation.sequence = 1;
    observation.capture_time = now();
    observation.state = {1.0F, 2.0F};
    observation.task = "move";
    observation.images.push_back(CameraImage{"camera", 3, 1, 2, {0, 1, 2, 3, 4, 5}});
    return observation;
}

void test_valid_snapshot() {
    const ObservationSnapshot observation = valid_observation();
    observation.validate();
}

void test_rejects_bad_pixel_count() {
    ObservationSnapshot observation = valid_observation();
    observation.images[0].pixels.pop_back();
    CHECK_THROWS(observation.validate());
}

void test_rejects_non_finite_state() {
    ObservationSnapshot observation = valid_observation();
    observation.state[0] = std::numeric_limits<float>::infinity();
    CHECK_THROWS(observation.validate());
}

void test_rejects_duplicate_camera_names() {
    ObservationSnapshot observation = valid_observation();
    observation.images.push_back(observation.images[0]);
    CHECK_THROWS(observation.validate());
}

}  // namespace

int main() {
    test_valid_snapshot();
    test_rejects_bad_pixel_count();
    test_rejects_non_finite_state();
    test_rejects_duplicate_camera_names();

    if (g_failures == 0) {
        std::printf("test_observation: all checks passed\n");
    } else {
        std::printf("test_observation: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
