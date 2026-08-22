#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "cerebellum/python_chunk_generator.hpp"
#include "cerebellum/python_simulator.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

namespace {

PythonSimulatorOptions options() {
  PythonSimulatorOptions value;
  value.python_executable = CEREBELLUM_TEST_PYTHON;
  value.python_package_path = CEREBELLUM_TEST_PYTHONPATH;
  value.backend = PythonSimulatorBackend::Synthetic;
  value.image_size = 4;
  value.startup_timeout = std::chrono::seconds(5);
  value.step_timeout = std::chrono::seconds(5);
  value.worker_poll_period = std::chrono::microseconds(20);
  return value;
}

bool wait_for_applied(PythonSimulatorAdapter &simulator, std::uint64_t count) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (simulator.stats().commands_applied >= count)
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void test_handshake_and_initial_observation() {
  PythonSimulatorAdapter simulator(options());
  CHECK(simulator.healthy());
  CHECK(simulator.action_dim() == 7);
  CHECK(simulator.state_dim() == 8);
  CHECK(simulator.task() == "move the object to the target");
  CHECK(simulator.image_schema().size() == 2);

  const auto observation = simulator.latest();
  CHECK(observation != nullptr);
  CHECK(observation->sequence == 0);
  CHECK(observation->state.size() == 8);
  CHECK(observation->images.size() == 2);
  CHECK(observation->images[0].pixels.size() == 3 * 4 * 4);
  CHECK(simulator.stats().observations == 1);
}

void test_emit_is_applied_by_worker_and_publishes_observation() {
  PythonSimulatorAdapter simulator(options());
  ActionEmission emission;
  emission.step = 9;
  for (int d = 0; d < simulator.action_dim(); ++d) {
    emission.action[static_cast<std::size_t>(d)] = static_cast<float>(d + 1);
  }
  simulator.emit(emission);
  CHECK(wait_for_applied(simulator, 1));

  const auto observation = simulator.latest();
  CHECK(observation->sequence == 1);
  for (int d = 0; d < simulator.action_dim(); ++d) {
    CHECK(observation->state[static_cast<std::size_t>(d)] ==
          static_cast<float>(d + 1));
  }
  const SimulatorStats stats = simulator.stats();
  CHECK(stats.commands_emitted == 1);
  CHECK(stats.commands_applied == 1);
  CHECK(stats.observations == 2);
  CHECK(stats.transport_errors == 0);
}

void test_latest_action_wins_without_blocking_control() {
  PythonSimulatorAdapter simulator(options());
  ActionEmission emission;
  constexpr int kWrites = 1000;
  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < kWrites; ++i) {
    emission.step = i;
    emission.action[0] = static_cast<float>(i);
    simulator.emit(emission);
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  CHECK(elapsed < std::chrono::milliseconds(20));
  CHECK(wait_for_applied(simulator, 1));
  CHECK(simulator.stats().commands_emitted == kWrites);
  CHECK(simulator.stats().commands_superseded > 0);
}

void test_runtime_closes_observation_inference_action_loop() {
  PythonSimulatorAdapter simulator(options());
  RuntimeConfig config;
  config.control_period = std::chrono::milliseconds(1);
  config.inference_budget_ms = 1.0;
  config.action_dim = 7;
  config.action_space = ActionSpace::Delta;
  PythonChunkGeneratorOptions generator_options;
  generator_options.python_executable = CEREBELLUM_TEST_PYTHON;
  generator_options.python_package_path = CEREBELLUM_TEST_PYTHONPATH;
  generator_options.startup_timeout = std::chrono::seconds(5);
  generator_options.inference_timeout = std::chrono::seconds(5);
  PythonChunkGenerator generator(config, simulator, generator_options);
  RuntimeLoop loop(config, generator, simulator, 100,
                   std::chrono::microseconds(20));

  loop.run_for(100);

  CHECK(loop.inference_stats().generated > 0);
  CHECK(loop.queue().consumer_stats().chunks_accepted > 0);
  CHECK(simulator.stats().commands_emitted == 100);
  CHECK(simulator.stats().commands_applied > 0);
  CHECK(simulator.latest()->sequence > 0);
}

void test_real_libero_when_requested() {
  const char *enabled = std::getenv("CEREBELLUM_RUN_LIBERO_SIMULATOR");
  if (!enabled || std::string(enabled) != "1")
    return;
  auto value = options();
  value.backend = PythonSimulatorBackend::Libero;
  value.image_size = 256;
  value.startup_timeout = std::chrono::minutes(2);
  if (const char *path = std::getenv("CEREBELLUM_OSMESA_LIBRARY_PATH")) {
    value.osmesa_library_path = path;
  }
  PythonSimulatorAdapter simulator(value);
  CHECK(simulator.healthy());
  CHECK(simulator.task().find("pick up") != std::string::npos);
  ActionEmission noop;
  noop.action[6] = -1.0F;
  simulator.emit(noop);
  CHECK(wait_for_applied(simulator, 1));
  CHECK(simulator.latest()->sequence == 1);
  CHECK(simulator.latest()->state.size() == 8);
  CHECK(simulator.latest()->images.size() == 2);
}

} // namespace

int main() {
  test_handshake_and_initial_observation();
  test_emit_is_applied_by_worker_and_publishes_observation();
  test_latest_action_wins_without_blocking_control();
  test_runtime_closes_observation_inference_action_loop();
  test_real_libero_when_requested();
  if (g_failures == 0) {
    std::printf("test_python_simulator: all checks passed\n");
  } else {
    std::printf("test_python_simulator: %d FAILURES\n", g_failures);
  }
  return g_failures;
}
