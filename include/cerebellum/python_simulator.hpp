#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cerebellum/loop.hpp"
#include "cerebellum/observation.hpp"

namespace cerebellum {

enum class PythonSimulatorBackend { Synthetic, Libero };
enum class SimulatorDelivery { LatestOnly, Acknowledged };

struct SimulatorImageSpec {
  std::string feature_name;
  std::uint16_t channels = 0;
  std::uint16_t height = 0;
  std::uint16_t width = 0;
};

struct PythonSimulatorOptions {
  std::string python_executable = "python3";
  std::string python_package_path = "python";
  PythonSimulatorBackend backend = PythonSimulatorBackend::Synthetic;
  SimulatorDelivery delivery = SimulatorDelivery::LatestOnly;
  std::string suite = "libero_spatial";
  int task_id = 0;
  int init_state = 0;
  int control_hz = kControlHz;
  int image_size = 256;
  int expected_action_dim = 7;
  // A user-local extracted libOSMesa directory can be supplied on machines
  // where EGL device access is unavailable. The child environment alone is
  // changed; Cerebellum's process and other Python workers are untouched.
  std::string osmesa_library_path;
  std::chrono::milliseconds startup_timeout{120'000};
  std::chrono::milliseconds step_timeout{30'000};
  std::chrono::microseconds worker_poll_period{100};
};

struct SimulatorStats {
  std::uint64_t commands_emitted = 0;
  std::uint64_t commands_applied = 0;
  std::uint64_t commands_superseded = 0;
  std::uint64_t observations = 0;
  std::uint64_t transport_errors = 0;
  std::uint64_t environment_step_ns = 0;
  std::uint64_t observation_build_ns = 0;
  std::uint64_t cpp_round_trip_ns = 0;
  bool terminated = false;
  bool success = false;
};

// One object is both ends of the simulated robot boundary:
//
//   RuntimeLoop control thread -> atomic newest-action mailbox
//   simulator worker          -> Python/MuJoCo -> newest observation snapshot
//   inference worker          -> atomic newest observation snapshot
//
// LatestOnly emit() therefore satisfies ActionSink's no-wait/no-I/O/no-allocation
// rule. Acknowledged delivery deliberately waits and is restricted to
// slower-than-real-time evaluation.
class PythonSimulatorAdapter final : public ActionSink,
                                     public ObservationSource {
public:
  explicit PythonSimulatorAdapter(PythonSimulatorOptions options = {});
  ~PythonSimulatorAdapter() override;

  PythonSimulatorAdapter(const PythonSimulatorAdapter &) = delete;
  PythonSimulatorAdapter &operator=(const PythonSimulatorAdapter &) = delete;

  void emit(const ActionEmission &emission) noexcept override;
  std::shared_ptr<const ObservationSnapshot> latest() noexcept override;

  bool healthy() const noexcept {
    return healthy_.load(std::memory_order_acquire);
  }
  int action_dim() const noexcept { return action_dim_; }
  int state_dim() const noexcept { return state_dim_; }
  const std::string &task() const noexcept { return task_; }
  const std::vector<SimulatorImageSpec> &image_schema() const noexcept {
    return images_;
  }
  SimulatorStats stats() const noexcept;
  std::string last_error() const;

private:
  struct PendingAction {
    std::uint64_t publication = 0;
    std::int64_t step = 0;
    Action action{};
  };

  bool read_handshake();
  std::shared_ptr<const ObservationSnapshot> read_observation();
  bool write_command(const PendingAction &pending);
  bool load_pending(std::uint64_t consumed,
                    PendingAction &pending) const noexcept;
  void worker_loop() noexcept;
  bool write_frame(const void *data, std::size_t size);
  bool read_frame(std::string &payload);
  bool write_all(const void *data, std::size_t size);
  bool read_all(void *data, std::size_t size);
  bool wait_for(short events);
  void transport_error(std::string message) noexcept;
  void stop_child() noexcept;

  PythonSimulatorOptions options_;
  std::chrono::milliseconds io_timeout_;
  int socket_ = -1;
  int child_pid_ = -1;
  int action_dim_ = 0;
  int state_dim_ = 0;
  std::string task_;
  std::vector<SimulatorImageSpec> images_;

  // Single control-thread writer, single simulator-thread reader seqlock.
  // Odd means a write is in progress; even values identify publications.
  alignas(64) std::atomic<std::uint64_t> mailbox_version_{0};
  std::atomic<std::int64_t> mailbox_step_{0};
  std::array<std::atomic<std::uint32_t>, kPaddedActionDim> mailbox_action_{};

  // GCC 11 does not expose atomic<shared_ptr>'s C++20 specialization, but the
  // standard atomic shared_ptr free functions provide the same ownership-safe
  // publication on every compiler Cerebellum currently supports.
  std::shared_ptr<const ObservationSnapshot> latest_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> healthy_{false};
  std::thread worker_;

  std::atomic<std::uint64_t> commands_emitted_{0};
  std::atomic<std::uint64_t> commands_applied_{0};
  std::atomic<std::uint64_t> commands_superseded_{0};
  std::atomic<std::uint64_t> observations_{0};
  std::atomic<std::uint64_t> transport_errors_{0};
  std::atomic<bool> terminated_{false};
  std::atomic<bool> success_{false};

  // Used only by acknowledged simulation. The control thread waits here while
  // the simulator worker performs one command; real-time delivery never locks.
  mutable std::mutex acknowledgement_mutex_;
  std::condition_variable acknowledgement_cv_;

  std::atomic<std::uint64_t> environment_step_ns_{0};
  std::atomic<std::uint64_t> observation_build_ns_{0};
  std::atomic<std::uint64_t> cpp_round_trip_ns_{0};

  mutable std::mutex error_mutex_;
  std::string last_error_;
};

} // namespace cerebellum
