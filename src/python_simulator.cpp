#include "cerebellum/python_simulator.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

extern char **environ;

namespace cerebellum {
namespace {

constexpr std::uint16_t kProtocolVersion = 3;
constexpr std::uint32_t kMaxFrameBytes = 16U * 1024U * 1024U;
constexpr std::string_view kHelloMagic = "CBSH";
constexpr std::string_view kCommandMagic = "CBSC";
constexpr std::string_view kObservationMagic = "CBSO";

void append_u16(std::string &out, std::uint16_t value) {
  const std::uint16_t encoded = htons(value);
  out.append(reinterpret_cast<const char *>(&encoded), sizeof(encoded));
}

void append_u32(std::string &out, std::uint32_t value) {
  const std::uint32_t encoded = htonl(value);
  out.append(reinterpret_cast<const char *>(&encoded), sizeof(encoded));
}

void append_u64(std::string &out, std::uint64_t value) {
  append_u32(out, static_cast<std::uint32_t>(value >> 32));
  append_u32(out, static_cast<std::uint32_t>(value));
}

void append_f32(std::string &out, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  append_u32(out, bits);
}

class Reader {
public:
  explicit Reader(std::string_view data) : data_(data) {}

  std::string_view bytes(std::size_t count) {
    if (count > data_.size() - offset_)
      throw std::runtime_error("truncated message");
    const auto result = data_.substr(offset_, count);
    offset_ += count;
    return result;
  }
  std::uint8_t u8() { return static_cast<std::uint8_t>(bytes(1)[0]); }
  std::uint16_t u16() {
    const auto value = bytes(2);
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, value.data(), sizeof(encoded));
    return ntohs(encoded);
  }
  std::uint32_t u32() {
    const auto value = bytes(4);
    std::uint32_t encoded = 0;
    std::memcpy(&encoded, value.data(), sizeof(encoded));
    return ntohl(encoded);
  }
  std::uint64_t u64() {
    return (static_cast<std::uint64_t>(u32()) << 32) | u32();
  }
  std::int64_t i64() { return static_cast<std::int64_t>(u64()); }
  float f32() {
    const std::uint32_t bits = u32();
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  bool empty() const noexcept { return offset_ == data_.size(); }

private:
  std::string_view data_;
  std::size_t offset_ = 0;
};

bool key_matches(const std::string &entry, std::string_view key) {
  return entry.size() > key.size() && entry.compare(0, key.size(), key) == 0 &&
         entry[key.size()] == '=';
}

std::vector<std::string>
child_environment(const PythonSimulatorOptions &options) {
  std::vector<std::string> result;
  for (char **entry = environ; entry && *entry; ++entry)
    result.emplace_back(*entry);
  if (options.osmesa_library_path.empty())
    return result;

  const auto set = [&result](std::string key, std::string value) {
    const std::string joined = key + "=" + value;
    for (std::string &entry : result) {
      if (key_matches(entry, key)) {
        entry = joined;
        return;
      }
    }
    result.push_back(joined);
  };
  std::string libraries = options.osmesa_library_path;
  if (const char *existing = std::getenv("LD_LIBRARY_PATH");
      existing && *existing) {
    libraries += ":";
    libraries += existing;
  }
  set("LD_LIBRARY_PATH", libraries);
  set("PYOPENGL_PLATFORM", "osmesa");
  set("MUJOCO_GL", "osmesa");
  return result;
}

} // namespace

PythonSimulatorAdapter::PythonSimulatorAdapter(PythonSimulatorOptions options)
    : options_(std::move(options)), io_timeout_(options_.startup_timeout) {
  if (options_.expected_action_dim <= 0 ||
      options_.expected_action_dim > kPaddedActionDim ||
      options_.image_size <= 0 ||
      options_.image_size > std::numeric_limits<std::uint16_t>::max() ||
      options_.control_hz <= 0 ||
      options_.worker_poll_period <= std::chrono::microseconds::zero() ||
      options_.startup_timeout <= std::chrono::milliseconds::zero() ||
      options_.step_timeout <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("invalid Python simulator options");
  }

  int sockets[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
    throw std::runtime_error(std::string("socketpair failed: ") +
                             std::strerror(errno));
  }

  const std::string code =
      "import sys; sys.path.insert(0, sys.argv.pop(1)); "
      "from cerebellum_model.simulator_server import main; main()";
  std::vector<std::string> arguments{
      options_.python_executable,
      "-c",
      code,
      options_.python_package_path,
      "--backend",
      options_.backend == PythonSimulatorBackend::Synthetic ? "synthetic"
                                                            : "libero",
      "--action-dim",
      std::to_string(options_.expected_action_dim),
      "--image-size",
      std::to_string(options_.image_size),
      "--suite",
      options_.suite,
      "--task-id",
      std::to_string(options_.task_id),
      "--init-state",
      std::to_string(options_.init_state),
      "--control-hz",
      std::to_string(options_.control_hz),
      "--video-path",
      options_.video_path,
      "--video-label",
      options_.video_label,
  };
  std::vector<char *> argv;
  argv.reserve(arguments.size() + 1);
  for (std::string &argument : arguments)
    argv.push_back(argument.data());
  argv.push_back(nullptr);

  std::vector<std::string> environment = child_environment(options_);
  std::vector<char *> envp;
  envp.reserve(environment.size() + 1);
  for (std::string &entry : environment)
    envp.push_back(entry.data());
  envp.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  int spawn_error = ::posix_spawn_file_actions_init(&actions);
  const bool actions_initialized = spawn_error == 0;
  if (spawn_error == 0)
    spawn_error = ::posix_spawn_file_actions_addclose(&actions, sockets[0]);
  if (spawn_error == 0) {
    spawn_error =
        ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
  }
  if (spawn_error == 0) {
    spawn_error =
        ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
  }
  if (spawn_error == 0)
    spawn_error = ::posix_spawn_file_actions_addclose(&actions, sockets[1]);

  pid_t pid = -1;
  if (spawn_error == 0) {
    spawn_error = ::posix_spawnp(&pid, options_.python_executable.c_str(),
                                 &actions, nullptr, argv.data(), envp.data());
  }
  if (actions_initialized)
    ::posix_spawn_file_actions_destroy(&actions);
  if (spawn_error != 0) {
    ::close(sockets[0]);
    ::close(sockets[1]);
    throw std::runtime_error(std::string("starting Python simulator failed: ") +
                             std::strerror(spawn_error));
  }

  ::close(sockets[1]);
  socket_ = sockets[0];
  child_pid_ = static_cast<int>(pid);
  if (!read_handshake()) {
    const std::string error = last_error();
    stop_child();
    throw std::runtime_error("Python simulator handshake failed: " + error);
  }
  std::shared_ptr<const ObservationSnapshot> initial = read_observation();
  if (!initial) {
    const std::string error = last_error();
    stop_child();
    throw std::runtime_error("Python simulator reset failed: " + error);
  }
  std::atomic_store_explicit(&latest_, std::move(initial),
                             std::memory_order_release);
  observations_.store(1, std::memory_order_relaxed);
  io_timeout_ = options_.step_timeout;
  healthy_.store(true, std::memory_order_release);
  worker_ = std::thread([this] { worker_loop(); });
}

PythonSimulatorAdapter::~PythonSimulatorAdapter() {
  stop_.store(true, std::memory_order_release);
  acknowledgement_cv_.notify_all();
  if (worker_.joinable())
    worker_.join();
  stop_child();
}

void PythonSimulatorAdapter::emit(const ActionEmission &emission) noexcept {
  std::uint64_t version = mailbox_version_.load(std::memory_order_relaxed);
  if ((version & 1U) != 0)
    ++version;
  mailbox_version_.store(version + 1, std::memory_order_release);
  mailbox_step_.store(emission.step, std::memory_order_relaxed);
  mailbox_observation_age_ns_.store(emission.observation_age.count(),
                                    std::memory_order_relaxed);
  mailbox_safety_flags_.store(emission.safety_flags, std::memory_order_relaxed);
  mailbox_fallback_.store(emission.fallback, std::memory_order_relaxed);
  mailbox_safety_rejected_.store(emission.safety_rejected, std::memory_order_relaxed);
  for (int d = 0; d < action_dim_; ++d) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &emission.action[static_cast<std::size_t>(d)],
                sizeof(bits));
    mailbox_action_[static_cast<std::size_t>(d)].store(
        bits, std::memory_order_relaxed);
  }
  mailbox_version_.store(version + 2, std::memory_order_release);
  const std::uint64_t target =
      commands_emitted_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (options_.delivery == SimulatorDelivery::Acknowledged) {
    try {
      std::unique_lock lock(acknowledgement_mutex_);
      acknowledgement_cv_.wait(lock, [this, target] {
        return commands_applied_.load(std::memory_order_acquire) >= target ||
               !healthy() || stop_.load(std::memory_order_acquire);
      });
    } catch (...) {
      transport_error("waiting for simulator acknowledgement failed");
    }
  }
}

std::shared_ptr<const ObservationSnapshot>
PythonSimulatorAdapter::latest() noexcept {
  return std::atomic_load_explicit(&latest_, std::memory_order_acquire);
}

SimulatorStats PythonSimulatorAdapter::stats() const noexcept {
  return SimulatorStats{
      commands_emitted_.load(std::memory_order_relaxed),
      commands_applied_.load(std::memory_order_relaxed),
      commands_superseded_.load(std::memory_order_relaxed),
      observations_.load(std::memory_order_relaxed),
      transport_errors_.load(std::memory_order_relaxed),
      environment_step_ns_.load(std::memory_order_relaxed),
      observation_build_ns_.load(std::memory_order_relaxed),
      cpp_round_trip_ns_.load(std::memory_order_relaxed),
      terminated_.load(std::memory_order_relaxed),
      success_.load(std::memory_order_relaxed),
  };
}

std::string PythonSimulatorAdapter::last_error() const {
  std::lock_guard lock(error_mutex_);
  return last_error_;
}

bool PythonSimulatorAdapter::read_handshake() {
  std::string payload;
  if (!read_frame(payload))
    return false;
  try {
    Reader reader(payload);
    if (reader.bytes(4) != kHelloMagic || reader.u16() != kProtocolVersion) {
      throw std::runtime_error("bad simulator handshake magic or version");
    }
    const std::uint8_t status = reader.u8();
    (void)reader.u8();
    action_dim_ = reader.u16();
    state_dim_ = reader.u16();
    const std::uint16_t image_count = reader.u16();
    const std::uint32_t detail_size = reader.u32();
    if (status != 0) {
      transport_error(std::string(reader.bytes(detail_size)));
      return false;
    }
    if (action_dim_ != options_.expected_action_dim || state_dim_ <= 0 ||
        image_count == 0 || detail_size == 0) {
      throw std::runtime_error(
          "simulator shape does not match requested configuration");
    }
    images_.clear();
    images_.reserve(image_count);
    for (std::uint16_t i = 0; i < image_count; ++i) {
      const std::uint16_t name_size = reader.u16();
      SimulatorImageSpec image;
      image.channels = reader.u16();
      image.height = reader.u16();
      image.width = reader.u16();
      image.feature_name = std::string(reader.bytes(name_size));
      if (image.feature_name.empty() || image.channels == 0 ||
          image.height == 0 || image.width == 0) {
        throw std::runtime_error("simulator sent invalid image schema");
      }
      images_.push_back(std::move(image));
    }
    task_ = std::string(reader.bytes(detail_size));
    if (task_.empty() || !reader.empty())
      throw std::runtime_error("invalid simulator task");
    return true;
  } catch (const std::exception &exc) {
    transport_error(exc.what());
    return false;
  }
}

std::shared_ptr<const ObservationSnapshot>
PythonSimulatorAdapter::read_observation() {
  std::string payload;
  if (!read_frame(payload))
    return nullptr;
  try {
    Reader reader(payload);
    if (reader.bytes(4) != kObservationMagic ||
        reader.u16() != kProtocolVersion) {
      throw std::runtime_error("bad simulator observation magic or version");
    }
    const std::uint8_t status = reader.u8();
    (void)reader.u8();
    const std::uint64_t sequence = reader.u64();
    const std::uint64_t capture_ns = reader.u64();
    (void)reader.i64(); // simulator step is diagnostic metadata
    (void)reader.f32(); // reward
    const bool terminated = reader.u8() != 0;
    const bool success = reader.u8() != 0;
    const std::uint16_t state_dim = reader.u16();
    const std::uint16_t image_count = reader.u16();
    const std::uint64_t environment_step_ns = reader.u64();
    const std::uint64_t observation_build_ns = reader.u64();
    const std::uint32_t error_size = reader.u32();
    if (status != 0) {
      transport_error(std::string(reader.bytes(error_size)));
      return nullptr;
    }
    if (error_size != 0 || state_dim != state_dim_ ||
        image_count != images_.size()) {
      throw std::runtime_error("simulator observation shape changed");
    }
    auto observation = std::make_shared<ObservationSnapshot>();
    observation->sequence = sequence;
    // Python monotonic_ns and C++ steady_clock share CLOCK_MONOTONIC on this
    // host/process boundary, preserving the observation-age measurement.
    observation->capture_time =
        TimePoint{Nanos{static_cast<std::int64_t>(capture_ns)}};
    observation->state.reserve(state_dim);
    for (std::uint16_t d = 0; d < state_dim; ++d)
      observation->state.push_back(reader.f32());
    observation->images.reserve(image_count);
    for (std::uint16_t i = 0; i < image_count; ++i) {
      const std::uint16_t name_size = reader.u16();
      CameraImage image;
      image.channels = reader.u16();
      image.height = reader.u16();
      image.width = reader.u16();
      const std::uint32_t pixel_size = reader.u32();
      image.feature_name = std::string(reader.bytes(name_size));
      const auto pixels = reader.bytes(pixel_size);
      image.pixels.assign(pixels.begin(), pixels.end());
      const SimulatorImageSpec &expected = images_[i];
      if (image.feature_name != expected.feature_name ||
          image.channels != expected.channels ||
          image.height != expected.height || image.width != expected.width) {
        throw std::runtime_error("simulator observation camera schema changed");
      }
      observation->images.push_back(std::move(image));
    }
    if (!reader.empty())
      throw std::runtime_error("simulator observation has trailing bytes");
    observation->task = task_;
    observation->validate();
    // Episode outcomes are sticky: a later action must not erase the fact that
    // the task succeeded or terminated at an earlier simulated step.
    if (terminated) terminated_.store(true, std::memory_order_relaxed);
    if (success) success_.store(true, std::memory_order_relaxed);
    environment_step_ns_.fetch_add(environment_step_ns, std::memory_order_relaxed);
    observation_build_ns_.fetch_add(observation_build_ns, std::memory_order_relaxed);
    return observation;
  } catch (const std::exception &exc) {
    transport_error(exc.what());
    return nullptr;
  }
}

bool PythonSimulatorAdapter::write_command(const PendingAction &pending) {
  std::string payload;
  payload.reserve(40 + static_cast<std::size_t>(action_dim_) * sizeof(float));
  payload.append(kCommandMagic);
  append_u16(payload, kProtocolVersion);
  payload.push_back('\0');
  payload.push_back('\0');
  append_u64(payload, pending.publication / 2);
  append_u64(payload, static_cast<std::uint64_t>(pending.step));
  append_u64(payload, static_cast<std::uint64_t>(
                          std::max<std::int64_t>(0, pending.observation_age_ns)));
  append_u32(payload, pending.safety_flags);
  payload.push_back(pending.fallback ? '\1' : '\0');
  payload.push_back(pending.safety_rejected ? '\1' : '\0');
  append_u16(payload, static_cast<std::uint16_t>(action_dim_));
  for (int d = 0; d < action_dim_; ++d) {
    append_f32(payload, pending.action[static_cast<std::size_t>(d)]);
  }
  return write_frame(payload.data(), payload.size());
}

bool PythonSimulatorAdapter::load_pending(
    std::uint64_t consumed, PendingAction &pending) const noexcept {
  for (;;) {
    const std::uint64_t before =
        mailbox_version_.load(std::memory_order_acquire);
    if (before == consumed || (before & 1U) != 0)
      return false;
    pending.publication = before;
    pending.step = mailbox_step_.load(std::memory_order_relaxed);
    pending.observation_age_ns =
        mailbox_observation_age_ns_.load(std::memory_order_relaxed);
    pending.safety_flags = mailbox_safety_flags_.load(std::memory_order_relaxed);
    pending.fallback = mailbox_fallback_.load(std::memory_order_relaxed);
    pending.safety_rejected = mailbox_safety_rejected_.load(std::memory_order_relaxed);
    for (int d = 0; d < action_dim_; ++d) {
      const std::uint32_t bits =
          mailbox_action_[static_cast<std::size_t>(d)].load(
              std::memory_order_relaxed);
      std::memcpy(&pending.action[static_cast<std::size_t>(d)], &bits,
                  sizeof(bits));
    }
    const std::uint64_t after =
        mailbox_version_.load(std::memory_order_acquire);
    if (before == after)
      return true;
  }
}

void PythonSimulatorAdapter::worker_loop() noexcept {
  std::uint64_t consumed = 0;
  while (!stop_.load(std::memory_order_acquire) && healthy()) {
    PendingAction pending;
    if (!load_pending(consumed, pending)) {
      std::this_thread::sleep_for(options_.worker_poll_period);
      continue;
    }
    if (pending.publication > consumed + 2) {
      commands_superseded_.fetch_add((pending.publication - consumed) / 2 - 1,
                                     std::memory_order_relaxed);
    }
    consumed = pending.publication;
    const TimePoint round_trip_start = now();
    if (!write_command(pending))
      break;
    std::shared_ptr<const ObservationSnapshot> observation = read_observation();
    if (!observation)
      break;
    std::atomic_store_explicit(&latest_, std::move(observation),
                               std::memory_order_release);
    commands_applied_.fetch_add(1, std::memory_order_relaxed);
    cpp_round_trip_ns_.fetch_add(
        static_cast<std::uint64_t>((now() - round_trip_start).count()),
        std::memory_order_relaxed);
    observations_.fetch_add(1, std::memory_order_relaxed);
    acknowledgement_cv_.notify_all();
  }
  healthy_.store(false, std::memory_order_release);
  acknowledgement_cv_.notify_all();
}

bool PythonSimulatorAdapter::write_frame(const void *data, std::size_t size) {
  if (size > kMaxFrameBytes) {
    transport_error("outgoing simulator frame exceeds limit");
    return false;
  }
  const std::uint32_t encoded = htonl(static_cast<std::uint32_t>(size));
  return write_all(&encoded, sizeof(encoded)) && write_all(data, size);
}

bool PythonSimulatorAdapter::read_frame(std::string &payload) {
  std::uint32_t encoded = 0;
  if (!read_all(&encoded, sizeof(encoded)))
    return false;
  const std::uint32_t size = ntohl(encoded);
  if (size > kMaxFrameBytes) {
    transport_error("incoming simulator frame exceeds limit");
    return false;
  }
  payload.resize(size);
  return size == 0 || read_all(payload.data(), size);
}

bool PythonSimulatorAdapter::write_all(const void *data, std::size_t size) {
  const char *cursor = static_cast<const char *>(data);
  while (size > 0) {
    if (!wait_for(POLLOUT))
      return false;
    const ssize_t written = ::send(socket_, cursor, size, MSG_NOSIGNAL);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      transport_error(std::string("write to Python simulator failed: ") +
                      std::strerror(errno));
      return false;
    }
    cursor += written;
    size -= static_cast<std::size_t>(written);
  }
  return true;
}

bool PythonSimulatorAdapter::read_all(void *data, std::size_t size) {
  char *cursor = static_cast<char *>(data);
  while (size > 0) {
    if (!wait_for(POLLIN))
      return false;
    const ssize_t received = ::recv(socket_, cursor, size, 0);
    if (received == 0) {
      transport_error("Python simulator closed the connection");
      return false;
    }
    if (received < 0) {
      if (errno == EINTR)
        continue;
      transport_error(std::string("read from Python simulator failed: ") +
                      std::strerror(errno));
      return false;
    }
    cursor += received;
    size -= static_cast<std::size_t>(received);
  }
  return true;
}

bool PythonSimulatorAdapter::wait_for(short events) {
  pollfd descriptor{socket_, events, 0};
  int result = 0;
  do {
    result = ::poll(&descriptor, 1, static_cast<int>(io_timeout_.count()));
  } while (result < 0 && errno == EINTR);
  if (stop_.load(std::memory_order_acquire))
    return false;
  if (result == 0) {
    transport_error("Python simulator timed out");
    return false;
  }
  if (result < 0 ||
      (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
    transport_error(result < 0 ? std::string("simulator poll failed: ") +
                                     std::strerror(errno)
                               : "Python simulator connection failed");
    return false;
  }
  return true;
}

void PythonSimulatorAdapter::transport_error(std::string message) noexcept {
  if (!stop_.load(std::memory_order_acquire)) {
    transport_errors_.fetch_add(1, std::memory_order_relaxed);
    try {
      std::lock_guard lock(error_mutex_);
      last_error_ = std::move(message);
    } catch (...) {
    }
  }
  healthy_.store(false, std::memory_order_release);
}

void PythonSimulatorAdapter::stop_child() noexcept {
  const int socket = socket_;
  socket_ = -1;
  // Let Python observe request EOF and close the simulator normally while its
  // response direction remains valid for final buffered output.
  if (socket >= 0)
    (void)::shutdown(socket, SHUT_WR);
  if (child_pid_ < 0) {
    if (socket >= 0)
      ::close(socket);
    return;
  }
  const pid_t pid = static_cast<pid_t>(child_pid_);
  int status = 0;
  for (int attempt = 0; attempt < 40; ++attempt) {
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      child_pid_ = -1;
      if (socket >= 0)
        ::close(socket);
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  (void)::kill(pid, SIGTERM);
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  child_pid_ = -1;
  if (socket >= 0)
    ::close(socket);
}

} // namespace cerebellum
