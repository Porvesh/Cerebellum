#include "cerebellum/python_chunk_generator.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <string_view>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

extern char** environ;

namespace cerebellum {
namespace {

constexpr std::uint16_t kProtocolVersion = 2;
constexpr std::uint32_t kMaxFrameBytes = 16U * 1024U * 1024U;
constexpr std::string_view kHelloMagic = "CBHI";
constexpr std::string_view kRequestMagic = "CBRQ";
constexpr std::string_view kResponseMagic = "CBRS";

void append_u16(std::string& out, std::uint16_t value) {
    const std::uint16_t encoded = htons(value);
    out.append(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
}

void append_u32(std::string& out, std::uint32_t value) {
    const std::uint32_t encoded = htonl(value);
    out.append(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
}

void append_u64(std::string& out, std::uint64_t value) {
    append_u32(out, static_cast<std::uint32_t>(value >> 32));
    append_u32(out, static_cast<std::uint32_t>(value));
}

void append_f32(std::string& out, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}

class Reader {
public:
    explicit Reader(std::string_view data) : data_(data) {}

    std::string_view bytes(std::size_t count) {
        if (count > data_.size() - offset_) throw std::runtime_error("truncated message");
        const auto value = data_.substr(offset_, count);
        offset_ += count;
        return value;
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
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    bool empty() const noexcept { return offset_ == data_.size(); }

private:
    std::string_view data_;
    std::size_t offset_ = 0;
};

std::uint8_t stitching_value(Stitching stitching) {
    switch (stitching) {
        case Stitching::Discard: return 0;
        case Stitching::Ensemble: return 1;
        case Stitching::Rtc: return 2;
    }
    return std::numeric_limits<std::uint8_t>::max();
}

}  // namespace

PythonChunkGenerator::PythonChunkGenerator(const RuntimeConfig& config,
                                           ObservationSource& observations,
                                           PythonChunkGeneratorOptions options)
    : config_(config), observations_(observations), options_(std::move(options)) {
    config_.validate();
    if (options_.timeout <= std::chrono::milliseconds::zero()) {
        throw std::invalid_argument("Python worker timeout must be positive");
    }

    int sockets[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
        throw std::runtime_error(std::string("socketpair failed: ") + std::strerror(errno));
    }

    const std::string code =
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from cerebellum_model.server import main; main()";
    std::vector<std::string> arguments{
        options_.python_executable,
        "-c",
        code,
        options_.python_package_path,
        "--runner",
        "synthetic",
        "--chunk-size",
        std::to_string(config_.chunk_size),
        "--model-dim",
        std::to_string(config_.padded_action_dim),
        "--robot-dim",
        std::to_string(config_.action_dim),
    };
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (std::string& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    int spawn_error = ::posix_spawn_file_actions_init(&actions);
    const bool actions_initialized = spawn_error == 0;
    if (spawn_error == 0) spawn_error = ::posix_spawn_file_actions_addclose(&actions, sockets[0]);
    if (spawn_error == 0) {
        spawn_error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    }
    if (spawn_error == 0) {
        spawn_error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    }
    if (spawn_error == 0) spawn_error = ::posix_spawn_file_actions_addclose(&actions, sockets[1]);

    pid_t pid = -1;
    if (spawn_error == 0) {
        spawn_error = ::posix_spawnp(&pid, options_.python_executable.c_str(), &actions,
                                     nullptr, argv.data(), environ);
    }
    if (actions_initialized) ::posix_spawn_file_actions_destroy(&actions);
    if (spawn_error != 0) {
        ::close(sockets[0]);
        ::close(sockets[1]);
        throw std::runtime_error(std::string("starting Python worker failed: ") +
                                 std::strerror(spawn_error));
    }

    ::close(sockets[1]);
    socket_ = sockets[0];
    child_pid_ = static_cast<int>(pid);
    if (!read_handshake()) {
        const std::string error = last_error_;
        stop_child();
        throw std::runtime_error("Python worker handshake failed: " + error);
    }
}

PythonChunkGenerator::~PythonChunkGenerator() { stop_child(); }

bool PythonChunkGenerator::generate(const InferenceRequest& request, Chunk& out) noexcept {
    try {
        if (!healthy()) {
            last_error_ = "Python worker is not running";
            return false;
        }

        const std::shared_ptr<const ObservationSnapshot> observation = observations_.latest();
        if (!observation) {
            last_error_ = "no observation is available";
            return false;
        }
        try {
            observation->validate();
        } catch (const std::exception& exc) {
            last_error_ = std::string("invalid observation: ") + exc.what();
            return false;
        }
        if (observation->state.size() > std::numeric_limits<std::uint32_t>::max() ||
            observation->images.size() > std::numeric_limits<std::uint16_t>::max() ||
            observation->task.size() > std::numeric_limits<std::uint32_t>::max()) {
            last_error_ = "observation metadata exceeds protocol limits";
            return false;
        }

        const std::uint64_t request_id = next_request_id_++;
        std::string payload;
        payload.reserve(66 + observation->state.size() * sizeof(float) +
                        observation->task.size());
        payload.append(kRequestMagic);
        append_u16(payload, kProtocolVersion);
        payload.push_back(static_cast<char>(stitching_value(request.stitching)));
        payload.push_back('\0');
        append_u64(payload, request_id);
        append_u64(payload, static_cast<std::uint64_t>(request.first_step));
        append_u64(payload, static_cast<std::uint64_t>(request.last_emitted_step));
        append_u32(payload, static_cast<std::uint32_t>(request.action_count));
        append_u32(payload, options_.seed);
        append_u64(payload, observation->sequence);
        append_u64(payload, static_cast<std::uint64_t>(std::chrono::duration_cast<Nanos>(
                                observation->capture_time.time_since_epoch()).count()));
        append_u32(payload, static_cast<std::uint32_t>(observation->state.size()));
        append_u16(payload, static_cast<std::uint16_t>(observation->images.size()));
        append_u32(payload, static_cast<std::uint32_t>(observation->task.size()));
        for (float value : observation->state) append_f32(payload, value);
        payload.append(observation->task);
        for (const CameraImage& image : observation->images) {
            if (image.feature_name.size() > std::numeric_limits<std::uint16_t>::max() ||
                image.pixels.size() > std::numeric_limits<std::uint32_t>::max()) {
                last_error_ = "camera metadata exceeds protocol limits";
                return false;
            }
            append_u16(payload, static_cast<std::uint16_t>(image.feature_name.size()));
            append_u16(payload, image.channels);
            append_u16(payload, image.height);
            append_u16(payload, image.width);
            append_u32(payload, static_cast<std::uint32_t>(image.pixels.size()));
            payload.append(image.feature_name);
            payload.append(reinterpret_cast<const char*>(image.pixels.data()), image.pixels.size());
        }
        if (!write_frame(payload.data(), payload.size())) return false;

        std::string response;
        if (!read_frame(response)) return false;
        Reader reader(response);
        if (reader.bytes(4) != kResponseMagic) throw std::runtime_error("bad response magic");
        if (reader.u16() != kProtocolVersion) throw std::runtime_error("protocol version mismatch");
        const std::uint8_t status = reader.u8();
        (void)reader.u8();
        const std::uint64_t received_id = reader.u64();
        const std::int64_t first_step = reader.i64();
        const std::uint64_t observation_sequence = reader.u64();
        const std::int64_t capture_ns = reader.i64();
        const std::uint32_t count = reader.u32();
        const std::uint16_t model_dim = reader.u16();
        const std::uint16_t robot_dim = reader.u16();
        const std::uint32_t error_size = reader.u32();

        if (received_id != request_id) throw std::runtime_error("response request ID mismatch");
        if (status != 0) {
            last_error_ = std::string(reader.bytes(error_size));
            return false;
        }
        if (error_size != 0 || first_step != request.first_step ||
            count != static_cast<std::uint32_t>(request.action_count) ||
            count > static_cast<std::uint32_t>(kChunkSize) ||
            model_dim != static_cast<std::uint16_t>(config_.padded_action_dim) ||
            robot_dim != static_cast<std::uint16_t>(config_.action_dim)) {
            throw std::runtime_error("Python response shape or timeline mismatch");
        }

        for (std::uint32_t i = 0; i < count; ++i) {
            for (std::uint16_t d = 0; d < model_dim; ++d) {
                const float value = reader.f32();
                if (!std::isfinite(value)) throw std::runtime_error("non-finite model action");
                out.model_actions[i][d] = value;
            }
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            out.actions[i].fill(0.0F);
            for (std::uint16_t d = 0; d < robot_dim; ++d) {
                const float value = reader.f32();
                if (!std::isfinite(value)) throw std::runtime_error("non-finite robot action");
                out.actions[i][d] = value;
            }
        }
        if (!reader.empty()) throw std::runtime_error("response contains trailing bytes");

        out.count = static_cast<int>(count);
        out.stamps.obs_seq = observation_sequence;
        out.stamps.t_obs_capture = TimePoint{Nanos{capture_ns}};
        last_error_.clear();
        return true;
    } catch (const std::exception& exc) {
        transport_error(exc.what());
        return false;
    } catch (...) {
        transport_error("unknown Python bridge failure");
        return false;
    }
}

bool PythonChunkGenerator::read_handshake() {
    std::string payload;
    if (!read_frame(payload)) return false;
    try {
        Reader reader(payload);
        if (reader.bytes(4) != kHelloMagic) throw std::runtime_error("bad handshake magic");
        if (reader.u16() != kProtocolVersion) throw std::runtime_error("protocol version mismatch");
        const std::uint8_t status = reader.u8();
        (void)reader.u8();
        const std::uint32_t chunk_size = reader.u32();
        const std::uint16_t model_dim = reader.u16();
        const std::uint16_t robot_dim = reader.u16();
        const std::uint32_t error_size = reader.u32();
        if (status != 0) {
            last_error_ = std::string(reader.bytes(error_size));
            return false;
        }
        if (error_size != 0 || !reader.empty()) throw std::runtime_error("invalid handshake length");
        if (chunk_size != static_cast<std::uint32_t>(config_.chunk_size) ||
            model_dim != static_cast<std::uint16_t>(config_.padded_action_dim) ||
            robot_dim != static_cast<std::uint16_t>(config_.action_dim)) {
            throw std::runtime_error("Python worker model shape does not match RuntimeConfig");
        }
        return true;
    } catch (const std::exception& exc) {
        last_error_ = exc.what();
        return false;
    }
}

bool PythonChunkGenerator::write_frame(const void* data, std::size_t size) {
    if (size > kMaxFrameBytes) {
        transport_error("outgoing frame exceeds limit");
        return false;
    }
    const std::uint32_t encoded_size = htonl(static_cast<std::uint32_t>(size));
    return write_all(&encoded_size, sizeof(encoded_size)) && write_all(data, size);
}

bool PythonChunkGenerator::read_frame(std::string& payload) {
    std::uint32_t encoded_size = 0;
    if (!read_all(&encoded_size, sizeof(encoded_size))) return false;
    const std::uint32_t size = ntohl(encoded_size);
    if (size > kMaxFrameBytes) {
        transport_error("incoming frame exceeds limit");
        return false;
    }
    payload.resize(size);
    return size == 0 || read_all(payload.data(), size);
}

bool PythonChunkGenerator::write_all(const void* data, std::size_t size) {
    const auto* cursor = static_cast<const char*>(data);
    while (size > 0) {
        if (!wait_for(POLLOUT)) return false;
        const ssize_t written = ::send(socket_, cursor, size, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            transport_error(std::string("write to Python worker failed: ") + std::strerror(errno));
            return false;
        }
        cursor += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}

bool PythonChunkGenerator::read_all(void* data, std::size_t size) {
    auto* cursor = static_cast<char*>(data);
    while (size > 0) {
        if (!wait_for(POLLIN)) return false;
        const ssize_t received = ::recv(socket_, cursor, size, 0);
        if (received == 0) {
            transport_error("Python worker closed the connection");
            return false;
        }
        if (received < 0) {
            if (errno == EINTR) continue;
            transport_error(std::string("read from Python worker failed: ") + std::strerror(errno));
            return false;
        }
        cursor += received;
        size -= static_cast<std::size_t>(received);
    }
    return true;
}

bool PythonChunkGenerator::wait_for(short events) {
    pollfd descriptor{socket_, events, 0};
    int result = 0;
    do {
        result = ::poll(&descriptor, 1, static_cast<int>(options_.timeout.count()));
    } while (result < 0 && errno == EINTR);
    if (result == 0) {
        transport_error("Python worker timed out");
        return false;
    }
    if (result < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        transport_error(result < 0
                            ? std::string("poll failed: ") + std::strerror(errno)
                            : "Python worker connection failed");
        return false;
    }
    return true;
}

void PythonChunkGenerator::transport_error(std::string message) noexcept {
    try {
        last_error_ = std::move(message);
    } catch (...) {
        // Still invalidate the transport if diagnostics cannot be allocated.
    }
    if (socket_ >= 0) {
        ::close(socket_);
        socket_ = -1;
    }
}

void PythonChunkGenerator::stop_child() noexcept {
    if (socket_ >= 0) {
        ::shutdown(socket_, SHUT_RDWR);
        ::close(socket_);
        socket_ = -1;
    }
    if (child_pid_ < 0) return;

    const pid_t pid = static_cast<pid_t>(child_pid_);
    int status = 0;
    for (int attempt = 0; attempt < 20; ++attempt) {
        const pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) {
            child_pid_ = -1;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    (void)::kill(pid, SIGTERM);
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    child_pid_ = -1;
}

}  // namespace cerebellum
