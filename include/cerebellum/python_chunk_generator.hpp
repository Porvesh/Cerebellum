#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "cerebellum/loop.hpp"
#include "cerebellum/observation.hpp"

namespace cerebellum {

struct PythonChunkGeneratorOptions {
    std::string python_executable = "python3";
    std::string python_package_path = "python";
    std::chrono::milliseconds timeout{5'000};
    std::uint32_t seed = 0;
};

// A persistent, crash-isolated Python worker behind the existing inference
// interface. Construction starts the child and validates its model shape;
// generate() is called only by RuntimeLoop's non-real-time inference worker.
class PythonChunkGenerator final : public ChunkGenerator {
public:
    PythonChunkGenerator(const RuntimeConfig& config, ObservationSource& observations,
                         PythonChunkGeneratorOptions options = {});
    ~PythonChunkGenerator() override;

    PythonChunkGenerator(const PythonChunkGenerator&) = delete;
    PythonChunkGenerator& operator=(const PythonChunkGenerator&) = delete;

    bool generate(const InferenceRequest& request, Chunk& out) noexcept override;

    bool healthy() const noexcept { return socket_ >= 0; }
    const std::string& last_error() const noexcept { return last_error_; }

private:
    bool read_handshake();
    bool write_frame(const void* data, std::size_t size);
    bool read_frame(std::string& payload);
    bool write_all(const void* data, std::size_t size);
    bool read_all(void* data, std::size_t size);
    bool wait_for(short events);
    void transport_error(std::string message) noexcept;
    void stop_child() noexcept;

    RuntimeConfig config_;
    ObservationSource& observations_;
    PythonChunkGeneratorOptions options_;
    int socket_ = -1;
    int child_pid_ = -1;
    std::uint64_t next_request_id_ = 1;
    std::string last_error_;
};

}  // namespace cerebellum
