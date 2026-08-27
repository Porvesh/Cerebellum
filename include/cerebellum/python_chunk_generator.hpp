#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cerebellum/loop.hpp"
#include "cerebellum/observation.hpp"

namespace cerebellum {

enum class PythonRunner { Synthetic, SmolVla };

struct WorkerImageSpec {
    std::string feature_name;
    std::uint16_t channels = 0;
    std::uint16_t height = 0;
    std::uint16_t width = 0;
};

struct WorkerObservationSchema {
    std::uint16_t state_dim = 0;
    std::vector<WorkerImageSpec> images;

    bool constrained() const noexcept { return state_dim != 0 || !images.empty(); }
};

// One successful generate() call, split at boundaries owned by Cerebellum and
// by the Python model worker. `total - python_model` is the headline
// Cerebellum overhead; the individual fields explain where that overhead went.
struct BridgeTiming {
    std::uint64_t request_id = 0;
    std::int64_t total_ns = 0;
    std::int64_t observation_lookup_ns = 0;
    std::int64_t observation_validation_ns = 0;
    std::int64_t request_encode_ns = 0;
    std::int64_t request_write_ns = 0;
    std::int64_t response_wait_ns = 0;
    std::int64_t response_decode_ns = 0;
    std::int64_t python_request_decode_ns = 0;
    std::int64_t python_observation_ns = 0;
    std::int64_t python_model_ns = 0;
    std::int64_t python_response_encode_ns = 0;

    std::int64_t cerebellum_overhead_ns() const noexcept {
        return total_ns > python_model_ns ? total_ns - python_model_ns : 0;
    }
};

struct PythonChunkGeneratorOptions {
    std::string python_executable = "python3";
    std::string python_package_path = "python";
    PythonRunner runner = PythonRunner::Synthetic;
    std::string model = "lerobot/smolvla_base";
    std::string device = "cuda";
    bool local_files_only = false;
    std::chrono::milliseconds startup_timeout{600'000};
    std::chrono::milliseconds inference_timeout{30'000};
    std::uint32_t seed = 0;
    // Keep benchmarks/replays bit-repeatable by default. Live policy adapters
    // may opt into a deterministic per-observation diffusion seed so repeated
    // replans do not reuse an identical noise trajectory.
    bool vary_seed_by_observation = false;
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
    const WorkerObservationSchema& observation_schema() const noexcept { return schema_; }
    const BridgeTiming& last_timing() const noexcept { return last_timing_; }

private:
    bool read_handshake();
    bool write_frame(const void* data, std::size_t size);
    bool read_frame(std::string& payload);
    bool write_all(const void* data, std::size_t size);
    bool read_all(void* data, std::size_t size);
    bool wait_for(short events);
    void transport_error(std::string message) noexcept;
    void stop_child() noexcept;
    bool observation_matches_schema(const ObservationSnapshot& observation);

    RuntimeConfig config_;
    ObservationSource& observations_;
    PythonChunkGeneratorOptions options_;
    WorkerObservationSchema schema_;
    std::chrono::milliseconds io_timeout_;
    int socket_ = -1;
    int child_pid_ = -1;
    std::uint64_t next_request_id_ = 1;
    std::string last_error_;
    BridgeTiming last_timing_{};
};

}  // namespace cerebellum
