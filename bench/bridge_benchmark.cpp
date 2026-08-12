#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cerebellum/python_chunk_generator.hpp"

using namespace cerebellum;

namespace {

struct Arguments {
    PythonRunner runner = PythonRunner::Synthetic;
    std::string device = "cuda";
    std::string model = "lerobot/smolvla_base";
    std::string output;
    int warmup = 5;
    int iterations = 50;
    bool local_files_only = true;
};

class StaticObservationSource final : public ObservationSource {
public:
    std::shared_ptr<const ObservationSnapshot> latest() noexcept override { return value_; }
    void publish(std::shared_ptr<const ObservationSnapshot> value) { value_ = std::move(value); }

private:
    std::shared_ptr<const ObservationSnapshot> value_;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&](std::string_view option) -> std::string_view {
            if (++i >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
            return argv[i];
        };
        if (arg == "--runner") {
            const std::string_view name = value(arg);
            if (name == "synthetic") args.runner = PythonRunner::Synthetic;
            else if (name == "smolvla") args.runner = PythonRunner::SmolVla;
            else throw std::invalid_argument("--runner must be synthetic or smolvla");
        } else if (arg == "--device") {
            args.device = value(arg);
        } else if (arg == "--model") {
            args.model = value(arg);
        } else if (arg == "--warmup") {
            args.warmup = std::stoi(std::string(value(arg)));
        } else if (arg == "--iterations") {
            args.iterations = std::stoi(std::string(value(arg)));
        } else if (arg == "--output") {
            args.output = value(arg);
        } else if (arg == "--allow-download") {
            args.local_files_only = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (args.warmup < 0 || args.iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations positive");
    }
    return args;
}

std::shared_ptr<const ObservationSnapshot> make_observation(
    const WorkerObservationSchema& schema) {
    auto observation = std::make_shared<ObservationSnapshot>();
    observation->sequence = 1;
    observation->capture_time = now();
    observation->task = "move the object to the target";

    if (schema.constrained()) {
        observation->state.assign(schema.state_dim, 0.0F);
        for (const WorkerImageSpec& expected : schema.images) {
            CameraImage image;
            image.feature_name = expected.feature_name;
            image.channels = expected.channels;
            image.height = expected.height;
            image.width = expected.width;
            image.pixels.resize(static_cast<std::size_t>(image.channels) * image.height * image.width);
            observation->images.push_back(std::move(image));
        }
    } else {
        // Match smolvla_base so synthetic vs real subtracts the model while
        // retaining equivalent Cerebellum serialization and IPC work.
        observation->state.assign(6, 0.0F);
        for (int camera = 1; camera <= 3; ++camera) {
            CameraImage image;
            image.feature_name = "observation.images.camera" + std::to_string(camera);
            image.channels = 3;
            image.height = 256;
            image.width = 256;
            image.pixels.resize(3U * 256U * 256U);
            observation->images.push_back(std::move(image));
        }
    }
    observation->validate();
    return observation;
}

struct Summary {
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

Summary summarize(std::vector<std::int64_t> values) {
    if (values.empty()) return {};
    std::sort(values.begin(), values.end());
    const auto percentile = [&](int percent) {
        const std::size_t rank =
            (static_cast<std::size_t>(percent) * values.size() + 99U) / 100U;
        return values[std::max<std::size_t>(1, rank) - 1];
    };
    return Summary{
        percentile(50) / 1e6,
        percentile(90) / 1e6,
        percentile(99) / 1e6,
        values.back() / 1e6,
    };
}

template <typename Getter>
Summary field(const std::vector<BridgeTiming>& samples, Getter getter) {
    std::vector<std::int64_t> values;
    values.reserve(samples.size());
    for (const BridgeTiming& timing : samples) values.push_back(getter(timing));
    return summarize(std::move(values));
}

void write_summary(std::ostream& out, std::string_view name, const Summary& summary,
                   bool trailing_comma = true) {
    out << "    \"" << name << "\": {\"p50\": " << summary.p50_ms
        << ", \"p90\": " << summary.p90_ms << ", \"p99\": " << summary.p99_ms
        << ", \"max\": " << summary.max_ms << "}";
    if (trailing_comma) out << ',';
    out << '\n';
}

void report(std::ostream& out, const Arguments& args, const std::vector<BridgeTiming>& samples) {
    const Summary total = field(samples, [](const BridgeTiming& t) { return t.total_ns; });
    const Summary model = field(samples, [](const BridgeTiming& t) { return t.python_model_ns; });
    const Summary overhead =
        field(samples, [](const BridgeTiming& t) { return t.cerebellum_overhead_ns(); });
    const double model_share = total.p50_ms > 0.0 ? 100.0 * model.p50_ms / total.p50_ms : 0.0;

    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"runner\": \""
        << (args.runner == PythonRunner::Synthetic ? "synthetic" : "smolvla") << "\",\n"
        << "  \"device\": \"" << args.device << "\",\n"
        << "  \"warmup\": " << args.warmup << ",\n"
        << "  \"iterations\": " << args.iterations << ",\n"
        << "  \"units\": \"milliseconds\",\n"
        << "  \"metrics\": {\n";
    write_summary(out, "total_observation_lookup_to_chunk", total);
    write_summary(out, "python_model", model);
    write_summary(out, "cerebellum_overhead", overhead);
    write_summary(out, "observation_lookup",
                  field(samples, [](const BridgeTiming& t) { return t.observation_lookup_ns; }));
    write_summary(out, "observation_validation",
                  field(samples, [](const BridgeTiming& t) { return t.observation_validation_ns; }));
    write_summary(out, "cpp_request_encode",
                  field(samples, [](const BridgeTiming& t) { return t.request_encode_ns; }));
    write_summary(out, "cpp_request_write",
                  field(samples, [](const BridgeTiming& t) { return t.request_write_ns; }));
    write_summary(out, "response_wait",
                  field(samples, [](const BridgeTiming& t) { return t.response_wait_ns; }));
    write_summary(out, "cpp_response_decode",
                  field(samples, [](const BridgeTiming& t) { return t.response_decode_ns; }));
    write_summary(out, "python_request_decode",
                  field(samples, [](const BridgeTiming& t) { return t.python_request_decode_ns; }));
    write_summary(out, "python_observation_conversion",
                  field(samples, [](const BridgeTiming& t) { return t.python_observation_ns; }));
    write_summary(out, "python_response_encode",
                  field(samples, [](const BridgeTiming& t) { return t.python_response_encode_ns; }),
                  false);
    out << "  },\n"
        << "  \"p50_share_percent\": {\"model\": " << model_share
        << ", \"cerebellum\": " << 100.0 - model_share << "}\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        RuntimeConfig config;
        StaticObservationSource source;
        PythonChunkGeneratorOptions options;
        options.python_executable = CEREBELLUM_BENCH_PYTHON;
        options.python_package_path = CEREBELLUM_BENCH_PYTHONPATH;
        options.runner = args.runner;
        options.model = args.model;
        options.device = args.device;
        options.local_files_only = args.local_files_only;
        options.startup_timeout = std::chrono::minutes(15);
        options.inference_timeout = std::chrono::minutes(2);

        PythonChunkGenerator generator(config, source, options);
        source.publish(make_observation(generator.observation_schema()));

        const int total_calls = args.warmup + args.iterations;
        std::vector<BridgeTiming> samples;
        samples.reserve(static_cast<std::size_t>(args.iterations));
        for (int iteration = 0; iteration < total_calls; ++iteration) {
            const std::int64_t first_step = static_cast<std::int64_t>(iteration) * config.chunk_size;
            Chunk chunk;
            if (!generator.generate(
                    InferenceRequest{first_step, first_step - 1, config.chunk_size,
                                     Stitching::Discard},
                    chunk)) {
                throw std::runtime_error("generation failed: " + generator.last_error());
            }
            if (iteration >= args.warmup) samples.push_back(generator.last_timing());
        }

        if (args.output.empty()) {
            report(std::cout, args, samples);
        } else {
            std::ofstream output(args.output);
            if (!output) throw std::runtime_error("cannot open output file: " + args.output);
            report(output, args, samples);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "bridge_benchmark: " << exc.what() << '\n';
        return 1;
    }
}
