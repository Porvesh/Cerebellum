#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "cerebellum/loop.hpp"
#include "cerebellum/observation.hpp"
#include "cerebellum/python_chunk_generator.hpp"

using namespace cerebellum;

namespace {

struct Arguments {
    PythonRunner runner = PythonRunner::Synthetic;
    std::string device = "cuda";
    std::string model = "lerobot/smolvla_base";
    std::string output;
    int ticks = 300;
    int warmup_inferences = 2;
    int refresh_trigger = 6;
    double synthetic_inference_ms = 149.0;
    RefreshPolicy refresh_policy = RefreshPolicy::Tail;
    bool local_files_only = true;
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
        } else if (arg == "--ticks") {
            args.ticks = std::stoi(std::string(value(arg)));
        } else if (arg == "--warmup-inferences") {
            args.warmup_inferences = std::stoi(std::string(value(arg)));
        } else if (arg == "--refresh-trigger") {
            args.refresh_trigger = std::stoi(std::string(value(arg)));
        } else if (arg == "--refresh-policy") {
            const std::string_view name = value(arg);
            if (name == "tail") args.refresh_policy = RefreshPolicy::Tail;
            else if (name == "continuous") args.refresh_policy = RefreshPolicy::Continuous;
            else throw std::invalid_argument("--refresh-policy must be tail or continuous");
        } else if (arg == "--inference-ms") {
            args.synthetic_inference_ms = std::stod(std::string(value(arg)));
        } else if (arg == "--output") {
            args.output = value(arg);
        } else if (arg == "--allow-download") {
            args.local_files_only = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (args.ticks <= 0 || args.warmup_inferences < 0 ||
        args.synthetic_inference_ms < 0.0) {
        throw std::invalid_argument(
            "ticks must be positive; warmup and inference time must be non-negative");
    }
    if (args.refresh_trigger <= 0 || args.refresh_trigger >= kChunkSize) {
        throw std::invalid_argument("refresh trigger must be between 1 and chunk_size - 1");
    }
    return args;
}

class CountingSink final : public ActionSink {
public:
    explicit CountingSink(std::size_t capacity) : emit_lateness(capacity) {}

    void emit(const ActionEmission& emission) noexcept override {
        ++emissions;
        if (emission.fallback) ++fallbacks;
        emit_lateness.record((now() - emission.deadline).count());
    }

    std::uint64_t emissions = 0;
    std::uint64_t fallbacks = 0;
    PercentileRecorder emit_lateness;
};

class DelayedGenerator final : public ChunkGenerator {
public:
    DelayedGenerator(int chunk_size, Nanos delay) : chunk_size_(chunk_size), delay_(delay) {}

    bool generate(const InferenceRequest& request, Chunk& out) noexcept override {
        const TimePoint capture = now();
        const std::uint64_t observation = ++calls_;
        std::this_thread::sleep_for(delay_);
        out.count = chunk_size_;
        out.stamps.obs_seq = observation;
        out.stamps.t_obs_capture = capture;
        for (int i = 0; i < chunk_size_; ++i) {
            const float value = static_cast<float>(request.first_step + i) +
                                static_cast<float>(observation) * 0.01F;
            for (std::size_t d = 0; d < kPaddedActionDim; ++d) {
                out.actions[static_cast<std::size_t>(i)][d] = value;
                out.model_actions[static_cast<std::size_t>(i)][d] = value;
            }
        }
        return true;
    }

private:
    int chunk_size_;
    Nanos delay_;
    std::uint64_t calls_ = 0;
};

class RefreshingObservationSource final : public ObservationSource {
public:
    void configure(const WorkerObservationSchema& schema) {
        value_ = std::make_shared<ObservationSnapshot>();
        value_->sequence = ++sequence_;
        value_->capture_time = now();
        value_->task = "move the object to the target";
        value_->state.assign(schema.state_dim, 0.0F);
        for (const WorkerImageSpec& expected : schema.images) {
            CameraImage image;
            image.feature_name = expected.feature_name;
            image.channels = expected.channels;
            image.height = expected.height;
            image.width = expected.width;
            image.pixels.resize(static_cast<std::size_t>(image.channels) * image.height * image.width);
            value_->images.push_back(std::move(image));
        }
        value_->validate();
    }

    std::shared_ptr<const ObservationSnapshot> latest() noexcept override {
        if (!value_) return {};
        value_->sequence = ++sequence_;
        value_->capture_time = now();
        return value_;
    }

private:
    std::shared_ptr<ObservationSnapshot> value_;
    std::uint64_t sequence_ = 0;
};

std::uint64_t positive_samples(const PercentileRecorder& recorder) {
    return static_cast<std::uint64_t>(std::count_if(
        recorder.samples().begin(), recorder.samples().end(),
        [](std::int64_t sample) { return sample > 0; }));
}

void write_summary(std::ostream& out, std::string_view name, const Summary& summary,
                   double divisor, bool trailing_comma = true) {
    out << "    \"" << name << "\": {\"min\": " << summary.min_ns / divisor
        << ", \"p50\": " << summary.p50_ns / divisor
        << ", \"p90\": " << summary.p90_ns / divisor
        << ", \"p99\": " << summary.p99_ns / divisor
        << ", \"max\": " << summary.max_ns / divisor << "}";
    if (trailing_comma) out << ',';
    out << '\n';
}

void report(std::ostream& out, const Arguments& args, RuntimeLoop& loop,
            CountingSink& sink, Nanos wall_time) {
    ControlMetrics& metrics = loop.metrics();
    const Summary lateness = sink.emit_lateness.summarize();
    const Summary staleness = metrics.staleness.summarize();
    const Summary ready_to_emit = metrics.ready_to_emit.summarize();
    const Summary seam = loop.queue().seam_linf().summarize();
    const InferenceStats& inference = loop.inference_stats();
    const ProducerStats& producer = loop.queue().producer_stats();
    const ConsumerStats& consumer = loop.queue().consumer_stats();
    const std::uint64_t real_actions = sink.emissions - sink.fallbacks;
    const double wall_ns = static_cast<double>(wall_time.count());
    const double wall_seconds = wall_ns / 1e9;

    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"runner\": \""
        << (args.runner == PythonRunner::Synthetic ? "synthetic" : "smolvla") << "\",\n"
        << "  \"stitching\": \"discard\",\n"
        << "  \"refresh_policy\": \""
        << (args.refresh_policy == RefreshPolicy::Tail ? "tail" : "continuous") << "\",\n"
        << "  \"device\": \"" << args.device << "\",\n"
        << "  \"ticks_requested\": " << args.ticks << ",\n"
        << "  \"control_hz\": " << kControlHz << ",\n"
        << "  \"staleness_bound_ms\": " << kMaxStalenessMs << ",\n"
        << "  \"chunk_size\": " << kChunkSize << ",\n"
        << "  \"refresh_trigger\": " << args.refresh_trigger << ",\n"
        << "  \"warmup_inferences\": ";
    if (args.runner == PythonRunner::SmolVla) out << args.warmup_inferences;
    else out << "null";
    out << ",\n"
        << "  \"synthetic_inference_ms\": ";
    if (args.runner == PythonRunner::Synthetic) out << args.synthetic_inference_ms;
    else out << "null";
    out << ",\n"
        << "  \"wall_time_ms\": " << to_ms(wall_time) << ",\n"
        << "  \"timing_ms\": {\n";
    write_summary(out, "action_emit_lateness", lateness, 1e6);
    write_summary(out, "observation_to_action_staleness", staleness, 1e6);
    write_summary(out, "chunk_ready_to_action_emit", ready_to_emit, 1e6, false);
    out << "  },\n"
        << "  \"action_metrics\": {\n";
    write_summary(out, "discard_seam_linf", seam,
                  static_cast<double>(ActionChunkQueue::kSeamScale), false);
    out << "  },\n"
        << "  \"counts\": {\n"
        << "    \"ticks\": " << metrics.ticks << ",\n"
        << "    \"strict_deadline_misses\": " << positive_samples(sink.emit_lateness) << ",\n"
        << "    \"real_actions\": " << real_actions << ",\n"
        << "    \"fallback_actions\": " << sink.fallbacks << ",\n"
        << "    \"underruns\": " << metrics.underruns << ",\n"
        << "    \"staleness_violations\": " << metrics.staleness_violations << ",\n"
        << "    \"steps_skipped\": " << consumer.steps_skipped << ",\n"
        << "    \"inference_requests\": " << inference.requests << ",\n"
        << "    \"chunks_generated\": " << inference.generated << ",\n"
        << "    \"generation_failures\": " << inference.generation_failed << ",\n"
        << "    \"invalid_chunks\": " << inference.invalid_chunks << ",\n"
        << "    \"publish_retries\": " << inference.publish_retries << ",\n"
        << "    \"chunks_published\": " << producer.chunks_published << ",\n"
        << "    \"publish_failures\": " << producer.publish_failed << ",\n"
        << "    \"acquire_failures\": " << producer.acquire_failed << ",\n"
        << "    \"chunks_accepted\": " << consumer.chunks_accepted << ",\n"
        << "    \"chunks_superseded\": " << consumer.chunks_superseded << ",\n"
        << "    \"actions_discarded\": " << consumer.actions_discarded << "\n"
        << "  },\n"
        << "  \"inference_load\": {\n"
        << "    \"generation_wall_ms\": " << inference.generation_wall_ns / 1e6 << ",\n"
        << "    \"generated_per_second\": "
        << (wall_seconds > 0.0 ? inference.generated / wall_seconds : 0.0) << ",\n"
        << "    \"worker_busy_percent\": "
        << (wall_ns > 0.0 ? 100.0 * inference.generation_wall_ns / wall_ns : 0.0) << "\n"
        << "  },\n"
        << "  \"rates_percent\": {\n"
        << "    \"underrun\": "
        << (metrics.ticks ? 100.0 * metrics.underruns / metrics.ticks : 0.0) << ",\n"
        << "    \"staleness_violation_of_real_actions\": "
        << (real_actions ? 100.0 * metrics.staleness_violations / real_actions : 0.0)
        << "\n  }\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        RuntimeConfig config;
        config.stitching = Stitching::Discard;
        config.refresh_policy = args.refresh_policy;
        config.refresh_trigger = args.refresh_trigger;
        config.queue_capacity = config.chunk_size + config.refresh_trigger;
        config.validate();

        RefreshingObservationSource observations;
        std::unique_ptr<DelayedGenerator> delayed;
        std::unique_ptr<PythonChunkGenerator> python;
        ChunkGenerator* generator = nullptr;

        if (args.runner == PythonRunner::Synthetic) {
            const Nanos delay = std::chrono::duration_cast<Nanos>(
                std::chrono::duration<double, std::milli>(args.synthetic_inference_ms));
            delayed = std::make_unique<DelayedGenerator>(config.chunk_size, delay);
            generator = delayed.get();
        } else {
            PythonChunkGeneratorOptions options;
            options.python_executable = CEREBELLUM_BENCH_PYTHON;
            options.python_package_path = CEREBELLUM_BENCH_PYTHONPATH;
            options.runner = PythonRunner::SmolVla;
            options.model = args.model;
            options.device = args.device;
            options.local_files_only = args.local_files_only;
            options.startup_timeout = std::chrono::minutes(15);
            options.inference_timeout = std::chrono::minutes(2);
            python = std::make_unique<PythonChunkGenerator>(config, observations, options);
            observations.configure(python->observation_schema());
            for (int i = 0; i < args.warmup_inferences; ++i) {
                Chunk warmup;
                if (!python->generate(
                        InferenceRequest{0, -1, config.chunk_size, Stitching::Discard}, warmup)) {
                    throw std::runtime_error("warmup failed: " + python->last_error());
                }
            }
            generator = python.get();
        }

        CountingSink sink(static_cast<std::size_t>(args.ticks));
        RuntimeLoop loop(config, *generator, sink, static_cast<std::size_t>(args.ticks));
        const TimePoint started = now();
        loop.run_for(static_cast<std::size_t>(args.ticks));
        const Nanos wall_time = now() - started;

        if (args.output.empty()) {
            report(std::cout, args, loop, sink, wall_time);
        } else {
            std::ofstream output(args.output);
            if (!output) throw std::runtime_error("cannot open output file: " + args.output);
            report(output, args, loop, sink, wall_time);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "runtime_benchmark: " << exc.what() << '\n';
        return 1;
    }
}
