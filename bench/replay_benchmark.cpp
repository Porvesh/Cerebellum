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

#include "cerebellum/python_chunk_generator.hpp"
#include "cerebellum/replay_episode.hpp"

using namespace cerebellum;

namespace {

struct Arguments {
    std::string episode;
    std::string actions_output;
    std::string report_output;
    std::string model = "lerobot/smolvla_base";
    std::string device = "cpu";
    PythonRunner runner = PythonRunner::Synthetic;
    Stitching stitching = Stitching::Discard;
    int warmup_inferences = 0;
    int rtc_denoise_steps = 5;
    int rtc_inference_delay = 8;
    int rtc_execution_horizon = 8;
    bool local_files_only = true;
};

Arguments parse_arguments(int argc, char **argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&](std::string_view option) -> std::string_view {
            if (++i >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
            return argv[i];
        };
        if (arg == "--episode") {
            args.episode = value(arg);
        } else if (arg == "--actions-output") {
            args.actions_output = value(arg);
        } else if (arg == "--report-output") {
            args.report_output = value(arg);
        } else if (arg == "--model") {
            args.model = value(arg);
        } else if (arg == "--device") {
            args.device = value(arg);
        } else if (arg == "--runner") {
            const std::string_view name = value(arg);
            if (name == "synthetic")
                args.runner = PythonRunner::Synthetic;
            else if (name == "smolvla")
                args.runner = PythonRunner::SmolVla;
            else
                throw std::invalid_argument("--runner must be synthetic or smolvla");
        } else if (arg == "--stitching") {
            const std::string_view name = value(arg);
            if (name == "discard")
                args.stitching = Stitching::Discard;
            else if (name == "rtc")
                args.stitching = Stitching::Rtc;
            else
                throw std::invalid_argument("--stitching must be discard or rtc");
        } else if (arg == "--warmup-inferences") {
            args.warmup_inferences = std::stoi(std::string(value(arg)));
        } else if (arg == "--rtc-denoise-steps") {
            args.rtc_denoise_steps = std::stoi(std::string(value(arg)));
        } else if (arg == "--rtc-inference-delay") {
            args.rtc_inference_delay = std::stoi(std::string(value(arg)));
        } else if (arg == "--rtc-execution-horizon") {
            args.rtc_execution_horizon = std::stoi(std::string(value(arg)));
        } else if (arg == "--allow-download") {
            args.local_files_only = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (args.episode.empty()) throw std::invalid_argument("--episode is required");
    if (args.warmup_inferences < 0) {
        throw std::invalid_argument("warmup inference count must be non-negative");
    }
    if (args.rtc_denoise_steps <= 0) {
        throw std::invalid_argument("RTC denoise steps must be positive");
    }
    check_horizons(args.rtc_inference_delay, args.rtc_execution_horizon, kChunkSize);
    return args;
}

void write_report(std::ostream &out, const Arguments &args, std::size_t episode_frames,
                  int reference_action_dim,
                  RuntimeLoop &loop, const ReplayActionSink &sink,
                  const ReplayQualitySummary *quality) {
    const Summary staleness = loop.metrics().staleness.summarize();
    const Summary ready_to_emit = loop.metrics().ready_to_emit.summarize();
    const InferenceStats &inference = loop.inference_stats();
    const ConsumerStats &consumer = loop.queue().consumer_stats();
    out << std::fixed << std::setprecision(6) << "{\n"
        << "  \"format\": \"cerebellum-replay-report-v1\",\n"
        << "  \"runner\": \""
        << (args.runner == PythonRunner::Synthetic ? "synthetic" : "smolvla") << "\",\n"
        << "  \"stitching\": \""
        << (args.stitching == Stitching::Rtc ? "rtc" : "discard") << "\",\n"
        << "  \"device\": \"" << args.device << "\",\n"
        << "  \"episode_frames\": " << episode_frames << ",\n"
        << "  \"reference_action_dim\": " << reference_action_dim << ",\n"
        << "  \"control_hz\": " << kControlHz << ",\n"
        << "  \"rtc_denoise_steps\": " << args.rtc_denoise_steps << ",\n"
        << "  \"rtc_inference_delay\": " << args.rtc_inference_delay << ",\n"
        << "  \"rtc_execution_horizon\": " << args.rtc_execution_horizon << ",\n"
        << "  \"timing_ms\": {\n"
        << "    \"staleness_p50\": " << staleness.p50_ns / 1e6 << ",\n"
        << "    \"staleness_p99\": " << staleness.p99_ns / 1e6 << ",\n"
        << "    \"ready_to_emit_p50\": " << ready_to_emit.p50_ns / 1e6 << ",\n"
        << "    \"ready_to_emit_p99\": " << ready_to_emit.p99_ns / 1e6 << "\n"
        << "  },\n"
        << "  \"counts\": {\n"
        << "    \"ticks\": " << loop.metrics().ticks << ",\n"
        << "    \"recorded_actions\": " << sink.records().size() << ",\n"
        << "    \"dropped_records\": " << sink.dropped_records() << ",\n"
        << "    \"underruns\": " << loop.metrics().underruns << ",\n"
        << "    \"chunks_generated\": " << inference.generated << ",\n"
        << "    \"generation_failures\": " << inference.generation_failed << ",\n"
        << "    \"actions_discarded\": " << consumer.actions_discarded << "\n"
        << "  },\n"
        << "  \"quality\": ";
    if (!quality) {
        out << "null\n";
    } else {
        out << "{\n"
            << "    \"compared_actions\": " << quality->compared_actions << ",\n"
            << "    \"fallback_actions\": " << quality->fallback_actions << ",\n"
            << "    \"missing_reference_actions\": "
            << quality->missing_reference_actions << ",\n"
            << "    \"mean_absolute_error\": " << quality->mean_absolute_error << ",\n"
            << "    \"root_mean_square_error\": " << quality->root_mean_square_error << ",\n"
            << "    \"max_linf_error\": " << quality->max_linf_error << "\n"
            << "  }\n";
    }
    out << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        std::ifstream input(args.episode, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open replay episode: " + args.episode);
        ReplayEpisode episode = read_replay_episode(input);

        RuntimeConfig config;
        if (episode.action_dim > 0) config.action_dim = episode.action_dim;
        config.stitching = args.stitching;
        config.refresh_policy = args.stitching == Stitching::Rtc ? RefreshPolicy::Horizon
                                                                 : RefreshPolicy::Continuous;
        config.rtc.denoise_steps = args.rtc_denoise_steps;
        config.rtc.inference_delay = args.rtc_inference_delay;
        config.rtc.execution_horizon = args.rtc_execution_horizon;
        config.validate();

        const std::size_t ticks = episode.frames.size();
        ReplayObservationSource source(std::move(episode.frames));
        PythonChunkGeneratorOptions generator_options;
        generator_options.python_executable = CEREBELLUM_REPLAY_PYTHON;
        generator_options.python_package_path = CEREBELLUM_REPLAY_PYTHONPATH;
        generator_options.runner = args.runner;
        generator_options.model = args.model;
        generator_options.device = args.device;
        generator_options.local_files_only = args.local_files_only;
        generator_options.startup_timeout = std::chrono::minutes(15);
        generator_options.inference_timeout = std::chrono::minutes(2);
        PythonChunkGenerator generator(config, source, generator_options);

        source.start();
        for (int i = 0; i < args.warmup_inferences; ++i) {
            Chunk warmup;
            if (!generator.generate(
                    InferenceRequest{0, -1, config.chunk_size, args.stitching}, warmup)) {
                throw std::runtime_error("replay warmup failed: " + generator.last_error());
            }
        }

        source.start();  // warmup time must not consume the recorded episode
        ReplayActionSink sink(ticks);
        RuntimeLoop loop(config, generator, sink, ticks);
        loop.run_for(ticks);

        if (!args.actions_output.empty()) {
            std::ofstream actions(args.actions_output);
            if (!actions) {
                throw std::runtime_error("cannot open action output: " + args.actions_output);
            }
            write_replay_actions_csv(actions, sink.records(), config.action_dim);
        }

        ReplayQualitySummary quality;
        const ReplayQualitySummary *quality_ptr = nullptr;
        if (episode.action_dim > 0) {
            quality = evaluate_replay_actions(sink.records(), episode.reference_actions,
                                              episode.action_dim);
            quality_ptr = &quality;
        }
        if (args.report_output.empty()) {
            write_report(std::cout, args, ticks, episode.action_dim, loop, sink, quality_ptr);
        } else {
            std::ofstream report(args.report_output);
            if (!report) {
                throw std::runtime_error("cannot open replay report: " + args.report_output);
            }
            write_report(report, args, ticks, episode.action_dim, loop, sink, quality_ptr);
        }
        return sink.dropped_records() == 0 ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << "replay_benchmark: " << error.what() << '\n';
        return 1;
    }
}
