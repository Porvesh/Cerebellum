#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "cerebellum/action_safety.hpp"
#include "cerebellum/python_chunk_generator.hpp"
#include "cerebellum/python_simulator.hpp"

using namespace cerebellum;

namespace {

constexpr int kLiberoActionDim = 7;

struct Arguments {
    std::string model = "HuggingFaceVLA/smolvla_libero";
    std::string device = "cuda";
    std::string suite = "libero_spatial";
    std::string osmesa_library_path;
    std::string output;
    int task_id = 0;
    int init_state = 0;
    int control_hz = 10;
    int ticks = 300;
    int warmup_inferences = 1;
    int refresh_trigger = 0;
    int rtc_denoise_steps = 5;
    int rtc_inference_delay = 0;
    int rtc_execution_horizon = 0;
    double inference_budget_ms = 300.0;
    double max_staleness_ms = 575.0;
    double max_observation_age_ms = 1000.0;
    Stitching stitching = Stitching::Discard;
    RefreshPolicy refresh_policy = RefreshPolicy::Continuous;
    ControlPacing pacing = ControlPacing::RealTime;
    bool local_files_only = true;
};

// The model process is loaded and warmed before LIBERO captures its initial
// observation. During warm-up it sees a shape-correct dummy; before either
// runtime thread starts, bind() permanently redirects reads to the simulator.
class SwitchableObservationSource final : public ObservationSource {
   public:
    std::shared_ptr<const ObservationSnapshot> latest() noexcept override {
        return delegate_ ? delegate_->latest() : fallback_;
    }

    void set_fallback(std::shared_ptr<const ObservationSnapshot> value) {
        fallback_ = std::move(value);
    }
    void bind(ObservationSource &delegate) noexcept { delegate_ = &delegate; }

   private:
    std::shared_ptr<const ObservationSnapshot> fallback_;
    ObservationSource *delegate_ = nullptr;
};

std::shared_ptr<const ObservationSnapshot> make_warmup_observation(
    const WorkerObservationSchema &schema) {
    if (!schema.constrained()) {
        throw std::runtime_error("SmolVLA did not provide an observation schema");
    }
    auto observation = std::make_shared<ObservationSnapshot>();
    observation->sequence = 0;
    observation->capture_time = now();
    observation->task = "LIBERO model warm-up";
    observation->state.assign(schema.state_dim, 0.0F);
    for (const WorkerImageSpec &expected : schema.images) {
        CameraImage image;
        image.feature_name = expected.feature_name;
        image.channels = expected.channels;
        image.height = expected.height;
        image.width = expected.width;
        image.pixels.resize(static_cast<std::size_t>(image.channels) * image.height * image.width);
        observation->images.push_back(std::move(image));
    }
    observation->validate();
    return observation;
}

Arguments parse_arguments(int argc, char **argv) {
    Arguments args;
    if (const char *path = std::getenv("CEREBELLUM_OSMESA_LIBRARY_PATH")) {
        args.osmesa_library_path = path;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&](std::string_view option) -> std::string_view {
            if (++i >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
            return argv[i];
        };
        if (arg == "--model") {
            args.model = value(arg);
        } else if (arg == "--device") {
            args.device = value(arg);
        } else if (arg == "--suite") {
            args.suite = value(arg);
        } else if (arg == "--task-id") {
            args.task_id = std::stoi(std::string(value(arg)));
        } else if (arg == "--init-state") {
            args.init_state = std::stoi(std::string(value(arg)));
        } else if (arg == "--ticks") {
            args.ticks = std::stoi(std::string(value(arg)));
        } else if (arg == "--control-hz") {
            args.control_hz = std::stoi(std::string(value(arg)));
        } else if (arg == "--warmup-inferences") {
            args.warmup_inferences = std::stoi(std::string(value(arg)));
        } else if (arg == "--refresh-trigger") {
            args.refresh_trigger = std::stoi(std::string(value(arg)));
        } else if (arg == "--rtc-denoise-steps") {
            args.rtc_denoise_steps = std::stoi(std::string(value(arg)));
        } else if (arg == "--rtc-inference-delay") {
            args.rtc_inference_delay = std::stoi(std::string(value(arg)));
        } else if (arg == "--rtc-execution-horizon") {
            args.rtc_execution_horizon = std::stoi(std::string(value(arg)));
        } else if (arg == "--inference-budget-ms") {
            args.inference_budget_ms = std::stod(std::string(value(arg)));
        } else if (arg == "--max-staleness-ms") {
            args.max_staleness_ms = std::stod(std::string(value(arg)));
        } else if (arg == "--max-observation-age-ms") {
            args.max_observation_age_ms = std::stod(std::string(value(arg)));
        } else if (arg == "--stitching") {
            const std::string_view name = value(arg);
            if (name == "discard")
                args.stitching = Stitching::Discard;
            else if (name == "rtc")
                args.stitching = Stitching::Rtc;
            else
                throw std::invalid_argument("--stitching must be discard or rtc");
        } else if (arg == "--refresh-policy") {
            const std::string_view name = value(arg);
            if (name == "tail")
                args.refresh_policy = RefreshPolicy::Tail;
            else if (name == "continuous")
                args.refresh_policy = RefreshPolicy::Continuous;
            else if (name == "horizon")
                args.refresh_policy = RefreshPolicy::Horizon;
            else
                throw std::invalid_argument(
                    "--refresh-policy must be tail, continuous, or horizon");
        } else if (arg == "--mode") {
            const std::string_view name = value(arg);
            if (name == "realtime")
                args.pacing = ControlPacing::RealTime;
            else if (name == "synchronous")
                args.pacing = ControlPacing::SynchronousEvaluation;
            else
                throw std::invalid_argument("--mode must be realtime or synchronous");
        } else if (arg == "--osmesa-library-path") {
            args.osmesa_library_path = value(arg);
        } else if (arg == "--output") {
            args.output = value(arg);
        } else if (arg == "--allow-download") {
            args.local_files_only = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (args.task_id < 0 || args.init_state < 0 || args.control_hz <= 0 || args.ticks <= 0 ||
        args.warmup_inferences < 0 || args.refresh_trigger < 0 ||
        args.refresh_trigger >= kChunkSize || !(args.inference_budget_ms > 0.0) ||
        !(args.max_staleness_ms > 0.0) ||
        !(args.max_observation_age_ms > 0.0)) {
        throw std::invalid_argument("invalid LIBERO rollout argument");
    }
    check_horizons(args.rtc_inference_delay, args.rtc_execution_horizon, kChunkSize);
    return args;
}

void validate_schema(const PythonSimulatorAdapter &simulator,
                     const WorkerObservationSchema &model) {
    if (simulator.state_dim() != model.state_dim) {
        throw std::runtime_error("simulator and model state dimensions differ");
    }
    if (simulator.image_schema().size() != model.images.size()) {
        throw std::runtime_error("simulator and model camera counts differ");
    }
    for (const WorkerImageSpec &expected : model.images) {
        bool found = false;
        for (const SimulatorImageSpec &actual : simulator.image_schema()) {
            if (expected.feature_name == actual.feature_name) {
                found = expected.channels == actual.channels && expected.height == actual.height &&
                        expected.width == actual.width;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("simulator does not provide model camera: " +
                                     expected.feature_name);
        }
    }
}

class MeasuredSimulatorSink final : public ActionSink {
   public:
    explicit MeasuredSimulatorSink(PythonSimulatorAdapter &simulator) : simulator_(simulator) {}

    void emit(const ActionEmission &emission) noexcept override {
        ++emissions;
        if (emission.fallback) ++fallbacks;
        if (emission.safety_rejected) ++safety_rejections;
        if (emission.safety_flags != 0) ++safety_modified;
        simulator_.emit(emission);
    }

    std::uint64_t emissions = 0;
    std::uint64_t fallbacks = 0;
    std::uint64_t safety_modified = 0;
    std::uint64_t safety_rejections = 0;

   private:
    PythonSimulatorAdapter &simulator_;
};

void write_clamp_diagnostics(std::ostream &out, const SafetyStats &stats, int action_dim) {
    constexpr std::array<std::string_view, kLiberoActionDim> names{
        "delta_x", "delta_y", "delta_z", "axis_angle_x",
        "axis_angle_y", "axis_angle_z", "gripper",
    };
    out << "    \"clamp_by_dimension\": [\n";
    for (int d = 0; d < action_dim; ++d) {
        const std::size_t i = static_cast<std::size_t>(d);
        out << "      {\"dimension\": " << d << ", \"name\": \"" << names[i]
            << "\", \"below_min\": " << stats.requested_below_min[i]
            << ", \"above_max\": " << stats.requested_above_max[i]
            << ", \"max_excess\": " << stats.max_bound_excess[i] << "}";
        if (d + 1 < action_dim) out << ',';
        out << '\n';
    }
    out << "    ]\n";
}

void report(std::ostream &out, const Arguments &args, RuntimeLoop &loop,
            const RuntimeConfig &config,
            const PythonSimulatorAdapter &simulator, const MeasuredSimulatorSink &sink,
            const ActionSafetyFilter &safety, Nanos wall_time) {
    const SimulatorStats sim = simulator.stats();
    const InferenceStats &inference = loop.inference_stats();
    const ConsumerStats &queue = loop.queue().consumer_stats();
    const SafetyStats &safety_stats = safety.stats();
    const Summary staleness = loop.metrics().staleness.summarize();
    PercentileRecorder eligible_staleness(loop.metrics().staleness.size());
    std::size_t cold_start_staleness = 0;
    for (const std::int64_t sample : loop.metrics().staleness.samples()) {
        if (to_ms(sample) <= args.max_observation_age_ms)
            eligible_staleness.record(sample);
        else
            ++cold_start_staleness;
    }
    const Summary eligible = eligible_staleness.summarize();
    const double wall_seconds = std::chrono::duration<double>(wall_time).count();
    const double control_rate_hz = wall_seconds > 0.0 ? sink.emissions / wall_seconds : 0.0;
    const double simulator_rate_hz =
        wall_seconds > 0.0 ? sim.commands_applied / wall_seconds : 0.0;
    const std::uint64_t resolved_commands = sim.commands_applied + sim.commands_superseded;
    const double command_delivery = resolved_commands > 0
                                        ? static_cast<double>(sim.commands_applied) /
                                              static_cast<double>(resolved_commands)
                                        : 0.0;
    const double minimum_simulator_rate_hz = 0.95 * args.control_hz;
    const bool control_cadence_live = control_rate_hz >= minimum_simulator_rate_hz;
    const bool simulator_throughput_live = simulator_rate_hz >= minimum_simulator_rate_hz;
    const bool command_delivery_live = command_delivery >= 0.99;
    const bool inference_live = inference.generation_failed == 0;
    const bool freshness_live = loop.metrics().staleness_violations == 0;
    const bool runtime_live = control_cadence_live && simulator_throughput_live &&
                              command_delivery_live && inference_live && freshness_live;
    const bool synchronous = args.pacing == ControlPacing::SynchronousEvaluation;
    const bool evaluation_pipeline_pass =
        sim.commands_applied == sink.emissions && sim.commands_superseded == 0 && inference_live;
    const double applied = static_cast<double>(sim.commands_applied);
    const double environment_step_ms = applied > 0.0 ? sim.environment_step_ns / applied / 1e6 : 0.0;
    const double observation_build_ms =
        applied > 0.0 ? sim.observation_build_ns / applied / 1e6 : 0.0;
    const double cpp_round_trip_ms = applied > 0.0 ? sim.cpp_round_trip_ns / applied / 1e6 : 0.0;
    out << std::boolalpha << std::fixed << std::setprecision(3) << "{\n"
        << "  \"model\": \"" << args.model << "\",\n"
        << "  \"suite\": \"" << args.suite << "\",\n"
        << "  \"task_id\": " << args.task_id << ",\n"
        << "  \"init_state\": " << args.init_state << ",\n"
        << "  \"task\": \"" << simulator.task() << "\",\n"
        << "  \"device\": \"" << args.device << "\",\n"
        << "  \"mode\": \"" << (synchronous ? "synchronous" : "realtime") << "\",\n"
      << "  \"control_hz\": " << args.control_hz << ",\n"
      << "  \"control_period_ms\": " << config.control_period_ms() << ",\n"
      << "  \"inference_budget_ms\": " << config.inference_budget_ms << ",\n"
      << "  \"max_staleness_ms\": " << config.max_staleness_ms << ",\n"
      << "  \"refresh_trigger\": " << config.effective_refresh_trigger() << ",\n"
      << "  \"rtc_inference_delay\": " << loop.queue().inference_delay() << ",\n"
      << "  \"rtc_execution_horizon\": " << loop.queue().execution_horizon() << ",\n"
        << "  \"stitching\": \"" << (args.stitching == Stitching::Rtc ? "rtc" : "discard")
        << "\",\n"
        << "  \"ticks_requested\": " << args.ticks << ",\n"
        << "  \"wall_time_ms\": " << to_ms(wall_time) << ",\n"
        << "  \"success\": " << sim.success << ",\n"
        << "  \"terminated\": " << sim.terminated << ",\n"
        << "  \"counts\": {\n"
        << "    \"control_emissions\": " << sink.emissions << ",\n"
        << "    \"fallback_actions\": " << sink.fallbacks << ",\n"
        << "    \"safety_modified\": " << sink.safety_modified << ",\n"
        << "    \"safety_rejections\": " << sink.safety_rejections << ",\n"
        << "    \"simulator_commands_applied\": " << sim.commands_applied << ",\n"
        << "    \"simulator_commands_superseded\": " << sim.commands_superseded << ",\n"
        << "    \"simulator_observations\": " << sim.observations << ",\n"
        << "    \"inference_requests\": " << inference.requests << ",\n"
        << "    \"chunks_generated\": " << inference.generated << ",\n"
        << "    \"generation_failures\": " << inference.generation_failed << ",\n"
        << "    \"chunks_accepted\": " << queue.chunks_accepted << ",\n"
        << "    \"real_actions\": " << sink.emissions - sink.fallbacks << ",\n"
        << "    \"underruns\": " << loop.metrics().underruns << ",\n"
        << "    \"staleness_violations\": " << loop.metrics().staleness_violations << "\n"
        << "  },\n"
        << "  \"safety\": {\n"
        << "    \"wall_clock_observation_age_applicable\": " << !synchronous << ",\n"
        << "    \"clamped\": " << safety_stats.action_clamped << ",\n"
        << "    \"stale_rejected\": " << safety_stats.stale_rejected << ",\n";
    write_clamp_diagnostics(out, safety_stats, config.action_dim);
    out << "  },\n"
        << "  \"simulator_timing_mean_ms\": {\n"
        << "    \"environment_step_including_render\": " << environment_step_ms << ",\n"
        << "    \"observation_build\": " << observation_build_ms << ",\n"
        << "    \"cpp_command_to_observation\": " << cpp_round_trip_ms << "\n"
        << "  },\n"
        << "  \"action_staleness_ms\": {\n"
        << "    \"samples\": " << staleness.count << ",\n"
        << "    \"p50\": " << to_ms(staleness.p50_ns) << ",\n"
        << "    \"p90\": " << to_ms(staleness.p90_ns) << ",\n"
        << "    \"p99\": " << to_ms(staleness.p99_ns) << ",\n"
        << "    \"max\": " << to_ms(staleness.max_ns) << "\n"
        << "  },\n"
        << "  \"safety_eligible_staleness_ms\": {\n"
        << "    \"definition\": \"observation age within safety limit\",\n"
        << "    \"samples\": " << eligible.count << ",\n"
        << "    \"cold_start_or_rejected_samples\": " << cold_start_staleness << ",\n"
        << "    \"p50\": " << to_ms(eligible.p50_ns) << ",\n"
        << "    \"p90\": " << to_ms(eligible.p90_ns) << ",\n"
        << "    \"p99\": " << to_ms(eligible.p99_ns) << ",\n"
        << "    \"max\": " << to_ms(eligible.max_ns) << "\n"
        << "  },\n"
        << "  \"liveness\": {\n"
        << "    \"real_time_applicable\": " << !synchronous << ",\n"
        << "    \"control_rate_hz\": " << control_rate_hz << ",\n"
        << "    \"simulator_step_rate_hz\": " << simulator_rate_hz << ",\n"
        << "    \"command_delivery_fraction\": " << command_delivery << ",\n"
        << "    \"minimum_rate_hz\": " << minimum_simulator_rate_hz << ",\n"
        << "    \"minimum_delivery_fraction\": 0.990,\n"
        << "    \"control_cadence_pass\": " << control_cadence_live << ",\n"
        << "    \"simulator_throughput_pass\": " << simulator_throughput_live << ",\n"
        << "    \"command_delivery_pass\": " << command_delivery_live << ",\n"
        << "    \"inference_pass\": " << inference_live << ",\n"
        << "    \"freshness_pass\": " << freshness_live << ",\n"
        << "    \"runtime_pass\": " << (!synchronous && runtime_live) << ",\n"
        << "    \"evaluation_pipeline_pass\": " << evaluation_pipeline_pass << "\n"
        << "  },\n"
        << "  \"simulator_healthy\": " << simulator.healthy() << "\n"
        << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);

        RuntimeConfig config;
        config.control_period = Nanos{1'000'000'000 / args.control_hz};
        config.inference_budget_ms = args.inference_budget_ms;
        config.max_staleness_ms = args.max_staleness_ms;
        config.action_dim = kLiberoActionDim;
        config.action_space = ActionSpace::Delta;
        config.stitching = args.stitching;
        config.refresh_policy = args.refresh_policy;
        config.refresh_trigger = args.refresh_trigger;
        config.queue_capacity = config.chunk_size + config.effective_refresh_trigger();
        config.rtc.denoise_steps = args.rtc_denoise_steps;
        config.rtc.inference_delay = args.rtc_inference_delay;
        config.rtc.execution_horizon = args.rtc_execution_horizon;
        config.validate();

        SwitchableObservationSource observations;
        PythonChunkGeneratorOptions generator_options;
        generator_options.python_executable = CEREBELLUM_LIBERO_PYTHON;
        generator_options.python_package_path = CEREBELLUM_LIBERO_PYTHONPATH;
        generator_options.runner = PythonRunner::SmolVla;
        generator_options.model = args.model;
        generator_options.device = args.device;
        generator_options.local_files_only = args.local_files_only;
        generator_options.startup_timeout = std::chrono::minutes(15);
        generator_options.inference_timeout = std::chrono::minutes(2);
        PythonChunkGenerator generator(config, observations, generator_options);
        observations.set_fallback(make_warmup_observation(generator.observation_schema()));

        for (int i = 0; i < args.warmup_inferences; ++i) {
            Chunk warmup;
            if (!generator.generate(InferenceRequest{0, -1, config.chunk_size, args.stitching},
                                    warmup)) {
                throw std::runtime_error("model warmup failed: " + generator.last_error());
            }
        }

        PythonSimulatorOptions simulator_options;
        simulator_options.python_executable = CEREBELLUM_LIBERO_PYTHON;
        simulator_options.python_package_path = CEREBELLUM_LIBERO_PYTHONPATH;
        simulator_options.backend = PythonSimulatorBackend::Libero;
        simulator_options.delivery = args.pacing == ControlPacing::SynchronousEvaluation
                                         ? SimulatorDelivery::Acknowledged
                                         : SimulatorDelivery::LatestOnly;
        simulator_options.suite = args.suite;
        simulator_options.task_id = args.task_id;
        simulator_options.init_state = args.init_state;
        simulator_options.control_hz = args.control_hz;
        simulator_options.osmesa_library_path = args.osmesa_library_path;
        simulator_options.startup_timeout = std::chrono::minutes(2);
        PythonSimulatorAdapter simulator(simulator_options);
        validate_schema(simulator, generator.observation_schema());
        observations.bind(simulator);

        SafetyConfig safety_config(config.action_dim);
        for (int d = 0; d < config.action_dim; ++d) {
            safety_config.min_action[static_cast<std::size_t>(d)] = -1.0F;
            safety_config.max_action[static_cast<std::size_t>(d)] = 1.0F;
        }
        safety_config.replacement_action[6] = -1.0F;
        // In synchronous evaluation, wall time intentionally runs slower than
        // simulated policy time. The observation is still the latest simulator
        // state, so deployment wall-clock age must be measured but not rejected.
        safety_config.max_observation_age =
            args.pacing == ControlPacing::SynchronousEvaluation
                ? Nanos::max()
                : std::chrono::duration_cast<Nanos>(
                      std::chrono::duration<double, std::milli>(args.max_observation_age_ms));
        ActionSafetyFilter safety(safety_config);
        MeasuredSimulatorSink sink(simulator);
        RuntimeLoop loop(config, generator, sink, static_cast<std::size_t>(args.ticks),
                         std::chrono::microseconds(100), &safety);
        const TimePoint started = now();
        loop.run_for(static_cast<std::size_t>(args.ticks), args.pacing);
        const Nanos wall_time = now() - started;

        if (!simulator.healthy() && !simulator.last_error().empty()) {
            throw std::runtime_error("simulator failed: " + simulator.last_error());
        }
        if (args.output.empty()) {
            report(std::cout, args, loop, config, simulator, sink, safety, wall_time);
        } else {
            std::ofstream output(args.output);
            if (!output) throw std::runtime_error("cannot open output file: " + args.output);
            report(output, args, loop, config, simulator, sink, safety, wall_time);
        }
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "libero_rollout: " << exc.what() << '\n';
        return 1;
    }
}
