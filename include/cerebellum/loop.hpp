// Two loops, two rates — fixed-cadence control plus an asynchronous inference
// worker (spec.md §4.1, §4.4, §7).

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include "cerebellum/action_safety.hpp"
#include "cerebellum/chunk_queue.hpp"
#include "cerebellum/config.hpp"
#include "cerebellum/timing.hpp"
#include "cerebellum/validate.hpp"

namespace cerebellum {

struct RtcConditioning {
    std::uint64_t source_chunk_id = 0;
    std::int64_t prefix_first_step = 0;
    int prefix_count = 0;
    int inference_delay = 0;
    int execution_horizon = 0;
    std::array<ModelAction, kChunkSize> prefix{};

    bool present() const noexcept { return prefix_count > 0; }
};

// Everything the inference side needs to decide which absolute steps its chunk
// describes. RTC carries normalized padded actions, never executable robot
// commands: guidance happens in the model's own action space.
struct InferenceRequest {
    std::int64_t first_step = 0;
    std::int64_t last_emitted_step = -1;
    int action_count = 0;
    Stitching stitching = Stitching::Discard;
    RtcConditioning rtc{};
};

// Blocking is allowed here: this method runs only on the inference worker. A
// PythonChunkGenerator implements the same boundary around a persistent worker.
// On success it fills executable actions, padded model actions, count, and
// observation provenance. RuntimeLoop owns first_step,
// chunk_id, and inference start/end stamps so every backend is measured alike.
class ChunkGenerator {
   public:
    virtual ~ChunkGenerator() = default;
    virtual bool generate(const InferenceRequest &request, Chunk &out) noexcept = 0;
};

struct ActionEmission {
    std::int64_t step = 0;
    TimePoint deadline{};
    Action action{};
    bool fallback = false;
    Nanos observation_age{};
    std::uint32_t safety_flags = 0;
    bool safety_rejected = false;
};

// Called on the control thread. Implementations must not block, allocate, or
// wait on I/O; a later transport may enqueue to a preallocated Axon topic.
class ActionSink {
   public:
    virtual ~ActionSink() = default;
    virtual void emit(const ActionEmission &emission) noexcept = 0;
};

// Written only by the inference worker and read after run_for() joins it.
struct InferenceStats {
    std::uint64_t requests = 0;
    std::uint64_t generated = 0;
    std::uint64_t generation_failed = 0;
    std::uint64_t invalid_chunks = 0;
    std::uint64_t publish_retries = 0;
    std::uint64_t generation_wall_ns = 0;
    std::uint64_t rtc_inference_overruns = 0;
};

// RTC uses a conservative recent latency rather than trusting the most recent
// (possibly lucky) request. Fixed storage keeps this estimator allocation-free.
class RtcDelayEstimator {
   public:
    explicit RtcDelayEstimator(int configured_floor) : floor_(std::max(1, configured_floor)) {}

    void observe(int steps) noexcept {
        samples_[next_] = std::max(1, steps);
        next_ = (next_ + 1) % samples_.size();
        count_ = std::min(count_ + 1, samples_.size());
    }

    int conservative_steps() const noexcept {
        int result = floor_;
        for (std::size_t i = 0; i < count_; ++i) result = std::max(result, samples_[i]);
        return result;
    }

   private:
    const int floor_;
    std::array<int, kRtcDelayWindow> samples_{};
    std::size_t next_ = 0;
    std::size_t count_ = 0;
};

class RuntimeLoop {
   public:
    RuntimeLoop(const RuntimeConfig &cfg, ChunkGenerator &generator, ActionSink &sink,
                std::size_t metrics_capacity = 4096,
                Nanos worker_poll_period = std::chrono::microseconds(100),
                ActionSafetyFilter *safety_filter = nullptr)
        : cfg_(cfg),
          generator_(generator),
          sink_(sink),
          queue_(cfg),
          metrics_(metrics_capacity),
          worker_poll_period_(worker_poll_period),
          safety_filter_(safety_filter) {
        cfg_.validate();
        if (worker_poll_period_ <= Nanos::zero()) {
            throw std::invalid_argument("worker_poll_period must be positive");
        }
        if (safety_filter_ && safety_filter_->config().action_dim != cfg_.action_dim) {
            throw std::invalid_argument("safety and runtime action dimensions differ");
        }
    }

    RuntimeLoop(const RuntimeLoop &) = delete;
    RuntimeLoop &operator=(const RuntimeLoop &) = delete;

    ~RuntimeLoop() {
        request_stop();
        join_worker();
    }

    // Runs control on the calling thread and inference on one worker thread.
    // The first deadline is one period in the future, giving the worker one
    // period to begin the initial inference without changing the fixed grid.
    void run_for(std::size_t max_ticks, Nanos control_period = kControlPeriod) {
        if (control_period <= Nanos::zero()) {
            throw std::invalid_argument("control_period must be positive");
        }
        if (started_.exchange(true)) {
            throw std::logic_error("RuntimeLoop is single-use");
        }
        active_control_period_ = control_period;

        worker_ = std::thread([this] { inference_loop(); });
        // Publish running only after the worker exists. request_stop() from a
        // thread that observed running()==true can no longer be overwritten by
        // startup initialization.
        running_.store(true, std::memory_order_release);

        const TimePoint origin = now() + control_period;
        const ScheduleClock schedule(origin, control_period);
        std::int64_t step = 0;

        for (std::size_t tick = 0; tick < max_ticks && !stop_.load(std::memory_order_acquire);
             ++tick) {
            const TimePoint deadline = schedule.deadline(step);
            sleep_until(deadline);
            control_tick(step, deadline);

            // Stay on the original grid. If control work crossed one or more
            // deadlines, jump to the grid point containing `now` instead of
            // executing obsolete actions late or slipping the schedule.
            const std::int64_t current_grid = schedule.index_at(now());
            step = std::max(step + 1, current_grid);
        }

        request_stop();
        join_worker();
        running_.store(false, std::memory_order_release);
    }

    void request_stop() noexcept { stop_.store(true, std::memory_order_release); }
    bool running() const noexcept { return running_.load(std::memory_order_acquire); }

    ActionChunkQueue &queue() noexcept { return queue_; }
    const ActionChunkQueue &queue() const noexcept { return queue_; }
    ControlMetrics &metrics() noexcept { return metrics_; }
    const ControlMetrics &metrics() const noexcept { return metrics_; }
    const InferenceStats &inference_stats() const noexcept { return inference_stats_; }

   private:
    void control_tick(std::int64_t step, TimePoint deadline) noexcept {
        Action action{};
        ActionRecord record{};
        const bool available = queue_.pop(step, deadline, action, record);

        Nanos observation_age{};
        if (available) {
            observation_age = now() - record.chunk.t_obs_capture;
            metrics_.on_emit(record);
        } else {
            action = fallback_action();
            metrics_.on_underrun();
        }

        std::uint32_t safety_flags = 0;
        bool safety_rejected = false;
        if (safety_filter_) {
            const SafetyResult safe =
                safety_filter_->apply(action, active_control_period_, available, observation_age);
            action = safe.action;
            safety_flags = safe.flags;
            safety_rejected = safe.rejected;
        }

        // Fallback history contains only the action that actually passed the
        // safety boundary, never the raw model prediction.
        remember(action);
        sink_.emit(ActionEmission{step, deadline, action, !available, observation_age,
                                  safety_flags, safety_rejected});
    }

    Action fallback_action() const noexcept {
        switch (cfg_.underrun) {
            case UnderrunPolicy::HoldLast:
                return have_last_ ? last_output_ : Action{};
            case UnderrunPolicy::Zeros:
                return Action{};
            case UnderrunPolicy::Extrapolate: {
                if (!have_last_) return Action{};
                if (!have_previous_) return last_output_;
                Action out = last_output_;
                for (int d = 0; d < cfg_.action_dim; ++d) {
                    const std::size_t i = static_cast<std::size_t>(d);
                    out[i] = last_output_[i] + (last_output_[i] - previous_output_[i]);
                }
                return out;
            }
        }
        return Action{};
    }

    void remember(const Action &action) noexcept {
        if (have_last_) {
            previous_output_ = last_output_;
            have_previous_ = true;
        }
        last_output_ = action;
        have_last_ = true;
    }

    void inference_loop() noexcept {
        Chunk *slot = nullptr;
        bool ready_to_publish = false;
        std::uint64_t next_chunk_id = 1;
        Chunk retained{};  // worker-private copy; never touches consumer-owned slots
        bool have_retained = false;
        RtcDelayEstimator delay_estimator(cfg_.rtc.inference_delay);

        while (!stop_.load(std::memory_order_acquire)) {
            if (ready_to_publish) {
                if (queue_.publish(slot)) {
                    slot = nullptr;
                    ready_to_publish = false;
                } else {
                    ++inference_stats_.publish_retries;
                    std::this_thread::sleep_for(worker_poll_period_);
                }
                continue;
            }

            if (!queue_.should_refresh()) {
                std::this_thread::sleep_for(worker_poll_period_);
                continue;
            }

            if (!slot) {
                slot = queue_.acquire();
                if (!slot) {
                    std::this_thread::sleep_for(worker_poll_period_);
                    continue;
                }
            }

            const std::int64_t last_emitted = queue_.last_emitted_step();
            InferenceRequest request{last_emitted + 1, last_emitted, cfg_.chunk_size,
                                     cfg_.stitching};
            request.rtc.inference_delay = queue_.inference_delay();
            request.rtc.execution_horizon = queue_.execution_horizon();
            if (cfg_.stitching == Stitching::Rtc && have_retained &&
                retained.covers(request.first_step)) {
                request.rtc.source_chunk_id = retained.stamps.chunk_id;
                request.rtc.prefix_first_step = request.first_step;
                const int offset = static_cast<int>(request.first_step - retained.first_step);
                request.rtc.prefix_count = std::min(retained.count - offset, cfg_.chunk_size);
                for (int i = 0; i < request.rtc.prefix_count; ++i) {
                    request.rtc.prefix[static_cast<std::size_t>(i)] =
                        retained.model_actions[static_cast<std::size_t>(offset + i)];
                }
            }

            slot->first_step = request.first_step;
            slot->count = 0;
            slot->stamps = ChunkStamps{};
            slot->stamps.t_infer_start = now();
            ++inference_stats_.requests;

            if (!generator_.generate(request, *slot)) {
                const TimePoint infer_end = now();
                inference_stats_.generation_wall_ns +=
                    static_cast<std::uint64_t>((infer_end - slot->stamps.t_infer_start).count());
                ++inference_stats_.generation_failed;
                std::this_thread::sleep_for(worker_poll_period_);
                continue;  // retain this producer-owned slot and retry it
            }

            slot->stamps.t_infer_end = now();
            inference_stats_.generation_wall_ns += static_cast<std::uint64_t>(
                (slot->stamps.t_infer_end - slot->stamps.t_infer_start).count());
            slot->first_step = request.first_step;  // backend cannot retarget it
            if (slot->count <= 0 || slot->count > cfg_.chunk_size) {
                ++inference_stats_.invalid_chunks;
                std::this_thread::sleep_for(worker_poll_period_);
                continue;
            }

            slot->stamps.chunk_id = next_chunk_id++;
            const auto generation_ns = std::chrono::duration_cast<Nanos>(slot->stamps.t_infer_end -
                                                                         slot->stamps.t_infer_start)
                                           .count();
            const auto period_ns = kControlPeriod.count();
            const int measured_delay =
                static_cast<int>((generation_ns + period_ns - 1) / period_ns);
            if (cfg_.stitching == Stitching::Rtc && measured_delay >= cfg_.chunk_size) {
                ++inference_stats_.rtc_inference_overruns;
            }
            delay_estimator.observe(measured_delay);
            queue_.update_rtc_timing(delay_estimator.conservative_steps());
            retained = *slot;
            have_retained = true;
            ++inference_stats_.generated;
            ready_to_publish = true;
        }
        // If stop arrives while this thread owns a slot, the RuntimeLoop is
        // being destroyed after join; no other thread can observe or reuse it.
    }

    void join_worker() noexcept {
        if (worker_.joinable()) worker_.join();
    }

    const RuntimeConfig cfg_;
    ChunkGenerator &generator_;
    ActionSink &sink_;
    ActionChunkQueue queue_;
    ControlMetrics metrics_;
    const Nanos worker_poll_period_;
    ActionSafetyFilter *const safety_filter_;
    Nanos active_control_period_ = kControlPeriod;

    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> started_{false};
    std::thread worker_;

    // Control-thread private fallback history.
    Action previous_output_{};
    Action last_output_{};
    bool have_previous_ = false;
    bool have_last_ = false;

    // Inference-worker private; read only after join.
    InferenceStats inference_stats_{};
};

}  // namespace cerebellum
