// The action chunk queue — the only thing between the two threads (spec.md
// §4.1, §4.2, §4.3).
//
// Deliberately NOT an SpscQueue<Action>. Three requirements make a per-action
// ring the wrong shape:
//
//   - push is a chunk of H actions, pop is one action. A per-item ring
//   publishes
//     an incomplete chunk to the consumer, and can't express "replace the
//     queued tail", which is what discard stitching does.
//   - discard and ensemble MUTATE data that is already queued. In a plain SPSC
//     ring that region belongs to the consumer, so writing it from the producer
//     is a race, not a policy choice.
//   - RTC needs the committed prefix read from the PRODUCER side (§4.5). A
//     pop-only interface cannot express it.
//
// So ownership of whole chunks moves through two SPSC rings — ready (producer
// to consumer) and free (consumer to producer) — over a fixed pool, and the
// action cursor lives entirely on the consumer. Emitting an action is a local
// index computation, not a concurrent operation, and stitching happens when the
// consumer accepts a chunk, on data no other thread is touching.
//
// Indexed by ABSOLUTE control step throughout, per §4.5. That is what makes
// "what did I promise for steps s..s+n" a slice, and it makes discarding a
// stale prefix fall out of the indexing rather than needing bookkeeping.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cerebellum/config.hpp"
#include "cerebellum/detail/spsc_queue.hpp"
#include "cerebellum/timing.hpp"

namespace cerebellum {

// Robot commands and RTC conditioning are different representations. Action is
// the postprocessed command consumed by the control loop (only action_dim
// values are meaningful); ModelAction is the normalized padded trajectory RTC
// needs. Collapsing them would either command normalized values or feed
// denormalized values back into diffusion.
using Action = std::array<float, kPaddedActionDim>;
using ModelAction = std::array<float, kPaddedActionDim>;

// Consumer holds two (current + previous, for the ensemble overlap), producer
// holds one under construction, one may be in flight in the ready ring.
inline constexpr std::size_t kChunkSlots = 4;

// A pool slot. Fixed size — kChunkSize is compiled in, so a config asking for a
// longer chunk is rejected in the constructor rather than allocating (#3).
struct Chunk {
    std::array<Action, kChunkSize> actions{};             // executable robot space
    std::array<ModelAction, kChunkSize> model_actions{};  // normalized RTC space
    int count = 0;                                        // actions actually written
    std::int64_t first_step = 0;                          // absolute control step of actions[0]
    ChunkStamps stamps{};

    bool covers(std::int64_t step) const { return step >= first_step && step < first_step + count; }
};

// Producer-only counters.
struct ProducerStats {
    std::uint64_t chunks_published = 0;
    std::uint64_t publish_failed = 0;  // ready ring full: consumer badly behind
    std::uint64_t acquire_failed = 0;  // pool dry
};

// Consumer-only counters.
struct ConsumerStats {
    std::uint64_t chunks_accepted = 0;
    std::uint64_t chunks_superseded = 0;  // newer chunk arrived before this was accepted
    std::uint64_t actions_discarded = 0;  // stale prefix skipped at a seam (§4.3)
    std::uint64_t steps_skipped = 0;      // the loop overran a whole period
    std::uint64_t underruns = 0;          // pops that could not be served (#2)
};

class ActionChunkQueue {
   public:
    // The seam discontinuity is an L-infinity distance in action units, not a
    // duration, but it wants the same percentiles as everything else. Stored in
    // millionths so it fits the integer recorder; divide with from_seam_units().
    static constexpr std::int64_t kSeamScale = 1'000'000;
    static double from_seam_units(std::int64_t v) {
        return static_cast<double>(v) / static_cast<double>(kSeamScale);
    }

    explicit ActionChunkQueue(const RuntimeConfig &cfg, std::size_t seam_capacity = 4096)
        : chunk_size_(cfg.chunk_size),
          action_dim_(cfg.action_dim),
          refresh_trigger_(cfg.refresh_trigger),
          stitching_(cfg.stitching),
          refresh_policy_(cfg.refresh_policy),
          base_execution_horizon_(cfg.rtc.execution_horizon),
          inference_delay_(cfg.rtc.inference_delay),
          execution_horizon_(cfg.rtc.execution_horizon),
          pool_(kChunkSlots),
          ready_(kChunkSlots),
          free_(kChunkSlots),
          seam_(seam_capacity) {
        if (cfg.chunk_size > kChunkSize) {
            throw std::invalid_argument(
                "chunk_size (" + std::to_string(cfg.chunk_size) +
                ") exceeds the compiled slot size " + std::to_string(kChunkSize) +
                ": recompile against the checkpoint, don't allocate at runtime");
        }
        if (cfg.action_dim > kPaddedActionDim) {
            throw std::invalid_argument("action_dim exceeds kPaddedActionDim");
        }
        for (std::uint32_t i = 0; i < kChunkSlots; ++i) free_.push(i);
    }

    // --- producer side: the inference worker ---------------------------------

    // A slot to write the next chunk into. nullptr means the pool is dry, which
    // means the consumer has not freed anything — do not spin on it, skip the
    // inference and let the trigger fire again.
    Chunk *acquire() {
        std::uint32_t slot = 0;
        if (!free_.pop(slot)) {
            ++pstats_.acquire_failed;
            return nullptr;
        }
        Chunk &c = pool_[slot];
        c.count = 0;
        return &c;
    }

    // Hands the chunk to the control thread. Never blocks (§7: block-on-full is
    // correct for logs and wrong for control). On failure the producer KEEPS the
    // slot — it must not push to free_, which the consumer owns as its producer
    // end, or the SPSC contract on that ring is broken by a second writer.
    bool publish(Chunk *c) {
        if (!ready_.push(slot_of(c))) {
            ++pstats_.publish_failed;
            return false;
        }
        ++pstats_.chunks_published;
        return true;
    }

    // §4.4's trigger. The ready check is what stops one slow accept from firing a
    // second inference for a chunk that is already computed and waiting.
    bool should_refresh() const {
        if (!ready_.empty()) return false;
        if (remaining() == 0) return true;
        switch (refresh_policy_) {
            case RefreshPolicy::Tail:
                return remaining() <= refresh_trigger_;
            case RefreshPolicy::Continuous:
                return true;
            case RefreshPolicy::Horizon:
                return actions_since_accept_.load(std::memory_order_relaxed) >=
                       std::max(0, execution_horizon() - inference_delay());
        }
        return false;
    }

    void update_rtc_timing(int measured_delay) noexcept {
        const int delay = std::max(1, std::min(measured_delay, chunk_size_));
        const int horizon = std::min(chunk_size_, std::max(base_execution_horizon_, delay + 1));
        inference_delay_.store(delay, std::memory_order_relaxed);
        execution_horizon_.store(horizon, std::memory_order_relaxed);
    }
    int inference_delay() const noexcept {
        return inference_delay_.load(std::memory_order_relaxed);
    }
    int execution_horizon() const noexcept {
        return execution_horizon_.load(std::memory_order_relaxed);
    }

    // Actions the consumer can still emit without a new chunk. Written by the
    // consumer, read here — relaxed on both ends, because R is slack by
    // construction and a trigger built on an exact count would be a trigger with
    // no margin.
    int remaining() const { return remaining_.load(std::memory_order_relaxed); }

    // Temporary in-process RTC feedback (§4.5). Under Stitching::Rtc what gets
    // emitted is bit-identical to what the worker generated (Rtc has discard
    // semantics — no blending), so an in-process worker that retains its last
    // generated chunk can slice it using this cursor.
    //
    // This is deliberately NOT the eventual process-boundary contract. The
    // Python model transport must carry {chunk_id, first_step, committed padded
    // actions}; a bare cursor cannot identify the active chunk after a
    // supersession. That snapshot belongs to the transport/loop milestone, not
    // this SPSC ownership primitive.
    std::int64_t last_emitted_step() const {
        return last_emitted_step_.load(std::memory_order_acquire);
    }

    // --- consumer side: the control thread ----------------------------------

    // Serve `step`. Returns false on underrun (invariant #2) — the caller applies
    // its UnderrunPolicy and counts it. Never blocks, never waits on the GPU
    // (invariant #1), allocates nothing.
    //
    // `rec` is filled with the §5.2 chain: chunk stamps, index within the chunk,
    // the deadline it was supposed to be popped at, and the instant it was.
    bool pop(std::int64_t step, TimePoint deadline, Action &out, ActionRecord &rec) {
        accept_ready(step);

        // A skipped tick must not emit the action that belonged to it. Absolute
        // step indexing makes that automatic; counting it is how the loop's
        // overrun surfaces at all.
        if (last_step_ >= 0 && step > last_step_ + 1) {
            cstats_.steps_skipped += static_cast<std::uint64_t>(step - last_step_ - 1);
        }

        if (!cur_ || !cur_->covers(step)) {
            ++cstats_.underruns;
            remaining_.store(0, std::memory_order_relaxed);
            return false;
        }

        const std::int64_t idx = step - cur_->first_step;
        out = cur_->actions[static_cast<std::size_t>(idx)];

        rec.chunk = cur_->stamps;

        // Temporal ensembling: average the overlapping predictions of two chunks
        // (§4.3). Only the real dims — the padded tail is not a prediction.
        if (stitching_ == Stitching::Ensemble && prev_ && prev_->covers(step)) {
            const Action &b = prev_->actions[static_cast<std::size_t>(step - prev_->first_step)];
            for (int d = 0; d < action_dim_; ++d) {
                out[static_cast<std::size_t>(d)] =
                    0.5f * (out[static_cast<std::size_t>(d)] + b[static_cast<std::size_t>(d)]);
            }
            // A blended action is two opinions formed from different
            // observations, so it inherits the age of the OLDER one. Pessimistic
            // on purpose: staleness is a bound, and reporting the newer stamp
            // would flatter it.
            if (prev_->stamps.t_obs_capture < rec.chunk.t_obs_capture) {
                rec.chunk.t_obs_capture = prev_->stamps.t_obs_capture;
            }
        }

        rec.index = static_cast<int>(idx);
        rec.t_deadline = deadline;
        rec.t_emit = now();

        last_ = out;
        last_step_ = step;
        last_emitted_step_.store(step, std::memory_order_release);
        remaining_.store(remaining_from(step + 1), std::memory_order_relaxed);
        actions_since_accept_.fetch_add(1, std::memory_order_relaxed);

        // The old opinion is done once the cursor passes its end.
        if (prev_ && !prev_->covers(step + 1)) {
            release(prev_);
            prev_ = nullptr;
        }

        return true;
    }

    // --- report -------------------------------------------------------------

    // Column 1 of §11.5's table: the jump at the join, per stitching policy. The
    // queue is the only place that sees both the last action emitted and what the
    // replacement chunk says for the same step.
    PercentileRecorder &seam_linf() { return seam_; }

    const ProducerStats &producer_stats() const { return pstats_; }
    const ConsumerStats &consumer_stats() const { return cstats_; }

    Stitching stitching() const { return stitching_; }
    int chunk_size() const { return chunk_size_; }

   private:
    std::uint32_t slot_of(const Chunk *c) const {
        return static_cast<std::uint32_t>(c - pool_.data());
    }
    void release(const Chunk *c) { free_.push(slot_of(c)); }

    // Newest-wins (§5's rule for the observation side, and the same argument
    // holds here): if two chunks are waiting, the older one describes a world
    // that has already moved. Take the newest, free the rest, count it.
    void accept_ready(std::int64_t step) {
        std::uint32_t slot = 0;
        std::uint32_t take = 0;
        bool got = false;
        while (ready_.pop(slot)) {
            if (got) {
                ++cstats_.chunks_superseded;
                free_.push(take);
            }
            take = slot;
            got = true;
        }
        if (!got) return;

        Chunk *c = &pool_[take];
        record_seam(*c, step);
        ++cstats_.chunks_accepted;
        actions_since_accept_.store(0, std::memory_order_relaxed);

        // Everything in the new chunk before `step` was predicted for timesteps
        // already executed (§4.3's overlap region). Nothing deletes it — absolute
        // indexing just never reads it — but it is the cost of the seam and it
        // gets counted.
        const std::int64_t stale = step - c->first_step;
        if (stale > 0) {
            cstats_.actions_discarded +=
                static_cast<std::uint64_t>(stale < c->count ? stale : c->count);
        }

        switch (stitching_) {
            case Stitching::Discard:
            case Stitching::Rtc:
                // Once the RTC sampler exists, it will generate this chunk
                // conditioned on the committed prefix, leaving no queue-side
                // seam to reconcile. Consumer-side behaviour is discard's.
                if (prev_) {
                    release(prev_);
                    prev_ = nullptr;
                }
                if (cur_) release(cur_);
                cur_ = c;
                break;
            case Stitching::Ensemble:
                if (prev_) release(prev_);
                prev_ = cur_;
                cur_ = c;
                break;
        }

        remaining_.store(remaining_from(last_step_ + 1), std::memory_order_relaxed);
    }

    // Emittable actions from `next_step` onward. Clamped at both ends: a chunk
    // whose prefix is already behind the cursor contributes only its tail, and a
    // chunk entirely behind it contributes nothing.
    int remaining_from(std::int64_t next_step) const {
        if (!cur_) return 0;
        const std::int64_t consumed = next_step - cur_->first_step;
        const std::int64_t left = cur_->count - (consumed > 0 ? consumed : 0);
        return left > 0 ? static_cast<int>(left) : 0;
    }

    void record_seam(const Chunk &next, std::int64_t step) {
        if (last_step_ < 0 || !next.covers(step)) return;
        const Action &a = next.actions[static_cast<std::size_t>(step - next.first_step)];
        float linf = 0.0f;
        for (int d = 0; d < action_dim_; ++d) {
            float diff = a[static_cast<std::size_t>(d)] - last_[static_cast<std::size_t>(d)];
            if (diff < 0.0f) diff = -diff;
            if (diff > linf) linf = diff;
        }
        seam_.record(static_cast<std::int64_t>(
            static_cast<double>(linf) * static_cast<double>(kSeamScale) + 0.5));
    }

    const int chunk_size_;
    const int action_dim_;
    const int refresh_trigger_;
    const Stitching stitching_;
    const RefreshPolicy refresh_policy_;
    const int base_execution_horizon_;
    std::atomic<int> inference_delay_;
    std::atomic<int> execution_horizon_;

    std::vector<Chunk> pool_;
    detail::SpscQueue<std::uint32_t> ready_;  // producer -> consumer
    detail::SpscQueue<std::uint32_t> free_;   // consumer -> producer

    // Consumer-private cursor. No atomics: emitting an action is not a concurrent
    // operation once whole chunks are what cross the thread boundary.
    Chunk *cur_ = nullptr;
    Chunk *prev_ = nullptr;
    std::int64_t last_step_ = -1;
    Action last_{};
    ConsumerStats cstats_{};
    PercentileRecorder seam_;

    ProducerStats pstats_{};

    // Consumer writes, producer reads.
    std::atomic<int> remaining_{0};
    std::atomic<int> actions_since_accept_{0};
    std::atomic<std::int64_t> last_emitted_step_{-1};
};

}  // namespace cerebellum
