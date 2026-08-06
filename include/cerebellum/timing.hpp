// The measurement harness (spec.md §9, §11).
//
// Three pieces, in the order §12 names them: a percentile recorder that keeps
// every sample, a schedule clock that measures against the deadline instead of
// against itself, and the §5.2 timestamp chain that travels with a chunk.
//
// This is not observability bolted on the side. The clock the recorder reads is
// the same one the control thread ticks on, staleness (invariant #4) is a
// subtraction between two of these stamps, and an underrun (#2) is only
// definable relative to a deadline. The percentiles are the deliverable; the
// clock is load-bearing.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "cerebellum/config.hpp"

namespace cerebellum {

// --- timebase ---------------------------------------------------------------
// One timebase everywhere, converted only at the edges (§5.2). libstdc++ maps
// steady_clock onto CLOCK_MONOTONIC, which is the clock the log schema names.

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Nanos = std::chrono::nanoseconds;

static_assert(Clock::is_steady,
              "the control loop cannot tick on a clock that can be adjusted");

inline TimePoint now() noexcept { return Clock::now(); }

inline double to_ms(std::int64_t ns) { return static_cast<double>(ns) / 1e6; }
inline double to_ms(Nanos d) { return to_ms(d.count()); }

// Wakeup granularity is ~50-100 us on a stock kernel, which is 0.3% of a 33 ms
// period and fine for the cadence — but not for a stage span. If p99 lateness
// turns out to be dominated by wakeup slop, loop.hpp switches to sleep-then-spin
// over the last millisecond; the grid it sleeps to does not change.
inline void sleep_until(TimePoint t) { std::this_thread::sleep_until(t); }

// --- percentiles ------------------------------------------------------------

// No mean. §9: a mean latency is meaningless against a deadline, and a running
// mean is the specific thing that makes the tail unrecoverable after the fact.
struct Summary {
    std::size_t count = 0;    // samples kept
    std::size_t warmup = 0;   // discarded up front; §9 says state N in the writeup
    std::size_t dropped = 0;  // arrived after capacity was full — see record()
    std::int64_t min_ns = 0;
    std::int64_t p50_ns = 0;
    std::int64_t p90_ns = 0;
    std::int64_t p99_ns = 0;
    std::int64_t p999_ns = 0;
    std::int64_t max_ns = 0;
};

// Stores every sample and computes percentiles at report time. Samples are
// signed nanoseconds because lateness can be negative (a tick that fired early).
class PercentileRecorder {
public:
    explicit PercentileRecorder(std::size_t capacity, std::size_t warmup = 0)
        : warmup_(warmup), warmup_remaining_(warmup) {
        samples_.reserve(capacity);
    }

    // Called from the control thread every tick. Allocates nothing: the vector
    // is at capacity from construction, and a sample past the end is counted
    // rather than grown (invariant #3 — zero allocation in the steady state).
    // A non-zero dropped() means the run outlived its capacity and the reported
    // percentiles describe a prefix; print it, don't quietly trust the p99.
    void record(std::int64_t ns) noexcept {
        if (warmup_remaining_ > 0) {
            --warmup_remaining_;
            return;
        }
        if (samples_.size() == samples_.capacity()) {
            ++dropped_;
            return;
        }
        samples_.push_back(ns);
        sorted_ = false;
    }

    void record(Nanos d) noexcept { record(d.count()); }

    std::size_t size() const noexcept { return samples_.size(); }
    std::size_t dropped() const noexcept { return dropped_; }
    std::size_t warmup() const noexcept { return warmup_; }
    bool empty() const noexcept { return samples_.empty(); }

    // The full distribution, for the CSV. §9 keeps every sample so a percentile
    // can be recomputed later without re-running the experiment.
    const std::vector<std::int64_t>& samples() const noexcept { return samples_; }

    // Nearest-rank, never interpolated: a reported p99 is a sample that really
    // happened. Sorts on first use, so this is a report-time call — never invoke
    // it from the control thread.
    //
    // The rank is integer permille arithmetic rather than ceil(p/100 * n):
    // 99.9/100.0*1000 is 999.0000000000001 in binary floating point, which
    // rounds up to rank 1000 and silently reports the max as p99.9.
    std::int64_t percentile(double p) {
        if (samples_.empty()) return 0;
        sort();
        const std::int64_t permille = static_cast<std::int64_t>(p * 10.0 + 0.5);
        const std::int64_t n = static_cast<std::int64_t>(samples_.size());
        std::int64_t rank = (permille * n + 999) / 1000;  // ceil, 1-based
        if (rank < 1) rank = 1;
        if (rank > n) rank = n;
        return samples_[static_cast<std::size_t>(rank - 1)];
    }

    Summary summarize() {
        Summary s;
        s.count = samples_.size();
        s.warmup = warmup_;
        s.dropped = dropped_;
        if (samples_.empty()) return s;
        sort();
        s.min_ns = samples_.front();
        s.max_ns = samples_.back();
        s.p50_ns = percentile(50.0);
        s.p90_ns = percentile(90.0);
        s.p99_ns = percentile(99.0);
        s.p999_ns = percentile(99.9);
        return s;
    }

    // Keeps the capacity, so a sweep can reuse one recorder per point without
    // reallocating between them.
    void reset() noexcept {
        samples_.clear();
        dropped_ = 0;
        warmup_remaining_ = warmup_;
        sorted_ = true;
    }

private:
    void sort() {
        if (!sorted_) {
            std::sort(samples_.begin(), samples_.end());
            sorted_ = true;
        }
    }

    std::vector<std::int64_t> samples_;
    std::size_t warmup_ = 0;
    std::size_t warmup_remaining_ = 0;
    std::size_t dropped_ = 0;
    bool sorted_ = true;
};

// --- the schedule clock -----------------------------------------------------
// §9's one non-negotiable rule. Deadlines live on an absolute grid anchored at
// t0 and never slip to "whenever the last tick finished", because a schedule
// that slips reports every miss as on-time — coordinated omission, and it erases
// exactly the tail this project exists to measure.
//
//   WRONG:  start = now(); work(); record(now() - start)
//   RIGHT:  deadline = t0 + i * period; work(); record(now() - deadline)
//
// There is deliberately no start()/stop() pair on this class. The API that
// produces the wrong number is the one that isn't offered.

class ScheduleClock {
public:
    explicit ScheduleClock(Nanos period = kControlPeriod)
        : t0_(now()), period_(period) {}
    ScheduleClock(TimePoint t0, Nanos period = kControlPeriod)
        : t0_(t0), period_(period) {}

    TimePoint origin() const noexcept { return t0_; }
    Nanos period() const noexcept { return period_; }

    // Grid point i. Multiplication, not accumulation: adding period_ each tick
    // would compound the third of a nanosecond that 1e9/30 truncates away
    // (config.hpp:16).
    TimePoint deadline(std::int64_t i) const noexcept { return t0_ + period_ * i; }

    // Signed: negative is early, positive is late. This is the number the
    // lateness recorder stores, and the only one that answers "did we hold the
    // cadence".
    std::int64_t lateness_ns(std::int64_t i, TimePoint t) const noexcept {
        return (t - deadline(i)).count();
    }
    std::int64_t lateness_ns(std::int64_t i) const noexcept {
        return lateness_ns(i, now());
    }

    // Which grid point `t` falls in. Lets the loop notice it overran a whole
    // period and skipped a tick, rather than quietly renumbering the schedule.
    std::int64_t index_at(TimePoint t) const noexcept { return (t - t0_) / period_; }

    // The control thread's cursor. Advancing is separate from measuring so a
    // tick that runs long still targets the next grid point instead of one
    // period after it happened to finish.
    std::int64_t index() const noexcept { return i_; }
    TimePoint next_deadline() const noexcept { return deadline(i_); }
    void advance() noexcept { ++i_; }

private:
    TimePoint t0_;
    Nanos period_;
    std::int64_t i_ = 0;
};

// --- stage breakdown (§11.1) ------------------------------------------------
// Wall spans only. A wall timer wrapped around an async launch measures nothing
// (§9), so GPU spans are timed with CUDA events on the side of the pybind seam
// that owns the stream and arrive here as already-elapsed durations.
//
// Expect the stage sum to exceed the pipelined end-to-end time. §9 says say so
// in the writeup: that is what overlap looks like, not a bug.

enum class Stage { Preprocess, H2D, Vision, Backbone, ActionHead, D2H };

inline constexpr std::size_t kStageCount = 6;

inline const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Preprocess: return "preprocess";
        case Stage::H2D:        return "h2d";
        case Stage::Vision:     return "vision";
        case Stage::Backbone:   return "backbone";
        case Stage::ActionHead: return "action_head";
        case Stage::D2H:        return "d2h";
    }
    return "unknown";
}

// Self-relative, and this is the one place that is correct: a stage duration
// measures work performed. Adherence to the cadence is the opposite case and
// goes through ScheduleClock — don't reach for this to time a tick.
class ScopedSpan {
public:
    explicit ScopedSpan(PercentileRecorder& into) noexcept
        : into_(into), start_(now()) {}
    ~ScopedSpan() { into_.record((now() - start_).count()); }

    ScopedSpan(const ScopedSpan&) = delete;
    ScopedSpan& operator=(const ScopedSpan&) = delete;

private:
    PercentileRecorder& into_;
    TimePoint start_;
};

// One recorder per stage — §11.1's table, p50/p99 per stage at batch 1.
class StageTable {
public:
    explicit StageTable(std::size_t capacity, std::size_t warmup = 0) {
        recorders_.reserve(kStageCount);
        for (std::size_t i = 0; i < kStageCount; ++i) {
            recorders_.emplace_back(capacity, warmup);
        }
    }

    PercentileRecorder& operator[](Stage s) noexcept {
        return recorders_[static_cast<std::size_t>(s)];
    }

    void print(std::FILE* out = stdout) {
        std::fprintf(out, "%-12s %8s %8s %8s %8s %8s   (ms)\n", "stage", "p50", "p90",
                     "p99", "p99.9", "max");
        for (std::size_t i = 0; i < kStageCount; ++i) {
            const Summary s = recorders_[i].summarize();
            std::fprintf(out, "%-12s %8.3f %8.3f %8.3f %8.3f %8.3f   n=%zu\n",
                         stage_name(static_cast<Stage>(i)), to_ms(s.p50_ns),
                         to_ms(s.p90_ns), to_ms(s.p99_ns), to_ms(s.p999_ns),
                         to_ms(s.max_ns), s.count);
        }
    }

private:
    std::vector<PercentileRecorder> recorders_;
};

// --- the §5.2 causal chain --------------------------------------------------
// Not just actions: enough to reconstruct what the policy saw, so a replay can
// be checked against it. The chunk queue carries ChunkStamps; the control thread
// completes an ActionRecord per pop.

struct ChunkStamps {
    std::uint64_t obs_seq = 0;  // which observation produced this chunk
    std::uint64_t chunk_id = 0;
    TimePoint t_obs_capture{};  // when the frame was captured
    TimePoint t_infer_start{};  // when the forward began
    TimePoint t_infer_end{};    // when the chunk was ready

    // What R has to cover (§15.2): observation to chunk-ready, not the forward
    // pass alone. In phase 1 the synthetic source is instantaneous and the two
    // coincide, which is exactly why the distinction is written down now.
    std::int64_t obs_to_ready_ns() const noexcept {
        return (t_infer_end - t_obs_capture).count();
    }
    std::int64_t forward_ns() const noexcept {
        return (t_infer_end - t_infer_start).count();
    }
};

struct ActionRecord {
    ChunkStamps chunk{};
    int index = 0;           // which action within the chunk
    TimePoint t_emit{};      // when the control thread popped it
    TimePoint t_deadline{};  // when it was SUPPOSED to be popped

    std::int64_t lateness_ns() const noexcept { return (t_emit - t_deadline).count(); }

    // Invariant #4, as a subtraction. Note what it measures from: the capture
    // stamp, not the start of the forward — an action inherits the age of the
    // observation that produced it plus its own position in the chunk.
    std::int64_t staleness_ns() const noexcept {
        return (t_emit - chunk.t_obs_capture).count();
    }
    double staleness_ms() const noexcept { return to_ms(staleness_ns()); }
    bool within_staleness_bound() const noexcept {
        return staleness_ms() <= kMaxStalenessMs;
    }
};

// --- what phase 1 reports ---------------------------------------------------
// §11's deliverables 1-3, plus the two invariants that are counted rather than
// asserted. Underruns and staleness violations don't throw: they are the
// measurement. §15.1 guarantees the staleness counter fires under any default
// config, and the §11.2 sweep is what settles that.

struct ControlMetrics {
    PercentileRecorder lateness;   // now - deadline, signed (§9)
    PercentileRecorder staleness;  // t_emit - t_obs_capture (invariant #4)

    std::uint64_t ticks = 0;
    std::uint64_t underruns = 0;             // invariant #2
    std::uint64_t staleness_violations = 0;  // invariant #4

    explicit ControlMetrics(std::size_t capacity, std::size_t warmup = 0)
        : lateness(capacity, warmup), staleness(capacity, warmup) {}

    // One call per tick from the control thread. Allocation-free.
    void on_emit(const ActionRecord& rec) noexcept {
        ++ticks;
        lateness.record(rec.lateness_ns());
        staleness.record(rec.staleness_ns());
        if (!rec.within_staleness_bound()) ++staleness_violations;
    }

    // A tick the queue could not serve. Counted, and the tick still happened —
    // whatever UnderrunPolicy produced is what went to the actuator.
    void on_underrun() noexcept {
        ++ticks;
        ++underruns;
    }

    void print(std::FILE* out = stdout) {
        const Summary l = lateness.summarize();
        const Summary s = staleness.summarize();
        std::fprintf(out,
                     "ticks=%llu  underruns=%llu  staleness_violations=%llu  warmup=%zu\n",
                     static_cast<unsigned long long>(ticks),
                     static_cast<unsigned long long>(underruns),
                     static_cast<unsigned long long>(staleness_violations), l.warmup);
        std::fprintf(out, "%-12s %8s %8s %8s %8s %8s   (ms)\n", "metric", "p50", "p90",
                     "p99", "p99.9", "max");
        std::fprintf(out, "%-12s %8.3f %8.3f %8.3f %8.3f %8.3f   n=%zu\n", "lateness",
                     to_ms(l.p50_ns), to_ms(l.p90_ns), to_ms(l.p99_ns), to_ms(l.p999_ns),
                     to_ms(l.max_ns), l.count);
        std::fprintf(out, "%-12s %8.3f %8.3f %8.3f %8.3f %8.3f   n=%zu  bound=%.1f\n",
                     "staleness", to_ms(s.p50_ns), to_ms(s.p90_ns), to_ms(s.p99_ns),
                     to_ms(s.p999_ns), to_ms(s.max_ns), s.count, kMaxStalenessMs);
    }
};

// Every sample, not a summary — the committed CSV in results/ is what makes a
// percentile recomputable without re-running the experiment.
inline void write_samples_csv(std::FILE* out, const char* label,
                              const PercentileRecorder& r) {
    std::fprintf(out, "label,i,ns\n");
    const auto& v = r.samples();
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::fprintf(out, "%s,%zu,%lld\n", label, i, static_cast<long long>(v[i]));
    }
}

}  // namespace cerebellum
