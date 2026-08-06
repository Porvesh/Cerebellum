// The measurement, as arithmetic (spec.md §9, §11).
//
// Nothing here times real work except one span — deadlines and stamps are
// synthesised so the assertions are exact. A test that measures the machine
// can only assert "greater than zero", which is not a contract.
//
// The claims that must survive a rewrite: percentiles are nearest-rank over
// every retained sample, lateness is measured against the schedule and not
// against the caller, warmup samples never reach the output, and the steady-state
// path allocates nothing.

#include <cstdio>
#include <vector>

#include "cerebellum/config.hpp"
#include "cerebellum/timing.hpp"
#include "cerebellum/validate.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond);   \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// --- percentiles ------------------------------------------------------------

static void test_percentiles_are_nearest_rank() {
    // 1..1000 ns, recorded out of order so the sort is doing real work.
    PercentileRecorder r(1000);
    for (int i = 1000; i >= 1; --i) r.record(i);

    const Summary s = r.summarize();
    CHECK(s.count == 1000);
    CHECK(s.min_ns == 1);
    CHECK(s.max_ns == 1000);

    // Nearest rank, ceil, 1-based: p50 of 1000 samples is the 500th, not an
    // average of the 500th and 501st. Every reported number is a real sample.
    CHECK(s.p50_ns == 500);
    CHECK(s.p90_ns == 900);
    CHECK(s.p99_ns == 990);

    // The one that breaks under ceil(p/100 * n) in floating point: 99.9/100*1000
    // computes as 999.0000000000001, rounds to rank 1000, and reports the max as
    // p99.9 — hiding the worst sample behind a percentile that isn't the worst.
    CHECK(s.p999_ns == 999);
    CHECK(s.p999_ns != s.max_ns);
}

static void test_percentiles_on_small_samples() {
    PercentileRecorder r(8);
    r.record(10);
    r.record(30);
    r.record(20);
    CHECK(r.percentile(50.0) == 20);   // ceil(1.5) -> 2nd of 3
    CHECK(r.percentile(99.9) == 30);
    CHECK(r.percentile(0.0) == 10);    // clamped to rank 1, never rank 0

    PercentileRecorder empty(4);
    CHECK(empty.empty());
    CHECK(empty.percentile(99.0) == 0);
    CHECK(empty.summarize().count == 0);
}

static void test_recording_after_summarize() {
    // summarize() sorts in place. A recorder that keeps being written to after a
    // report must not return the stale ordering.
    PercentileRecorder r(16);
    r.record(5);
    r.record(1);
    CHECK(r.summarize().max_ns == 5);
    r.record(9);
    CHECK(r.summarize().max_ns == 9);
    CHECK(r.summarize().count == 3);
}

// --- warmup and capacity ----------------------------------------------------

static void test_warmup_is_discarded() {
    // §9: discard the first N, and state N in the writeup — so N survives into
    // the Summary rather than living only in the experiment script.
    PercentileRecorder r(100, /*warmup=*/50);
    for (int i = 1; i <= 60; ++i) r.record(i * 1000);

    const Summary s = r.summarize();
    CHECK(s.count == 10);
    CHECK(s.warmup == 50);
    CHECK(s.min_ns == 51'000);   // sample 51 is the first kept
    CHECK(s.max_ns == 60'000);
}

static void test_steady_state_does_not_allocate() {
    // Invariant #3. The vector is at capacity from construction; a sample past
    // the end is counted, not grown, because a reallocation on the control
    // thread is a latency spike that lands in the very tail being measured.
    PercentileRecorder r(4);
    const auto* base = r.samples().data();
    const std::size_t cap = r.samples().capacity();

    for (int i = 0; i < 100; ++i) r.record(7);

    CHECK(r.size() == 4);
    CHECK(r.dropped() == 96);
    CHECK(r.samples().capacity() == cap);
    CHECK(r.samples().data() == base);   // never reallocated

    // dropped() is loud on purpose: these percentiles describe a prefix of the
    // run, and the summary carries that fact.
    CHECK(r.summarize().dropped == 96);

    r.reset();
    CHECK(r.size() == 0);
    CHECK(r.dropped() == 0);
    CHECK(r.samples().capacity() == cap);   // reusable across sweep points
    CHECK(r.samples().data() == base);
}

// --- the schedule clock -----------------------------------------------------

static void test_deadlines_live_on_a_fixed_grid() {
    const TimePoint t0 = now();
    ScheduleClock sched(t0);

    CHECK(sched.period() == kControlPeriod);
    CHECK(sched.deadline(0) == t0);
    CHECK(sched.deadline(1) - sched.deadline(0) == kControlPeriod);

    // Multiplied, not accumulated: after 108'000 ticks (an hour) the grid is
    // still exactly 108'000 periods from t0. Accumulating would have compounded
    // the third of a nanosecond 1e9/30 truncates away (config.hpp:16).
    CHECK(sched.deadline(108'000) - t0 == kControlPeriod * 108'000);

    CHECK(sched.index_at(t0) == 0);
    CHECK(sched.index_at(t0 + kControlPeriod * 2) == 2);
    CHECK(sched.index_at(t0 + kControlPeriod * 2 + kControlPeriod / 2) == 2);
}

static void test_lateness_is_measured_against_the_schedule() {
    const TimePoint t0 = now();
    ScheduleClock sched(t0);

    // Tick 4 fires 5 ms late.
    const TimePoint fired = sched.deadline(4) + std::chrono::milliseconds(5);
    CHECK(sched.lateness_ns(4, fired) == 5'000'000);

    // Early is negative, not zero and not an error — the distribution has a left
    // side and clamping it would flatter the p50.
    const TimePoint early = sched.deadline(4) - std::chrono::milliseconds(2);
    CHECK(sched.lateness_ns(4, early) == -2'000'000);
}

static void test_the_schedule_does_not_slip_after_an_overrun() {
    // Coordinated omission, made concrete. Tick 4 runs 40 ms long — more than a
    // whole period — and tick 5 therefore starts already behind.
    const TimePoint t0 = now();
    ScheduleClock sched(t0);

    const TimePoint tick4_done = sched.deadline(4) + std::chrono::milliseconds(40);

    // Self-relative timing of tick 5's work: it does 1 ms of work and looks fine.
    const TimePoint tick5_done = tick4_done + std::chrono::milliseconds(1);
    const std::int64_t self_relative = (tick5_done - tick4_done).count();
    CHECK(to_ms(self_relative) < 2.0);   // "1 ms, we're healthy"

    // Deadline-relative: tick 5 was 6.7 ms late before it started and finished
    // 7.7 ms past its grid point. Same instant, and the only version that
    // reports the miss.
    const std::int64_t against_schedule = sched.lateness_ns(5, tick5_done);
    CHECK(against_schedule > 0);
    CHECK(to_ms(against_schedule) > 7.0 && to_ms(against_schedule) < 8.0);

    // And the overrun is visible as a skipped grid point rather than a renumbered
    // schedule: tick 4 finished inside tick 5's slot.
    CHECK(sched.index_at(tick4_done) == 5);
}

static void test_cursor_advances_independently_of_the_clock() {
    ScheduleClock sched(now());
    CHECK(sched.index() == 0);
    const TimePoint first = sched.next_deadline();
    sched.advance();
    CHECK(sched.index() == 1);
    // The next target is one period after the previous GRID POINT, not one period
    // after whenever advance() happened to be called.
    CHECK(sched.next_deadline() - first == kControlPeriod);
}

// --- stage spans ------------------------------------------------------------

static void test_scoped_span_measures_work() {
    // The one self-relative measurement that is correct: a stage duration
    // measures work, not adherence to a deadline.
    StageTable stages(64);
    {
        ScopedSpan span(stages[Stage::Preprocess]);
        sleep_until(now() + std::chrono::milliseconds(2));
    }
    CHECK(stages[Stage::Preprocess].size() == 1);
    CHECK(stages[Stage::Preprocess].percentile(50.0) >= 1'000'000);   // >= 1 ms

    // Every stage in §11.1's table gets its own recorder, independently warmed.
    StageTable warmed(64, /*warmup=*/2);
    for (int i = 0; i < 5; ++i) warmed[Stage::Backbone].record(90'000'000);
    CHECK(warmed[Stage::Backbone].size() == 3);
    CHECK(warmed[Stage::D2H].size() == 0);
}

// --- the §5.2 chain and the invariants it counts -----------------------------

static ActionRecord synth_action(TimePoint capture, const ScheduleClock& sched,
                                 std::int64_t tick, int index_in_chunk,
                                 double budget_ms) {
    ActionRecord rec;
    rec.chunk.obs_seq = 7;
    rec.chunk.chunk_id = 1;
    rec.chunk.t_obs_capture = capture;
    rec.chunk.t_infer_start = capture;
    rec.chunk.t_infer_end =
        capture + Nanos{static_cast<std::int64_t>(budget_ms * 1e6)};
    rec.index = index_in_chunk;
    rec.t_deadline = sched.deadline(tick);
    rec.t_emit = rec.t_deadline;   // emitted exactly on time
    return rec;
}

static void test_staleness_accumulates_across_a_chunk() {
    // staleness(i) = budget + i * period (§15.1), measured from CAPTURE — an
    // action inherits the age of the observation plus its own position in the
    // chunk. This is the same arithmetic validate.hpp's
    // max_actions_within_staleness() does, reached from the other direction.
    const TimePoint capture = now();
    const std::int64_t budget_ns = static_cast<std::int64_t>(kBudgetTargetMs * 1e6);
    ScheduleClock sched(capture + Nanos{budget_ns});

    const ActionRecord a0 = synth_action(capture, sched, 0, 0, kBudgetTargetMs);
    CHECK(a0.staleness_ms() > 145.9 && a0.staleness_ms() < 146.1);
    CHECK(a0.within_staleness_bound());
    CHECK(a0.lateness_ns() == 0);
    CHECK(a0.chunk.obs_to_ready_ns() == budget_ns);

    const ActionRecord a1 = synth_action(capture, sched, 1, 1, kBudgetTargetMs);
    CHECK(a1.staleness_ms() > 179.0 && a1.staleness_ms() < 179.4);
    CHECK(a1.within_staleness_bound());

    // The third action of the chunk is already past the bound, and validate.hpp
    // says exactly two fit. Two independent statements of §15.1 that must agree.
    const ActionRecord a2 = synth_action(capture, sched, 2, 2, kBudgetTargetMs);
    CHECK(a2.staleness_ms() > 212.0);
    CHECK(!a2.within_staleness_bound());
    CHECK(max_actions_within_staleness() == 2);
}

static void test_control_metrics_count_the_invariants() {
    // Invariants #2 and #4 are counted and reported, not thrown on. A runtime
    // that refused to run under §15.1 would just be a runtime nobody can run.
    const TimePoint capture = now();
    ScheduleClock sched(capture + Nanos{static_cast<std::int64_t>(kBudgetTargetMs * 1e6)});
    ControlMetrics m(128);

    m.on_emit(synth_action(capture, sched, 0, 0, kBudgetTargetMs));   // in bound
    m.on_emit(synth_action(capture, sched, 1, 1, kBudgetTargetMs));   // in bound
    m.on_emit(synth_action(capture, sched, 2, 2, kBudgetTargetMs));   // violation
    m.on_underrun();

    CHECK(m.ticks == 4);
    CHECK(m.underruns == 1);
    CHECK(m.staleness_violations == 1);

    // An underrun contributes a tick but no lateness or staleness sample: there
    // was no action to be late or stale. Three emits, three samples.
    CHECK(m.lateness.size() == 3);
    CHECK(m.staleness.size() == 3);
    CHECK(m.lateness.percentile(99.0) == 0);   // all emitted exactly on time
}

int main() {
    test_percentiles_are_nearest_rank();
    test_percentiles_on_small_samples();
    test_recording_after_summarize();
    test_warmup_is_discarded();
    test_steady_state_does_not_allocate();
    test_deadlines_live_on_a_fixed_grid();
    test_lateness_is_measured_against_the_schedule();
    test_the_schedule_does_not_slip_after_an_overrun();
    test_cursor_advances_independently_of_the_clock();
    test_scoped_span_measures_work();
    test_staleness_accumulates_across_a_chunk();
    test_control_metrics_count_the_invariants();

    if (g_failures == 0) {
        std::printf("test_timing: all checks passed\n");
    } else {
        std::printf("test_timing: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
