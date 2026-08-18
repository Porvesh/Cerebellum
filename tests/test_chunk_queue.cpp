// The queue's logic, single-threaded (spec.md §4.2, §4.3, §4.4).
//
// Chunk ownership crossing threads is test_chunk_queue_mt.cpp's problem, under
// TSAN. Everything here is the consumer's cursor arithmetic, which is where the
// stitching policies actually live and where absolute-step indexing either works
// or quietly emits an action that belonged to a tick that already passed.

#include <cstdio>

#include "cerebellum/chunk_queue.hpp"
#include "cerebellum/config.hpp"
#include "cerebellum/timing.hpp"
#include "cerebellum/validate.hpp"

using namespace cerebellum;

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_THROWS(expr)                                                                \
    do {                                                                                  \
        bool threw = false;                                                               \
        try {                                                                             \
            (void)(expr);                                                                 \
        } catch (const std::exception&) {                                                 \
            threw = true;                                                                 \
        }                                                                                 \
        if (!threw) {                                                                     \
            std::printf("  FAIL %s:%d  expected throw: %s\n", __FILE__, __LINE__, #expr); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

// A chunk whose every action is `value` in every real dim, so a seam or a blend
// is readable as a single number.
static Chunk* fill(ActionChunkQueue& q, std::int64_t first_step, int count, float value,
                   std::uint64_t chunk_id = 0, TimePoint capture = now()) {
    Chunk* c = q.acquire();
    if (!c) return nullptr;
    c->first_step = first_step;
    c->count = count;
    for (int i = 0; i < count; ++i) {
        for (std::size_t d = 0; d < kPaddedActionDim; ++d) {
            c->actions[static_cast<std::size_t>(i)][d] =
                d < static_cast<std::size_t>(kActionDim) ? value : 0.0f;
        }
    }
    c->stamps.chunk_id = chunk_id;
    c->stamps.obs_seq = chunk_id;
    c->stamps.t_obs_capture = capture;
    c->stamps.t_infer_start = capture;
    c->stamps.t_infer_end = capture + std::chrono::milliseconds(90);
    return c;
}

static RuntimeConfig cfg_with(Stitching s) {
    RuntimeConfig cfg{};
    cfg.stitching = s;
    if (s == Stitching::Rtc) {
        cfg.refresh_policy = RefreshPolicy::Horizon;
        cfg.rtc.execution_horizon = cfg.rtc.inference_delay;
    }
    cfg.validate();
    return cfg;
}

// --- the plain path ---------------------------------------------------------

static void test_publish_then_drain() {
    ActionChunkQueue q(cfg_with(Stitching::Discard));
    ScheduleClock sched(now());

    CHECK(q.publish(fill(q, 0, kChunkSize, 1.0f, /*chunk_id=*/7)));

    Action a{};
    ActionRecord rec;
    for (std::int64_t step = 0; step < kChunkSize; ++step) {
        CHECK(q.pop(step, sched.deadline(step), a, rec));
        CHECK(a[0] == 1.0f);
        CHECK(rec.index == static_cast<int>(step));  // index within the chunk
        CHECK(rec.chunk.chunk_id == 7);              // §5.2 provenance survives
        CHECK(rec.t_deadline == sched.deadline(step));
        CHECK(q.last_emitted_step() == step);
    }
    CHECK(q.consumer_stats().chunks_accepted == 1);
    CHECK(q.consumer_stats().underruns == 0);
    CHECK(q.remaining() == 0);

    // Step 50 is past the chunk: underrun, and it must not wrap to actions[0].
    CHECK(!q.pop(kChunkSize, sched.deadline(kChunkSize), a, rec));
    CHECK(q.consumer_stats().underruns == 1);
}

static void test_underrun_on_empty_queue() {
    ActionChunkQueue q(cfg_with(Stitching::Discard));
    Action a{};
    ActionRecord rec;
    // Nothing published. pop() returns false rather than waiting — invariant #1
    // is the whole architecture, and this is the one line that could break it.
    CHECK(!q.pop(0, now(), a, rec));
    CHECK(q.consumer_stats().underruns == 1);
    CHECK(q.remaining() == 0);
}

// --- §4.4, the refresh trigger ----------------------------------------------

static void test_refresh_trigger() {
    RuntimeConfig cfg = cfg_with(Stitching::Discard);
    ActionChunkQueue q(cfg);
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    // Nothing queued: the trigger is already asserted, which is how the loop
    // gets its first chunk.
    CHECK(q.should_refresh());

    q.publish(fill(q, 0, kChunkSize, 1.0f));
    for (std::int64_t step = 0; step < 10; ++step) {
        q.pop(step, sched.deadline(step), a, rec);
    }
    CHECK(q.remaining() == kChunkSize - 10);
    CHECK(!q.should_refresh());

    // Drain to exactly R remaining.
    for (std::int64_t step = 10; step < kChunkSize - cfg.refresh_trigger; ++step) {
        q.pop(step, sched.deadline(step), a, rec);
    }
    CHECK(q.remaining() == cfg.refresh_trigger);
    CHECK(q.should_refresh());

    // A published-but-unaccepted chunk suppresses the trigger. Without this the
    // worker fires a second inference for a chunk that is already computed and
    // sitting in the ring — wasted GPU and an extra seam, for nothing.
    q.publish(fill(q, kChunkSize - cfg.refresh_trigger, kChunkSize, 2.0f));
    CHECK(!q.should_refresh());
}

static void test_continuous_refresh_starts_after_each_accept() {
    RuntimeConfig cfg = cfg_with(Stitching::Discard);
    cfg.refresh_policy = RefreshPolicy::Continuous;
    ActionChunkQueue q(cfg);
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    CHECK(q.should_refresh());
    CHECK(q.publish(fill(q, 0, kChunkSize, 1.0f)));
    CHECK(!q.should_refresh());  // one completed chunk is waiting for control
    CHECK(q.pop(0, sched.deadline(0), a, rec));
    CHECK(q.remaining() == kChunkSize - 1);
    CHECK(q.should_refresh());  // do not wait for the tail trigger
}

static void test_rtc_preserves_configured_refresh_gap() {
    RuntimeConfig zero_gap = cfg_with(Stitching::Rtc);
    zero_gap.rtc.execution_horizon = zero_gap.rtc.inference_delay;
    ActionChunkQueue earliest(zero_gap);
    earliest.update_rtc_timing(8);
    CHECK(earliest.inference_delay() == 8);
    CHECK(earliest.execution_horizon() == 8);

    RuntimeConfig cfg = cfg_with(Stitching::Rtc);
    cfg.rtc.inference_delay = 5;
    cfg.rtc.execution_horizon = 7;  // start after s-d = two actions
    ActionChunkQueue q(cfg);
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    // A slower measurement moves d and s together; the two-action scheduling
    // gap remains a deliberate configuration rather than collapsing to one.
    q.update_rtc_timing(8);
    CHECK(q.inference_delay() == 8);
    CHECK(q.execution_horizon() == 10);

    CHECK(q.publish(fill(q, 0, kChunkSize, 1.0f)));
    CHECK(q.pop(0, sched.deadline(0), a, rec));
    CHECK(!q.should_refresh());
    CHECK(q.pop(1, sched.deadline(1), a, rec));
    CHECK(q.should_refresh());
}

// --- §4.3, discard ----------------------------------------------------------

static void test_discard_skips_the_stale_prefix() {
    ActionChunkQueue q(cfg_with(Stitching::Discard));
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    q.publish(fill(q, 0, 20, 0.0f, /*chunk_id=*/1));
    for (std::int64_t step = 0; step < 5; ++step) {
        q.pop(step, sched.deadline(step), a, rec);
    }

    // Chunk 2 was computed from the observation at step 3 and lands while the
    // consumer is at 5: its actions for steps 3 and 4 describe timesteps already
    // executed. §4.3's overlap region, and exactly what R has to cover.
    q.publish(fill(q, 3, 20, 0.25f, /*chunk_id=*/2));

    CHECK(q.pop(5, sched.deadline(5), a, rec));
    CHECK(a[0] == 0.25f);  // the new chunk, not the old one
    CHECK(rec.chunk.chunk_id == 2);
    CHECK(rec.index == 2);  // step 5 is index 2 of a chunk starting at 3
    CHECK(q.consumer_stats().actions_discarded == 2);

    // The jump at the join is the thing §11.5 has a column for and nobody has
    // published: 0.0 to 0.25 in every real dim.
    CHECK(q.seam_linf().size() == 1);
    const double linf = ActionChunkQueue::from_seam_units(q.seam_linf().percentile(50.0));
    CHECK(linf > 0.2499 && linf < 0.2501);
}

// --- §4.3, temporal ensembling ----------------------------------------------

static void test_ensemble_blends_the_overlap() {
    ActionChunkQueue q(cfg_with(Stitching::Ensemble));
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    const TimePoint old_capture = now();
    q.publish(fill(q, 0, 20, 0.0f, /*chunk_id=*/1, old_capture));
    for (std::int64_t step = 0; step < 5; ++step) {
        q.pop(step, sched.deadline(step), a, rec);
    }

    // Covers 5..24, so it overlaps the old chunk's 0..19 and then runs past it.
    const TimePoint new_capture = old_capture + std::chrono::milliseconds(100);
    q.publish(fill(q, 5, 20, 1.0f, /*chunk_id=*/2, new_capture));

    // Overlap: both chunks have an opinion about step 5. Averaged.
    CHECK(q.pop(5, sched.deadline(5), a, rec));
    CHECK(a[0] == 0.5f);
    // Padded dims are not predictions and are not blended into something else.
    CHECK(a[kPaddedActionDim - 1] == 0.0f);

    // A blend inherits the age of the OLDER opinion. Staleness is a bound, so
    // reporting the newer capture would flatter it.
    CHECK(rec.chunk.t_obs_capture == old_capture);

    // Steps 6..19 are still overlap; from 20 the old chunk has nothing to say and
    // the new one is emitted unblended. That transition is also where prev_ gets
    // returned to the pool.
    for (std::int64_t step = 6; step < 25; ++step) {
        CHECK(q.pop(step, sched.deadline(step), a, rec));
        CHECK(a[0] == (step < 20 ? 0.5f : 1.0f));
        // Once the blend ends, so does the pessimistic capture stamp.
        CHECK(rec.chunk.t_obs_capture == (step < 20 ? old_capture : new_capture));
    }
}

static void test_ensemble_releases_the_old_chunk() {
    // The pool is 4 slots. If prev_ is never freed, ensemble leaks a slot per
    // seam and acquire() starts returning nullptr after four chunks.
    ActionChunkQueue q(cfg_with(Stitching::Ensemble));
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    for (int k = 0; k < 12; ++k) {
        const std::int64_t base = k * 5;
        Chunk* c = fill(q, base, 10, static_cast<float>(k));
        CHECK(c != nullptr);  // the pool must recycle
        if (!c) break;
        q.publish(c);
        for (std::int64_t step = base; step < base + 5; ++step) {
            CHECK(q.pop(step, sched.deadline(step), a, rec));
        }
    }
    CHECK(q.producer_stats().acquire_failed == 0);
    CHECK(q.consumer_stats().underruns == 0);
}

// --- newest-wins and skipped ticks ------------------------------------------

static void test_newest_chunk_wins() {
    ActionChunkQueue q(cfg_with(Stitching::Discard));
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    q.publish(fill(q, 0, 20, 1.0f, 1));
    q.publish(fill(q, 0, 20, 2.0f, 2));  // both waiting, neither accepted yet

    // §5's rule applied to this side of the runtime: the older chunk describes a
    // world that has already moved. Take the newest, free the rest, count it.
    CHECK(q.pop(0, sched.deadline(0), a, rec));
    CHECK(a[0] == 2.0f);
    CHECK(rec.chunk.chunk_id == 2);
    CHECK(q.consumer_stats().chunks_superseded == 1);
    CHECK(q.consumer_stats().chunks_accepted == 1);
}

static void test_skipped_tick_is_not_emitted_late() {
    ActionChunkQueue q(cfg_with(Stitching::Discard));
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    Chunk* c = q.acquire();
    c->first_step = 0;
    c->count = 20;
    for (int i = 0; i < 20; ++i) c->actions[static_cast<std::size_t>(i)][0] = static_cast<float>(i);
    q.publish(c);

    CHECK(q.pop(0, sched.deadline(0), a, rec));
    CHECK(a[0] == 0.0f);

    // The loop overran and steps 1 and 2 never happened. Step 3 must get step
    // 3's action — a queue that popped the next item in sequence would hand out
    // action 1 here, three periods late, and nothing downstream could tell.
    CHECK(q.pop(3, sched.deadline(3), a, rec));
    CHECK(a[0] == 3.0f);
    CHECK(rec.index == 3);
    CHECK(q.consumer_stats().steps_skipped == 2);
}

// --- RTC's requirement (§4.5) ------------------------------------------------

static void test_rtc_exposes_the_in_process_cursor() {
    RuntimeConfig cfg = cfg_with(Stitching::Rtc);
    ActionChunkQueue q(cfg);
    ScheduleClock sched(now());
    Action a{};
    ActionRecord rec;

    CHECK(q.last_emitted_step() == -1);  // nothing promised yet

    q.publish(fill(q, 0, kChunkSize, 1.0f, 1));
    for (std::int64_t step = 0; step < 4; ++step) {
        q.pop(step, sched.deadline(step), a, rec);
    }

    // This is sufficient only for an in-process worker that retains its own last
    // generated chunk. The Python process boundary will carry the active chunk
    // identity and committed padded actions explicitly; a cursor alone is
    // ambiguous after newest-wins supersession.
    CHECK(q.last_emitted_step() == 3);

    // And the emitted values really are the worker's, unmodified — no blending
    // under Rtc, which is the assumption that removes the seqlock.
    CHECK(a[0] == 1.0f);
}

// --- construction -----------------------------------------------------------

static void test_rejects_a_chunk_longer_than_the_slot() {
    // Slots are kChunkSize, compiled in from the checkpoint. A config asking for
    // more is a rebuild, not a runtime allocation (invariant #3).
    RuntimeConfig cfg{};
    cfg.chunk_size = kChunkSize + 1;
    cfg.queue_capacity = 128;
    CHECK_THROWS(ActionChunkQueue{cfg});
}

static void test_pool_exhaustion_is_reported_not_blocked() {
    ActionChunkQueue q(cfg_with(Stitching::Discard));
    for (std::size_t i = 0; i < kChunkSlots; ++i) {
        CHECK(fill(q, 0, 10, 1.0f) != nullptr);
    }
    // Fifth acquire with nothing freed: nullptr, immediately. The worker skips
    // the inference; it does not spin waiting for the control thread.
    CHECK(q.acquire() == nullptr);
    CHECK(q.producer_stats().acquire_failed == 1);
}

int main() {
    test_publish_then_drain();
    test_underrun_on_empty_queue();
    test_refresh_trigger();
    test_continuous_refresh_starts_after_each_accept();
    test_rtc_preserves_configured_refresh_gap();
    test_discard_skips_the_stale_prefix();
    test_ensemble_blends_the_overlap();
    test_ensemble_releases_the_old_chunk();
    test_newest_chunk_wins();
    test_skipped_tick_is_not_emitted_late();
    test_rtc_exposes_the_in_process_cursor();
    test_rejects_a_chunk_longer_than_the_slot();
    test_pool_exhaustion_is_reported_not_blocked();

    if (g_failures == 0) {
        std::printf("test_chunk_queue: all checks passed\n");
    } else {
        std::printf("test_chunk_queue: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
