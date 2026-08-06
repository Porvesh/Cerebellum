// Two threads over the queue, for ThreadSanitizer (spec.md §4.1, §8).
//
// Build with -DCEREBELLUM_TSAN=ON. The claim under test is the one the design
// rests on: whole chunks cross the thread boundary through the two rings, so the
// handoff is release/acquire on a single index and the payload needs no locking.
// If that is wrong, TSAN says so here and the queue's whole shape is wrong.
//
// The correctness argument, written down because TSAN can only refute it:
//   producer writes slot -> ready_.push (release) -> consumer's ready_.pop
//   (acquire) -> consumer reads slot -> free_.push (release) -> producer's
//   free_.pop (acquire) -> producer reuses slot.
// Two ordered edges per slot per cycle, and no other thread ever touches it.
//
// Cadence is compressed (200 us periods, ~600 us "inference") so the test runs
// in under a second. The RATIOS are what matter — inference spans ~3 periods and
// R is 6, the same relationship as 90 ms against 33 ms.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "cerebellum/chunk_queue.hpp"
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

namespace {

constexpr auto kTestPeriod = std::chrono::microseconds(200);
constexpr int kTestSteps = 2000;

std::atomic<bool> g_stop{false};

// Every dim of every action in chunk k carries k. A chunk published before it
// was fully written shows up as an action whose dims disagree, or whose value
// disagrees with the stamps that arrived with it — the slot was last used by a
// different chunk, so the stale value is a different number.
void inference_worker(ActionChunkQueue& q, std::atomic<std::uint64_t>& published) {
    std::uint64_t chunk_id = 1;
    // Deterministic variance, no Math.random equivalent needed: 3 to 5 periods.
    int jitter = 0;

    while (!g_stop.load(std::memory_order_relaxed)) {
        if (!q.should_refresh()) {
            std::this_thread::sleep_for(kTestPeriod / 4);
            continue;
        }

        Chunk* c = q.acquire();
        if (!c) {
            std::this_thread::sleep_for(kTestPeriod / 4);
            continue;
        }

        // The observation is for the step after the last one promised. By the
        // time this lands the consumer has moved on, which is the stale prefix
        // discard has to skip (§4.3).
        const std::int64_t obs_step = q.last_emitted_step() + 1;
        const TimePoint capture = now();

        const float v = static_cast<float>(chunk_id);
        c->first_step = obs_step;
        c->count = kChunkSize;
        for (int i = 0; i < kChunkSize; ++i) {
            for (std::size_t d = 0; d < kPaddedActionDim; ++d) {
                c->actions[static_cast<std::size_t>(i)][d] = v;
            }
        }
        c->stamps.chunk_id = chunk_id;
        c->stamps.obs_seq = chunk_id;
        c->stamps.t_obs_capture = capture;
        c->stamps.t_infer_start = capture;

        // The forward pass, with variance — deliverable §11.3 is underruns
        // against R under exactly this, and a fake worker controls the
        // distribution instead of hoping the GPU produces an interesting one.
        jitter = (jitter + 1) % 3;
        std::this_thread::sleep_for(kTestPeriod * (3 + jitter));

        c->stamps.t_infer_end = now();
        if (q.publish(c)) {
            published.fetch_add(1, std::memory_order_relaxed);
            ++chunk_id;
        }
        // On failure the producer keeps the slot and retries with it — it must
        // never push to free_, which would put two writers on that ring.
    }
}

}  // namespace

int main() {
    RuntimeConfig cfg{};
    cfg.validate();

    ActionChunkQueue q(cfg, /*seam_capacity=*/kTestSteps);
    ControlMetrics metrics(kTestSteps);
    std::atomic<std::uint64_t> published{0};

    std::uint64_t torn = 0;
    std::uint64_t mismatched_stamps = 0;

    std::thread worker(inference_worker, std::ref(q), std::ref(published));

    // The control thread. Fixed cadence, absolute grid, never blocks on the
    // producer (invariant #1).
    ScheduleClock sched(now(), Nanos{kTestPeriod});
    Action a{};
    ActionRecord rec;

    for (int i = 0; i < kTestSteps; ++i) {
        const std::int64_t step = sched.index();
        sleep_until(sched.deadline(step));

        if (q.pop(step, sched.deadline(step), a, rec)) {
            // Internal consistency: all 32 dims came from one chunk.
            for (std::size_t d = 1; d < kPaddedActionDim; ++d) {
                if (a[d] != a[0]) {
                    ++torn;
                    break;
                }
            }
            // And the payload agrees with the stamps that travelled with it.
            if (static_cast<std::uint64_t>(a[0]) != rec.chunk.chunk_id) {
                ++mismatched_stamps;
            }
            metrics.on_emit(rec);
        } else {
            metrics.on_underrun();
        }
        sched.advance();
    }

    g_stop.store(true, std::memory_order_relaxed);
    worker.join();

    // The whole point of the file.
    CHECK(torn == 0);
    CHECK(mismatched_stamps == 0);

    // Sanity: the run actually exercised the handoff rather than idling.
    CHECK(published.load() > 10);
    CHECK(q.consumer_stats().chunks_accepted > 10);
    CHECK(q.consumer_stats().actions_discarded > 0);   // there were real seams
    CHECK(metrics.ticks == static_cast<std::uint64_t>(kTestSteps));

    // Not asserted: an underrun count of zero. Under injected variance the
    // honest output is the number, and the number is deliverable §11.3 — a test
    // that demanded zero would be tuning R by editing the assertion.
    std::printf("test_chunk_queue_mt: published=%llu accepted=%llu superseded=%llu\n",
                static_cast<unsigned long long>(published.load()),
                static_cast<unsigned long long>(q.consumer_stats().chunks_accepted),
                static_cast<unsigned long long>(q.consumer_stats().chunks_superseded));
    std::printf("test_chunk_queue_mt: underruns=%llu discarded=%llu skipped=%llu\n",
                static_cast<unsigned long long>(metrics.underruns),
                static_cast<unsigned long long>(q.consumer_stats().actions_discarded),
                static_cast<unsigned long long>(q.consumer_stats().steps_skipped));
    metrics.print();

    if (g_failures == 0) {
        std::printf("test_chunk_queue_mt: all checks passed\n");
    } else {
        std::printf("test_chunk_queue_mt: %d FAILURES\n", g_failures);
    }
    return g_failures;
}
