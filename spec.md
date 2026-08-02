# Project Spec — `cerebellum`
### A real-time inference runtime for vision-language-action policies

Serves a VLA policy at a fixed control cadence on a single GPU, and proves it with
tail-latency measurements rather than throughput numbers.

---

## 0. Why this name

The existing projects are named for the sensorimotor pathway:

| Project | Biological role | What it does |
|---|---|---|
| **Retina** | Transduces light, sends it down the optic nerve | Frame capture, lossy transport, multi-camera alignment |
| **Axon** | Carries signals between cells | Zero-copy shared-memory ring, `io_uring` persistence |
| **Cerebellum** | Timing and smooth motor coordination | Turns observations into actions on a deadline |

The cerebellum is the structure that handles *timing* — it doesn't decide what to do, it
makes movement smooth and correctly phased, and it does so with feed-forward prediction
rather than waiting for feedback. That is exactly what this component does: it doesn't
train the policy, it schedules it.

---

## 1. Problem statement

A robot controller needs a new action at a fixed cadence — say every 33 ms for 30 Hz.
A vision-language-action policy takes roughly 90–150 ms per forward pass on an H100 at
batch size 1.

That is a 3–5x deficit, and it cannot be closed by making the forward pass faster. It is
closed by changing the shape of the problem:

- The policy emits **many future actions per forward** (action chunking).
- Inference runs **asynchronously**, overlapping with execution of the previous chunk.
- A **queue** decouples the slow, irregular producer from the fast, fixed-rate consumer.

Cerebellum is that queue plus the machinery around it.

### The invariant

Everything in this document exists to satisfy one statement:

> An action is available at the actuator every 33 ms, computed from an observation
> no older than 200 ms.

Two numbers, two distinct failure modes:

- **Miss the cadence** → the controller sees jitter, which can destabilise it. This is a
  *latency* failure.
- **Blow the staleness bound** → actions arrive smoothly but describe a world that has
  moved. This is a *staleness* failure.

Latency and staleness are different quantities and are traded against each other on
purpose. Chunking drives latency to near zero (pop from a queue) and drives staleness up
(acting on a prediction made up to a second ago). Never collapse the two into one word.

---

## 2. Scope

### In scope
- Serving-side scheduling, buffering, and cadence control
- GPU execution optimisation: CUDA graphs, pinned memory, stream discipline
- Latency and staleness measurement methodology
- Integration with Retina (transport) and Axon (IPC + logging)

### Explicitly out of scope
- Training, fine-tuning, or any policy learning
- Model architecture changes
- Writing custom CUDA kernels
- Task-success evaluation (accuracy is only measured as a regression check on quantisation)
- Real robot hardware

If you find yourself reading a policy-learning paper, you have drifted.

---

## 3. Layer ownership

```
┌────────────────────────────────────────────────────────────┐
│  Data collection and training              NOT OURS        │
│  skipped entirely                                          │
├────────────────────────────────────────────────────────────┤
│  Model weights and architecture            NOT OURS        │
│  downloaded, frozen, called as a function                  │
├────────────────────────────────────────────────────────────┤
│  SERVING RUNTIME                           ← CEREBELLUM    │
│  cadence, chunk queue, stitching, CUDA graphs              │
├────────────────────────────────────────────────────────────┤
│  TRANSPORT / IPC / LOGGING                 ← RETINA, AXON  │
│  lossy UDP, jitter buffer, shm ring, io_uring writer       │
├────────────────────────────────────────────────────────────┤
│  CUDA runtime and driver                   NOT OURS        │
│  configured, not written                                   │
├────────────────────────────────────────────────────────────┤
│  Robot and actuator                        EMULATED        │
│  a fixed-cadence consumer that never blocks                │
└────────────────────────────────────────────────────────────┘
```

The VLA is a black box. The value of this project is entirely in the layer wrapped
around it.

---

## 4. Architecture

### 4.1 Steady-state dataflow

```
   observation                                  actions
        │                                          ▲
        ▼                                          │
  ┌───────────────┐                        ┌───────────────┐
  │  ObsSource    │                        │  Control      │
  │  latest()     │                        │  thread       │
  │  non-blocking │                        │  33 ms tick   │
  └───────┬───────┘                        └───────▲───────┘
          │                                        │ pop
          ▼                                        │
  ┌───────────────┐        push chunk      ┌───────┴───────┐
  │  Inference    │──────────────────────▶ │  Action chunk │
  │  worker       │                        │  queue        │
  │  ~90 ms/pass  │                        │  ~50 actions  │
  └───────────────┘                        └───────────────┘
          │                                        │
          └──────────── ActionSink ◀───────────────┘
                    (log obs_seq, timings, chunk id)
```

Two threads, two rates. The queue is the only thing between them.

**The control thread never blocks on the GPU.** This is invariant #1 and the entire
architecture follows from it.

### 4.2 The queue is a jitter buffer

The action chunk queue is structurally identical to Retina's frame jitter buffer:

| | Retina jitter buffer | Cerebellum chunk queue |
|---|---|---|
| Producer | network, bursty and lossy | GPU, slow and variable |
| Consumer | display, fixed rate | actuator, fixed rate |
| Absorbs | arrival jitter | inference-time variance |
| Costs | added latency | added staleness |
| Fails by | underrun → dropped frame | underrun → missed deadline |

Same shape, different payload. Reuse the intuition.

### 4.3 Chunking and async inference

**Naive synchronous** — the robot stalls at every chunk boundary:

```
infer[k]───▶│ exec a0 … a49 ││ infer[k+1]───▶│ exec b0 … b49 │
                             ^^^^^^^^^^^^^^^
                             GPU busy, queue empty, robot stalls ~90 ms
```

**Async with overlap** — next chunk computed while current one drains:

```
infer[k]───▶│ exec a0 a1 … a44 a45 a46 a47 a48 a49 │
                        infer[k+1]───▶│ exec b0 b1 …
                                      ▲
                        b0..b3 were predicted for timesteps
                        already executed from chunk k
```

The overlap region is the interesting part. Chunk `k+1` was computed from an observation
at time *t* but lands at *t + 90 ms*, by which point 2–3 more actions of chunk `k` have
already gone out. Two options:

- **Discard the stale prefix.** Simple. Produces a discontinuity at the seam — the
  actuator jerks.
- **Temporal ensembling.** Average the overlapping predictions from both chunks. Smooth,
  but you are blending two opinions formed from different observations.

Start with discard, measure the seam discontinuity, then implement ensembling and show
the improvement. The before/after is a good result on its own.

### 4.4 Refresh trigger

Fire inference for chunk `k+1` when the queue has `R` actions remaining.

- `R` too small → inference doesn't finish in time → underrun → missed deadline
- `R` too large → you re-infer while plenty of valid actions remain → wasted GPU, and
  more seams than necessary

`R` must cover the p99 inference time, not the mean:

```
R ≥ ceil(p99_inference_ms / control_period_ms)

worked:  p99 = 118 ms, period = 33 ms  →  R ≥ 4 actions of headroom
```

This is why the measurement work comes before the tuning work. You cannot pick `R`
without a trustworthy p99. (See §15.2 — in phase 2 the term to cover is the whole
observation→chunk-ready path, not the forward pass alone.)

---

## 5. Interfaces

Write these on day one, even though phase 1 only needs the fake implementations.
Retrofitting the abstraction after hardcoding tensors everywhere is miserable.

```python
class ObservationSource:
    def latest(self) -> Obs | None:
        """Non-blocking. Returns the NEWEST observation, or None if nothing new.
        Never returns a queued backlog — stale observations are worthless."""

class ActionSink:
    def publish(self, chunk: ActionChunk, meta: ChunkMeta) -> None:
        """Non-blocking. Must never stall the inference worker."""
```

| Phase | ObservationSource | ActionSink |
|---|---|---|
| 1 | `SyntheticSource` — random tensors at a fixed rate | `NullSink` — in-memory trace |
| 2 | `AxonRingSource` — zero-copy shm reader | `AxonLogSink` — `io_uring` columnar log |

The runtime between them does not change across phases. That is the point.

### 5.1 The Axon seam (phase 2)

The policy talks to **Axon only**. It never sees RTP, FEC, or jitter buffers — Retina's
job ends when it deposits an aligned frame set into the ring. If the policy knew about
the network, you could not test it without one.

**Do not copy across the language boundary.** Naive pybind11 returning `bytes` duplicates
the frame and deletes Axon's zero-copy claim.

```cpp
// returns a memoryview into the mapped shm region — no copy
py::memoryview read_latest(uint64_t* out_seq);
```

Then `np.frombuffer(view)` → `torch.from_numpy(...)` → `.to('cuda', non_blocking=True)`
into a pre-allocated pinned staging buffer. Exactly one copy, performed by the DMA engine.

**Torn reads are a real bug here.** Axon's ring overwrites. If the producer wraps while
you are reading, you get a frame that is half old and half new, with no error raised — the
policy silently sees garbage. Use seqlock discipline:

```
seq0 = read_seq()
if seq0 is odd:      retry      # writer is mid-update
copy into pinned buffer
seq1 = read_seq()
if seq1 != seq0:     retry      # clobbered mid-copy, discard
```

Cost is one wasted memcpy under contention. Do **not** solve this by reserving a slot the
writer must skip — that reintroduces exactly the producer/consumer coupling Axon exists
to avoid.

### 5.2 What gets logged

Not just actions — the causal chain, so replay can reconstruct what the policy saw:

```
obs_seq             which observation produced this chunk
t_obs_capture       when the frame was captured
t_infer_start       when the forward began
t_infer_end         when the chunk was ready
chunk_id, index     which action within which chunk
t_emit              when the control thread popped it
t_deadline          when it was SUPPOSED to be popped
```

One timebase: `CLOCK_MONOTONIC`, everywhere, no exceptions. Convert only at the edges.

---

## 6. Latency budget

Write the budget before writing code, then hold yourself to it. This is a target, not a
measurement.

```
capture: exposure + sensor readout            8 ms
encode + fragment                             3 ms
network transit (emulated, p99)              12 ms
jitter buffer depth                          20 ms   ← the free knob
multi-camera alignment wait                   5 ms
Axon ring handoff                           0.1 ms
preprocess + H2D                              4 ms
forward pass                                 92 ms   ← 63% of the total
postprocess + queue push                      2 ms
                                          -------
                                            146 ms   ← staleness at chunk start,
                                                       grows to 146 + chunk duration
```

The budget already tells you where the work is, before any profiling: the forward pass
dominates, and the jitter buffer is the only variable that costs nothing but depth.

### Why the forward pass is what it is

At batch size 1 you are **memory-bandwidth bound, not compute bound.** You read every
parameter once and do almost no arithmetic with it. For a π0-class 3B model:

```
3B params × 2 bytes (bf16) = 6 GB per forward
6 GB ÷ ~3 TB/s HBM         ≈ 2 ms theoretical floor
measured                     ≈ 45 ms backbone
```

Tensor core utilisation is on the order of a couple percent. Consequences:

- FP8 buys latency almost linearly — it halves the bytes read
- A bigger batch does not help; you have exactly one robot
- A bigger GPU does not help much either

SmolVLA (~450M) is ~7x smaller, so its floor and its measured forward are both far lower —
keep every quoted number tagged with the model it came from (§15.3).

### Kernel launch overhead

30 layers × ~10 kernels each = 300+ launches per forward. Each kernel runs 30–80 µs with
5–10 µs of launch gap, so 10–20% of wall time is the GPU sitting idle between kernels.
CUDA graphs capture the forward as one replayable graph — one launch total.

The flow-matching action head is the best target: 10 sequential tiny denoise steps is
pure launch overhead.

---

## 7. Design knobs

| Knob | Increasing it… |
|---|---|
| chunk size `H` | staleness ↑, inference pressure ↓, seams ↓ |
| refresh trigger `R` | underrun risk ↓, wasted compute ↑ |
| stitching policy | discard = responsive but jerky; ensemble = smooth but blended |
| queue overflow policy | block = correct for logs, wrong for control |
| precision (bf16 → fp8) | forward time ↓, accuracy risk ↑ |
| CUDA graphs on/off | launch overhead ↓, shape rigidity ↑ |
| jitter buffer depth | loss resilience ↑, staleness ↑ |

---

## 8. Invariants

Violations of these are what you count and report.

1. **The control thread never blocks on the GPU.** If it does, you have built a
   synchronous system with extra steps.
2. **The queue never underruns.** An underrun is a missed deadline is a broken robot.
3. **Zero allocation in the steady-state loop** — CPU or GPU. No `cudaMalloc`, no tensor
   creation, no Python object churn.
4. **Staleness at every emitted action stays under the bound.**
5. **The policy consumer reads newest-wins.** A 500 ms old frame is worthless; never
   drain a backlog.

---

## 9. Measurement methodology

This is the part that makes or breaks the project. A serving wrapper with no rigorous
latency study is a mediocre project. The measurement *is* the deliverable.

### Rules

- **Percentiles only.** Mean latency is meaningless for a deadline system. Report
  p50 / p99 / p99.9 / max. Store every sample; never accumulate a running mean.
- **Measure against the schedule, not against yourself.** This is coordinated omission
  and you will walk into it:

  ```
  WRONG:  start = now(); do_work(); record(now() - start)
  RIGHT:  deadline = t0 + i * period; do_work(); record(now() - deadline)
  ```

  Timing from "when I called it" systematically erases every iteration where you were
  already behind — precisely the tail you are hunting. Read Gil Tene on this.
- **Warmup discipline.** Discard the first N iterations (autotuning, JIT, allocator
  growth) and state N in the writeup.
- **Variance isolation.** One idle GPU with no other CUDA context. Pinned threads,
  `isolcpus`, no frequency scaling. Any other tenant contaminates p99, and p99 is the
  product.
- **CUDA events for GPU spans, `perf_counter_ns` for wall spans.** Wall-clock timers
  wrapped around an async launch measure nothing.
- **Stage breakdown needs explicit syncs**, and the stage sum will exceed the pipelined
  end-to-end time. Note that in the writeup — it is expected, not a bug.

### Tooling, in order of use

| Tool | When |
|---|---|
| Your own percentile harness | Always. Primary. Nothing else knows your deadline. |
| `torch.profiler` | First reach. Per-op CPU+CUDA breakdown, Chrome trace. |
| Nsight Systems + NVTX ranges | When you need to see the *gaps* — launch overhead, missing overlap, hidden syncs. |
| `perf stat -e context-switches,page-faults` | When p99 spikes look CPU-shaped. |
| eBPF / bpftrace | Phase 2 only — proving the `io_uring` writer's disk stalls don't perturb the control thread. |
| Nsight Compute | Probably never. You are not writing kernels. |

Install nsys now, use `torch.profiler` on day one, let the others arrive when a specific
number confuses you.

---

## 10. Phases

### Phase 0 — clear the box

The sglang server currently holds ~75 GB on all four H100s (preallocated KV pool, not
real usage). You need one completely idle GPU — not for capacity, for variance.

```bash
pkill -f sglang
# or confine it:
CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
  --model-path <model> --tp 2 --mem-fraction-static 0.5
```

### Phase 1 — the runtime, standalone

Synthetic input only. No Retina, no Axon. Independently demo-able.

1. **One forward, correct shapes.** Random image tensors + dummy instruction → an action
   chunk. Don't time anything. Getting shapes and action normalisation right is fiddlier
   than it sounds.
2. **The timing harness.** Build this second, not last. Schedule-based measurement,
   warmup discard, full sample retention, percentile output.
3. **Stage breakdown.** Instrument preprocess / H2D / vision / backbone / action head /
   D2H. Produce the baseline table.
4. **The two-thread loop.** Fixed-cadence executor + inference worker + chunk queue.
   Count underruns.
5. **Optimise.** Pinned memory, CUDA graphs, prefix KV caching across denoise steps.
   Re-measure after each. One change at a time.
6. **Sweep.** Chunk size vs. staleness and p99 jitter. Refresh trigger vs. underrun count
   under injected inference variance.

**Ships when:** you have the chunk-size sweep plot and a stable p99.

### Phase 2 — integration

A thin repo pulling Retina, Axon, and Cerebellum together as submodules. Mostly glue and
instrumentation, almost no new logic.

1. Swap `SyntheticSource` → `AxonRingSource`, `NullSink` → `AxonLogSink`
2. Wire Retina's output into the Axon ring
3. Frame-loss policy: when FEC can't recover, does the policy get a stale frame, an
   interpolated one, or does it skip the inference and coast?
4. End-to-end glass-to-action p99 under injected packet loss, decomposed by stage
5. Deterministic replay: feed logged observations back through the policy — are the
   actions bit-identical? Probably not (cuDNN algorithm selection, non-deterministic
   atomic reductions). Making replay bit-exact is a strong result on its own.
6. The payoff measurement: does the log writer's disk stall move the control thread's
   p99? If Axon's isolation claim holds, it does not.

**On honesty about the network:** on a single box the network hop is emulated loopback
with an impairment layer. Say so plainly, and make the impairment model principled —
measured loss and jitter distributions, not uniform random drops. Framed as "emulated WAN
with a configurable loss model," it holds up fine.

---

## 11. Deliverables

Three artifacts, in priority order:

1. **Stage latency table** — p50/p99 per stage, batch=1, idle H100, N=1000 after 50 warmup
2. **Chunk-size sweep** — staleness and p99 control jitter as functions of `H`. Nobody
   publishes this curve.
3. **Underrun count vs. refresh trigger** under injected inference-time variance
4. *(phase 2)* **Glass-to-action p99 vs. packet loss rate**, stacked by stage

Plus the nsys before/after screenshot around CUDA graph capture — 300 scattered launches
with visible gaps, then one dense block. That one image explains the whole optimisation.

---

## 12. Repo layout

```
cerebellum/
├── README.md              the three plots, then everything else
├── cerebellum/
│   ├── config.py          CONTROL_HZ, MAX_STALENESS_MS, chunk size, refresh trigger
│   ├── sources.py         ObservationSource implementations
│   ├── sinks.py           ActionSink implementations
│   ├── runner.py          model load, single forward, CUDA graph capture
│   ├── queue.py           action chunk queue + stitching policies
│   ├── loop.py            control thread + inference worker
│   └── timing.py          percentile recorder, CUDA event wrapper, schedule clock
├── bench/
│   ├── stage_breakdown.py
│   ├── chunk_sweep.py
│   └── underrun.py
└── results/               committed CSVs + plots
```

---

## 13. Environment

```bash
conda create -n cerebellum python=3.11 && conda activate cerebellum
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install lerobot          # SmolVLA lives here
```

Driver 580.x handles cu128 wheels. Confirm `nsys` is present, install it if not.

**Model:** start with **SmolVLA** (~450M, ~2 GB, clean chunking interface, iterates in
seconds). Move to π0 / openpi once the harness is solid. Do not start with a 3B model.

**Data:** none needed. Random tensors of the correct shape and dtype are *better* for
latency work — deterministic, and they isolate the thing being measured. Real data only
matters if you check whether quantisation broke accuracy.

**Do not install vLLM, sglang, or TensorRT-LLM for this.** You are writing the serving
loop. Handing that layer to a framework deletes the part of the project that is yours.

---

## 14. Day one

Put these in `config.py` before anything else:

```python
CONTROL_HZ = 30
MAX_STALENESS_MS = 200
```

Every optimisation gets judged against those two numbers. Without them the project drifts
into "make it faster," which has no stopping condition and produces a benchmark instead
of a system.

---

## 15. Open tensions phase 1 must settle

Four places where the numbers above don't yet close. None of them change the
architecture; all of them change what a passing run means, so they get resolved with
measurements rather than by editing the invariant.

### 15.1 The staleness bound and the chunk length are in conflict

Staleness of the *i*-th action of a chunk is `budget + i × period`. With the §6 budget and
the §1 bound:

```
staleness(i) = 146 ms + i × 33.3 ms  ≤  200 ms   →   i ≤ 1
```

Two actions per chunk. A 50-action chunk drained at 30 Hz reaches ~1.8 s of staleness at
its tail, 9x over the bound — so `H = 50` and `MAX_STALENESS_MS = 200` cannot both hold
under a 146 ms budget. Exactly one of these is true, and phase 1 has to say which:

- the bound applies **at chunk start** (a much weaker claim, and then invariant #4 needs
  rewording), or
- the bound is per-action and the real budget is a *much* smaller number than 146 ms
  (which means phase 2's network path is out of the question), or
- `MAX_STALENESS_MS` is simply wrong for chunked control and belongs nearer 1000 ms, in
  which case say what 200 ms was protecting against.

The chunk-size sweep (§11.2) is the experiment that settles it — it is measuring precisely
this trade. Until it runs, treat `MAX_STALENESS_MS` as the number under test rather than a
number already met.

### 15.2 `R` covers more than the forward pass

The trigger has to cover everything between "decide to re-infer" and "chunk is in the
queue", which in phase 2 includes capture, transport, jitter buffer and alignment — ~54 ms
of the §6 budget before the forward even starts:

```
R ≥ ceil(p99_obs_to_chunk_ready_ms / period)     not     ceil(p99_forward_ms / period)
```

With the §6 numbers that is `ceil(146 / 33.3) = 5`, not 4. In phase 1 the two are the same
thing (the synthetic source is instantaneous), which is exactly why this needs writing
down now rather than being discovered as an underrun spike in phase 2.

### 15.3 The bandwidth argument is for a model we're not starting with

§6's `3B params × 2 bytes = 6 GB` is π0-scale. SmolVLA is ~450M, so ~0.9 GB per forward
and a ~0.3 ms bandwidth floor. The *argument* (batch-1 is bandwidth bound, FP8 buys
latency near-linearly, a bigger GPU doesn't help) survives the substitution; the numbers
don't. Every measured number in `results/` carries the model it came from.

### 15.4 The underrun fallback is unspecified

Invariant #2 says the queue never underruns, and §11.3 counts underruns — so the runtime
needs a defined behaviour for the tick where it happens. Hold the last action, emit zeros,
or extrapolate? Holding is the obvious first choice (it is the only one of the three that
can't inject a discontinuity of its own), but pick it deliberately and count the event
either way: an underrun that gets papered over is a missed deadline you never hear about.
