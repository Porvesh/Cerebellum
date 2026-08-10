# Cerebellum — Architecture

A serving runtime that turns a slow, irregular VLA policy into a fixed-cadence action
stream. Companion to `spec.md`: the spec argues *why*, this draws *what*.

---

## 1. The problem

```
   control loop needs an action every ......  33 ms   (30 Hz)
   one VLA forward pass takes .............  90–150 ms
                                             ─────────
                                             3–5x deficit
```

Not closable by making the forward faster. Closed by changing the shape:

```
   ONE forward  ──▶  a CHUNK of 50 future actions  ──▶  drains over ~1.7 s

   forward passes per second required:  30  ──▶  ~0.6
   the cost is paid in STALENESS, not latency
```

Two failure modes, traded against each other on purpose, never collapsed into one word:

```
   LATENCY failure     action is late or missing     → controller jitters, robot destabilises
   STALENESS failure   action is on time but old     → robot acts on a world that has moved
```

---

## 2. Layer ownership

```
   ┌──────────────────────────────────────────────────────────┐
   │  training / data collection                  NOT OURS    │
   ├──────────────────────────────────────────────────────────┤
   │  model weights + architecture                NOT OURS    │
   │  frozen, called as a function                            │
   ├──────────────────────────────────────────────────────────┤
   │  SERVING RUNTIME                          ← CEREBELLUM   │
   │  cadence · chunk queue · stitching · CUDA graphs         │
   ├──────────────────────────────────────────────────────────┤
   │  transport / IPC / logging                ← RETINA, AXON │
   ├──────────────────────────────────────────────────────────┤
   │  CUDA runtime + driver                       NOT OURS    │
   ├──────────────────────────────────────────────────────────┤
   │  robot / actuator                            EMULATED    │
   └──────────────────────────────────────────────────────────┘
```

The VLA is a black box. All the value is in the layer wrapped around it.

---

## 3. Steady-state dataflow

```
                        ┌────────────────────────┐
                        │   ObservationSource    │
                        │   newest-wins          │
                        │   never a backlog      │
                        └───────────┬────────────┘
                                    │ observation
                                    ▼
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  INFERENCE WORKER            irregular · ~0.6 Hz · owns the GPU          ║
   ║                                                                          ║
   ║    wait for refresh trigger → take a free slot → read newest observation  ║
   ║    → forward pass (~90 ms) → write chunk into the slot → hand it over     ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                    │                                       ▲
        chunk ready │                                       │ slot recycled
                    ▼                                       │
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                        ACTION CHUNK QUEUE                                │
   │                                                                          │
   │        ═══════════ ready ═══════════▶     chunks handed forward          │
   │        ◀══════════ free ════════════      slots handed back              │
   │                                                                          │
   │        ── remaining ───────────────▶      "how much runway is left"      │
   │        ── committed cursor ────────▶      "how far execution has got"    │
   └──────────────────────────────────────────────────────────────────────────┘
                    │                                       ▲
         one action │                                       │
         per tick   ▼                                       │
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  CONTROL THREAD              fixed 30 Hz · NEVER blocks on the GPU       ║
   ║                                                                          ║
   ║    sleep to absolute deadline → pop the action for this step             ║
   ║    → on miss, apply the underrun fallback → send to actuator             ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │      ActionSink        │
                        │   non-blocking log     │
                        └────────────────────────┘
```

**The control thread never blocks on the GPU.** Everything else follows from that one rule.

**The queue is a jitter buffer.** Same shape as a video jitter buffer: bursty producer,
fixed-rate consumer, absorbs producer variance, pays for it in added delay, fails by
underrun.

---

## 4. Two threads, and what each one owns

The queue is the only thing between them. Ownership is what removes the need for locks.

```
   ════════════════════════════════════════════════════════════════════════════════
     CONTROL THREAD                              INFERENCE WORKER
     fixed rate · never blocks                   irregular · may block on GPU
   ════════════════════════════════════════════════════════════════════════════════

     OWNS OUTRIGHT                               OWNS OUTRIGHT
     ┌───────────────────────────┐               ┌───────────────────────────┐
     │ the current chunk         │               │ the chunk being built     │
     │ the previous chunk        │               │ its CUDA stream           │
     │ the action cursor         │               │ its own timing tables     │
     │ its timing recorders      │               │                           │
     └───────────────────────────┘               └───────────────────────────┘
                    │                                          │
                    │      ═══ THE ONLY SHARED STATE ═══       │
                    ▼                                          ▼
     ┌────────────────────────────────────────────────────────────────────────┐
     │   a small fixed POOL of chunk slots                                     │
     │       whole slots change hands; contents are never touched by two       │
     │       threads at once                                                   │
     │                                                                         │
     │   two one-way rings   ready (worker → control), free (control → worker) │
     │       they carry slot IDs, never action data                            │
     │                                                                         │
     │   two counters        runway remaining, committed cursor                │
     └────────────────────────────────────────────────────────────────────────┘
```

Because whole chunks cross the boundary, emitting an action is a local array index — not a
concurrent operation. Stitching likewise happens on data no other thread can see.

---

## 5. The slot lifecycle

A slot is in exactly one place at any instant. That is the whole safety argument.

```
                    ┌──────────────────────────────┐
       ┌───────────▶│   FREE POOL                  │◀────────────┐
       │            │   nobody is looking at this  │             │
       │            └──────────────┬───────────────┘             │
       │                           │  worker takes a slot        │
       │                           ▼                             │
       │            ┌──────────────────────────────┐             │
       │            │   BEING WRITTEN              │             │
       │            │   worker · ~90 ms of GPU     │             │
       │            └──────────────┬───────────────┘             │
       │                           │  handed over                │
       │                           ▼                             │
       │            ┌──────────────────────────────┐             │
       │ superseded │   IN FLIGHT                  │             │
       ├────────────┤   ready, not yet taken       │             │
       │ (newest    └──────────────┬───────────────┘             │
       │  wins)                    │  control thread accepts     │
       │                           ▼                             │
       │            ┌──────────────────────────────┐             │
       │            │   CURRENT                    │             │
       │            │   draining, one per tick     │             │
       │            └──────────────┬───────────────┘             │
       │                           │                             │
       │        discard / RTC ─────┤───── ensemble               │
       │        freed at once      │        │                    │
       │                           │        ▼                    │
       │                           │   ┌────────────────────┐    │
       │                           │   │   PREVIOUS         │    │
       │                           │   │   kept only for    │    │
       │                           │   │   the overlap      │    │
       │                           │   └─────────┬──────────┘    │
       └───────────────────────────┴─────────────┴───────────────┘
                            slot returned to the free pool
```

**Pool sizing.** In the worst legal steady state the consumer holds two (current +
previous), the worker holds one under construction, and one is in flight — four. The pool
is sized to exactly that, and stays honest because the refresh trigger refuses to start a
second inference while one is already waiting.

**Newest-wins.** If two chunks are waiting, the older one describes a world that has
already moved. Take the newest, free the rest, count what was dropped.

---

## 6. Timing — overlap and the refresh trigger

Naive synchronous execution stalls the robot at every chunk boundary:

```
   infer ───▶│ execute a0 … a49 ││ infer ───▶│ execute b0 …
                                  ^^^^^^^^^^
                                  GPU busy, queue empty, robot stalls ~90 ms
```

Async overlap: the next chunk is computed while the current one is still draining.

```
   tick   0        10        20        30        40    45   50
          ├─────────┼─────────┼─────────┼─────────┼─────┼────┤

   chunk k ████████████████████████████████████████████░░░░░░
                                                   ▲
                                                   │  runway drops to the
                                                   │  refresh trigger
                                                   ▼
   worker                                        ┌── infer k+1 ──┐
                                                 │    ~90 ms     │
                                                 └───────┬───────┘
                                                         │ handed over
                                                         ▼
   chunk k+1                                       ░░░░░░████████████
                                                   ▲    ▲
                                                   │    └─ first action actually used
                                                   └─ predicted for ticks that already
                                                      executed — the OVERLAP
```

**Where the trigger sits.** Fire when the remaining runway drops to `R` actions.

```
   R too small  →  inference doesn't land in time  →  underrun  →  missed deadline
   R too large  →  re-infer while good actions remain  →  wasted GPU, extra seams

   R must cover the p99 of the whole observation→chunk-ready path, not the mean,
   and not the forward pass alone.
```

This is why the measurement work comes before the tuning work: you cannot pick `R` without
a trustworthy p99.

**Second condition.** The trigger also requires that nothing is already waiting to be
accepted — otherwise it re-fires every tick for a chunk that is already computed.

---

## 7. The seam — three stitching policies

The overlap region is the interesting part. A new chunk lands a few ticks after the
observation it was computed from, so its first actions were predicted for ticks that have
already executed.

```
                          the join
                             │
      chunk k    … a45  a46  a47 ┊
      chunk k+1        b0   b1   b2   b3   b4 …
                       └────┴────┘
                        already executed
```

```
   ┌────────────┬──────────────────────────┬──────────────────────────────────────┐
   │ DISCARD    │  … a47 │ b3  b4  b5 …     │  simple. Discontinuity at the join — │
   │            │        ▲                  │  the actuator jerks.                 │
   │            │        └ the seam         │                                      │
   ├────────────┼──────────────────────────┼──────────────────────────────────────┤
   │ ENSEMBLE   │  … a47 │ avg  avg │ b5 …  │  smooth, but every blended action is │
   │            │  average the overlapping  │  two opinions formed from DIFFERENT  │
   │            │  predictions of both      │  observations. It inherits the age   │
   │            │  chunks                   │  of the older one.                   │
   ├────────────┼──────────────────────────┼──────────────────────────────────────┤
   │ RTC        │  … a47 │ b3  b4  b5 …     │  no seam to reconcile. b3 was        │
   │            │  (looks like discard —    │  GENERATED already agreeing with     │
   │            │   because there is        │  a47. The cost moves out of the      │
   │            │   nothing to fix up)      │  queue and into the denoise loop.    │
   └────────────┴──────────────────────────┴──────────────────────────────────────┘
```

All three live behind one policy switch, are measured by the same harness, on the same
clock. That equal footing is the reason RTC is written here rather than imported — a
three-way comparison built from three codebases measures implementations, not policies.

---

## 8. Real-time chunking — the feedback arrow

Discard and ensemble both operate on a chunk that already exists. RTC attacks the problem
one layer down: it treats the overlap as **inpainting inside the denoise loop**. Actions
guaranteed to execute are frozen, and the rest are generated conditioned on them.

This adds the one arrow the other two policies don't need: **execution state flowing back
to the worker.**

```
   CONTROL THREAD                              INFERENCE WORKER
   ──────────────                              ────────────────
   emits action for step s
          │
          │  committed cursor
          └────────────────────────────▶   "execution has reached s"
                                                    │
                                                    ▼
                                           the committed prefix:
                                           actions already promised for
                                           the next few steps
                                                    │
                                                    ▼
                                           ┌─────────────────────────────┐
                                           │  DENOISE LOOP               │
                                           │                             │
                                           │  prefix ── FROZEN ──┐       │
                                           │  ramp   ── bent toward       │
                                           │  tail   ── free              │
                                           └─────────────────────────────┘
                                                    │
                                                    ▼
                                           a chunk that agrees at the join
                                           by construction
```

Masking across the horizon:

```
   step:  0 ─────────── d ─────────── h ──────────────── H

          │◀ inference ▶│◀ soft ramp ▶│◀───── free ─────▶│
          │   delay     │              │
          hard: already   bend toward    generate freely
          promised, do    these, but
          not move        not at any cost
```

The band between the two boundaries is where the algorithm actually lives — outside it,
RTC is either a hard constraint or absent.

**Two things this demands of the runtime, both architectural:**

1. **The queue must expose what has been committed** — which is why it is indexed by
   absolute control step. "What did I promise for steps *s*…*s+n*" becomes a slice rather
   than a bookkeeping exercise.
2. **Inference delay is measured, not assumed.** The timing harness stops being a reporting
   tool and becomes an input to the algorithm.

---

## 9. Where the padded action space comes in

The policy denoises in a wide padded space and slices down to the robot's real degrees of
freedom only at the end. That slice point is an architectural decision, not a detail:

```
                 model's padded space (wide)              robot's real DOF (narrow)
                 ─────────────────────────────            ─────────────────────────

   worker    ────▶ generates here ────────────────────▶ postprocesses here
   queue     ────▶ stores model copy                   stores command copy ───▶
   stitching                                             blends commands only
   RTC prefix ───▶ must be handed back here
   actuator  ────────────────────────────────────────────────────────────────▶
```

Each chunk therefore holds two representations: the padded normalized model trajectory for
RTC and the narrow postprocessed robot commands for execution. A prefix handed to RTC in the
narrow space would leave the padded dimensions implicitly zero, while commanding the model
copy would bypass postprocessing. Either error is silent, so the types and bridge protocol keep
the two arrays separate.

---

## 10. Instrumentation — the causal chain

Timing is not bolted on; it is the deliverable. Every emitted action carries the chain back
to the frame that caused it.

```
   observation      forward        chunk        deadline      actually
   captured         begins         ready        for this tick emitted
        │              │              │              │            │
   ─────●──────────────●──────────────●──────────────●────────────●────▶
        │                             │              │            │
        │◀──── what the trigger ─────▶│              │◀ lateness ▶│
        │      must cover                            │   (signed)
        │
        │◀──────────────────── staleness ─────────────────────────▶│
                          bounded, per the invariant
```

One timebase everywhere, converted only at the edges. What gets recorded:

```
   lateness      did the action arrive on its tick
   staleness     how old was the world it describes
   seam          how far the actuator jumped at the join   ← the stitching comparison
   stage times   where the forward pass actually goes
```

---

## 11. Invariants the architecture exists to satisfy

```
   1  the control thread never blocks on the GPU
      — otherwise this is a synchronous system with extra steps

   2  the queue never underruns
      — an underrun is a missed deadline is a broken robot

   3  zero allocation in the steady-state loop, CPU or GPU
      — chunk sizes are fixed up front; a config asking for more is rejected,
        not allocated

   4  staleness at every emitted action stays under the bound

   5  the observation consumer reads newest-wins
      — a 500 ms old frame is worthless; never drain a backlog
```

Violations are counted and reported rather than hidden: underruns, skipped ticks,
superseded chunks, discarded overlap, and inferences dropped because the pool was dry.
