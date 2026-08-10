# Cerebellum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A real-time inference runtime for vision-language-action policies.**

Serves a VLA policy at a fixed control cadence on a single GPU, and proves it with
tail-latency measurements rather than throughput numbers.

A robot controller needs a new action every 33 ms. A VLA forward pass takes 90–150 ms on
an H100 at batch size 1. That deficit isn't closed by making the forward faster — it's
closed by action chunking, asynchronous inference, and a queue between the slow irregular
producer and the fast fixed-rate consumer. Cerebellum is that queue plus the machinery
around it.

## The invariant

> An action is available at the actuator every 33 ms, computed from an observation
> no older than 200 ms.

Two numbers, two distinct failure modes: miss the cadence and the controller sees jitter
(a *latency* failure); blow the staleness bound and actions arrive smoothly but describe a
world that has moved (a *staleness* failure). They are traded against each other on
purpose and never collapsed into one word.

## Where this sits

Third of three, named for the sensorimotor pathway:

| Project | Biological role | What it does |
|---|---|---|
| [**Retina**](https://github.com/Porvesh/Retina) | Transduces light, sends it down the optic nerve | Frame capture, lossy transport, multi-camera alignment |
| [**Axon**](https://github.com/Porvesh/Axon) | Carries signals between cells | Zero-copy shared-memory ring, `io_uring` persistence |
| **Cerebellum** | Timing and smooth motor coordination | Turns observations into actions on a deadline |

The cerebellum handles *timing* — it doesn't decide what to do, it makes movement smooth
and correctly phased, using feed-forward prediction rather than waiting for feedback. This
component doesn't train the policy, it schedules it. The VLA is a black box; the value is
entirely in the layer wrapped around it.

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
```

Two threads, two rates. The queue is the only thing between them, and **the control thread
never blocks on the GPU.**

## Status

The standalone phase-1 skeleton is implemented: configuration validation, fixed-grid timing,
a lock-free action-chunk queue, and the two-thread control/inference runtime. A persistent,
framed local-process bridge now connects C++ `ChunkGenerator::generate()` to a synthetic Python
worker. Its versioned handshake rejects mismatched chunk and action dimensions before control
starts. The Python model side also loads `lerobot/smolvla_base` and returns both representations
needed by the system:

- `model_actions`: `50 × 32`, padded model space retained for future RTC conditioning.
- `robot_actions`: `50 × 6`, sliced and postprocessed for execution.

The base checkpoint smoke test proves loading, preprocessing, denoising, and output shapes; it
does not establish useful robot behavior without task-specific fine-tuning.

Build and exercise the C++ → Python → C++ synthetic round trip using the no-sudo environment:

```bash
cmake -S . -B build -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The bridge currently creates a synthetic observation inside Python. Sending live camera/state
observations and selecting the real SmolVLA runner are the next integration step. RTC requests
are rejected explicitly until committed-prefix conditioning is implemented.

Run the dependency-light contract tests:

```bash
PYTHONPATH=python conda run -n cerebellum python -m pytest -q
```

Run one real checkpoint forward from the local Hugging Face cache:

```bash
PYTHONPATH=python conda run -n cerebellum \
  python -m cerebellum_model.smoke --device cuda --local-files-only
```

Use `--device cpu` when no GPU has enough free memory. Install the pinned optional dependencies
without sudo using `python/requirements-smolvla.txt`.

See [`spec.md`](spec.md) for the problem statement, latency budget, invariants, measurement
methodology, and phased plan. Performance numbers in the spec remain targets until the benchmark
tables and sweeps are committed.

Phase 1 is the runtime standalone on synthetic input; phase 2 wires in Retina and Axon.
The deliverables, in priority order, are a per-stage p50/p99 latency table, a chunk-size
vs. staleness/jitter sweep, and underrun count vs. refresh trigger.

## License

[MIT](LICENSE)
