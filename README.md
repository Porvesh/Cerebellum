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
> no older than 350 ms.

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
worker. Each request carries an immutable observation snapshot containing robot state, task text,
and camera-native HWC bytes; Python converts the images to CHW floats for the runner. Its
versioned handshake rejects mismatched chunk and action dimensions before control starts. The
Python model side also loads `lerobot/smolvla_base` and returns both representations needed by
the system:

- `model_actions`: `50 × 32`, padded model space used for RTC conditioning.
- `robot_actions`: `50 × 6`, sliced and postprocessed for execution.

The base checkpoint smoke test proves loading, preprocessing, denoising, and output shapes; it
does not establish useful robot behavior without task-specific fine-tuning.

Build and exercise the C++ → Python → C++ synthetic round trip using the no-sudo environment:

```bash
cmake -S . -B build -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The bridge currently uses an in-memory observation source. The worker can run either the fast
synthetic backend or the real SmolVLA checkpoint; SmolVLA advertises its state and camera schema
during startup, and C++ rejects mismatched observations before sending inference requests.
RTC requests carry the retained chunk identity and aligned committed model-space prefix. The
SmolVLA worker applies gradient-guided inpainting during every denoising step; the control
thread still only sees postprocessed robot-space actions.

The real cross-process C++ → SmolVLA → C++ test is opt-in because it loads the checkpoint:

```bash
HF_HOME=/path/to/huggingface/cache \
CEREBELLUM_RUN_SMOLVLA_BRIDGE=1 \
CEREBELLUM_SMOLVLA_DEVICE=cpu \
./build/test_python_chunk_generator
```

Use `CEREBELLUM_SMOLVLA_DEVICE=cuda` when a GPU has enough free memory. Model startup and
per-request inference use separate configurable timeouts in `PythonChunkGeneratorOptions`.
Set `CEREBELLUM_RUN_SMOLVLA_RTC=1` to run the direct two-request RTC smoke test; it verifies
that guided denoising moves the new committed prefix closer to the retained chunk prefix.

Benchmark the same full C++ → Python → model → C++ path with realistic three-camera payloads:

```bash
# Cerebellum transport/runtime baseline without model cost
./build/bench_bridge --runner synthetic --device cpu --warmup 10 --iterations 100

# Real model split: total, model, and total-minus-model Cerebellum overhead
CUDA_VISIBLE_DEVICES=2 HF_HOME=/path/to/huggingface/cache \
  ./build/bench_bridge --runner smolvla --device cuda \
  --warmup 10 --iterations 100 --output results/smolvla_h100_baseline.json
```

The report gives p50/p90/p99/max from C++ observation lookup to a decoded chunk, the Python
model call, Cerebellum overhead, and each serialization/IPC/conversion stage. The headline
overhead is computed per request as `total - model`; `response_wait` intentionally contains
the Python stages and therefore must not be added to them again. Observation-capture-to-action
staleness remains a separate control-loop metric.

Benchmark the complete 30 Hz inference-worker → chunk-queue → action-sink runtime using
Discard stitching:

```bash
# Model-delay simulation using the measured 149 ms H100 latency
./build/bench_runtime --runner synthetic --device cpu --inference-ms 149 \
  --ticks 300 --refresh-trigger 6 --refresh-policy tail

# Freshness/power comparison: continuously generate replacement chunks
./build/bench_runtime --runner synthetic --device cpu --inference-ms 149 \
  --ticks 300 --refresh-trigger 6 --refresh-policy continuous

# Real SmolVLA runtime
CUDA_VISIBLE_DEVICES=2 HF_HOME=/path/to/huggingface/cache \
  ./build/bench_runtime --runner smolvla --device cuda --warmup-inferences 10 \
  --ticks 300 --refresh-trigger 6 --refresh-policy tail \
  --output results/runtime_discard_smolvla_h100_baseline.json

# RTC: start after s-d actions and inpaint actions committed during inference
./build/bench_runtime --runner synthetic --device cpu --inference-ms 149 \
  --ticks 300 --refresh-trigger 6 --stitching rtc --refresh-policy horizon

# Compare RTC guidance cost and committed-prefix agreement at 5/6/8/10 steps.
# CUDA_VISIBLE_DEVICES maps physical GPU 3 to logical cuda:0 in this process.
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python conda run -n cerebellum \
  python -m cerebellum_model.rtc_sweep --device cuda --physical-gpu 3 \
  --steps 5 6 8 10 --warmup 1 --iterations 10 \
  --output results/rtc_denoise_sweep_gpu3.json

# Run the winning setting through the complete C++ two-loop runtime.
CUDA_VISIBLE_DEVICES=3 ./build/bench_runtime --runner smolvla --device cuda \
  --stitching rtc --refresh-policy horizon --rtc-denoise-steps 5 \
  --rtc-inference-delay 8 --rtc-execution-horizon 8 \
  --warmup-inferences 2 --ticks 300 --refresh-trigger 6
```

This report covers all action emissions, including fallbacks, and separately reports action
lateness, observation-to-action staleness, chunk-ready-to-action queue age, underruns, skipped
steps, discarded actions, seam size, and inference/queue counters.
`inference_load.worker_busy_percent` is the fraction of runtime spent inside
`ChunkGenerator::generate()`; it is a comparable workload proxy, not measured GPU wattage.

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
