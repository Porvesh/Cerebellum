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

The inference worker can run either the fast synthetic backend or the real SmolVLA checkpoint;
SmolVLA advertises its state and camera schema
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

### LIBERO simulation

`PythonSimulatorAdapter` is both the runtime's `ActionSink` and `ObservationSource`. The 30 Hz
control thread only writes a preallocated atomic newest-action mailbox. A third, non-real-time
worker owns Python and MuJoCo I/O, applies the newest command, and atomically publishes the next
immutable 8D-state/two-camera observation for inference. This closes the same control → robot →
observation loop a physical robot adapter will eventually implement.

Install LIBERO into the existing project environment (no separate Conda environment is needed):

```bash
python -m pip install -e '.[libero]'
```

The real simulator test is opt-in. On a machine where EGL render-device access works, omit the
OSMesa variable. This workstation instead uses a user-local extracted `libOSMesa.so` because no
`sudo` or `/dev/dri/render*` access is available:

```bash
CEREBELLUM_RUN_LIBERO_SIMULATOR=1 \
CEREBELLUM_OSMESA_LIBRARY_PATH="$HOME/.local/lib/cerebellum-osmesa/usr/lib/x86_64-linux-gnu" \
./build/test_python_simulator
```

The test starts a persistent LIBERO `libero_spatial` task at 30 Hz, resets from its recorded init
state, transports both 256×256 cameras plus the flattened state into C++, applies one 7D no-op
command, and verifies the next observation. The default test backend is synthetic and requires
neither MuJoCo nor a GPU.

Run the LIBERO-trained SmolVLA checkpoint through the complete simulator loop:

```bash
CUDA_VISIBLE_DEVICES=3 \
HF_HOME="$HOME/.cache/cerebellum-hf" \
CEREBELLUM_OSMESA_LIBRARY_PATH="$HOME/.local/lib/cerebellum-osmesa/usr/lib/x86_64-linux-gnu" \
./build/run_libero --ticks 300 --warmup-inferences 1 \
  --model HuggingFaceVLA/smolvla_libero
```

The executable validates the checkpoint's 8D state, two-camera, and 7D action schema before
control starts. It clamps commands to LIBERO's `[-1, 1]` action space and reports model chunks,
fallbacks, simulator steps, safety decisions, staleness, termination, and task success.
The rollout defaults to the official `lerobot/libero` dataset's native 10 Hz action rate. A
future 30 Hz deployment must explicitly resample these delta actions; consuming them three times
as fast changes their physical meaning and is not a valid evaluation.
`RuntimeConfig::control_period` is the single runtime clock source: refresh floors, chunk duration,
RTC delay/horizon defaults, safety timing, and measured inference-delay steps derive from it.

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

## Action safety

`ActionSafetyFilter` is an optional, allocation-free boundary between the chunk queue and the
`ActionSink`. It rejects non-finite or over-age model actions, clamps command bounds, and limits
per-dimension action rate and acceleration. Rejected actions hold the last safe command (or a
configured replacement before the first accepted command). The runtime remembers only the
filtered result, so an unsafe prediction cannot later reappear through `HoldLast` or extrapolation.

Physical values are deliberately not built into Cerebellum. A robot adapter constructs
`SafetyConfig` with its command bounds, rate limits, acceleration limits, maximum observation age,
and replacement action. For an absolute joint-position policy, action bounds are joint limits and
action rate is joint velocity; an adapter for another command space maps its own units instead.
Before starting control, the adapter should call `reset()` with the measured starting pose or
command and pass the filter to `RuntimeLoop`. Hardware collision protection, torque limits, and
the emergency stop remain the responsibility of the robot controller.

Every emitted action includes `safety_flags` and `safety_rejected`. Offline replay CSVs preserve
those fields along with `observation_age_ns`, making safety interventions inspectable after a run.

## Offline replay

`ReplayObservationSource` runs prerecorded observations through the same newest-wins
`ObservationSource` contract used by a live deployment. Frames carry offsets from the beginning
of an episode; `start()` rebases their capture stamps onto the current monotonic clock, and the
inference worker receives the newest frame whose offset has elapsed. Recorded steady-clock epochs
are deliberately ignored because they are meaningless in another process or machine.

`ReplayActionSink` reserves its complete output capacity before control starts. It records actions
without allocating or performing file I/O on the control thread; after the loop stops,
`write_replay_actions_csv()` writes steps, relative deadlines, emission times, fallback flags, and
action values. Dataset decoding is intentionally outside the real-time core: an adapter loads an
episode into `std::vector<ReplayObservationFrame>`, then the existing `RuntimeLoop` and either the
synthetic or SmolVLA `ChunkGenerator` execute it unchanged.

The dependency-free `replay` test exercises timestamp rebasing, newest-frame selection, bounded
recording, CSV output, and a complete replay through the asynchronous inference and control loops.
No GPU is required for this test.

Convert a portable NumPy episode and run it through the CPU synthetic worker:

```bash
PYTHONPATH=python python -m cerebellum_model.replay_export \
  --input episode.npz --output episode.cbr

./build/bench_replay --episode episode.cbr --runner synthetic --device cpu \
  --actions-output predicted-actions.csv --report-output replay-report.json
```

The NPZ contract and binary format are documented in [`docs/replay.md`](docs/replay.md). The same
`bench_replay` command accepts `--runner smolvla --device cuda`; use that only when a GPU is free.

See [`spec.md`](spec.md) for the problem statement, latency budget, invariants, measurement
methodology, and phased plan. Performance numbers in the spec remain targets until the benchmark
tables and sweeps are committed.

Phase 1 is the runtime standalone on synthetic input; phase 2 wires in Retina and Axon.
The deliverables, in priority order, are a per-stage p50/p99 latency table, a chunk-size
vs. staleness/jitter sweep, and underrun count vs. refresh trigger.

## License

[MIT](LICENSE)
