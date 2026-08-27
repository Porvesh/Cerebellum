# Cerebellum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time execution for chunked vision-language-action (VLA) policies.

Cerebellum keeps robot control on a fixed clock while model inference runs asynchronously. The
control thread never waits for the GPU: a worker publishes action chunks into a lock-free queue,
and the runtime measures both missed deadlines and observation-to-action staleness.

```text
observation -> inference worker -> action-chunk queue -> fixed-rate control -> robot
                    slow/variable             fast/non-blocking
```

## What is implemented

- C++20 two-thread runtime with a fixed-grid control loop and bounded SPSC queue.
- Persistent framed C++/Python bridge for synthetic and SmolVLA inference.
- Three chunk-stitching policies: Discard, temporal Ensemble, and RTC.
- Model-space RTC conditioning with robot-space execution.
- Observation freshness checks, action bounds, rate/acceleration limits, and safe fallback.
- LIBERO adapter with separate policy cadence and simulator frequency, non-blocking video
  recording, action traces, and automatic task-completion stop.
- Portable offline replay through the same runtime path.

## RTC and the Physical Intelligence paper

Cerebellum implements the inference-time method from Physical Intelligence's
[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
([project page](https://www.pi.website/research/real_time_chunking)). While the current chunk is
executing, RTC generates its replacement, freezes the actions already committed to execution, and
inpaints the remaining action horizon during flow denoising. No policy retraining is required.

This repository combines that method with the systems work needed to run it: asynchronous
scheduling, conservative delay/horizon tracking, padded model-space prefix alignment, queue
handoff, safety filtering, and end-to-end timing. Discard remains available as the
checkpoint-compatible behavior baseline, and temporal ensembling provides a simpler overlap
alternative.

## See the difference

[Watch the matched Discard vs RTC-5 rollout](results/libero_discard_vs_rtc5.mp4)

The side-by-side video uses the same LIBERO task, initial state, checkpoint, and 10 Hz experimental
runtime. RTC reduced the median raw chunk-boundary jump from `1.990` to `0.082` L-infinity—about
96%—and is visibly smoother at typical seams. The p90 seam remained near `2.0`, and neither run
completed the task, so this is evidence of continuity rather than policy success.

| Matched rollout | Discard | RTC-5 |
|---|---:|---:|
| Commands delivered | 299 / 300 | 299 / 300 |
| Median boundary jump, L-inf | 1.990 | 0.082 |
| p90 boundary jump, L-inf | 2.032 | 2.051 |
| Task success | no | no |

The individual videos and action traces are listed in [results](results/README.md).

## Verified LIBERO behavior

The compatible default for `HuggingFaceVLA/smolvla_libero` replans after every executed action:

- 3 Hz external policy cadence; LIBERO remains at its native 20 Hz simulation/controller rate.
- Fresh-start chunk alignment, full 10-step inference, and reproducible per-observation noise.
- `libero_spatial` task 0 completed after 83 actions in 27.67 seconds.
- 83/83 commands delivered, with no underruns, stale-action violations, or inference failures.

See the machine-readable [successful rollout](results/libero_compatible_task0.json). RTC currently
uses three denoising steps as an experimental smoother; it preserves freshness but has not yet
preserved task success on this checkpoint.

## Build and test

Requirements: CMake 3.24+, a C++20 compiler, Python 3.11, and NumPy.

```bash
cmake -S . -B build -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=python python -m pytest -q
```

Install the optional model and simulator dependencies:

```bash
python -m pip install -e '.[libero,video]'
```

## Run LIBERO

```bash
: "${GPU_ID:?Set GPU_ID to an available physical GPU index}"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/path/to/huggingface/cache \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$GPU_ID" \
./build/run_libero \
  --model HuggingFaceVLA/smolvla_libero \
  --output results/libero_run.json \
  --video results/libero_run.mp4 \
  --actions-csv results/libero_run_actions.csv
```

The default command runs the verified 3 Hz Discard profile. Add `--stitching rtc` only when
evaluating the experimental RTC path. EGL/OSMesa setup and all CLI controls are documented in
[docs/libero.md](docs/libero.md).

## Design boundaries

Cerebellum schedules a policy; it does not train one. A base checkpoint smoke test proves loading,
preprocessing, denoising, and shapes, not useful behavior without task-specific weights. Hardware
collision protection, torque limits, and emergency stop remain the robot controller's job.

The core runtime is intentionally independent of LIBERO and SmolVLA. `ObservationSource`,
`ChunkGenerator`, and `ActionSink` are the adapter seams for another camera stack, policy, or robot.

See [architecture](ARCH_DIAG.md), [runtime specification](spec.md), [LIBERO operation](docs/libero.md),
[replay format](docs/replay.md), and [results](results/README.md).

## License

[MIT](LICENSE)
