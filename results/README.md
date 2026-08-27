# Results

This directory keeps the machine-readable evidence behind the README. The current claims are the
checkpoint-compatible LIBERO result and the matched Discard/RTC continuity comparison; older
profiling and scheduler reports remain as raw artifacts, not headline results.

## Checkpoint-compatible LIBERO result

`HuggingFaceVLA/smolvla_libero` completed `libero_spatial` task 0, init state 0, through the full
LIBERO → C++ → Python/SmolVLA → queue → safety → simulator loop.

| Metric | Result |
|---|---:|
| External policy cadence | 3 Hz |
| Internal simulator/controller frequency | 20 Hz |
| Completion time | 27.67 s |
| Model commands delivered | 83 / 83 |
| Underruns / stale violations | 0 / 0 |
| Superseded commands | 0 |
| Inference failures | 0 |
| Task success | yes |

The profile uses Discard stitching, full ten-step inference, fresh-start alignment, and a
reproducible noise seed derived from each observation. It stops as soon as LIBERO acknowledges
task completion.

- [Machine-readable report](libero_compatible_task0.json)

## Matched smoothness comparison

[Watch Discard and RTC-5 side by side](libero_discard_vs_rtc5.mp4)

Both 30-second rollouts use the same LIBERO task, initial state, checkpoint, GPU/EGL path, and
experimental 10 Hz command cadence. RTC follows the inference-time conditioning method in
[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339).

| Runtime metric | Discard | RTC-5 |
|---|---:|---:|
| Control rate | 10.000 Hz | 9.948 Hz |
| Commands delivered | 299 / 300 | 299 / 300 |
| Median executed action change, L-inf | 0.210 | 0.082 |
| Median executed second difference, L-inf | 1.995 | 0.109 |
| Median raw chunk-boundary jump, L-inf | 1.990 | 0.082 |
| p90 raw chunk-boundary jump, L-inf | 2.032 | 2.051 |
| Task success | no | no |

RTC reduced the typical raw boundary jump by about 96%. It did not fix the gripper-dominated tail,
and guided inference made actions older. This comparison supports a smoothness claim only; it does
not support a task-performance claim.

- [Discard video](libero_realtime_discard.mp4) · [RTC-5 video](libero_realtime_rtc5.mp4)
- [Discard report](libero_realtime_discard_video.json) · [RTC-5 report](libero_realtime_rtc5_video.json)
- [Discard actions](libero_realtime_discard_actions.csv) · [RTC-5 actions](libero_realtime_rtc5_actions.csv)

## RTC-3 follow-up

Profiling showed that denoiser forwards and RTC gradient/VJP work dominate conditioned inference;
prefix preparation and correction are negligible. Reducing the LIBERO RTC path from five to three
denoising steps restored freshness and retained a roughly 95% median seam reduction, but task 0
still failed. RTC-3 is therefore an experimental runtime candidate, not the behavior default.

- [RTC-3 rollout](libero_realtime_rtc3.json)
- [RTC-3 actions](libero_realtime_rtc3_actions.csv)
- [Denoising comparison](libero_rtc_steps345_gpu3.json)
- [Profile attribution](libero_rtc5_nsight_summary.json)

## Bridge baseline

The full C++ → Python → C++ bridge transported realistic three-camera payloads. With the base
SmolVLA checkpoint on an isolated H100, p50 end-to-end request latency was 149.39 ms, of which
148.26 ms was the model call and 1.09 ms was Cerebellum overhead.

- [Synthetic bridge](synthetic_bridge_baseline.json)
- [SmolVLA/H100 bridge](smolvla_h100_baseline.json)

## Reading older artifacts

The remaining JSON and CSV files capture development-time scheduler sweeps, replay smoke tests,
rendering comparisons, and superseded 10/30 Hz profiles. They are retained for reproducibility,
but should not be mixed with the verified 3 Hz task result. In particular:

- Synchronous runs measure policy behavior, not real-time freshness.
- Blank-camera replay runs validate plumbing, not action quality.
- Synthetic runs validate timing and contracts, not model behavior.
- Earlier 10 Hz LIBERO runs predate the checkpoint-compatible `n_action_steps=1` profile.
