# Bridge baseline benchmark

Measured 2026-08-20 on one isolated NVIDIA H100 PCIe with driver 580.105.08,
PyTorch 2.10.0+cu128, CUDA 12.8, and `lerobot/smolvla_base`. Model loading is
excluded: ten warm-up requests run after the persistent worker has loaded the
checkpoint, followed by 100 measured requests.

Both runs transport the same six-value state, task string, and three 256×256 RGB
camera frames through the full C++ → Python → C++ bridge. The synthetic backend
therefore measures Cerebellum with realistic payload size rather than a tiny test
message.

| Warm metric | Synthetic | SmolVLA/H100 |
|---|---:|---:|
| Total p50 | 1.05 ms | 149.39 ms |
| Total p99 | 2.37 ms | 153.66 ms |
| Model-call p50 | 0.28 ms | 148.26 ms |
| Cerebellum-overhead p50 | 0.77 ms | 1.09 ms |
| Cerebellum-overhead p99 | 1.93 ms | 1.44 ms |
| Model share of p50 total | 26.61% synthetic stub | 99.25% |
| Cerebellum share of p50 total | 73.39% | 0.75% |

`python_model` is the complete `runner.predict()` call, including the model's
LeRobot preprocessing and postprocessing. `cerebellum_overhead` is calculated
per request as `total_observation_lookup_to_chunk - python_model`; it includes C++
observation handling, binary framing, IPC, Python request decoding and image
conversion, response encoding, and C++ response decoding.

The total begins when C++ looks up the latest observation; capture-to-action
staleness and queue/control cadence are reported separately by `ControlMetrics`.

`response_wait` overlaps the Python stages, so it must not be added to them. It
exists to explain the C++ thread's wall time, not as another term in the total.

Raw reports:

- [`synthetic_bridge_baseline.json`](synthetic_bridge_baseline.json)
- [`smolvla_h100_baseline.json`](smolvla_h100_baseline.json)

## LIBERO closed-loop integration

The `HuggingFaceVLA/smolvla_libero` checkpoint ran through the complete
LIBERO → C++ observation adapter → SmolVLA → action queue → safety filter →
MuJoCo loop on physical GPU 3. The checkpoint schema matched without adaptation:
two 256×256 cameras, an 8D Panda state, 50×32 model-space chunks, and 50×7
postprocessed robot commands. All 100 inference requests completed successfully.

The rollout used the official `lerobot/libero` dataset's native 10 Hz action
rate. Its 100 ms period dynamically produced a three-action refresh trigger;
the live latency estimator converged to `d=s=3`. It did not complete task 0.
User-local CPU OSMesa rendering advanced only 158 of 300 commands during the
30-second control interval; the newest-wins mailbox superseded 140 commands. This is a
simulator-throughput limitation, so
the failed task is not a clean policy-quality measurement. GPU/EGL rendering or
a synchronous evaluation mode is needed before reporting success rate.

The run also showed that 200 emitted commands needed `[-1, 1]` action-space
clamping and 270 of 298 model actions exceeded Cerebellum's 350 ms freshness
bound under Discard stitching. Both require investigation before claiming a
deployment-quality LIBERO controller.

A follow-up smoke run made those requirements machine-readable instead of
inferring them from raw counters. The LIBERO profile uses a provisional 300 ms
inference budget, which derives a four-action refresh trigger and `d=s=3` at
10 Hz. The control loop sustained 9.999 Hz and inference had no failures, but
CPU OSMesa sustained only 5.000 simulator steps/s and delivered 51.7% of
resolved commands. The report therefore marks simulator throughput, command
delivery, freshness, and overall runtime liveness as failed while leaving the
process-health result true. The thresholds are 95% of the policy-native action
rate and 99% command delivery; the existing 350 ms freshness bound was not
loosened to make the simulator pass.

- [`libero_smolvla_gpu3_discard.json`](libero_smolvla_gpu3_discard.json)
- [`libero_liveness_gpu3_smoke.json`](libero_liveness_gpu3_smoke.json)

### Synchronous LIBERO evaluation

The same task then ran for all 300 policy steps in synchronous mode. This mode
advances the C++ absolute step only after the simulator acknowledges the current
action, so it measures policy behavior without pretending that CPU OSMesa is
real-time. All 300 emitted commands were applied in order, none were
superseded, all 150 inference requests succeeded, and the evaluation pipeline
passed. Deployment freshness is reported but is explicitly inapplicable to this
slower-than-real-time mode; its wall-clock age is not used to reject actions.

The 30 simulated seconds took 59.331 wall-clock seconds and the task succeeded.
Episode success is sticky: later motion cannot erase the fact that LIBERO's
success condition was reached earlier. This changes the interpretation of the
real-time failure: SmolVLA can solve this task when every selected action is
applied, and dropping roughly half the trajectory was consequential.

The new stage timings locate the throughput problem: LIBERO `env.step()`
including camera rendering averaged 195.890 ms/action, observation construction
averaged 0.962 ms/action, and the complete C++ command to decoded-observation
round trip averaged 197.603 ms/action. Nearly all of the time is inside the
simulator environment step, not Cerebellum IPC or image copying. The safety
filter clamped 166 actions without rejecting any, which is the next policy-path
issue to investigate.

- [`libero_synchronous_gpu3_discard.json`](libero_synchronous_gpu3_discard.json)

### Real-time NVIDIA EGL rollout

NVIDIA's 580.105.08 userspace EGL libraries were extracted into the user's home directory to
match the loaded kernel driver; no root installation or render-group membership was required.
On physical GPU 3, EGL reduced LIBERO `env.step()` including rendering from 195.890 ms/action
under CPU OSMesa to 41.925 ms/action. The complete command-to-observation round trip fell from
197.603 ms to 43.326 ms. The simulator and C++ control loop both sustained exactly 10 Hz for the
full 300-action episode, with 300/300 commands applied and none superseded.

This removes simulator throughput as the explanation for the remaining failure. An initial run
showed that the old 350 ms bound was below normal operation, so the report now separates raw
cold-start staleness from safety-eligible steady-state staleness. Across repeated 300-action runs,
the latter measured about 457 ms p50, 558 ms p90, 561.5 ms p99, and 568.4 ms maximum. The LIBERO
profile therefore uses 575 ms: the smallest round bound with scheduling margin above the observed
maximum. The base 30 Hz profile remains at 350 ms.

With the 575 ms target, all 295 safety-eligible actions passed. The only three violations were
25-second-old startup observations captured before model loading completed; the safety filter
rejected them independently. Synchronous evaluation succeeded because the roughly 300 ms model
inference consumed fewer simulated action intervals there; in real time it produces only one
fresh chunk per approximately three actions. Cold-start observation handling and RTC/inference
freshness remain separate follow-up work.

The model is now loaded and warmed against a schema-correct dummy observation before LIBERO is
constructed. The following rollout therefore captured the real initial state after model startup:
all 298 model actions were younger than the one-second safety limit and stale safety rejections
fell from three to zero. A real scheduling tail remained—four actions exceeded 575 ms, with
655.6 ms p99 and 755.6 ms maximum—so the report does not claim a freshness pass. Queue prefill
and RTC should reduce that tail; increasing the requirement again from one outlier run would hide
it.

Per-dimension diagnostics also explain the high clamp count. Of 220 clamped ticks, the raw model
crossed the gripper bound 215 times (75 below -1 and 140 above +1, maximum excess 0.049) and moved
`delta_z` below -1 seven times (maximum excess 0.055). No x/y or rotation dimension crossed its
bounds. SmolVLA mean/std unnormalization does not enforce environment bounds; Robosuite already
clips controller input and the Panda gripper uses the action sign. Cerebellum's clamp is therefore
the expected safety boundary rather than evidence of double-normalization.

- [`libero_egl_gpu3_realtime_discard.json`](libero_egl_gpu3_realtime_discard.json)

### Matched Discard and RTC-5 videos

The same real-time EGL setup recorded matched 300-action episodes for Discard and five-step RTC.
Each MP4 contains the two observations seen by the policy plus per-action control metadata. Both
runs applied 299 of 300 commands, passed the 99% delivery requirement, and sustained the 9.5 Hz
liveness floor. One latest-mailbox supersession per run is materially different from the earlier
CPU-rendering loss of 140 commands, but neither episode is a perfect all-command evaluation.

| Runtime metric | Discard | RTC, 5 denoising steps |
|---|---:|---:|
| Control rate | 10.000 Hz | 9.948 Hz |
| Commands delivered | 299 / 300 | 299 / 300 |
| Real actions / fallbacks | 298 / 2 | 299 / 1 |
| Inference requests | 100 | 76 |
| Staleness p50 | 457.3 ms | 558.7 ms |
| Staleness p90 | 558.5 ms | 756.7 ms |
| Staleness maximum | 656.4 ms | 901.1 ms |
| Actions above 575 ms | 1 | 149 |
| Median executed action change, L-inf | 0.210 | 0.082 |
| Median executed second difference, L-inf | 1.995 | 0.109 |
| Median executed third difference, L-inf | 1.034 | 0.390 |
| Median raw chunk-boundary jump, L-inf | 1.990 | 0.082 |
| p90 raw chunk-boundary jump, L-inf | 2.032 | 2.051 |
| Task success | no | no |

RTC clearly improves typical smoothness: its median raw boundary jump is about 96% smaller, while
the executed first, second, and third finite differences fall by about 61%, 95%, and 62%. It does
not remove the tail: p90 boundary jumps remain near 2.0. The action CSV confirms the gripper
dimension reaches about 2.0 at those boundaries while RTC keeps every arm dimension below 0.26.
Its autograd/VJP guidance also makes each request more expensive; the horizon
scheduler issued fewer requests, and actions were older. RTC-5 therefore demonstrates its intended
continuity benefit but is not deployment-ready. Guided-inference latency and the remaining tail
transitions are the next functional bottlenecks.

- [`libero_realtime_discard.mp4`](libero_realtime_discard.mp4)
- [`libero_realtime_rtc5.mp4`](libero_realtime_rtc5.mp4)
- [`libero_realtime_discard_video.json`](libero_realtime_discard_video.json)
- [`libero_realtime_rtc5_video.json`](libero_realtime_rtc5_video.json)
- [`libero_realtime_discard_actions.csv`](libero_realtime_discard_actions.csv)
- [`libero_realtime_rtc5_actions.csv`](libero_realtime_rtc5_actions.csv)

### Non-blocking recording and RTC-3 optimization

Video composition and H.264 encoding now run on a bounded background queue rather than the
simulator I/O thread. A repeated 300-tick Discard rollout produced 300 applied commands, zero
supersessions, and 300 video frames at exactly 10 Hz. The adapter also drains the final mailbox
publication before reporting, preventing an applied command from being misclassified as pending.

Nsight Systems 2026.4.1 then captured one five-step conditioned request using the exact LIBERO
checkpoint. Under tracing, the five forward denoiser calls consumed 222.8 ms and the five VJPs
248.6 ms—87% of the enclosing RTC range together. Prefix preparation and correction application
used only 1.1 ms. The optimization target is therefore the number or implementation of
forward/backward pairs, not Cerebellum transport or Retina.

An unprofiled exact-checkpoint sweep measured the latency/conditioning tradeoff:

| RTC steps | Latency p50 | Latency p99 | Guided / unguided prefix MSE |
|---:|---:|---:|---:|
| 3 | 229.2 ms | 231.7 ms | 10.31% |
| 4 | 299.0 ms | 304.1 ms | 4.50% |
| 5 | 340.5 ms | 370.2 ms | 3.42% |

Three steps were then run through the full 300-tick real-time loop. RTC-3 applied all 300 commands,
had no freshness violations, and measured 457.0 ms p50, 557.7 ms p90, and 566.0 ms maximum action
age. Its median raw boundary jump was 0.090, about 95% below Discard's 1.990, while its p90 boundary
jump improved to 1.808. The task still failed, so this is a runtime/freshness success rather than a
policy-success claim. Three steps are now the LIBERO-specific default; other checkpoints retain
their own configuration and must be remeasured.

- [`libero_async_video_verification.json`](libero_async_video_verification.json)
- [`libero_rtc5_nsight_summary.json`](libero_rtc5_nsight_summary.json)
- [`libero_rtc_steps345_gpu3.json`](libero_rtc_steps345_gpu3.json)
- [`libero_realtime_rtc3.json`](libero_realtime_rtc3.json)
- [`libero_realtime_rtc3_actions.csv`](libero_realtime_rtc3_actions.csv)

## Discard refresh-policy baseline

The complete `RuntimeLoop` ran for 300 ticks at 30 Hz with the default 50-action
chunk and refresh trigger of 6. Tail drains the chunk until the trigger;
Continuous requests the next chunk immediately after accepting the previous one.
Neither mode prefills the queue, so startup behavior is included.

| Runtime metric | Synthetic Tail | Synthetic Continuous | H100 Tail | H100 Continuous |
|---|---:|---:|---:|---:|
| Generation wall / request | 149 ms | 149 ms | ~229 ms | ~239 ms |
| Worker busy | 10.44% | 89.48% | 16.03% | 90.63% |
| Chunks / 10 s | 7 | 60 | 7 | 38 |
| Staleness p50 | 866.52 ms | 233.25 ms | 933.19 ms | 366.64 ms |
| Staleness p99 | 1599.93 ms | 300.01 ms | 1666.60 ms | 499.99 ms |
| Violations / real actions | 254 / 296 | 0 / 296 | 264 / 288 | 180 / 293 |
| Fallbacks / underruns | 4 | 4 | 12 | 7 |
| Skipped control steps | 0 | 0 | 5 | 0 |
| Actions discarded | 28 | 241 | 46 | 262 |

The controlled synthetic comparison isolates scheduling: Continuous satisfies
the 350 ms bound, but increases worker duty from 10.44% to 89.48% and discards
241 predicted actions. This is the freshness-versus-compute tradeoff RTC will
inherit; RTC is expected to improve the frequent seams, not remove inference
cost.

The refresh-policy H100 runs were **not isolated**. External processes reserved
about 74 GiB on every GPU (0% utilization at launch), and observed SmolVLA
generation rose from the earlier isolated ~149 ms baseline to ~229–239 ms.
These contention-sensitive results deliberately show why a 350 ms target only
holds when the latency assumption holds: Continuous p99 reached ~500 ms. They
must not be presented as clean power or model-performance measurements.

`strict_deadline_misses` counts every positive offset from the exact deadline;
normal scheduler wakeup latency therefore counts even when it is only about
0.1 ms. The distribution and `steps_skipped` are more informative than that
strict count by itself.

Raw runtime reports:

- [`runtime_discard_synthetic_baseline.json`](runtime_discard_synthetic_baseline.json)
- [`runtime_discard_continuous_synthetic_baseline.json`](runtime_discard_continuous_synthetic_baseline.json)
- [`runtime_discard_smolvla_h100_baseline.json`](runtime_discard_smolvla_h100_baseline.json)
- [`runtime_discard_continuous_smolvla_h100_baseline.json`](runtime_discard_continuous_smolvla_h100_baseline.json)

## RTC synthetic integration check

The horizon scheduler and committed-prefix path ran for 90 ticks with the same
149 ms controlled model delay. It generated 18 chunks, recorded no RTC inference
overruns and no staleness violations among 86 real actions; p99 staleness was
300.01 ms. The four underruns are cold-start ticks before the first chunk exists.

This synthetic backend verifies scheduling, prefix alignment, IPC, and queue
behavior. It does not measure the extra GPU cost or action quality of gradient
guidance. A real SmolVLA/H100 RTC baseline remains required when an isolated GPU
has enough memory; all four GPUs were externally occupied during this change.

- [`runtime_rtc_synthetic_baseline.json`](runtime_rtc_synthetic_baseline.json)

## Real SmolVLA RTC baseline and Nsight attribution

An isolated H100 run verified both the direct two-request RTC path and the full
C++ → Python → SmolVLA → C++ bridge. The 300-tick unprofiled runtime generated
26 chunks with no RTC inference overrun, but guided generation averaged about
368 ms. This made the conservative horizon expand and produced 766.66 ms p99
staleness: 290 of 296 real actions exceeded the 350 ms target.

Nsight Systems then profiled one conditioned 10-step request. Under tracing,
the ten base-denoiser ranges totaled 249.75 ms and the ten RTC VJP/autograd
ranges totaled 268.62 ms. Prefix preparation plus correction application took
less than 2.5 ms. The important conclusion is attribution, not traced latency:
the true denoiser VJP costs roughly another model pass, while Cerebellum's RTC
bookkeeping is negligible. The next controlled experiment is the paper's
five-step RTC setting and a 5/6/8/10-step quality/latency sweep.

- [`runtime_rtc_smolvla_h100_baseline.json`](runtime_rtc_smolvla_h100_baseline.json)
- [`rtc_nsight_systems_summary.json`](rtc_nsight_systems_summary.json)

## RTC denoising-step sweep on GPU 3

A controlled run on an otherwise idle physical GPU 3 loaded SmolVLA once and
measured the same observation, noise seed, and committed prefix at 5, 6, 8, and
10 denoising steps. Each setting had one warm-up and ten measured conditioned
requests. Prefix MSE compares the generated committed region with the retained
actions it was asked to preserve; the ratio is relative to the same unguided
chunk transition.

| Steps | RTC latency p50 | RTC latency p99 | Guided/baseline prefix MSE | Peak allocation |
|---:|---:|---:|---:|---:|
| 5 | 196.32 ms | 207.65 ms | 27.00% | 1027.73 MiB |
| 6 | 274.54 ms | 284.42 ms | 19.58% | 1027.74 MiB |
| 8 | 350.94 ms | 367.16 ms | 15.62% | 1027.75 MiB |
| 10 | 451.95 ms | 463.98 ms | 13.51% | 1027.76 MiB |

Five steps is the latency winner and still reduces prefix error by about 73%
relative to no RTC guidance. In the complete 300-tick C++ runtime, five steps
also beat six: p50 staleness was 366.61 ms versus 433.18 ms, with 173/298 versus
220/297 real actions exceeding the 350 ms bound. This is an improvement, not a
pass: the five-step worker remained 93.28% busy and still violated the bound on
58.05% of real actions. The next optimization must address refresh cadence and
end-to-end freshness while checking real action quality.

- [`rtc_denoise_sweep_gpu3.json`](rtc_denoise_sweep_gpu3.json)
- [`runtime_rtc_steps5_gpu3.json`](runtime_rtc_steps5_gpu3.json)
- [`runtime_rtc_steps6_gpu3.json`](runtime_rtc_steps6_gpu3.json)

## RTC scheduler sweep on GPU 3

The complete C++ two-loop runtime then swept the paper's timing variables with
five denoising steps. Here `d` is the conservative inference delay, `s` is the
execution horizon, and `s-d` is how many actions the scheduler waits after
accepting a chunk before requesting its replacement. Every run used 300 ticks
at 30 Hz on isolated physical GPU 3. The latency estimator was active rather
than artificially fixed.

| Configured d/s | Gap | Final d/s | Staleness p50 | Staleness p99 | Violations | Worker busy |
|---:|---:|---:|---:|---:|---:|---:|
| 6/6 | 0 | 8/8 | 366.56 ms | 499.96 ms | 167/298 (56.04%) | 92.42% |
| 6/7 | 1 | 8/9 | 366.62 ms | 499.96 ms | 175/298 (58.72%) | 90.44% |
| 6/8 | 2 | 8/10 | 399.88 ms | 533.30 ms | 187/298 (62.75%) | 80.94% |
| 6/10 | 4 | 8/12 | 399.96 ms | 599.96 ms | 199/298 (66.78%) | 66.84% |
| 7/7 | 0 | 8/8 | 366.51 ms | 499.94 ms | 153/298 (51.34%) | 94.08% |
| 8/8 | 0 | 8/8 | 366.61 ms | 499.98 ms | 170/298 (57.05%) | 91.94% |

The violation count moves with request/control phase because many samples land
on the 333/367 ms bins around the 350 ms threshold; p50 and p99 are the stable
comparison. All delay floors converged to eight ticks. A zero gap is the best
schedule and a floor of eight safely describes the measured worst recent
latency for the five-step candidate, so its benchmark setting is `d=s=8`.
The general RTC default remains unchanged until the next action-quality check
decides whether five denoising steps can replace the checkpoint's ten steps.

Scheduling cannot close the remaining gap. A single serial worker spends about
230–235 ms generating each conditioned chunk. The newest observation is already
that old when the chunk arrives, and it continues aging while the next serial
request runs, producing the measured roughly 500 ms p99. Meeting a strict
350 ms bound needs conditioned generation near 175 ms or overlapping inference;
waiting longer only saves GPU duty at the cost of freshness.

- [`rtc_scheduler_sweep_gpu3.json`](rtc_scheduler_sweep_gpu3.json)
- Raw reports: [`d6/s6`](runtime_rtc_d6_s6_gpu3.json),
  [`d6/s7`](runtime_rtc_d6_s7_gpu3.json),
  [`d6/s8`](runtime_rtc_d6_s8_gpu3.json),
  [`d6/s10`](runtime_rtc_d6_s10_gpu3.json),
  [`d7/s7`](runtime_rtc_d7_s7_gpu3.json), and
  [`d8/s8`](runtime_rtc_d8_s8_gpu3.json)

## File-backed real SmolVLA replay smoke test

The complete NPZ → CBR → C++ replay → Python SmolVLA → action CSV path ran on
physical GPU 3 for 60 ticks at 30 Hz. Both runs used the identical generated
episode: six zero state values, three blank 256×256 RGB cameras, and the task
`move the object to the target`. The episode intentionally contains no
reference actions.

| Runtime metric | 5-step RTC (`d=s=8`) | 10-step RTC (`d=s=14`) |
|---|---:|---:|
| Real actions | 58 / 60 | 56 / 60 |
| Cold-start fallbacks | 2 | 4 |
| Chunks generated | 9 | 6 |
| Staleness p50 | 333.59 ms | 500.25 ms |
| Staleness p99 | 500.26 ms | 733.59 ms |
| Ready-to-emit p99 | 245.53 ms | 379.55 ms |
| Superseded action prefixes | 47 | 46 |

On the 56 steps where both configurations emitted real actions, their commands
had MAE 0.0851, RMSE 0.1237, and maximum L-infinity disagreement 0.3784. The
ten-step output was smoother on this input: mean consecutive-action L2 was
0.0593 versus 0.0804 for five steps.

These are **pipeline and relative-behavior measurements, not task-quality
results**. Blank cameras and zero state are out-of-distribution, and there is no
correct reference trajectory. The run proves that file replay drives real RTC
and quantifies the operational cost difference. Deciding whether five steps is
good enough still requires a recorded episode from the target policy/task.

- [`replay_rtc5_blank_gpu3.json`](replay_rtc5_blank_gpu3.json)
- [`replay_rtc10_blank_gpu3.json`](replay_rtc10_blank_gpu3.json)
- [`replay_rtc5_vs_rtc10_blank_gpu3.json`](replay_rtc5_vs_rtc10_blank_gpu3.json)
