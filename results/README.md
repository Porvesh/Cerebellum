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
rate. It did not complete task 0. User-local CPU OSMesa rendering advanced only
161 of 300 commands during the 30-second control interval; the newest-wins
mailbox superseded 138 commands. This is a simulator-throughput limitation, so
the failed task is not a clean policy-quality measurement. GPU/EGL rendering or
a synchronous evaluation mode is needed before reporting success rate.

The run also showed that 216 emitted commands needed `[-1, 1]` action-space
clamping and 275 of 298 model actions exceeded Cerebellum's 350 ms freshness
bound under Discard stitching. Both require investigation before claiming a
deployment-quality LIBERO controller.

- [`libero_smolvla_gpu3_discard.json`](libero_smolvla_gpu3_discard.json)

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
