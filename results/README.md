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
