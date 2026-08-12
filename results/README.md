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

## Discard runtime baseline

The complete `RuntimeLoop` ran for 300 ticks at 30 Hz with the default 50-action
chunk and refresh trigger of 6. The synthetic generator slept for 149 ms; the
real run used SmolVLA on the same H100 after ten model warmups. Neither run
prefilled the queue, so startup behavior is included.

| Runtime metric | Synthetic 149 ms | SmolVLA/H100 |
|---|---:|---:|
| Action-emission lateness p50 | ~0.10 ms | ~0.08 ms |
| Action-emission lateness p99 | ~0.11 ms | ~0.10 ms |
| Skipped control steps | 0 | 0 |
| Fallbacks / underruns | 4 / 300 | 4 / 300 |
| Staleness p50 | 866.53 ms | 866.49 ms |
| Staleness p99 | 1599.99 ms | 1599.90 ms |
| Staleness violations | 283 / 296 | 283 / 296 |
| Actions discarded at seams | 28 | 28 |

The refresh trigger successfully hides approximately 149 ms of inference after
startup: there are no later underruns or skipped periods. It does **not** satisfy
the 200 ms staleness contract. Discard continues executing most of each
50-action chunk, so queue age—not Cerebellum bridge overhead—dominates
observation-to-action staleness.

`strict_deadline_misses` counts every positive offset from the exact deadline;
normal scheduler wakeup latency therefore counts even when it is only about
0.1 ms. The distribution and `steps_skipped` are more informative than that
strict count by itself.

Raw runtime reports:

- [`runtime_discard_synthetic_baseline.json`](runtime_discard_synthetic_baseline.json)
- [`runtime_discard_smolvla_h100_baseline.json`](runtime_discard_smolvla_h100_baseline.json)
