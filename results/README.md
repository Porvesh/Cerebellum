# Baseline benchmark

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
