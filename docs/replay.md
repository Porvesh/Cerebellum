# Offline replay

Replay separates model/runtime evaluation from robot actuation. One episode contains timestamped
observations and, optionally, the reference commands recorded at the corresponding absolute
control steps. Cerebellum plays the observations on a new monotonic clock and writes what it would
have commanded instead of moving hardware.

## NPZ interchange contract

`python -m cerebellum_model.replay_export` accepts one `.npz` file with these arrays:

| Key | Shape and type | Meaning |
|---|---|---|
| `task` | scalar Unicode string | Episode-wide language instruction |
| `offset_ns` | `(N,) int64` | Nanoseconds from episode start; first is zero, strictly increasing |
| `state` | `(N, S)` finite numeric | Robot state at each observation |
| `sequence` | optional `(N,) uint64` | Strictly increasing source IDs; defaults to `1..N` |
| `action` | optional `(N, A)` finite numeric | Reference robot commands, `1 <= A <= 32` |
| `image.<feature>` | `(N, H, W, C) uint8` | Camera-native NHWC frames |

There must be at least one image key. The suffix after `image.` is transported unchanged, so for
SmolVLA it must match the checkpoint feature name, such as
`observation.images.camera1`. All observations in an episode have one task and fixed state, camera,
and action shapes.

Example creation:

```python
import numpy as np

np.savez(
    "episode.npz",
    task=np.asarray("pick up the blue block"),
    offset_ns=np.arange(frame_count, dtype=np.int64) * 33_333_333,
    state=states.astype(np.float32),
    action=recorded_actions.astype(np.float32),
    **{
        "image.observation.images.camera1": camera1.astype(np.uint8),
        "image.observation.images.camera2": camera2.astype(np.uint8),
        "image.observation.images.camera3": camera3.astype(np.uint8),
    },
)
```

## CBR binary

The exporter converts NPZ into Cerebellum Binary Replay (`CBRPLY1`). CBR uses little-endian fixed
width numeric fields, one episode-wide task/camera schema, then fixed-shape frame records. Each
frame contains its source sequence, offset, float32 state, camera bytes, and optional float32
reference action. The C++ loader rejects bad versions, truncation, trailing data, non-finite
values, non-monotonic timelines, and schema changes before the runtime starts.

NPZ is the easy interchange format for dataset adapters; CBR is the dependency-free format the
C++ runtime reads. Recorded `steady_clock` epochs are never stored because they have no meaning in
a later process. `ReplayObservationSource::start()` rebases capture times onto the playback clock.

## Running and interpreting quality

```bash
./build/bench_replay \
  --episode episode.cbr \
  --runner synthetic \
  --device cpu \
  --actions-output predicted-actions.csv \
  --report-output replay-report.json
```

For RTC, add:

```text
--runner smolvla --device cuda --stitching rtc
--rtc-denoise-steps 5 --rtc-inference-delay 8 --rtc-execution-horizon 8
```

The report includes runtime staleness/underruns plus reference-action MAE, RMSE, and maximum
L-infinity error. Fallback commands are counted separately and excluded from model-error averages;
otherwise a held safety command would be misreported as a model prediction. These open-loop errors
compare policies on identical input, but they do not prove closed-loop task success.

When an episode has no trustworthy reference actions, two replay outputs can still be compared:

```bash
PYTHONPATH=python python -m cerebellum_model.replay_compare \
  --left rtc5-actions.csv --left-label rtc-5-step \
  --right rtc10-actions.csv --right-label rtc-10-step \
  --output rtc5-vs-rtc10.json
```

This reports disagreement on steps where both runs emitted real actions and reports per-run action
smoothness. It measures how two configurations differ; it cannot say which trajectory is correct.
