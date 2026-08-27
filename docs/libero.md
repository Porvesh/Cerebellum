# LIBERO operation

`run_libero` connects the C++ runtime, the persistent SmolVLA worker, and LIBERO's Robosuite
environment. The policy cadence and simulator controller frequency are independent.

## Install

```bash
python -m pip install -e '.[libero,video]'
```

The checkpoint must advertise an 8D state, two cameras, and a 7D robot action. Cerebellum rejects
schema mismatches before the control loop starts.

## Rendering

For GPU rendering, set MuJoCo to EGL and select the matching device:

```bash
: "${GPU_ID:?Set GPU_ID to an available physical GPU index}"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID="$GPU_ID" \
./build/run_libero --model HuggingFaceVLA/smolvla_libero
```

If the system GLVND installation cannot load the NVIDIA EGL driver, provide a user-owned vendor
manifest and library directory whose version matches `/proc/driver/nvidia/version`:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/path/to/10_nvidia.json \
LD_LIBRARY_PATH=/path/to/nvidia/userspace \
./build/run_libero --model HuggingFaceVLA/smolvla_libero
```

OSMesa is the CPU fallback:

```bash
CEREBELLUM_OSMESA_LIBRARY_PATH=/path/to/libOSMesa.so \
./build/run_libero --model HuggingFaceVLA/smolvla_libero \
  --osmesa-library-path /path/to/libOSMesa.so
```

## Behavior profiles

The default is the verified checkpoint-compatible profile:

```bash
./build/run_libero \
  --stitching discard \
  --control-hz 3 \
  --simulation-hz 20 \
  --chunk-alignment fresh-start \
  --max-staleness-ms 700
```

It runs full ten-step inference and executes action zero from every accepted plan, matching the
checkpoint's `n_action_steps=1` evaluation behavior.

Experimental RTC uses model-space prefix conditioning and absolute chunk alignment:

```bash
./build/run_libero \
  --stitching rtc \
  --chunk-alignment absolute \
  --rtc-denoise-steps 3
```

RTC is smoother in the matched 10 Hz comparison, but it has not preserved task success on this
checkpoint. Do not present it as the behavior default.

## Outputs

```bash
./build/run_libero \
  --output results/run.json \
  --video results/run.mp4 \
  --actions-csv results/run_actions.csv
```

- JSON records cadence, delivery, freshness, inference, safety, simulator, and task outcomes.
- MP4 places both policy cameras side by side with live control metadata. Encoding runs off the
  simulator and control threads.
- CSV records each safety-filtered command, source chunk/index, and exact boundary markers.

The run stops early when LIBERO reports completion. Use `--ticks` to set the maximum action count.

## Important controls

| Option | Purpose |
|---|---|
| `--suite`, `--task-id`, `--init-state` | Select the recorded LIBERO episode state. |
| `--control-hz` | External policy command cadence. |
| `--simulation-hz` | Internal Robosuite controller frequency. |
| `--max-staleness-ms` | Runtime freshness requirement. |
| `--max-observation-age-ms` | Safety rejection threshold. |
| `--gripper-rate-limit` | Limits binary-sign reversals in faster experiments. |
| `--mode synchronous` | Applies every command for policy evaluation; not a real-time claim. |
| `--allow-download` | Permits a missing checkpoint to be fetched. |

The report is the source of truth. A process exit without failed inference is not enough to claim
real-time delivery, freshness, or task success.
