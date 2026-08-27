# Architecture

Cerebellum separates slow, irregular policy inference from fixed-rate robot control.

```text
                        non-real-time
 observation snapshot ──> inference worker ──> ActionChunk
          ▲                                      │
          │                               bounded SPSC queue
          │                                      │
       robot/sim <── ActionSink <── safety <── control thread
                         fixed cadence / no GPU waits
```

## Components

`ObservationSource` publishes immutable, timestamped state, task text, and camera frames. `latest()`
is non-blocking and newest-wins.

`ChunkGenerator` turns one observation into both padded model-space actions and postprocessed
robot-space actions. The Python implementation uses a persistent framed subprocess and validates
the model/action/image schema during its handshake.

`ActionChunkQueue` transfers complete chunks from the inference worker to the control thread. It
is single-producer/single-consumer, bounded, allocation-free after construction, and ordered by
generation.

`RuntimeLoop` owns the two threads, refresh scheduling, chunk selection, fallback behavior,
safety filtering, and timing metrics.

`ActionSink` receives one resolved command per tick. The LIBERO adapter uses a newest-action
mailbox so Python/MuJoCo and video encoding never block the control thread.

## Control thread

At each absolute deadline the control thread:

1. Drains newly generated chunks.
2. Accepts or rejects them according to the stitching policy and alignment mode.
3. Selects the action for the current absolute step.
4. Applies the safety filter or configured fallback.
5. Emits once to the sink and records causal timing metadata.

It performs no model work, file I/O, Python calls, or unbounded allocation.

## Inference worker

The worker waits for the active refresh policy, snapshots the newest observation, and generates a
replacement chunk. Only the worker touches the model process. Failed or late generations are
counted and never stall control.

For RTC, the request also carries the retained chunk identity and the model-space actions that are
guaranteed to execute before the new chunk becomes available.

## Stitching policies

### Discard

Accept a new chunk and discard the unexecuted tail of the old one. This is the lowest-cost policy,
but independent generations can jump at the boundary.

### Ensemble

Blend overlapping predictions for the same absolute action step. This reduces seams without
changing the model, at the cost of retaining overlapping chunks and some temporal lag.

### Real-time chunking

RTC follows [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339).
Let `d` be the conservative inference delay in control steps and `s` the execution horizon. The
worker starts the replacement after `s-d` actions. During denoising, actions committed from the old
chunk are frozen through gradient-guided inpainting; the remainder stays free to react to the new
observation.

Conditioning happens in the checkpoint's padded model space. Execution always uses the
postprocessed robot space. Keeping both representations in every chunk prevents normalization or
padding errors at the seam.

## Alignment

Absolute alignment maps chunk index `i` to the control timeline and is required for RTC overlap.
Fresh-start alignment executes index zero when a new chunk is accepted. The latter matches the
LIBERO checkpoint's evaluation behavior when replanning after each action.

## Timing and safety

The system reports three different quantities:

- Deadline lateness: when the command was emitted relative to its fixed deadline.
- Observation staleness: capture-to-command age of the observation behind that action.
- Queue age: chunk-ready-to-command time inside Cerebellum.

These are not interchangeable. A loop can meet cadence while using stale actions.

`ActionSafetyFilter` rejects non-finite or over-age actions, clamps bounds, and limits configured
rates and acceleration. Rejection holds the last safe command or a startup replacement. Hardware
collision, torque, and emergency-stop protection remain below this runtime.

## Core invariants

- The control thread never waits for inference or simulator work.
- Exactly one resolved command is offered per control tick.
- Published observations and chunks are immutable.
- An unsafe prediction cannot reappear through fallback.
- Model-space conditioning and robot-space execution are never confused.
- Task success is reported only after the sink acknowledges environment completion.
