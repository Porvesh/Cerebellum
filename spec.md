# Runtime specification

Cerebellum is a real-time execution layer for chunked vision-language-action policies. It turns a
slow, variable-latency model into a fixed-cadence action stream without hiding freshness failures.

## Scope

In scope:

- asynchronous observation → chunk inference;
- deterministic control scheduling and bounded handoff;
- Discard, temporal Ensemble, and real-time chunking (RTC);
- safety checks at the model/robot boundary;
- per-stage latency, freshness, seam, and delivery metrics;
- synthetic, SmolVLA, LIBERO, and replay adapters.

Out of scope:

- policy training or fine-tuning;
- camera capture/transport implementations;
- hardware collision, torque, and emergency-stop systems;
- claiming task quality from synthetic or out-of-distribution inputs.

## Runtime contract

The control period is the single clock source. Refresh thresholds, chunk duration, RTC delay and
horizon, safety timing, and measured inference-delay steps derive from it.

The base runtime target is:

> Emit one resolved action on every 33.33 ms control tick, based on an observation no older than
> 350 ms.

Adapters may define a different validated profile. LIBERO uses 3 Hz control, a separate 20 Hz
simulator/controller frequency, and a 700 ms freshness bound because full checkpoint inference
runs between actions. A slower profile must never silently weaken the base target.

## Interfaces

```cpp
struct ObservationSource {
  virtual std::shared_ptr<const ObservationSnapshot> latest() noexcept = 0;
};

struct ChunkGenerator {
  virtual bool generate(const InferenceRequest&, Chunk& out) noexcept = 0;
};

struct ActionSink {
  virtual void emit(const ActionEmission&) noexcept = 0;
  virtual bool stop_requested() const noexcept { return false; }
};
```

Observations carry a monotonic capture timestamp and immutable state/text/image payloads. Chunks
carry their source observation, generation timing, absolute start step, robot-space actions, and
padded model-space actions.

The concrete interfaces in `include/cerebellum/` are authoritative; the sketch above describes
ownership rather than exact signatures.

## Scheduling

The control thread uses absolute deadlines, not repeated relative sleeps. It drains ready chunks,
resolves the current action, applies safety, emits, and records metrics. Inference runs only on the
worker thread. The queue and sink path are bounded and preallocated.

Refresh policies:

- `tail`: request before the current chunk runs out;
- `continuous`: request again after every accepted chunk;
- `horizon`: RTC request timing based on conservative delay `d` and execution horizon `s`.

Cold-start and exhausted-queue behavior is explicit: hold the last safe command, emit a configured
replacement, or extrapolate only when the adapter has enabled it. Every fallback is counted.

## Stitching

Discard replaces the active chunk and drops its unexecuted tail. Ensemble blends predictions that
target the same absolute action step.

RTC implements the inference-time algorithm from Physical Intelligence's
[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339). A
replacement chunk is generated while the current chunk executes. The prefix guaranteed to execute
before inference completes is aligned into the new model-space chunk and enforced during each flow
denoising step; the later actions remain free to adapt to the newest observation.

RTC requires absolute timeline alignment. Fresh-start alignment is separate behavior for policies,
such as the current LIBERO checkpoint, evaluated by executing action zero after every replan.

## Safety

The optional action safety boundary:

- rejects non-finite values and observations over the configured age;
- clamps per-dimension command bounds;
- limits configured action rate and acceleration;
- retains only the filtered result for later fallback.

Every emission records safety flags, rejection state, source chunk/index, observation age, and
whether it was a fallback.

## Measurement

Reports must keep these metrics separate:

- control deadline lateness and skipped steps;
- observation capture-to-action staleness;
- chunk-ready-to-action queue age;
- model, transport, serialization, and simulator stage latency;
- generated/accepted/dropped chunks, underruns, fallbacks, and delivery fraction;
- raw chunk-boundary jump and executed finite differences;
- task success/termination when the environment provides them.

Use p50, p90, p99, and maximum where sample size permits. Model warm-up and loading are excluded
from warm inference latency but cold-start behavior remains visible in runtime reports. Synchronous
runs are labeled non-real-time. Synthetic and blank replay results are plumbing evidence only.

## Validation gates

A deployment-profile report passes only when all applicable gates pass:

1. Control cadence reaches at least 95% of its configured rate.
2. At least 99% of resolved commands are acknowledged by the sink.
3. Inference has no generation failures.
4. Applicable actions meet the profile's freshness requirement.
5. The simulator/robot adapter remains healthy.
6. Task success is separately reported and never inferred from liveness.

## Current status

The standalone runtime, Python bridge, safety path, all three stitching policies, replay, and LIBERO
closed loop are implemented. The checkpoint-compatible LIBERO profile has one successful task-0
rollout with full delivery and freshness. RTC demonstrates substantially smaller typical seams but
has not yet preserved task success on that checkpoint.

Integration with external camera transport and physical robot adapters is intentionally deferred.

## References

- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
- [LIBERO](https://libero-project.github.io/)
