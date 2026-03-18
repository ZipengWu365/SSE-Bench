# SSE-Bench Benchmark Logic (Cross-Disciplinary Overview)

SSE-Bench is an event-level early-warning benchmark for social diffusion.

The core question is not "which post gets more likes", but:

Given only early, time-available signals about an emerging event, can we forecast whether it will enter a rare **super-spreading regime** (an explosive diffusion phase), and characterize that regime?

## What Counts As an "Event"

An "event" is the unit of prediction. Different datasets can define events differently:

* A news URL's multi-platform engagement trajectory (proxy eventization).
* A diffusion cascade over a network (cascade eventization).
* A simulated cascade with known structure (synthetic control track).

All datasets must map into the same `Event` object contract (see `benchmark_spec.md`).

## The Four Tasks (One Event, Four Questions)

All tasks share the same input constraint: models only see information up to an observation window `w` (no future leakage).

1. Task 1: SSE Detection
Predict whether the event will become an SSE.
Use case: early triage and resource allocation.

2. Task 2: Time-to-SSE
Predict when the event enters the SSE phase.
Use case: "how much time do we have" for interventions or monitoring.

3. Task 3: Final Cascade Size
Predict the eventual magnitude of the cascade.
Use case: impact estimation and planning.

4. Task 4: Historical Analogue Retrieval
Retrieve similar past events based on early trajectories.
Use case: interpretability-by-comparison ("what did similar cases do next").

## What "SSE" Means In v0.1

In v0.1, `is_sse` is an operational proxy combining:

* unusually large final size (relative to a cohort), and
* an unusually large early burst (relative to positive-growth statistics).

This is a pragmatic starting point. It should not be marketed as a definitive, graph-grounded phase-transition label.

## Why Include a Synthetic Control Track

Real datasets have missingness, measurement noise, and partial observability. A synthetic control track is useful because:

* the diffusion structure can be explicit and internally consistent by construction,
* labels can reflect the simulated mechanism (ground truth for that mechanism),
* it is a low-cost way to sanity-check evaluation code and graph-aware baselines.

Limits:

* synthetic performance does not imply real-world robustness,
* synthetic diffusion mechanisms are not a substitute for real structural evidence.

## How to Read Results

Across tasks and datasets, interpret results in three layers:

* Predictive performance within a dataset (chronological split).
* Sensitivity to the observation window `w` (early-warning behavior).
* Stability across datasets (requires explicit cross-dataset transfer evaluation; planned/optional).

For claim boundaries and recommended language, see `docs/validity_notes.md`.

