# SSE-Bench Validity Notes

## Why This Note Exists

SSE-Bench is being built in phases. The goal of this note is to make the claim boundary explicit so the project stays scientifically honest while still moving quickly.

## Current Validity Level (v0.1)

What v0.1 supports well:

* reproducible event-level early forecasting
* strict chronological evaluation
* multiple downstream tasks tied to the same event object
* operational SSE labels based on scale and burstiness
* explicit transfer experiments when a documented cross-dataset protocol is run

What v0.1 does not yet fully support:

* strong graph-grounded claims about super-spreading
* cross-community diffusion measurement
* broad multi-dataset generalization claims
* cross-dataset robustness claims beyond the specific evaluated source-target pairs

## How v0.2 Improves Validity

v0.2 should improve validity by adding:

* a second dataset with cascade-oriented structure
* graph-aware event signals and graph-aware baselines
* better separation between proxy labels and stronger structural labels
* a synthetic graph-grounded control track that improves internal validity and helps catch implementation bugs
* cross-dataset transfer evaluation with a documented, leakage-safe protocol

## Claiming Guidance For Papers And Reports

Safe claim style:

* "SSE-Bench is an event-level early warning benchmark with an operational proxy-SSE definition."
* "v0.1 provides a reproducible starting point for studying early diffusion explosions."
* "SSE-Bench includes a synthetic control track with known cascade structure to validate graph-aware evaluation and debug models."

Claims to avoid until graph-grounded coverage is complete:

* "SSE-Bench is already a definitive benchmark for super-spreading dynamics."
* "The current label directly measures real transmission structure."
* "Results already establish cross-domain robustness."

Claims to avoid even when the synthetic control track is enabled:

* "Synthetic-track performance implies real-world performance."
* "Synthetic results establish external validity for social super-spreading."

Claims to avoid even when cross-dataset transfer is reported:

* "The method generalizes across domains/datasets."
* "Results demonstrate cross-platform robustness."
* "A few source-target transfer results establish external validity."
