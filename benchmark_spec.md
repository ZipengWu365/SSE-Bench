# SSE-Bench Benchmark Specification v0.1 -> v0.2 Transition

## Objective

SSE-Bench evaluates whether a model can identify, from only early observations, that an emerging social event will become a super-spreading event. The benchmark is event-level by construction and is intended for early warning, timing, impact estimation, and historical analogue retrieval.

## Scope By Phase

### v0.1

Stable and runnable:

* three dataset paths (`uci_news_sse`, `infopath_sse`, `synthetic_sse`)
* strict chronological evaluation
* Tasks 1-4 defined and runnable on the current adapter
* diagnostics for label sensitivity and data quality
* a graph-aware Task 1 baseline for adapters with cascade structure
* optional cross-dataset transfer evaluation over shared feature intersections
* reproducible paper-table and plotting exports from tracked artifacts

### v0.2

Target direction:

* one more real dataset with stronger structural diffusion evidence
* richer graph-aware baselines and graph-derived event signals
* stronger construct validity for the term "super-spreading"
* stronger external-validity evidence beyond synthetic and current proxy datasets

## Core Tasks

### Task 1: SSE Detection

Input:

* early observations up to time window `w`

Output:

```text
P(event becomes SSE)
```

Metrics:

* `AUROC`
* `AUPRC`
* `F1`
* optional calibration metrics

### Task 2: Time-to-SSE

Output:

```text
time_to_sse_minutes
```

Metrics:

* `MAE`
* `Concordance index`

v0.1 note:

* evaluated only on SSE-positive events with defined onset times

### Task 3: Final Cascade Size

Output:

```text
log(final_cascade_size)
```

Metrics:

* `RMSE`
* `Spearman correlation`

### Task 4: Historical Analogue Retrieval

Output:

```text
Top-k similar historical events
```

Metrics:

* `Recall@k`
* `NDCG`

v0.1 note:

* retrieval relevance is currently operationalized by full-trajectory similarity, not human-validated analogue labels

## Event Object Contract

All datasets are normalized into the following object shape.

```python
class Event:
    event_id: str
    dataset: str
    platform: str
    topic: str | None
    start_time: datetime
    engagement_series: list[int]
    sentiment_series: list[float] | None
    cascade_graph: dict | None
    is_sse: bool
    time_to_sse_minutes: int | None
    final_cascade_size: int
    split: str | None
    metadata: dict
```

Interpretation:

* `engagement_series` stores cumulative observations over fixed time slices
* `sentiment_series` may be static or dynamic depending on source data
* `cascade_graph` is optional in v0.1 but expected to become more important in v0.2
* `metadata` is the escape hatch for adapter-specific fields and diagnostics

## Evaluation Protocol

SSE-Bench uses strict chronological evaluation.

Rules:

1. train events occur no later than validation events
2. validation events occur no later than test events
3. features must be available at prediction time
4. labels may use full trajectories, but model inputs must be truncated to the chosen observation window

Default windows:

* `20 minutes`
* `1 hour`
* `6 hours`
* `24 hours`

Each adapter can expose only the windows supported by its temporal resolution.

## Cross-Dataset Transfer (Optional Experimental Track)

Cross-dataset transfer evaluation is implemented as an optional analysis for assessing whether early-warning signals learned from one domain generalize to another.

Current protocol:

* train on dataset A (chronological split within A)
* select hyperparameters using validation within A
* test on dataset B using only features available within the chosen observation window `w`

Current constraints:

* transfer should be reported only when adapters expose compatible feature sets, or when a documented feature intersection is used
* synthetic control tracks can be used to validate mechanics, but do not count as real-domain transfer evidence
* transfer results should be interpreted as stress tests for portability, not as standalone proof of generalization

## SSE Labeling Policy

The default v0.1 label is cohort-relative.

```text
size_threshold = quantile(final_cascade_size, q)
growth_threshold = mean(positive_growth) + sigma * std(positive_growth)
```

Defaults:

```text
q = 0.99
sigma = 3.0
```

Then:

```text
is_sse = (
    final_cascade_size >= size_threshold
    and
    max(slice_to_slice_growth) >= growth_threshold
)
```

For SSE-positive events:

```text
time_to_sse_minutes = first bin where slice_to_slice_growth >= growth_threshold
```

This should be described as an **operational proxy** unless the adapter also supports stronger structural evidence.

### Synthetic Control Track Labeling

For the synthetic control track, labels come from the generator and are therefore *ground-truth with respect to the synthetic mechanism*:

* `is_sse` reflects the simulated regime switch into a super-spreading process (as defined by the generator)
* `time_to_sse_minutes` reflects the simulated onset time when the regime switch occurs
* `final_cascade_size` is the realized cascade size from the simulated diffusion process

This track is intended for:

* sanity checking that baselines and evaluation code behave as expected when diffusion structure is known
* controlled comparisons of graph-aware vs non-graph models

This track is not intended as evidence of real-world generalization.

## Adapter Notes

Current adapter paths:

* UCI news popularity adapter
* eventized as platform-specific engagement trajectories
* suitable for early-warning experiments, but not yet graph-grounded
* InfoPath keyword-cascade adapter
* eventized as cascade-level web diffusion trajectories with a proxy `cascade_graph`
* suitable for graph-aware experimentation, while still requiring careful claim discipline because the graph is proxy-constructed

* Synthetic graph-grounded control track
* eventized as simulated cascades with explicit `cascade_graph` including community assignments
* useful for internal validity and debugging, but not a replacement for real diffusion datasets

Graph-aware baselines should consume either `cascade_graph` or graph summary fields in `metadata`.

## Artifacts

Current repo artifacts include:

* processed event objects
* per-dataset index tables
* label threshold summaries
* data-quality diagnostics
* label-sensitivity diagnostics
* per-dataset baseline reports
* per-dataset window-sweep reports
* a cross-dataset benchmark overview
* optional cross-dataset transfer reports
* paper-ready tables exported from the above JSON/Markdown artifacts
* static plot bundles for cross-window comparisons

If all-dataset wrappers are present, they should orchestrate the same per-dataset entrypoints rather than replace them.

## Validity Statement

SSE-Bench v0.1 is valid as a benchmark for **event-level early diffusion forecasting under an operational SSE proxy**. It is not yet valid to market v0.1 as a definitive graph-grounded benchmark of social super-spreading. That stronger claim requires at least one additional adapter with richer structural diffusion evidence and corresponding graph-aware baselines.
