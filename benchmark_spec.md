# SSE-Bench Benchmark Specification v0.1

## Objective

SSE-Bench evaluates whether models can detect an upcoming super-spreading event from only early observable signals. The benchmark is designed for event-level forecasting rather than single-post popularity prediction.

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

Input:

* early observations up to time window `w`

Output:

```text
time_to_sse_minutes
```

Metrics:

* `MAE`
* `Concordance index`

v0.1 evaluation note:

* this task is evaluated only on SSE-positive events with observed onset times

### Task 3: Final Cascade Size

Input:

* early observations up to time window `w`

Output:

```text
log(final_cascade_size)
```

Metrics:

* `RMSE`
* `Spearman correlation`

### Task 4: Historical Analogue Retrieval

Input:

* early observations up to time window `w`

Output:

```text
Top-k similar historical events
```

Metrics:

* `Recall@k`
* `NDCG`

v0.1 retrieval oracle:

* relevant analogues are defined by full-trajectory nearest neighbours among historical SSE-positive events
* baselines retrieve using only early-window trajectories and are scored against that full-trajectory oracle

## Event Object

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

* `engagement_series` stores cumulative engagement over fixed time slices
* `sentiment_series` may be dynamic or static depending on the raw dataset
* `cascade_graph` is optional because many public datasets lack explicit diffusion graphs
* `metadata` preserves adapter-specific fields without breaking the shared schema

## Chronological Protocol

SSE-Bench uses strict chronological evaluation.

Rules:

1. all training events must occur no later than validation events
2. all validation events must occur no later than test events
3. models may only use observations available before the prediction horizon
4. labels are created from full trajectories, but features must be truncated to the chosen observation window

The default split policy in v0.1 is:

* `train`: first 70% of events by publication time
* `val`: next 15%
* `test`: final 15%

## Observation Windows

The benchmark API allows arbitrary windows, but each adapter only exposes windows consistent with its temporal resolution.

Default windows:

* `20 minutes`
* `1 hour`
* `6 hours`
* `24 hours`

The UCI adapter uses 20-minute slices, so `20m`, `1h`, `6h`, and `24h` map to `1`, `3`, `18`, and `72` bins respectively.

## SSE Label Generation

The default v0.1 label is cohort-relative.

For each dataset cohort:

```text
cohort = dataset + platform + topic
```

Compute:

```text
size_threshold = quantile(final_cascade_size, q)
growth_threshold = mean(positive_growth) + sigma * std(positive_growth)
```

with defaults:

```text
q = 0.99
sigma = 3.0
```

Then define:

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

This keeps the onset definition dynamic while preserving the final-scale requirement.

## UCI Adapter Details

v0.1 includes a concrete adapter for:

* `News Popularity in Multiple Social Media Platforms`

Official source:

* [UCI dataset page](https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms)

Adapter mapping:

* one event per `(platform, IDLink)` pair
* `News_Final.csv` provides metadata and final popularity
* `Facebook_*`, `GooglePlus_*`, and `LinkedIn_*` tables provide 144 cumulative feedback bins
* negative counts in the raw time series are treated as missing, then prefix-filled with `0` and converted into monotone cumulative trajectories
* title/headline sentiment is represented as a static one-step sentiment series and also copied into `metadata`
* the adapter emits data-quality and label-sensitivity artifacts so proxy-label assumptions are auditable

## Processed Artifacts

The preparation pipeline emits:

* `data/processed/uci_news_sse/events.jsonl.gz`
* `data/processed/uci_news_sse/event_index.parquet`
* `artifacts/uci_news_summary.json`
* `artifacts/uci_news_cohort_thresholds.csv`
* `artifacts/uci_news_data_quality.csv`
* `artifacts/uci_news_label_sensitivity.csv`
* `artifacts/uci_news_baselines.json`

`events.jsonl.gz` contains the full schema objects. `event_index.parquet` stores scalar metadata and precomputed early-window features for fast experimentation.

## Baselines In This Repo

Implemented:

* early-growth heuristic
* XGBoost classifier for Task 1
* XGBoost regressor for Task 2
* XGBoost regressor for Task 3
* trajectory-retrieval baseline for Task 4

Planned:

* temporal neural baselines for sequence modeling
* retrieval baselines with richer distance functions
* graph-aware baselines once a cascade-graph adapter is added

## Validity Notes

The UCI adapter is intentionally framed as a proxy-SSE benchmark component. It supports early event-level forecasting and reproducible temporal evaluation, but it does not yet provide explicit diffusion graphs or direct cross-community spread labels. Claims about digital super-spreading should therefore be interpreted as operational rather than fully graph-grounded in v0.1.
