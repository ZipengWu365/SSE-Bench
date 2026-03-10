# SSE-Bench

## Super-Spreading Event Forecasting Benchmark

Predict whether an emerging social event will become a **super-spreading event (SSE)** using only early observations.

SSE-Bench studies phase transitions in social diffusion systems: when ordinary propagation turns into explosive cascades. The benchmark is event-centric, not post-centric. The core question is whether an emerging event will cross into a super-spreading regime.

## Tasks

SSE-Bench v0.1 defines four benchmark tasks.

1. `SSE Detection`: predict `P(event becomes SSE)`.
2. `Time-to-SSE`: predict the onset time of the super-spreading phase.
3. `Final Cascade Size`: predict `log(final_cascade_size)`.
4. `Historical Analogue Retrieval`: retrieve the most similar past events from early trajectories.

Default evaluation metrics:

* Classification: `AUROC`, `AUPRC`, `F1`
* Regression: `RMSE`, `Spearman`
* Time-to-event: `MAE`, `Concordance index`
* Retrieval: `Recall@k`, `NDCG`
* Calibration: `Brier score`, `Expected Calibration Error`

## SSE Definition

An event is an SSE when it satisfies both a size criterion and a burst criterion.

```text
final_size >= top_q_percentile
and
max_growth >= mean_positive_growth + 3 * std_positive_growth
```

The default `q` is `99%`. Thresholds are computed within a dataset cohort so that topic and platform scale differences do not dominate the label.

The framework also leaves room for future structural SSE definitions, such as cross-community spread or graph-depth transitions.

## Event Schema

Each adapter maps raw data into a unified `Event` object.

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

## Implemented In v0.1

This repository now includes:

* `benchmark_spec.md` with the benchmark protocol
* `schema/event.py` with serialization and validation helpers
* a UCI adapter for `News Popularity in Multiple Social Media Platforms`
* preprocessing code that downloads raw data and emits unified `Event` objects
* feature extraction utilities
* baseline code for an early-growth heuristic and an XGBoost classifier
* evaluation helpers for classification, regression, retrieval, and calibration

## Dataset Adapter In v0.1

The first concrete adapter uses the official UCI dataset:

`News Popularity in Multiple Social Media Platforms`

Source:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms)

Adapter decisions:

* each `(platform, news item)` pair is treated as one event
* 144 cumulative engagement bins at 20-minute resolution define the trajectory
* default observation windows are `20m`, `1h`, `6h`, and `24h`
* title and headline sentiment are stored as a length-1 static sentiment series
* strict chronological `train/val/test` splits are assigned from publication time

## Quickstart

Prepare the UCI dataset and generate processed SSE-Bench artifacts:

```bash
python -m scripts.prepare_uci_news
```

Run the simple heuristic baseline on the processed events:

```bash
python -m baselines.early_growth --window-minutes 360
```

Run the XGBoost baseline:

```bash
python -m baselines.xgboost_baseline --window-minutes 360
```

## Repository Layout

```text
sse-bench/
|- README.md
|- benchmark_spec.md
|- pyproject.toml
|- artifacts/
|- baselines/
|- datasets/
|  `- adapters/
|- evaluation/
|- features/
|- retrieval/
|- schema/
|- scripts/
`- data/
```

## Research Direction

SSE-Bench is aimed at studying digital super-spreading dynamics:

* early warning signals of diffusion explosions
* onset timing of super-spreading behavior
* uncertainty-aware forecasting
* retrieval of historical analogues

Longer term, the same benchmark format can extend to intervention evaluation, counterfactual analysis, and simulation-based policy studies.

## Citation

```bibtex
@misc{ssebench2026,
  title={SSE-Bench: Super-Spreading Event Forecasting Benchmark},
  year={2026}
}
```
