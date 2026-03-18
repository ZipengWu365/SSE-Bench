# SSE-Bench

## Super-Spreading Event Forecasting Benchmark

SSE-Bench studies early warning for **super-spreading social events**: can we tell, from only the first slice of signals, whether an emerging event will transition from ordinary diffusion into an explosive cascade?

For a cross-disciplinary explanation of the benchmark logic and the four tasks, see `docs/benchmark_logic.md`.

## Current Status

SSE-Bench is currently a **v0.1 proxy-SSE benchmark** with working local pipelines for:

* `uci_news_sse`
* `infopath_sse`
* `synthetic_sse` (synthetic control track; graph-grounded; optional)

It also includes:

* cross-dataset Task 1 transfer evaluation over shared early-window features
* paper-ready Markdown/LaTeX tables built from tracked benchmark artifacts
* static benchmark plots for cross-window inspection

The current repo state is a bridge toward **v0.2 multi-dataset SSE benchmarking** with graph-aware support and a second cascade-oriented adapter already wired into the benchmark workflow.

This means:

* v0.1 is real and runnable.
* v0.1 should be described honestly as an **operational proxy** for SSE, not yet a graph-grounded gold standard.
* the current development direction is to add a second dataset and graph-aware baselines without changing the event-level benchmark identity.

## Benchmark Tasks

SSE-Bench defines four linked tasks around the same event object.

1. `SSE Detection`
   Predict `P(event becomes SSE)`.
   Metrics: `AUROC`, `AUPRC`, `F1`, calibration metrics.

2. `Time-to-SSE`
   Predict when the event enters the super-spreading phase.
   Metrics: `MAE`, `Concordance index`.

3. `Final Cascade Size`
   Predict `log(final_cascade_size)`.
   Metrics: `RMSE`, `Spearman`.

4. `Historical Analogue Retrieval`
   Retrieve the most similar historical events from early trajectories.
   Metrics: `Recall@k`, `NDCG`.

## SSE Definition (Operational Default)

The default operational SSE label combines size and burstiness:

```text
is_sse = (
  final_cascade_size >= top_q_percentile
  and
  max_growth >= mean_positive_growth + sigma * std_positive_growth
)
```

Defaults:

* `q = 0.99`
* `sigma = 3.0`

Thresholds are computed within a dataset cohort so platform or topic scale does not dominate the label. This is the current **proxy-SSE** definition; future graph-aware adapters may add structural criteria such as cross-community spread or cascade depth.

## Event Object

All adapters map raw data into a unified event representation.

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

The benchmark is event-centric rather than post-centric. The goal is not to rank isolated posts, but to understand whether an unfolding event is entering a super-spreading regime.

## What Is Implemented

Current repo contents include:

* event schema and serialization helpers
* a working UCI dataset adapter and preprocessing pipeline
* a working InfoPath cascade adapter and preprocessing pipeline
* (optional) a synthetic graph-grounded control track adapter for sanity checks
* early-warning baselines for Tasks 1-4 on the current proxy dataset
* a graph-aware Task 1 classifier for datasets that expose cascade structure
* a cross-dataset Task 1 transfer script for feature-intersection experiments
* diagnostics for data quality, label sensitivity, and baseline results
* paper-ready table export and static plot generation from benchmark artifacts
* all-dataset orchestration entrypoints
* pytest coverage for schema, graph features, retrieval, and orchestration logic
* tracked validity notes for publication-safe claim boundaries

Current artifacts are centered on:

* UCI data preparation: `python -m scripts.prepare_uci_news`
* InfoPath data preparation: `python -m scripts.prepare_infopath`
* UCI task suite: `python -m scripts.run_uci_benchmark_suite`
* InfoPath task suite: `python -m scripts.run_infopath_benchmark_suite`

All-dataset wrappers are also available:

```bash
python -m scripts.prepare_all_datasets
python -m scripts.run_all_benchmark_suites
python -m scripts.run_window_sweep --datasets synthetic infopath uci --windows 20 60 360 1440
python -m scripts.build_benchmark_overview
python -m scripts.build_benchmark_plots
python -m scripts.build_paper_tables
```

Otherwise, continue using the per-dataset entrypoints above.

## Quickstart

Prepare the current UCI benchmark artifacts:

```bash
python -m scripts.prepare_uci_news
```

Prepare the current InfoPath benchmark artifacts:

```bash
python -m scripts.prepare_infopath --stream-remote
```

Prepare the synthetic control track (if present):

```bash
python -m scripts.prepare_synthetic_sse
```

Run the current UCI baseline suite:

```bash
python -m scripts.run_uci_benchmark_suite --window-minutes 360
```

Run the current InfoPath baseline suite:

```bash
python -m scripts.run_infopath_benchmark_suite --window-minutes 360
```

Run the synthetic baseline suite (if present):

```bash
python -m scripts.run_synthetic_benchmark_suite --window-minutes 360
```

Run cross-dataset transfer on the shared feature set:

```bash
python -m scripts.run_cross_dataset_transfer --index-paths data/processed/uci_news_sse/event_index.parquet data/processed/infopath_sse/event_index.parquet data/processed/synthetic_sse/event_index.parquet --windows 60 360 1440
```

Run individual baselines:

```bash
python -m baselines.early_growth --window-minutes 360
python -m baselines.xgboost_baseline --window-minutes 360
python -m baselines.time_to_sse_regression --window-minutes 360
python -m baselines.final_size_regression --window-minutes 360
python -m baselines.trajectory_retrieval --window-minutes 360
python -m baselines.graph_cascade_classifier --events-path data/processed/infopath_sse/events.jsonl.gz
```

Run the test suite:

```bash
python -m pytest
```

Export paper tables and static plots:

```bash
python -m scripts.build_paper_tables
python -m scripts.build_benchmark_plots
```

## Validity Boundary

What is currently safe to claim:

* SSE-Bench is a reproducible **event-level early forecasting benchmark**.
* The current UCI implementation supports strict chronological evaluation and multiple downstream tasks.
* The current SSE label is an operational proxy based on trajectory scale and growth bursts.
* The synthetic control track (when enabled) is useful for graph-aware sanity checks because it has known diffusion structure by construction.
* cross-dataset transfer can be reported when the explicit transfer protocol and shared feature intersection are stated.

What is **not** safe to claim yet:

* that v0.1 fully captures graph-grounded super-spreading
* that the benchmark is already multi-domain or multi-platform in a publication-grade sense
* that the current proxy label is equivalent to a gold-standard diffusion-phase-transition label
* that performance on the synthetic control track implies real-world robustness
* that a single cross-dataset transfer result establishes broad external validity

For more detail, see `docs/validity_notes.md` and `docs/benchmark_logic.md`.

## Citation

```bibtex
@misc{ssebench2026,
  title={SSE-Bench: Super-Spreading Event Forecasting Benchmark},
  year={2026}
}
```
