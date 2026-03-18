from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.early_growth import run_baseline as run_early_growth
from baselines.final_size_regression import run_baseline as run_final_size_regression
from baselines.graph_cascade_classifier import run_baseline as run_graph_classification
from baselines.time_to_sse_regression import run_baseline as run_time_to_sse_regression
from baselines.trajectory_retrieval import run_baseline as run_trajectory_retrieval
from baselines.xgboost_baseline import run_baseline as run_xgboost_classification


DEFAULT_INDEX_PATH = Path("data/processed/synthetic_sse/event_index.parquet")
DEFAULT_EVENTS_PATH = Path("data/processed/synthetic_sse/events.jsonl.gz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic control-track baseline suite for all implemented SSE-Bench tasks."
    )
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to processed event index parquet.")
    parser.add_argument("--events-path", default=str(DEFAULT_EVENTS_PATH), help="Path to processed Event objects.")
    parser.add_argument("--window-minutes", type=int, default=360, help="Observation window in minutes.")
    parser.add_argument("--retrieval-k", type=int, default=10, help="Retrieval cutoff for reported metrics.")
    parser.add_argument("--oracle-k", type=int, default=25, help="Full-trajectory oracle neighbourhood size.")
    parser.add_argument(
        "--output-path",
        default="artifacts/synthetic_sse_baselines.json",
        help="Where to write the aggregated baseline report.",
    )
    args = parser.parse_args()

    index_path = Path(args.index_path)
    events_path = Path(args.events_path)
    report = {
        "dataset": "synthetic_sse_control",
        "window_minutes": args.window_minutes,
        "task_1_heuristic": run_early_growth(index_path=index_path, window_minutes=args.window_minutes),
        "task_1_xgboost": run_xgboost_classification(index_path=index_path, window_minutes=args.window_minutes),
        "task_1_graph": run_graph_classification(events_path=events_path),
        "task_2_time_to_sse": run_time_to_sse_regression(index_path=index_path, window_minutes=args.window_minutes),
        "task_3_final_size": run_final_size_regression(index_path=index_path, window_minutes=args.window_minutes),
        "task_4_retrieval": run_trajectory_retrieval(
            events_path=events_path,
            window_minutes=args.window_minutes,
            k=args.retrieval_k,
            oracle_k=args.oracle_k,
        ),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
