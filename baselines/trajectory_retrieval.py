from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from baselines.common import DEFAULT_EVENTS_PATH
from evaluation.metrics import retrieval_metrics
from features.early_observations import window_to_steps
from retrieval.trajectory import trajectory_matrix
from schema.event import load_events_jsonl


def _infer_slice_minutes(events, default: int = 20) -> int:
    for event in events:
        value = event.metadata.get("slice_minutes")
        if value is None:
            continue
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            continue
        if minutes > 0:
            return minutes
    return default


def run_baseline(
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    window_minutes: int = 360,
    k: int = 10,
    oracle_k: int = 25,
    max_queries: int | None = None,
    seed: int = 42,
    slice_minutes: int | None = None,
) -> dict[str, object]:
    events = load_events_jsonl(Path(events_path))
    candidates = [event for event in events if event.is_sse and event.split in {"train", "val"}]
    queries = [event for event in events if event.is_sse and event.split == "test"]

    if max_queries is not None and len(queries) > max_queries:
        rng = random.Random(seed)
        queries = rng.sample(queries, max_queries)
        queries.sort(key=lambda event: event.event_id)

    if not queries or not candidates:
        raise ValueError("Trajectory retrieval requires positive SSE queries and historical candidates.")

    effective_slice_minutes = slice_minutes or _infer_slice_minutes(events)
    early_steps = window_to_steps(window_minutes, slice_minutes=effective_slice_minutes)
    full_steps = max(len(event.engagement_series) for event in (*queries, *candidates))
    query_early = trajectory_matrix(queries, steps=early_steps)
    query_full = trajectory_matrix(queries, steps=full_steps)
    candidate_early = trajectory_matrix(candidates, steps=early_steps)
    candidate_full = trajectory_matrix(candidates, steps=full_steps)

    predicted_scores = query_early @ candidate_early.T
    oracle_scores = query_full @ candidate_full.T
    truth = np.zeros_like(predicted_scores)

    for row_index in range(len(queries)):
        oracle_row = oracle_scores[row_index]
        top_indices = np.argsort(oracle_row)[::-1][:oracle_k]
        truth[row_index, top_indices] = oracle_row[top_indices]

    metrics = retrieval_metrics(truth, predicted_scores, k=k)
    return {
        "window_minutes": window_minutes,
        "slice_minutes": effective_slice_minutes,
        "query_events": len(queries),
        "candidate_events": len(candidates),
        "k": k,
        "oracle_k": oracle_k,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trajectory-retrieval baseline on SSE-positive events.")
    parser.add_argument(
        "--events-path",
        default=str(DEFAULT_EVENTS_PATH),
        help="Path to processed Event objects.",
    )
    parser.add_argument("--window-minutes", type=int, default=360, help="Observation window in minutes.")
    parser.add_argument("--k", type=int, default=10, help="Retrieval cutoff for reported metrics.")
    parser.add_argument("--oracle-k", type=int, default=25, help="Oracle full-trajectory neighbour set size.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on the number of SSE-positive test queries.",
    )
    parser.add_argument(
        "--slice-minutes",
        type=int,
        default=None,
        help="Override the engagement-series slice width in minutes. Defaults to adapter metadata or 20.",
    )
    args = parser.parse_args()

    output = run_baseline(
        events_path=Path(args.events_path),
        window_minutes=args.window_minutes,
        k=args.k,
        oracle_k=args.oracle_k,
        max_queries=args.max_queries,
        slice_minutes=args.slice_minutes,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
