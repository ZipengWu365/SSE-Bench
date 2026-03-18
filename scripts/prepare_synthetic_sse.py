from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.adapters.synthetic_sse import SyntheticSseConfig, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic graph-grounded SSE control dataset.")
    parser.add_argument("--processed-dir", default="data/processed/synthetic_sse", help="Directory for processed artifacts.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for tracked summaries.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--events", type=int, default=20000, help="Number of synthetic events to generate.")
    parser.add_argument("--sse-probability", type=float, default=0.05, help="Probability an event is a true SSE.")
    parser.add_argument("--slice-minutes", type=int, default=20, help="Observation bin width in minutes.")
    parser.add_argument("--max-observation-bins", type=int, default=72, help="Number of bins to retain (default=24h at 20-min bins).")
    parser.add_argument("--communities", type=int, default=4, help="Number of communities in the synthetic graph.")
    parser.add_argument("--nodes-per-community", type=int, default=250, help="Candidate nodes per community.")
    parser.add_argument("--base-reproduction", type=float, default=0.8, help="Reproduction rate before onset (and for non-SSE).")
    parser.add_argument("--sse-reproduction", type=float, default=1.6, help="Reproduction rate after onset for SSE.")
    parser.add_argument("--base-cross-community-prob", type=float, default=0.02, help="Cross-community edge probability in base regime.")
    parser.add_argument("--sse-cross-community-prob", type=float, default=0.15, help="Cross-community edge probability after onset in SSE.")
    parser.add_argument("--onset-min-minutes", type=int, default=40, help="Minimum SSE onset time in minutes.")
    parser.add_argument("--onset-max-minutes", type=int, default=360, help="Maximum SSE onset time in minutes.")
    parser.add_argument("--max-graph-nodes", type=int, default=64, help="Maximum nodes retained in cascade_graph.")
    parser.add_argument("--max-graph-edges", type=int, default=256, help="Maximum edges retained in cascade_graph.")
    args = parser.parse_args()

    config = SyntheticSseConfig(
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed,
        events=args.events,
        sse_probability=args.sse_probability,
        slice_minutes=args.slice_minutes,
        max_observation_bins=args.max_observation_bins,
        communities=args.communities,
        nodes_per_community=args.nodes_per_community,
        base_reproduction=args.base_reproduction,
        sse_reproduction=args.sse_reproduction,
        base_cross_community_prob=args.base_cross_community_prob,
        sse_cross_community_prob=args.sse_cross_community_prob,
        onset_min_minutes=args.onset_min_minutes,
        onset_max_minutes=args.onset_max_minutes,
        max_graph_nodes=args.max_graph_nodes,
        max_graph_edges=args.max_graph_edges,
    )
    summary = prepare_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

