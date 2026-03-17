from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.adapters.infopath import InfoPathConfig, prepare_dataset


def _normalize_limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or stream and preprocess SNAP InfoPath keyword cascades.")
    parser.add_argument("--raw-dir", default="data/raw/infopath", help="Directory for raw downloads.")
    parser.add_argument("--processed-dir", default="data/processed/infopath_sse", help="Directory for processed artifacts.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for tracked summaries.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local archive.")
    parser.add_argument("--stream-remote", action="store_true", help="Stream the remote archive instead of downloading it first.")
    parser.add_argument("--extract-selected-files", action="store_true", default=True, help="Write selected raw member files to the extract directory.")
    parser.add_argument("--no-extract-selected-files", action="store_true", help="Do not write selected raw member files to disk.")
    parser.add_argument("--max-keyword-files", type=int, default=3, help="Maximum number of keyword files to process. <=0 means unlimited.")
    parser.add_argument("--max-events", type=int, default=15000, help="Maximum number of cascades to process. <=0 means unlimited.")
    parser.add_argument("--size-quantile", type=float, default=0.99, help="Final-size quantile for SSE labels.")
    parser.add_argument("--growth-sigma", type=float, default=3.0, help="Sigma multiplier for burst thresholding.")
    parser.add_argument("--min-topic-events", type=int, default=50, help="Minimum events per topic before using topic-local thresholds.")
    parser.add_argument("--slice-minutes", type=int, default=60, help="Observation bin width in minutes.")
    parser.add_argument("--max-observation-bins", type=int, default=336, help="Maximum number of observation bins to retain.")
    parser.add_argument("--max-graph-nodes", type=int, default=64, help="Maximum number of nodes retained in proxy cascade_graph.")
    parser.add_argument("--max-graph-edges", type=int, default=256, help="Maximum number of edges retained in proxy cascade_graph.")
    args = parser.parse_args()

    config = InfoPathConfig(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        force_download=args.force_download,
        stream_remote=args.stream_remote,
        extract_selected_files=not args.no_extract_selected_files,
        max_keyword_files=_normalize_limit(args.max_keyword_files),
        max_events=_normalize_limit(args.max_events),
        size_quantile=args.size_quantile,
        growth_sigma=args.growth_sigma,
        min_topic_events=args.min_topic_events,
        slice_minutes=args.slice_minutes,
        max_observation_bins=args.max_observation_bins,
        max_graph_nodes=args.max_graph_nodes,
        max_graph_edges=args.max_graph_edges,
    )
    summary = prepare_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
