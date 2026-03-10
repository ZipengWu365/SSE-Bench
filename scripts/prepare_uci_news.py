from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.adapters.uci_news import UciNewsConfig, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess the UCI news popularity dataset.")
    parser.add_argument("--raw-dir", default="data/raw/uci_news", help="Directory for raw downloads.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/uci_news_sse",
        help="Directory for processed event artifacts.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for small tracked summaries.")
    parser.add_argument("--size-quantile", type=float, default=0.99, help="Final-size quantile for SSE labels.")
    parser.add_argument("--growth-sigma", type=float, default=3.0, help="Sigma multiplier for growth bursts.")
    parser.add_argument("--force-download", action="store_true", help="Redownload and re-extract the raw zip.")
    args = parser.parse_args()

    config = UciNewsConfig(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        size_quantile=args.size_quantile,
        growth_sigma=args.growth_sigma,
        force_download=args.force_download,
    )
    summary = prepare_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
