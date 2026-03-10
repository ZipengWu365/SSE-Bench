from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import calibration_metrics, classification_metrics
from features.early_observations import build_feature_frame
from schema.event import load_events_jsonl


class EarlyGrowthHeuristic:
    def __init__(self) -> None:
        self.threshold_ = 0.5

    def fit(self, train_frame) -> None:
        score = self._score(train_frame)
        thresholds = np.linspace(0.05, 0.95, 19)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            metrics = classification_metrics(train_frame["is_sse"], score, threshold=threshold)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = float(threshold)
        self.threshold_ = best_threshold

    def _score(self, frame) -> np.ndarray:
        growth_rank = frame["growth_rate"].rank(pct=True).to_numpy(dtype=float)
        increment_rank = frame["max_increment"].rank(pct=True).to_numpy(dtype=float)
        burst_rank = frame["burstiness"].rank(pct=True).to_numpy(dtype=float)
        return np.clip(0.5 * growth_rank + 0.35 * increment_rank + 0.15 * burst_rank, 0.0, 1.0)

    def predict_proba(self, frame) -> np.ndarray:
        return self._score(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the early-growth heuristic baseline.")
    parser.add_argument(
        "--events-path",
        default="data/processed/uci_news_sse/events.jsonl.gz",
        help="Path to processed Event objects.",
    )
    parser.add_argument("--window-minutes", type=int, default=360, help="Observation window in minutes.")
    args = parser.parse_args()

    events = load_events_jsonl(Path(args.events_path))
    frame = build_feature_frame(events, window_minutes=args.window_minutes)
    train_frame = frame[frame["split"] == "train"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    model = EarlyGrowthHeuristic()
    model.fit(train_frame)
    probabilities = model.predict_proba(test_frame)

    output = {
        "window_minutes": args.window_minutes,
        "threshold": model.threshold_,
        "classification": classification_metrics(test_frame["is_sse"], probabilities, threshold=model.threshold_),
        "calibration": calibration_metrics(test_frame["is_sse"], probabilities),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
