from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.common import DEFAULT_INDEX_PATH, load_index
from evaluation.metrics import calibration_metrics, classification_metrics


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


def _frame_for_window(index_frame, window_minutes: int):
    prefix = f"w{window_minutes}_"
    columns = {
        f"{prefix}growth_rate": "growth_rate",
        f"{prefix}max_increment": "max_increment",
        f"{prefix}burstiness": "burstiness",
    }
    frame = index_frame[list(columns) + ["split", "is_sse"]].rename(columns=columns).copy()
    return frame


def run_baseline(index_path: str | Path = DEFAULT_INDEX_PATH, window_minutes: int = 360) -> dict[str, object]:
    index_frame = load_index(index_path)
    frame = _frame_for_window(index_frame, window_minutes=window_minutes)
    train_frame = frame[frame["split"] == "train"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    model = EarlyGrowthHeuristic()
    model.fit(train_frame)
    probabilities = model.predict_proba(test_frame)

    return {
        "window_minutes": window_minutes,
        "threshold": model.threshold_,
        "classification": classification_metrics(test_frame["is_sse"], probabilities, threshold=model.threshold_),
        "calibration": calibration_metrics(test_frame["is_sse"], probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the early-growth heuristic baseline.")
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to processed event index parquet.",
    )
    parser.add_argument("--window-minutes", type=int, default=360, help="Observation window in minutes.")
    args = parser.parse_args()

    output = run_baseline(index_path=Path(args.index_path), window_minutes=args.window_minutes)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
