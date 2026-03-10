from __future__ import annotations

import argparse
import json
from pathlib import Path

from xgboost import XGBClassifier

from evaluation.metrics import calibration_metrics, classification_metrics
from features.early_observations import build_feature_frame
from schema.event import load_events_jsonl


FEATURE_COLUMNS = [
    "observed_size",
    "growth_rate",
    "max_increment",
    "mean_increment",
    "burstiness",
    "acceleration",
    "sentiment_mean",
    "sentiment_abs",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XGBoost SSE classification baseline.")
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
    val_frame = frame[frame["split"] == "val"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    train_x = train_frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    val_x = val_frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    test_x = test_frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    train_y = train_frame["is_sse"].to_numpy(dtype=int)
    val_y = val_frame["is_sse"].to_numpy(dtype=int)
    test_y = test_frame["is_sse"].to_numpy(dtype=int)

    positive = max(int(train_y.sum()), 1)
    negative = max(int((1 - train_y).sum()), 1)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=negative / positive,
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)

    probabilities = model.predict_proba(test_x)[:, 1]
    output = {
        "window_minutes": args.window_minutes,
        "features": FEATURE_COLUMNS,
        "classification": classification_metrics(test_y, probabilities, threshold=0.5),
        "calibration": calibration_metrics(test_y, probabilities),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
