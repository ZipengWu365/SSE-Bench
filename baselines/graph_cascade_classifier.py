from __future__ import annotations

import argparse
import json
from pathlib import Path

from xgboost import XGBClassifier

from baselines.common import DEFAULT_EVENTS_PATH
from evaluation.metrics import calibration_metrics, classification_metrics
from features.graph_features import GRAPH_FEATURE_COLUMNS, build_graph_feature_frame
from schema.event import load_events_jsonl


def _prepare_split_frames(frame):
    signal = frame[frame["graph_has_signal"] > 0].copy()
    if signal.empty:
        raise ValueError(
            "No graph features found. This baseline requires `event.cascade_graph` or graph summary statistics in `event.metadata`."
        )
    train = signal[signal["split"] == "train"].copy()
    val = signal[signal["split"] == "val"].copy()
    test = signal[signal["split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Insufficient graph-aware split coverage. Found train={len(train)}, val={len(val)}, test={len(test)} events with graph signal."
        )
    return train, val, test


def _prepare_features(train_frame, val_frame, test_frame):
    usable_columns = [
        column
        for column in GRAPH_FEATURE_COLUMNS
        if any(frame[column].notna().any() for frame in (train_frame, val_frame, test_frame))
    ]
    if not usable_columns:
        raise ValueError("Graph signal exists but no usable numeric graph columns were found for training.")

    medians = train_frame[usable_columns].median(numeric_only=True).fillna(0.0)
    train_x = train_frame[usable_columns].fillna(medians).fillna(0.0).to_numpy(dtype=float)
    val_x = val_frame[usable_columns].fillna(medians).fillna(0.0).to_numpy(dtype=float)
    test_x = test_frame[usable_columns].fillna(medians).fillna(0.0).to_numpy(dtype=float)
    return usable_columns, train_x, val_x, test_x


def run_baseline(events_path: str | Path = DEFAULT_EVENTS_PATH) -> dict[str, object]:
    events = load_events_jsonl(Path(events_path))
    frame = build_graph_feature_frame(events)
    train_frame, val_frame, test_frame = _prepare_split_frames(frame)
    feature_columns, train_x, val_x, test_x = _prepare_features(train_frame, val_frame, test_frame)

    train_y = train_frame["is_sse"].to_numpy(dtype=int)
    val_y = val_frame["is_sse"].to_numpy(dtype=int)
    test_y = test_frame["is_sse"].to_numpy(dtype=int)
    positive = max(int(train_y.sum()), 1)
    negative = max(int((1 - train_y).sum()), 1)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=negative / positive,
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    probabilities = model.predict_proba(test_x)[:, 1]

    return {
        "features": feature_columns,
        "graph_signal_events": int((frame["graph_has_signal"] > 0).sum()),
        "train_events": int(len(train_frame)),
        "val_events": int(len(val_frame)),
        "test_events": int(len(test_frame)),
        "classification": classification_metrics(test_y, probabilities, threshold=0.5),
        "calibration": calibration_metrics(test_y, probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a graph-aware cascade classification baseline.")
    parser.add_argument(
        "--events-path",
        default=str(DEFAULT_EVENTS_PATH),
        help="Path to processed Event objects (jsonl.gz).",
    )
    args = parser.parse_args()
    output = run_baseline(events_path=Path(args.events_path))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
