from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from baselines.common import DEFAULT_INDEX_PATH, load_index
from evaluation.metrics import time_to_event_metrics
from features import window_feature_columns


def run_baseline(index_path: str | Path = DEFAULT_INDEX_PATH, window_minutes: int = 360) -> dict[str, object]:
    frame = load_index(index_path)
    positive = frame[frame["time_to_sse_minutes"].notna()].copy()
    feature_columns = window_feature_columns(window_minutes)

    train_frame = positive[positive["split"] == "train"].copy()
    val_frame = positive[positive["split"] == "val"].copy()
    test_frame = positive[positive["split"] == "test"].copy()

    train_x = train_frame[feature_columns].to_numpy(dtype=float)
    val_x = val_frame[feature_columns].to_numpy(dtype=float)
    test_x = test_frame[feature_columns].to_numpy(dtype=float)

    train_y = np.log1p(train_frame["time_to_sse_minutes"].to_numpy(dtype=float))
    val_y = np.log1p(val_frame["time_to_sse_minutes"].to_numpy(dtype=float))
    test_y = test_frame["time_to_sse_minutes"].to_numpy(dtype=float)

    model = XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=3,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)

    prediction = np.expm1(model.predict(test_x))
    return {
        "window_minutes": window_minutes,
        "features": feature_columns,
        "training_events": int(len(train_frame)),
        "validation_events": int(len(val_frame)),
        "test_events": int(len(test_frame)),
        "time_to_event": time_to_event_metrics(test_y, prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the time-to-SSE regression baseline.")
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
