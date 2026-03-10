from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from baselines.common import DEFAULT_INDEX_PATH, load_index
from evaluation.metrics import regression_metrics
from features import window_feature_columns


def run_baseline(index_path: str | Path = DEFAULT_INDEX_PATH, window_minutes: int = 360) -> dict[str, object]:
    frame = load_index(index_path)
    feature_columns = window_feature_columns(window_minutes)

    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    train_x = train_frame[feature_columns].to_numpy(dtype=float)
    val_x = val_frame[feature_columns].to_numpy(dtype=float)
    test_x = test_frame[feature_columns].to_numpy(dtype=float)

    train_y = np.log1p(train_frame["final_cascade_size"].to_numpy(dtype=float))
    val_y = np.log1p(val_frame["final_cascade_size"].to_numpy(dtype=float))
    test_y = np.log1p(test_frame["final_cascade_size"].to_numpy(dtype=float))

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)

    prediction = model.predict(test_x)
    raw_truth = test_frame["final_cascade_size"].to_numpy(dtype=float)
    raw_pred = np.expm1(prediction)
    return {
        "window_minutes": window_minutes,
        "features": feature_columns,
        "log_regression": regression_metrics(test_y, prediction),
        "raw_rmse": float(np.sqrt(np.mean((raw_truth - raw_pred) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final-cascade-size regression baseline.")
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
