from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluation.metrics import calibration_metrics, classification_metrics
from features import window_feature_columns


WINDOW_COLUMN_PATTERN = re.compile(r"^w(?P<minutes>\d+)_")


@dataclass(frozen=True)
class DatasetIndex:
    name: str
    index_path: Path
    frame: pd.DataFrame


def _infer_name(index_path: Path, frame: pd.DataFrame) -> str:
    if "dataset" in frame.columns:
        values = [str(v) for v in pd.unique(frame["dataset"].dropna())]
        if len(values) == 1:
            return values[0]
        if values:
            return "+".join(sorted(values))
    return index_path.stem


def _supported_windows_from_frame(frame: pd.DataFrame) -> list[int]:
    windows: set[int] = set()
    for col in frame.columns:
        match = WINDOW_COLUMN_PATTERN.match(str(col))
        if match:
            windows.add(int(match.group("minutes")))
    return sorted(windows)


def _validate_index_frame(frame: pd.DataFrame, index_path: Path) -> None:
    required = {"split", "is_sse"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Index parquet {index_path} is missing required columns: {sorted(missing)}")


def load_indices(index_paths: Iterable[str | Path]) -> list[DatasetIndex]:
    indices: list[DatasetIndex] = []
    for path in index_paths:
        index_path = Path(path)
        frame = pd.read_parquet(index_path)
        _validate_index_frame(frame, index_path)
        name = _infer_name(index_path, frame)
        indices.append(DatasetIndex(name=name, index_path=index_path, frame=frame))
    # Deterministic order.
    indices.sort(key=lambda item: (item.name, str(item.index_path)))
    return indices


def _threshold_grid() -> np.ndarray:
    return np.linspace(0.05, 0.95, 19)


def _pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] = classification_metrics(y_true, y_prob, threshold=0.5)
    best_f1 = best_metrics.get("f1", -1.0)
    for threshold in _threshold_grid():
        metrics = classification_metrics(y_true, y_prob, threshold=float(threshold))
        f1 = metrics.get("f1", -1.0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _train_xgboost(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> XGBClassifier:
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
    return model


def run_transfer(
    indices: list[DatasetIndex],
    windows: list[int],
    include_self: bool = False,
) -> dict[str, Any]:
    if len(indices) < 2 and not include_self:
        raise ValueError("Cross-dataset transfer requires at least two index parquets (or pass --include-self).")

    dataset_meta: dict[str, Any] = {}
    for item in indices:
        counts = {k: int(v) for k, v in item.frame["split"].value_counts().sort_index().items()}
        dataset_meta[item.name] = {
            "index_path": str(item.index_path),
            "supported_windows": _supported_windows_from_frame(item.frame),
            "split_counts": counts,
            "sse_rate": float(pd.Series(item.frame["is_sse"]).mean()),
        }

    results: list[dict[str, Any]] = []
    for window_minutes in windows:
        features = window_feature_columns(window_minutes)
        for source in indices:
            if any(col not in source.frame.columns for col in features):
                missing = [col for col in features if col not in source.frame.columns]
                raise ValueError(f"Source {source.name} missing window feature columns: {missing}")
            source_train = source.frame[source.frame["split"] == "train"].copy()
            source_val = source.frame[source.frame["split"] == "val"].copy()
            if source_train.empty or source_val.empty:
                raise ValueError(f"Source {source.name} does not contain both train and val splits.")

            train_x = source_train[features].to_numpy(dtype=float)
            val_x = source_val[features].to_numpy(dtype=float)
            train_y = source_train["is_sse"].to_numpy(dtype=int)
            val_y = source_val["is_sse"].to_numpy(dtype=int)
            model = _train_xgboost(train_x, train_y, val_x, val_y)

            val_prob = model.predict_proba(val_x)[:, 1]
            threshold, source_val_cls = _pick_threshold(val_y, val_prob)
            source_val_cal = calibration_metrics(val_y, val_prob)

            for target in indices:
                if (not include_self) and (target.name == source.name):
                    continue
                if any(col not in target.frame.columns for col in features):
                    missing = [col for col in features if col not in target.frame.columns]
                    raise ValueError(f"Target {target.name} missing window feature columns: {missing}")
                target_test = target.frame[target.frame["split"] == "test"].copy()
                if target_test.empty:
                    raise ValueError(f"Target {target.name} does not contain a test split.")

                test_x = target_test[features].to_numpy(dtype=float)
                test_y = target_test["is_sse"].to_numpy(dtype=int)
                test_prob = model.predict_proba(test_x)[:, 1]
                target_test_cls = classification_metrics(test_y, test_prob, threshold=threshold)
                target_test_cal = calibration_metrics(test_y, test_prob)

                results.append(
                    {
                        "window_minutes": window_minutes,
                        "features": features,
                        "source": source.name,
                        "target": target.name,
                        "threshold": threshold,
                        "source_val": {"classification": source_val_cls, "calibration": source_val_cal},
                        "target_test": {"classification": target_test_cls, "calibration": target_test_cal},
                    }
                )

    results.sort(key=lambda row: (row["window_minutes"], row["source"], row["target"]))
    return {
        "windows": windows,
        "include_self": include_self,
        "datasets": dataset_meta,
        "pairs": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cross-dataset transfer (Task 1 XGBoost) on Event index parquets.")
    parser.add_argument(
        "--index-paths",
        nargs="+",
        required=True,
        help="One or more event_index.parquet paths (from different datasets).",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        default=None,
        help="Observation windows in minutes. Defaults to intersection across provided indices.",
    )
    parser.add_argument("--include-self", action="store_true", help="Include source==target pairs.")
    parser.add_argument(
        "--output-path",
        default="artifacts/cross_dataset_transfer.json",
        help="Where to write the all-pairs transfer report.",
    )
    args = parser.parse_args(argv)

    indices = load_indices(args.index_paths)
    if args.windows is None:
        supported_sets = [set(_supported_windows_from_frame(item.frame)) for item in indices]
        windows = sorted(set.intersection(*supported_sets)) if supported_sets else []
    else:
        windows = list(args.windows)
    if not windows:
        raise ValueError("No common observation windows found across indices (or none were provided).")

    report = run_transfer(indices=indices, windows=windows, include_self=args.include_self)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

