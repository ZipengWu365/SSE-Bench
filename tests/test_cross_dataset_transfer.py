from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_index(path: Path, dataset: str, split_rows: list[tuple[str, int, float]]) -> Path:
    # split_rows: (split, is_sse, signal_feature)
    rows = []
    for split, is_sse, signal in split_rows:
        rows.append(
            {
                "event_id": f"{dataset}::{split}::{len(rows)}",
                "dataset": dataset,
                "platform": "web",
                "topic": None,
                "split": split,
                "is_sse": int(is_sse),
                # Minimal window feature set for w60_*
                "w60_observed_size": float(signal),
                "w60_growth_rate": float(signal),
                "w60_max_increment": float(signal),
                "w60_mean_increment": float(signal),
                "w60_burstiness": float(signal),
                "w60_acceleration": float(signal),
                "w60_sentiment_mean": 0.0,
                "w60_sentiment_abs": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return path


def test_cross_dataset_transfer_all_pairs_with_dummy_model(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_cross_dataset_transfer as mod

    class DummyXGBClassifier:
        def __init__(self, *args, **kwargs) -> None:
            self.bias = 0.0

        def fit(self, X, y, eval_set=None, verbose: bool = False):
            # Simple: bias = mean label, so probs are in (0,1) and deterministic.
            self.bias = float(np.asarray(y, dtype=float).mean()) if len(y) else 0.0
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            signal = X[:, 0] if X.ndim == 2 and X.shape[1] else np.zeros(len(X))
            # Map to [0,1] using a fixed squashing around bias.
            p1 = 1.0 / (1.0 + np.exp(-(signal + self.bias)))
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

    monkeypatch.setattr(mod, "XGBClassifier", DummyXGBClassifier)

    src_path = _make_index(
        tmp_path / "src.parquet",
        dataset="src_ds",
        split_rows=[
            ("train", 0, -2.0),
            ("train", 1, 2.0),
            ("val", 0, -1.0),
            ("val", 1, 1.0),
            ("test", 0, -1.0),
            ("test", 1, 1.0),
        ],
    )
    tgt_path = _make_index(
        tmp_path / "tgt.parquet",
        dataset="tgt_ds",
        split_rows=[
            ("train", 0, -2.0),
            ("train", 0, -1.0),
            ("val", 0, -1.0),
            ("val", 1, 1.0),
            ("test", 0, -1.0),
            ("test", 1, 1.0),
        ],
    )

    indices = mod.load_indices([src_path, tgt_path])
    report = mod.run_transfer(indices=indices, windows=[60], include_self=False)
    pairs = report["pairs"]
    assert len(pairs) == 2  # src->tgt and tgt->src (ordered pairs)

    forward = next((row for row in pairs if row["source"] == "src_ds" and row["target"] == "tgt_ds"), None)
    assert forward is not None
    assert forward["window_minutes"] == 60
    assert 0.0 < float(forward["threshold"]) <= 1.0

    source_val = forward["source_val"]
    target_test = forward["target_test"]
    assert "classification" in source_val
    assert "calibration" in source_val
    assert "classification" in target_test
    assert "calibration" in target_test

    for blob in (source_val["classification"], target_test["classification"]):
        for key in ("auprc", "f1", "precision", "recall", "auroc"):
            assert key in blob

    for blob in (source_val["calibration"], target_test["calibration"]):
        for key in ("brier_score", "ece"):
            assert key in blob
