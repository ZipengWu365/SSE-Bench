from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_paper_tables_smoke(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    overview = {
        "datasets": [
            {
                "slug": "toy",
                "dataset_name": "toy_dataset",
                "summary": {
                    "dataset": "toy_dataset",
                    "total_events": 100,
                    "sse_events": 5,
                    "sse_rate": 0.05,
                    "slice_minutes": 20,
                },
            }
        ]
    }
    sweep = {
        "dataset_key": "toy",
        "artifact_slug": "toy",
        "supported_windows": [20, 360],
        "windows": [
            {
                "window_minutes": 20,
                "task_1_xgboost": {"auroc": 0.6, "auprc": 0.1},
                "task_3_final_size": {"rmse": 1.2},
                "task_4_retrieval": {"recall_at_k": 0.02},
            },
            {
                "window_minutes": 360,
                "task_1_xgboost": {"auroc": 0.8, "auprc": 0.2},
                "task_1_graph": {"auroc": 0.9, "auprc": 0.3},
                "task_2_time_to_sse": {"mae": 123.4},
                "task_3_final_size": {"rmse": 0.9},
                "task_4_retrieval": {"recall_at_k": 0.12},
            },
        ],
    }

    _write_json(artifacts_dir / "benchmark_overview.json", overview)
    _write_json(artifacts_dir / "toy_window_sweep.json", sweep)

    from scripts.build_paper_tables import build_paper_tables

    output_dir = artifacts_dir / "paper_tables"
    outputs = build_paper_tables(artifacts_dir=artifacts_dir, output_dir=output_dir)

    for key, output_path in outputs.items():
        assert Path(output_path).exists(), f"Missing output {key} at {output_path}"

    md = (output_dir / "headline_metrics.md").read_text(encoding="utf-8")
    assert "toy_dataset" in md
    assert "| 360 |" in md
    assert "0.800" in md

    tex = (output_dir / "dataset_summary.tex").read_text(encoding="utf-8")
    assert "\\begin{table}" in tex
    assert "toy\\_dataset" in tex

