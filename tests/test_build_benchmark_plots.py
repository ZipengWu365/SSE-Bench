from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_build_benchmark_plots_writes_manifest_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.build_benchmark_plots as mod

    artifacts_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "artifacts" / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Minimal overview + sweep for a single dataset slug.
    (artifacts_dir / "benchmark_overview.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "slug": "toy",
                        "dataset_name": "toy_dataset",
                        "summary": {"total_events": 10, "sse_rate": 0.1, "slice_minutes": 20},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "toy_window_sweep.json").write_text(
        json.dumps(
            {
                "dataset_key": "toy",
                "artifact_slug": "toy",
                "suite_module": "scripts.run_toy_benchmark_suite",
                "supported_windows": [60],
                "windows": [
                    {
                        "window_minutes": 60,
                        "task_1_xgboost": {"auroc": 0.7, "auprc": 0.2},
                        "task_1_graph": {"auroc": 0.9, "auprc": 0.4},
                        "task_3_final_size": {"rmse": 1.0},
                        "task_4_retrieval": {"recall_at_k": 0.1, "ndcg_at_k": 0.2},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Avoid requiring matplotlib for unit tests; we only validate file outputs.
    def _touch_png(*args, **kwargs) -> None:
        path = kwargs.get("output_path") or args[2]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(mod, "_require_matplotlib", lambda: None)
    monkeypatch.setattr(mod, "_save_line_plot", lambda *a, **k: _touch_png(output_path=k["output_path"]))
    monkeypatch.setattr(
        mod,
        "_save_bar_plot",
        lambda *a, **k: _touch_png(output_path=k["output_path"]),
    )

    manifest = mod.build_plots(artifacts_dir=artifacts_dir, output_dir=output_dir)

    manifest_path = output_dir / "plot_manifest.json"
    report_md = output_dir / "benchmark_plots.md"
    assert manifest_path.exists()
    assert report_md.exists()
    assert "toy" in manifest["datasets"]
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["output_dir"] == str(output_dir)
    assert Path(loaded_manifest["outputs"]["report_markdown"]).exists()

    # Ensure markdown references expected files.
    text = report_md.read_text(encoding="utf-8")
    assert "task1_xgb_auroc_vs_window.png" in text
    assert "headline_task1_xgb_bar.png" in text
