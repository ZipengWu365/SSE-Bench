from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACTS_DIR / "plots"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _discover_window_sweeps(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    sweeps: dict[str, dict[str, Any]] = {}
    for path in sorted(artifacts_dir.glob("*_window_sweep.json")):
        slug = path.stem.removesuffix("_window_sweep")
        sweeps[slug] = _load_json(path)
    return sweeps


def _discover_summaries(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(artifacts_dir.glob("*_summary.json")):
        slug = path.stem.removesuffix("_summary")
        summaries[slug] = _load_json(path)
    return summaries


def _overview_datasets(artifacts_dir: Path) -> list[dict[str, Any]]:
    overview_path = artifacts_dir / "benchmark_overview.json"
    if overview_path.exists():
        overview = _load_json(overview_path)
        return list(overview.get("datasets", []))
    datasets = []
    summaries = _discover_summaries(artifacts_dir)
    for slug, summary in summaries.items():
        datasets.append({"slug": slug, "dataset_name": summary.get("dataset"), "summary": summary})
    return datasets


def _sweep_metric_frame(sweeps: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slug, sweep in sweeps.items():
        for window in sweep.get("windows", []):
            window_minutes = window.get("window_minutes")
            if window_minutes is None:
                continue
            rows.append(
                {
                    "slug": slug,
                    "window_minutes": int(window_minutes),
                    "task1_xgb_auroc": _safe_float(window.get("task_1_xgboost", {}).get("auroc")),
                    "task1_xgb_auprc": _safe_float(window.get("task_1_xgboost", {}).get("auprc")),
                    "task1_graph_auroc": _safe_float(window.get("task_1_graph", {}).get("auroc")),
                    "task1_graph_auprc": _safe_float(window.get("task_1_graph", {}).get("auprc")),
                    "task3_rmse": _safe_float(window.get("task_3_final_size", {}).get("rmse")),
                    "task4_recall": _safe_float(window.get("task_4_retrieval", {}).get("recall_at_k")),
                    "task4_ndcg": _safe_float(window.get("task_4_retrieval", {}).get("ndcg_at_k")),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["slug", "window_minutes"]).reset_index(drop=True)
    return frame


def _headline_frame(datasets: list[dict[str, Any]], sweeps: dict[str, dict[str, Any]]) -> pd.DataFrame:
    summary_lookup = {item.get("slug"): (item.get("summary") or {}) for item in datasets}
    name_lookup = {item.get("slug"): (item.get("dataset_name") or item.get("slug")) for item in datasets}
    rows: list[dict[str, Any]] = []
    for slug, sweep in sweeps.items():
        windows = sweep.get("windows", [])
        if not windows:
            continue
        baseline = None
        for candidate in windows:
            if candidate.get("window_minutes") == 360:
                baseline = candidate
                break
        if baseline is None:
            baseline = windows[len(windows) // 2]
        summary = summary_lookup.get(slug, {})
        rows.append(
            {
                "slug": slug,
                "dataset_name": name_lookup.get(slug, slug),
                "events": summary.get("total_events"),
                "sse_rate": _safe_float(summary.get("sse_rate")),
                "slice_minutes": summary.get("slice_minutes"),
                "headline_window_minutes": int(baseline.get("window_minutes")),
                "task1_xgb_auroc": _safe_float(baseline.get("task_1_xgboost", {}).get("auroc")),
                "task1_xgb_auprc": _safe_float(baseline.get("task_1_xgboost", {}).get("auprc")),
                "task1_graph_auroc": _safe_float(baseline.get("task_1_graph", {}).get("auroc")),
                "task1_graph_auprc": _safe_float(baseline.get("task_1_graph", {}).get("auprc")),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("slug").reset_index(drop=True)
    return frame


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting.\n"
            "Install it with: pip install '.[plot]'\n"
            "Or: pip install matplotlib\n"
        ) from exc


def _save_line_plot(
    frame: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if frame.empty or metric not in frame.columns:
        return
    filtered = frame.dropna(subset=[metric]).copy()
    if filtered.empty:
        return

    plt.figure(figsize=(7.5, 4.5))
    for slug, group in filtered.groupby("slug", sort=True):
        group = group.sort_values("window_minutes")
        plt.plot(group["window_minutes"], group[metric], marker="o", linewidth=2, label=slug)
    plt.xscale("log", base=10)
    plt.xticks(sorted(filtered["window_minutes"].unique().tolist()), labels=[str(v) for v in sorted(filtered["window_minutes"].unique())])
    plt.xlabel("Observation Window (minutes)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_bar_plot(
    frame: pd.DataFrame,
    metric_columns: list[str],
    output_path: Path,
    title: str,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if frame.empty:
        return

    plot_frame = frame[["slug", "dataset_name", *metric_columns]].copy()
    plot_frame = plot_frame.dropna(subset=metric_columns, how="all")
    if plot_frame.empty:
        return

    x = range(len(plot_frame))
    width = 0.35 if len(metric_columns) == 2 else 0.25

    plt.figure(figsize=(8.2, 4.6))
    for idx, metric in enumerate(metric_columns):
        values = plot_frame[metric].fillna(0.0).to_numpy(dtype=float)
        positions = [pos + (idx - (len(metric_columns) - 1) / 2) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric)

    plt.xticks(list(x), plot_frame["slug"].tolist(), rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_plots(artifacts_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweeps = _discover_window_sweeps(artifacts_dir)
    datasets = _overview_datasets(artifacts_dir)
    sweep_frame = _sweep_metric_frame(sweeps)
    headline_frame = _headline_frame(datasets, sweeps)

    outputs: dict[str, str] = {}
    _save_line_plot(
        sweep_frame,
        metric="task1_xgb_auroc",
        output_path=output_dir / "task1_xgb_auroc_vs_window.png",
        title="Task 1 XGBoost AUROC vs Observation Window",
        ylabel="AUROC",
    )
    outputs["task1_xgb_auroc_vs_window"] = str(output_dir / "task1_xgb_auroc_vs_window.png")

    _save_line_plot(
        sweep_frame,
        metric="task1_xgb_auprc",
        output_path=output_dir / "task1_xgb_auprc_vs_window.png",
        title="Task 1 XGBoost AUPRC vs Observation Window",
        ylabel="AUPRC",
    )
    outputs["task1_xgb_auprc_vs_window"] = str(output_dir / "task1_xgb_auprc_vs_window.png")

    _save_line_plot(
        sweep_frame,
        metric="task3_rmse",
        output_path=output_dir / "task3_final_size_rmse_vs_window.png",
        title="Task 3 Final Size (log) RMSE vs Observation Window",
        ylabel="RMSE (log scale target)",
    )
    outputs["task3_final_size_rmse_vs_window"] = str(output_dir / "task3_final_size_rmse_vs_window.png")

    _save_line_plot(
        sweep_frame,
        metric="task4_recall",
        output_path=output_dir / "task4_retrieval_recall_vs_window.png",
        title="Task 4 Retrieval Recall@k vs Observation Window",
        ylabel="Recall@k",
    )
    outputs["task4_retrieval_recall_vs_window"] = str(output_dir / "task4_retrieval_recall_vs_window.png")

    _save_bar_plot(
        headline_frame,
        metric_columns=["task1_xgb_auroc", "task1_xgb_auprc"],
        output_path=output_dir / "headline_task1_xgb_bar.png",
        title="Headline Task 1 XGBoost Metrics (AUROC/AUPRC)",
    )
    outputs["headline_task1_xgb_bar"] = str(output_dir / "headline_task1_xgb_bar.png")

    report_md = output_dir / "benchmark_plots.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        "\n".join(
            [
                "# SSE-Bench Plots",
                "",
                "Generated from `artifacts/*_window_sweep.json` and `artifacts/benchmark_overview.json`.",
                "",
                "## Cross-Window",
                "",
                "![Task 1 XGB AUROC](task1_xgb_auroc_vs_window.png)",
                "",
                "![Task 1 XGB AUPRC](task1_xgb_auprc_vs_window.png)",
                "",
                "![Task 3 RMSE](task3_final_size_rmse_vs_window.png)",
                "",
                "![Task 4 Recall@k](task4_retrieval_recall_vs_window.png)",
                "",
                "## Headline",
                "",
                "![Headline Task 1 XGB](headline_task1_xgb_bar.png)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["report_markdown"] = str(report_md)

    manifest = {
        "artifacts_dir": str(artifacts_dir),
        "output_dir": str(output_dir),
        "outputs": outputs,
        "datasets": sorted(sweeps),
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static plots from SSE-Bench benchmark artifacts.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Directory containing benchmark artifacts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for plot outputs.")
    args = parser.parse_args()

    manifest = build_plots(artifacts_dir=Path(args.artifacts_dir), output_dir=Path(args.output_dir))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

