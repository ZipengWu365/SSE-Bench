from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OVERVIEW_JSON_PATH = ARTIFACTS_DIR / "benchmark_overview.json"
OVERVIEW_MD_PATH = ARTIFACTS_DIR / "benchmark_overview.md"


def _artifact_slug(path: Path) -> str:
    stem = path.stem
    for suffix in ("_summary", "_baselines", "_window_sweep"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def _load_json_artifacts(pattern: str) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(ARTIFACTS_DIR.glob(pattern)):
        loaded[_artifact_slug(path)] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _dataset_entry(
    slug: str,
    summary: dict[str, Any] | None,
    baselines: dict[str, Any] | None,
    sweep: dict[str, Any] | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "slug": slug,
        "dataset_name": (summary or {}).get("dataset"),
        "summary": summary,
        "baselines": baselines,
        "window_sweep": sweep,
    }
    if baselines:
        task_1_xgb = baselines.get("task_1_xgboost", {}).get("classification", {})
        task_1_graph = baselines.get("task_1_graph", {}).get("classification", {})
        task_2 = baselines.get("task_2_time_to_sse", {}).get("time_to_event", {})
        task_3 = baselines.get("task_3_final_size", {}).get("log_regression", {})
        task_4 = baselines.get("task_4_retrieval", {}).get("metrics", {})
        entry["headline_metrics"] = {
            "window_minutes": baselines.get("window_minutes"),
            "xgboost_auroc": task_1_xgb.get("auroc"),
            "xgboost_auprc": task_1_xgb.get("auprc"),
            "graph_auroc": task_1_graph.get("auroc"),
            "graph_auprc": task_1_graph.get("auprc"),
            "time_to_sse_mae": task_2.get("mae"),
            "final_size_rmse": task_3.get("rmse"),
            "retrieval_recall_at_k": task_4.get("recall_at_k"),
        }
    return entry


def build_overview() -> dict[str, Any]:
    summaries = _load_json_artifacts("*_summary.json")
    baselines = _load_json_artifacts("*_baselines.json")
    sweeps = _load_json_artifacts("*_window_sweep.json")
    all_slugs = sorted(set(summaries) | set(baselines) | set(sweeps))
    datasets = [
        _dataset_entry(
            slug=slug,
            summary=summaries.get(slug),
            baselines=baselines.get(slug),
            sweep=sweeps.get(slug),
        )
        for slug in all_slugs
    ]
    overview = {
        "datasets": datasets,
        "available_summaries": sorted(summaries),
        "available_baselines": sorted(baselines),
        "available_window_sweeps": sorted(sweeps),
    }
    return overview


def _dataset_row(entry: dict[str, Any]) -> str:
    summary = entry.get("summary") or {}
    headline = entry.get("headline_metrics") or {}
    return "| {name} | {events} | {sse} | {rate} | {slice_minutes} | {window} | {auroc} | {auprc} | {graph_auroc} | {mae} | {rmse} | {recall} |".format(
        name=entry.get("dataset_name") or entry["slug"],
        events=summary.get("total_events", "-"),
        sse=summary.get("sse_events", "-"),
        rate=_format_float(summary.get("sse_rate")),
        slice_minutes=summary.get("slice_minutes", "-"),
        window=headline.get("window_minutes", "-"),
        auroc=_format_float(headline.get("xgboost_auroc")),
        auprc=_format_float(headline.get("xgboost_auprc")),
        graph_auroc=_format_float(headline.get("graph_auroc")),
        mae=_format_float(headline.get("time_to_sse_mae")),
        rmse=_format_float(headline.get("final_size_rmse")),
        recall=_format_float(headline.get("retrieval_recall_at_k")),
    )


def _sweep_section(entry: dict[str, Any]) -> list[str]:
    sweep = entry.get("window_sweep")
    if not sweep:
        return []
    lines = [
        f"### {entry.get('dataset_name') or entry['slug']} Window Sweep",
        "",
        "| Window (min) | Task1 XGB AUROC | Task1 XGB AUPRC | Task3 RMSE | Task4 Recall@k |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sweep.get("windows", []):
        lines.append(
            "| {window} | {auroc} | {auprc} | {rmse} | {recall} |".format(
                window=row.get("window_minutes", "-"),
                auroc=_format_float(row.get("task_1_xgboost", {}).get("auroc")),
                auprc=_format_float(row.get("task_1_xgboost", {}).get("auprc")),
                rmse=_format_float(row.get("task_3_final_size", {}).get("rmse")),
                recall=_format_float(row.get("task_4_retrieval", {}).get("recall_at_k")),
            )
        )
    lines.append("")
    return lines


def build_markdown(overview: dict[str, Any]) -> str:
    lines = [
        "# SSE-Bench Overview",
        "",
        "| Dataset | Events | SSE | SSE Rate | Slice (min) | Headline Window | Task1 XGB AUROC | Task1 XGB AUPRC | Task1 Graph AUROC | Task2 MAE | Task3 RMSE | Task4 Recall@k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in overview["datasets"]:
        lines.append(_dataset_row(entry))
    lines.append("")
    for entry in overview["datasets"]:
        lines.extend(_sweep_section(entry))
    return "\n".join(lines) + "\n"


def main() -> None:
    overview = build_overview()
    OVERVIEW_JSON_PATH.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    OVERVIEW_MD_PATH.write_text(build_markdown(overview), encoding="utf-8")
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
