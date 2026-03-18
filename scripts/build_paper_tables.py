from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACTS_DIR / "paper_tables"


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


def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _latex_escape(text: str) -> str:
    # Minimal escape for tabular content.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _to_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _to_latex_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    align: str,
) -> str:
    escaped_headers = [_latex_escape(header) for header in headers]
    escaped_rows = [[_latex_escape(cell) for cell in row] for row in rows]
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(escaped_headers) + " \\\\",
        "\\midrule",
    ]
    for row in escaped_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


@dataclass(frozen=True)
class DatasetEntry:
    slug: str
    dataset_name: str
    summary: dict[str, Any]


def _load_overview_entries(artifacts_dir: Path) -> list[DatasetEntry]:
    overview_path = artifacts_dir / "benchmark_overview.json"
    overview = _load_json(overview_path)
    entries: list[DatasetEntry] = []
    for item in overview.get("datasets", []):
        slug = str(item.get("slug") or "")
        if not slug:
            continue
        summary = dict(item.get("summary") or {})
        dataset_name = str(item.get("dataset_name") or summary.get("dataset") or slug)
        entries.append(DatasetEntry(slug=slug, dataset_name=dataset_name, summary=summary))
    entries.sort(key=lambda entry: entry.slug)
    return entries


def _load_window_sweeps(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    sweeps: dict[str, dict[str, Any]] = {}
    for path in sorted(artifacts_dir.glob("*_window_sweep.json")):
        slug = path.stem.removesuffix("_window_sweep")
        sweeps[slug] = _load_json(path)
    return sweeps


def _pick_headline_window(sweep: dict[str, Any], prefer: int = 360) -> int | None:
    supported = sweep.get("supported_windows") or []
    try:
        supported_int = sorted({int(v) for v in supported})
    except Exception:
        supported_int = []
    if prefer in supported_int:
        return prefer
    return supported_int[-1] if supported_int else None


def _headline_metrics_from_sweep(sweep: dict[str, Any], window_minutes: int) -> dict[str, Any]:
    for row in sweep.get("windows", []):
        if int(row.get("window_minutes", -1)) == int(window_minutes):
            task1_xgb = row.get("task_1_xgboost", {}) or {}
            task1_graph = row.get("task_1_graph", {}) or {}
            task2 = row.get("task_2_time_to_sse", {}) or {}
            task3 = row.get("task_3_final_size", {}) or {}
            task4 = row.get("task_4_retrieval", {}) or {}
            return {
                "window_minutes": window_minutes,
                "task1_xgb_auroc": task1_xgb.get("auroc"),
                "task1_xgb_auprc": task1_xgb.get("auprc"),
                "task1_graph_auroc": task1_graph.get("auroc"),
                "task1_graph_auprc": task1_graph.get("auprc"),
                "task2_mae": task2.get("mae"),
                "task3_rmse": task3.get("rmse"),
                "task4_recall": task4.get("recall_at_k"),
            }
    return {"window_minutes": window_minutes}


def _dataset_summary_rows(entries: Iterable[DatasetEntry]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        summary = entry.summary
        rows.append(
            [
                entry.dataset_name,
                entry.slug,
                _fmt_int(summary.get("total_events")),
                _fmt_int(summary.get("sse_events")),
                _fmt(summary.get("sse_rate"), digits=4),
                _fmt_int(summary.get("slice_minutes")),
            ]
        )
    return rows


def _headline_rows(entries: Iterable[DatasetEntry], sweeps: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        sweep = sweeps.get(entry.slug)
        window = _pick_headline_window(sweep) if sweep else None
        metrics = _headline_metrics_from_sweep(sweep, window) if sweep and window is not None else {}
        rows.append(
            [
                entry.dataset_name,
                entry.slug,
                _fmt_int(metrics.get("window_minutes")),
                _fmt(metrics.get("task1_xgb_auroc")),
                _fmt(metrics.get("task1_xgb_auprc")),
                _fmt(metrics.get("task1_graph_auroc")),
                _fmt(metrics.get("task1_graph_auprc")),
                _fmt(metrics.get("task2_mae"), digits=1),
                _fmt(metrics.get("task3_rmse")),
                _fmt(metrics.get("task4_recall")),
            ]
        )
    return rows


def build_paper_tables(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    entries = _load_overview_entries(artifacts_dir)
    sweeps = _load_window_sweeps(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_headers = ["Dataset", "Slug", "Events", "SSE", "SSE Rate", "Slice (min)"]
    dataset_rows = _dataset_summary_rows(entries)
    dataset_md = _to_markdown_table(dataset_headers, dataset_rows)
    dataset_tex = _to_latex_table(
        dataset_headers,
        dataset_rows,
        caption="Dataset summary for SSE-Bench.",
        label="tab:ssebench_dataset_summary",
        align="llrrrr",
    )

    headline_headers = [
        "Dataset",
        "Slug",
        "Window",
        "T1 XGB AUROC",
        "T1 XGB AUPRC",
        "T1 Graph AUROC",
        "T1 Graph AUPRC",
        "T2 MAE",
        "T3 RMSE",
        "T4 Recall@k",
    ]
    headline_rows = _headline_rows(entries, sweeps)
    headline_md = _to_markdown_table(headline_headers, headline_rows)
    headline_tex = _to_latex_table(
        headline_headers,
        headline_rows,
        caption="Headline metrics (selected window) for SSE-Bench baselines.",
        label="tab:ssebench_headline_metrics",
        align="llrrrrrrrr",
    )

    outputs = {
        "dataset_summary_md": str(output_dir / "dataset_summary.md"),
        "dataset_summary_tex": str(output_dir / "dataset_summary.tex"),
        "headline_metrics_md": str(output_dir / "headline_metrics.md"),
        "headline_metrics_tex": str(output_dir / "headline_metrics.tex"),
    }
    Path(outputs["dataset_summary_md"]).write_text(dataset_md, encoding="utf-8")
    Path(outputs["dataset_summary_tex"]).write_text(dataset_tex, encoding="utf-8")
    Path(outputs["headline_metrics_md"]).write_text(headline_md, encoding="utf-8")
    Path(outputs["headline_metrics_tex"]).write_text(headline_tex, encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready tables from SSE-Bench artifacts.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Directory containing benchmark artifacts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated paper tables.")
    args = parser.parse_args()

    outputs = build_paper_tables(artifacts_dir=Path(args.artifacts_dir), output_dir=Path(args.output_dir))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

