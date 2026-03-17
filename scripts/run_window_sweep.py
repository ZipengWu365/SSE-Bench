from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_all_benchmark_suites import discover_suite_scripts, detect_processed_inputs


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
WINDOW_COLUMN_PATTERN = re.compile(r"^w(?P<minutes>\d+)_")


def _artifact_slug_for_dataset(dataset: str) -> str:
    candidate_files = sorted(ARTIFACTS_DIR.glob(f"*{dataset}*_summary.json"))
    if candidate_files:
        return candidate_files[0].stem.removesuffix("_summary")
    if dataset == "uci":
        return "uci_news"
    return dataset


def _index_path_for_dataset(dataset: str) -> Path | None:
    for path in detect_processed_inputs(dataset):
        if path.name == "event_index.parquet":
            return path
    return None


def _summary_for_slug(slug: str) -> dict[str, Any] | None:
    summary_path = ARTIFACTS_DIR / f"{slug}_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _supported_windows(index_path: Path, slice_minutes: int | None = None) -> list[int]:
    frame = pd.read_parquet(index_path)
    windows = set()
    for column in frame.columns:
        match = WINDOW_COLUMN_PATTERN.match(column)
        if match:
            windows.add(int(match.group("minutes")))
    if slice_minutes is not None:
        windows = {window for window in windows if window >= slice_minutes}
    return sorted(windows)


def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_minutes": report.get("window_minutes"),
        "task_1_heuristic": report.get("task_1_heuristic", {}).get("classification", {}),
        "task_1_xgboost": report.get("task_1_xgboost", {}).get("classification", {}),
        "task_1_graph": report.get("task_1_graph", {}).get("classification", {}),
        "task_2_time_to_sse": report.get("task_2_time_to_sse", {}).get("time_to_event", {}),
        "task_3_final_size": report.get("task_3_final_size", {}).get("log_regression", {}),
        "task_4_retrieval": report.get("task_4_retrieval", {}).get("metrics", {}),
    }


def _run_suite(
    module: str,
    python_executable: str,
    window_minutes: int,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        python_executable,
        "-m",
        module,
        "--window-minutes",
        str(window_minutes),
        "--output-path",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Window sweep command failed for {module} at window={window_minutes}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_sweep(
    datasets: list[str] | None = None,
    windows: list[int] | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    discovered = discover_suite_scripts()
    selected = datasets or sorted(discovered)
    aggregate: dict[str, Any] = {"datasets": []}

    for dataset in selected:
        module = discovered.get(dataset)
        if module is None:
            aggregate["datasets"].append({"dataset": dataset, "status": "skipped", "reason": "suite not discovered"})
            continue

        index_path = _index_path_for_dataset(dataset)
        if index_path is None:
            aggregate["datasets"].append({"dataset": dataset, "status": "skipped", "reason": "event index not found"})
            continue

        slug = _artifact_slug_for_dataset(dataset)
        summary = _summary_for_slug(slug) or {}
        supported = _supported_windows(index_path, slice_minutes=summary.get("slice_minutes"))
        selected_windows = [window for window in (windows or supported) if window in supported]
        rows = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for window_minutes in selected_windows:
                temp_output = temp_root / f"{slug}_w{window_minutes}.json"
                report = _run_suite(
                    module=module,
                    python_executable=python_executable,
                    window_minutes=window_minutes,
                    output_path=temp_output,
                )
                rows.append(_extract_metrics(report))

        payload = {
            "dataset_key": dataset,
            "artifact_slug": slug,
            "suite_module": module,
            "supported_windows": supported,
            "windows": rows,
        }
        (ARTIFACTS_DIR / f"{slug}_window_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        aggregate["datasets"].append({"dataset": dataset, "status": "ok", **payload})

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suites across multiple observation windows.")
    parser.add_argument("--datasets", nargs="*", help="Dataset keys to sweep. Defaults to all discovered suites.")
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        help="Observation windows to evaluate. Defaults to all windows found in each dataset index.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocess suites.")
    args = parser.parse_args()

    aggregate = run_sweep(datasets=args.datasets, windows=args.windows, python_executable=args.python)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
