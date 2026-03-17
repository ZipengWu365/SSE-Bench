from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


def discover_suite_scripts() -> dict[str, str]:
    discovered: dict[str, str] = {}
    for path in sorted(SCRIPTS_DIR.glob("run_*_benchmark_suite.py")):
        if path.name == "run_all_benchmark_suites.py":
            continue
        stem = path.stem
        dataset_key = stem.removeprefix("run_").removesuffix("_benchmark_suite")
        discovered[dataset_key] = f"scripts.{stem}"
    return discovered


def detect_processed_inputs(dataset: str) -> list[Path]:
    candidates = [
        PROCESSED_ROOT / dataset / "event_index.parquet",
        PROCESSED_ROOT / dataset / "events.jsonl.gz",
        PROCESSED_ROOT / f"{dataset}_sse" / "event_index.parquet",
        PROCESSED_ROOT / f"{dataset}_sse" / "events.jsonl.gz",
    ]
    if PROCESSED_ROOT.exists():
        for directory in PROCESSED_ROOT.iterdir():
            if directory.is_dir() and dataset in directory.name:
                candidates.append(directory / "event_index.parquet")
                candidates.append(directory / "events.jsonl.gz")
    existing = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            existing.append(path)
            seen.add(path)
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all discovered benchmark suite entrypoints.")
    parser.add_argument("--datasets", nargs="*", help="Dataset keys to run. Defaults to all discovered suites.")
    parser.add_argument("--list", action="store_true", help="List discovered suite scripts and exit.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocesses.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately on the first suite failure.")
    parser.add_argument(
        "--force-run-missing-inputs",
        action="store_true",
        help="Run even if no processed dataset artifacts are detected.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show the commands that would be run.")
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    discovered = discover_suite_scripts()
    if args.list:
        print(json.dumps(discovered, indent=2))
        return 0

    selected = args.datasets or list(discovered)
    results: list[dict[str, object]] = []
    exit_code = 0

    for dataset in selected:
        module = discovered.get(dataset)
        if module is None:
            results.append({"dataset": dataset, "status": "skipped", "reason": "not discovered"})
            continue

        detected_inputs = [str(path) for path in detect_processed_inputs(dataset)]
        if not detected_inputs and not args.force_run_missing_inputs:
            results.append(
                {
                    "dataset": dataset,
                    "status": "skipped",
                    "reason": "no processed artifacts detected",
                }
            )
            continue

        command = [args.python, "-m", module, *passthrough]
        if args.dry_run:
            results.append(
                {
                    "dataset": dataset,
                    "status": "dry-run",
                    "command": command,
                    "detected_inputs": detected_inputs,
                }
            )
            continue

        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "ok" if completed.returncode == 0 else "failed"
        results.append(
            {
                "dataset": dataset,
                "status": status,
                "command": command,
                "detected_inputs": detected_inputs,
                "returncode": completed.returncode,
            }
        )
        if completed.returncode != 0:
            exit_code = 1
            if args.stop_on_error:
                break

    summary = {
        "discovered": discovered,
        "selected": selected,
        "passthrough_args": passthrough,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
