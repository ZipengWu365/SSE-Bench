from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def discover_prepare_scripts() -> dict[str, str]:
    discovered: dict[str, str] = {}
    for path in sorted(SCRIPTS_DIR.glob("prepare_*.py")):
        if path.name in {"prepare_all_datasets.py", "prepare.py"}:
            continue
        stem = path.stem
        dataset_key = stem.removeprefix("prepare_")
        discovered[dataset_key] = f"scripts.{stem}"
    return discovered


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all discovered dataset preparation entrypoints.")
    parser.add_argument("--datasets", nargs="*", help="Dataset keys to run. Defaults to all discovered datasets.")
    parser.add_argument("--list", action="store_true", help="List discovered prepare scripts and exit.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocesses.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately on the first preparation failure.")
    parser.add_argument("--dry-run", action="store_true", help="Only show the commands that would be run.")
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    discovered = discover_prepare_scripts()
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

        command = [args.python, "-m", module, *passthrough]
        if args.dry_run:
            results.append({"dataset": dataset, "status": "dry-run", "command": command})
            continue

        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "ok" if completed.returncode == 0 else "failed"
        results.append(
            {
                "dataset": dataset,
                "status": status,
                "command": command,
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
