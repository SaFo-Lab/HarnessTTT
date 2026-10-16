#!/usr/bin/env python3
"""Fail unless every discovered agent result contains a saved executor history."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _has_assistant_turn(trajectory) -> bool:
    if not isinstance(trajectory, list) or not trajectory:
        return False
    for message in trajectory:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "")
        role = getattr(role, "value", role)
        if str(role).lower().split(".")[-1] == "assistant":
            return True
    return False


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if p.parent.name == "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="result JSON file(s) or roots containing results/*.json")
    args = parser.parse_args()

    files = sorted({p.resolve() for root in args.paths for p in _result_files(root)})
    failures: list[tuple[Path, str]] = []
    if not files:
        failures.append((args.paths[0], "no per-task result JSON files found"))

    for path in files:
        try:
            result = json.loads(path.read_text())
        except Exception as exc:
            failures.append((path, f"unreadable JSON: {type(exc).__name__}: {exc}"))
            continue
        if not isinstance(result, dict):
            failures.append((path, "result is not a JSON object"))
        elif not _has_assistant_turn(result.get("trajectory")):
            failures.append((path, "missing trajectory or no assistant turn"))

    invalid_files = sum(1 for path, _ in failures if path in files)
    report = {"checked": len(files), "valid": len(files) - invalid_files,
              "invalid": len(failures)}
    print(f"[trajectory-audit] {json.dumps(report, sort_keys=True)}")
    for path, reason in failures:
        print(f"[trajectory-audit] INVALID {path}: {reason}", file=sys.stderr)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
