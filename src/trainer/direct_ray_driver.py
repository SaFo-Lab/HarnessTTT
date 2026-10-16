#!/usr/bin/env python3
"""Run an OpenClaw training entrypoint on an already-started local Ray head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import ray


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-env-json", required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    if not args.train_script.is_file():
        parser.error(f"training entrypoint does not exist: {args.train_script}")

    runtime_env = json.loads(args.runtime_env_json)
    if not isinstance(runtime_env, dict):
        parser.error("--runtime-env-json must decode to an object")
    address = os.environ.get("RAY_ADDRESS", "auto")
    context = ray.init(address=address, runtime_env=runtime_env)
    print(
        "DIRECT_RAY_DRIVER_CONNECTED "
        f"address={address} node_id={context.address_info.get('node_id', '')}",
        flush=True,
    )

    sys.argv = [str(args.train_script), *args.train_args]
    runpy.run_path(str(args.train_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
