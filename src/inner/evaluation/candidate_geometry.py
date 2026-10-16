"""Meterable, sandboxed extraction of public CP26 candidate geometry."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from inner.tasks.eft_task import EFTTask


_WORKER = Path(__file__).with_name("_geometry_worker.py")
_TIMEOUT_MARGIN_S = 5.0


@dataclass
class CandidateGeometryOutcome:
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    wall_s: float = 0.0
    timed_out: bool = False


def extract_candidate_geometry(
    task: EFTTask,
    program: str,
    *,
    timeout_s: float = 120.0,
    python_exe: str = sys.executable,
    cardinality: int = 26,
    dimension: int = 2,
) -> CandidateGeometryOutcome:
    """Execute ``run_packing`` out of process and return normalized arrays."""

    timeout = max(1.0, min(float(timeout_s), 120.0))
    with tempfile.TemporaryDirectory(prefix="sah_geometry_") as raw_tmp:
        tmp = Path(raw_tmp)
        program_path = tmp / "candidate.py"
        result_path = tmp / "result.json"
        request_path = tmp / "request.json"
        program_path.write_text(program)
        request_path.write_text(json.dumps({
            "program_path": str(program_path),
            "result_path": str(result_path),
            "shim_path": str(getattr(task, "shim_path", "") or ""),
            "task_dir": str(getattr(task, "task_dir", "") or ""),
            "cardinality": int(cardinality),
            "dimension": int(dimension),
        }))
        env = dict(os.environ)
        extras = [
            str(path) for path in (
                getattr(task, "shim_path", None),
                getattr(task, "task_dir", None),
            ) if path
        ]
        if extras:
            previous = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                extras + ([previous] if previous else [])
            )
        started = time.monotonic()
        process = subprocess.Popen(
            [python_exe, str(_WORKER), str(request_path)],
            cwd=str(tmp), env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout + _TIMEOUT_MARGIN_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        wall_s = time.monotonic() - started
        if timed_out:
            return CandidateGeometryOutcome(
                error=f"candidate geometry timed out after {timeout + _TIMEOUT_MARGIN_S:.0f}s",
                wall_s=wall_s, timed_out=True,
            )
        if not result_path.is_file():
            return CandidateGeometryOutcome(
                error="candidate geometry worker produced no result", wall_s=wall_s
            )
        try:
            result = json.loads(result_path.read_text())
        except Exception as exc:
            return CandidateGeometryOutcome(
                error=f"invalid candidate geometry worker result: {exc}", wall_s=wall_s
            )
        if not result.get("ok"):
            return CandidateGeometryOutcome(
                error=str(result.get("error") or "candidate geometry failed"),
                data={"traceback": result.get("traceback") or []}, wall_s=wall_s,
            )
        return CandidateGeometryOutcome(
            data={
                "centers": result["centers"],
                "radii": result["radii"],
                "sum_radii": float(result["sum_radii"]),
            },
            wall_s=wall_s,
        )
