"""Isolated worker for the trusted current-candidate geometry bridge."""
from __future__ import annotations

import importlib.util
import json
import math
import resource
import sys
import traceback
from pathlib import Path


def _finite_matrix(value, rows: int, cols: int):
    if not isinstance(value, (list, tuple)) and not hasattr(value, "tolist"):
        raise ValueError("centers must be an array-like value")
    raw = value.tolist() if hasattr(value, "tolist") else value
    if len(raw) != rows:
        raise ValueError(f"centers shape must be ({rows}, {cols}); got {len(raw)} rows")
    out = []
    for index, row in enumerate(raw):
        row = row.tolist() if hasattr(row, "tolist") else row
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise ValueError(f"centers[{index}] must contain exactly {cols} values")
        converted = [float(item) for item in row]
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"centers[{index}] contains a non-finite value")
        out.append(converted)
    return out


def _finite_vector(value, size: int):
    if not isinstance(value, (list, tuple)) and not hasattr(value, "tolist"):
        raise ValueError("radii must be an array-like value")
    raw = value.tolist() if hasattr(value, "tolist") else value
    if len(raw) != size:
        raise ValueError(f"radii shape must be ({size},); got {len(raw)} values")
    out = [float(item) for item in raw]
    if not all(math.isfinite(item) for item in out):
        raise ValueError("radii contains a non-finite value")
    return out


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    result_path = Path(request["result_path"])
    try:
        # The parent also enforces a process-group wall timeout.  These limits
        # bound accidental runaway code before that deadline.
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
        for raw in (request.get("shim_path"), request.get("task_dir")):
            if raw and str(raw) not in sys.path:
                sys.path.insert(0, str(raw))
        program_path = Path(request["program_path"])
        spec = importlib.util.spec_from_file_location(
            f"_sah_geometry_{abs(hash(str(program_path)))}", program_path
        )
        if spec is None or spec.loader is None:
            raise ValueError("candidate module could not be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_packing = getattr(module, "run_packing", None)
        if not callable(run_packing):
            raise ValueError("candidate does not define callable run_packing()")
        value = run_packing()
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("run_packing() must return (centers, radii, ...)")
        cardinality = int(request["cardinality"])
        dimension = int(request["dimension"])
        centers = _finite_matrix(value[0], cardinality, dimension)
        radii = _finite_vector(value[1], cardinality)
        total = float(sum(radii))
        if not math.isfinite(total):
            raise ValueError("sum of radii is non-finite")
        result = {
            "ok": True,
            "centers": centers,
            "radii": radii,
            "sum_radii": total,
        }
    except BaseException as exc:  # worker verdict must survive candidate errors
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3).splitlines()[-4:],
        }
    result_path.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
