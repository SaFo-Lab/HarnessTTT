"""Subprocess worker: evaluate one candidate program with a task's evaluator.

Invoked as ``python _eval_worker.py <request.json>``. Keeps the (potentially
crashy, import-polluting, slow) evaluator import + call out of the harness
process. Writes a single JSON result to the ``result_path`` in the request;
never writes the result to stdout (evaluators print freely to stdout/stderr).

request.json = {
  "evaluator_path": "...evaluator.py",
  "program_path":   "...candidate.py",
  "shim_path":      "...runtime/skydiscover_min",   # on sys.path so evaluator imports
  "result_path":    "...result.json"
}
"""
import contextlib
import importlib.util
import inspect
import io
import json
import operator
import sys
import traceback
from pathlib import Path


def _eplb_topology_error(evaluator, program_path: str):
    """Return an error when an EPLB program violates the node/group contract.

    The upstream evaluator checks tensor shapes and replica counts, but its
    scalar score does not enforce that each logical expert group remains on a
    single node.  A program can therefore obtain an artificially high score by
    ignoring ``num_groups``/``num_nodes`` and solving the easier global packing
    problem.  Validate outputs rather than source text so the same semantic
    rule applies to every search method and implementation.
    """
    if "/ADRS/eplb/" not in str(getattr(evaluator, "__file__", "")):
        return None

    import torch

    spec = importlib.util.spec_from_file_location(
        f"_eplb_candidate_{abs(hash(program_path))}", program_path
    )
    if spec is None or spec.loader is None:
        return "EPLB topology guard could not import the candidate"
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    rebalance = getattr(candidate, "rebalance_experts", None)
    if rebalance is None:
        return "EPLB topology guard: missing rebalance_experts"

    workloads = evaluator.load_workloads(evaluator.WORKLOAD_PATH)
    # The topology invariant is structural, but checking both ends catches
    # implementations that branch on workload values at negligible cost next
    # to the full evaluator pass.
    probe_workloads = [workloads[0]]
    if len(workloads) > 1:
        probe_workloads.append(workloads[-1])

    replicas = int(evaluator.NUM_REPLICAS)
    groups = int(evaluator.NUM_GROUPS)
    nodes = int(evaluator.NUM_NODES)
    gpus = int(evaluator.NUM_GPUS)
    if replicas % nodes:
        return "EPLB topology guard: replicas are not divisible by nodes"
    slots_per_node = replicas // nodes

    for probe_index, workload in enumerate(probe_workloads):
        phy2log, log2phy, logcnt = rebalance(
            workload, replicas, groups, nodes, gpus
        )
        layers, logical_experts = workload.shape
        expected_phy_shape = (layers, replicas)
        expected_count_shape = (layers, logical_experts)
        if tuple(phy2log.shape) != expected_phy_shape:
            return (f"EPLB topology guard: phy2log shape {tuple(phy2log.shape)} "
                    f"!= {expected_phy_shape}")
        if tuple(logcnt.shape) != expected_count_shape:
            return (f"EPLB topology guard: logcnt shape {tuple(logcnt.shape)} "
                    f"!= {expected_count_shape}")
        if logical_experts % groups:
            return "EPLB topology guard: logical experts are not divisible by groups"
        if (phy2log < 0).any() or (phy2log >= logical_experts).any():
            return "EPLB topology guard: phy2log contains an invalid logical id"
        if (logcnt < 0).any():
            return "EPLB topology guard: logcnt contains a negative count"

        phy_cpu = phy2log.to(dtype=torch.int64, device="cpu")
        count_cpu = logcnt.to(dtype=torch.int64, device="cpu")
        group_size = logical_experts // groups
        node_of_slot = torch.arange(replicas, dtype=torch.int64) // slots_per_node
        for layer in range(layers):
            observed_counts = torch.bincount(
                phy_cpu[layer], minlength=logical_experts
            )
            if not torch.equal(observed_counts, count_cpu[layer]):
                return (f"EPLB topology guard: replica-count mismatch at "
                        f"probe {probe_index}, layer {layer}")

            logical_group_of_slot = phy_cpu[layer] // group_size
            groups_per_node = [set() for _ in range(nodes)]
            for group in range(groups):
                group_slots = logical_group_of_slot == group
                group_nodes = torch.unique(node_of_slot[group_slots]).tolist()
                if len(group_nodes) != 1:
                    return (
                        "EPLB topology guard: logical group "
                        f"{group} spans nodes {group_nodes} at probe "
                        f"{probe_index}, layer {layer}"
                    )
                groups_per_node[int(group_nodes[0])].add(group)
            expected_groups_per_node = groups // nodes
            if any(len(node_groups) != expected_groups_per_node
                   for node_groups in groups_per_node):
                return (
                    "EPLB topology guard: nodes do not each own exactly "
                    f"{expected_groups_per_node} logical groups"
                )
    return None


def _prism_success_error(evaluator, normalized_result):
    """Reject PRISM scores computed from a cherry-picked subset of test cases.

    The upstream scalar is ``1 / mean(KVPR of successes) + success_rate``.
    Without a validity gate, deliberately failing hard cases can increase the
    first term far more than the small success-rate penalty.  The benchmark
    requires a placement for every generated case, and all reference/main-table
    programs have success_rate=1.
    """
    if "/ADRS/prism/" not in str(getattr(evaluator, "__file__", "")):
        return None
    success_rate = float(
        (normalized_result.get("metrics") or {}).get("success_rate", 0.0)
    )
    if success_rate < 1.0 - 1e-12:
        return (
            "PRISM success guard: evaluator scored only "
            f"{success_rate:.1%} of generated test cases"
        )
    return None


def _is_txn_evaluator(evaluator) -> bool:
    return "/ADRS/txn_scheduling/" in str(
        getattr(evaluator, "__file__", "")
    )


def _evaluate_txn_guarded(evaluator, program_path: str):
    """Evaluate Transaction Scheduling with a complete-permutation guard.

    The imported ADRS evaluator validates a schedule against ``range(len(seq))``
    instead of the workload's transaction count.  A one-element schedule such
    as ``[0]`` therefore passes and receives an artificially large reciprocal
    score.  Run the candidate once, require one exact permutation per workload,
    and recompute the makespan from those guarded schedules.  This replaces the
    broken upstream scalar for every adaptation route rather than filtering
    particular programs or source patterns.
    """
    makespan, schedules = evaluator.run_with_timeout(
        program_path, timeout_seconds=600
    )

    from txn_simulator import Workload
    from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

    workloads = [Workload(raw) for raw in (WORKLOAD_1, WORKLOAD_2, WORKLOAD_3)]
    if not isinstance(schedules, (list, tuple)) or len(schedules) != len(workloads):
        return {
            "makespan": float(makespan) if isinstance(makespan, (int, float)) else 0.0,
            "schedule": float(len(schedules)) if isinstance(schedules, (list, tuple)) else 0.0,
            "validity": 0.0,
            "combined_score": 0.0,
            "error": (
                "Transaction legality guard: expected exactly three workload "
                "schedules"
            ),
            "txn_legality_guard": 0.0,
        }

    guarded_schedules = []
    for workload_index, (workload, schedule) in enumerate(zip(workloads, schedules)):
        if not isinstance(schedule, (list, tuple)):
            return {
                "makespan": float(makespan) if isinstance(makespan, (int, float)) else 0.0,
                "schedule": float(len(schedules)),
                "validity": 0.0,
                "combined_score": 0.0,
                "error": (
                    "Transaction legality guard: workload "
                    f"{workload_index} schedule is not a sequence"
                ),
                "txn_legality_guard": 0.0,
            }
        try:
            normalized_schedule = [operator.index(value) for value in schedule]
        except (TypeError, ValueError, OverflowError):
            normalized_schedule = []
        expected = list(range(workload.num_txns))
        if len(normalized_schedule) != workload.num_txns or sorted(normalized_schedule) != expected:
            return {
                "makespan": float(makespan) if isinstance(makespan, (int, float)) else 0.0,
                "schedule": float(len(schedules)),
                "validity": 0.0,
                "combined_score": 0.0,
                "error": (
                    "Transaction legality guard: workload "
                    f"{workload_index} must be an exact permutation of "
                    f"0..{workload.num_txns - 1}; got length "
                    f"{len(normalized_schedule)}"
                ),
                "txn_legality_guard": 0.0,
            }
        guarded_schedules.append(normalized_schedule)

    guarded_makespan = sum(
        workload.get_opt_seq_cost(schedule)
        for workload, schedule in zip(workloads, guarded_schedules)
    )
    return {
        "makespan": float(guarded_makespan),
        "reported_makespan": float(makespan),
        "schedule": float(len(guarded_schedules)),
        "validity": 1.0,
        "combined_score": float(1_000_000.0 / (1.0 + guarded_makespan)),
        "txn_legality_guard": 1.0,
    }


_HINT_MARKERS = ("Traceback", "Error", "error", "failed", "Failed", "overlap", "outside", "NaN", "Timeout", "timed out")


def _error_hint_from_output(captured: str, max_lines: int = 3, max_chars: int = 220):
    """Return the last few diagnostic lines an evaluator printed, or None.

    Several upstream evaluators (e.g. circle_packing) catch the candidate's
    exception, print it, and return a zero-score dict WITHOUT an ``error`` key,
    so the executor only ever saw ``validity = 0``: on 2026-09-21 43% of the
    circle-packing executor evaluations were silent SyntaxError/NameError
    crashes it could not debug. This surfaces the printed diagnostics for
    zero-score results only; positive results are untouched.
    """
    try:
        lines = [ln.strip() for ln in captured[-20000:].splitlines() if ln.strip()]
        hits = [ln for ln in lines if any(m in ln for m in _HINT_MARKERS)]
        if not hits:
            return None
        return " | ".join(ln[:max_chars] for ln in hits[-max_lines:])
    except Exception:
        return None


def _normalize(result):
    """Return {combined_score, validity, error, metrics} from any evaluator return."""
    # skydiscover EvaluationResult -> flatten .metrics; its artifacts carry the failure text (error / failure_stage) that
    # the GPU-mode evaluators never put into metrics, and a correctness of 0 is an invalid program, not a score of 0
    if hasattr(result, "metrics") and isinstance(getattr(result, "metrics"), dict):
        d = dict(result.metrics)
        arts = getattr(result, "artifacts", None)
        if isinstance(arts, dict) and arts.get("error") and not d.get("error"):
            stage = arts.get("failure_stage")
            d["error"] = f"[{stage}] {arts['error']}" if stage else str(arts["error"])
        if "correctness" in d and float(d.get("correctness") or 0.0) < 1.0 and "validity" not in d:
            d["validity"] = 0.0
            d.setdefault("error", "correctness check failed")
    elif hasattr(result, "to_dict"):
        d = dict(result.to_dict())
    elif isinstance(result, dict):
        d = dict(result)
    else:
        return {"combined_score": 0.0, "validity": 0.0,
                "error": f"unrecognized evaluator return type: {type(result)}", "metrics": {}}
    score = d.get("combined_score", d.get("score", 0.0))
    try:
        score = float(score)
    except Exception:
        score = 0.0
    return {
        "combined_score": score,
        "validity": float(d.get("validity", 1.0 if d.get("error") in (None, "") else 0.0)),
        "error": d.get("error"),
        "metrics": {k: v for k, v in d.items() if isinstance(v, (int, float))},
    }


def main() -> None:
    req = json.loads(Path(sys.argv[1]).read_text())
    out = {"combined_score": 0.0, "validity": 0.0, "error": None, "metrics": {}}
    try:
        if req.get("subsample"):
            # cheap-probe mode: evaluator sees only the first N rows of any CSV
            import pandas as _pd
            _orig_read_csv = _pd.read_csv
            _n = int(req["subsample"])

            def _sub_read_csv(*a, **kw):
                kw.setdefault("nrows", _n)
                return _orig_read_csv(*a, **kw)

            _pd.read_csv = _sub_read_csv
        ev_path = Path(req["evaluator_path"])
        # make the evaluator importable: its own dir, the runtime shim, and the task dir
        for p in (req.get("shim_path"), str(ev_path.parent), str(ev_path.parent.parent)):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        spec = importlib.util.spec_from_file_location("_eft_evaluator", str(ev_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        evaluate = getattr(mod, "evaluate")
        # The upstream Transaction evaluator has an incomplete-schedule
        # validity bug, so its guarded route executes exactly once and replaces
        # the scalar.  Other tasks use their upstream evaluator unchanged.
        captured = io.StringIO()
        if _is_txn_evaluator(mod):
            result = _evaluate_txn_guarded(mod, req["program_path"])
        else:
            # call with program_path only (all task evaluators accept this;
            # extras have defaults). Evaluator prints are captured so that a
            # zero-score result can carry the printed diagnostics.
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                try:
                    result = evaluate(req["program_path"])
                except TypeError:
                    sig = inspect.signature(evaluate)
                    if "config" in sig.parameters:
                        result = evaluate(req["program_path"], None)
                    else:
                        raise
        out = _normalize(result)
        if (out["combined_score"] == 0.0 or out["validity"] < 1.0) and out.get("error") in (None, ""):
            hint = _error_hint_from_output(captured.getvalue())
            if hint:
                out["error"] = f"evaluator output: {hint}"
        if out["validity"] >= 1.0 and out.get("error") in (None, ""):
            semantic_error = (
                _eplb_topology_error(mod, req["program_path"])
                or _prism_success_error(mod, out)
            )
            if semantic_error:
                out["metrics"]["unguarded_combined_score"] = out["combined_score"]
                out["combined_score"] = 0.0
                out["validity"] = 0.0
                out["error"] = semantic_error
    except Exception as e:  # never raise out of the worker
        out = {"combined_score": 0.0, "validity": 0.0,
               "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc().splitlines()[-3:], "metrics": {}}
    Path(req["result_path"]).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
