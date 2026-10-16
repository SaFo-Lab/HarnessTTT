"""Run the M0 + H2 baseline over the EFT held-out tasks.

Modes
-----
--seed-only : evaluate each task's seed program only (no model needed). Produces
              the H_simple baseline table + validates the eval pipeline.
default     : run the H2 NexAU agent (Qwen3.5-9B via a vLLM endpoint) per task.

Outputs a per-task result JSON + a summary.json/summary.csv under --out, with
run provenance (plan.md §21).

Examples
--------
  # pipeline check, no model:
  python run_baseline.py --seed-only --tiers tier0_cpu_nosetup tier1_cpu_lightsetup

  # real baseline against a served endpoint:
  python run_baseline.py --tiers tier0_cpu_nosetup tier1_cpu_lightsetup \\
      --base-url http://127.0.0.1:8800/v1 --model qwen3.5-9b \\
      --max-evals 10 --eval-python /path/to/env/bin/python --out runs/inner_baseline
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put `src/` on path -> import inner.*

from inner.tasks.eft_task import load_tasks, get_task, EFTTask  # noqa: E402
from inner.evaluation.eval_runner import evaluate_program  # noqa: E402
from inner.runtime.package_hash import h2_sha256  # noqa: E402


def _has_executor_trajectory(trajectory) -> bool:
    """Return whether a saved history contains at least one executor turn."""
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


def _llm_transport_provenance(args) -> dict:
    """Describe the effective non-streaming timeout/retry layers.

    Candidate H2 packages are materialized with stream disabled. Recording
    both retry layers makes a command-line value of zero independently
    auditable instead of relying on argv alone.
    """

    retries = int(args.llm_max_retries)
    outer_attempts = 1
    return {
        "schema": "sah-llm-transport/1.0",
        "stream": False,
        "request_timeout_seconds": float(args.llm_timeout),
        "http_read_timeout_seconds": float(args.llm_timeout),
        "http_connection_reuse": (
            os.environ.get("NEXAU_HTTP_CONNECTION_CLOSE") != "1"
        ),
        "sdk_max_retries": retries,
        "agent_total_attempts": outer_attempts,
        "maximum_provider_attempts": outer_attempts * (retries + 1),
        "retry_scope": "provider_request_only",
        "trajectory_retry_attempts": 0,
    }


def _select(args) -> list:
    if args.ids:
        return [get_task(i) for i in args.ids]
    tiers = args.tiers or None
    tasks = load_tasks(include_simpletes_crossref=not args.no_simpletes, cost_tiers=tiers)
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def _seed_only(task: EFTTask, args) -> dict:
    out = evaluate_program(task, task.initial_program,
                           timeout_s=args.eval_timeout, python_exe=args.eval_python)
    return {
        "task_id": task.task_id, "source": task.source, "cost_tier": task.cost_tier,
        "mode": "seed_only", "seed_score": out.combined_score, "best_score": out.combined_score,
        "delta": 0.0, "validity": out.validity, "error": out.error, "wall_s": out.wall_s,
        "evaluations": 1,
    }


def _agent_run(task: EFTTask, args, out_dir: Path) -> dict:
    from inner.runtime.harness_runner import LLMEndpoint, H2Config, run_task  # lazy (needs nexau)

    ep = LLMEndpoint(model=args.model, base_url=args.base_url, api_key=args.api_key,
                     temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                     timeout=args.llm_timeout, max_retries=args.llm_max_retries,
                     enable_thinking=args.thinking,
                     seed=args.seed)
    if args.max_iters > 0:
        max_iters = args.max_iters
    elif args.harness_dir:
        max_iters = None  # candidate package: respect its own agent.yaml max_iterations
    else:
        max_iters = 3 * args.max_evals + 8
    h2 = H2Config(max_evaluator_calls=args.max_evals, max_iterations=max_iters,
                  eval_timeout_s=args.eval_timeout, python_exe=args.eval_python)
    ckpt = str(out_dir / "checkpoints" / f"{task.task_id}.json")
    h2_package = (
        Path(args.harness_dir).resolve()
        if args.harness_dir else Path(__file__).resolve().parents[1] / "harness"
    )
    h2_sha_before = h2_sha256(h2_package)
    res = run_task(task, endpoint=ep, h2=h2, keep_trajectory=not args.no_trajectory,
                   checkpoint_path=ckpt,
                   harness_dir=Path(args.harness_dir) if args.harness_dir else None)
    h2_sha_after = h2_sha256(h2_package)
    if h2_sha_after != h2_sha_before:
        audit_error = "H2MutationError: harness package changed during rollout"
        res.error = f"{res.error}; {audit_error}" if res.error else audit_error
        res.stop_reason = "h2_package_mutated"
        res.score_eligible = False
        if res.trajectory is not None:
            res.trajectory.append({"role": "framework", "content": audit_error})
    trajectory_ok = _has_executor_trajectory(res.trajectory)
    if args.require_trajectory and not trajectory_ok:
        audit_error = ("TrajectoryAuditError: required executor trajectory is "
                       "missing or contains no assistant turn")
        res.error = f"{res.error}; {audit_error}" if res.error else audit_error
        if res.stop_reason == "completed":
            res.stop_reason = "trajectory_missing"
        # Preserve the diagnostic program/score in ``_full`` while preventing
        # a missing conversation from entering reward, replay, or a curve.
        res.score_eligible = False
    full = asdict(res)
    full["h2_package_provenance"] = {
        "path": str(h2_package),
        "sha256": h2_sha_before,
        "sha256_after": h2_sha_after,
        "stable_during_rollout": h2_sha_before == h2_sha_after,
        "hash_scheme": "canonical-h2-v1",
    }
    seed_provenance = dict(getattr(task, "_seed_program_provenance", {}) or {})
    seed_program = task.initial_program
    seed_provenance.update({
        "program_sha256": hashlib.sha256(seed_program.encode()).hexdigest(),
        "observed_seed_score": float(res.seed_score),
    })
    full["seed_program_provenance"] = seed_provenance
    full["llm_transport_provenance"] = _llm_transport_provenance(args)
    published_best = res.best_score if res.score_eligible else None
    published_delta = (res.best_score - res.seed_score) if res.score_eligible else None
    return {**{k: v for k, v in full.items() if k != "trajectory"},
            "mode": "agent", "best_score": published_best,
            "delta": published_delta,
            "evaluations": res.ledger.get("evaluator_calls", 0),
            "_full": full, "_trajectory_ok": trajectory_ok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-only", action="store_true", help="evaluate seeds only (no model)")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="cost tiers to include, e.g. tier0_cpu_nosetup tier1_cpu_lightsetup")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit task_ids")
    ap.add_argument("--no-simpletes", action="store_true", help="exclude the 3 SimpleTES cross-ref tasks")
    ap.add_argument("--limit", type=int, default=0)
    # endpoint / sampling
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8800/v1"))
    ap.add_argument("--model", default=os.environ.get("INNER_MODEL", "qwen3.5-9b"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--llm-timeout", type=float, default=600.0)
    ap.add_argument("--llm-max-retries", type=int, default=2)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking (default off)")
    ap.add_argument("--seed", type=int, default=None,
                    help="explicit decode seed recorded in every H2 result")
    # budget / eval
    ap.add_argument("--max-evals", type=int, default=10, help="evaluator-call budget per task")
    ap.add_argument("--max-iters", type=int, default=0, help="agent-loop cap; 0 = auto (3*max_evals+8; candidate packages keep their agent.yaml value)")
    ap.add_argument("--harness-dir", default=None,
                    help="run a candidate H2 package (dir with agent.yaml) instead of the built-in inner/harness")
    ap.add_argument("--eval-timeout", type=float, default=None, help="override per-eval timeout (s)")
    ap.add_argument("--eval-python", default=os.environ.get("INNER_EVAL_PYTHON", sys.executable),
                    help="interpreter with the task deps (numpy/scipy/jax/...) for eval subprocess")
    trajectory_group = ap.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--no-trajectory", action="store_true",
        help="do not save agent history (manual debugging only; never use for reported runs)")
    trajectory_group.add_argument(
        "--require-trajectory", action="store_true",
        help="save agent history and fail the run unless it contains an executor turn")
    ap.add_argument("--seed-programs-file", default=None,
                    help="JSON {task_id: program_text | {program: text}}; overrides the task's initial program (best-program inheritance across outer rounds)")
    ap.add_argument("--out", default=None, help="output dir (default runs/inner_baseline/<ts>)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    default_root = Path(__file__).resolve().parents[3] / "runs" / "inner_baseline"
    out_dir = (Path(args.out) / ts) if args.out else (default_root / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)

    tasks = _select(args)
    seed_registry_path = (
        Path(args.seed_programs_file).resolve()
        if args.seed_programs_file else None
    )
    seed_registry_sha256 = None
    if seed_registry_path is not None and seed_registry_path.is_file():
        seed_registry_sha256 = hashlib.sha256(
            seed_registry_path.read_bytes()
        ).hexdigest()
    for task in tasks:
        initial = task.initial_program
        task._seed_program_provenance = {
            "mode": "task_initial",
            "program_path": str(Path(task.initial_program_path).resolve()),
            "program_sha256": hashlib.sha256(initial.encode()).hexdigest(),
            "registry_path": str(seed_registry_path) if seed_registry_path else None,
            "registry_sha256": seed_registry_sha256,
            "registry_entry_present": False,
            "claimed_score": None,
        }
    if args.seed_programs_file and Path(args.seed_programs_file).exists():
        # A missing/empty seed-programs file means "no inheritance yet" (fresh
        # workspace, e.g. a new adaptive A/B run) — fall back to each task's
        # initial program instead of crashing the whole rollout (FileNotFound
        # took down all 8 hadamard rollouts on adaptive round940).
        try:
            inherited = json.loads(Path(args.seed_programs_file).read_text() or "{}")
        except Exception as e:
            print(f"[inherit] WARNING: seed-programs-file unreadable ({e}); "
                  f"starting from each task's initial program")
            inherited = {}
        for task in tasks:
            ent = inherited.get(task.task_id)
            if not ent:
                continue
            text = ent["program"] if isinstance(ent, dict) else ent
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"seed-program registry entry for {task.task_id} has no program text"
                )
            ov = out_dir / f"seed_override__{task.task_id}.py"
            ov.write_text(text)
            task.initial_program_path = ov
            task._seed_program_provenance = {
                "mode": "inherited_registry",
                "program_path": str(ov.resolve()),
                "program_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "registry_path": str(seed_registry_path),
                "registry_sha256": seed_registry_sha256,
                "registry_entry_present": True,
                "claimed_score": (
                    float(ent["score"])
                    if isinstance(ent, dict) and ent.get("score") is not None
                    else None
                ),
                "registry_round": ent.get("round") if isinstance(ent, dict) else None,
                "registry_k": ent.get("k") if isinstance(ent, dict) else None,
            }
            parents = ent.get("parents") if isinstance(ent, dict) else None
            if parents:
                task.crossover_parents = parents  # consumed by harness_runner
            print(f"[inherit] {task.task_id}: rollout starts from inherited best program "
                  f"({len(text)} chars, {len(parents or [])} crossover parents)", flush=True)
    provenance = {
        "timestamp": ts, "mode": "seed_only" if args.seed_only else "agent",
        "n_tasks": len(tasks), "task_ids": [t.task_id for t in tasks],
        "base_url": args.base_url, "model": args.model, "temperature": args.temperature,
        "top_p": args.top_p, "max_tokens": args.max_tokens, "max_evals": args.max_evals,
        "decode_seed": args.seed,
        "llm_transport": (
            None if args.seed_only else _llm_transport_provenance(args)
        ),
        "seed_program_registry": {
            "path": str(seed_registry_path) if seed_registry_path else None,
            "sha256": seed_registry_sha256,
        },
        "eval_python": args.eval_python,
        "trajectory_policy": ("required" if args.require_trajectory else
                              "disabled" if args.no_trajectory else "saved"),
        "generated_middleware_policy": (
            "gate+compile+mount+invoke>=1+errors=0; fires may be 0; "
            "dict returns may enforce a require_tools gate (finish exempt, "
            "auto-lift after 2 refusals, refusals consume no budget)"
        ),
        "argv": sys.argv,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[run] {provenance['mode']} | {len(tasks)} tasks | out={out_dir}")

    rows = []
    trajectory_failures = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.task_id} ({task.cost_tier}) ...", flush=True)
        trajectory_ok = None
        try:
            row = _seed_only(task, args) if args.seed_only else _agent_run(task, args, out_dir)
        except Exception as e:
            row = {"task_id": task.task_id, "source": task.source, "mode": "error",
                   "seed_score": None, "best_score": None, "delta": None,
                   "error": f"{type(e).__name__}: {e}"}
        trajectory_ok = row.pop("_trajectory_ok", trajectory_ok)
        full = row.pop("_full", None)
        (out_dir / "results" / f"{task.task_id}.json").write_text(
            json.dumps(full if full is not None else row, indent=2))
        rows.append(row)
        if args.require_trajectory and not args.seed_only and trajectory_ok is not True:
            trajectory_failures.append(task.task_id)
        print(f"      best={row.get('best_score')} seed={row.get('seed_score')} "
              f"delta={row.get('delta')} evals={row.get('evaluations')} err={row.get('error')}")

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    cols = ["task_id", "source", "cost_tier", "mode", "seed_score", "best_score",
            "delta", "validity", "evaluations", "score_eligible", "stop_reason",
            "error"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    scored = [r for r in rows if r.get("best_score") is not None]
    if scored and not args.seed_only:
        improved = sum(1 for r in scored if (r.get("delta") or 0) > 1e-9)
        print(f"\n[done] {len(scored)} scored | improved over seed: {improved}/{len(scored)}")
    print(f"[done] summary -> {out_dir}/summary.csv")
    if trajectory_failures:
        joined = ", ".join(trajectory_failures)
        print(f"[trajectory-audit] ERROR: missing required executor history: {joined}",
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
