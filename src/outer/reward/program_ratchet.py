"""Route-independent program-incumbent transition for outer-loop campaigns.

The canonical reward-routing comparison uses ``strict_single``: one incumbent
per task, promoted only when the candidate program that actually produced the
reported score strictly beats both the incoming H2 base and the previous
program incumbent.  No QD elites or crossover parents are carried in this
mode.  Keeping this transition in one module prevents the proposer and context
arms from silently using different program memories.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from inner.runtime.package_hash import h2_sha256


STRICT_SINGLE = "strict_single"
LEGACY_QD = "legacy_qd"
SUPPORTED_MODES = {STRICT_SINGLE, LEGACY_QD}


def suppress_task_training(
    batch_rows: Iterable[Dict[str, Any]], task_id: str, reason: Optional[str],
) -> int:
    """Zero all task advantages when its apparent gain lacks program credit."""

    count = 0
    for row in batch_rows:
        if row.get("task_id") != task_id:
            continue
        row["advantage"] = 0.0
        row["training_suppressed"] = True
        row["training_suppression_reason"] = reason
        count += 1
    return count


def _result_files(round_dir: Path, task_id: str, k: Optional[int] = None) -> Iterable[Path]:
    root = Path(round_dir) / "rollouts" / task_id
    pattern = "cand*/*/results/*.json" if k is None else f"cand{k:02d}/*/results/*.json"
    return sorted(root.glob(pattern))


def _best_program_result(
    round_dir: Path, task_id: str, k: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for path in _result_files(round_dir, task_id, k):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict) or row.get("task_id") != task_id:
                continue
            program = row.get("best_program")
            score = row.get("best_score")
            if not program or score is None or row.get("score_eligible") is False:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if best is None or score > float(best["score"]):
                seed_provenance = row.get("seed_program_provenance") or {}
                best = {
                    "score": score,
                    "program": str(program),
                    "result_path": str(path),
                    "eval_time": (row.get("best_metrics") or {}).get("eval_time"),
                    "seed_score": row.get("seed_score"),
                    "seed_program_provenance": seed_provenance,
                    "h2_package_provenance": row.get(
                        "h2_package_provenance"
                    ) or {},
                }
    return best


def _eval_time_cap(task_id: str) -> Optional[float]:
    """Opt-in promotion guard against programs that sit at the evaluator's time
    limit (they score once and then time out as the next round's seed).

    ``SAH_RATCHET_EVAL_TIME_FRAC`` (e.g. 0.8) times the per-task limit from
    ``SAH_RATCHET_EVAL_LIMITS`` (JSON task_id -> seconds) or, by default,
    ``SAH9B_EVAL_TIMEOUT``. Unset/0 -> disabled (None).
    """
    try:
        frac = float(os.environ.get("SAH_RATCHET_EVAL_TIME_FRAC", "0") or 0)
    except ValueError:
        return None
    if frac <= 0:
        return None
    limits: Dict[str, Any] = {}
    raw = os.environ.get("SAH_RATCHET_EVAL_LIMITS", "").strip()
    if raw:
        try:
            limits = json.loads(raw) or {}
        except ValueError:
            limits = {}
    limit = limits.get(task_id, os.environ.get("SAH9B_EVAL_TIMEOUT"))
    try:
        return frac * float(limit)
    except (TypeError, ValueError):
        return None


def mark_over_cap_candidates(
    round_dir: Path, task_id: str, candidates: Iterable[Dict[str, Any]],
) -> list:
    """OVERCAP-001 (opt-in via ``SAH_OVER_CAP_AS_INVALID=1``).

    A *valid* proposal whose best program ran longer than the ratchet's
    eval-time cap (``_eval_time_cap``) is re-labelled as an INVALID proposal
    *before* rewards are computed: it takes ``INVALID_REWARD``, can neither
    become ``best_k`` nor a beam parent, and therefore no longer blocks the
    promotion of a slower-but-under-cap improvement nor suppresses the whole
    task's training (which the strict ratchet does when ``best_k`` itself is
    refused for ``candidate_eval_time_over_cap``).  The candidate dict is
    mutated in place (``valid=False``, ``invalid_reason``, ``over_cap``) and
    the list of marks ``{k, score, eval_time, cap}`` is returned (empty when
    the policy is off, the cap is unset, or nothing is over the cap).
    """
    if os.environ.get("SAH_OVER_CAP_AS_INVALID", "0") != "1":
        return []
    cap = _eval_time_cap(task_id)
    if cap is None:
        return []
    marks: list = []
    for cand in candidates:
        if not isinstance(cand, dict) or not cand.get("valid"):
            continue
        try:
            k = int(cand["k"])
        except (KeyError, TypeError, ValueError):
            continue
        result = _best_program_result(Path(round_dir), task_id, k)
        eval_time = None if result is None else result.get("eval_time")
        try:
            eval_time = None if eval_time is None else float(eval_time)
        except (TypeError, ValueError):
            eval_time = None
        if eval_time is None or eval_time <= cap:
            continue
        mark = {"k": k, "score": float(result["score"]), "eval_time": eval_time, "cap": float(cap)}
        cand["valid"] = False
        cand["invalid_reason"] = "eval_time_over_cap"
        cand["over_cap"] = mark
        marks.append(mark)
    return marks


def _incoming_score(bases_in: Dict[str, Any], task_id: str, fallback: float) -> float:
    entry = bases_in.get(task_id) or {}
    try:
        return float(entry.get("score", fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _normalise_single(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict) or not entry.get("program"):
        return None
    out = {
        "score": float(entry.get("score", float("-inf"))),
        "program": str(entry["program"]),
    }
    for key in ("round", "k", "result_path", "program_sha256"):
        if entry.get(key) is not None:
            out[key] = entry[key]
    out["program_sha256"] = hashlib.sha256(out["program"].encode()).hexdigest()
    return out


def update_strict_single(
    *, round_dir: Path, groups: Dict[str, Any], bases_in: Dict[str, Any],
    previous: Dict[str, Any], round_id: int, eps: float = 1e-12,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply the canonical single-incumbent transition without route knowledge."""

    state: Dict[str, Any] = {}
    for task_id, entry in (previous or {}).items():
        clean = _normalise_single(entry)
        if clean is not None:
            state[task_id] = clean

    audit: Dict[str, Any] = {
        "schema": "program-ratchet/1.0",
        "mode": STRICT_SINGLE,
        "round": int(round_id),
        "tasks": {},
    }
    try:
        round_meta = json.loads((Path(round_dir) / "round.json").read_text())
        expected_seed_registry_sha = (
            round_meta.get("cross_round_inputs", {})
            .get("seed_programs", {})
            .get("snapshot_sha256")
        )
    except Exception:
        expected_seed_registry_sha = None
    for task_id, group in groups.items():
        prior = state.get(task_id)
        group_base = float(group.get("base_score", float("-inf")))
        input_score = _incoming_score(bases_in, task_id, group_base)
        prior_score = float(prior.get("score", float("-inf"))) if prior else float("-inf")
        threshold = max(group_base, input_score, prior_score)
        best_k = group.get("program_best_k", group.get("best_k"))
        result = _best_program_result(round_dir, task_id, best_k) if best_k is not None else None
        group_score = group.get("program_best_score", group.get("best_score"))
        group_improved = group.get("program_improved", group.get("improved"))
        reason = "no_scored_program"
        promoted = False
        cap = _eval_time_cap(task_id)

        if result is not None and group_score is not None:
            reported = float(group_score)
            observed = float(result["score"])
            program_sha = hashlib.sha256(result["program"].encode()).hexdigest()
            seed_provenance = result.get("seed_program_provenance") or {}
            seed_sha = seed_provenance.get("program_sha256")
            seed_registry_sha = seed_provenance.get("registry_sha256")
            expected_seed_sha = prior.get("program_sha256") if prior else None
            h2_provenance = result.get("h2_package_provenance") or {}
            expected_h2 = (
                Path(round_dir) / "tasks" / task_id / f"cand{int(best_k):02d}"
            )
            if abs(reported - observed) > max(1e-9, 1e-8 * max(abs(reported), 1.0)):
                reason = "program_score_mismatch"
            elif not seed_sha:
                reason = "seed_program_provenance_missing"
            elif not expected_seed_registry_sha or not seed_registry_sha:
                reason = "seed_registry_provenance_missing"
            elif seed_registry_sha != expected_seed_registry_sha:
                reason = "seed_registry_provenance_mismatch"
            elif expected_seed_sha and seed_sha != expected_seed_sha:
                reason = "seed_program_mismatch"
            elif prior is None and seed_provenance.get("mode") != "task_initial":
                reason = "unexpected_inherited_seed"
            elif program_sha == seed_sha:
                reason = "no_program_change"
            elif result.get("seed_score") is None:
                reason = "seed_score_missing"
            elif not h2_provenance.get("sha256"):
                reason = "h2_package_provenance_missing"
            elif h2_provenance.get("stable_during_rollout") is not True:
                reason = "h2_package_mutated_during_rollout"
            elif not expected_h2.is_dir() \
                    or h2_sha256(expected_h2) != h2_provenance.get("sha256"):
                reason = "h2_package_provenance_mismatch"
            elif (
                cap is not None and result.get("eval_time") is not None
                and float(result["eval_time"]) > cap
            ):
                reason = "candidate_eval_time_over_cap"
            elif observed <= float(result["seed_score"]) + eps:
                reason = "did_not_beat_rollout_seed"
            elif not bool(group_improved):
                reason = "did_not_beat_h2_base"
            elif observed <= threshold + eps:
                reason = "did_not_beat_program_incumbent"
            else:
                program = result["program"]
                state[task_id] = {
                    "score": observed,
                    "program": program,
                    "round": int(round_id),
                    "k": int(best_k),
                    "result_path": result["result_path"],
                    "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                }
                promoted = True
                reason = "strict_improvement"

        audit["tasks"][task_id] = {
            "incoming_h2_score": input_score,
            "incoming_program_score": None if prior is None else prior_score,
            "threshold": threshold,
            "candidate_k": best_k,
            "candidate_score": None if result is None else result["score"],
            "candidate_eval_time": None if result is None else result.get("eval_time"),
            "eval_time_cap": cap,
            "candidate_program_sha256": (
                None if result is None
                else hashlib.sha256(result["program"].encode()).hexdigest()
            ),
            "rollout_seed_score": None if result is None else result.get("seed_score"),
            "rollout_seed_program_sha256": (
                None if result is None
                else (result.get("seed_program_provenance") or {}).get(
                    "program_sha256"
                )
            ),
            "expected_seed_registry_sha256": expected_seed_registry_sha,
            "rollout_seed_registry_sha256": (
                None if result is None
                else (result.get("seed_program_provenance") or {}).get(
                    "registry_sha256"
                )
            ),
            "rollout_h2_sha256": (
                None if result is None
                else (result.get("h2_package_provenance") or {}).get("sha256")
            ),
            "rollout_h2_stable": (
                None if result is None
                else (result.get("h2_package_provenance") or {}).get(
                    "stable_during_rollout"
                )
            ),
            "group_reported_score": group_score,
            "promoted": promoted,
            "reason": reason,
            "outgoing_program_sha256": (
                state.get(task_id, {}).get("program_sha256")
            ),
        }
    return state, audit


def update_legacy_qd(
    *, round_dir: Path, groups: Dict[str, Any], bases_in: Dict[str, Any],
    previous: Dict[str, Any], round_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Preserve the historical QD/crossover transition for non-paper runs."""

    state = copy.deepcopy(previous or {})
    audit: Dict[str, Any] = {
        "schema": "program-ratchet/1.0",
        "mode": LEGACY_QD,
        "round": int(round_id),
        "tasks": {},
    }
    for task_id, group in groups.items():
        best_k = group.get("program_best_k", group.get("best_k"))
        score = group.get("program_best_score", group.get("best_score"))
        prior = state.get(task_id, {})
        result = _best_program_result(round_dir, task_id, best_k) if best_k is not None else None
        promoted = False
        if result is not None and score is not None and float(prior.get("score", float("-inf"))) < float(score):
            parents = list(prior.get("parents") or []) if isinstance(prior, dict) else []
            if isinstance(prior, dict) and prior.get("program") and prior["program"] != result["program"]:
                parents.insert(0, {"score": prior["score"], "program": prior["program"]})
            state[task_id] = {
                "score": float(score), "program": result["program"],
                "round": int(round_id), "k": int(best_k), "parents": parents[:2],
            }
            promoted = True

        entry = state.get(task_id)
        if isinstance(entry, dict) and entry.get("program"):
            best_program = entry["program"]
            best_score = float(entry.get("score", 0.0))
            floor = best_score - 0.15 * abs(best_score)
            pool = [e for e in (entry.get("elites") or []) if e.get("program")]
            seen = []
            for path in _result_files(round_dir, task_id):
                try:
                    payload = json.loads(path.read_text())
                except Exception:
                    continue
                rows = payload if isinstance(payload, list) else [payload]
                for row in rows:
                    try:
                        program, candidate_score = row.get("best_program"), float(row.get("best_score"))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if program and candidate_score >= floor and program != best_program:
                        seen.append((candidate_score, program))
            similarity = lambda a, b: difflib.SequenceMatcher(None, a[:4000], b[:4000]).ratio()
            for candidate_score, program in sorted(seen, key=lambda item: -item[0]):
                if similarity(program, best_program) >= 0.7:
                    continue
                twin = next((e for e in pool if similarity(program, e["program"]) >= 0.7), None)
                if twin is not None:
                    if candidate_score > float(twin.get("score", float("-inf"))):
                        twin.update(score=candidate_score, program=program)
                    continue
                pool.append({"score": candidate_score, "program": program})
            pool = sorted(pool, key=lambda row: -float(row.get("score", 0.0)))[:3]
            if pool:
                entry["elites"] = pool
                entry["parents"] = [
                    {"score": row["score"], "program": row["program"]} for row in pool
                ]
        audit["tasks"][task_id] = {"promoted": promoted}
    return state, audit


def update_program_ratchet(
    *, mode: str, round_dir: Path, groups: Dict[str, Any], bases_in: Dict[str, Any],
    previous: Dict[str, Any], round_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if mode == STRICT_SINGLE:
        return update_strict_single(
            round_dir=round_dir, groups=groups, bases_in=bases_in,
            previous=previous, round_id=round_id,
        )
    if mode == LEGACY_QD:
        return update_legacy_qd(
            round_dir=round_dir, groups=groups, bases_in=bases_in,
            previous=previous, round_id=round_id,
        )
    raise ValueError(f"unsupported program ratchet mode {mode!r}; expected {sorted(SUPPORTED_MODES)}")
