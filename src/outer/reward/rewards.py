"""Rewards + GRPO advantages, instance-wise (plan.md §2.2, §8.3).

Per task instance tau, K candidates are generated FOR tau. Legacy/v2/v3 modes
score them against tau's historical current-best-harness scalar:

    r = clip((score - base_score) / (|base_score| + eps), -1, 1)

Failed/absent rollouts score 0 -> r = -1 after clipping; an INVALID candidate
(no submission / schema failure / duplicate) gets the fixed INVALID_REWARD.
The GRPO group = the K candidates of ONE task:

    A_k = (r_k - mean_group) / (std_group + eps)

Paired mode instead runs each candidate and its incoming parent H2 under one or
more matched decode seeds and rewards their mean counterfactual difference.
Group-relative normalization within the task removes cross-task scale.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from outer.reward.component_evidence import component_evidence, promotion_gate

EPS = 1e-9
INVALID_REWARD = -1.0
# Asymmetric clip: relative gain is naturally bounded below by -1 (scores >= 0);
# the upper bound must NOT flatten winner ordering (step-3 lesson: 0.37/0.42/0.51
# on a 0.14 base all clipped to +1 and got identical advantages).
CLIP_LOW, CLIP_HIGH = -1.0, 5.0


def task_reward(score: Optional[float], base_score: float) -> float:
    if score is None:
        score = 0.0
    r = (score - base_score) / (abs(base_score) + EPS)
    return max(CLIP_LOW, min(CLIP_HIGH, r))


def load_rollout_score(rollout_dir: Path, task_id: str) -> Optional[float]:
    """Best score for ``task_id`` from a candidate's run_baseline output
    (final summary authoritative; wall-safe checkpoint only if no terminal task
    row exists).

    ``run_baseline`` evaluates the seed before constructing the candidate
    harness, so a harness ConfigError can leave a valid seed checkpoint next to
    an explicit terminal row whose ``best_score`` is null.  Falling back in
    that case falsely credits a broken harness with the incumbent score and
    leaks it into analyzer feedback.  Checkpoints are therefore used only for
    genuinely interrupted runs that never wrote a terminal row for this task.
    """
    best: Optional[float] = None
    saw_terminal_task_row = False
    for summ in rollout_dir.glob("*/summary.json"):
        try:
            payload = json.loads(summ.read_text())
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if not isinstance(row, dict) or row.get("task_id") != task_id:
                    continue
                saw_terminal_task_row = True
                if row.get("score_eligible") is not False \
                        and row.get("best_score") is not None:
                    s = float(row["best_score"])
                    best = s if best is None else max(best, s)
        except Exception:
            pass
    if saw_terminal_task_row:
        return best
    if best is None:
        for ck in rollout_dir.glob(f"*/checkpoints/{task_id}.json"):
            try:
                s = float(json.loads(ck.read_text()).get("best_score", 0.0))
                best = s if best is None else max(best, s)
            except Exception:
                pass
    return best


def _load_scored_results(
    rollout_dir: Path, task_id: str,
) -> List[Dict[str, Any]]:
    """Return every eligible terminal result plus its run provenance.

    Paired-repeat artifacts add a stable ``repeatNN`` directory before the
    timestamped directory created by ``run_baseline``.  Search recursively so
    both the legacy ``candNN/<timestamp>`` layout and the repeat-complete
    ``candNN/repeatNN/<timestamp>`` layout are consumed.  Duplicate decode
    seeds are still rejected by the paired caller, so recursive discovery
    cannot silently cherry-pick one of multiple terminal runs.
    """

    found: List[Dict[str, Any]] = []
    for result_path in rollout_dir.rglob("results/*.json"):
        try:
            payload = json.loads(result_path.read_text())
            rows = payload if isinstance(payload, list) else [payload]
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("task_id") != task_id:
                continue
            if row.get("score_eligible") is False or row.get("best_score") is None:
                continue
            try:
                score = float(row["best_score"])
            except (TypeError, ValueError):
                continue
            run_dir = result_path.parent.parent
            try:
                provenance = json.loads((run_dir / "provenance.json").read_text())
            except Exception:
                provenance = None
            item = {
                "score": score,
                "result": row,
                "result_path": str(result_path),
                "provenance": provenance,
            }
            found.append(item)
    return found


def _paired_contract_issues(
    *, task_id: str, candidate: Dict[str, Any], control: Dict[str, Any],
    expected_candidate_h2_sha256: Optional[str] = None,
    expected_control_h2_sha256: Optional[str] = None,
    expected_compute_contract: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate that two terminal results form an auditable counterfactual."""

    issues: List[str] = []
    cand_result = candidate["result"]
    ctrl_result = control["result"]
    cand_prov = candidate.get("provenance")
    ctrl_prov = control.get("provenance")
    if not isinstance(cand_prov, dict) or not isinstance(ctrl_prov, dict):
        return ["run provenance missing"]

    def require_equal(label: str, left: Any, right: Any) -> None:
        if left is None or right is None:
            issues.append(f"{label} missing")
        elif left != right:
            issues.append(f"{label} mismatch: candidate={left!r} control={right!r}")

    require_equal(
        "decode_seed", cand_result.get("decode_seed"), ctrl_result.get("decode_seed")
    )
    require_equal(
        "provenance.decode_seed",
        cand_prov.get("decode_seed"), ctrl_prov.get("decode_seed"),
    )
    require_equal("model", cand_prov.get("model"), ctrl_prov.get("model"))
    require_equal("task_ids", cand_prov.get("task_ids"), ctrl_prov.get("task_ids"))
    if cand_prov.get("task_ids") != [task_id]:
        issues.append(
            f"candidate task_ids do not bind exactly {task_id!r}: "
            f"{cand_prov.get('task_ids')!r}"
        )
    require_equal("max_evals", cand_prov.get("max_evals"), ctrl_prov.get("max_evals"))
    require_equal(
        "ledger.max_evaluator_calls",
        (cand_result.get("ledger") or {}).get("max_evaluator_calls"),
        (ctrl_result.get("ledger") or {}).get("max_evaluator_calls"),
    )
    require_equal(
        "ledger.max_probe_calls",
        (cand_result.get("ledger") or {}).get("max_probe_calls"),
        (ctrl_result.get("ledger") or {}).get("max_probe_calls"),
    )
    cap_fields = (
        ("model_max_context_tokens", "max_executor_context_tokens", None),
        ("max_executor_model_calls", "max_executor_model_calls", "llm_calls"),
        ("max_executor_input_tokens", "max_executor_input_tokens", "llm_input_tokens"),
        ("max_executor_output_tokens", "max_executor_output_tokens", "llm_output_tokens"),
        ("max_full_evaluations", "max_evaluator_calls", "evaluator_calls"),
        ("max_probe_cost", "max_probe_calls", "probe_calls"),
        ("max_sandbox_seconds", "max_sandbox_seconds", "sandbox_seconds"),
    )
    compute = expected_compute_contract or {}
    for label, result in (("candidate", cand_result), ("control", ctrl_result)):
        ledger = result.get("ledger") or {}
        for contract_key, cap_key, usage_key in cap_fields:
            if contract_key not in compute:
                continue  # read compatibility for pre-contract artifacts
            expected = compute[contract_key]
            observed_cap = ledger.get(cap_key)
            if observed_cap is None:
                issues.append(f"{label} ledger.{cap_key} missing")
            elif float(observed_cap) != float(expected):
                issues.append(
                    f"{label} ledger.{cap_key} mismatch: "
                    f"expected={expected!r} observed={observed_cap!r}"
                )
            if usage_key is None:
                continue
            usage = ledger.get(usage_key)
            if usage is None:
                issues.append(
                    f"{label} ledger.{usage_key} missing under metered compute contract"
                )
            elif float(usage) > float(expected) + EPS:
                issues.append(
                    f"{label} ledger.{usage_key} exceeds {contract_key}: "
                    f"usage={usage!r} cap={expected!r}"
                )
    cand_seed = cand_result.get("seed_program_provenance") or {}
    ctrl_seed = ctrl_result.get("seed_program_provenance") or {}
    require_equal(
        "seed_program_sha256",
        cand_seed.get("program_sha256"), ctrl_seed.get("program_sha256"),
    )
    require_equal(
        "seed_registry_sha256",
        cand_seed.get("registry_sha256"), ctrl_seed.get("registry_sha256"),
    )
    require_equal(
        "provenance.seed_registry_sha256",
        (cand_prov.get("seed_program_registry") or {}).get("sha256"),
        (ctrl_prov.get("seed_program_registry") or {}).get("sha256"),
    )
    for label, result in (("candidate", cand_result), ("control", ctrl_result)):
        h2 = result.get("h2_package_provenance") or {}
        if not h2.get("sha256"):
            issues.append(f"{label} H2 sha256 missing")
        if h2.get("stable_during_rollout") is not True:
            issues.append(f"{label} H2 was not stable during rollout")
    control_h2 = ctrl_result.get("h2_package_provenance") or {}
    candidate_h2 = cand_result.get("h2_package_provenance") or {}
    if expected_candidate_h2_sha256 and (
        candidate_h2.get("sha256") != expected_candidate_h2_sha256
    ):
        issues.append(
            "candidate H2 is not the declared materialized proposal: "
            f"expected={expected_candidate_h2_sha256!r} "
            f"observed={candidate_h2.get('sha256')!r}"
        )
    if expected_control_h2_sha256 and (
        control_h2.get("sha256") != expected_control_h2_sha256
    ):
        issues.append(
            "control H2 is not the declared incoming parent: "
            f"expected={expected_control_h2_sha256!r} "
            f"observed={control_h2.get('sha256')!r}"
        )
    return issues


def compute_task_group(
    *, task_id: str, candidates: List[Dict[str, Any]], rollout_root: Path,
    base_score: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        k = cand["k"]
        if not cand.get("valid"):
            score, reward = None, INVALID_REWARD
        else:
            score = load_rollout_score(rollout_root / task_id / f"cand{k:02d}", task_id)
            reward = task_reward(score, base_score)
        rows.append({"k": k, "valid": bool(cand.get("valid")), "score": score,
                     "reward": reward, "spec_hash": cand.get("spec_hash", ""),
                     "changed_fields": cand.get("changed_fields", []),
                     "review_log": cand.get("review_log", [])})

    rewards = [r["reward"] for r in rows]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    for r in rows:
        r["advantage"] = (r["reward"] - mean) / (std + EPS)

    valid = [r for r in rows if r["valid"] and r["score"] is not None]
    best = max(valid, key=lambda r: r["score"]) if valid else None
    return {"task_id": task_id, "base_score": base_score,
            "reward_mean": mean, "reward_std": std, "rows": rows,
            "best_k": best["k"] if best else None,
            "best_score": best["score"] if best else None,
            "improved": bool(best and best["score"] > base_score)}

# --------------------------------------------------------------------------- #
# v2 discovery-oriented rewards/advantages (2026-07-24 algorithm exploration)
#
# Motivations, each backed by campaign evidence:
#   1. GAP-NORMALIZED REWARD  — near saturation the relative-gain reward
#      vanishes (round019 AC1: +0.0001-level rewards) and GRPO's std division
#      inflates that noise into +/-2.6 advantages. Reward becomes "fraction of
#      the remaining gap to the task ceiling closed":  r = (s-b)/(ceil-b).
#   2. RLOO BASELINE, NO STD DIVISION — leave-one-out mean is unbiased at
#      small K and stops noise amplification.
#   3. MAX-WEIGHTED SHARPENING — discovery banks best-of-K, not the mean.
#      A = alpha*(softmax(r/tau) - 1/K) + (1-alpha)*A_rloo  approximates a
#      risk-seeking / E[max] gradient while retaining a stable mean term.
#   4. ZERO-SIGNAL FILTER — groups whose valid rewards are all identical
#      (round014 llm_sql: eight 0.0934s) carry only "submit anything valid"
#      pressure, which is drift fuel; their advantages are zeroed so the
#      training step skips them.
# --------------------------------------------------------------------------- #

def task_reward_v2(score: Optional[float], base_score: float,
                   ceiling: Optional[float]) -> float:
    if score is None:
        score = 0.0
    if ceiling is not None and ceiling > base_score + EPS:
        r = (score - base_score) / (ceiling - base_score)
        # soft cap: bounded ~3 but strictly monotonic, so two above-ceiling
        # candidates keep their ordering (hard clip made r020 cand0/cand4 tie)
        return 3.0 * math.tanh(r / 3.0) if r > 0 else max(-1.0, r)
    return task_reward(score, base_score)


def _rloo(rewards: List[float]) -> List[float]:
    n = len(rewards)
    if n < 2:
        return [0.0 for _ in rewards]
    tot = sum(rewards)
    return [r - (tot - r) / (n - 1) for r in rewards]


def _max_weights(rewards: List[float], tau: float) -> List[float]:
    m = max(rewards)
    exps = [math.exp((r - m) / max(tau, EPS)) for r in rewards]
    z = sum(exps)
    n = len(rewards)
    return [e / z - 1.0 / n for e in exps]


def compute_task_group_v2(
    *, task_id: str, candidates: List[Dict[str, Any]], rollout_root: Path,
    base_score: float, ceiling: Optional[float] = None,
    sharpen_alpha: float = 0.3, zero_range_eps: float = 1e-3,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        k = cand["k"]
        repaired = bool((cand.get("repair") or {}).get("succeeded"))
        if not cand.get("valid"):
            score, reward = None, INVALID_REWARD
        else:
            score = load_rollout_score(rollout_root / task_id / f"cand{k:02d}", task_id)
            reward = task_reward_v2(score, base_score, ceiling)
            if repaired:
                # REPAIR-001: phi trains at minimum reward for a submission
                # the frozen-M0 repair agent produced; the H2 itself stays a
                # legal candidate for the round.
                reward = INVALID_REWARD
        rows.append({"k": k, "valid": bool(cand.get("valid")),
                     "repaired": repaired, "score": score,
                     "reward": reward, "spec_hash": cand.get("spec_hash", ""),
                     "changed_fields": cand.get("changed_fields", []),
                     "review_log": cand.get("review_log", [])})

    valid_rewards = [r["reward"] for r in rows if r["valid"]]
    rng = (max(valid_rewards) - min(valid_rewards)) if valid_rewards else 0.0
    all_valid = all(r["valid"] for r in rows)
    no_signal = all_valid and rng < zero_range_eps
    if valid_rewards and rng < zero_range_eps and not all_valid:
        # valids are effectively tied: collapse their micro-differences so the
        # only surviving signal is valid-vs-invalid (no fake ordering from 1e-6
        # score noise — the r014 llm_sql pathology)
        vm = sum(valid_rewards) / len(valid_rewards)
        for r in rows:
            if r["valid"]:
                r["reward"] = vm
    # REPAIR-001 invariant survives any collapse: a repaired slot always
    # trains at exactly the minimum reward.
    for r in rows:
        if r.get("repaired"):
            r["reward"] = INVALID_REWARD
    rewards = [r["reward"] for r in rows]

    if no_signal:
        for r in rows:
            r["advantage"] = 0.0
    else:
        base_adv = _rloo(rewards)
        tau = max(rng / 4.0, 1e-6)
        mw = _max_weights(rewards, tau)
        for r, a, w in zip(rows, base_adv, mw):
            r["advantage"] = (1.0 - sharpen_alpha) * a + sharpen_alpha * w

    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    valid = [r for r in rows if r["valid"] and r["score"] is not None]
    best = max(valid, key=lambda r: r["score"]) if valid else None
    return {"task_id": task_id, "base_score": base_score, "ceiling": ceiling,
            "reward_mean": mean, "reward_std": std, "rows": rows,
            "adv_mode": "no_signal" if no_signal else f"rloo+max(a={sharpen_alpha})",
            "best_k": best["k"] if best else None,
            "best_score": best["score"] if best else None,
            "improved": bool(best and best["score"] > base_score)}


def compute_task_group_paired(
    *, task_id: str, candidates: List[Dict[str, Any]], rollout_root: Path,
    control_root: Path, base_score: float, ceiling: Optional[float] = None,
    sharpen_alpha: float = 0.3, zero_range_eps: float = 1e-3,
    min_effect: float = 1e-4,
    expected_control_h2_sha256: Optional[str] = None,
    expected_repeats: int = 1,
    min_promotion_repeats: int = 3,
) -> Dict[str, Any]:
    """Credit H1 only for improvement over a same-seed parent-H2 rollout.

    An executor can discover a strong program for reasons already present in
    the public task or from sampling noise.  Comparing that score only with a
    historical scalar incorrectly assigns the discovery to the candidate H2.
    In paired mode every candidate rollout has a counterfactual control: the
    incoming parent harness, same task/program/evaluation budget/model/decode
    seed.  Proposer reward is the paired effect ``candidate - control``.

    ``base_score`` remains the historical ratchet.  H2 promotion is deliberately
    conservative: the causal winner must both beat its paired parent and exceed
    the historical score.  Absolute-best diagnostics are retained separately.
    A missing parent control for a valid proposal fails closed because no causal
    credit can be computed.
    """

    expected_repeats = int(expected_repeats)
    if expected_repeats < 1:
        raise ValueError("expected_repeats must be >= 1")
    min_promotion_repeats = int(min_promotion_repeats)
    if min_promotion_repeats < 1:
        raise ValueError("min_promotion_repeats must be >= 1")

    def by_decode_seed(
        results: List[Dict[str, Any]], label: str,
    ) -> Dict[int, Dict[str, Any]]:
        keyed: Dict[int, Dict[str, Any]] = {}
        for item in results:
            raw_seed = item["result"].get("decode_seed")
            if raw_seed is None:
                raise ValueError(f"{label}: terminal result has no decode_seed")
            seed = int(raw_seed)
            if seed in keyed:
                raise ValueError(f"{label}: duplicate terminal decode_seed {seed}")
            keyed[seed] = item
        return keyed

    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        k = int(cand["k"])
        proposal_valid = bool(cand.get("valid"))
        parent_compute = cand.get("parent_compute_contract")
        candidate_compute = cand.get("candidate_compute_contract")
        if proposal_valid:
            if not isinstance(parent_compute, dict) or not isinstance(
                candidate_compute, dict
            ):
                raise ValueError(
                    f"{task_id}/cand{k:02d}: H2 compute contract missing; "
                    "refusing unaudited paired credit"
                )
            if parent_compute.get("sha256") != candidate_compute.get("sha256"):
                raise ValueError(
                    f"{task_id}/cand{k:02d}: candidate/control H2 compute "
                    f"signatures differ: parent={parent_compute} "
                    f"candidate={candidate_compute}"
                )
        expected_candidate_h2_sha256 = cand.get("package_sha256")
        if proposal_valid and not expected_candidate_h2_sha256:
            raise ValueError(
                f"{task_id}/cand{k:02d}: materialized candidate H2 hash "
                "missing; refusing unbound proposer credit"
            )
        candidate_results = (
            _load_scored_results(
                rollout_root / task_id / f"cand{k:02d}", task_id
            )
            if proposal_valid else []
        )
        control_results = _load_scored_results(
            control_root / task_id / f"cand{k:02d}", task_id
        )
        if proposal_valid and len(control_results) != expected_repeats:
            raise ValueError(
                f"{task_id}/cand{k:02d}: expected {expected_repeats} eligible "
                f"parent-control repeat(s), found {len(control_results)}; "
                "refusing unpaired proposer credit"
            )
        if proposal_valid and len(candidate_results) > expected_repeats:
            raise ValueError(
                f"{task_id}/cand{k:02d}: found {len(candidate_results)} "
                f"eligible candidate repeats, expected {expected_repeats}; "
                "refusing ambiguous/cherry-picked proposer credit"
            )

        pair_issues: List[str] = []
        pair_samples: List[Dict[str, Any]] = []
        candidate_complete = bool(
            proposal_valid and len(candidate_results) == expected_repeats
        )
        score: Optional[float] = None
        control_score: Optional[float] = None
        if proposal_valid:
            control_score = sum(
                float(item["score"]) for item in control_results
            ) / expected_repeats
        if candidate_complete:
            candidate_by_seed = by_decode_seed(
                candidate_results, f"{task_id}/cand{k:02d}/candidate"
            )
            control_by_seed = by_decode_seed(
                control_results, f"{task_id}/cand{k:02d}/control"
            )
            if set(candidate_by_seed) != set(control_by_seed):
                raise ValueError(
                    f"{task_id}/cand{k:02d}: candidate/control decode-seed "
                    f"sets differ: {sorted(candidate_by_seed)} vs "
                    f"{sorted(control_by_seed)}"
                )
            for decode_seed in sorted(candidate_by_seed):
                candidate_result = candidate_by_seed[decode_seed]
                control_result = control_by_seed[decode_seed]
                issues = _paired_contract_issues(
                    task_id=task_id,
                    candidate=candidate_result,
                    control=control_result,
                    expected_candidate_h2_sha256=
                        expected_candidate_h2_sha256,
                    # BEAM-001: under beam inheritance a slot's control rolls
                    # out ITS OWN parent, so the expected control hash is the
                    # slot's parent when one is recorded, else the task head.
                    expected_control_h2_sha256=(
                        cand.get("parent_package_sha256")
                        or expected_control_h2_sha256
                    ),
                    expected_compute_contract=candidate_compute,
                )
                pair_issues.extend(
                    f"seed {decode_seed}: {issue}" for issue in issues
                )
                pair_samples.append({
                    "decode_seed": decode_seed,
                    "candidate_score": candidate_result["score"],
                    "control_score": control_result["score"],
                    "causal_delta": (
                        float(candidate_result["score"])
                        - float(control_result["score"])
                    ),
                    "candidate_result": candidate_result["result_path"],
                    "control_result": control_result["result_path"],
                })
            if pair_issues:
                raise ValueError(
                    f"{task_id}/cand{k:02d}: paired provenance contract "
                    f"failed: {'; '.join(pair_issues)}"
                )
            score = sum(
                float(item["candidate_score"]) for item in pair_samples
            ) / expected_repeats

        attribution_valid = bool(
            proposal_valid and score is not None and control_score is not None
        )
        activation = component_evidence(
            cand,
            [item["result"] for item in candidate_results]
            if candidate_complete else [],
        )
        causal_delta: Optional[float] = None
        component_credit_delta: Optional[float] = None
        if not proposal_valid:
            reward = INVALID_REWARD
            attribution_status = "invalid_h1_proposal"
        elif score is None:
            reward = INVALID_REWARD
            attribution_status = (
                "candidate_rollout_ineligible_or_incomplete:"
                f"{len(candidate_results)}/{expected_repeats}"
            )
        else:
            assert control_score is not None
            causal_delta = float(score) - float(control_score)
            component_credit_delta = causal_delta
            reward = task_reward_v2(score, float(control_score), ceiling)
            attribution_status = "paired_same_seed"
            if activation["activation_required"] and not activation[
                "positive_credit_eligible"
            ]:
                component_credit_delta = min(causal_delta, 0.0)
                reward = min(float(reward), 0.0)
                attribution_status += ":" + str(activation["status"])

        repaired = bool((cand.get("repair") or {}).get("succeeded"))
        if repaired and proposal_valid:
            # REPAIR-001 (user-directed credit rule): the slot phi failed
            # trains at the MINIMUM reward — the frozen-M0-repaired H2 still
            # rolls out, keeps its measured paired outcome for diagnostics,
            # and may win the round for the harness lineage, but phi gets no
            # credit for a submission it did not produce.
            reward = INVALID_REWARD
            attribution_status = "repaired_by_frozen_m0_no_phi_credit"

        promotion = promotion_gate(
            evidence=activation,
            pair_samples=pair_samples,
            score=score,
            base_score=base_score,
            causal_delta=component_credit_delta,
            min_effect=min_effect,
            min_repeats=min_promotion_repeats,
        )

        rows.append({
            "k": k,
            "repaired": repaired,
            "valid": proposal_valid,
            "score": score,
            "control_score": control_score,
            "causal_delta": causal_delta,
            "component_credit_delta": component_credit_delta,
            "component_evidence": activation,
            "attribution_valid": attribution_valid,
            "attribution_status": attribution_status,
            "promotion": promotion,
            "promotion_eligible": bool(
                attribution_valid and not repaired and promotion["eligible"]
            ),
            "pair_contract": {
                "passed": not pair_issues if attribution_valid else None,
                "issues": pair_issues,
                "expected_repeats": expected_repeats,
                "observed_candidate_repeats": len(candidate_results),
                "observed_control_repeats": len(control_results),
                "samples": pair_samples,
                "expected_candidate_h2_sha256":
                    expected_candidate_h2_sha256,
                "expected_control_h2_sha256": expected_control_h2_sha256,
            },
            "reward": reward,
            "spec_hash": cand.get("spec_hash", ""),
            "changed_fields": cand.get("changed_fields", []),
            "review_log": cand.get("review_log", []),
        })

    credited_rewards = [
        float(row["reward"]) for row in rows if row["attribution_valid"]
    ]
    reward_range = (
        max(credited_rewards) - min(credited_rewards)
        if credited_rewards else 0.0
    )
    all_creditable = all(row["attribution_valid"] for row in rows)
    no_signal = all_creditable and reward_range < zero_range_eps
    if credited_rewards and reward_range < zero_range_eps and not all_creditable:
        tied_mean = sum(credited_rewards) / len(credited_rewards)
        for row in rows:
            if row["attribution_valid"]:
                row["reward"] = tied_mean
    # REPAIR-001 invariant survives any collapse.
    for row in rows:
        if row.get("repaired"):
            row["reward"] = INVALID_REWARD
        evidence = row.get("component_evidence") or {}
        if evidence.get("activation_required") and not evidence.get(
            "positive_credit_eligible"
        ):
            # A lucky paired score from an intervention that never occurred is
            # analysis data, not a positive H1 learning signal.
            row["reward"] = min(float(row["reward"]), 0.0)

    rewards = [float(row["reward"]) for row in rows]
    credited_rewards = [
        float(row["reward"]) for row in rows if row["attribution_valid"]
    ]
    reward_range = (
        max(credited_rewards) - min(credited_rewards)
        if credited_rewards else 0.0
    )
    all_creditable = all(row["attribution_valid"] for row in rows)
    no_signal = all_creditable and reward_range < zero_range_eps
    if no_signal:
        for row in rows:
            row["advantage"] = 0.0
    else:
        base_advantages = _rloo(rewards)
        tau = max(reward_range / 4.0, 1e-6)
        max_weights = _max_weights(rewards, tau)
        for row, advantage, max_weight in zip(
            rows, base_advantages, max_weights
        ):
            row["advantage"] = (
                (1.0 - sharpen_alpha) * advantage
                + sharpen_alpha * max_weight
            )
    for row in rows:
        evidence = row.get("component_evidence") or {}
        inactive = (
            evidence.get("activation_required")
            and not evidence.get("positive_credit_eligible")
        )
        row["advantage_clamped_for_inactive_component"] = bool(inactive)
        if inactive:
            row["advantage"] = min(float(row["advantage"]), 0.0)

    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    credited = [row for row in rows if row["attribution_valid"]]
    causal_best = max(
        credited,
        key=lambda row: (
            float(row["reward"]),
            float(row["component_credit_delta"]),
            float(row["score"]),
        ),
    ) if credited else None
    promotion_rows = [row for row in credited if row["promotion_eligible"]]
    promotion_best = max(
        promotion_rows,
        key=lambda row: (
            float(row["component_credit_delta"]), float(row["score"])
        ),
    ) if promotion_rows else None
    absolute_best = max(
        credited, key=lambda row: float(row["score"])
    ) if credited else None
    program_samples = [
        (row["k"], sample)
        for row in credited
        for sample in row["pair_contract"]["samples"]
    ]
    program_best = max(
        program_samples, key=lambda item: float(item[1]["candidate_score"])
    ) if program_samples else None
    improved = promotion_best is not None
    return {
        "task_id": task_id,
        "base_score": base_score,
        "ceiling": ceiling,
        "reward_mean": mean,
        "reward_std": std,
        "rows": rows,
        "adv_mode": (
            "paired_same_seed:no_signal"
            if no_signal else f"paired_same_seed:rloo+max(a={sharpen_alpha})"
        ),
        "credit_basis": "candidate_minus_parent_same_decode_seed",
        "paired_repeats": expected_repeats,
        "min_promotion_repeats": min_promotion_repeats,
        "min_effect": float(min_effect),
        "best_k": (
            promotion_best["k"] if promotion_best else
            causal_best["k"] if causal_best else None
        ),
        "best_score": (
            promotion_best["score"] if promotion_best else
            causal_best["score"] if causal_best else None
        ),
        "best_control_score": (
            promotion_best["control_score"] if promotion_best else
            causal_best["control_score"] if causal_best else None
        ),
        "best_causal_delta": (
            promotion_best["causal_delta"] if promotion_best else
            causal_best["causal_delta"] if causal_best else None
        ),
        "best_component_credit_delta": (
            promotion_best["component_credit_delta"] if promotion_best else
            causal_best["component_credit_delta"] if causal_best else None
        ),
        "diagnostic_causal_best_k": (
            causal_best["k"] if causal_best else None
        ),
        "promotion_best_k": (
            promotion_best["k"] if promotion_best else None
        ),
        "promotion_best_score": (
            promotion_best["score"] if promotion_best else None
        ),
        "promotion_best_component_credit_delta": (
            promotion_best["component_credit_delta"]
            if promotion_best else None
        ),
        "absolute_best_k": absolute_best["k"] if absolute_best else None,
        "absolute_best_score": (
            absolute_best["score"] if absolute_best else None
        ),
        "program_best_k": program_best[0] if program_best else None,
        "program_best_score": (
            program_best[1]["candidate_score"] if program_best else None
        ),
        "program_improved": bool(
            program_best
            and float(program_best[1]["candidate_score"]) > base_score + EPS
        ),
        "improved": improved,
    }


def compute_task_group_v3(
    *, task_id: str, candidates: List[Dict[str, Any]], rollout_root: Path,
    base_score: float, ceiling: Optional[float] = None,
    sharpen_alpha: float = 0.3, zero_range_eps: float = 1e-3,
    hist_lambda: float = 0.3,
) -> Dict[str, Any]:
    """v2 + HISTORICAL-FRONTIER RESCUE (the '历史 trajectory reward 对比' idea).

    v2 zeros a group whenever its K valid rewards are tied within
    ``zero_range_eps`` (no within-group signal). But ``base_score`` IS the
    historical frontier (the ratchet best), so a tied batch that sits UNIFORMLY
    above/below the frontier still carries a real cross-round signal that v2
    throws away. Empirically (AC1) that discards genuine gains — e.g. a round
    where all 8 candidates beat base by 20%% of the gap was zeroed.

    Rescue: for an off-frontier tied group, give a uniform directional advantage
    ``hist_lambda * softcap(vm)`` where ``vm`` is the shared gap-normalized
    reward vs base (>0 batch beat the frontier -> reinforce this phi; <0 ->
    discourage). A truly-plateaued group (vm ~ 0, batch reproduced the frontier
    program) still gets 0 — no signal can be manufactured from zero variance;
    that regime needs exploration (higher K/temperature), not a baseline.

    Only the previously-zeroed rounds change; groups with within-group signal are
    byte-identical to v2. Still fully on-policy: the historical frontier only
    sets the baseline/magnitude; we never backprop through past trajectories.
    """
    g = compute_task_group_v2(
        task_id=task_id, candidates=candidates, rollout_root=rollout_root,
        base_score=base_score, ceiling=ceiling, sharpen_alpha=sharpen_alpha,
        zero_range_eps=zero_range_eps)
    if g["adv_mode"] != "no_signal":
        return g  # within-group signal present -> identical to v2
    valid_rewards = [r["reward"] for r in g["rows"] if r["valid"]]
    vm = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
    if abs(vm) < zero_range_eps:
        g["adv_mode"] = "no_signal(true-plateau)"        # genuinely nothing to learn
        return g
    nudge = hist_lambda * (3.0 * math.tanh(vm / 3.0))    # bounded, monotonic in vm
    for r in g["rows"]:
        r["advantage"] = nudge if r["valid"] else INVALID_REWARD
    g["adv_mode"] = f"hist_rescue(vm={vm:+.3f},lam={hist_lambda})"
    return g


def compute_task_group_grpo(
    *, task_id: str, candidates: List[Dict[str, Any]], rollout_root: Path,
    base_score: float, ceiling: Optional[float] = None,
    zero_range_eps: float = 1e-3, std_eps: float = 1e-6, **_ignored: Any,
) -> Dict[str, Any]:
    """STANDARD GRPO credit (ablation of the RLOO + max-sharpening default).

    Same v2 rewards (gap-normalized vs the incumbent, invalid = -1, repaired =
    -1), but the advantage is the plain group-normalized reward
        A_i = (r_i - mean(r)) / (std(r) + std_eps)
    over ALL slots of the proposal group (GRPO, Shao et al. 2024): no
    leave-one-out baseline, no max-sharpening term, no historical rescue.  A
    zero-variance group gets all-zero advantages (nothing to learn), exactly as
    a GRPO implementation would.  Selection/promotion is untouched: the ratchet
    and the beam read scores, never advantages.
    """
    g = compute_task_group_v2(
        task_id=task_id, candidates=candidates, rollout_root=rollout_root,
        base_score=base_score, ceiling=ceiling, sharpen_alpha=0.0,
        zero_range_eps=zero_range_eps)
    rewards = [float(r["reward"]) for r in g["rows"]]
    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / n) if n else 0.0
    if n < 2 or std < 1e-12:
        for r in g["rows"]:
            r["advantage"] = 0.0
        g["adv_mode"] = "grpo:no_signal(zero-variance)"
    else:
        for r, x in zip(g["rows"], rewards):
            r["advantage"] = (x - mean) / (std + std_eps)
        g["adv_mode"] = "grpo(mean-std)"
    g["reward_mean"], g["reward_std"] = mean, std
    return g


def _rms_normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    rms = math.sqrt(sum(v * v for v in values) / len(values))
    return [v / rms for v in values] if rms > EPS else list(values)


def compute_task_group_anchored(
    *, task_id: str, candidates: List[Dict[str, Any]], rollout_root: Path,
    base_score: float, ceiling: Optional[float] = None, **_ignored: Any,
) -> Dict[str, Any]:
    """Adaptive-ported anchored signed credit (reward.impl=anchored).

    Per candidate the raw advantage is a bounded blend
        0.70*tanh(delta/0.25) + 0.20*rank_term + 0.30*record_bonus
    where delta = gap-normalized gain vs base, rank_term is the within-group
    rank in [-1,1], and record_bonus fires only on a genuine new record. Signs
    are locked (valid positive delta >= +0.05, negative <= -0.05, invalid <=
    -1, behavior-equivalent = 0) so a high raw score that regressed against its
    matched parent can never read as a gain. Advantages are RMS-normalized
    across the group. Validity-gated best selection is inherited from the same
    load_rollout_score path the other impls use.
    """
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        k = cand["k"]
        valid = bool(cand.get("valid"))
        score = (load_rollout_score(rollout_root / task_id / f"cand{k:02d}", task_id)
                 if valid else None)
        rows.append({"k": k, "valid": valid, "score": score,
                     "spec_hash": cand.get("spec_hash", ""),
                     "changed_fields": cand.get("changed_fields", []),
                     "review_log": cand.get("review_log", [])})

    denom = (ceiling - base_score) if (ceiling is not None and ceiling > base_score + EPS) \
        else (abs(base_score) + EPS)
    valid_scores = [r["score"] for r in rows if r["valid"] and r["score"] is not None]
    ranked = sorted(valid_scores)
    raw: List[float] = []
    for r in rows:
        if not r["valid"] or r["score"] is None:
            raw.append(-1.0)
            r["reward"] = INVALID_REWARD
            continue
        delta = (r["score"] - base_score) / denom
        hard_zero = abs(delta) <= EPS
        anchor = math.tanh(delta / 0.25)
        rank = (ranked.index(r["score"]) / max(1, len(ranked) - 1)) * 2 - 1 if len(ranked) > 1 else 0.0
        is_record = r["score"] > base_score + EPS and r["score"] >= max(valid_scores)
        record_bonus = math.tanh(max(0.0, delta) / 0.05) if is_record else 0.0
        value = 0.70 * anchor + 0.20 * rank + 0.30 * record_bonus
        if hard_zero:
            value = 0.0
        elif delta > 0.0:
            value = max(value, 0.05)
        elif delta < 0.0:
            value = min(value, -0.05)
        raw.append(value)
        r["reward"] = delta
    for r, adv in zip(rows, _rms_normalize(raw)):
        r["advantage"] = adv

    valid = [r for r in rows if r["valid"] and r["score"] is not None]
    best = max(valid, key=lambda r: r["score"]) if valid else None
    return {"task_id": task_id, "base_score": base_score, "ceiling": ceiling,
            "reward_mean": sum(r["reward"] for r in rows) / len(rows),
            "reward_std": 0.0, "rows": rows, "adv_mode": "anchored",
            "best_k": best["k"] if best else None,
            "best_score": best["score"] if best else None,
            "improved": bool(best and best["score"] > base_score)}
