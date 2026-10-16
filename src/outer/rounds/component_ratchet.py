"""COMPRATCHET-001: selection at the COMPONENT level, with evidence.

The program ratchet promotes whole harnesses, so a component with a positive
ISOLATED causal effect dies whenever its candidate loses the round — measured
across campaigns: the one skill with Δ+0.0439 never reached the next round's
base.  Evolution needs the missing operator: components are admitted into the
next round's base on their own merit, independently of who won.

Admission contract (every clause checkable from artifacts):
  * exactly one heritable unit was changed (one component XOR one parameter),
    so joint effects are never split into individually credited grafts;
  * the EXECUTOR actually used the component — a tool that returned ``ok:true``, a skill
    loaded or injected before it wrote code, a middleware that fired.  Being
    mounted is not participation: generated tools were mounted 99 times and
    called 18 across 21 campaigns;
  * the pair passed attribution (``attribution_valid``);
  * ``causal_delta >= CURVE_COMPONENT_RATCHET_MIN_DELTA`` (default 1e-4);
  * a component's only prompt difference is its exact canonical declaration;
  * capacity: the genome caps per type bound the heritable library
    (tools 3 / skills 2 / middlewares 2); incumbent-carried components are
    never evicted — overflow admissions are recorded as skipped;
  * the grafted base keeps every naming invariant (declaration lines are
    appended; the prompt cap is respected or the admission is skipped).

Every admission/skip is appended to ``rounds/component_ledger.jsonl`` with
the full evidence chain (round, slot, candidate dir, delta, control/candidate
scores, authored-by), and ``next_bases.json`` records what was grafted, from
where, and why — so the evolutionary history is reconstructible after the
fact.  ``CURVE_COMPONENT_RATCHET=0`` disables the operator.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from outer.genome import harness_spec as hs
from outer.compiling.materialize import materialize

_KIND_FIELDS = {
    "new_tools": "tool",
    "new_skills": "skill",
    "new_middlewares": "middleware",
}
# The fifth dimension: an operating-point move (sampling / agent budget /
# built-in middleware policy) measured in isolation is exactly as heritable
# as a component — it is a scalar field, so admission overwrites it on the
# next base and the highest-delta proposal wins a contested field.
_PARAM_GROUPS = ("sampling", "agent", "middleware")
_TYPE_CAPS = {"new_tools": 3, "new_skills": 2, "new_middlewares": 2}
_PROMPT_CAP = 10000
_CORE_TOOLS = {"edit_solution", "evaluate_solution", "finish",
               "probe_solution", "read_solution"}
_CORE_MIDDLEWARES = {"budget_reminder", "long_tool_output",
                     "round_and_token_reminder", "stall_restart"}


def _used_by_executor(round_dir: Path, tid: str, k: int) -> set:
    """Generated components the executor actually engaged during the rollout.

    A tool counts only when it returned a top-level ``ok:true`` result, a skill
    when it was loaded or injected before the executor wrote code, and a
    middleware when it fired. Transport invocation and mounting are
    not participation: across 21 campaigns generated tools were mounted 99
    times and called 18, so "the harness carried it" says nothing about
    whether it did any work.  Inheritance is for components the executor used.
    """
    used: set = set()
    base = round_dir / "rollouts" / tid / f"cand{k:02d}"
    if not base.is_dir():
        return used
    for result in base.rglob("results/*.json"):
        try:
            payload = json.loads(result.read_text())
        except Exception:
            continue
        for item in (payload if isinstance(payload, list) else [payload]):
            if not isinstance(item, dict):
                continue
            for name, audit in (item.get("tool_audit") or {}).items():
                if name in _CORE_TOOLS or not isinstance(audit, dict):
                    continue
                if int(audit.get("semantic_successful_invocations") or 0) > 0:
                    used.add(str(name))
            for name, audit in (item.get("skill_audit") or {}).items():
                if not isinstance(audit, dict):
                    continue
                if any(int(audit.get(field) or 0) > 0 for field in
                       ("loads", "loads_before_first_edit",
                        "runtime_injections")):
                    used.add(str(name))
            for name, audit in (item.get("middleware_audit") or {}).items():
                if name in _CORE_MIDDLEWARES or not isinstance(audit, dict):
                    continue
                if int(audit.get("fires") or 0) > 0:
                    used.add(str(name))
    return used


def _component_entries(spec: Dict[str, Any], names: set) -> List[Dict[str, Any]]:
    out = []
    for field in _KIND_FIELDS:
        for entry in (spec.get(field) or []):
            if isinstance(entry, dict) and str(entry.get("name")) in names:
                out.append({"field": field, "entry": entry})
    return out


def apply_component_ratchet(
    *,
    round_dir: Path,
    tid: str,
    round_index: int,
    candidates: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    improved: bool,
    best_k: Optional[int],
    next_base_entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Graft admissible components into the next base; mutate the entry.

    Returns the ratchet record (also stored on the entry) or None when the
    operator is disabled or nothing was admissible.
    """
    if os.environ.get("CURVE_COMPONENT_RATCHET", "1") != "1":
        return None
    min_delta = float(
        os.environ.get("CURVE_COMPONENT_RATCHET_MIN_DELTA", "1e-4") or 1e-4)

    row_by_k = {int(r["k"]): r for r in rows}
    admissible: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for cand in candidates:
        k = int(cand.get("k", -1))
        if not cand.get("valid"):
            continue
        comp_names = {
            str(f).split(".", 1)[1] for f in (cand.get("changed_fields") or [])
            if str(f).split(".", 1)[0] in _KIND_FIELDS
        }
        param_fields = [
            str(f) for f in (cand.get("changed_fields") or [])
            if str(f).split(".", 1)[0] in _PARAM_GROUPS and "." in str(f)
        ]
        if not comp_names and not param_fields:
            continue
        proposed_comp_names = set(comp_names)
        row = row_by_k.get(k) or {}
        delta = row.get("causal_delta")
        evidence = {
            "round": round_index, "k": k,
            "candidate": str(round_dir / "tasks" / tid / f"cand{k:02d}"),
            "causal_delta": delta,
            "score": row.get("score"),
            "control_score": row.get("control_score"),
            "authored_by": ("repair" if cand.get("stop_reason") ==
                            "repaired_submission" else "proposer"),
        }
        if improved and best_k is not None and k == int(best_k):
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "already_in_promoted"})
            continue
        heritable_units = sorted(proposed_comp_names) + param_fields
        if len(heritable_units) != 1:
            skipped.append({
                **evidence,
                "names": heritable_units,
                "reason": "non_atomic_heritable_bundle_requires_exactly_one",
            })
            continue
        # USAGE-001: a component is heritable when the EXECUTOR used it and
        # the pair came out ahead — not because the slot it came from carried
        # a label.  Admission used to require `component_only`, a label
        # decided before H1 ran; slots are gone, and a label was never
        # evidence anyway.  Participation is read from the rollout audits.
        used = _used_by_executor(round_dir, tid, k)
        unused = sorted(comp_names - used)
        if unused:
            skipped.append({**evidence,
                            "names": unused,
                            "reason": "not_used_by_executor:" + ",".join(unused)})
        comp_names &= used
        if not comp_names and not param_fields:
            continue
        evidence["used_by_executor"] = sorted(comp_names)
        # Change-set edits that are neither the sole heritable unit nor its
        # canonical prompt declaration make the paired delta uninterpretable.
        confounders = sorted(
            str(f) for f in (cand.get("changed_fields") or [])
            if str(f) != "system_prompt"
            and str(f).split(".", 1)[0] not in _KIND_FIELDS
            and str(f).split(".", 1)[0] not in _PARAM_GROUPS
        )
        if confounders:
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "delta_confounded_by:" + ",".join(confounders)})
            continue
        if not row.get("attribution_valid"):
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "attribution_invalid"})
            continue
        component_promotion = row.get("promotion")
        required_promotion_fields = {
            "blockers", "observed_repeats", "positive_sign_consistent",
            "minimum_effect", "minimum_repeats",
        }
        if not isinstance(component_promotion, dict) or not (
            required_promotion_fields <= set(component_promotion)
        ):
            skipped.append({
                **evidence,
                "names": heritable_units,
                "reason": "promotion_evidence_missing",
            })
            continue
        try:
            observed_repeats = int(component_promotion["observed_repeats"])
            minimum_repeats = int(component_promotion["minimum_repeats"])
            promotion_min_effect = float(
                component_promotion["minimum_effect"]
            )
        except (TypeError, ValueError):
            skipped.append({
                **evidence,
                "names": heritable_units,
                "reason": "promotion_evidence_invalid",
            })
            continue
        component_blockers = [
            str(reason) for reason in component_promotion.get("blockers", [])
            if str(reason) != "candidate_not_above_historical_base"
        ]
        if observed_repeats < minimum_repeats:
            component_blockers.append(
                f"paired_repeats_below_{minimum_repeats}"
            )
        if component_promotion.get("positive_sign_consistent") is not True:
            component_blockers.append("paired_delta_sign_inconsistent")
        if delta is None or float(delta) <= promotion_min_effect:
            component_blockers.append(
                f"mean_delta_not_above_{promotion_min_effect:g}"
            )
        component_blockers = list(dict.fromkeys(component_blockers))
        if component_blockers:
            skipped.append({
                **evidence,
                "names": sorted(proposed_comp_names) + param_fields,
                "reason": "inheritance_promotion_blocked:"
                + ",".join(component_blockers),
            })
            continue
        if delta is None or float(delta) < min_delta:
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": f"delta_below_{min_delta}"})
            continue
        cand_dir = Path(evidence["candidate"])
        try:
            cand_spec = hs.read_base_spec(cand_dir)
        except Exception as exc:
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": f"unreadable_candidate: {exc}"})
            continue
        parent_dir = Path(
            cand.get("parent_package") or next_base_entry["package"]
        )
        try:
            parent_spec = hs.read_base_spec(parent_dir)
        except Exception as exc:
            skipped.append({
                **evidence,
                "names": heritable_units,
                "reason": f"unreadable_parent: {exc}",
            })
            continue
        candidate_prompt = str(cand_spec.get("system_prompt") or "")
        parent_prompt = str(parent_spec.get("system_prompt") or "")
        evidence["prompt_also_changed"] = (
            hs.normalized_prompt(candidate_prompt)
            != hs.normalized_prompt(parent_prompt)
        )
        if comp_names:
            component_name = next(iter(comp_names))
            declaration = hs.isolated_generated_component_declaration(
                candidate_prompt=candidate_prompt,
                parent_prompt=parent_prompt,
                name=component_name,
            )
            if declaration is None:
                skipped.append({
                    **evidence,
                    "names": [component_name],
                    "reason": "prompt_not_isolated_component_declaration",
                })
                continue
            component_entries = _component_entries(cand_spec, comp_names)
            if len(component_entries) != 1:
                skipped.append({
                    **evidence,
                    "names": [component_name],
                    "reason": "component_entry_missing_or_ambiguous",
                })
                continue
            item = component_entries[0]
            admissible.append({
                **evidence,
                "field": item["field"],
                "name": str(item["entry"].get("name")),
                "entry": item["entry"],
                "prompt_declaration": declaration,
            })
        elif hs.normalized_prompt(candidate_prompt) != hs.normalized_prompt(
            parent_prompt
        ):
            skipped.append({
                **evidence,
                "names": param_fields,
                "reason": "delta_confounded_by:system_prompt",
            })
            continue
        for field in param_fields:
            group, key = field.split(".", 1)
            value = (cand_spec.get(group) or {}).get(key)
            if value is None:
                continue
            admissible.append({**evidence, "field": group, "name": key,
                               "param_value": value})

    record: Dict[str, Any] = {"admitted": [], "skipped": skipped}
    if admissible:
        admissible.sort(key=lambda a: -float(a["causal_delta"]))
        base_pkg = Path(next_base_entry["package"])
        base_spec = hs.read_base_spec(base_pkg)
        merged = json.loads(json.dumps(base_spec))  # deep copy, JSON-safe
        prompt = str(merged.get("system_prompt") or "")
        existing = {
            field: {str(e.get("name")) for e in (merged.get(field) or [])
                    if isinstance(e, dict)}
            for field in _KIND_FIELDS
        }
        admitted_params: set = set()
        for adm in admissible:
            field, name = adm["field"], adm["name"]
            public = {k: v for k, v in adm.items() if k != "entry"}
            if field in _PARAM_GROUPS:
                current = (merged.get(field) or {}).get(name)
                if (field, name) in admitted_params:
                    # admissible is sorted by DESCENDING delta, so the first
                    # writer is the strongest proposal; without this guard a
                    # weaker later one overwrote it (last-write-wins) while
                    # both were recorded as admitted.
                    skipped.append({**public,
                                    "reason": "param_contested_lower_delta"})
                    continue
                if current == adm["param_value"]:
                    skipped.append({**public, "reason": "param_already_set"})
                    continue
                admitted_params.add((field, name))
                merged.setdefault(field, {})
                merged[field][name] = adm["param_value"]
                public["previous_value"] = current
                record["admitted"].append(public)
                continue
            if name in existing[field]:
                skipped.append({**public, "reason": "name_already_in_base"})
                continue
            if len(existing[field]) >= _TYPE_CAPS[field]:
                skipped.append({**public, "reason": "capacity_full"})
                continue
            declaration = str(adm.get("prompt_declaration") or "").strip()
            if not declaration:
                skipped.append({**public, "reason": "prompt_declaration_missing"})
                continue
            if hs.generated_component_prompt_sections(prompt, name):
                skipped.append({**public, "reason": "prompt_declaration_collision"})
                continue
            if len(prompt) + len(declaration) + 2 > _PROMPT_CAP - 200:
                skipped.append({**public, "reason": "prompt_cap"})
                continue
            merged.setdefault(field, [])
            merged[field] = list(merged[field]) + [adm["entry"]]
            prompt = prompt.rstrip("\n") + "\n\n" + declaration + "\n"
            existing[field].add(name)
            record["admitted"].append(public)

        if record["admitted"]:
            merged["system_prompt"] = prompt
            prompt_issues = hs.component_prompt_issues(merged)
            if prompt_issues:
                for item in record["admitted"]:
                    skipped.append({
                        **item,
                        "reason": "grafted_prompt_validation_failed",
                        "validation_issues": prompt_issues,
                    })
                record["admitted"] = []
            else:
                graft_dir = round_dir / "grafted_bases" / tid
                materialize(
                    merged, graft_dir, meta={"effective": merged},
                    validate_prompt=True,
                )
                record["previous_package"] = str(base_pkg)
                record["grafted_package"] = str(graft_dir)
                record["spec_hash"] = hs.spec_hash(merged)
                next_base_entry["package"] = str(graft_dir)
                next_base_entry["score_basis"] = "pre_graft"
                next_base_entry["component_ratchet"] = {
                    "admitted": record["admitted"],
                    "skipped_count": len(skipped),
                    "previous_package": str(base_pkg),
                }
                beam = next_base_entry.get("beam")
                if beam:
                    beam[0] = dict(beam[0])
                    beam[0]["package"] = str(graft_dir)
                    beam[0]["from"] = (
                        beam[0].get("from", "incumbent") + "+graft"
                    )

    if record["admitted"] or skipped:
        ledger = round_dir.parent / "component_ledger.jsonl"
        with ledger.open("a") as fh:
            for kind, items in (("admitted", record["admitted"]),
                                ("skipped", skipped)):
                for item in items:
                    fh.write(json.dumps({
                        "event": kind, "task_id": tid,
                        "round": round_index, **item,
                    }) + "\n")
    return record if (record["admitted"] or skipped) else None
