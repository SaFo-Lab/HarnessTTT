"""Proposer I/O — round-context builder + versioning (NOT the H1 harness).

The H1 harness itself (system prompt, tools, skill, sampling) is the versioned
declarative NexAU package at ``src/outer/harness/``. This
module is only the surrounding glue: it builds the deterministic per-condition
USER message, retains an offline feedback formatter, and hashes the package for
provenance. Feedback is never injected into proposer-policy GRPO context.
Renamed from ``h1.py`` (that name wrongly implied it *was* H1).

It builds the round-varying USER message for ONE task instance): the task's public spec, its seed program (excerpt),
the current best harness spec for this task, and its scores.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

H1_VERSION = "v1.1"
H1_PACKAGE = Path(__file__).resolve().parents[1] / "harness"

_SEED_PROGRAM_CAP = 5000  # chars of the seed program shown to the proposer

PROPOSAL_FLOW_INSTRUCTION = """Final proposal-flow instruction: follow this order: (1) analyze the current
best H2 harness and its existing behavior; (2) analyze the public task,
executor failure modes, and reward-hacking risks; (3) make an explicit plan for
how the current best harness should change; (4) assess prompts, tools, skills, middlewares, LLM configuration, and sampling
separately, deciding for each whether to keep, modify, remove, or add it—a
justified keep/no-change decision is valid; (5) decide whether any new
component is actually needed; (6) for each needed capability, first consult
the existing NexAU tools and middlewares and inspect relevant mounted schemas,
bindings, descriptions, and parameters, deciding whether to reuse them
unchanged or adapt their configuration or description; only when neither
works, design a task-specific tool that efficiently guides the executor toward
better solutions or prevents task-specific reward hacking without embedding a
solution; and (7) summarize all modifications, organize them into one coherent
harness, validate it, then call `submit_harness()`."""

USER_TEMPLATE = """# Task instance: {task_id}

## Public task description
{task_spec}

## Seed program (the executor edits the EVOLVE-BLOCK region){seed_note}
```python
{seed_program}
```

## Scores on this task (combined_score, higher is better; budget {max_evals} evals)
- seed program alone: {seed_score:.6g}
- current harness best: {base_score:.6g}{stuck_tag}

# Current H2 harness
A private filesystem copy of the current H2 is available through the harness
tools. Preload records explicitly mark `complete: true` files as inspected;
truncated/omitted records are not inspected. Atomic component upsert does not
require manually reading inherited files. Before any raw edit of an incomplete
inherited file, inspect that exact path.

# Task
Design an H2 tailored to THIS task by editing that filesystem. Diagnose why
the current H2 reaches only {base_score:.6g}. Before editing, analyze the task,
the incumbent's likely failure modes, and what intervention could make search
faster or more reliable. Compare behaviorally distinct alternatives, then choose
the smallest coherent portfolio yourself. Prompt, skill, budget-neutral
sampling, tool, middleware, and required plumbing are all eligible when their
surface is available and the causal diagnosis supports them. Multiple changes
are allowed only when each has distinct marginal value and the composition is
necessary; do not bundle unrelated edits.

The seed code above is context only: NEVER copy it wholesale into an H2 file or
create an EVOLVE-BLOCK. Use public task evidence only to design reusable
executor behavior that addresses the failure mode you diagnosed; do not embed
a solution answer. Two anti-anchoring rules: never
write absolute claims like "X is known to be optimal" into H2 files — they
bias every later round; and never re-implement items from the incumbent
prompt's own plan text as though they were new capabilities. The rollout runtime supplies the
actual program to the future executor. Prefer atomic upserts for generated
skills, tools, or middleware; the framework owns their files, mounts,
descriptors, frontmatter, wrappers, and canonical declarations. Use exposed
file tools for other supported surfaces. Validate after each selected change
and again after the complete portfolio. A valid final workspace is
deterministically auto-finalized if an explicit submit is missed. Raw
validation is observational: incomplete fallback edits are preserved and
reported."""


def finalize_proposer_user_message(message: str, *suffixes: str) -> str:
    """Place exactly one fixed proposal-flow instruction at the true end."""

    core = str(message).rstrip()
    if core.endswith(PROPOSAL_FLOW_INSTRUCTION):
        core = core[:-len(PROPOSAL_FLOW_INSTRUCTION)].rstrip()
    combined = core + "".join(str(suffix) for suffix in suffixes)
    return combined.rstrip() + "\n\n" + PROPOSAL_FLOW_INSTRUCTION


def build_user_message(*, task_id: str, task_spec: str, seed_program: str,
                       seed_score: float, base_score: float,
                       base_spec: Dict[str, Any], max_evals: int = 20) -> str:
    prog = seed_program.strip()
    note = ""
    if len(prog) > _SEED_PROGRAM_CAP:
        prog = prog[:_SEED_PROGRAM_CAP]
        note = f" (truncated to {_SEED_PROGRAM_CAP} chars)"
    stuck = "  <-- STUCK AT SEED: the harness makes no progress here" \
        if abs(base_score - seed_score) < 1e-9 else ""
    message = USER_TEMPLATE.format(
        task_id=task_id,
        task_spec=task_spec.strip() or "(no public description; rely on the seed program)",
        seed_note=note,
        seed_program=prog,
        seed_score=seed_score,
        base_score=base_score,
        stuck_tag=stuck,
        max_evals=max_evals,
    )
    return finalize_proposer_user_message(message)


def h1_hash() -> str:
    """Hash the whole H1 package (every file) for provenance."""
    h = hashlib.sha256()
    for f in sorted(H1_PACKAGE.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(H1_PACKAGE)).encode())
            h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()[:16]


def proposer_contract_hash() -> str:
    """Hash every source file that can accept/reject a v1.1 proposal.

    ``h1_hash`` historically covered only the declarative NexAU package. The
    slot portfolio and file-workspace gates live around that package, so v1.1
    records a second, complete contract hash for rollout traceability.
    """

    repo = Path(__file__).resolve().parents[3]
    files = [
        *[
            p for p in H1_PACKAGE.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
            and not p.name.endswith((".pyc", ".pyo"))
        ],
        Path(__file__),
        repo / "src/outer/proposing/component_portfolio.py",
        repo / "src/outer/proposing/component_scaffold.py",
        repo / "src/outer/proposing/propose.py",
        repo / "src/outer/workspace/propose_session.py",
        repo / "src/outer/workspace/h2_workspace.py",
        repo / "src/outer/genome/harness_spec.py",
        repo / "src/outer/compiling/materialize.py",
        repo / "src/outer/safety/static_gates.py",
        repo / "src/outer/safety/task_profiles.py",
        repo / "src/outer/safety/reviewer/reviewer.py",
        repo / "src/outer/safety/reviewer/selftest.py",
        repo / "src/outer/safety/reviewer/middleware_selftest.py",
        repo / "src/inner/runtime/harness_sdk.py",
        repo / "src/inner/runtime/session.py",
        repo / "src/inner/runtime/program_structure.py",
        repo / "src/inner/evaluation/candidate_geometry.py",
        repo / "src/inner/evaluation/_geometry_worker.py",
        repo / "src/inner/harness/middleware/generated_context.py",
        repo / "src/inner/runtime/token_ledger.py",
    ]
    h = hashlib.sha256()
    for path in sorted({p.resolve() for p in files if p.is_file()}):
        h.update(str(path.relative_to(repo)).encode())
        h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()[:16]


def source_revision(repo: Optional[Path] = None) -> Dict[str, Any]:
    """Return fail-closed Git provenance without refreshing the index.

    ``git status`` refreshes the worktree index and takes an ``index.lock``.
    On Lustre that refresh can exceed a short subprocess timeout, leaving both
    null provenance and a stale lock after an otherwise successful rollout.
    ``git diff HEAD`` verifies all tracked worktree bytes without updating the
    index, while ``git ls-files --others`` accounts for all untracked files.
    The whole worktree is sealed because proposal execution imports task,
    config, entrypoint, runtime, and evaluator code outside ``src/outer``.
    """

    repo = (
        Path(repo).resolve()
        if repo is not None else Path(__file__).resolve().parents[3]
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            capture_output=True, check=True, timeout=300,
        ).stdout.strip()
        expected = os.environ.get("SAH_EXPECTED_COMMIT", "").strip()
        if expected and commit != expected:
            raise RuntimeError(
                f"source HEAD {commit} != SAH_EXPECTED_COMMIT {expected}"
            )
        branch_result = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo, text=True, capture_output=True, check=False, timeout=300,
        )
        if branch_result.returncode not in (0, 1):
            raise RuntimeError(
                "git symbolic-ref failed: " + branch_result.stderr.strip()
            )
        tracked_result = subprocess.run(
            ["git", "diff", "--quiet", "--no-ext-diff", "HEAD", "--"],
            cwd=repo, text=True, capture_output=True, check=False, timeout=300,
        )
        if tracked_result.returncode not in (0, 1):
            raise RuntimeError(
                "git diff failed: " + tracked_result.stderr.strip()
            )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--"],
            cwd=repo, text=True, capture_output=True, check=True, timeout=300,
        ).stdout.strip()
        return {
            "commit": commit,
            "branch": branch_result.stdout.strip() or None,
            "proposer_sources_dirty": bool(
                tracked_result.returncode == 1 or untracked
            ),
            "tracked_check": "git_diff_head_no_index_refresh",
            "untracked_check": "git_ls_files_others",
            "checked_paths": ["."],
        }
    except Exception as exc:
        raise RuntimeError(
            "unable to capture trustworthy source revision: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def render_workspace(draft_dir):
    """Render bounded, integrity-labelled mutable H2 records into context.

    PRELOAD-001: the historical H1 loop spent most of its iterations on
    ``cat`` round-trips before the first edit.  Injecting every mutable file
    up front lets the proposer state its edits directly and keeps the whole
    trajectory within a small iteration budget (edits -> <=5 validation
    cycles -> submit).  Read-before-edit is satisfied by construction — the
    session marks these files pre-inspected.

    Returns ``(rendered_text, rendered_relative_paths)``.
    """
    from pathlib import Path

    draft_dir = Path(draft_dir)
    per_file_cap = 5000
    total_cap = 14000
    ordered = [draft_dir / "agent.yaml", draft_dir / "prompt.md"]
    for pattern in ("tools/*.tool.yaml", "custom_tools/*.py",
                    "skills/*/SKILL.md", "middlewares/*.py",
                    "middlewares/*.middleware.yaml"):
        ordered.extend(sorted(draft_dir.glob(pattern)))
    parts = [
        "\n\n## CURRENT H2 WORKSPACE (bounded integrity records)\n"
        "Each JSON record says whether its content is complete. Only "
        "complete=true counts as inspected; read an incomplete record before "
        "raw editing. Atomic upsert does not require manual reinspection.\n"
    ]
    used = 0
    rendered = []
    for path in ordered:
        if not path.is_file():
            continue
        relative = path.relative_to(draft_dir).as_posix()
        text = path.read_text(errors="replace")
        complete = len(text) <= per_file_cap
        record = {
            "path": relative,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
            "complete": complete,
            "content": text if complete else text[:per_file_cap],
            **({"reason": "per-file cap; cat before raw edit"}
               if not complete else {}),
        }
        block = "\n```json\n" + json.dumps(
            record, ensure_ascii=False
        ) + "\n```\n"
        if used + len(block) > total_cap:
            record["complete"] = False
            record["content"] = None
            record["reason"] = "total preload cap; cat before raw edit"
            block = "\n```json\n" + json.dumps(
                record, ensure_ascii=False
            ) + "\n```\n"
        else:
            used += len(block)
        parts.append(block)
        if record["complete"]:
            rendered.append(relative)
    return "".join(parts), rendered


def render_feedback(fb: dict) -> str:
    """Render previous-visit telemetry for offline reports, never H1 input."""
    lines = [
        "\n\n## Telemetry from the previous visit (round %s)" % fb.get("round"),
    ]
    if fb.get("analyst_note"):
        lines.append("ANALYST NOTE: " + fb["analyst_note"])
    best = fb.get("best_score") or fb.get("base_score", 0.0)
    accepted = fb.get("accepted_improvement")
    accepted_score = fb.get("outgoing_base_score", fb.get("base_score", 0.0))
    if accepted is False and best > fb.get("base_score", 0.0):
        score_line = (
            "The raw candidate maximum was %.6g, but it was not accepted "
            "(%s); the incumbent remains %.6g."
            % (best, fb.get("program_ratchet_reason", "attribution failed"), accepted_score)
        )
    else:
        score_line = (
            "The starting score was %.6g; the best of 8 candidate harnesses reached %.6g."
            % (fb.get("base_score", 0.0), best)
        )
    lines += [
        score_line,
        "%s of the candidates made NO progress past the starting program." % fb.get("n_stuck_at_base", "?"),
        "Per-candidate causal/behavioral outcomes:",
    ]
    for c in fb.get("candidates", []):
        if c.get("invalid"):
            lines.append("  k%s: INVALID H2 (never rolled out); taxonomy=invalid_component" % c["k"])
            continue
        if c.get("tools"):
            lines.append("  k%s: tools=%s" % (c["k"], ", ".join(c["tools"])))
        taxonomy = c.get("failure_taxonomy") or "unclassified"
        progression = json.dumps(
            c.get("score_progression") or [], separators=(",", ":")
        )
        reasons = json.dumps(
            c.get("component_reject_reasons") or {}, separators=(",", ":")
        )
        lines.append("  k%s: candidate=%s control=%s raw_delta=%s "
                     "component_credit_delta=%s activation=%s consistent=%s "
                     "evals=%s "
                     "probes=%s llm=%s tokens=%s/%s sandbox_s=%s stop=%s "
                     "changed=%s component_calls=%s usable=%s semantic_ok=%s "
                     "semantic_rejections=%s reject_reasons=%s "
                     "middleware_fires=%s "
                     "geometry_calls=%s valid_edit_rate=%s new_best_edits=%s "
                     "families=%s progression=%s structure_gate=%s/%s "
                     "taxonomy=%s%s" % (
            c["k"],
            ("%.6g" % c["score"]) if c.get("score") is not None else "?",
            ("%.6g" % c["control_score"])
            if c.get("control_score") is not None else "?",
            ("%+.6g" % c["causal_delta"])
            if c.get("causal_delta") is not None else "?",
            ("%+.6g" % c["component_credit_delta"])
            if c.get("component_credit_delta") is not None else "?",
            c.get("component_activation_status"),
            c.get("component_activation_consistent"),
            c.get("evals"), c.get("probes"), c.get("llm_calls"),
            c.get("input_tokens"), c.get("output_tokens"),
            c.get("sandbox_seconds"), c.get("stop"),
            ",".join(c.get("changed", [])), c.get("component_invocations"),
            c.get("component_usable_outputs"),
            c.get("component_semantic_successes"),
            c.get("component_semantic_rejections"), reasons,
            c.get("middleware_fires"), c.get("candidate_geometry_calls"),
            c.get("valid_edit_rate"), c.get("new_best_edits"),
            c.get("distinct_program_families"), progression,
            c.get("structure_gate_satisfied"),
            c.get("structure_postcondition_failed"), taxonomy,
            (" err=" + c["err"]) if c.get("err") else ""))
    lines.append(
        "Use component_credit_delta and a semantic activation witness—not raw "
        "score luck or transport invocation—to select a failure mode. A tool "
        "needs a top-level ok:true result in every paired repeat; a middleware "
        "must fire in every paired repeat. "
        "Do not resubmit a delivered-but-inactive, causally neutral, "
        "duplicate-family, or invalid component.")
    return "\n".join(lines)


def render_prior_actions(actions: list) -> str:
    """Sequential-sampling context: the VALID actions already proposed for THIS
    task in THIS batch, so this sample proposes something genuinely different
    (Adaptive within-batch diversity). Only the changed axes + a compact spec
    outline are shown — no scores, no rollout outcomes (those aren't available
    yet mid-batch, and withholding them keeps the channel leak-free)."""
    if not actions:
        return ""
    lines = ["\n\n## Already proposed this batch (do NOT paraphrase these)"]
    for a in actions:
        fields = ", ".join(a.get("changed_fields") or []) or "(no changes)"
        spec = a.get("spec") or {}
        tools = [t.get("name") for t in (spec.get("new_tools") or []) if isinstance(t, dict)]
        skills = [s.get("name") for s in (spec.get("new_skills") or []) if isinstance(s, dict)]
        extra = ""
        if tools:
            extra += " new_tools=[%s]" % ", ".join(filter(None, tools))
        if skills:
            extra += " new_skills=[%s]" % ", ".join(filter(None, skills))
        lines.append("  - k%s changed: %s%s" % (a.get("k"), fields, extra))
    lines.append("Propose a design that explores a DIFFERENT axis or strategy "
                 "than every entry above.")
    return "\n".join(lines)
