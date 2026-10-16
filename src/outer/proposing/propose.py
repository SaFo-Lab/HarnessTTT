"""Run H1 to edit private candidate H2 packages (instance-wise).

Each candidate = one full H1 agent run (inspect -> edit -> validate -> submit)
against the M_phi endpoint, conditioned on ONE task instance. All K runs in an
exchangeable task/parent/policy group share the exact prompt, H2 workspace,
sampling configuration, and budget. Every sample gets the same safe
prompt/skill/sampling/tool/middleware editing surface and may compose a small
evidence-backed portfolio. Diversity within a group comes only from independent
H1 sampling seeds.
outer_round threads run_once across all (task, k) jobs and serving replicas;
the propose-session contextvar is thread-local, so sessions never collide.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import yaml

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from outer.genome import harness_spec as hs
from outer.proposing.proposer_io import (
    H1_PACKAGE,  # noqa: F401 (used by run_once)
    finalize_proposer_user_message,
    render_workspace,
)
from outer.workspace.propose_session import ProposeSession, propose_scope
from outer.compiling.materialize import materialize
from outer.proposing.component_scaffold import placeholder_bodies
from inner.runtime.program_structure import (
    STRUCTURAL_CANONICAL_SCHEMA,
    normalized_ast_digest,
    normalized_ast_payload,
)


COMPONENT_FINGERPRINT_SCHEMA = "sah-component-semantic-fingerprint/2.0"
COMPONENT_BEHAVIOR_SCHEMA = "sah-component-behavior-fingerprint/1.0"


def _portfolio_tool_names(required_kind: str) -> tuple[str, ...]:
    """Complete safe H1 editing surface shared by every proposer sample.

    ``required_kind`` remains in artifact metadata for backward compatibility,
    but must never restrict model choice. Raw editors make prompt, skill,
    sampling, registration, and multi-file changes reachable; atomic component
    transactions remain preferred for generated skills, tools, and middleware.
    """

    _ = required_kind
    return (
        "harness_shell", "write_harness_file", "edit_harness_file",
        "delete_harness_file", "upsert_skill", "upsert_tool", "upsert_middleware",
        "repair_component", "validate_harness", "submit_harness",
    )


@dataclass
class CandidateRecord:
    k: int
    valid: bool
    errors: List[str] = field(default_factory=list)
    raw_submission: str = ""
    spec: Optional[Dict[str, Any]] = None        # validated partial spec
    effective: Optional[Dict[str, Any]] = None   # spec folded over base
    review_log: list = field(default_factory=list)  # validate_harness component audit
    changed_fields: List[str] = field(default_factory=list)
    spec_hash: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    stop_reason: str = ""
    auto_finalized: bool = False
    schema_guard: Dict[str, Any] = field(default_factory=dict)  # TOOLSCHEMA-001
    # REPAIR-001: frozen-M0 takeover of a failed slot. ``trajectory`` always
    # stays the PRIMARY (phi) trajectory — phi's training credit binds to its
    # own actions; the repair trajectory is retained separately.
    repair: Dict[str, Any] = field(default_factory=dict)
    repair_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # EXPLORE-001 (v2.5)
    behavior_policy: str = "trained_phi"
    # BEAM-001: which beam member this slot descended from
    parent_package: Optional[str] = None
    parent_score: Optional[float] = None
    parent_spec_hash: str = ""
    parent_compute_contract: Dict[str, Any] = field(default_factory=dict)
    # PROPOSALARTIFACT-001: preserve the exact model input and final private
    # filesystem even for invalid/unsubmitted candidates.  Without these, a
    # trajectory can be inspected only as text and its attempted harness
    # cannot be replayed or reviewed file-by-file.
    exact_user_message: str = ""
    repair_user_message: str = ""
    primary_file_events: List[Dict[str, str]] = field(default_factory=list)
    repair_file_events: List[Dict[str, str]] = field(default_factory=list)
    workspace_archive: Optional[str] = None
    primary_workspace_archive: Optional[str] = None
    repair_workspace_archive: Optional[str] = None
    primary_component_transactions: List[Dict[str, Any]] = field(
        default_factory=list
    )
    repair_component_transactions: List[Dict[str, Any]] = field(
        default_factory=list
    )
    artifact_errors: List[str] = field(default_factory=list)
    validations: int = 0
    failed_validations: int = 0
    # Exact resolved primary H1 sampling/iteration contract. Sampling seeds are
    # intentionally excluded: they create exchangeable draws within a group.
    h1_config: Dict[str, Any] = field(default_factory=dict)
    component_transactions: List[Dict[str, Any]] = field(default_factory=list)
    cross_group_duplicates: List[Dict[str, Any]] = field(default_factory=list)
    component_semantic_fingerprint: str = ""
    component_semantic_fingerprint_record: Dict[str, Any] = field(
        default_factory=dict
    )
    within_group_semantic_duplicates: List[Dict[str, Any]] = field(
        default_factory=list
    )
    component_behavior_fingerprint: str = ""
    component_behavior_fingerprint_record: Dict[str, Any] = field(
        default_factory=dict
    )
    within_group_behavior_duplicates: List[Dict[str, Any]] = field(
        default_factory=list
    )
    validation_profile: Optional[Dict[str, Any]] = None
    h1_runtime_events: List[Dict[str, Any]] = field(default_factory=list)
    infrastructure_failure: bool = False
    training_block_reason: Optional[str] = None
    proposal_group_id: str = ""
    proposal_condition: Dict[str, Any] = field(default_factory=dict)
    training_eligible: bool = False


def _semantic_schema(value: Any) -> Any:
    """Strip prose-only JSON-Schema fields while preserving call semantics."""

    if isinstance(value, dict):
        return {
            str(key): _semantic_schema(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in {"description", "title", "$comment", "examples", "example"}
        }
    if isinstance(value, list):
        return [_semantic_schema(child) for child in value]
    return value


def component_semantic_fingerprint_record(
    effective: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a replayable, runtime-independent executable fingerprint record.

    This is deliberately narrower than semantic equivalence.  It catches the
    measured false-novelty class (renames, literal tweaks, pass/self-assign,
    neutral algebra/dead branches, and descriptor prose) without declaring
    different algorithms equivalent. The canonical AST payload is retained so
    an artifact can be independently recomputed without the original Python
    minor version.
    """

    if not isinstance(effective, dict):
        return {}
    rows: List[Dict[str, Any]] = []
    for tool in effective.get("new_tools") or []:
        if not isinstance(tool, dict) or not tool.get("implementation_py"):
            continue
        implementation = str(tool["implementation_py"])
        ast_payload = normalized_ast_payload(implementation)
        if ast_payload is None:
            continue
        rows.append({
            "kind": "tool",
            "ast_digest": normalized_ast_digest(implementation),
            "ast_payload": json.loads(ast_payload),
            "input_schema": _semantic_schema(tool.get("input_schema") or {}),
        })
    for middleware in effective.get("new_middlewares") or []:
        if not isinstance(middleware, dict) \
                or not middleware.get("implementation_py"):
            continue
        implementation = str(middleware["implementation_py"])
        ast_payload = normalized_ast_payload(implementation)
        if ast_payload is None:
            continue
        rows.append({
            "kind": "middleware",
            "hook": str(middleware.get("hook") or "before_model"),
            "ast_digest": normalized_ast_digest(implementation),
            "ast_payload": json.loads(ast_payload),
        })
    if not rows:
        return {}
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    payload = {
        "schema": COMPONENT_FINGERPRINT_SCHEMA,
        "ast_schema": STRUCTURAL_CANONICAL_SCHEMA,
        "components": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {
        "schema": COMPONENT_FINGERPRINT_SCHEMA,
        "fingerprint": "h2sem_" + digest[:24],
        "canonical_payload_sha256": digest,
        "canonical_payload": payload,
        # Diagnostic only: excluded from the canonical payload and fingerprint.
        "producer_runtime": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
        },
    }


def component_semantic_fingerprint(effective: Optional[Dict[str, Any]]) -> str:
    """Compatibility accessor for the replayable fingerprint string."""

    return component_semantic_fingerprint_record(effective).get(
        "fingerprint", ""
    )


def component_behavior_fingerprint_record(
    effective: Optional[Dict[str, Any]], base_spec: Dict[str, Any],
    validation_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fingerprint changed middleware by trusted bounded truth-table effects."""

    if not isinstance(effective, dict):
        return {}
    from outer.safety.reviewer.middleware_selftest import (
        middleware_behavior_fingerprint_record,
    )
    rows: List[Dict[str, Any]] = []
    for middleware in hs.changed_generated_components(
        effective, base_spec, "new_middlewares"
    ):
        if not isinstance(middleware, dict):
            continue
        record = middleware_behavior_fingerprint_record(
            str(middleware.get("implementation_py") or ""),
            hook=str(middleware.get("hook") or "before_model"),
            description=str(middleware.get("description") or ""),
            validation_profile=validation_profile,
        )
        payload = record.get("canonical_payload")
        if not isinstance(payload, dict):
            continue
        # Component names and descriptor prose do not create behavioral
        # novelty; the replayable case/effect matrix is authoritative.
        rows.append({
            "kind": "middleware",
            "behavior": payload,
        })
    if not rows:
        return {}
    rows.sort(key=lambda row: json.dumps(
        row, sort_keys=True, separators=(",", ":")
    ))
    payload = {
        "schema": COMPONENT_BEHAVIOR_SCHEMA,
        "components": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {
        "schema": COMPONENT_BEHAVIOR_SCHEMA,
        "fingerprint": "h2beh_" + digest[:24],
        "canonical_payload_sha256": digest,
        "canonical_payload": payload,
    }


def _dump_history(agent) -> List[Dict[str, Any]]:
    out = []
    try:
        for msg in agent.history:
            try:
                out.append(msg.model_dump())
            except Exception:
                out.append({"role": str(getattr(msg, "role", "?")),
                            "content": str(getattr(msg, "content", ""))[:4000]})
    except Exception:
        pass
    return out


def _declarative_h1_surface(
    *, allowed_tool_names: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Return the exact declarative H1 surface that creates runtime objects.

    ``AgentConfig.from_yaml`` materializes middleware rows into instances whose
    default string includes a process-local address. Runtime introspection is
    also unsafe for grouping: behavior-changing constructor values such as
    LongToolOutputMiddleware's ``bypass_tool_names`` live in private fields.
    The YAML rows are the canonical constructor contract, so retain them
    directly and bind every callable tool schema by content hash.
    """

    agent_path = H1_PACKAGE / "agent.yaml"
    raw = agent_path.read_bytes()
    document = yaml.safe_load(raw) or {}
    if not isinstance(document, dict):
        raise RuntimeError("H1 agent.yaml must contain a mapping")
    tools = document.get("tools") or []
    if not isinstance(tools, list):
        raise RuntimeError("H1 agent.yaml tools must contain a list")
    if allowed_tool_names is not None:
        tools = [
            row for row in tools
            if isinstance(row, dict)
            and str(row.get("name") or "") in allowed_tool_names
        ]
    schema_hashes: Dict[str, str] = {}
    for row in tools:
        if not isinstance(row, dict):
            raise RuntimeError("H1 agent.yaml tool rows must be mappings")
        name = str(row.get("name") or "")
        relative = str(row.get("yaml_path") or "")
        path = (H1_PACKAGE / relative).resolve()
        try:
            path.relative_to(H1_PACKAGE.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"H1 tool schema escapes the package: {relative!r}"
            ) from exc
        if not name or not path.is_file():
            raise RuntimeError(
                f"H1 tool declaration is incomplete: name={name!r}, "
                f"yaml_path={relative!r}"
            )
        schema_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "agent_yaml_sha256": hashlib.sha256(raw).hexdigest(),
        "tool_surface": (
            "portfolio_full" if allowed_tool_names is not None else "full"
        ),
        "system_prompt_path": document.get("system_prompt"),
        "tool_call_mode": document.get("tool_call_mode"),
        "sandbox_config": document.get("sandbox_config"),
        "tools": tools,
        "tool_schema_sha256": schema_hashes,
        "stop_tools": document.get("stop_tools") or [],
        "skills": document.get("skills") or [],
        "middlewares": document.get("middlewares") or [],
        "tracers": document.get("tracers") or [],
    }


def _configure_transport(config, *, timeout: float, max_retries: int) -> None:
    """Apply the same request-recovery contract used by H2 executors.

    ``LLMConfig.max_retries`` belongs to the provider SDK and retries only the
    failed HTTP request.  ``AgentConfig.retry_attempts`` is a separate outer
    layer that can replay an entire agent iteration.  Keep the latter at one
    total attempt so a transient connection reset never creates an extra H1
    trajectory.
    """

    llm = config.llm_config
    llm.timeout = timeout
    llm.max_retries = max_retries
    if not bool(getattr(llm, "stream", False)):
        llm.stream_idle_timeout_ms = max(1000, int(timeout * 1000))
    config.retry_attempts = 1


def archive_workspace(draft_dir: Path, destination: Path) -> Path:
    """Copy one final private H2 workspace to an immutable artifact path.

    The destination must not exist.  Refusing overwrite keeps retries and
    review iterations from silently replacing the filesystem that produced a
    recorded trajectory.
    """

    source = Path(draft_dir)
    target = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"candidate workspace does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"candidate artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return target.resolve()


REPAIR_MSG = (
    "\n\n## REPAIR SESSION (frozen-executor takeover)\n"
    "A previous proposer session left this H2 workspace INVALID and "
    "unsubmitted. Recorded failure: {failure}\n"
    "Your ONLY job is to drive THIS existing workspace to a valid submission "
    "with the smallest coherent portfolio supported by the task diagnosis. "
    "Preserve valid high-value changes; repair, remove, or replace only the "
    "surface that prevents a safe submission. Do not replace a failed tool "
    "with quick_probe/score/budget relay code. If the current workspace "
    "files are included in this message, only records marked complete=true "
    "count as inspected. Read each truncated/omitted exact path before a raw "
    "edit; atomic upsert needs no re-inspection. Procedure: "
    "1) validate_harness to see the current errors; 2) apply minimal fixes "
    "— shrink, remove, or replace an unfixable change while preserving the "
    "strongest coherent validated subset. Prefer atomic upsert for generated "
    "components; raw validation only reports missing plumbing "
    "and never writes it. If a LENGTH CAP is reported "
    "(e.g. prompt.md over "
    "10000 chars), REPLACE a long section with a shorter one "
    "(old_text/new_text) — appending is refused over the cap; "
    "3) validate_harness; 4) submit_harness."
)


def _run_h1_agent(
    *, user_message: str, base_url: str, model: str, api_key: str,
    seed: Optional[int], timeout: float, max_retries: int,
    max_iterations: Optional[int] = None,
) -> tuple:
    """One NexAU H1 agent pass over the CURRENT propose session's draft.

    Returns (trajectory, err, resolved_config). The caller owns the
    session/contextvar scope.
    """
    from nexau import Agent, AgentConfig

    config = AgentConfig.from_yaml(H1_PACKAGE / "agent.yaml")
    allowed_tool_names: Optional[set[str]] = None
    try:
        from outer.harness.tools.runtime import get_session
        role = get_session().portfolio_role()
    except Exception:
        role = None
    if role is not None:
        allowed_tool_names = set(_portfolio_tool_names(
            str(getattr(role, "required_kind", "open_portfolio"))
        ))
        config.tools = [
            row for row in (config.tools or [])
            if str(getattr(row, "name", "")) in allowed_tool_names
        ]
    from inner.runtime.harness_runner import _clamp_context
    _clamp_context(config)
    llm = config.llm_config
    llm.model = model
    llm.base_url = base_url
    llm.api_key = api_key
    _configure_transport(config, timeout=timeout, max_retries=max_retries)
    if max_iterations is not None:
        config.max_iterations = int(max_iterations)
    extra = getattr(llm, "extra_params", None)
    if not isinstance(extra, dict):
        extra = {}
        try:
            llm.extra_params = extra
        except Exception:
            pass
    eb = extra.setdefault("extra_body", {})
    eb.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if seed is not None:
        eb["seed"] = seed

    declarative_surface = _declarative_h1_surface(
        allowed_tool_names=allowed_tool_names,
    )
    resolved_config = {
        "max_iterations": int(config.max_iterations),
        "max_context_tokens": int(config.max_context_tokens),
        "sampling": {
            "max_tokens": int(llm.max_tokens),
            "temperature": float(llm.temperature),
            "top_p": float(llm.top_p),
            "top_k": extra.get("top_k", eb.get("top_k")),
            "enable_thinking": False,
        },
        "transport": {
            "timeout_seconds": float(timeout),
            "max_retries": int(max_retries),
            "agent_retry_attempts": int(config.retry_attempts),
        },
        "tools": [
            str(getattr(row, "name", "")) for row in (config.tools or [])
        ],
        "middlewares": declarative_surface["middlewares"],
        "declarative_agent": declarative_surface,
    }

    trajectory: List[Dict[str, Any]] = []
    err: Optional[str] = None
    agent = None
    try:
        agent = Agent(config=config)
        agent.run(message=user_message)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        if agent is not None:
            trajectory = _dump_history(agent)
    return trajectory, err, resolved_config


def maybe_auto_finalize(session, trajectory) -> bool:
    """FINALIZE-001: submit a finished-but-unsubmitted workspace, once.

    7/11 invalid slots in the fix3 taxonomy ended with a complete, often
    valid workspace and no submit call.  The normal submit path runs with
    every guard live (placeholder refusal, novelty guard, no-op check), so
    an untouched scaffold or a futile repeat still comes back invalid —
    this only rescues work the model actually finished, and the flag is
    recorded so analysis can separate auto-finalized slots.
    """
    if session.submitted:
        return False
    try:
        if session.validated_revision != session.workspace_revision:
            v = session.validate_harness()
            if not str(v).startswith("VALID"):
                trajectory.append({
                    "role": "framework",
                    "content": "[FINALIZE-001] auto-finalize skipped: the "
                               "final workspace does not validate and was "
                               "preserved unchanged for attribution: "
                               + str(v)[:200],
                })
                return False
        out = session.submit_harness()
    except Exception as exc:
        session.record_h1_runtime_event({
            "kind": "auto_finalize_infrastructure_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }, training_blocking=True)
        trajectory.append({
            "role": "framework",
            "content": "[FINALIZE-001] trusted auto-finalize failed; do not "
                       "rewrite the H2 for this framework witness: "
                       f"{type(exc).__name__}: {str(exc)[:160]}",
        })
        return False
    # the failure shape is "SUBMITTED but INVALID (minimum reward)" — only a
    # real acceptance counts
    if isinstance(out, str) and "Candidate H2 accepted" in out:
        trajectory.append({
            "role": "framework",
            "content": "[FINALIZE-001] iteration budget ended without "
                       "submit_harness; the workspace validated clean and "
                       "was auto-finalized: " + out[:200],
        })
        return True
    trajectory.append({
        "role": "framework",
        "content": "[FINALIZE-001] auto-finalize attempted and REFUSED: "
                   + str(out)[:200],
    })
    return False



def _operator_brief_suffix(task_id: Optional[str]) -> str:
    """Optional operator-authored research brief for ONE task.

    Enabled by ``SAH_H1_OPERATOR_BRIEF_DIR=<dir>`` containing ``<task_id>.md``.
    The brief is method-level guidance for the proposer (problem structure,
    optimisation recipe, pitfalls) and is appended to the proposer user message
    before the workspace preload. It must never contain a solution; the H1
    system prompt's no-embedded-solution rule still applies to every draft.
    Absent directory or file -> no change to the message.
    """
    d = os.environ.get("SAH_H1_OPERATOR_BRIEF_DIR", "").strip()
    if not d or not task_id:
        return ""
    path = Path(d) / f"{task_id}.md"
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not text:
        return ""
    cap = int(os.environ.get("SAH_H1_OPERATOR_BRIEF_CAP", "12000"))
    if len(text) > cap:
        text = text[:cap].rstrip() + "\n[... operator brief truncated ...]"
    return ("\n\n## Operator research brief (task-specific method guidance from the "
            "campaign operator; authoritative on problem structure and on which "
            "optimisation recipe reaches the target; NOT a solution - never "
            "hard-code coordinates or copy any external solution)\n" + text)

def run_once(k: int, *, base_spec: Dict[str, Any], user_message: str,
             futile_signatures=None,
              task_id: Optional[str] = None,
              base_url: str, model: str, api_key: str, seed: Optional[int],
              timeout: float, max_retries: int = 2,
              repair_base_url: Optional[str] = None,
              repair_model: Optional[str] = None,
              repair_policy_identity: Optional[str] = None,
              repair_endpoint_role: str = "frozen_base_executor",
              repair_max_iters: int = 16,
              workspace_archive: Optional[Path] = None,
              round_index: int = 0) -> CandidateRecord:
    trajectory: List[Dict[str, Any]] = []
    err: Optional[str] = None
    operator_brief = _operator_brief_suffix(task_id)
    if operator_brief and k == 0:
        print(f"[propose] operator brief attached for {task_id}: "
              f"{len(operator_brief)} chars sha256 "
              f"{hashlib.sha256(operator_brief.encode()).hexdigest()[:12]}")
    primary_message = finalize_proposer_user_message(user_message, operator_brief)
    repair_message = ""
    primary_file_events: List[Dict[str, str]] = []
    repair_file_events: List[Dict[str, str]] = []
    archived_workspace: Optional[str] = None
    primary_archived_workspace: Optional[str] = None
    repair_archived_workspace: Optional[str] = None
    primary_component_transactions: List[Dict[str, Any]] = []
    repair_component_transactions: List[Dict[str, Any]] = []
    primary_h1_runtime_events: List[Dict[str, Any]] = []
    primary_training_block_reason: Optional[str] = None
    artifact_errors: List[str] = []
    h1_config: Dict[str, Any] = {}
    # Every parallel proposer gets a complete private copy.  H1 can inspect and
    # edit real H2 files, but can never mutate the shared accepted package or
    # another candidate's draft.
    with TemporaryDirectory(prefix=f"h1_candidate_{k:02d}_") as td:
        draft_dir = Path(td) / "h2"
        try:
            materialize(
                base_spec,
                draft_dir,
                meta={"effective": base_spec},
                # Historical parents may predate proposer-owned component
                # catalogs. Preserve their prompt byte-for-byte in the private
                # draft so H1—not runtime—must repair it before submission.
                validate_prompt=False,
            )
        except Exception as e:
            return CandidateRecord(
                k=k,
                valid=False,
                errors=[f"cannot initialize private H2 workspace: {type(e).__name__}: {e}"],
                stop_reason="workspace_error",
                h1_runtime_events=[{
                    "kind": "workspace_initialization_error",
                    "error": f"{type(e).__name__}: {e}",
                    "training_blocking": True,
                }],
                infrastructure_failure=True,
                training_block_reason="workspace_initialization_error",
                exact_user_message=primary_message,
            )
        # NOTHING is pre-written into the draft. Earlier revisions seeded a
        # mounted starter tool (BAILOUT-005 / SCAFFOLD-001) so a slot under a
        # hard "must ship a tool" contract could always satisfy it.  Both the
        # contract and the seeding are gone: a component that exists because
        # the pipeline wrote it measures nothing, and it polluted the record —
        # 148 of 168 generated-tool proposals across 21 campaigns were that
        # untouched starter. The proposer now chooses the smallest portfolio
        # from the task evidence; the pipeline never writes a free component.
        # PRELOAD-001: render the whole mutable workspace into the initial
        # context so the proposer edits directly instead of cat-ing files for
        # half its budget; the iteration cap shrinks accordingly.
        preload = os.environ.get("SAH_H1_PRELOAD", "1") == "1"
        h1_max_iters: Optional[int] = None
        session = ProposeSession(
            base_spec=base_spec,
            draft_dir=draft_dir,
            slot_index=k,
            portfolio_round=round_index,
            task_id=task_id,
            futile_signatures=set(futile_signatures or ()),
        )
        if preload:
            ws_text, rendered = render_workspace(draft_dir)
            primary_message = finalize_proposer_user_message(
                user_message, operator_brief, ws_text
            )
            session.mark_preloaded(rendered)
            # Atomic transactions and direct workspace editors share the same
            # fixed 12-call H1 budget in every condition group.
            h1_max_iters = int(os.environ.get(
                "SAH_H1_MAX_ITERS",
                os.environ.get("SAH_H1_COMPONENT_MAX_ITERS", "12")))
        with propose_scope(session):
            trajectory, err, h1_config = _run_h1_agent(
                user_message=primary_message, base_url=base_url, model=model,
                api_key=api_key, seed=seed, timeout=timeout,
                max_retries=max_retries, max_iterations=h1_max_iters,
            )
        primary_file_events = list(session.file_events)
        primary_guard = session.schema_guard_summary()
        primary_component_transactions = list(session.component_transactions)
        if err and session.h1_training_block_reason() is None:
            session.record_h1_runtime_event({
                "kind": "h1_agent_exception",
                "error": err,
            }, training_blocking=True)
        # Deterministic finalization belongs to phi and runs before any frozen
        # repair policy is considered. This avoids a takeover when the primary
        # workspace is already valid and the model merely omitted submit.
        auto_finalized = (not err) and maybe_auto_finalize(session, trajectory)
        primary_h1_runtime_events = list(session.h1_runtime_events)
        primary_training_block_reason = session.h1_training_block_reason()

        # REPAIR-001: a slot phi failed is taken over by the FROZEN executor
        # model on an isolated clone of the primary draft. Capability parity keeps the
        # no-strong-model-leakage invariant; phi's training credit stays bound
        # to its own (failed) trajectory — the collector assigns it the
        # minimum reward.
        repair_info: Dict[str, Any] = {}
        repair_trajectory: List[Dict[str, Any]] = []
        primary_valid = bool(
            session.submitted and session.effective is not None
            and session.changed_fields and not session.errors
        )
        if (not primary_valid and repair_base_url
                and primary_training_block_reason is None):
            original_errors = list(
                session.errors or ["proposer never called submit_harness"]
            )
            failure = "; ".join(original_errors)[:600]
            if workspace_archive is not None:
                try:
                    primary_archived_workspace = str(archive_workspace(
                        draft_dir,
                        Path(workspace_archive).parent / "primary_h2",
                    ))
                except Exception as exc:
                    artifact_errors.append(
                        "cannot archive primary pre-repair workspace: "
                        f"{type(exc).__name__}: {exc}"
                    )

            repair_draft_dir = Path(td) / "repair_h2"
            try:
                shutil.copytree(
                    draft_dir, repair_draft_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
            except Exception as exc:
                repair_info = {
                    "attempted": False,
                    "succeeded": False,
                    "blocked": "cannot clone primary workspace",
                    "repair_error": f"{type(exc).__name__}: {exc}",
                    "original_errors": original_errors,
                }
            else:
                repair_session = ProposeSession(
                    base_spec=base_spec, draft_dir=repair_draft_dir,
                    slot_index=k,
                    portfolio_round=round_index,
                    task_id=task_id,
                    futile_signatures=set(futile_signatures or ()),
                )
                repair_suffix = REPAIR_MSG.format(failure=failure)
                if preload:
                    ws_text, rendered = render_workspace(repair_draft_dir)
                    repair_suffix += ws_text
                    repair_session.mark_preloaded(rendered)
                repair_message = finalize_proposer_user_message(
                    user_message, repair_suffix
                )
                with propose_scope(repair_session):
                    resolved_repair_model = repair_model or model
                    repair_trajectory, repair_err, repair_h1_config = _run_h1_agent(
                        user_message=repair_message,
                        base_url=repair_base_url, model=resolved_repair_model,
                        api_key=api_key,
                        seed=(seed + 500000) if seed is not None else None,
                        timeout=timeout, max_retries=max_retries,
                        max_iterations=repair_max_iters,
                    )
                if repair_err and repair_session.h1_training_block_reason() is None:
                    repair_session.record_h1_runtime_event({
                        "kind": "h1_agent_exception",
                        "error": repair_err,
                    }, training_blocking=True)
                repair_file_events = list(repair_session.file_events)
                repair_component_transactions = list(
                    repair_session.component_transactions
                )
                repair_valid = bool(
                    repair_session.submitted
                    and repair_session.effective is not None
                    and repair_session.changed_fields
                    and not repair_session.errors
                )
                repair_info = {
                    "attempted": True,
                    "succeeded": repair_valid,
                    "repair_model": resolved_repair_model,
                    "repair_policy_identity": (
                        repair_policy_identity or resolved_repair_model
                    ),
                    "repair_endpoint_role": repair_endpoint_role,
                    "repair_base_url": repair_base_url,
                    "original_errors": original_errors,
                    "repair_error": repair_err,
                    "llm_calls": sum(
                        1 for m in repair_trajectory
                        if str(m.get("role", "")).lower().endswith("assistant")
                    ),
                    "h1_config": repair_h1_config,
                    "component_transactions": repair_component_transactions,
                    "h1_runtime_events": list(repair_session.h1_runtime_events),
                    "infrastructure_failure": bool(
                        repair_session.h1_training_block_reason()
                    ),
                }
                if workspace_archive is not None:
                    try:
                        repair_archived_workspace = str(archive_workspace(
                            repair_draft_dir,
                            Path(workspace_archive).parent / "repair_h2",
                        ))
                    except Exception as exc:
                        artifact_errors.append(
                            "cannot archive frozen repair workspace: "
                            f"{type(exc).__name__}: {exc}"
                        )
                if repair_valid:
                    session = repair_session

        # Archive before TemporaryDirectory cleanup.  This is intentionally
        # independent of proposal validity: failed code is often the most
        # useful evidence for the next proposer-harness iteration.
        if workspace_archive is not None:
            try:
                archived_workspace = str(
                    archive_workspace(
                        Path(session.draft_dir), Path(workspace_archive)
                    )
                )
                if primary_archived_workspace is None:
                    primary_archived_workspace = archived_workspace
            except Exception as exc:
                artifact_errors.append(
                    "cannot archive final candidate workspace: "
                    f"{type(exc).__name__}: {exc}"
                )
        if repair_info:
            repair_info["primary_workspace_archive"] = primary_archived_workspace
            repair_info["repair_workspace_archive"] = repair_archived_workspace
            repair_info["selected_workspace_archive"] = archived_workspace

    valid = bool(session.submitted and session.effective is not None
                 and session.changed_fields and not session.errors)
    repaired = bool(repair_info.get("succeeded"))
    rec = CandidateRecord(
        k=k, valid=valid,
        raw_submission=session.raw_submission,
        spec=session.partial_spec, effective=session.effective,
        changed_fields=session.changed_fields,
        spec_hash=hs.spec_hash(session.effective) if session.effective else "",
        trajectory=trajectory,
        llm_calls=sum(1 for m in trajectory
                      if str(m.get("role", "")).lower().endswith("assistant")),
        stop_reason=(
                    "h1_infrastructure_failure"
                    if primary_training_block_reason else
                    "repaired_submission" if repaired else
                    ("harness_error" if err else
                     ("auto_finalized" if auto_finalized and valid
                      else "submitted" if session.submitted and session.effective
                      else "invalid_submission" if session.submitted
                      else "no_submission"))),
        exact_user_message=primary_message,
        repair_user_message=repair_message,
        primary_file_events=primary_file_events,
        repair_file_events=repair_file_events,
        workspace_archive=archived_workspace,
        primary_workspace_archive=primary_archived_workspace,
        repair_workspace_archive=repair_archived_workspace,
        primary_component_transactions=primary_component_transactions,
        repair_component_transactions=repair_component_transactions,
        artifact_errors=artifact_errors,
        validations=session.validations,
        failed_validations=session.failed_validations,
        h1_config=h1_config,
        component_transactions=list(session.component_transactions),
        validation_profile=session.validation_profile(),
        h1_runtime_events=primary_h1_runtime_events,
        infrastructure_failure=bool(primary_training_block_reason),
        training_block_reason=primary_training_block_reason,
    )
    rec.auto_finalized = auto_finalized
    rec.review_log = list(session.component_audit)
    rec.schema_guard = primary_guard
    rec.repair = repair_info
    rec.repair_trajectory = repair_trajectory
    if repaired:
        rec.errors = []
    elif err:
        rec.errors = [err]
    elif not session.submitted:
        rec.errors = ["proposer never called submit_harness"]
    elif session.errors:
        rec.errors = session.errors
    return rec


def dedup_group(records: List[CandidateRecord], base_spec: Dict[str, Any]) -> None:
    """Invalidate byte/spec and cosmetic-semantic duplicates within one group."""
    parent_hashes = {
        str(rec.parent_spec_hash) for rec in records if rec.parent_spec_hash
    }
    seen = parent_hashes or {hs.spec_hash(base_spec)}
    seen_semantic: Dict[str, CandidateRecord] = {}
    seen_behavior: Dict[str, CandidateRecord] = {}
    # SAH_SEMANTIC_DEDUP=rollout: keep cosmetic-semantic / behavioural duplicates
    # as valid slots (they still get distinct executor decode seeds, i.e. extra
    # trials of the same structure); byte/spec duplicates stay rejected.
    keep_dups = os.environ.get("SAH_SEMANTIC_DEDUP", "strict") == "rollout"
    for rec in sorted(records, key=lambda r: r.k):
        if not rec.valid:
            continue
        if rec.spec_hash in seen:
            rec.valid = False
            rec.errors = ["duplicate of another candidate (or of the base)"]
            continue
        seen.add(rec.spec_hash)
        rec.component_semantic_fingerprint_record = (
            component_semantic_fingerprint_record(rec.effective)
        )
        rec.component_semantic_fingerprint = (
            rec.component_semantic_fingerprint_record.get("fingerprint", "")
        )
        rec.component_behavior_fingerprint_record = (
            component_behavior_fingerprint_record(
                rec.effective, base_spec, rec.validation_profile,
            )
        )
        rec.component_behavior_fingerprint = (
            rec.component_behavior_fingerprint_record.get("fingerprint", "")
        )
        fingerprint = rec.component_semantic_fingerprint
        if not fingerprint:
            continue
        prior = seen_semantic.get(fingerprint)
        if prior is None:
            seen_semantic[fingerprint] = rec
        else:
            if not keep_dups:
                rec.valid = False
                rec.errors = [
                    "semantic duplicate within exact-condition group: executable "
                    f"structure is equivalent to candidate k={prior.k} after "
                    "literal/identifier/no-op normalization"
                ]
            rec.within_group_semantic_duplicates.append({
                "k": prior.k,
                "fingerprint": fingerprint,
                "duplicate_of": True,
            })
            prior.within_group_semantic_duplicates.append({
                "k": rec.k,
                "fingerprint": fingerprint,
                "duplicate_of": False,
            })
            if not keep_dups:
                continue
        behavior = rec.component_behavior_fingerprint
        if not behavior:
            continue
        behavior_prior = seen_behavior.get(behavior)
        if behavior_prior is None:
            seen_behavior[behavior] = rec
            continue
        if not keep_dups:
            rec.valid = False
            rec.errors = [
                "behavioral duplicate within exact-condition group: trusted "
                f"middleware truth table matches candidate k={behavior_prior.k}"
            ]
        rec.within_group_behavior_duplicates.append({
            "k": behavior_prior.k,
            "fingerprint": behavior,
            "duplicate_of": True,
        })
        behavior_prior.within_group_behavior_duplicates.append({
            "k": rec.k,
            "fingerprint": behavior,
            "duplicate_of": False,
        })


PROPOSAL_GATE_MODES = ("off", "all")


def parse_gate_mode(mode: str, k: int) -> tuple:
    """Return (kind, min_valid) for a SAH_PROPOSAL_GATE value.

    ``off`` -> no enforcement; ``all`` -> every charged slot valid;
    ``min<N>`` (e.g. ``min6``) -> at least N of K slots must hold a valid
    materialized candidate (REPAIR-001 relaxation: invalid slots are charged,
    trained at minimum reward, and simply do not roll out).
    """

    if mode in ("off", "all"):
        return mode, k if mode == "all" else 0
    if mode.startswith("min") and mode[3:].isdigit():
        n = int(mode[3:])
        if 1 <= n <= k:
            return "min", n
    raise ValueError(
        f"unknown SAH_PROPOSAL_GATE mode {mode!r}; expected off, all, or min<N>"
    )


def proposal_gate_issues(
    candidates: List[Dict[str, Any]], *, k: int,
) -> List[str]:
    """Fail-closed ``all`` proposal gate over one task's candidate entries.

    A causal method-comparison round may only charge intervention slots that
    hold a valid, materialized candidate H2. The fair16-v5 campaign showed why
    this must be a hard gate: 16 of 19 update_harness circle rounds proposed
    0/8 valid H2, every slot silently ran the incumbent-fallback package, and
    the rounds still completed as official curve points. Returns a list of
    per-slot issues; empty means the task passes.
    """

    issues: List[str] = []
    if len(candidates) != k:
        issues.append(
            f"charged slot count {len(candidates)} != K={k}"
        )
    for cand in sorted(candidates, key=lambda c: int(c.get("k", -1))):
        slot = int(cand.get("k", -1))
        if not cand.get("valid"):
            reason = cand.get("stop_reason") or "invalid"
            first_error = (cand.get("errors") or ["unspecified"])[0]
            issues.append(
                f"k{slot:02d}: invalid proposal ({reason}: {first_error})"
            )
        elif not cand.get("dir"):
            issues.append(f"k{slot:02d}: valid proposal was never materialized")
    return issues
