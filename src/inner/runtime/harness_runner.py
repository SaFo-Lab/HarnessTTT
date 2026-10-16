"""Run H2 (M0 + the declarative NexAU harness package) on one EFT task.

Loads ``inner/harness/agent.yaml`` (the harness surface: system prompt + tools +
skills + middlewares + sampling), injects the frozen-executor serving endpoint
and the eval budget at runtime, and drives one task's edit->evaluate loop.

The harness *definition* lives entirely in ``inner/harness/`` — this module only
wires the endpoint/budget and collects results, so the outer proposer can later
mutate the package without touching this code.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ensure `inner...` (and the agent.yaml tool/middleware bindings) are importable
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inner.tasks.eft_task import EFTTask  # noqa: E402
from inner.evaluation.eval_runner import EvalOutcome  # noqa: E402
from inner.runtime.session import BudgetLedger, InnerSession, session_scope  # noqa: E402

AGENT_YAML = Path(__file__).resolve().parents[1] / "harness" / "agent.yaml"


@dataclass
class LLMEndpoint:
    model: str = "qwen3.5-9b"
    base_url: str = "http://127.0.0.1:8800/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = 20
    max_tokens: int = 8192
    timeout: float = 600.0
    max_retries: int = 2
    enable_thinking: bool = False  # Qwen3.5: keep outputs clean/fast (matches the earlier runtime)
    seed: Optional[int] = None


@dataclass
class H2Config:
    max_evaluator_calls: int = 10
    max_probe_calls: int = 30
    max_iterations: Optional[int] = None  # None -> keep agent.yaml's value
    eval_timeout_s: Optional[float] = None
    python_exe: Optional[str] = None      # interpreter with task deps for eval subprocess


@dataclass
class InnerResult:
    task_id: str
    source: str
    best_score: float
    seed_score: float
    best_metrics: Dict[str, float]
    best_program: str
    stop_reason: str
    ledger: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    middleware_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skill_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    decode_seed: Optional[int] = None
    score_eligible: bool = True
    trajectory: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


def _expected_generated_middlewares(agent_yaml: Path) -> List[str]:
    """Names of generated middleware that must mount and enter their hook."""

    payload = yaml.safe_load(Path(agent_yaml).read_text()) or {}
    names: List[str] = []
    for row in payload.get("middlewares", []):
        if not isinstance(row, dict):
            continue
        binding = str(row.get("import", ""))
        if not binding.endswith(":GeneratedMiddleware"):
            continue
        module = binding.split(":", 1)[0]
        names.append(module.rsplit(".", 1)[-1])
    return names


def _expected_generated_components(agent_yaml: Path) -> Dict[str, Dict[str, str]]:
    payload = yaml.safe_load(Path(agent_yaml).read_text()) or {}
    tools: Dict[str, str] = {}
    for row in payload.get("tools", []):
        if isinstance(row, dict) and "custom_runtime" in str(row.get("binding", "")):
            extra_kwargs = row.get("extra_kwargs") or {}
            tools[str(row.get("name"))] = str(
                extra_kwargs.get("_sah_py_path")
                or extra_kwargs.get("py_path", "")
            )
    skills: Dict[str, str] = {}
    for source in payload.get("skills", []):
        name = Path(str(source)).name
        if name != "discovery-optimization":
            skills[name] = str(source)
    return {"tools": tools, "skills": skills}


def _required_generated_skills(
    harness_dir: Path, mounted_skills: Dict[str, str],
) -> List[str]:
    """Skills changed by *this* H1 proposal and therefore required for credit.

    A materialized package contains inherited generated skills as well as the
    additions/updates made by the current proposal. Every mounted skill is
    enacted for stable harness semantics; component lineage is the separate,
    authoritative credit boundary. Only ``added`` and ``updated`` skills are
    interventions introduced by this candidate.

    Legacy packages without usable lineage predate that distinction.  Fail
    conservatively by requiring all mounted generated skills for those packages.
    """

    manifest_path = Path(harness_dir) / "component_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        lineage = (manifest.get("lineage") or {}).get("new_skills")
        if not isinstance(lineage, list):
            raise ValueError("new_skills lineage is absent")
        required = {
            str(row.get("name"))
            for row in lineage
            if isinstance(row, dict)
            and str(row.get("status")) in {"added", "updated"}
        }
        return sorted(required.intersection(mounted_skills))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return sorted(mounted_skills)


def _mounted_skill_playbooks(
    agent_yaml: Path,
    mounted_skills: Dict[str, str],
    active_skills: List[str],
) -> Dict[str, str]:
    """Read exact generated playbooks for deterministic pre-edit delivery."""

    playbooks: Dict[str, str] = {}
    for name in active_skills:
        source = Path(mounted_skills[name])
        if not source.is_absolute():
            source = Path(agent_yaml).parent / source
        try:
            body = (source / "SKILL.md").read_text().strip()
        except OSError:
            continue
        if body:
            playbooks[name] = body
    return playbooks


def _resolved_component_path(agent_yaml: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else Path(agent_yaml).parent / path


def _yaml_description(path: Path) -> str:
    """Best-effort component description without hiding a mounted component."""

    try:
        payload = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, TypeError, yaml.YAMLError):
        return "description unavailable; inspect the callable schema before use"
    description = str(payload.get("description", "")).strip()
    if not description:
        return "description unavailable; inspect the callable schema before use"
    # Generated schemas are bounded at validation, but inherited packages may
    # predate that gate. Keep the authoritative contract compact and explicit.
    return " ".join(description.split())[:1200]


def _runtime_component_contract(
    agent_yaml: Path,
    *,
    evaluator_budget: int,
    probe_budget: int,
    generated_skill_playbooks: Optional[Dict[str, str]] = None,
    current_candidate_skills: Optional[List[str]] = None,
) -> str:
    """Render the actual mounted H2 surface as one authoritative contract.

    Proposer prose is useful for strategy, but it is not a reliable registry:
    cand03 mounted a generated skill while stale optional-use wording caused the
    executor to ignore it. This contract is generated from the exact agent.yaml
    that NexAU loads, distinguishes delivery semantics, and gives generated
    tools an explicit decision protocol without forcing irrelevant calls.
    """

    agent_yaml = Path(agent_yaml)
    payload = yaml.safe_load(agent_yaml.read_text()) or {}
    playbooks = generated_skill_playbooks or {}
    current = set(current_candidate_skills or [])
    lines = [
        "# Authoritative runtime component contract",
        "<RuntimeComponentContract>",
        (
            "This block is generated by the trusted runtime from the exact "
            "mounted agent package. It overrides any stale or contradictory "
            "component catalog in proposer-authored prose."
        ),
        f"Official evaluator budget: {int(evaluator_budget)} call(s).",
        f"Approximate probe budget: {int(probe_budget)} call(s).",
        "",
        "## Callable tools",
    ]
    for row in payload.get("tools", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip() or "unnamed-tool"
        binding = str(row.get("binding", ""))
        generated = "custom_runtime" in binding
        description = _yaml_description(
            _resolved_component_path(agent_yaml, str(row.get("yaml_path", "")))
        )
        if generated:
            lines.append(
                f"- `{name}` [GENERATED, CONDITIONAL]: {description}"
            )
        else:
            lines.append(f"- `{name}` [CORE]: {description}")
    lines.extend([
        "- `LoadSkill` [FRAMEWORK]: load the base skill named below; generated "
        "skills marked AUTO-ENACTED are already delivered and must not be reloaded.",
        "",
        "Generated-tool rule: before your first `edit_solution`, state in the "
        "same assistant turn whether each GENERATED tool's described trigger "
        "currently applies and cite task/search evidence. If it applies, use the "
        "tool at the first relevant point before spending the corresponding "
        "official evaluation. If it does not apply, do not call it merely for "
        "compliance.",
        "",
        "## Skills",
    ])
    for source in payload.get("skills", []):
        name = Path(str(source)).name
        if name == "discovery-optimization":
            lines.append(
                "- `discovery-optimization` [BASE, LOAD REQUIRED]: call "
                "`LoadSkill` for it before the first program edit."
            )
            continue
        delivery = (
            "AUTO-ENACTED, CURRENT CANDIDATE"
            if name in current else "AUTO-ENACTED, INHERITED"
        )
        if name in playbooks:
            lines.append(
                f"- `{name}` [{delivery}]: its complete playbook appears below. "
                "Follow it in this rollout; no `LoadSkill` call is needed."
            )
        else:
            lines.append(
                f"- `{name}` [DELIVERY ERROR]: mounted but its complete playbook "
                "could not be read. Do not begin editing."
            )
    lines.extend(["", "## Automatic middleware"])
    for row in payload.get("middlewares", []):
        if not isinstance(row, dict):
            continue
        binding = str(row.get("import", ""))
        module, _, class_name = binding.partition(":")
        generated = class_name == "GeneratedMiddleware"
        name = module.rsplit(".", 1)[-1] if generated else (class_name or module)
        kind = "GENERATED, AUTOMATIC" if generated else "RUNTIME, AUTOMATIC"
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        description = ""
        if generated:
            descriptor = agent_yaml.parent / "middlewares" / f"{name}.middleware.yaml"
            description = _yaml_description(descriptor)
        suffix = f": {description}" if description else ""
        lines.append(
            f"- `{name}` [{kind}, params={json.dumps(params, sort_keys=True)}]"
            f"{suffix}"
        )
    lines.extend([
        "",
        "Middleware rule: middleware runs automatically. Never try to call it. "
        "Treat its framework messages and tool gates as active runtime state.",
        "",
        "## First-edit obligation",
        "Before the first `edit_solution`, load the BASE skill, read every "
        "AUTO-ENACTED playbook below, and include a concise `Component plan:` "
        "in the same assistant turn as the edit. Map the edit to the skill "
        "guidance and record the GENERATED-tool trigger decisions. Component "
        "names alone are not evidence of enactment.",
        "</RuntimeComponentContract>",
    ])
    return "\n".join(lines)


def _is_score_eligible(stop_reason: str, participation_issues: List[str]) -> bool:
    """Publish only a completed harness route whose proposed components enacted."""

    return stop_reason != "harness_error" and not participation_issues


def _record_skill_loads(session: InnerSession, trajectory: Optional[List[Dict[str, Any]]]) -> None:
    """Audit successful LoadSkill results and whether they preceded editing.

    Counting the assistant's requested tool call alone would incorrectly mark a
    missing/failed skill as enacted.  Pair each request with its successful tool
    result and retain ordering relative to the first program edit.
    """

    pending: Dict[str, tuple[str, bool]] = {}
    editing_started = False
    for msg_index, message in enumerate(trajectory or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        blocks = content if isinstance(content, list) else []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", ""))
            if block_type == "tool_use":
                tool_name = str(block.get("name", ""))
                if tool_name == "edit_solution":
                    editing_started = True
                    continue
                if tool_name != "LoadSkill":
                    continue
                args = block.get("input", {})
                skill_name = str(args.get("skill_name", "")) \
                    if isinstance(args, dict) else ""
                if skill_name not in session.skill_audit:
                    continue
                call_id = str(
                    block.get("id")
                    or block.get("tool_use_id")
                    or f"{msg_index}:{block_index}"
                )
                pending[call_id] = (skill_name, not editing_started)
            elif block_type == "tool_result":
                call_id = str(block.get("tool_use_id") or block.get("id") or "")
                requested = pending.pop(call_id, None)
                if requested is None or bool(block.get("is_error", False)):
                    continue
                skill_name, before_first_edit = requested
                session.record_skill_load(
                    skill_name, before_first_edit=before_first_edit
                )


def _initial_message(
    task: EFTTask,
    seed_score: float,
    seed_valid: float,
    budget: int,
    generated_skill_playbooks: Optional[Dict[str, str]] = None,
    component_contract: Optional[str] = None,
) -> str:
    msg = (
        f"# Task\n{task.spec.strip()}\n\n"
        f"# Current program\n```python\n{task.initial_program}\n```\n\n"
        f"# Baseline\nThe seed program scores combined_score = {seed_score:.6g} "
        f"(validity {seed_valid:g}). Beat it. You have {budget} evaluations.\n\n"
    )
    parents = getattr(task, "crossover_parents", None)
    if parents:
        msg += "# Alternative high-scoring approaches (different search basins)\n"
        msg += ("These reached similar scores via DIFFERENT strategies. Consider "
                "hybridizing their ideas with the current program — crossover often "
                "escapes local optima that pure mutation cannot:\n")
        for i, par in enumerate(parents):
            msg += (f"\n## Alternative {i+1} (score {par.get('score', 0):.6g})\n"
                    f"```python\n{par.get('program', '')[:4000]}\n```\n")
        msg += "\n"
    if component_contract:
        msg += f"{component_contract.strip()}\n\n"
    msg += "Load the discovery-optimization skill first."
    if generated_skill_playbooks:
        rendered = "\n\n".join(
            f"## `{name}`\n<SkillDetails>\n{body}\n</SkillDetails>"
            for name, body in generated_skill_playbooks.items()
        )
        msg += (
            "\n\n# Automatically enacted proposer-generated skills\n"
            "The trusted runtime has inserted the complete playbooks below "
            "before your first edit. They are interventions in the candidate "
            "H2 being evaluated: follow them in this rollout. You do not need "
            "to call `LoadSkill` for these already-delivered playbooks.\n\n"
            f"{rendered}\n"
        )
    msg += (
        "\nThen propose an improved EVOLVE-BLOCK with edit_solution and score "
        "it with evaluate_solution."
    )
    return msg


def _override_llm(config, ep: LLMEndpoint, *, preserve_sampling: bool = False,
                  top_k_override: Optional[int] = None) -> None:
    """Inject the serving endpoint into the loaded agent config.

    preserve_sampling: keep the agent.yaml's temperature/top_p/max_tokens
    (candidate H2 packages own their sampling — it is part of the genome);
    only the endpoint/model/timeouts are injected. top_k_override applies the
    candidate's top_k (which lives outside agent.yaml, in its meta.json).
    """
    llm = config.llm_config
    llm.model = ep.model
    llm.base_url = ep.base_url
    llm.api_key = ep.api_key
    llm.timeout = ep.timeout
    llm.max_retries = ep.max_retries
    # NexAU's default HTTP read timeout is a separate 300-second stream-idle
    # budget.  These local vLLM calls are non-streaming, so leaving that default
    # silently truncates a requested 1800-second response timeout to 300s.
    if not bool(getattr(llm, "stream", False)):
        llm.stream_idle_timeout_ms = max(1000, int(ep.timeout * 1000))
    # NexAU has two independent retry layers. LLMConfig.max_retries retries the
    # failed provider request in place; AgentConfig.retry_attempts can replay
    # an outer agent iteration. Never let the transport setting create a
    # second logical solution trajectory.
    config.retry_attempts = 1
    if not preserve_sampling:
        llm.temperature = ep.temperature
        llm.top_p = ep.top_p
        llm.max_tokens = ep.max_tokens
    extra = getattr(llm, "extra_params", None)
    if not isinstance(extra, dict):
        extra = {}
        try:
            llm.extra_params = extra
        except Exception:
            return
    extra_body = extra.setdefault("extra_body", {})
    top_k = top_k_override if top_k_override is not None else ep.top_k
    if top_k is not None:
        extra_body["top_k"] = top_k
    extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = ep.enable_thinking
    if ep.seed is not None:
        extra_body["seed"] = int(ep.seed)


def _extract_trajectory(agent) -> List[Dict[str, Any]]:
    traj: List[Dict[str, Any]] = []
    try:
        for msg in agent.history:
            try:
                traj.append(msg.model_dump())
            except Exception:
                traj.append({"role": str(getattr(msg, "role", "?")),
                             "content": str(getattr(msg, "content", ""))})
    except Exception:
        pass
    return traj


def _count_llm_calls(agent) -> int:
    n = 0
    try:
        for msg in agent.history:
            if str(getattr(msg, "role", "")).lower().endswith("assistant"):
                n += 1
    except Exception:
        pass
    return n


def _finalize_llm_accounting(ledger: BudgetLedger, agent: Any) -> None:
    """Keep provider calls authoritative and history turns observational.

    One provider attempt need not produce one assistant history message: an
    exception may have no message, while framework correction can produce a
    different turn count.  The immutable usage middleware records attempts;
    assistant history is only a fallback for legacy packages where that
    middleware never emitted a record.
    """

    ledger.assistant_turns = _count_llm_calls(agent)
    if not ledger.llm_usage_records:
        ledger.llm_calls = ledger.assistant_turns
        ledger.llm_calls_source = "assistant_history_fallback"



def _clamp_context(config) -> None:
    """Hold the agent's context to what the SERVER was actually started with.

    CONTEXT-001.  agent.yaml carries max_context_tokens as a literal, and the
    proposer may inherit or edit it, so it can drift away from the vLLM
    ``--max-model-len`` the pool launched with.  When the agent believes it has
    more room than the server allows, the server rejects the request mid-
    rollout and the whole trajectory is lost -- observed on an earlier campaign
    at 24577 input tokens against a 32768 server.  The campaign exports one
    number (SAH_MAX_CONTEXT_TOKENS, = CURVE_MAX_MODEL_LEN) and everything is
    clamped to it; unset, the packaged value stands.
    """
    import os
    raw = os.environ.get("SAH_MAX_CONTEXT_TOKENS")
    if not raw or not raw.isdigit():
        return
    cap = int(raw)
    current = getattr(config, "max_context_tokens", None)
    if isinstance(current, int) and current > cap:
        config.max_context_tokens = cap
    for mw in (getattr(config, "middlewares", None) or []):
        params = getattr(mw, "params", None)
        if isinstance(params, dict) and isinstance(
                params.get("max_context_tokens"), int) \
                and params["max_context_tokens"] > cap:
            params["max_context_tokens"] = cap

def run_task(
    task: EFTTask,
    *,
    endpoint: LLMEndpoint,
    h2: H2Config,
    keep_trajectory: bool = True,
    checkpoint_path: Optional[str] = None,
    harness_dir: Optional[Path] = None,
    seed_outcome: Optional[EvalOutcome] = None,
) -> InnerResult:
    """Run H2 on one EFT task; return best program + full ledger/trajectory.

    harness_dir: run a *candidate* H2 package (outer loop) instead of the
    built-in ``inner/harness``. The dir must contain ``agent.yaml``; it is put
    first on ``sys.path`` so its ``middlewares/`` package resolves. One
    candidate per process — different candidates may define the same module
    names, so never load two candidate packages into one interpreter.
    """
    from nexau import Agent, AgentConfig  # lazy: needs the nexau env

    agent_yaml = AGENT_YAML
    cand_top_k: Optional[int] = None
    if harness_dir is not None:
        harness_dir = Path(harness_dir).resolve()
        agent_yaml = harness_dir / "agent.yaml"
        if str(harness_dir) not in sys.path:
            sys.path.insert(0, str(harness_dir))
        meta_file = harness_dir / "meta.json"
        if meta_file.exists():
            try:
                import json as _json
                cand_top_k = (_json.loads(meta_file.read_text())
                              .get("effective", {}).get("sampling", {}).get("top_k"))
            except Exception:
                cand_top_k = None
    expected_middlewares = (
        _expected_generated_middlewares(agent_yaml) if harness_dir is not None else []
    )
    expected_components = (
        _expected_generated_components(agent_yaml) if harness_dir is not None
        else {"tools": {}, "skills": {}}
    )
    credit_skills = (
        _required_generated_skills(harness_dir, expected_components["skills"])
        if harness_dir is not None else []
    )
    active_skills = sorted(expected_components["skills"])
    active_playbooks = _mounted_skill_playbooks(
        agent_yaml, expected_components["skills"], active_skills
    )

    ledger = BudgetLedger(
        max_evaluator_calls=h2.max_evaluator_calls,
        max_probe_calls=h2.max_probe_calls,
    )
    component_contract = _runtime_component_contract(
        agent_yaml,
        evaluator_budget=ledger.max_evaluator_calls,
        probe_budget=ledger.max_probe_calls,
        generated_skill_playbooks=active_playbooks,
        current_candidate_skills=credit_skills,
    )
    session = InnerSession(task=task, ledger=ledger,
                           eval_timeout_s=h2.eval_timeout_s, python_exe=h2.python_exe,
                           checkpoint_path=checkpoint_path,
                           harness_dir=str(harness_dir) if harness_dir is not None else None)
    for name, source in expected_components["tools"].items():
        session.register_tool(name, source)
    for name, source in expected_components["skills"].items():
        session.register_skill(
            name, source, required_for_credit=name in credit_skills
        )
    for name in active_playbooks:
        session.record_skill_load(
            name,
            before_first_edit=True,
            delivery_mode="runtime_injection",
        )
    seed = (
        session.seed_baseline_from_outcome(seed_outcome)
        if seed_outcome is not None
        else session.seed_baseline()
    )
    seed_score = seed.combined_score

    config = AgentConfig.from_yaml(agent_yaml)
    _clamp_context(config)
    _override_llm(config, endpoint,
                  preserve_sampling=harness_dir is not None,
                  top_k_override=cand_top_k)
    if h2.max_iterations is not None:
        config.max_iterations = h2.max_iterations

    stop_reason, err, trajectory = "completed", None, None
    agent = None
    with session_scope(session):
        try:
            agent = Agent(config=config)
            agent.run(message=_initial_message(
                task,
                seed_score,
                seed.validity,
                h2.max_evaluator_calls,
                generated_skill_playbooks=active_playbooks,
                component_contract=component_contract,
            ))
        except Exception as e:  # a harness failure must not lose the seed baseline
            stop_reason, err = "harness_error", f"{type(e).__name__}: {e}"
        finally:
            # Preserve partial histories too.  Executor failures are often the
            # most useful case-study evidence, so an exception must not discard
            # the assistant/tool transcript accumulated before it occurred.
            if agent is not None:
                _finalize_llm_accounting(ledger, agent)
                if keep_trajectory:
                    trajectory = _extract_trajectory(agent)

    _record_skill_loads(session, trajectory)
    middleware_issues = session.middleware_participation_issues(
        expected_middlewares
    )
    skill_issues = session.skill_participation_issues(
        active_skills
    )
    participation_issues = [
        *(f"middleware {issue}" for issue in middleware_issues),
        *(f"skill {issue}" for issue in skill_issues),
    ]
    # A harness exception may leave the evaluator's seed checkpoint in the
    # session.  Keep that diagnostic value, but never publish it as a scored
    # executor trajectory: no completed executor action produced it.
    score_eligible = _is_score_eligible(stop_reason, participation_issues)
    if participation_issues:
        if skill_issues:
            participation_error = (
                "ComponentParticipationError: "
                + "; ".join(participation_issues)
            )
        else:
            participation_error = (
                "MiddlewareParticipationError: "
                + "; ".join(middleware_issues)
            )
        session.history_note(participation_error)
        if trajectory is not None:
            trajectory.append({
                "role": "framework",
                "content": participation_error,
            })
        err = f"{err}; {participation_error}" if err else participation_error
        if stop_reason == "completed":
            stop_reason = (
                "component_participation_failed"
                if skill_issues
                else "middleware_participation_failed"
            )

    s = session.summary()
    return InnerResult(
        task_id=task.task_id, source=task.source,
        best_score=session.best_score if session.best_score != float("-inf") else seed_score,
        seed_score=seed_score, best_metrics=session.best_metrics,
        best_program=session.best_program or task.initial_program,
        stop_reason=stop_reason, ledger=s["ledger"], steps=s["steps"],
        middleware_audit=s["middleware_audit"], tool_audit=s["tool_audit"],
        skill_audit=s["skill_audit"], decode_seed=endpoint.seed,
        score_eligible=score_eligible,
        trajectory=trajectory, error=err,
    )
