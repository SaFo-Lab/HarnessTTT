"""Materialize an effective HarnessSpec into a full candidate H2 package.

Layout (matches the scaffold in src/inner/harness_candidate/candNN/):

    <cand_dir>/
    ├── agent.yaml                  # NexAU agent config (spec sampling/iteration/middleware params)
    ├── prompt.md                   # system prompt (spec.system_prompt)
    ├── spec.yaml                   # the raw partial spec M_phi generated (provenance)
    ├── meta.json                   # round/k/hash/changed-fields/raw generation pointer
    ├── tools/                      # tool schemas (descriptions from spec); bindings = shared executor code
    ├── skills/discovery-optimization/SKILL.md
    └── middlewares/                # per-candidate copy of the middleware code, imported as
        └── budget_reminder.py      #   middlewares.budget_reminder (cand_dir goes on sys.path)

Deterministic compilation: same effective spec -> byte-identical package
(modulo meta.json provenance fields).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from outer.genome.harness_spec import (
    RUNTIME_TOOL_PATH_ARG,
    component_prompt_issues,
    generated_component_inventory,
)

INNER_HARNESS = Path(__file__).resolve().parents[2] / "inner" / "harness"

_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "edit_solution": {
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string",
                                    "description": "Either SEARCH/REPLACE diff block(s) to apply to the current program, or the full replacement body for the EVOLVE-BLOCK region."}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    "evaluate_solution": {
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "probe_solution": {
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "default_description": (
            "Cheaply score the CURRENT program on SUBSAMPLED data (~2000 rows). "
            "Fast; does NOT consume the evaluation budget; approximate and NOT "
            "comparable to full scores — use it to rank variants when full "
            "evaluation is slow, then confirm with evaluate_solution."),
    },
    "finish": {
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "minLength": 1,
                                       "description": "One line on the final approach and best score."}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}


_BUILTIN_OPTIONAL = {"probe_solution"}

# Generated-middleware wrapper: the model supplies just the hook function body.
# Runtime errors remain diagnostic-fail-open so we retain the rest of the
# executor trajectory, but the rollout-level participation audit is
# reward-fail-closed: a missing mount/invocation or any hook error makes the
# rollout score ineligible.
_MW_WRAPPER = '''"""Generated middleware (h2spec/1.0), lifecycle-audited."""
from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput, HookResult, Middleware)
from nexau.core.messages import Message, Role, TextBlock
from inner.harness.middleware.generated_context import GeneratedHookTracker
from inner.harness.tools.runtime import get_session

# --USER-HOOK-START--
{user_code}
# --USER-HOOK-END--

class GeneratedMiddleware(Middleware):
    def __init__(self):
        self._name = {name_repr}
        self._hook = {hook_repr}
        self._tracker = GeneratedHookTracker()
        self._reported_errors = set()
        self._registered = False
        # NexAU instantiates middlewares at AgentConfig.from_yaml, BEFORE the
        # runner opens session_scope, so registration here can only be
        # best-effort; _ensure_registered() retries at every hook entry.
        self._ensure_registered()

    def _ensure_registered(self):
        if self._registered:
            return
        try:
            get_session().register_middleware(self._name, self._hook)
            self._registered = True
        except Exception:
            pass

    def _audit(self, text):
        try:
            get_session().history_note("[middleware:" + {name_repr} + "] " + str(text)[:320])
        except Exception:
            pass

    def _record(self, event, iteration, error=None):
        try:
            get_session().record_middleware_event(
                self._name, event, iteration=iteration, error=error)
        except Exception:
            pass

    def {hook}(self, hook_input):
        self._ensure_registered()
        iteration = int(getattr(hook_input, "current_iteration", 0) or 0)
        self._record("invoked", iteration)
        try:
            context = self._tracker.snapshot(hook_input, get_session())
            result = {hook}(context)
        except Exception as exc:
            self._record("error", iteration, error=str(exc))
            key = (type(exc).__name__, str(exc)[:200])
            if key not in self._reported_errors:
                self._reported_errors.add(key)
                self._audit("ERROR " + key[0] + ": " + key[1])
            return HookResult.no_changes()
        if not result:
            return HookResult.no_changes()
        # The hook stays a pure function.  It may return either an advisory
        # note (str) or a dict {{"note": str?, "require_tools": [tool, ...]?,
        # "require_structure_change": bool?}}.
        # The trusted wrapper applies effects; malformed shapes are hook
        # errors, never silent no-ops.
        note = None
        require_tools = None
        require_structure_change = False
        if isinstance(result, dict):
            unknown = set(result) - {{
                "note", "require_tools", "require_structure_change"
            }}
            if unknown:
                self._record("error", iteration,
                             error="unknown hook-result keys: " + repr(sorted(unknown)))
                self._audit("ERROR unknown hook-result keys " + repr(sorted(unknown)))
                return HookResult.no_changes()
            note = result.get("note")
            require_tools = result.get("require_tools")
            require_structure_change = result.get(
                "require_structure_change", False)
            if not isinstance(require_structure_change, bool):
                self._record("error", iteration,
                             error="require_structure_change must be boolean")
                self._audit("ERROR require_structure_change must be boolean")
                return HookResult.no_changes()
            if require_structure_change and require_tools != ["edit_solution"]:
                self._record(
                    "error", iteration,
                    error=("require_structure_change=true requires exactly "
                           "require_tools=['edit_solution']"))
                self._audit("ERROR structure postcondition requires edit_solution")
                return HookResult.no_changes()
        else:
            note = result
        fired = False
        if require_tools:
            try:
                fired = bool(get_session().request_tool_gate(
                    self._name, require_tools,
                    require_structure_change_from=(
                        context.get("last_family")
                        if require_structure_change else None
                    )))
                if fired:
                    self._audit("ENFORCED iteration=" + str(iteration)
                                + " require_tools=" + repr(list(require_tools))
                                + " structure_change="
                                + repr(require_structure_change))
                    # Gate semantics are mechanical.  Never expose an
                    # H1-authored note that can contradict the already-active
                    # immediate gate; derive the executor message solely from
                    # the trusted require_tools tuple.
                    if require_structure_change:
                        note = (
                            "[tool-gate:" + self._name
                            + "] NEXT CALL: edit_solution. Its resulting "
                              "normalized Python AST must differ from family "
                            + str(context.get("last_family"))
                            + "; literal tweaks, identifier renames, pass, and "
                              "self-assignment do not satisfy this gate. No "
                              "intermediate action is allowed."
                        )
                    else:
                        note = (
                            "[tool-gate:" + self._name + "] NEXT CALL: "
                            + " or ".join(str(tool) for tool in require_tools)
                            + ". No intermediate action is allowed."
                        )
                else:
                    # First-wins runtime arbitration suppressed this request.
                    # Do not deliver prose claiming that the losing gate is
                    # active; retain the conflict only in trusted audit data.
                    self._audit("SUPPRESSED iteration=" + str(iteration)
                                + " require_tools=" + repr(list(require_tools)))
                    note = None
            except Exception as exc:
                self._record("error", iteration, error=str(exc))
                self._audit("ENFORCE_ERROR " + type(exc).__name__ + ": " + str(exc)[:200])
                return HookResult.no_changes()
        if not note and not fired:
            return HookResult.no_changes()
        try:
            if fired:
                self._record("fired", iteration)
            if note:
                msg = Message(role=Role.FRAMEWORK, content=[TextBlock(text=str(note)[:2000])])
                if not fired:
                    self._record("fired", iteration)
                self._audit("FIRED iteration=" + str(context.get("iteration", 0))
                            + " note=" + str(note)[:180])
                return HookResult.with_modifications(messages=[*hook_input.messages, msg])
            return HookResult.no_changes()
        except Exception as exc:
            self._record("error", iteration, error=str(exc))
            self._audit("EMIT_ERROR " + type(exc).__name__ + ": " + str(exc)[:200])
            return HookResult.no_changes()
'''


def render_generated_skill(name: str, description: str, body: str) -> str:
    """Render canonical, lossless generated-skill bytes."""

    frontmatter = yaml.safe_dump({
        "name": str(name),
        "description": str(description),
    }, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{frontmatter}\n---\n\n{str(body).strip()}\n"


def _build_skill_list(effective: Dict[str, Any], cand_dir: Path) -> list:
    """Materialize M_phi-generated skills into skills/<name>/SKILL.md."""
    out = []
    for sk in effective.get("new_skills", []):
        name = sk["name"]
        d = cand_dir / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        # NexAU requires YAML frontmatter (--- name/description ---) before the body.
        # Keep it byte-identical to the atomic H1 transaction; description
        # punctuation is semantic data and must not be silently rewritten.
        desc = (sk.get("description", "") or f"Task-specific playbook: {name}").strip()
        (d / "SKILL.md").write_text(
            render_generated_skill(name, desc, sk["body"])
        , encoding="utf-8")
        out.append(f"./skills/{name}")
    return out


def _build_custom_middlewares(effective: Dict[str, Any], cand_dir: Path) -> list:
    """Materialize generated middleware code (gated during H1 validation)
    into middlewares/<name>.py wrapped in a lifecycle-audited Middleware."""
    out = []
    for mw in effective.get("new_middlewares", []):
        name = mw["name"]
        (cand_dir / "middlewares").mkdir(parents=True, exist_ok=True)
        code = _MW_WRAPPER.format(user_code=mw["implementation_py"].strip(),
                                  hook=mw["hook"], name_repr=repr(name),
                                  hook_repr=repr(mw["hook"]))
        compile(code, str(cand_dir / "middlewares" / f"{name}.py"), "exec")
        (cand_dir / "middlewares" / f"{name}.py").write_text(code, encoding="utf-8")
        (cand_dir / "middlewares" / f"{name}.middleware.yaml").write_text(
            _yaml_str({
                "name": name,
                "hook": mw["hook"],
                "description": mw.get("description", ""),
            })
        )
        out.append({"import": f"middlewares.{name}:GeneratedMiddleware", "params": {}})
    return out


def _build_tool_list(effective: Dict[str, Any], cand_dir: Path) -> list:
    """Assemble agent.yaml tool entries: core built-ins (minus removed
    optionals) + generated custom tools (h2spec/1.0)."""
    removed = set(effective.get("remove_tools", []))
    builtins = [n for n in ("edit_solution", "evaluate_solution",
                            "probe_solution", "finish")
                if n not in (removed & _BUILTIN_OPTIONAL)]
    tools = [{"name": n, "yaml_path": f"./tools/{n}.tool.yaml",
              "binding": f"inner.harness.tools.discovery:{n}"} for n in builtins]
    for t in effective.get("new_tools", []):
        name = t["name"]
        (cand_dir / "custom_tools").mkdir(exist_ok=True)
        (cand_dir / "custom_tools" / f"{name}.py").write_text(t["implementation_py"], encoding="utf-8")
        src = f"./custom_tools/{name}.py"
        input_schema = json.loads(json.dumps(
            t.get("input_schema") or {"type": "object", "properties": {}}
        ))
        properties = input_schema.setdefault("properties", {})
        properties[RUNTIME_TOOL_PATH_ARG] = {
            "type": "string",
            "const": src,
            "description": "Trusted dispatcher binding; omit from tool calls.",
        }
        (cand_dir / "tools" / f"{name}.tool.yaml").write_text(_yaml_str({
            "type": "tool", "name": name,
            "description": t["description"],
            "input_schema": input_schema,
        }), encoding="utf-8")
        # Keep packages relocatable and prevent a stale absolute temp path from
        # selecting code outside the mounted H2 package at execution time.
        tools.append({"name": name, "yaml_path": f"./tools/{name}.tool.yaml",
                      "binding": "inner.harness.tools.custom_runtime:custom_tool",
                      "extra_kwargs": {RUNTIME_TOOL_PATH_ARG: src}})
    return tools


def _yaml_str(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def _agent_component_inventory(agent: Dict[str, Any]) -> Dict[str, list]:
    """Generated components actually mounted by a materialized agent config."""

    tools = [
        str(row.get("name")) for row in agent.get("tools", [])
        if "custom_runtime" in str(row.get("binding", ""))
    ]
    skills = [
        Path(str(binding)).name for binding in agent.get("skills", [])
        if Path(str(binding)).name != "discovery-optimization"
    ]
    middlewares = []
    for row in agent.get("middlewares", []):
        binding = str(row.get("import", ""))
        if binding.endswith(":GeneratedMiddleware"):
            middlewares.append(binding.split(":", 1)[0].rsplit(".", 1)[-1])
    return {
        "new_tools": tools,
        "new_skills": skills,
        "new_middlewares": middlewares,
    }


def materialize(effective: Dict[str, Any], cand_dir: Path, *,
                raw_spec_text: str = "", meta: Optional[Dict[str, Any]] = None,
                validate_prompt: bool = True) -> Path:
    """Write the full candidate package for ``effective`` (a FULL merged spec)."""
    cand_dir = Path(cand_dir)
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    (cand_dir / "tools").mkdir(parents=True)
    (cand_dir / "skills" / "discovery-optimization").mkdir(parents=True)
    (cand_dir / "middlewares").mkdir(parents=True)

    prompt_issues = component_prompt_issues(effective)
    if validate_prompt and prompt_issues:
        raise ValueError("invalid proposer-owned system_prompt: "
                         + "; ".join(prompt_issues))
    # The executor system prompt is H2 genome authored by the proposer.
    # Normal materialization verifies it but never appends, repairs, or
    # rewrites it.  ``validate_prompt=False`` exists only to seed H1's private
    # draft from a legacy parent; final candidates always use the strict path.
    (cand_dir / "prompt.md").write_text(
        effective["system_prompt"].rstrip() + "\n"
    , encoding="utf-8")

    # --- skill --- #
    skill = "---\nname: discovery-optimization\ndescription: " + \
        json.dumps(effective.get("skill_description", "").strip()) + \
        "\n---\n\n" + effective.get("skill_body", "").rstrip() + "\n"
    (cand_dir / "skills" / "discovery-optimization" / "SKILL.md").write_text(skill, encoding="utf-8")

    # --- tools (descriptions from spec; bindings = fixed executor contract) --- #
    tool_descs = effective.get("tool_descriptions", {})
    for name, schema in _TOOL_SCHEMAS.items():
        doc = {"type": "tool", "name": name,
               "description": tool_descs.get(name, "").strip()
                              or schema.get("default_description")
                              or f"The {name} tool.",
               "input_schema": schema["input_schema"]}
        (cand_dir / "tools" / f"{name}.tool.yaml").write_text(_yaml_str(doc), encoding="utf-8")

    # --- middleware code copy (imported per-process as middlewares.budget_reminder) --- #
    (cand_dir / "middlewares" / "__init__.py").write_text("# candidate middleware package\n", encoding="utf-8")
    src_mw = INNER_HARNESS / "middleware" / "budget_reminder.py"
    mw_text = src_mw.read_text(encoding="utf-8").replace(
        "from inner.harness.tools.runtime import get_session",
        "from inner.harness.tools.runtime import get_session  # shared session bridge (fixed runtime)",
    )
    (cand_dir / "middlewares" / "budget_reminder.py").write_text(mw_text, encoding="utf-8")
    sr_text = (INNER_HARNESS / "middleware" / "stall_restart.py").read_text(encoding="utf-8").replace(
        "from inner.harness.tools.runtime import get_session",
        "from inner.harness.tools.runtime import get_session  # shared session bridge (fixed runtime)",
    )
    (cand_dir / "middlewares" / "stall_restart.py").write_text(sr_text, encoding="utf-8")

    # --- agent.yaml --- #
    sampling = effective.get("sampling", {})
    agent_p = effective.get("agent", {})
    mw_p = effective.get("middleware", {})
    agent = {
        "type": "agent",
        "name": "inner_h2_candidate",
        "description": "Candidate H2 generated by the outer loop (M_phi + H1).",
        "max_context_tokens": 131072,
        "system_prompt": "./prompt.md",
        "system_prompt_type": "file",
        "tool_call_mode": "structured",
        # Longer solution trajectory (SAH_MIN_ITERS floor): the inner rollout is
        # capped by agent iterations, not the eval budget — measured llm_calls
        # p50=59 while only ~10 of 40 evals were used, so M0 runs out of turns
        # before it finishes refining. Raising the floor lets the edit ->
        # evaluate -> adjust-from-feedback loop run much longer (user insight:
        # a longer trajectory that keeps writing code against feedback solves
        # harder). floor=0 (default) keeps the proposer's chosen value.
        "max_iterations": max(int(agent_p.get("max_iterations", 36)),
                              int(__import__("os").environ.get("SAH_MIN_ITERS", "0") or "0")),
        "retry_attempts": 2,
        "retry_backoff_max_seconds": 30,
        "timeout": 600,
        "llm_config": {
            "model": "qwen3.5-9b",           # overridden at runtime by harness_runner
            "base_url": "http://127.0.0.1:8800/v1",
            "api_key": "EMPTY",
            "api_type": "openai_chat_completion",
            "max_tokens": int(sampling.get("max_tokens", 8192)),
            "temperature": float(sampling.get("temperature", 0.7)),
            "top_p": float(sampling.get("top_p", 0.95)),
            # Keep every proposer-owned sampling field visible in agent.yaml.
            # harness_runner also reads the canonical effective spec from
            # meta.json and applies the same value at execution time.
            # LLMConfig has no explicit ``extra_params`` constructor field;
            # unknown keys become ``llm.extra_params``.  Emit ``extra_body``
            # directly or the OpenAI client receives an invalid literal
            # ``extra_params=...`` keyword before the first H2 model call.
            "extra_body": {"top_k": int(sampling.get("top_k", 20))},
            "stream": False,
        },
        "sandbox_config": {"type": "local", "work_dir": "/tmp"},
        "tools": _build_tool_list(effective, cand_dir),
        "stop_tools": ["finish"],
        "skills": ["./skills/discovery-optimization"] + _build_skill_list(effective, cand_dir),
        "middlewares": _build_custom_middlewares(effective, cand_dir) + [
            {"import": "middlewares.budget_reminder:BudgetReminderMiddleware",
             "params": {"remind_from_left": int(mw_p.get("budget_reminder_from_left", 3))}},
            {"import": "middlewares.stall_restart:StallRestartMiddleware",
             "params": {"stall_after": int(mw_p.get("stall_after", 8)),
                        "max_restarts": int(mw_p.get("max_restarts", 2))}},
            {"import": "nexau.archs.main_sub.execution.middleware.long_tool_output:LongToolOutputMiddleware",
             "params": {"max_output_chars": int(mw_p.get("long_tool_output_max_chars", 8000)),
                        "head_lines": 40, "tail_lines": 20,
                        "head_chars": 4000, "tail_chars": 4000,
                        "bypass_tool_names": ["finish", "LoadSkill"]}},
            {"import": "nexau.archs.main_sub.execution.middleware.round_and_token_reminder:RoundAndTokenReminderMiddleware",
             "params": {"max_context_tokens": 131072,
                        "desired_max_tokens": int(sampling.get("max_tokens", 8192))}},
            {"import": "inner.runtime.token_ledger:LLMUsageLedgerMiddleware",
             "params": {}},
        ],
        "tracers": [{"import": "nexau.archs.tracer.adapters.in_memory:InMemoryTracer"}],
    }
    expected_inventory = generated_component_inventory(effective)
    mounted_inventory = _agent_component_inventory(agent)
    if mounted_inventory != expected_inventory:
        raise RuntimeError(
            "generated component materialization mismatch: "
            f"expected={expected_inventory}, mounted={mounted_inventory}"
        )
    (cand_dir / "agent.yaml").write_text(_yaml_str(agent), encoding="utf-8")

    # --- provenance --- #
    if raw_spec_text:
        (cand_dir / "spec.yaml").write_text(raw_spec_text.rstrip() + "\n", encoding="utf-8")
    meta_payload = dict(meta or {})
    meta_payload["component_inventory"] = expected_inventory
    (cand_dir / "component_manifest.json").write_text(json.dumps({
        "schema": "generated-components/1.0",
        "inventory": expected_inventory,
        "lineage": meta_payload.get("component_lineage", {}),
    }, indent=2))
    (cand_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    # note: the candidate's spec sampling top_k is applied at runtime via extra_body
    return cand_dir
