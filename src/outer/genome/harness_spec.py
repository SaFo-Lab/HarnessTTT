"""Typed semantic genome used to validate and canonicalize H2 packages.

H1 now edits a private materialized H2 filesystem.  After validation, that
directory is parsed into this representation for hashing, provenance, static
gates, and deterministic recompilation.  The genome includes proposer-owned
executor prompts, skills, generated tool/middleware code, mounts, sampling, and
iteration settings.  Core runtime bindings and the evaluation budget remain
externally fixed.

The legacy partial-YAML interface remains supported internally. Missing fields
inherit the base; generated component removal is explicit. A candidate
identical to its base is invalid.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SCHEMA_VERSION = "h2spec/1.0"
# 0.1 specs are still accepted (declarative-only surface); 1.0 adds new_tools[]
# with generated implementation code, gated + reviewed at propose time.
_ACCEPTED_SCHEMAS = {"h2spec/0.1", "h2spec/1.0"}

# generated-tool structural limits (code SAFETY is enforced by outer.static_gates
# + the reviewer self-test, NOT here — this only checks the spec shape)
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
_RESERVED_TOOL_NAMES = {"edit_solution", "evaluate_solution", "probe_solution",
                        "finish"}
RUNTIME_TOOL_PATH_ARG = "_sah_py_path"
_MAX_NEW_TOOLS = 3

# field -> (type, max_chars) for text fields.
# system_prompt MUST match propose_session._PROMPT_CAP: the write-time guard
# and this validation cap are two views of the SAME genome budget — a
# mismatch (write 10000 / validate 8000) burned round003 of cp10div: every
# session edited past 8000, validation refused, and 12 iterations died in
# the shrink loop as no_submission.
_TEXT_FIELDS = {
    "system_prompt": 10000,
    "skill_description": 600,
    "skill_body": 8000,
}
_TOOL_DESC_FIELDS = {  # tool name -> max chars
    "edit_solution": 1600,
    "evaluate_solution": 1000,
    "probe_solution": 1000,
    "finish": 600,
}
# numeric field -> (min, max, is_int)
_SAMPLING_FIELDS = {
    "temperature": (0.0, 1.5, False),
    "top_p": (0.05, 1.0, False),
    "top_k": (1, 100, True),
    "max_tokens": (1024, 16384, True),
}
_AGENT_FIELDS = {
    "max_iterations": (8, 80, True),
}
_MIDDLEWARE_FIELDS = {
    "budget_reminder_from_left": (0, 10, True),
    "long_tool_output_max_chars": (2000, 20000, True),
    "stall_after": (2, 30, True),
    "max_restarts": (0, 5, True),
}
_TOP_KEYS = {"schema", "system_prompt", "skill_description", "skill_body",
             "tool_descriptions", "sampling", "agent", "middleware",
             "new_tools", "remove_tools", "new_skills", "new_middlewares",
             "remove_generated"}

_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{2,39}$")
_MW_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
# Generated middleware currently injects framework messages into the model
# context.  Other NexAU hook types have different input/output contracts and
# are rejected until they receive their own audited adapters.
_MW_HOOKS = {"before_model"}
_MAX_NEW_SKILLS = 2
_MAX_NEW_MIDDLEWARES = 2
GENERATED_COMPONENT_FIELDS = ("new_tools", "new_skills", "new_middlewares")
_REMOVE_GENERATED_KEYS = {
    "tools": "new_tools",
    "skills": "new_skills",
    "middlewares": "new_middlewares",
}
_CORE_H2_TOOLS = ("edit_solution", "evaluate_solution", "probe_solution", "finish")
_CORE_H2_SKILLS = ("discovery-optimization",)
_CORE_H2_MIDDLEWARES = (
    "BudgetReminderMiddleware", "StallRestartMiddleware",
    "LongToolOutputMiddleware", "RoundAndTokenReminderMiddleware",
    "LLMUsageLedgerMiddleware",
)

# Component declarations in prompt.md are executable-interface claims, not
# ordinary prose. Keep the syntax intentionally narrow so normal mentions of
# "tool" or "skill" do not become false positives, while headings such as
# ``# Generated Tool: structure_analyzer`` cannot advertise an unmounted
# component to H2.
_COMPONENT_DECLARATION_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+"
    r"(?P<generated>generated\s+)?"
    r"(?P<kind>tool|skill|middleware)\s*:\s*"
    r"`?(?P<name>[A-Za-z][A-Za-z0-9_-]{1,63})`?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_GENERATED_SECTION_RE = re.compile(
    r"^generated\s+(tools?|skills?|middlewares?)\b", re.IGNORECASE
)
_COMPONENT_BULLET_RE = re.compile(
    r"^\s*[-*]\s+`(?P<name>[a-z][a-z0-9_-]{1,63})`"
)
_GENERATED_SUBHEADING_RE = re.compile(
    r"^`?(?P<name>[a-z][a-z0-9_-]{1,63})`?\b.*\bgenerated\b",
    re.IGNORECASE,
)


def _generated_section_declarations(prompt: str) -> List[Tuple[str, str]]:
    """Return component names explicitly listed in generated-* sections."""

    singular = {
        "tool": "tool", "tools": "tool",
        "skill": "skill", "skills": "skill",
        "middleware": "middleware", "middlewares": "middleware",
    }
    declarations: List[Tuple[str, str]] = []
    active: Optional[Tuple[str, int]] = None
    component_level: Optional[int] = None
    for line in prompt.splitlines():
        heading = _MARKDOWN_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if active and level <= active[1]:
                active = None
                component_level = None
            section = _GENERATED_SECTION_RE.match(title)
            if section:
                active = (singular[section.group(1).lower()], level)
                component_level = None
                continue
            if active and level > active[1]:
                declared = _GENERATED_SUBHEADING_RE.match(title)
                if declared:
                    declarations.append((active[0], declared.group("name")))
                    component_level = level
                elif component_level is not None and level <= component_level:
                    component_level = None
            continue
        if active and component_level is None:
            bullet = _COMPONENT_BULLET_RE.match(line)
            if bullet:
                declarations.append((active[0], bullet.group("name")))
    return declarations


def generated_component_prompt_sections(
    prompt: str, name: str,
) -> List[Tuple[str, int, int]]:
    """Return exact canonical Markdown sections for one generated component."""

    matches = list(re.finditer(
        rf"(?mi)^\s{{0,3}}(?P<marks>#{{1,6}})\s+Generated\s+"
        rf"(?:Tool|Skill|Middleware):\s+`?{re.escape(str(name))}`?\s*$",
        str(prompt),
    ))
    sections: List[Tuple[str, int, int]] = []
    for match in matches:
        level = len(match.group("marks"))
        end = len(prompt)
        for heading in re.finditer(
            r"(?m)^\s{0,3}(?P<marks>#{1,6})\s+.+$",
            prompt[match.end():],
        ):
            if len(heading.group("marks")) <= level:
                end = match.end() + heading.start()
                break
        sections.append((prompt[match.start():end].strip(), match.start(), end))
    return sections


def normalized_prompt(prompt: str) -> str:
    """Normalize only immaterial edge/trailing whitespace for comparisons."""

    return "\n".join(
        line.rstrip() for line in str(prompt).strip().splitlines()
    )


def isolated_generated_component_declaration(
    *, candidate_prompt: str, parent_prompt: str, name: str,
) -> Optional[str]:
    """Return the declaration iff only its canonical section can differ."""

    sections = generated_component_prompt_sections(candidate_prompt, name)
    if len(sections) != 1:
        return None
    text, start, end = sections[0]
    candidate_without = candidate_prompt[:start] + candidate_prompt[end:]
    parent_sections = generated_component_prompt_sections(parent_prompt, name)
    if len(parent_sections) > 1:
        return None
    if parent_sections:
        _parent_text, parent_start, parent_end = parent_sections[0]
        parent_without = (
            parent_prompt[:parent_start] + parent_prompt[parent_end:]
        )
    else:
        parent_without = parent_prompt
    if normalized_prompt(candidate_without) != normalized_prompt(parent_without):
        return None
    return text


@dataclass
class SpecValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)
    spec: Optional[Dict[str, Any]] = None  # canonical (validated, defaults NOT folded in)


def _check_num(errors: List[str], group: str, key: str, val: Any,
               lo: float, hi: float, is_int: bool) -> Any:
    if is_int:
        if isinstance(val, bool) or not isinstance(val, int):
            errors.append(f"{group}.{key}: expected int, got {type(val).__name__}")
            return None
    elif not isinstance(val, (int, float)) or isinstance(val, bool):
        errors.append(f"{group}.{key}: expected number, got {type(val).__name__}")
        return None
    if not (lo <= val <= hi):
        errors.append(f"{group}.{key}: {val} outside [{lo}, {hi}]")
        return None
    return int(val) if is_int else float(val)


def parse_and_validate(text: str) -> SpecValidation:
    """Parse a raw YAML spec (optionally inside ```yaml fences) fail-closed."""
    # Only unwrap a fence when it encloses the *entire* external payload.
    # ``system_prompt`` and generated skill bodies may legitimately contain
    # Markdown code fences; searching for the first fence would parse that
    # inner snippet as the H2 spec and report a false ``got str`` failure.
    m = re.fullmatch(
        r"\s*```(?:yaml|yml)?\s*\n(?P<body>.*?)\n```\s*",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    raw = m.group("body") if m else text
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        return SpecValidation(False, [f"yaml parse error: {str(e)[:200]}"])
    return validate_mapping(data)


def validate_mapping(data: Any) -> SpecValidation:
    """Validate an already parsed H2 mapping without serializing it again.

    Trusted workspace reconstruction produces a Python mapping.  Keeping that
    path typed prevents model-authored Markdown inside string values from ever
    re-entering the external text parser.
    """
    if not isinstance(data, dict):
        return SpecValidation(False, [f"spec must be a mapping, got {type(data).__name__}"])

    errors: List[str] = []
    unknown = set(data) - _TOP_KEYS
    if unknown:
        errors.append(f"unknown top-level keys (fail closed): {sorted(unknown)}")

    out: Dict[str, Any] = {"schema": SCHEMA_VERSION}
    if "schema" in data and data["schema"] not in _ACCEPTED_SCHEMAS:
        errors.append(f"schema must be one of {sorted(_ACCEPTED_SCHEMAS)}, got {data['schema']!r}")

    for key, cap in _TEXT_FIELDS.items():
        if key in data:
            v = data[key]
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{key}: must be a non-empty string")
            elif len(v) > cap:
                errors.append(f"{key}: {len(v)} chars exceeds cap {cap}")
            else:
                out[key] = v.strip()

    if "tool_descriptions" in data:
        td = data["tool_descriptions"]
        if not isinstance(td, dict):
            errors.append("tool_descriptions: must be a mapping")
        else:
            bad = set(td) - set(_TOOL_DESC_FIELDS)
            if bad:
                errors.append(f"tool_descriptions: unknown tools {sorted(bad)}")
            good = {}
            for name, cap in _TOOL_DESC_FIELDS.items():
                if name in td:
                    v = td[name]
                    if not isinstance(v, str) or not v.strip():
                        errors.append(f"tool_descriptions.{name}: must be a non-empty string")
                    elif len(v) > cap:
                        errors.append(f"tool_descriptions.{name}: {len(v)} chars exceeds cap {cap}")
                    else:
                        good[name] = v.strip()
            if good:
                out["tool_descriptions"] = good

    for group, fields_def in (("sampling", _SAMPLING_FIELDS), ("agent", _AGENT_FIELDS),
                              ("middleware", _MIDDLEWARE_FIELDS)):
        if group in data:
            g = data[group]
            if not isinstance(g, dict):
                errors.append(f"{group}: must be a mapping")
                continue
            bad = set(g) - set(fields_def)
            if bad:
                errors.append(f"{group}: unknown keys {sorted(bad)}")
            good = {}
            for key, (lo, hi, is_int) in fields_def.items():
                if key in g:
                    v = _check_num(errors, group, key, g[key], lo, hi, is_int)
                    if v is not None:
                        good[key] = v
            if good:
                out[group] = good

    # --- generative surface (h2spec/1.0): new_tools[] + remove_tools[] ------
    if "remove_tools" in data:
        rt = data["remove_tools"]
        if not isinstance(rt, list) or not all(isinstance(x, str) for x in rt):
            errors.append("remove_tools: must be a list of tool names")
        else:
            # only optional built-ins may be removed; core edit/evaluate/finish stay
            removable = {"probe_solution"}
            bad = set(rt) - removable
            if bad:
                errors.append(f"remove_tools: not removable {sorted(bad)} "
                              f"(only {sorted(removable)})")
            else:
                out["remove_tools"] = sorted(set(rt))

    if "new_tools" in data:
        nt = data["new_tools"]
        if not isinstance(nt, list):
            errors.append("new_tools: must be a list")
        elif len(nt) > _MAX_NEW_TOOLS:
            errors.append(f"new_tools: {len(nt)} exceeds cap {_MAX_NEW_TOOLS}")
        else:
            seen, good_tools = set(), []
            for i, t in enumerate(nt):
                if not isinstance(t, dict):
                    errors.append(f"new_tools[{i}]: must be a mapping")
                    continue
                extra = set(t) - {"name", "description", "input_schema",
                                  "implementation_py"}
                if extra:
                    errors.append(f"new_tools[{i}]: unknown keys {sorted(extra)}")
                name = t.get("name")
                if not isinstance(name, str) or not _TOOL_NAME_RE.match(name or ""):
                    errors.append(f"new_tools[{i}].name: must match [a-z][a-z0-9_]{{2,31}}")
                elif name in _RESERVED_TOOL_NAMES:
                    errors.append(f"new_tools[{i}].name: {name!r} is reserved")
                elif name in seen:
                    errors.append(f"new_tools[{i}].name: duplicate {name!r}")
                else:
                    seen.add(name)
                desc = t.get("description")
                if not isinstance(desc, str) or not desc.strip() or len(desc) > 800:
                    errors.append(f"new_tools[{i}].description: non-empty string <=800 chars")
                code = t.get("implementation_py")
                if not isinstance(code, str) or "def run" not in code:
                    errors.append(f"new_tools[{i}].implementation_py: must be code defining run(ctx, args)")
                sch = t.get("input_schema", {"type": "object", "properties": {}})
                if not isinstance(sch, dict):
                    errors.append(f"new_tools[{i}].input_schema: must be a JSON-schema mapping")
                elif sch.get("type") != "object" \
                        or not isinstance(sch.get("properties", {}), dict):
                    errors.append(
                        f"new_tools[{i}].input_schema: must describe an object "
                        "with a properties mapping"
                    )
                elif RUNTIME_TOOL_PATH_ARG in sch.get("properties", {}):
                    errors.append(
                        f"new_tools[{i}].input_schema: property "
                        f"{RUNTIME_TOOL_PATH_ARG!r} is reserved for the trusted runtime"
                    )
                if name and name in seen and isinstance(code, str) and isinstance(desc, str):
                    good_tools.append({"name": name, "description": desc.strip(),
                                       "input_schema": sch,
                                       "implementation_py": code})
            if good_tools:
                out["new_tools"] = good_tools

    # --- new_skills[]: extra skill playbooks (pure text, no code risk) -------
    if "new_skills" in data:
        ns = data["new_skills"]
        if not isinstance(ns, list):
            errors.append("new_skills: must be a list")
        elif len(ns) > _MAX_NEW_SKILLS:
            errors.append(f"new_skills: {len(ns)} exceeds cap {_MAX_NEW_SKILLS}")
        else:
            seen, good = set(), []
            for i, s in enumerate(ns):
                if not isinstance(s, dict):
                    errors.append(f"new_skills[{i}]: must be a mapping"); continue
                extra = set(s) - {"name", "description", "body"}
                if extra:
                    errors.append(f"new_skills[{i}]: unknown keys {sorted(extra)}")
                name = s.get("name")
                if not isinstance(name, str) or not _SKILL_NAME_RE.match(name or ""):
                    errors.append(f"new_skills[{i}].name: must match [a-z][a-z0-9-]{{2,39}}")
                elif name in seen or name == "discovery-optimization":
                    errors.append(f"new_skills[{i}].name: duplicate/reserved {name!r}")
                else:
                    seen.add(name)
                desc = s.get("description", "")
                body = s.get("body")
                if not isinstance(body, str) or not body.strip() or len(body) > 8000:
                    errors.append(f"new_skills[{i}].body: non-empty string <=8000 chars")
                if name in seen and isinstance(body, str) and body.strip():
                    good.append({"name": name, "description": str(desc).strip()[:600],
                                 "body": body.strip()})
            if good:
                out["new_skills"] = good

    # --- new_middlewares[]: generated hooks (code — same safety chain) --------
    if "new_middlewares" in data:
        nm = data["new_middlewares"]
        if not isinstance(nm, list):
            errors.append("new_middlewares: must be a list")
        elif len(nm) > _MAX_NEW_MIDDLEWARES:
            errors.append(f"new_middlewares: {len(nm)} exceeds cap {_MAX_NEW_MIDDLEWARES}")
        else:
            seen, good = set(), []
            for i, mw in enumerate(nm):
                if not isinstance(mw, dict):
                    errors.append(f"new_middlewares[{i}]: must be a mapping"); continue
                extra = set(mw) - {"name", "description", "hook", "implementation_py"}
                if extra:
                    errors.append(f"new_middlewares[{i}]: unknown keys {sorted(extra)}")
                name = mw.get("name")
                if not isinstance(name, str) or not _MW_NAME_RE.match(name or ""):
                    errors.append(f"new_middlewares[{i}].name: must match [a-z][a-z0-9_]{{2,31}}")
                elif name in seen:
                    errors.append(f"new_middlewares[{i}].name: duplicate {name!r}")
                else:
                    seen.add(name)
                hook = mw.get("hook")
                if hook not in _MW_HOOKS:
                    errors.append(f"new_middlewares[{i}].hook: must be one of {sorted(_MW_HOOKS)}")
                code = mw.get("implementation_py")
                if not isinstance(code, str) or "def " not in code:
                    errors.append(f"new_middlewares[{i}].implementation_py: must be a hook function body")
                desc = mw.get("description", "")
                if name in seen and isinstance(code, str) and hook in _MW_HOOKS:
                    good.append({"name": name, "hook": hook,
                                 "description": str(desc).strip()[:600],
                                 "implementation_py": code})
            if good:
                out["new_middlewares"] = good

    # Explicit deletion is required because omission means inheritance.  The
    # file-native H1 workflow derives this mapping when a component is removed
    # from agent.yaml and its files are deleted.
    if "remove_generated" in data:
        rg = data["remove_generated"]
        if not isinstance(rg, dict):
            errors.append("remove_generated: must be a mapping")
        else:
            extra = set(rg) - set(_REMOVE_GENERATED_KEYS)
            if extra:
                errors.append(
                    f"remove_generated: unknown component groups {sorted(extra)}"
                )
            good_removed: Dict[str, List[str]] = {}
            for group, field_name in _REMOVE_GENERATED_KEYS.items():
                if group not in rg:
                    continue
                names = rg[group]
                if not isinstance(names, list) or not all(
                    isinstance(name, str) for name in names
                ):
                    errors.append(
                        f"remove_generated.{group}: must be a list of names"
                    )
                    continue
                pattern = _SKILL_NAME_RE if field_name == "new_skills" else _TOOL_NAME_RE
                bad = [name for name in names if not pattern.match(name)]
                if bad:
                    errors.append(
                        f"remove_generated.{group}: invalid names {sorted(set(bad))}"
                    )
                    continue
                if names:
                    good_removed[group] = sorted(set(names))
            if good_removed:
                out["remove_generated"] = good_removed

    mutated = set(out) - {"schema"}
    if not mutated:
        errors.append("spec mutates nothing (all fields missing/invalid)")

    if errors:
        return SpecValidation(False, errors)
    return SpecValidation(True, [], out)


def canonical_json(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def spec_hash(spec: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:16]


def _rows_by_name(rows: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["name"]): row
        for row in (rows or [])
        if isinstance(row, dict) and row.get("name")
    }


def merge_named_components(
    inherited: Any, proposed: Any,
) -> List[Dict[str, Any]]:
    """Return a stable, name-keyed union of generated components.

    Components are part of the ratcheted harness genome.  A proposal with a
    new name appends one component; a proposal reusing an inherited name
    explicitly updates that component in place.  Omitting a name never removes
    it.  Deep copies keep proposal review/repair from mutating the base spec.
    """

    out = json.loads(json.dumps(list(inherited or [])))
    positions = {
        str(row.get("name")): i
        for i, row in enumerate(out)
        if isinstance(row, dict) and row.get("name")
    }
    for row in proposed or []:
        copied = json.loads(json.dumps(row))
        name = str(copied["name"])
        if name in positions:
            out[positions[name]] = copied
        else:
            positions[name] = len(out)
            out.append(copied)
    return out


def changed_generated_components(
    effective: Dict[str, Any], base: Dict[str, Any], field: str,
) -> List[Dict[str, Any]]:
    """Generated components added or explicitly updated by this candidate."""

    if field not in GENERATED_COMPONENT_FIELDS:
        raise ValueError(f"not a generated component field: {field}")
    base_by_name = _rows_by_name(base.get(field))
    return [
        row for row in effective.get(field, [])
        if base_by_name.get(str(row.get("name"))) != row
    ]


def generated_component_inventory(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Compact ordered inventory used by package provenance and audits."""

    return {
        field: [str(row["name"]) for row in spec.get(field, [])]
        for field in GENERATED_COMPONENT_FIELDS
    }


def added_generated_component_names(
    base: Dict[str, Any], effective: Dict[str, Any], field: str,
) -> List[str]:
    """Names newly introduced by a candidate, excluding in-place updates."""

    if field not in GENERATED_COMPONENT_FIELDS:
        raise ValueError(f"not a generated component field: {field}")
    base_names = set(generated_component_inventory(base)[field])
    return [
        name for name in generated_component_inventory(effective)[field]
        if name not in base_names
    ]


def h2_component_catalog(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Every component name the proposer-owned H2 prompt must expose."""

    removed = set(spec.get("remove_tools", []))
    tools = [name for name in _CORE_H2_TOOLS if name not in removed]
    tools += generated_component_inventory(spec)["new_tools"]
    return {
        "tools": tools,
        "skills": list(_CORE_H2_SKILLS)
                  + generated_component_inventory(spec)["new_skills"],
        "middlewares": generated_component_inventory(spec)["new_middlewares"]
                       + list(_CORE_H2_MIDDLEWARES),
    }


def component_prompt_issues(spec: Dict[str, Any]) -> List[str]:
    """Find missing or falsely advertised executor components in the prompt.

    Presence checks make mounted components discoverable. Declaration checks
    enforce the reverse direction for explicit component headings: an H1 may
    not describe a fictional ``Generated Tool`` in prose without mounting it.
    """

    prompt = str(spec.get("system_prompt", ""))
    issues: List[str] = []
    for group, names in h2_component_catalog(spec).items():
        missing = []
        for name in names:
            pattern = r"(?<![A-Za-z0-9_-])" + re.escape(name) \
                      + r"(?![A-Za-z0-9_-])"
            if re.search(pattern, prompt) is None:
                missing.append(name)
        if missing:
            issues.append(
                f"system_prompt does not name current {group}: {missing}"
            )

    catalog = h2_component_catalog(spec)
    generated = generated_component_inventory(spec)
    kind_to_group = {
        "tool": "tools",
        "skill": "skills",
        "middleware": "middlewares",
    }
    kind_to_field = {
        "tool": "new_tools",
        "skill": "new_skills",
        "middleware": "new_middlewares",
    }
    declarations: set[Tuple[str, str, bool]] = set()
    for match in _COMPONENT_DECLARATION_RE.finditer(prompt):
        kind = match.group("kind").lower()
        name = match.group("name")
        is_generated = bool(match.group("generated"))
        declarations.add((kind, name, is_generated))
    declarations.update(
        (kind, name, True)
        for kind, name in _generated_section_declarations(prompt)
    )
    for kind, name, is_generated in sorted(declarations):
        allowed = (
            generated[kind_to_field[kind]]
            if is_generated else catalog[kind_to_group[kind]]
        )
        if name not in allowed:
            qualifier = "generated " if is_generated else ""
            issues.append(
                f"system_prompt declares unmounted {qualifier}{kind} {name!r}"
            )
    return issues


def diff_to_partial(
    effective: Dict[str, Any], base: Dict[str, Any],
) -> Dict[str, Any]:
    """Encode a full effective H2 as a minimal, round-local mutation.

    This is also the bridge from the file-native proposer workspace back to
    h2spec provenance.  Unlike the legacy omission-only representation, it
    records component deletions explicitly.
    """

    out: Dict[str, Any] = {"schema": SCHEMA_VERSION}
    for key in _TEXT_FIELDS:
        if effective.get(key, "").strip() != base.get(key, "").strip():
            out[key] = effective.get(key, "")

    changed_descs = {}
    for name in _TOOL_DESC_FIELDS:
        value = effective.get("tool_descriptions", {}).get(name, "")
        if value.strip() != base.get("tool_descriptions", {}).get(name, "").strip():
            changed_descs[name] = value
    if changed_descs:
        out["tool_descriptions"] = changed_descs

    for group in ("sampling", "agent", "middleware"):
        values = {
            key: value
            for key, value in effective.get(group, {}).items()
            if value != base.get(group, {}).get(key)
        }
        if values:
            out[group] = values

    for field in GENERATED_COMPONENT_FIELDS:
        changed = changed_generated_components(effective, base, field)
        if changed:
            out[field] = json.loads(json.dumps(changed))

    removed: Dict[str, List[str]] = {}
    reverse_groups = {field: group for group, field in _REMOVE_GENERATED_KEYS.items()}
    for field in GENERATED_COMPONENT_FIELDS:
        effective_names = set(_rows_by_name(effective.get(field)))
        names = [
            name for name in _rows_by_name(base.get(field))
            if name not in effective_names
        ]
        if names:
            removed[reverse_groups[field]] = names
    if removed:
        out["remove_generated"] = removed

    if effective.get("remove_tools", []) != base.get("remove_tools", []):
        out["remove_tools"] = list(effective.get("remove_tools", []))
    return out


def generated_component_lineage(
    base: Dict[str, Any], effective: Dict[str, Any],
) -> Dict[str, List[Dict[str, str]]]:
    """Label every materialized component as inherited, added, or updated."""

    lineage: Dict[str, List[Dict[str, str]]] = {}
    for field in GENERATED_COMPONENT_FIELDS:
        base_by_name = _rows_by_name(base.get(field))
        rows = []
        effective_names = set()
        for row in effective.get(field, []):
            name = str(row["name"])
            effective_names.add(name)
            if name not in base_by_name:
                status = "added"
            elif base_by_name[name] == row:
                status = "inherited"
            else:
                status = "updated"
            rows.append({"name": name, "status": status})
        for row in base.get(field, []):
            name = str(row["name"])
            if name not in effective_names:
                rows.append({"name": name, "status": "removed"})
        lineage[field] = rows
    return lineage


# --------------------------------------------------------------------------- #
# Base-spec extraction: read the current best H2 package into spec form so the
# proposer sees the genome it is mutating and diffs are well-defined.
# --------------------------------------------------------------------------- #
def read_base_spec(
    package_dir: Path, *, verify_provenance: bool = True,
) -> Dict[str, Any]:
    """Extract the mutable surface of an existing H2 package as a full spec."""
    package_dir = Path(package_dir)
    agent = yaml.safe_load((package_dir / "agent.yaml").read_text())

    sys_file = package_dir / str(agent.get("system_prompt", "./system.md")).lstrip("./")
    if not sys_file.exists():  # candidate packages use prompt.md
        for alt in ("system.md", "prompt.md"):
            if (package_dir / alt).exists():
                sys_file = package_dir / alt
                break
    skill_dirs = [package_dir / str(s).lstrip("./") for s in agent.get("skills", [])]
    skill_md = next(
        (d / "SKILL.md" for d in skill_dirs
         if d.name == "discovery-optimization" and (d / "SKILL.md").exists()),
        None,
    )

    skill_desc, skill_body = "", ""
    if skill_md is not None:
        text = skill_md.read_text()
        fm = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if fm:
            meta = yaml.safe_load(fm.group(1)) or {}
            skill_desc = str(meta.get("description", ""))
            skill_body = fm.group(2).strip()
        else:
            skill_body = text.strip()

    tool_descs = {}
    # Read core schemas whether mounted or not.  In particular, removing the
    # optional probe tool from agent.yaml must not accidentally erase its
    # inherited description from the H2 genome.
    for name in _TOOL_DESC_FIELDS:
        ty = package_dir / "tools" / f"{name}.tool.yaml"
        if ty.exists():
            tdoc = yaml.safe_load(ty.read_text())
            if tdoc.get("name") in _TOOL_DESC_FIELDS:
                tool_descs[tdoc["name"]] = str(tdoc.get("description", "")).strip()

    llm = agent.get("llm_config", {}) or {}
    # Current NexAU YAML uses direct unknown keys, which LLMConfig stores in
    # ``llm.extra_params``.  Retain read compatibility with packages emitted by
    # the older, incorrectly nested representation so their genome can still be
    # inspected and normalized on the next materialization.
    extra_body = (
        llm.get("extra_body")
        or ((llm.get("extra_params") or {}).get("extra_body"))
        or {}
    )
    mw_params: Dict[str, int] = {}
    for mw in agent.get("middlewares", []):
        imp, params = str(mw.get("import", "")), mw.get("params", {}) or {}
        if "budget_reminder" in imp:
            mw_params["budget_reminder_from_left"] = int(params.get("remind_from_left", 3))
        if "long_tool_output" in imp:
            mw_params["long_tool_output_max_chars"] = int(params.get("max_output_chars", 8000))
        if "stall_restart" in imp:
            mw_params["stall_after"] = int(params.get("stall_after", 8))
            mw_params["max_restarts"] = int(params.get("max_restarts", 2))

    mounted_tool_names = {
        str(row.get("name")) for row in agent.get("tools", [])
        if isinstance(row, dict)
    }
    removed_optional = [
        name for name in sorted({"probe_solution"})
        if name not in mounted_tool_names
    ]

    return {
        "schema": SCHEMA_VERSION,
        "system_prompt": sys_file.read_text().strip(),
        "skill_description": skill_desc.strip(),
        "skill_body": skill_body,
        "tool_descriptions": tool_descs,
        "sampling": {
            "temperature": float(llm.get("temperature", 0.7)),
            "top_p": float(llm.get("top_p", 0.95)),
            "top_k": int(extra_body.get("top_k", 20)),
            "max_tokens": int(llm.get("max_tokens", 8192)),
        },
        "agent": {"max_iterations": int(agent.get("max_iterations", 36))},
        "middleware": mw_params or {"budget_reminder_from_left": 3,
                                    "long_tool_output_max_chars": 8000,
                                    "stall_after": 8,
                                    "max_restarts": 2},
        **({"remove_tools": removed_optional} if removed_optional else {}),
        **_read_generated(
            package_dir, agent, verify_provenance=verify_provenance
        ),
    }


def _read_generated(
    package_dir: Path, agent: Dict[str, Any], *, verify_provenance: bool = True,
) -> Dict[str, Any]:
    """Recover generated tools/skills/middlewares from a materialized package so
    the harness ratchet carries them forward — without this, every candidate's
    invented tools/skills/hooks vanish and M_phi must reinvent them each round.
    Reconstructs the h2spec/1.0 new_* fields from the components ACTUALLY
    mounted by ``agent.yaml``.  Declared-but-missing components fail closed;
    silently omitting one would corrupt the cross-round harness lineage."""
    out: Dict[str, Any] = {}
    reserved = {"discovery-optimization"}

    meta_effective: Dict[str, Any] = {}
    meta_path = package_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if isinstance(meta.get("effective"), dict):
                meta_effective = meta["effective"]
        except Exception as exc:
            raise ValueError(f"invalid package meta.json: {exc}") from exc
    meta_rows = {
        field: _rows_by_name(meta_effective.get(field))
        for field in GENERATED_COMPONENT_FIELDS
    }

    def package_path(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else package_dir / str(value).lstrip("./")

    # Generated tools: agent.yaml is authoritative.  Do not inherit stale .py
    # files that happen to remain in custom_tools/ but are not mounted.
    tools = []
    for tool in agent.get("tools", []):
        if "custom_runtime" not in str(tool.get("binding", "")):
            continue
        name = str(tool.get("name", ""))
        if not name:
            raise ValueError("mounted generated tool has no name")
        schema_path = package_path(tool.get("yaml_path", ""))
        if not schema_path.is_file():
            raise ValueError(f"mounted generated tool {name!r} is missing {schema_path}")
        doc = yaml.safe_load(schema_path.read_text()) or {}
        local_code = package_dir / "custom_tools" / f"{name}.py"
        extra_kwargs = tool.get("extra_kwargs") or {}
        declared_code = package_path(
            extra_kwargs.get(RUNTIME_TOOL_PATH_ARG)
            or extra_kwargs.get("py_path", "")  # legacy package support
        )
        code_path = local_code if local_code.is_file() else declared_code
        if not code_path.is_file():
            raise ValueError(f"mounted generated tool {name!r} has no implementation")
        prior = meta_rows["new_tools"].get(name, {})
        input_schema = doc.get("input_schema") or prior.get("input_schema") \
            or {"type": "object", "properties": {}}
        # The compiler adds a const-valued internal dispatcher argument to the
        # runtime schema. It is not part of the proposer genome and must not be
        # inherited or exposed back to H1 as an editable input.
        if isinstance(input_schema, dict):
            input_schema = json.loads(json.dumps(input_schema))
            properties = input_schema.get("properties")
            if isinstance(properties, dict):
                properties.pop(RUNTIME_TOOL_PATH_ARG, None)
            required = input_schema.get("required")
            if isinstance(required, list):
                input_schema["required"] = [
                    key for key in required if key != RUNTIME_TOOL_PATH_ARG
                ]
        tools.append({
            "name": name,
            "description": str(doc.get("description") or prior.get("description")
                               or f"generated tool {name}").strip(),
            "input_schema": input_schema,
            "implementation_py": code_path.read_text(),
        })
    if tools:
        out["new_tools"] = tools

    # Generated skills: inherit only skill directories listed in agent.yaml.
    skills = []
    for binding in agent.get("skills", []):
        skill_dir = package_path(binding)
        name = skill_dir.name
        if name in reserved:
            continue
        skill_md = skill_dir if skill_dir.name == "SKILL.md" else skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise ValueError(f"mounted generated skill {name!r} is missing {skill_md}")
        text = skill_md.read_text()
        fm = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        prior = meta_rows["new_skills"].get(name, {})
        if fm:
            frontmatter = yaml.safe_load(fm.group(1)) or {}
            if frontmatter.get("name") not in (None, name):
                raise ValueError(
                    f"generated skill frontmatter name mismatch for {name!r}"
                )
            body = fm.group(2).strip()
            desc = str(frontmatter.get("description") or prior.get("description") or "")
        else:
            body = text.strip()
            desc = str(prior.get("description") or "")
        skills.append({"name": name, "description": desc.strip(), "body": body})
    if skills:
        out["new_skills"] = skills

    # generated middlewares: middlewares/<name>.py bound as GeneratedMiddleware
    gen_mw = [m for m in agent.get("middlewares", [])
              if str(m.get("import", "")).endswith(":GeneratedMiddleware")]
    mws = []
    for m in gen_mw:
        module = str(m["import"]).split(":", 1)[0]
        name = module.rsplit(".", 1)[-1]
        py = package_dir / "middlewares" / f"{name}.py"
        if not py.is_file():
            raise ValueError(f"mounted generated middleware {name!r} is missing {py}")
        src = py.read_text()
        # recover the ORIGINAL user hook body (between sentinels), not the
        # nexau-importing wrapper — re-gating the wrapper would fail the import
        # whitelist and drop the inherited middleware.
        seg = re.search(r"# --USER-HOOK-START--\n(.*?)\n# --USER-HOOK-END--", src, re.DOTALL)
        prior = meta_rows["new_middlewares"].get(name, {})
        descriptor_path = package_dir / "middlewares" / f"{name}.middleware.yaml"
        descriptor: Dict[str, Any] = {}
        if descriptor_path.is_file():
            descriptor = yaml.safe_load(descriptor_path.read_text()) or {}
            if descriptor.get("name") not in (None, name):
                raise ValueError(
                    f"middleware descriptor name mismatch for {name!r}"
                )
        if seg:
            user_code = seg.group(1)
        elif isinstance(prior.get("implementation_py"), str):
            user_code = prior["implementation_py"]
        elif re.search(
            r"def\s+before_model\s*\(\s*hook_input", src
        ):
            # A newly authored draft may contain the raw user hook.  The final
            # materializer wraps it in GeneratedMiddleware after validation.
            user_code = src
        else:
            raise ValueError(
                f"cannot recover user hook body for generated middleware {name!r}"
            )
        hook_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*hook_input", user_code)
        hook = str(descriptor.get("hook") or prior.get("hook")
                   or (hook_match.group(1) if hook_match else ""))
        if hook not in _MW_HOOKS:
            raise ValueError(f"generated middleware {name!r} uses unsupported hook {hook!r}")
        mws.append({"name": name, "hook": hook,
                    "description": str(descriptor.get("description")
                                       or prior.get("description")
                                       or f"generated middleware {name}"),
                    "implementation_py": user_code})
    if mws:
        out["new_middlewares"] = mws

    # If provenance says a component was effective but agent.yaml does not
    # mount it, stop here.  Continuing would silently sever the ratchet.
    actual = generated_component_inventory(out)
    if verify_provenance:
        for field in GENERATED_COMPONENT_FIELDS:
            recorded = list(meta_rows[field])
            if recorded and recorded != actual[field]:
                raise ValueError(
                    f"package component mismatch for {field}: "
                    f"meta={recorded}, mounted={actual[field]}"
                )
    manifest_path = package_dir / "component_manifest.json"
    if verify_provenance and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        recorded_inventory = manifest.get("inventory") or {}
        if recorded_inventory != actual:
            raise ValueError(
                "component_manifest.json does not match mounted package: "
                f"manifest={recorded_inventory}, mounted={actual}"
            )
    return out


def merge_with_base(spec: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a (validated, partial) spec over the base spec -> full effective spec."""
    eff = json.loads(json.dumps(base))  # deep copy
    for key in _TEXT_FIELDS:
        if key in spec:
            eff[key] = spec[key]
    if "tool_descriptions" in spec:
        eff.setdefault("tool_descriptions", {}).update(spec["tool_descriptions"])
    for group in ("sampling", "agent", "middleware"):
        if group in spec:
            eff.setdefault(group, {}).update(spec[group])
    # Generated components are a ratcheted, name-keyed genome.  New names are
    # appended; reusing a name explicitly updates it; omission inherits it.
    for field in GENERATED_COMPONENT_FIELDS:
        if field in spec:
            merged = merge_named_components(base.get(field), spec[field])
            if merged:
                eff[field] = merged
    for group, field in _REMOVE_GENERATED_KEYS.items():
        removed = set((spec.get("remove_generated") or {}).get(group, []))
        if not removed:
            continue
        kept = [
            row for row in eff.get(field, [])
            if str(row.get("name")) not in removed
        ]
        if kept:
            eff[field] = kept
        else:
            eff.pop(field, None)
    if "remove_tools" in spec:
        eff["remove_tools"] = spec["remove_tools"]
    eff["schema"] = SCHEMA_VERSION
    return eff


def differs_from_base(effective: Dict[str, Any], base: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Which fields does the effective spec actually change vs the base?"""
    changed: List[str] = []
    for key in _TEXT_FIELDS:
        if effective.get(key, "").strip() != base.get(key, "").strip():
            changed.append(key)
    for name in _TOOL_DESC_FIELDS:
        if (effective.get("tool_descriptions", {}).get(name, "").strip()
                != base.get("tool_descriptions", {}).get(name, "").strip()):
            changed.append(f"tool_descriptions.{name}")
    for group in ("sampling", "agent", "middleware"):
        for key, val in effective.get(group, {}).items():
            if val != base.get(group, {}).get(key):
                changed.append(f"{group}.{key}")
    for field in GENERATED_COMPONENT_FIELDS:
        for row in changed_generated_components(effective, base, field):
            changed.append(f"{field}.{row.get('name', '?')}")
        effective_names = set(_rows_by_name(effective.get(field)))
        for name in _rows_by_name(base.get(field)):
            if name not in effective_names:
                changed.append(f"{field}.{name}.removed")
    if effective.get("remove_tools", []) != base.get("remove_tools", []):
        changed.append("remove_tools")
    return (len(changed) > 0, changed)
