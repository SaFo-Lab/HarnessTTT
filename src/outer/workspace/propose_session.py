"""Per-run proposer session and candidate-isolated H2 filesystem.

Mirrors inner/session.py: the propose driver sets the active session in a
contextvar for the duration of one H1 agent run; the tool bindings and the
submit-reminder middleware resolve it via get_session(). Thread-safe: each
proposer thread runs in its own context.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import re
import shlex
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from outer.genome import harness_spec as hs
from outer.workspace import h2_workspace as h2ws


@dataclass
class ProposeSession:
    base_spec: Dict[str, Any]
    draft_dir: Optional[Path] = None
    slot_index: Optional[int] = None
    portfolio_round: int = 0
    task_id: Optional[str] = None
    validations: int = 0
    submitted: bool = False
    raw_submission: str = ""
    partial_spec: Optional[Dict[str, Any]] = None
    effective: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    file_events: List[Dict[str, str]] = field(default_factory=list)
    agent_yaml_inspected: bool = False
    inspected_files: Set[str] = field(default_factory=set)
    workspace_revision: int = 0
    validated_revision: Optional[int] = None
    component_audit: List[Dict[str, Any]] = field(default_factory=list)
    schema_guard_events: List[Dict[str, Any]] = field(default_factory=list)
    component_transactions: List[Dict[str, Any]] = field(default_factory=list)
    failed_component_fingerprints: Dict[str, str] = field(default_factory=dict)
    failed_component_attempts: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    h1_runtime_events: List[Dict[str, Any]] = field(default_factory=list)
    failed_validations: int = 0
    # NOVELTY-001: component signatures proven causally futile in previous
    # rounds; a submission repeating one is refused (twice) with redirect
    # guidance, then accepted with a flag so the loop can never deadlock.
    futile_signatures: Set[str] = field(default_factory=set)
    novelty_refusals: int = 0
    novelty_flag: Optional[str] = None
    _validated_check: Optional[Any] = None
    _active_component_fingerprint: Optional[str] = None
    _active_component_attempt: Optional[Dict[str, Any]] = None

    # The pre-2.10 forced-slot starter body. The current scaffold seeds
    # component_scaffold.TOOL_PLACEHOLDER instead (picked up via
    # placeholder_bodies below), but base packages inherited from older
    # campaigns still carry this one, so it stays recognised as scaffolding.
    _STARTER_BODY = (
        "def run(ctx, args):\n"
        "    best = ctx.best_score()\n"
        "    return {\"best_score\": best, \"budget_left\": ctx.budget_left()}\n"
    )

    def _unchanged_starters(self) -> Set[str]:
        """Names of generated tools still byte-identical to the placeholder.

        Identity is decided by CONTENT, not by name: a starter that the
        proposer genuinely rewrote is a real component even though it kept
        the seeded file name, and a freshly written tool that happens to be
        the placeholder is not.
        """
        if self.draft_dir is None:
            return set()
        d = Path(self.draft_dir) / "custom_tools"
        if not d.is_dir():
            return set()
        bodies = {self._STARTER_BODY}
        try:
            from outer.proposing.component_scaffold import placeholder_bodies
            bodies |= placeholder_bodies()
        except Exception:
            pass
        names = {f.stem for f in d.glob("*.py") if f.read_text() in bodies}
        mw = Path(self.draft_dir) / "middlewares"
        if mw.is_dir():
            for f in mw.glob("*.py"):
                text = f.read_text()
                if any(b.strip() in text for b in bodies if "before_model" in b):
                    names.add(f.stem)
        return names

    def generated_component_targets(self) -> List[str]:
        """Paths of generated-component files this session wrote or edited.

        The bail-out escalation (BAILOUT-001) uses this to name the exact
        files a stuck proposer should delete before falling back to a
        minimal valid change.
        """
        targets: List[str] = []
        for event in self.file_events:
            path = str(event.get("target", ""))
            if (path.startswith("custom_tools/")
                    or (path.startswith("tools/")
                        and path.endswith(".tool.yaml"))
                    or path.startswith("middlewares/")
                    or path.startswith("skills/")) \
                    and event.get("operation") in ("write", "edit") \
                    and path not in targets:
                targets.append(path)
        return targets

    def failed_component_targets(self) -> List[str]:
        """Stable kind/name identities for rolled-back atomic attempts."""

        return sorted(self.failed_component_attempts)

    def record_h1_runtime_event(
        self, event: Dict[str, Any], *, training_blocking: bool = False,
    ) -> None:
        """Record H1 infrastructure provenance outside model-authored text."""

        row = dict(event)
        row["training_blocking"] = bool(training_blocking)
        self.h1_runtime_events.append(row)

    def h1_training_block_reason(self) -> Optional[str]:
        """Return the first infrastructure reason that invalidates GRPO."""

        for event in self.h1_runtime_events:
            if event.get("training_blocking"):
                return str(event.get("kind") or "h1_infrastructure_failure")
        return None

    def mark_preloaded(self, relative_paths) -> None:
        """Mark only integrity records rendered with ``complete=true``."""
        for relative in relative_paths:
            self.inspected_files.add(str(relative))
            if str(relative) == "agent.yaml":
                self.agent_yaml_inspected = True

    def record_schema_guard(self, event: Dict[str, Any]) -> None:
        """Audit trail for the tool schema guard (TOOLSCHEMA-001)."""
        self.schema_guard_events.append(dict(event))

    def schema_guard_summary(self) -> Dict[str, Any]:
        """Compact per-slot summary for round.json attribution."""
        repaired = [e for e in self.schema_guard_events if e.get("action") == "repaired"]
        refused = [e for e in self.schema_guard_events if e.get("action") == "refused"]
        by_tool: Dict[str, int] = {}
        for event in refused:
            by_tool[event["tool"]] = by_tool.get(event["tool"], 0) + 1
        return {
            "repaired_calls": len(repaired),
            "refused_calls": len(refused),
            "escalated": any(e.get("escalated") for e in refused),
            "refusals_by_tool": by_tool,
        }

    def _contract_errors(self, effective: Dict[str, Any]) -> List[str]:
        """Contracts a submission must satisfy to be valid.

        General workspace validation owns prompt, skill, sampling, component,
        and composition safety. The cross-candidate contract only freezes the
        parent H2 compute ceiling so a portfolio cannot buy a higher score.
        """
        if self.slot_index is None:
            return []
        from outer.proposing.component_portfolio import contract_errors
        return contract_errors(
            self.base_spec, effective, self.slot_index, self.portfolio_round
        )

    def portfolio_role(self):
        """The role used by prompt, validator, reminder, and artifacts."""

        if self.slot_index is None:
            return None
        from outer.proposing.component_portfolio import (
            portfolio_enabled, role_for_slot,
        )
        if not portfolio_enabled():
            return None
        return role_for_slot(self.slot_index, self.portfolio_round)

    def validation_profile(self) -> Optional[Dict[str, Any]]:
        """Trusted task/role profile; never derived from H1-authored text."""

        role = self.portfolio_role()
        from outer.safety.task_profiles import trusted_component_profile
        return trusted_component_profile(
            self.task_id, getattr(role, "key", None),
        )

    def _drop_scaffold_fields(self, changed):
        """Seeded plumbing the proposer never wrote is not a changed field.

        The forced-tool slot starts with a mounted placeholder so the
        proposer does not have to fight agent.yaml.  That placeholder is a
        real spec difference, so the diff reports `new_tools.<name>` — and
        every downstream consumer (round.json, the component ratchet, the
        futile-signature detector, the artifacts) then reads it as a tool
        the proposer invented.  Across 21 campaigns that inflated the
        generated-tool count from 20 real tools to 168.  Credit the body,
        not the scaffold.
        """
        placeholders = self._unchanged_starters()
        if not placeholders:
            return changed
        return [f for f in changed
                if str(f).split(".", 1)[-1] not in placeholders]

    def _check(self, spec_yaml: str):
        v = hs.parse_and_validate(spec_yaml)
        if not v.valid:
            return None, None, [], v.errors
        eff = hs.merge_with_base(v.spec, self.base_spec)
        differs, changed = hs.differs_from_base(eff, self.base_spec)
        changed = self._drop_scaffold_fields(changed)
        if not differs:
            return v.spec, eff, [], ["spec is identical to the current harness (no-op)"]
        prompt_errors = hs.component_prompt_issues(eff)
        if prompt_errors:
            return v.spec, eff, changed, prompt_errors
        contract_errors = self._contract_errors(eff)
        if contract_errors:
            return v.spec, eff, changed, contract_errors
        return v.spec, eff, changed, []

    def validate(self, spec_yaml: str) -> str:
        self.validations += 1
        _, _, changed, errors = self._check(spec_yaml)
        if errors:
            return "INVALID:\n- " + "\n- ".join(errors)
        return ("VALID. This spec changes: " + ", ".join(changed) +
                ". You can refine further or submit_spec now.")

    def submit(self, spec_yaml: str) -> str:
        self.submitted = True
        self.raw_submission = spec_yaml
        spec, eff, changed, errors = self._check(spec_yaml)
        if errors:
            self.errors = errors
            return "SUBMITTED but INVALID (minimum reward):\n- " + "\n- ".join(errors)
        self.partial_spec, self.effective, self.changed_fields = spec, eff, changed
        return f"SUBMITTED. Candidate spec accepted; changes: {', '.join(changed)}."

    def _require_draft(self) -> Path:
        if self.draft_dir is None:
            raise RuntimeError("this proposer session has no H2 file workspace")
        return Path(self.draft_dir)

    def inspect_harness(self, command: str) -> str:
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []
        is_agent_cat = (
            len(argv) == 2
            and argv[0] == "cat"
            and Path(argv[1]).as_posix().lstrip("./") == "agent.yaml"
        )
        if not self.agent_yaml_inspected and not is_agent_cat:
            return (
                "ERROR: first inspect the actual mount graph with "
                '`harness_shell(command="cat agent.yaml")`.'
            )
        output = h2ws.inspect(self._require_draft(), command)
        if not output.startswith("ERROR:") and len(argv) == 2 and argv[0] == "cat":
            try:
                relative = h2ws.relative_path(
                    self._require_draft(), argv[1], must_exist=True
                )
                self.inspected_files.add(relative)
                if relative == "agent.yaml":
                    self.agent_yaml_inspected = True
            except ValueError:
                pass
        self.file_events.append({"operation": "inspect", "target": command})
        return output

    # PROMPTCAP-001: the genome caps system_prompt at 8000 chars, but the
    # validator only reports the breach AFTER the session has spent turns.
    # Refuse the write that would cross the cap, with the fix in the message
    # — a drift-prone proposer otherwise appends its way into an unfixable
    # state that repair budgets cannot trim back in time.
    _PROMPT_CAP = 30000

    def _prompt_cap_refusal(self, path: str, *, content: Optional[str] = None,
                            old_text: Optional[str] = None,
                            new_text: Optional[str] = None,
                            append_text: Optional[str] = None) -> Optional[str]:
        if path != "prompt.md" or self.draft_dir is None:
            return None
        target = Path(self.draft_dir) / "prompt.md"
        current = target.read_text() if target.is_file() else ""
        if content is not None:
            prospective = len(content)
        elif append_text is not None:
            prospective = len(current) + len(append_text) + 1
        elif old_text is not None and old_text in current:
            prospective = len(current) - len(old_text) + len(new_text or "")
        else:
            return None
        if prospective <= self._PROMPT_CAP:
            return None
        return (
            f"REFUSED: this edit would make prompt.md {prospective} chars — "
            f"over the {self._PROMPT_CAP}-char genome cap (current: "
            f"{len(current)}). Do NOT append more text. Instead REPLACE or "
            "TRIM an existing section with edit_harness_file(old_text=..., "
            "new_text=<shorter text>) so the total stays under the cap, then "
            "validate_harness."
        )

    def _require_read_before_existing_edit(self, path: str) -> Optional[str]:
        if not self.agent_yaml_inspected:
            return (
                "ERROR: first inspect the actual mount graph with "
                '`harness_shell(command="cat agent.yaml")`.'
            )
        try:
            relative = h2ws.relative_path(
                self._require_draft(), path, must_exist=False
            )
            exists = (self._require_draft() / relative).exists()
        except ValueError as exc:
            return f"ERROR: {exc}"
        if exists and relative not in self.inspected_files:
            return (
                f"ERROR: read existing file first with "
                f'`harness_shell(command="cat {relative}")`.'
            )
        return None

    def _record_mutation(self, output: str) -> None:
        if not output.startswith("ERROR:"):
            self.workspace_revision += 1
            self.validated_revision = None
            self._validated_check = None
            self.component_audit = []

    def write_harness_file(self, path: str, content: str) -> str:
        blocked = self._require_read_before_existing_edit(path)
        if blocked:
            return blocked
        blocked = self._prompt_cap_refusal(path, content=content)
        if blocked:
            return blocked
        output = h2ws.write_file(self._require_draft(), path, content)
        self._record_mutation(output)
        self.file_events.append({"operation": "write", "target": path,
                                 "result": output})
        return output

    def edit_harness_file(
        self,
        path: str,
        old_text: str = "",
        new_text: str = "",
        append_text: str = "",
    ) -> str:
        blocked = self._require_read_before_existing_edit(path)
        if blocked:
            return blocked
        blocked = self._prompt_cap_refusal(
            path, old_text=old_text or None, new_text=new_text,
            append_text=append_text or None,
        )
        if blocked:
            return blocked
        output = h2ws.edit_file(
            self._require_draft(), path, old_text, new_text, append_text
        )
        self._record_mutation(output)
        self.file_events.append({"operation": "edit", "target": path,
                                 "result": output})
        return output

    def delete_harness_file(self, path: str) -> str:
        blocked = self._require_read_before_existing_edit(path)
        if blocked:
            return blocked
        output = h2ws.delete_file(self._require_draft(), path)
        self._record_mutation(output)
        self.file_events.append({"operation": "delete", "target": path,
                                 "result": output})
        return output

    @staticmethod
    def _component_intent(
        *, failure_mode: str, hypothesis: str,
        expected_behavior_change: str, novelty_basis: str, risk: str,
    ) -> Dict[str, str]:
        intent = {
            "failure_mode": str(failure_mode).strip(),
            "hypothesis": str(hypothesis).strip(),
            "expected_behavior_change": str(expected_behavior_change).strip(),
            "novelty_basis": str(novelty_basis).strip(),
            "risk": str(risk).strip(),
        }
        missing = [key for key, value in intent.items() if not value]
        if missing:
            raise ValueError(
                "component intent fields must be non-empty: " + ", ".join(missing)
            )
        for key, value in intent.items():
            if len(value) > 800:
                raise ValueError(f"{key} exceeds 800 characters")
        return intent

    def _begin_component_attempt(
        self, *, kind: str, name: str, implementation: str,
        contract: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return the first witness for an identical failed component contract."""

        payload = json.dumps({
            "kind": str(kind),
            "name": str(name),
            "implementation": str(implementation),
            "contract": contract or {},
            "intent": intent or {},
        }, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        self._active_component_fingerprint = fingerprint
        self._active_component_attempt = {
            "kind": str(kind),
            "name": str(name),
            "implementation": str(implementation),
            "contract": json.loads(json.dumps(contract or {}, default=str)),
            "intent": json.loads(json.dumps(intent or {}, default=str)),
            "fingerprint": fingerprint,
        }
        return self.failed_component_fingerprints.get(fingerprint)

    @staticmethod
    def _transaction_error_records(errors: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Attach stable ownership to validator witnesses.

        H1 should learn from semantic/component mistakes, not from failures in
        trusted transaction plumbing.  Framework-owned failures therefore
        block training and do not ask the proposer to rewrite its component.
        """

        records: List[Dict[str, Any]] = []
        for error in errors or []:
            message = str(error)
            framework_owned = message.startswith(
                "transaction infrastructure error:"
            )
            records.append({
                "owner": "framework" if framework_owned else "proposer",
                "code": (
                    "component_transaction_infrastructure_error"
                    if framework_owned else "component_validation_error"
                ),
                "retryable_by_proposer": not framework_owned,
                "message": message,
            })
        return records

    @staticmethod
    def _transaction_result(
        *, ok: bool, kind: str, name: str, intent: Dict[str, str],
        errors: Optional[List[str]] = None,
        changed_fields: Optional[List[str]] = None,
        audit: Optional[List[Dict[str, Any]]] = None,
        normalizations: Optional[List[str]] = None,
    ) -> str:
        error_text = "\n".join(str(error) for error in (errors or []))
        repair_surface = "implementation"
        for field in (
            "failure_mode", "hypothesis", "expected_behavior_change",
            "novelty_basis", "risk",
        ):
            if f"{field} exceeds" in error_text:
                repair_surface = field
                break
        if "description must be" in error_text:
            repair_surface = "description"
        elif kind == "skill" and "body must be" in error_text:
            repair_surface = "body"
        elif "usage_guidance must be" in error_text:
            repair_surface = "usage_guidance"
        elif "input_schema" in error_text:
            repair_surface = "input_schema"
        optional_repairs_failed = any(marker in error_text for marker in (
            "repair surface", "repair_suggestions", "wall repair",
        ))
        error_records = ProposeSession._transaction_error_records(errors)
        if ok:
            next_action = (
                "continue the selected portfolio or stop; the runtime will "
                "auto-finalize this validated workspace if no submit call occurs"
            )
        elif any(row["owner"] == "framework" for row in error_records):
            next_action = (
                "do not rewrite the component for this witness; trusted "
                "framework plumbing failed and this trajectory is excluded "
                "from proposer training"
            )
        elif optional_repairs_failed:
            next_action = (
                "repair_suggestions are optional for this checker. Call "
                "repair_component(surface='implementation', "
                "replacement_text=<complete corrected implementation>) to "
                "remove them while preserving verified diagnostics"
            )
        else:
            replacement_arg = (
                "replacement_schema=<complete corrected object>"
                if repair_surface == "input_schema"
                else "replacement_text=<complete corrected surface>"
            )
            next_action = (
                "call repair_component with surface=" + repr(repair_surface)
                + " and " + replacement_arg + "; use a full upsert only "
                  "when the component's causal design changes materially"
            )
        # Keep the immediate response repairable under the 6k tool-output cap:
        # errors come first, intent is summarized, and audit history stays in
        # component_transactions artifacts instead of repeating the error.
        intent_summary = {
            key: (value if len(value) <= 180 else value[:177] + "...")
            for key, value in intent.items()
        }
        payload = {
            "ok": bool(ok),
            "errors": errors or [],
            "error_records": error_records,
            "next_action": next_action,
            "repair_surface": None if ok else repair_surface,
            "component": {"kind": kind, "name": name},
            "workspace_changed": bool(ok),
            "changed_fields": changed_fields or [],
            "trusted_normalizations": normalizations or [],
            "intent_summary": intent_summary,
            "audit_records": len(audit or []),
        }
        label = (
            "VALID COMPONENT TRANSACTION"
            if ok else "INVALID COMPONENT TRANSACTION (workspace unchanged)"
        )
        return label + "\n" + json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _upsert_prompt_declaration(
        root: Path, *, kind: str, name: str, usage_guidance: str,
        expected_behavior_change: str,
    ) -> None:
        prompt_path = root / "prompt.md"
        prompt = prompt_path.read_text()
        title = {
            "tool": "Tool", "skill": "Skill", "middleware": "Middleware",
        }.get(kind)
        if title is None:
            raise ValueError(f"unsupported generated component kind: {kind!r}")
        block = h2ws.generated_component_prompt_block(
            kind, name, usage_guidance=usage_guidance,
            expected_behavior_change=expected_behavior_change,
        )
        pattern = re.compile(
            rf"(?ms)^#{{1,6}}\s+Generated\s+{title}:\s+`?{re.escape(name)}`?"
            rf"[^\n]*\n.*?(?=^#{{1,6}}\s|\Z)"
        )
        if pattern.search(prompt):
            updated = pattern.sub(block + "\n\n", prompt, count=1)
        else:
            updated = prompt.rstrip() + "\n\n" + block + "\n"
        if len(updated) > ProposeSession._PROMPT_CAP:
            raise ValueError(
                f"canonical {kind} declaration would exceed prompt.md cap "
                f"({len(updated)} > {ProposeSession._PROMPT_CAP})"
            )
        prompt_path.write_text(updated.rstrip() + "\n")

    def _commit_component_transaction(
        self, *, transaction_root: Path, kind: str, name: str,
        intent: Dict[str, str], check: h2ws.WorkspaceCheck,
        normalizations: Optional[List[str]] = None,
    ) -> str:
        draft = self._require_draft()
        backup = draft.parent / f".{draft.name}.backup-{uuid.uuid4().hex}"
        draft.rename(backup)
        try:
            transaction_root.rename(draft)
        except Exception:
            backup.rename(draft)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        self.workspace_revision += 1
        self.validated_revision = self.workspace_revision
        self._validated_check = check
        self.component_audit = list(check.component_audit)
        event = {
            "operation": f"upsert_{kind}",
            "target": name,
            "result": "committed_and_validated",
            "intent": intent,
            "changed_fields": list(check.changed_fields),
            "trusted_normalizations": list(normalizations or []),
        }
        self.file_events.append(event)
        self.component_transactions.append(event)
        self.failed_component_attempts.pop(f"{kind}:{name}", None)
        self._active_component_fingerprint = None
        self._active_component_attempt = None
        return self._transaction_result(
            ok=True, kind=kind, name=name, intent=intent,
            changed_fields=check.changed_fields, audit=check.component_audit,
            normalizations=normalizations,
        )

    def _transaction_failure(
        self, *, kind: str, name: str, intent: Dict[str, str],
        errors: List[str], audit: Optional[List[Dict[str, Any]]] = None,
        normalizations: Optional[List[str]] = None,
    ) -> str:
        self.failed_validations += 1
        error_records = self._transaction_error_records(errors)
        event = {
            "operation": f"upsert_{kind}", "target": name,
            "result": "rolled_back", "intent": intent,
            "errors": list(errors),
            "error_records": error_records,
            "audit": list(audit or []),
            "trusted_normalizations": list(normalizations or []),
        }
        self.file_events.append(event)
        self.component_transactions.append(event)
        active_attempt = self._active_component_attempt
        if active_attempt is not None:
            retained = dict(active_attempt)
            if intent:
                retained["intent"] = dict(intent)
            retained["last_errors"] = list(errors)
            self.failed_component_attempts[f"{kind}:{name}"] = retained
        if any(row["owner"] == "framework" for row in error_records):
            self.record_h1_runtime_event({
                "kind": "component_transaction_infrastructure_error",
                "component_kind": kind,
                "component_name": name,
                "errors": error_records,
            }, training_blocking=True)
        if self._active_component_fingerprint:
            # Preserve the original actionable witness.  Repeated-identical
            # refusals must not recursively overwrite it with an ever-growing
            # "previous witness" wrapper that consumes the H1 context budget.
            self.failed_component_fingerprints.setdefault(
                self._active_component_fingerprint,
                "; ".join(str(error) for error in errors)[:1600],
            )
        self._active_component_fingerprint = None
        self._active_component_attempt = None
        return self._transaction_result(
            ok=False, kind=kind, name=name, intent=intent,
            errors=errors, audit=audit, normalizations=normalizations,
        )

    def upsert_skill(
        self, *, name: str, body: str, description: str,
        usage_guidance: str, failure_mode: str, hypothesis: str,
        expected_behavior_change: str, novelty_basis: str, risk: str,
    ) -> str:
        """Atomically author a skill while the framework owns all plumbing."""

        self.validations += 1
        raw_intent = {
            "failure_mode": str(failure_mode).strip(),
            "hypothesis": str(hypothesis).strip(),
            "expected_behavior_change": str(expected_behavior_change).strip(),
            "novelty_basis": str(novelty_basis).strip(),
            "risk": str(risk).strip(),
        }
        prior_failure = self._begin_component_attempt(
            kind="skill", name=str(name), implementation=str(body),
            contract={
                "description": description,
                "usage_guidance": usage_guidance,
            },
            intent=raw_intent,
        )
        if prior_failure is not None:
            return self._transaction_failure(
                kind="skill", name=str(name), intent={}, errors=[
                    "REPEATED IDENTICAL IMPLEMENTATION: no skill body or "
                    "contract changed since the previous failed transaction. "
                    f"Previous witness: {prior_failure}. Revise the named "
                    "semantic surface before retrying."
                ]
            )
        try:
            intent = self._component_intent(**raw_intent)
        except ValueError as exc:
            return self._transaction_failure(
                kind="skill", name=str(name), intent=raw_intent,
                errors=[str(exc)],
            )
        if self._active_component_attempt is not None:
            self._active_component_attempt["intent"] = dict(intent)

        errors: List[str] = []
        skill_name = str(name).strip()
        skill_body = str(body).strip()
        skill_description = str(description).strip()
        skill_usage = str(usage_guidance).strip()
        if (not hs._SKILL_NAME_RE.match(skill_name)
                or skill_name == "discovery-optimization"):
            errors.append(
                "name must match [a-z][a-z0-9-]{2,39} and not be a core skill"
            )
        if not skill_body or len(skill_body) > 8000:
            errors.append("body must be non-empty and <=8000 characters")
        if re.match(r"^---\s*$", skill_body, re.MULTILINE):
            errors.append(
                "body must omit YAML frontmatter; the framework writes the "
                "canonical name/description frontmatter"
            )
        if not skill_description or len(skill_description) > 600:
            errors.append("description must be non-empty and <=600 characters")
        if not skill_usage or len(skill_usage) > 800:
            errors.append("usage_guidance must be non-empty and <=800 characters")
        if errors:
            return self._transaction_failure(
                kind="skill", name=skill_name, intent=intent, errors=errors,
            )

        trusted_normalizations = [
            "skill_frontmatter_framework_owned",
            "skill_mount_framework_owned",
            "skill_prompt_declaration_framework_owned",
        ]
        draft = self._require_draft()
        transaction_root = draft.parent / f".{draft.name}.skill-{uuid.uuid4().hex}"
        try:
            shutil.copytree(draft, transaction_root)
            skill_dir = transaction_root / "skills" / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            from outer.compiling.materialize import render_generated_skill
            (skill_dir / "SKILL.md").write_text(
                render_generated_skill(
                    skill_name, skill_description, skill_body
                )
            )
            agent_path = transaction_root / "agent.yaml"
            agent = yaml.safe_load(agent_path.read_text()) or {}
            mounts = [
                str(binding) for binding in (agent.get("skills") or [])
                if Path(str(binding).rstrip("/")).name != skill_name
            ]
            mounts.append(f"./skills/{skill_name}")
            agent["skills"] = mounts
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
            self._upsert_prompt_declaration(
                transaction_root, kind="skill", name=skill_name,
                usage_guidance=skill_usage,
                expected_behavior_change=expected_behavior_change,
            )
            check = h2ws.validate_workspace(
                transaction_root, self.base_spec,
                validation_profile=self.validation_profile(),
            )
            contract_errors = (
                self._contract_errors(check.effective or {}) if check.valid else []
            )
            all_errors = [*check.errors, *contract_errors]
            if all_errors:
                return self._transaction_failure(
                    kind="skill", name=skill_name, intent=intent,
                    errors=all_errors, audit=check.component_audit,
                    normalizations=trusted_normalizations,
                )
            return self._commit_component_transaction(
                transaction_root=transaction_root, kind="skill",
                name=skill_name, intent=intent, check=check,
                normalizations=trusted_normalizations,
            )
        except Exception as exc:
            return self._transaction_failure(
                kind="skill", name=skill_name, intent=intent,
                errors=[
                    f"transaction infrastructure error: "
                    f"{type(exc).__name__}: {exc}"
                ],
                normalizations=trusted_normalizations,
            )
        finally:
            if transaction_root.exists():
                shutil.rmtree(transaction_root)

    def upsert_tool(
        self, *, name: str, implementation: str, description: str,
        input_schema: Dict[str, Any], usage_guidance: str,
        failure_mode: str, hypothesis: str,
        expected_behavior_change: str, novelty_basis: str, risk: str,
    ) -> str:
        """Atomically author code, schema, mount, declaration, and validation."""

        self.validations += 1
        raw_intent = {
            "failure_mode": str(failure_mode).strip(),
            "hypothesis": str(hypothesis).strip(),
            "expected_behavior_change": str(expected_behavior_change).strip(),
            "novelty_basis": str(novelty_basis).strip(),
            "risk": str(risk).strip(),
        }
        prior_failure = self._begin_component_attempt(
            kind="tool", name=str(name), implementation=str(implementation),
            contract={
                "description": description,
                "input_schema": input_schema,
                "usage_guidance": usage_guidance,
            },
            intent=raw_intent,
        )
        if prior_failure is not None:
            return self._transaction_failure(
                kind="tool", name=str(name), intent={}, errors=[
                    "REPEATED IDENTICAL IMPLEMENTATION: no code or contract "
                    "changed since the previous failed transaction. Previous "
                    f"witness: {prior_failure}. Modify the implementation in "
                    "response to that witness before retrying."
                ]
            )
        try:
            intent = self._component_intent(**raw_intent)
        except ValueError as exc:
            return self._transaction_failure(
                kind="tool", name=str(name), intent=raw_intent,
                errors=[str(exc)]
            )
        if self._active_component_attempt is not None:
            self._active_component_attempt["intent"] = dict(intent)
        errors: List[str] = []
        if not hs._TOOL_NAME_RE.match(str(name)) or name in hs._RESERVED_TOOL_NAMES:
            errors.append(
                "name must match [a-z][a-z0-9_]{2,31} and not be a core tool"
            )
        if not isinstance(description, str) or not description.strip() \
                or len(description) > 800:
            errors.append("description must be non-empty and <=800 characters")
        schema = json.loads(json.dumps(input_schema)) \
            if isinstance(input_schema, dict) else {}
        if schema.get("type") != "object" or not isinstance(
            schema.get("properties"), dict
        ):
            errors.append("input_schema must be an object schema with properties")
        if hs.RUNTIME_TOOL_PATH_ARG in (schema.get("properties") or {}):
            errors.append(f"input_schema property {hs.RUNTIME_TOOL_PATH_ARG!r} is reserved")
        required = schema.get("required", [])
        if not isinstance(required, list) or not set(required) <= set(
            (schema.get("properties") or {}).keys()
        ):
            errors.append("input_schema.required must name declared properties")
        schema["additionalProperties"] = False
        if errors:
            return self._transaction_failure(
                kind="tool", name=str(name), intent=intent, errors=errors
            )
        from outer.safety.task_profiles import normalize_tool_schema
        schema, trusted_normalizations = normalize_tool_schema(
            self.validation_profile(), schema
        )
        from outer.safety.reviewer.reviewer import review_tool_code
        outcome = review_tool_code(
            str(implementation), input_schema=schema,
            description=str(description), repair_fn=None,
            max_rounds=0, selftest_timeout=10.0, require_utility=True,
            validation_profile=self.validation_profile(),
            contract_text="\n".join((
                str(usage_guidance), str(expected_behavior_change),
            )),
        )
        if not outcome.ok:
            return self._transaction_failure(
                kind="tool", name=str(name), intent=intent,
                errors=[outcome.final_error or "tool preflight failed"],
                audit=[{"kind": "tool", "name": name, "ok": False,
                        "history": outcome.history}],
                normalizations=trusted_normalizations,
            )

        draft = self._require_draft()
        transaction_root = draft.parent / f".{draft.name}.tool-{uuid.uuid4().hex}"
        try:
            shutil.copytree(draft, transaction_root)
            (transaction_root / "custom_tools").mkdir(exist_ok=True)
            (transaction_root / "tools").mkdir(exist_ok=True)
            (transaction_root / "custom_tools" / f"{name}.py").write_text(
                str(implementation).rstrip() + "\n"
            )
            (transaction_root / "tools" / f"{name}.tool.yaml").write_text(
                yaml.safe_dump({
                    "type": "tool", "name": name,
                    "description": description.strip(),
                    "input_schema": schema,
                }, sort_keys=False, allow_unicode=True, width=100)
            )
            agent_path = transaction_root / "agent.yaml"
            agent = yaml.safe_load(agent_path.read_text()) or {}
            mounts = [
                row for row in (agent.get("tools") or [])
                if str(row.get("name")) != name
            ]
            mounts.append({
                "name": name,
                "yaml_path": f"./tools/{name}.tool.yaml",
                "binding": "inner.harness.tools.custom_runtime:custom_tool",
                "extra_kwargs": {
                    hs.RUNTIME_TOOL_PATH_ARG: f"./custom_tools/{name}.py"
                },
            })
            agent["tools"] = mounts
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
            self._upsert_prompt_declaration(
                transaction_root, kind="tool", name=name,
                usage_guidance=usage_guidance,
                expected_behavior_change=expected_behavior_change,
            )
            check = h2ws.validate_workspace(
                transaction_root, self.base_spec,
                validation_profile=self.validation_profile(),
            )
            contract_errors = (
                self._contract_errors(check.effective or {}) if check.valid else []
            )
            all_errors = [*check.errors, *contract_errors]
            if all_errors:
                return self._transaction_failure(
                    kind="tool", name=name, intent=intent, errors=all_errors,
                    audit=check.component_audit,
                    normalizations=trusted_normalizations,
                )
            return self._commit_component_transaction(
                transaction_root=transaction_root, kind="tool", name=name,
                intent=intent, check=check,
                normalizations=trusted_normalizations,
            )
        except Exception as exc:
            return self._transaction_failure(
                kind="tool", name=name, intent=intent,
                errors=[f"transaction infrastructure error: {type(exc).__name__}: {exc}"],
                normalizations=trusted_normalizations,
            )
        finally:
            if transaction_root.exists():
                shutil.rmtree(transaction_root)

    def upsert_middleware(
        self, *, name: str, implementation: str, description: str,
        usage_guidance: str, failure_mode: str, hypothesis: str,
        expected_behavior_change: str, novelty_basis: str, risk: str,
    ) -> str:
        """Atomically author a typed before_model middleware transaction."""

        self.validations += 1
        raw_intent = {
            "failure_mode": str(failure_mode).strip(),
            "hypothesis": str(hypothesis).strip(),
            "expected_behavior_change": str(expected_behavior_change).strip(),
            "novelty_basis": str(novelty_basis).strip(),
            "risk": str(risk).strip(),
        }
        prior_failure = self._begin_component_attempt(
            kind="middleware", name=str(name), implementation=str(implementation),
            contract={
                "description": description,
                "usage_guidance": usage_guidance,
            },
            intent=raw_intent,
        )
        if prior_failure is not None:
            return self._transaction_failure(
                kind="middleware", name=str(name), intent={}, errors=[
                    "REPEATED IDENTICAL IMPLEMENTATION: no hook code changed "
                    "since the previous failed transaction. Previous witness: "
                    f"{prior_failure}. Change the condition or returned effect "
                    "before retrying."
                ]
            )
        try:
            intent = self._component_intent(**raw_intent)
        except ValueError as exc:
            return self._transaction_failure(
                kind="middleware", name=str(name), intent=raw_intent,
                errors=[str(exc)]
            )
        if self._active_component_attempt is not None:
            self._active_component_attempt["intent"] = dict(intent)
        errors: List[str] = []
        if not hs._MW_NAME_RE.match(str(name)):
            errors.append("name must match [a-z][a-z0-9_]{2,31}")
        if not isinstance(description, str) or not description.strip() \
                or len(description) > 600:
            errors.append("description must be non-empty and <=600 characters")
        hook_src = h2ws._extract_hook_def(str(implementation), "before_model")
        if hook_src is None:
            errors.append(
                "implementation must contain def before_model(hook_input)"
            )
        if errors:
            return self._transaction_failure(
                kind="middleware", name=str(name), intent=intent, errors=errors
            )
        from outer.safety.static_gates import (
            check_middleware_code, check_middleware_utility,
        )
        ok_static, static_errors = check_middleware_code(
            hook_src, "before_model"
        )
        ok_utility, utility_errors = (False, ["skipped after static failure"])
        if ok_static:
            ok_utility, utility_errors = check_middleware_utility(
                hook_src, "before_model", description=str(description),
                validation_profile=self.validation_profile(),
                contract_text=h2ws.generated_component_prompt_block(
                    "middleware", str(name)
                ),
            )
        preflight_errors = [*static_errors, *(
            [] if ok_utility else [f"utility gate: {e}" for e in utility_errors]
        )]
        if preflight_errors:
            return self._transaction_failure(
                kind="middleware", name=name, intent=intent,
                errors=preflight_errors,
                audit=[{"kind": "middleware", "name": name, "ok": False,
                        "history": preflight_errors}],
            )

        draft = self._require_draft()
        transaction_root = draft.parent / f".{draft.name}.mw-{uuid.uuid4().hex}"
        try:
            shutil.copytree(draft, transaction_root)
            middleware_dir = transaction_root / "middlewares"
            middleware_dir.mkdir(exist_ok=True)
            (middleware_dir / f"{name}.py").write_text(
                h2ws._wrap_user_hook(name, "before_model", hook_src)
            )
            (middleware_dir / f"{name}.middleware.yaml").write_text(
                yaml.safe_dump({
                    "name": name, "hook": "before_model",
                    "description": description.strip(),
                }, sort_keys=False, allow_unicode=True, width=100)
            )
            agent_path = transaction_root / "agent.yaml"
            agent = yaml.safe_load(agent_path.read_text()) or {}
            binding = f"middlewares.{name}:GeneratedMiddleware"
            mounts = [
                row for row in (agent.get("middlewares") or [])
                if str(row.get("import")) != binding
            ]
            mounts.insert(0, {"import": binding, "params": {}})
            agent["middlewares"] = mounts
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
            self._upsert_prompt_declaration(
                transaction_root, kind="middleware", name=name,
                usage_guidance=usage_guidance,
                expected_behavior_change=expected_behavior_change,
            )
            check = h2ws.validate_workspace(
                transaction_root, self.base_spec,
                validation_profile=self.validation_profile(),
            )
            contract_errors = (
                self._contract_errors(check.effective or {}) if check.valid else []
            )
            all_errors = [*check.errors, *contract_errors]
            if all_errors:
                return self._transaction_failure(
                    kind="middleware", name=name, intent=intent,
                    errors=all_errors, audit=check.component_audit,
                )
            return self._commit_component_transaction(
                transaction_root=transaction_root, kind="middleware", name=name,
                intent=intent, check=check,
            )
        except Exception as exc:
            return self._transaction_failure(
                kind="middleware", name=name, intent=intent,
                errors=[f"transaction infrastructure error: {type(exc).__name__}: {exc}"],
            )
        finally:
            if transaction_root.exists():
                shutil.rmtree(transaction_root)

    def repair_component(
        self, *, kind: str, name: str, old_text: str = "",
        new_text: str = "", surface: str = "implementation",
        replacement_text: str = "",
        replacement_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Replace or patch one retained failed surface and rerun it.

        Rollback intentionally keeps invalid bytes out of H2.  This method
        keeps the complete failed transaction in trusted session memory, so a
        repair turn can provide the corrected semantic surface directly.  The
        old exact-fragment mode remains available for small code patches.
        """

        kind = str(kind).strip().lower()
        name = str(name).strip()
        surface = str(surface or "implementation").strip()
        key = f"{kind}:{name}"
        attempt = self.failed_component_attempts.get(key)
        if attempt is None:
            return (
                "INVALID COMPONENT REPAIR: no retained rolled-back attempt "
                f"for {key!r}; call the matching full upsert transaction"
            )
        text_surfaces = {
            "implementation",
            "body",
            "description",
            "usage_guidance",
            "failure_mode",
            "hypothesis",
            "expected_behavior_change",
            "novelty_basis",
            "risk",
        }
        if surface not in text_surfaces | {"input_schema"}:
            return (
                "INVALID COMPONENT REPAIR: surface must be implementation, body, "
                "description, usage_guidance, input_schema, failure_mode, hypothesis, "
                "expected_behavior_change, novelty_basis, or risk"
            )
        if surface == "input_schema" and kind != "tool":
            return (
                "INVALID COMPONENT REPAIR: input_schema exists only on tools"
            )
        if surface == "body" and kind != "skill":
            return "INVALID COMPONENT REPAIR: body exists only on skills"
        contract = dict(attempt.get("contract") or {})
        intent = dict(attempt.get("intent") or {})
        if surface in {"implementation", "body"}:
            current_value: Any = str(attempt.get("implementation") or "")
        elif surface in contract:
            current_value = contract.get(surface)
        elif surface in intent:
            current_value = intent.get(surface)
        else:
            return (
                "INVALID COMPONENT REPAIR: retained transaction has no "
                f"surface {surface!r}; call the matching full upsert"
            )
        direct_schema = replacement_schema is not None
        direct_text = bool(replacement_text)
        if direct_schema and direct_text:
            return (
                "INVALID COMPONENT REPAIR: provide only replacement_schema "
                "or replacement_text"
            )
        if direct_schema:
            if surface != "input_schema":
                return (
                    "INVALID COMPONENT REPAIR: replacement_schema requires "
                    "surface='input_schema'"
                )
            try:
                repaired_value: Any = json.loads(json.dumps(replacement_schema))
            except (TypeError, ValueError) as exc:
                return (
                    "INVALID COMPONENT REPAIR: replacement_schema must be "
                    f"JSON-serializable: {exc}"
                )
            repair_mode = "structured_replacement"
            prior_rendered = json.dumps(
                current_value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )
            repaired_rendered = json.dumps(
                repaired_value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )
        elif direct_text:
            if surface == "input_schema":
                return (
                    "INVALID COMPONENT REPAIR: input_schema requires a JSON "
                    "object in replacement_schema"
                )
            repaired_value = str(replacement_text)
            repair_mode = "structured_replacement"
            prior_rendered = str(current_value or "")
            repaired_rendered = repaired_value
        else:
            old_text = str(old_text)
            new_text = str(new_text)
            if not old_text:
                return (
                    "INVALID COMPONENT REPAIR: provide replacement_text, "
                    "replacement_schema, or a non-empty old_text patch"
                )
            if surface == "input_schema":
                rendered = json.dumps(
                    current_value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                )
            else:
                rendered = str(current_value or "")
            occurrences = rendered.count(old_text)
            if occurrences != 1:
                return (
                    "INVALID COMPONENT REPAIR: old_text must match exactly once "
                    f"in retained {surface}; matched {occurrences} times"
                )
            if old_text == new_text:
                return "INVALID COMPONENT REPAIR: replacement is a no-op"
            if len(new_text) > 6000:
                return (
                    "INVALID COMPONENT REPAIR: new_text exceeds the "
                    "6000-character exact-patch cap; use replacement_text"
                )
            repaired_text = rendered.replace(old_text, new_text, 1)
            if surface == "input_schema":
                try:
                    repaired_value = json.loads(repaired_text)
                except json.JSONDecodeError as exc:
                    return (
                        "INVALID COMPONENT REPAIR: repaired input_schema is "
                        f"not valid JSON: {exc}"
                    )
            else:
                repaired_value = repaired_text
            repair_mode = "exact_patch"
            prior_rendered = rendered
            repaired_rendered = repaired_text
        if surface == "input_schema" and not isinstance(repaired_value, dict):
            return (
                "INVALID COMPONENT REPAIR: repaired input_schema must be a "
                "JSON object"
            )
        if prior_rendered == repaired_rendered:
            return "INVALID COMPONENT REPAIR: replacement is a no-op"
        if len(repaired_rendered) > 40_000:
            return (
                "INVALID COMPONENT REPAIR: replacement exceeds the "
                "40000-character workspace surface cap"
            )
        implementation = str(attempt.get("implementation") or "")
        if surface in {"implementation", "body"}:
            implementation = str(repaired_value)
        elif surface in contract:
            contract[surface] = repaired_value
        else:
            intent[surface] = str(repaired_value)
        required_intent = {
            "failure_mode", "hypothesis", "expected_behavior_change",
            "novelty_basis", "risk",
        }
        if set(intent) != required_intent:
            return (
                "INVALID COMPONENT REPAIR: retained attempt has no complete "
                "causal-intent record; call the matching full upsert"
            )
        parent_fingerprint = str(attempt.get("fingerprint") or "")
        kwargs = {
            "name": name,
            "implementation": implementation,
            **contract,
            **intent,
        }
        if kind == "tool":
            result = self.upsert_tool(**kwargs)
        elif kind == "middleware":
            result = self.upsert_middleware(**kwargs)
        elif kind == "skill":
            kwargs["body"] = kwargs.pop("implementation")
            result = self.upsert_skill(**kwargs)
        else:
            return (
                "INVALID COMPONENT REPAIR: kind must be skill, tool, or middleware"
            )
        self.component_transactions.append({
            "operation": "repair_component",
            "target": key,
            "surface": surface,
            "mode": repair_mode,
            "result": (
                "committed_and_validated"
                if str(result).startswith("VALID COMPONENT TRANSACTION")
                else "rolled_back"
            ),
            "parent_attempt_fingerprint": parent_fingerprint,
            "prior_value_sha256": hashlib.sha256(
                prior_rendered.encode()
            ).hexdigest(),
            "replacement_value_sha256": hashlib.sha256(
                repaired_rendered.encode()
            ).hexdigest(),
            "prior_value_chars": len(prior_rendered),
            "replacement_value_chars": len(repaired_rendered),
        })
        return result

    def _check_workspace(self) -> h2ws.WorkspaceCheck:
        return h2ws.validate_workspace(
            self._require_draft(), self.base_spec,
            validation_profile=self.validation_profile(),
        )

    def validate_harness(self) -> str:
        self.validations += 1
        if not self.agent_yaml_inspected:
            return (
                "INVALID H2 WORKSPACE:\n- agent.yaml was not inspected; start "
                'with harness_shell(command="cat agent.yaml")'
            )
        if "prompt.md" not in self.inspected_files:
            return (
                "INVALID H2 WORKSPACE:\n- prompt.md was not inspected; read the "
                "executor system prompt before validating H2"
            )
        check = self._check_workspace()
        self.component_audit = list(check.component_audit)
        if not check.valid:
            self.failed_validations += 1
            return "INVALID H2 WORKSPACE:\n- " + "\n- ".join(check.errors)
        contract_errors = self._contract_errors(check.effective or {})
        if contract_errors:
            self.failed_validations += 1
            return "INVALID H2 WORKSPACE:\n- " + "\n- ".join(contract_errors)
        self.validated_revision = self.workspace_revision
        self._validated_check = check
        return (
            "VALID H2 WORKSPACE. Changes: "
            + ", ".join(check.changed_fields)
            + ". You may refine files further or call submit_harness."
        )

    def submit_harness(self) -> str:
        self.submitted = True
        # a previous refusal left self.errors set, and nothing cleared it on
        # success — so the standard weak-model sequence (submit -> "validate
        # first" -> validate -> submit ACCEPTED) still read as invalid: the
        # candidate was discarded and a repair session was launched over a
        # perfectly good draft.
        self.errors = []
        if self.validated_revision != self.workspace_revision:
            self.errors = [
                "submit_harness requires a successful validate_harness after "
                "the most recent file edit"
            ]
            return "SUBMITTED but INVALID (minimum reward):\n- " + self.errors[0]
        # Reuse the exact validated object: submit_harness is atomic and never
        # invokes a reviewer or rewrites component bytes after H1 stops.
        check = self._validated_check
        if check is None:
            self.errors = ["validated workspace result is missing"]
            return "SUBMITTED but INVALID (minimum reward):\n- " + self.errors[0]
        if not check.valid:
            self.errors = check.errors
            return "SUBMITTED but INVALID (minimum reward):\n- " + "\n- ".join(
                check.errors
            )
        # A pre-seeded starter the proposer never touched is scaffolding, not
        # a component, so it is excluded from the signature — but only while
        # its body is still the placeholder (v2.6: a rewritten starter counts).
        placeholders = self._unchanged_starters()
        sig = ";".join(sorted(
            str(f) for f in check.changed_fields
            if not str(f).startswith("system_prompt")
            and str(f).split(".", 1)[-1] not in placeholders
        ))
        # subset match: a component proven futile is still a repeat when the
        # candidate bundles something else alongside it
        sig_set = set(sig.split(";")) if sig else set()
        repeat = next(
            (f for f in self.futile_signatures
             if f and set(f.split(";")) <= sig_set), None)
        if repeat:
            sig = repeat
            if self.novelty_refusals < 2:
                self.novelty_refusals += 1
                self.submitted = False
                # Preserve the bytes, but reopen the revision for editing.
                # Otherwise SubmitReminder sees a validated revision and
                # refuses every action except another identical submit.
                self.validated_revision = None
                self._validated_check = None
                return (
                    "REFUSED (repeat proposal): the component change ["
                    + sig + "] was already tried in previous rounds and its "
                    "paired causal effect was ~zero every time. Propose "
                    "something genuinely different (see the campaign evidence "
                    "in your briefing): a different construction family in "
                    "prompt.md, a different capability tool, or a different "
                    "skill. Edit, validate_harness, then submit again. "
                    f"({2 - self.novelty_refusals} refusal(s) left before a "
                    "repeat is accepted.)"
                )
            self.novelty_flag = "repeat_accepted_after_warnings"
        self.partial_spec = check.partial
        self.effective = check.effective
        # Same rule as the spec path: scaffolding the proposer never wrote is
        # not a changed field.  This is the path campaigns actually take, so
        # applying it only in _check() would have left every file-native
        # submission still crediting an inherited placeholder.
        self.changed_fields = self._drop_scaffold_fields(check.changed_fields)
        self.component_audit = list(check.component_audit)
        self.raw_submission = yaml.safe_dump(
            check.partial, sort_keys=False, allow_unicode=True, width=100
        ).strip()
        return (
            "SUBMITTED. Candidate H2 accepted; changes: "
            + ", ".join(check.changed_fields)
            + "."
        )


_CURRENT: "contextvars.ContextVar[Optional[ProposeSession]]" = contextvars.ContextVar(
    "propose_session", default=None
)


def get_session() -> ProposeSession:
    s = _CURRENT.get()
    if s is None:
        raise RuntimeError("no active ProposeSession (tool called outside propose_scope)")
    return s


@contextmanager
def propose_scope(session: ProposeSession):
    token = _CURRENT.set(session)
    try:
        yield session
    finally:
        _CURRENT.reset(token)
