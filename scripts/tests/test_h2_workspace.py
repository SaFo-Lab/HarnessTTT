from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outer.workspace import h2_workspace as ws  # noqa: E402
from outer.genome import harness_spec as hs  # noqa: E402
from outer.compiling.materialize import INNER_HARNESS, materialize  # noqa: E402
from outer.workspace.propose_session import ProposeSession  # noqa: E402


class H2WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = hs.read_base_spec(INNER_HARNESS)

    def _materialize(self, root: Path, spec: dict | None = None) -> Path:
        effective = spec or self.base
        package = root / "h2"
        materialize(effective, package, meta={"effective": effective})
        return package

    def test_spec_parser_only_unwraps_a_whole_document_yaml_fence(self) -> None:
        data = {
            "schema": hs.SCHEMA_VERSION,
            "system_prompt": (
                "Use this ordinary example inside the prompt:\n\n"
                "```text\nr_i + r_j <= distance(i, j)\n```"
            ),
        }
        rendered = yaml.safe_dump(data, sort_keys=False)
        direct = hs.parse_and_validate(rendered)
        fenced = hs.parse_and_validate(f"```yaml\n{rendered}```")
        mapped = hs.validate_mapping(data)
        self.assertTrue(direct.valid, direct.errors)
        self.assertTrue(fenced.valid, fenced.errors)
        self.assertTrue(mapped.valid, mapped.errors)
        self.assertIn("```text", direct.spec["system_prompt"])

    def test_materialized_openai_extra_body_is_not_nested_under_extra_params(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            llm = yaml.safe_load((package / "agent.yaml").read_text())["llm_config"]
            self.assertIn("extra_body", llm)
            self.assertEqual(llm["extra_body"]["top_k"], self.base["sampling"]["top_k"])
            self.assertNotIn("extra_params", llm)

            # Exercise the exact NexAU adapter that feeds OpenAI's create()
            # call.  A nested YAML key would survive here as the unsupported
            # literal parameter named ``extra_params``.
            from nexau.archs.llm.llm_config import LLMConfig

            openai_params = LLMConfig(**llm).to_openai_params()
            self.assertEqual(openai_params["extra_body"], {"top_k": 20})
            self.assertNotIn("extra_params", openai_params)

    def _add_tool(self, package: Path, name: str = "alpha_probe") -> None:
        agent_path = package / "agent.yaml"
        agent = yaml.safe_load(agent_path.read_text())
        agent["tools"].append({
            "name": name,
            "yaml_path": f"./tools/{name}.tool.yaml",
            "binding": "inner.harness.tools.custom_runtime:custom_tool",
            "extra_kwargs": {"_sah_py_path": f"./custom_tools/{name}.py"},
        })
        self.assertTrue(
            ws.write_file(
                package, "agent.yaml",
                yaml.safe_dump(agent, sort_keys=False, allow_unicode=True),
            ).startswith("WROTE")
        )
        schema = {
            "type": "tool",
            "name": name,
            "description": "Return a compact structural probe before editing.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
        ws.write_file(
            package, f"tools/{name}.tool.yaml",
            yaml.safe_dump(schema, sort_keys=False),
        )
        ws.write_file(
            package, f"custom_tools/{name}.py",
            "def run(ctx, args):\n"
            "    program = ctx.get_program()\n"
            "    lines = [line.strip() for line in program.splitlines() "
            "if line.strip()]\n"
            "    functions = [line for line in lines if line.startswith('def ')]\n"
            "    return {'nonempty_lines': len(lines), "
            "'function_headers': functions}\n",
        )
        prompt = (package / "prompt.md").read_text().rstrip()
        ws.write_file(
            package, "prompt.md",
            prompt + "\n\n" + ws.generated_component_prompt_block(
                "tool", name,
                usage_guidance="Call once to inspect the current search state.",
                expected_behavior_change=(
                    "Return structural measurements before the next edit."
                ),
            ),
        )

    def _add_skill_and_middleware(self, package: Path) -> None:
        agent = yaml.safe_load((package / "agent.yaml").read_text())
        agent["skills"].append("./skills/ac2-search")
        agent["middlewares"].insert(0, {
            "import": "middlewares.diversity_guard:GeneratedMiddleware",
            "params": {},
        })
        ws.write_file(
            package, "agent.yaml",
            yaml.safe_dump(agent, sort_keys=False, allow_unicode=True),
        )
        ws.write_file(
            package, "skills/ac2-search/SKILL.md",
            "---\nname: ac2-search\n"
            "description: Search AC2 structures.\n---\n\n"
            "# AC2\nUse multiple representations.",
        )
        ws.write_file(
            package, "middlewares/diversity_guard.py",
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if (state.get('stalled_evals', 0) >= 2 and\n"
            "            state.get('family_streak', 0) >= 4 and\n"
            "            state.get('last_step_kind') in ('diff', 'full_rewrite') and\n"
            "            not state.get('pending_evaluation', False) and\n"
            "            state.get('last_edit_staged') is not False and\n"
            "            not state.get('active_tool_gate') and\n"
            "            state.get('evals_remaining', 0) > 0):\n"
            "        return {'note': 'NEXT CALL: use edit_solution to switch family.',\n"
            "                'require_tools': ['edit_solution'],\n"
            "                'require_structure_change': True}\n"
            "    return None",
        )
        ws.write_file(
            package, "middlewares/diversity_guard.middleware.yaml",
            "name: diversity_guard\n"
            "hook: before_model\n"
            "description: Force a family switch after a stall.",
        )
        prompt = (package / "prompt.md").read_text().rstrip()
        ws.write_file(
            package, "prompt.md",
            prompt
            + "\n\n# Generated Skill: ac2-search\n"
            + "Load when AC2 structure search stalls."
            + "\n\n"
            + ws.generated_component_prompt_block(
                "middleware", "diversity_guard"
            ),
        )

    def test_shell_feedback_starts_from_agent_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            output = ws.inspect(package, "cat agent.yaml")
            self.assertIn("system_prompt: ./prompt.md", output)
            self.assertIn("tools:", output)
            self.assertIn("probe_solution", ws.inspect(package, "ls tools/"))
            self.assertIn("ERROR", ws.inspect(package, "cat ../outside"))

    def test_added_tool_requires_file_mount_and_prompt_then_is_submitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            self._add_tool(package)
            check = ws.validate_workspace(package, self.base)
            self.assertTrue(check.valid, check.errors)
            self.assertEqual(
                hs.generated_component_inventory(check.effective or {})["new_tools"],
                ["alpha_probe"],
            )
            self.assertIn("new_tools.alpha_probe", check.changed_fields)
            self.assertIn("system_prompt", check.changed_fields)

            session = ProposeSession(base_spec=self.base, draft_dir=package)
            self.assertIn(
                "system_prompt: ./prompt.md",
                session.inspect_harness("cat agent.yaml"),
            )
            session.inspect_harness("cat prompt.md")
            self.assertIn("VALID H2 WORKSPACE", session.validate_harness())
            self.assertIn("SUBMITTED", session.submit_harness())
            self.assertTrue(session.submitted)
            self.assertIn("new_tools", session.partial_spec or {})

            final_package = Path(td) / "final"
            materialize(
                session.effective or {},
                final_package,
                meta={"effective": session.effective or {}},
            )
            runtime_schema = yaml.safe_load(
                (final_package / "tools" / "alpha_probe.tool.yaml").read_text()
            )["input_schema"]
            self.assertEqual(
                runtime_schema["properties"]["_sah_py_path"]["const"],
                "./custom_tools/alpha_probe.py",
            )
            self.assertNotIn(
                "_sah_py_path",
                (session.effective or {})["new_tools"][0]["input_schema"]
                    .get("properties", {}),
            )

    def test_unadvertised_or_orphan_tool_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            self._add_tool(package)
            prompt = (package / "prompt.md").read_text()
            ws.write_file(
                package, "prompt.md",
                prompt.replace(
                    ws.generated_component_prompt_block(
                        "tool", "alpha_probe",
                        usage_guidance=(
                            "Call once to inspect the current search state."
                        ),
                        expected_behavior_change=(
                            "Return structural measurements before the next edit."
                        ),
                    ),
                    "",
                ),
            )
            prompt_without_declaration = (package / "prompt.md").read_bytes()
            check = ws.validate_workspace(package, self.base)
            self.assertFalse(check.valid)
            self.assertEqual(
                prompt_without_declaration, (package / "prompt.md").read_bytes()
            )
            self.assertTrue(any(
                "does not name current tools" in error for error in check.errors
            ), check.errors)

            # If the files exist but agent.yaml does not mount them, they are
            # not silently accepted as proposed components either.
            agent = yaml.safe_load((package / "agent.yaml").read_text())
            agent["tools"] = [row for row in agent["tools"]
                              if row.get("name") != "alpha_probe"]
            ws.write_file(
                package, "agent.yaml",
                yaml.safe_dump(agent, sort_keys=False, allow_unicode=True),
            )
            check = ws.validate_workspace(package, self.base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "orphan tool 'alpha_probe'" in error for error in check.errors
            ), check.errors)
            agent = yaml.safe_load((package / "agent.yaml").read_text())
            self.assertFalse(any(row.get("name") == "alpha_probe"
                                 for row in agent["tools"]))

    def test_fixed_usage_ledger_mount_deletion_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            agent_path = package / "agent.yaml"
            agent = yaml.safe_load(agent_path.read_text())
            mount = "inner.runtime.token_ledger:LLMUsageLedgerMiddleware"
            agent["middlewares"] = [
                row for row in agent["middlewares"]
                if row.get("import") != mount
            ]
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
            attempted = agent_path.read_bytes()
            check = ws.validate_workspace(package, self.base)
            self.assertFalse(check.valid)
            self.assertEqual(attempted, agent_path.read_bytes())
            self.assertTrue(any(
                "agent.yaml contains unsupported" in error
                for error in check.errors
            ), check.errors)

    def test_prompt_cannot_claim_an_unmounted_generated_component(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            prompt = (package / "prompt.md").read_text().rstrip()
            ws.write_file(
                package,
                "prompt.md",
                prompt
                + "\n\n# Generated Tool: structure_analyzer\n"
                + "Call `structure_analyzer` before a major redesign.\n",
            )
            before = (package / "prompt.md").read_bytes()
            check = ws.validate_workspace(package, self.base)
            self.assertFalse(check.valid)
            self.assertEqual(before, (package / "prompt.md").read_bytes())
            self.assertIn("structure_analyzer", (package / "prompt.md").read_text())

    def test_generated_component_catalog_cannot_list_an_unmounted_tool(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            prompt = (package / "prompt.md").read_text().rstrip()
            ws.write_file(
                package,
                "prompt.md",
                prompt
                + "\n\n## Generated tools (conditional capabilities)\n"
                + "- `kvpr_analyzer`: inspect placement pressure.\n",
            )
            check = ws.validate_workspace(package, self.base)
            self.assertFalse(check.valid)
            self.assertIn(
                "system_prompt declares unmounted generated tool "
                "'kvpr_analyzer'",
                check.errors,
            )

    def test_v11_portfolio_accepts_a_prompt_only_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=package, slot_index=0
            )
            session.inspect_harness("cat agent.yaml")
            session.inspect_harness("cat prompt.md")
            session.edit_harness_file(
                "prompt.md",
                append_text="## Task-specific search\nTry a wider search family.",
            )
            result = session.validate_harness()
            self.assertIn("VALID H2 WORKSPACE", result)
            self.assertIn("system_prompt", result)
            self.assertNotIn("requires a generated TOOL implementation", result)

    def test_file_level_delete_is_explicit_and_does_not_reinherit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = self._materialize(root)
            self._add_tool(first)
            added = ws.validate_workspace(first, self.base)
            self.assertTrue(added.valid, added.errors)
            inherited = added.effective or {}

            second = root / "second"
            materialize(inherited, second, meta={"effective": inherited})
            agent = yaml.safe_load((second / "agent.yaml").read_text())
            agent["tools"] = [row for row in agent["tools"]
                              if row.get("name") != "alpha_probe"]
            ws.write_file(
                second, "agent.yaml",
                yaml.safe_dump(agent, sort_keys=False, allow_unicode=True),
            )
            ws.delete_file(second, "tools/alpha_probe.tool.yaml")
            ws.delete_file(second, "custom_tools/alpha_probe.py")
            prompt = (second / "prompt.md").read_text()
            ws.write_file(
                second, "prompt.md",
                prompt.replace(
                    ws.generated_component_prompt_block(
                        "tool", "alpha_probe",
                        usage_guidance=(
                            "Call once to inspect the current search state."
                        ),
                        expected_behavior_change=(
                            "Return structural measurements before the next edit."
                        ),
                    ),
                    "",
                ),
            )

            removed = ws.validate_workspace(second, inherited)
            self.assertTrue(removed.valid, removed.errors)
            self.assertEqual(
                (removed.partial or {}).get("remove_generated"),
                {"tools": ["alpha_probe"]},
            )
            self.assertEqual(
                hs.generated_component_inventory(removed.effective or {})["new_tools"],
                [],
            )

    def test_inherited_tool_skill_and_middleware_are_editable_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = self._materialize(root)
            self._add_tool(first)
            self._add_skill_and_middleware(first)
            added = ws.validate_workspace(first, self.base)
            self.assertTrue(added.valid, added.errors)
            inherited = added.effective or {}

            second = root / "second"
            materialize(inherited, second, meta={"effective": inherited})
            self.assertIn(
                "EDITED",
                ws.edit_file(
                    second,
                    "custom_tools/alpha_probe.py",
                    "'function_headers': functions",
                    "'function_headers_v2': functions",
                ),
            )
            self.assertIn(
                "EDITED",
                ws.edit_file(
                    second,
                    "skills/ac2-search/SKILL.md",
                    "Use multiple representations.",
                    "Probe and compare multiple representations.",
                ),
            )
            self.assertIn(
                "EDITED",
                ws.edit_file(
                    second,
                    "middlewares/diversity_guard.py",
                    "NEXT CALL: use edit_solution to switch family.",
                    "NEXT CALL: use edit_solution to switch representation family.",
                ),
            )

            updated = ws.validate_workspace(second, inherited)
            self.assertTrue(updated.valid, updated.errors)
            self.assertEqual(updated.changed_fields, [
                "new_tools.alpha_probe",
                "new_skills.ac2-search",
                "new_middlewares.diversity_guard",
            ])
            self.assertEqual(
                hs.generated_component_inventory(updated.effective or {}),
                hs.generated_component_inventory(inherited),
            )

    def test_session_enforces_inspect_edit_validate_submit_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = self._materialize(Path(td))
            session = ProposeSession(base_spec=self.base, draft_dir=package)
            self.assertIn("first inspect", session.inspect_harness("ls tools/"))
            self.assertIn(
                "first inspect",
                session.write_harness_file(
                    "custom_tools/too_early.py", "def run(ctx, args): return {}"
                ),
            )
            self.assertIn(
                "first inspect",
                session.edit_harness_file(
                    "prompt.md", "Be decisive", "Be systematic"
                ),
            )
            session.inspect_harness("cat agent.yaml")
            self.assertIn(
                "read existing file first",
                session.edit_harness_file(
                    "prompt.md", "Be decisive", "Be systematic"
                ),
            )
            session.inspect_harness("cat prompt.md")
            self.assertIn(
                "EDITED",
                session.edit_harness_file(
                    "prompt.md", "Be decisive", "Be systematic"
                ),
            )
            self.assertIn(
                "requires a successful validate_harness",
                session.submit_harness(),
            )

            # A fresh session shows the successful path because submit_harness
            # is a stop tool: calling it too early correctly invalidates the
            # previous session.
            session = ProposeSession(base_spec=self.base, draft_dir=package)
            session.inspect_harness("cat agent.yaml")
            session.inspect_harness("cat prompt.md")
            self.assertIn(
                "APPENDED",
                session.edit_harness_file(
                    "prompt.md",
                    append_text=(
                        "## Task-specific search\n"
                        "Use autocorrelation symmetry to rank structural edits."
                    ),
                ),
            )
            self.assertIn(
                "append_text cannot be combined",
                session.edit_harness_file(
                    "prompt.md", "Be decisive", "Be systematic", "extra"
                ),
            )
            self.assertIn("VALID H2 WORKSPACE", session.validate_harness())
            self.assertIn("Candidate H2 accepted", session.submit_harness())

    def test_legacy_parent_is_copied_but_only_proposer_can_repair_prompt(self) -> None:
        legacy = dict(self.base)
        legacy["system_prompt"] = "Legacy executor instructions without a catalog."
        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "h2"
            with self.assertRaisesRegex(ValueError, "proposer-owned system_prompt"):
                materialize(legacy, package, meta={"effective": legacy})

            materialize(
                legacy,
                package,
                meta={"effective": legacy},
                validate_prompt=False,
            )
            invalid = ws.validate_workspace(package, legacy)
            self.assertFalse(invalid.valid)
            self.assertTrue(any("does not name" in err for err in invalid.errors))

            catalog = hs.h2_component_catalog(legacy)
            repaired_prompt = (
                "Executor workflow.\nTools: " + ", ".join(catalog["tools"])
                + "\nSkills: " + ", ".join(catalog["skills"])
                + "\nMiddleware: " + ", ".join(catalog["middlewares"])
            )
            ws.write_file(package, "prompt.md", repaired_prompt)
            repaired = ws.validate_workspace(package, legacy)
            self.assertTrue(repaired.valid, repaired.errors)
            self.assertEqual(repaired.changed_fields, ["system_prompt"])

    def test_middleware_wrapper_edits_outside_user_hook_are_not_silently_lost(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = self._materialize(root)
            self._add_skill_and_middleware(first)
            added = ws.validate_workspace(first, self.base)
            self.assertTrue(added.valid, added.errors)
            inherited = added.effective or {}

            second = root / "second"
            materialize(inherited, second, meta={"effective": inherited})
            ws.edit_file(
                second,
                "middlewares/diversity_guard.py",
                '"""Generated middleware (h2spec/1.0), lifecycle-audited."""',
                '"""A proposer edit outside the user-hook region."""',
            )
            prompt = (second / "prompt.md").read_text().rstrip()
            ws.write_file(second, "prompt.md", prompt + "\nUse explicit diagnostics.")
            # Validation is observational: a raw wrapper edit is rejected and
            # preserved byte-for-byte for attribution.
            edited = (second / "middlewares/diversity_guard.py").read_bytes()
            check = ws.validate_workspace(second, inherited)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "changed code outside its USER-HOOK" in error
                for error in check.errors
            ), check.errors)
            self.assertEqual(
                edited,
                (second / "middlewares/diversity_guard.py").read_bytes(),
            )

            # but a rewrite that DESTROYS the hook still fails closed
            # (keep the sentinels so this exercises the normalisation branch)
            gutted = edited.decode().replace("def before_model", "def wrong_hook")
            ws.write_file(second, "middlewares/diversity_guard.py", gutted)
            broken = ws.validate_workspace(second, inherited)
            self.assertFalse(broken.valid)
            self.assertTrue(
                any("no def for its hook" in err for err in broken.errors),
                broken.errors,
            )


if __name__ == "__main__":
    unittest.main()


class Scaffold004NormalisationTest(unittest.TestCase):
    """Validation reports incomplete plumbing and never mutates H1 bytes."""

    def _draft(self, tmp):
        base = hs.read_base_spec(INNER_HARNESS)
        d = Path(tmp) / "h2"
        materialize(base, d, meta={"effective": base}, validate_prompt=False)
        return d, base

    def test_implementationless_mount_is_preserved_and_rejected(self):
        import tempfile, yaml
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            agent = yaml.safe_load((d / "agent.yaml").read_text())
            agent["tools"].append({
                "name": "ghost_tool",
                "yaml_path": "./tools/ghost_tool.tool.yaml",
                "binding": "inner.harness.tools.custom_runtime:custom_tool",
                "extra_kwargs": {"py_path": "./custom_tools/ghost_tool.py"}})
            (d / "agent.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
            (d / "prompt.md").write_text(
                (d / "prompt.md").read_text() + "\nPrefer step functions.\n")
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertTrue(any("ghost_tool" in error for error in check.errors))
            agent = yaml.safe_load((d / "agent.yaml").read_text())
            self.assertTrue(
                any(m.get("name") == "ghost_tool" for m in agent["tools"]))

    def test_valid_orphan_middleware_is_preserved_and_rejected(self):
        import tempfile, yaml
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            (d / "middlewares" / "family_guard.py").write_text(
                "def before_model(hook_input):\n"
                "    if hook_input.get('state', {}).get('family_streak', 0) >= 4:\n"
                "        return 'Switch family now.'\n"
                "    return None\n")
            before = (d / "middlewares" / "family_guard.py").read_bytes()
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "orphan middleware 'family_guard'" in error
                for error in check.errors
            ), check.errors)
            self.assertEqual(
                before, (d / "middlewares" / "family_guard.py").read_bytes()
            )
            agent = yaml.safe_load((d / "agent.yaml").read_text())
            self.assertFalse(any("family_guard" in str(m.get("import", ""))
                                 for m in agent["middlewares"]))
            self.assertFalse(
                (d / "middlewares" / "family_guard.middleware.yaml").exists())

    def test_hasattr_is_allowed_in_tool_bodies(self):
        from outer.safety.static_gates import check_tool_code
        ok, errs = check_tool_code(
            "def run(ctx, args):\n"
            "    return {'has': bool(hasattr(ctx, 'best_score'))}\n")
        self.assertTrue(ok, errs)
        ok2, errs2 = check_tool_code(
            "def run(ctx, args):\n"
            "    return {'x': getattr(ctx, 'best_score')()}\n")
        self.assertFalse(ok2)

    def test_valid_orphan_tool_is_preserved_and_rejected(self):
        import tempfile, yaml
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            (d / "custom_tools").mkdir(exist_ok=True)
            (d / "custom_tools" / "structured_init.py").write_text(
                "def run(ctx, args):\n"
                "    program = ctx.get_program()\n"
                "    blocks = [part.strip() for part in program.split('def ') "
                "if part.strip()]\n"
                "    sizes = [len(part.splitlines()) for part in blocks]\n"
                "    return {'blocks': len(blocks), 'line_counts': sizes}\n")
            before = (d / "custom_tools" / "structured_init.py").read_bytes()
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "orphan tool 'structured_init'" in error
                for error in check.errors
            ), check.errors)
            self.assertEqual(
                before, (d / "custom_tools" / "structured_init.py").read_bytes()
            )
            agent = yaml.safe_load((d / "agent.yaml").read_text())
            self.assertFalse(any(m.get("name") == "structured_init"
                                 for m in agent["tools"]))

    def test_orphan_skill_is_preserved_and_rejected(self):
        import tempfile, yaml
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            sd = d / "skills" / "variable-radius-packing"
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text(
                "---\nname: variable-radius-packing\ndescription: alt family\n"
                "---\n# Variable radius\nUse mixed radii.\n")
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "orphan skill 'variable-radius-packing'" in error
                for error in check.errors
            ), check.errors)
            agent = yaml.safe_load((d / "agent.yaml").read_text())
            self.assertFalse(any(
                "variable-radius-packing" in str(m)
                for m in (agent.get("skills") or [])
            ))

    def test_gate_failing_orphan_tool_is_preserved_with_exact_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            (d / "custom_tools").mkdir(exist_ok=True)
            (d / "custom_tools" / "bad_tool.py").write_text(
                "import os\n"
                "def run(ctx, args):\n"
                "    return {'cwd': os.getcwd()}\n")
            (d / "prompt.md").write_text(
                (d / "prompt.md").read_text() + "\nPrefer step functions.\n")
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertTrue((d / "custom_tools" / "bad_tool.py").is_file())
            self.assertTrue(any(
                "orphan tool 'bad_tool'" in error and "forbidden import: os" in error
                for error in check.errors
            ), check.errors)

    def test_gate_failing_orphan_middleware_is_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            path = d / "middlewares" / "broken_gate.py"
            path.write_text(
                "def before_model(hook_input):\n"
                "    return hook_input['undocumented']\n"
            )
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertTrue(path.is_file())
            self.assertTrue(any(
                "orphan middleware 'broken_gate'" in error
                for error in check.errors
            ), check.errors)

    def test_stale_prompt_declaration_is_preserved_and_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d, base = self._draft(tmp)
            (d / "prompt.md").write_text(
                (d / "prompt.md").read_text()
                + "\nPrefer step functions.\n"
                + "\n# Generated Tool: step_initializer\nCall it early.\n")
            before = (d / "prompt.md").read_bytes()
            check = ws.validate_workspace(d, base)
            self.assertFalse(check.valid)
            self.assertEqual(before, (d / "prompt.md").read_bytes())
            self.assertIn("step_initializer", (d / "prompt.md").read_text())
