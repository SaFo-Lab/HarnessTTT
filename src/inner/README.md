# src/inner — the inner loop (M0 + H2)

> RESTRUCTURE-001: files now live in themed subpackages — see `../README.md` for the map; old import paths still work via shims.

**Inner loop:** `M0 + H2 → solution + reward`. **Outer loop** (separate, later):
`M_phi + H1 → new H2`. This package is the inner loop: a permanently frozen
executor **M0** (Qwen3.5-9B) running the discovery harness **H2** over the EFT
held-out tasks. **H2 is a declarative NexAU agent package** (`harness/`) — the
exact surface the outer proposer will later generate/mutate — with its prompts
aligned to **OpenEvolve** (what M0/Finch was trained on). Here H2 is the initial,
hand-written baseline harness.

## The harness package — `harness/` (the evolvable surface)

```
harness/
├── agent.yaml                     # the harness config: prompt + tools + skills + middlewares + sampling
├── system.md                      # system prompt (OpenEvolve framing + SEARCH/REPLACE format)
├── tools/
│   ├── discovery.py               # bindings: edit_solution, evaluate_solution, finish
│   ├── runtime.py                 # contextvar bridge -> active InnerSession
│   ├── edit_solution.tool.yaml    # accepts SEARCH/REPLACE diff (preferred) or full rewrite
│   ├── evaluate_solution.tool.yaml
│   └── finish.tool.yaml           # stop tool
├── skills/discovery-optimization/SKILL.md   # the discovery method
└── middleware/budget_reminder.py  # custom middleware: warns when eval budget is low
```

`agent.yaml` references NexAU built-in middlewares (`long_tool_output`,
`round_and_token_reminder`) + a tracer, exactly like the earlier runtime's `harness/nexau`.

## What H2 does

For one task, an edit→evaluate loop under a strict evaluation budget: the agent
sees the task + seed program, calls `edit_solution` (a targeted `SEARCH/REPLACE`
diff — M0's trained format — or a full EVOLVE-BLOCK rewrite), then
`evaluate_solution` (subprocess-isolated scoring → `combined_score`, validity,
error, best-so-far, evals-left), and repeats until budget is spent, then
`finish`. Reward = best `combined_score`; the seed is scored once as baseline.
All usage is tracked in an external `BudgetLedger`.

## Supporting modules

| File | Role |
|---|---|
| `eft_task.py` | Registry — 21 runnable EFT held-out tasks (18 OpenEvolve + 3 SimpleTES) from `MANIFEST_eft.json` |
| `program_edit.py` | EVOLVE-BLOCK split/splice + `SEARCH/REPLACE` diff + full-rewrite extraction |
| `_eval_worker.py` / `eval_runner.py` | Subprocess-isolated eval with hard timeout |
| `session.py` | `InnerSession` (program, best, ledger) + contextvar bridge |
| `harness_runner.py` | Loads `harness/agent.yaml`, injects endpoint+budget at runtime, runs one task |
| `run_baseline.py` | CLI over the suite → `summary.{json,csv}` + `provenance.json` (has `--seed-only`) |

## OpenEvolve alignment (why the prompts look like this)

M0 = Qwen3.5-9B = Finch's base, trained on OpenEvolve-generated trajectories.
So H2 uses OpenEvolve's system framing ("expert iteratively improving a codebase
to maximize the metric") and its `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE`
edit format. Reference (not vendored): `code/evolution-fine-tuning/openevolve/openevolve/prompts/defaults/`.

## Status (verified on the login node, no model)

- ✅ Splice preserves the fixed entry; `SEARCH/REPLACE` diff **and** full-rewrite apply via the `edit_solution` tool.
- ✅ Subprocess eval isolation + timeout; broken edits handled (no crash).
- ✅ Full tool loop: seed baseline (uncharged), budget tracking/exhaustion, best-tracking, ledger, summary.
- ✅ Package imports + tool bindings resolve standalone; `run_baseline.py --seed-only` (circle_packing=0.364, hadamard=0.143).
- ⏳ The NexAU agent path is written to the verified NexAU API but **not yet run** — needs the env + a served endpoint.

## Running

```bash
source config/env.sh   # HarnessTTT root

# pipeline check, no model:
cd src && python3 -m inner.run_baseline --seed-only --tiers tier0_cpu_nosetup

# real baseline (one-time env, then a 4-GPU serve+run job):
bash scripts/setup/setup_runtime_env.sh            # -> $SAH_RUNTIME_ENV (Python 3.12 env with NexAU and the evaluator deps)
sbatch scripts/serve_and_run_baseline.sbatch       # serves Qwen3.5-9B + runs run_baseline
```

Note: run as a package (`python -m inner.run_baseline` with `src/` on the path);
the tool/middleware bindings in `agent.yaml` import `inner.harness.*`.

## Next steps

- Run the baseline (env + GPU job).
- Optional OpenEvolve-style generation-loop harness variant (database-backed
  context, no tool-calling) for strict EFT-parity numbers — templates in the
  upstream OpenEvolve clone.
- tier2 tasks (GPU/Docker/Rust); `symbolic_regression` (non-standard runner).
