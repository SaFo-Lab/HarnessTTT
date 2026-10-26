<p align="center">
  <img src="assets/harnessttt_logo.svg" alt="HarnessTTT" width="450">
</p>

<p align="center">
  <a href="https://gray311.github.io/harnessrl/"><strong>Website</strong></a> &ensp;|&ensp;
  <a href="#quick-start"><strong>Quick Start</strong></a> &ensp;|&ensp;
  <a href="https://arxiv.org/abs/2610.17370"><strong>Paper</strong></a> &ensp;|&ensp;
  <a href="https://huggingface.co/papers/2610.17370"><strong>Hugging Face</strong></a> &ensp;|&ensp;
  <a href="https://huggingface.co/datasets/gray311/HarnessRL"><strong>Artifacts</strong></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.17370"><img src="https://img.shields.io/badge/arXiv-2610.17370-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2610.17370"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow?style=flat" alt="Hugging Face paper"></a>
  <a href="https://huggingface.co/datasets/gray311/HarnessRL"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HarnessRL-blue?style=flat" alt="Artifacts"></a>
  <a href="https://github.com/SaFo-Lab/Self_Adapt_Harness"><img src="https://img.shields.io/github/stars/SaFo-Lab/Self_Adapt_Harness?style=flat&logo=github&color=181717" alt="GitHub stars"></a>
  <!-- TODO(authors): add the license badge once LICENSE is chosen -->
</p>

<p align="center">
  <strong>Learn the harness, not the weights — HarnessTTT trains a proposer that writes executable harnesses for a frozen model, at test time.</strong>
</p>

---

## What is HarnessTTT?

An agent is a frozen **executor** model plus a **harness**: the system prompt, skills, tools, middleware and execution controls it runs with. **HarnessTTT** learns a **proposer** at test time. The proposer writes complete, executable harnesses for one discovery task; each harness drives the frozen executor through a budgeted search; the score that search reaches updates the proposer. The executor is never touched.

Executor RL (TTT-Discover) writes reward into the executor's weights. Fixed-proposer search (OpenEvolve, GEPA) never updates the proposer. HarnessTTT routes the reward to the proposer instead. What comes out is a readable harness package — `agent.yaml`, a prompt, skills, generated tools, middleware — that moves to other executors.

### Key Results

| | |
|---|---|
| **11 / 11** | With a frozen 9B executor, HarnessTTT improves on OpenEvolve with the same executor on all eleven benchmark tasks ([paper](https://arxiv.org/abs/2610.17370), Table 1) |
| **1.17 → 2.64** | Circle packing (n=26), sum of radii: OpenEvolve vs. HarnessTTT with the same frozen Qwen3.5-9B ([Table 1](https://arxiv.org/abs/2610.17370)) |
| **up to 92%** | Improvement over the discovery-fine-tuned Finch-9B, on every one of the eleven tasks ([Table 1](https://arxiv.org/abs/2610.17370)) |
| **9 / 11** | Best open-weight sub-10B result ([Table 1](https://arxiv.org/abs/2610.17370)) |
| **0.380919** | Erdős minimum overlap (↓), below the best human reference of 0.380927 ([paper](https://arxiv.org/abs/2610.17370)) |
| **9 / 14** | Harnesses learned with a frozen Qwen executor, moved to GPT-5.5, match or exceed the strongest reported system of any backbone ([Table 3](https://arxiv.org/abs/2610.17370)); the 27B proposer–executor pair improves on 11 of 14 |

---

## Installation

```bash
git clone https://github.com/SaFo-Lab/Self_Adapt_Harness.git HarnessTTT && cd HarnessTTT
pip install -e .                                   # packages: harnessttt, inner, outer, trainer
python3 scripts/setup/setup_dataset.py             # task snapshot from pinned public sources (+ --tiers core,gpu for the kernel tasks)
cp config/local.env.example config/local.env       # site paths: executor checkpoint, container image, envs, RUN_ROOT, Slurm account
bash scripts/setup/setup_deps.sh                   # NexAU agent runtime (pinned commit + patch) and Megatron-LM
bash scripts/setup/setup_runtime_env.sh            # Python 3.12 env for the executor runtime (NexAU + evaluator deps); run it where python3.12 exists, e.g. inside your container
bash scripts/setup/setup_slime.sh                  # slime (pinned, patched) — only needed to train the proposer
bash scripts/setup/preflight.sh                    # checks pins, dataset checksums, site settings, imports (--executor-only skips training)
```

`setup_dataset.py` builds `dataset/self_adapt_harness/` (git-ignored, ~350 MB for the 11-task protocol) from pinned commits of evolution-fine-tuning, SimpleTES, SkyDiscover and TTT-Discover plus sha256-checked downloads, and verifies every file against `config/dataset/CHECKSUMS.json` ([`dataset/README.md`](dataset/README.md)). Python ≥ 3.9 runs the package (task registry, pipeline driver, `harnessttt` API) from an editable install of the clone (the loop needs the repository's scripts and templates); running a harness needs the NexAU runtime, which requires **Python 3.12**. The paper's executor and training environments are Python 3.12 with vLLM 0.17.1 and torch 2.9.1 (`config/dependencies.lock.json`). Operations guide for the full loop: [`docs/pipeline_guide.md`](docs/pipeline_guide.md).

---

## Quick Start

### Optimize a harness for circle packing

The user gives a task (description, one seed program, evaluator), the path of the initial executor harness, and the models; HarnessTTT returns the evolved harness package and its score.

```python
import harnessttt

# A discovery task: a description, ONE seed program, and a black-box evaluator
task = harnessttt.examples.circle_packing.init_task()
print(task.description[:60], task.initial_program_path, task.evaluator_path)

model = "/models/Qwen3.5-9B"                       # local snapshot of Qwen/Qwen3.5-9B

result = harnessttt.optimize(
    task=task,
    seed_harness=harnessttt.INITIAL_HARNESS,       # path to the initial executor harness (src/inner/harness)
    executor_lm=model,                             # frozen: never updated
    proposer_lm=model,                             # same base; its LoRA is trained at test time
    rounds=15,
    proposals_per_round=8,
    max_metric_calls=20,                           # evaluator calls per rollout
)

print("Evolved harness:", result.best_harness_path)   # agent.yaml, prompt.md, skills/, tools/, custom_tools/, middlewares/ (the initial harness uses middleware/)
print("Best score:", result.best_metric, task.metric)  # sum of radii
result.write_best_program("cp26_best.py")
```

**Result:** With a frozen Qwen3.5-9B executor, circle packing goes from 1.477767 → 2.635983 (sum of radii, higher is better). These are paper numbers (Table 2, *initial* vs. *weight* condition), not the output of this snippet.

The same run as a script: `python examples/quickstart_circle_packing.py --rounds 1 --proposals 2 --evals 3` (one Slurm round job; `--dry-run` only writes the job script).

**Hardware:** the loop runs as Slurm jobs. A round job needs 2 GPUs (vLLM serves the frozen executor and the proposer, `round_gpus`), a proposer update needs 8 GPUs (slime, FSDP LoRA, `train_gpus`); the paper's 27B campaign used 8 GPUs per round job. There is no API-model mode for the loop itself; API models can run a *released* harness (next subsection). `optimize()` is the configuration-file entry point `python3 src/pipeline.py <config.yaml>` in Python form, and it writes the same run directory (`RUN_ROOT/<campaign>/<name>/`).

What you get back is a directory. The released circle-packing package ([`harnesses/circle_packing_n26`](https://huggingface.co/datasets/gray311/HarnessRL/tree/main/harnesses/circle_packing_n26), produced by the paper's 9B campaign) looks like this:

```text
circle_packing_n26/
├── agent.yaml                         # the harness spec after materialization
├── prompt.md                          # evolved executor system prompt
├── spec.yaml, meta.json               # the proposer's artifact and its provenance (round, changed fields)
├── skills/discovery-optimization/     # base skill
├── skills/hex-pack-complete-guide/    # generated skill
├── tools/*.tool.yaml                  # tool schemas (4 fixed + hexagonal_construction)
├── custom_tools/hexagonal_construction.py
└── middlewares/                       # budget_reminder, stall_restart + 2 generated reminders
```

An excerpt of that package's `agent.yaml` (the proposer's additions, verbatim):

```yaml
- name: hexagonal_construction
  yaml_path: ./tools/hexagonal_construction.tool.yaml
  binding: inner.harness.tools.custom_runtime:custom_tool
  extra_kwargs:
    py_path: ./custom_tools/hexagonal_construction.py
skills:
- ./skills/discovery-optimization
- ./skills/hex-pack-complete-guide
middlewares:
- import: middlewares.indexing_reminder:GeneratedMiddleware
```

### Run an evolved harness on a different executor

Weak-to-strong transfer: download a released package and run it with another executor through any OpenAI-compatible endpoint, with no training.

```python
from huggingface_hub import snapshot_download          # pip install huggingface_hub
import harnessttt

root = snapshot_download("gray311/HarnessRL", repo_type="dataset",
                         allow_patterns=["harnesses/circle_packing_n26/*"])

row = harnessttt.run_harness(
    f"{root}/harnesses/circle_packing_n26",            # the evolved package (dir with agent.yaml)
    harnessttt.examples.circle_packing.init_task(),
    base_url="https://api.openai.com/v1", model="gpt-5.5", api_key="...",
    max_evals=20,                                      # evaluator calls for this rollout
    out_dir="runs/transfer",
)
print(row["best_score"], row["out_dir"])
```

`run_harness` calls `python -m inner.cli.run_baseline --harness-dir ... --base-url ... --model ...` under the executor runtime: the Python 3.12 environment with NexAU that `setup_runtime_env.sh` builds (`SAH_PYTHON` in `config/local.env`, or `python=...`); the API key travels as `OPENAI_API_KEY`, never on the command line. For a hosted API host it adds `--no-vllm-extras`: vLLM-only fields (`top_k`, `chat_template_kwargs`) are not sent, the token cap goes as `max_completion_tokens`, and `temperature`/`top_p` are left to the API's defaults (pass `extra_args=["--keep-sampling"]` to send them). A failed run raises `RuntimeError` with the executor's own error text.

### Bring your own task

Tasks are registered by files, not by a Python call. A task is a folder with a seed program whose editable region is delimited by `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`, an `evaluator.py` that returns `combined_score` (higher is better) and `validity`, and a `config.yaml` whose `prompt.system_message` is the task description. Add a record to `config/dataset/MANIFEST_eft.json` (installed into the snapshot by `setup_dataset.py`; or edit `dataset/self_adapt_harness/MANIFEST_eft.json` directly), rebuild the task-text registry, and give the task an alias in `src/inner/tasks/aliases.py` (`TASK_IDS`, and `DISPLAY` for the native metric). Optional record fields: `per_eval_timeout_s`, `python_exe`, `extra_pythonpath`, `env` (paths may use `${VARIABLES}` from `config/local.env`):

```json
{"task_id": "eft__math__circle_packing", "rel_path": "raw/eft-aac2e79/benchmarks/math/circle_packing",
 "evaluator_rel": "evaluator.py", "initial_program": "initial_program.py",
 "cost_tier": "tier0_cpu_nosetup", "plan_family": "geometric_construction",
 "domain": "combinatorial_construction", "deps": ["numpy"], "scorable": true}
```

```bash
python3 scripts/tools/build_task_text_registry.py     # refresh config/task_text_registry.json
python3 -c "import harnessttt; print(harnessttt.load_task('eft__math__circle_packing').evaluator_path)"
```

The evaluator-call budget is `max_metric_calls` (pipeline `max_evals`); the per-evaluation timeout comes from the task's `config.yaml` (`evaluator.timeout`) unless the manifest record sets `per_eval_timeout_s`.

---

## How It Works

Each round turns one search outcome into one proposer update:

1. **Propose** — the proposer reads the task description, the incumbent program and harness, and an experience digest, and writes K harness specs.
2. **Gate** — each spec is validated (schema, mounts), and generated code passes static checks and a sandboxed self-test before it can run.
3. **Roll out** — each compiled harness drives the frozen executor from the incumbent program, under a hard budget of B evaluator calls.
4. **Score** — each rollout's best score becomes a gap-normalized reward; advantages are leave-one-out, sharpened toward the group maximum.
5. **Update & ratchet** — only the proposer's LoRA weights change; the best program and harness carry into the next round.

A **harness spec** is a partial `agent.yaml` merged over the incumbent spec. The upper block overrides components the executor already has; the lower block changes the set of components (paper, Table 4):

| Field | Type | Meaning |
|---|---|---|
| `system_prompt` | text | executor system prompt |
| `skill_description` | text | skill card shown in the tool list |
| `skill_body` | text | skill playbook body |
| `tool_descriptions` | map | descriptions of the fixed tools |
| `sampling` | map | temperature, top-p, top-k, max tokens |
| `agent` | map | iteration cap |
| `middleware` | map | budget-reminder onset; stall-restart threshold/cap; output truncation |
| `new_tools` | list (≤3) | name, description, JSON schema, Python implementation |
| `new_skills` | list (≤2) | additional skill packages |
| `new_middlewares` | list (≤2) | name, hook, Python implementation |
| `remove_tools` | list | removable optional tools only |

Generated tools are plain Python files with one `run(ctx, args)` entry point; `ctx` exposes the program state, `stage_edit`, a cheap `probe()`, the budgeted `evaluate()` and read-only task inputs, and nothing else (no network, filesystem or subprocess). The trained object is the proposer because the executor's weights are the one thing that is kept constant across tasks and executors: a harness written for a 9B executor is a file tree, so it can be inspected, diffed, and handed to a larger model. Code map: proposer side `src/outer/` (H1 package, gates in `outer/safety/`, rewards and ratchet in `outer/reward/`), executor side `src/inner/` (task registry, evaluator subprocess, H2 runtime, initial harness), update `src/trainer/`, loop `src/pipeline.py`.

---

## Tasks

| Task | Registry id | Domain | Metric | Direction | In this release |
|---|---|---|---|---|---|
| Erdős minimum overlap | `eft__math__erdos_min_overlap` | mathematics | c₅ | ↓ | yes |
| AC1 (first autocorrelation inequality) | `eft__math__first_autocorr_ineq` | mathematics | c₁ | ↓ | yes |
| AC2 (second autocorrelation inequality) | `eft__math__second_autocorr_ineq` | mathematics | c₂ | ↑ | yes |
| Circle packing, n=26 (CP26) | `eft__math__circle_packing` | mathematics | sum of radii | ↑ | yes |
| Hadamard maximal determinant | `eft__math__hadamard_maximal_det` | mathematics | determinant ratio | ↑ | yes |
| AHC039 | `eft__ahc_simpletes__ahc039` | heuristic programming (AtCoder) | total score | ↑ | yes |
| AHC058 | `eft__ahc_simpletes__ahc058` | heuristic programming (AtCoder) | total score (M) | ↑ | yes |
| EPLB | `adrs__eplb` | systems (ADRS) | balancedness | ↑ | yes |
| PRISM | `adrs__prism` | systems (ADRS) | combined | ↑ | yes |
| LLM-SQL | `adrs__llm_sql` | systems (ADRS) | combined | ↑ | yes (no result claims in the paper) |
| Txn scheduling | `adrs__txn_scheduling` | systems (ADRS) | combined | ↑ | yes (no result claims in the paper) |
| TriMul (GPU MODE, leaderboard protocol) | `gpu_mode__trimul_lb` | GPU kernels | geometric-mean latency (µs) | ↓ | yes (needs an H100 evaluator GPU) |
| V2 (Voyager 2) | `eft__astro_simpletes__voyager_2` | astrodynamics | total ΔV | ↓ | yes |
| scRNA denoising | `eft__bio_simpletes__scrna_denoising` | biology | denoising score | ↑ | yes |
| Mariner 10, TriMul (SkyDiscover protocol), MLA decode, ZAPBench | `eft__astro_simpletes__mariner_10`, `gpu_mode__trimul`, `gpu_mode__mla_decode`, `eft__zapbench_simpletes__whole_brain_forecasting` | — | — | — | registry only, no paper result |

"In this release" means the task is in the registry (`config/dataset/MANIFEST_eft.json`) and runs once `setup_dataset.py` has built the snapshot; sources are pinned upstream commits of evolution-fine-tuning / SkyDiscover, SimpleTES and TTT-Discover (`config/dataset/sources.json`). The astrodynamics, biology and ZAPBench tasks (`--tiers extra`) also need the data their own setup scripts prepare.

---

## Released Harnesses

The Hugging Face dataset [`gray311/HarnessRL`](https://huggingface.co/datasets/gray311/HarnessRL) holds the initial harness (`harnesses/initial`, the package in `src/inner/harness`) and nine evolved packages from the 9B campaigns: `erdos_min_overlap`, `autocorrelation_i`, `autocorrelation_ii`, `circle_packing_n26`, `hadamard_max_det`, `ahc058`, `eplb`, `llm_sql`, `txn_scheduling`. Every package is a complete NexAU harness:

```text
harnesses/<task>/
├── agent.yaml          # agent configuration and component bindings
├── prompt.md           # evolved task guidance
├── spec.yaml           # proposer artifact before materialization
├── meta.json           # proposer round, changed fields, provenance
├── skills/*/SKILL.md   # loadable procedural knowledge
├── custom_tools/*.py   # generated tools (capability-limited)
├── tools/*.tool.yaml   # tool schemas
└── middlewares/*.py    # lifecycle hooks
```

Load one with `huggingface_hub.snapshot_download(..., allow_patterns=["harnesses/<task>/*"])` and run it with `harnessttt.run_harness(package_dir, task, base_url=..., model=...)` (or `python -m inner.cli.run_baseline --harness-dir <package_dir>`). Use the whole directory: prompt, skills, schemas, middleware bindings and control parameters define the artifact together. Two packages (`circle_packing_n26`, `hadamard_max_det`) carry a `repair_provenance.json` describing a post-hoc mounting repair; the dataset's README explains the difference from the as-run package.

---

## When HarnessTTT Shines

- **Frozen or API-only executors** — the executor is never fine-tuned; what is learned is a file tree it runs with.
- **Expensive evaluators with a small call budget** — every rollout is capped at B evaluator calls, and generated tools get a cheaper `probe()` outside that budget.
- **Discovery tasks with one seed program and a black-box metric** — a task is a description, a seed program with a delimited editable region, an evaluator, a direction and a budget.
- **Auditable artifacts** — every edit the proposer makes is a named file: a prompt, a skill, a tool schema with its implementation, a middleware hook.
- **Weak-to-strong transfer** — learn the harness with a small executor, then run it on a larger one.

---

## Release Scope

**Included:** the evolve loop (`src/pipeline.py`, Slurm workers and templates in `scripts/pipeline/`); the proposer-side package H1 with its gates (`src/outer/harness`, `src/outer/safety/`: static AST gates over a shared code policy, tool-schema guard, resource-limited tool and middleware self-tests, leak guard); the executor-side containment of generated tools (`src/inner/harness/tools/custom_runtime.py`: the static gate re-run at load, whitelisted imports only, no I/O builtins, guarded module attributes, no string-driven attribute lookup); rewards, advantages and the strict program ratchet (`src/outer/reward/`, `src/outer/rounds/`); the executor runtime with the initial harness (`src/inner/`); the proposer update (`src/trainer/` plus the slime patch in `scripts/patches/`); contract tests that run the whole loop without GPUs (`pytest scripts/tests`); pinned dependencies (`config/dependencies.lock.json`); the dataset recipe (`config/dataset/`: sources, manifest, overlay, checksums); example lineage configs (`config/pipelines/`). The containment of generated code is defence in depth against a misbehaving proposer, not an OS sandbox: run rollout jobs in a container without credentials, as the Slurm templates do.

**Not included:** the task snapshot payload (`dataset/self_adapt_harness/`, rebuilt from public sources by `scripts/setup/setup_dataset.py`), the slime and NexAU checkouts (fetched and patched by `scripts/setup/`), model weights, the container image and Python environments (site payloads referenced from `config/local.env`), the campaign run directories and their manifests, and per-task evolved packages (released on Hugging Face instead). Two things the paper describes are not exposed as options here: a no-training *context* mode (the loop always trains the proposer between rounds), and a custom seed harness (`optimize()` starts from `src/inner/harness`).

---

## Further Reading

- **Paper:** [HarnessTTT: Learn to Optimize Harness at Test-Time for Scientific Discovery (arXiv:2610.17370)](https://arxiv.org/abs/2610.17370)
- **Project page:** [gray311.github.io/harnessrl](https://gray311.github.io/harnessrl/)
- **Hugging Face paper page:** [huggingface.co/papers/2610.17370](https://huggingface.co/papers/2610.17370)
- **Released harnesses:** [huggingface.co/datasets/gray311/HarnessRL](https://huggingface.co/datasets/gray311/HarnessRL)
- **Operations guide:** [`docs/pipeline_guide.md`](docs/pipeline_guide.md), [`scripts/pipeline/README.md`](scripts/pipeline/README.md), [`src/inner/README.md`](src/inner/README.md), [`src/outer/README.md`](src/outer/README.md), [`src/trainer/README.md`](src/trainer/README.md)

---

## Contributing

New tasks (a folder plus a manifest record, see *Bring your own task*) and new harness components (skills, tools, middleware under `src/inner/harness`) are welcome. Run `pytest scripts/tests` before opening a pull request.

---

## Citation

```bibtex
@misc{ma2026harnessttt,
  title         = {HarnessTTT: Learn to Optimize Harness at Test-Time for Scientific Discovery},
  author        = {Yingzi Ma and Yulong Cao and Nanxi Li and Fang Wu and Yizhe Zhang and James Zou and Yejin Choi and Chaowei Xiao},
  year          = {2026},
  eprint        = {2610.17370},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.17370}
}
```
