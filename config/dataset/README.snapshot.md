# HarnessTTT task snapshot

This directory is the frozen evaluator snapshot the pipeline runs against.  It is built by
`scripts/setup/setup_dataset.py` from pinned public commits (`config/dataset/sources.json`); nothing in it is read from a
moving checkout at runtime.

| path | content | source |
|---|---|---|
| `MANIFEST_eft.json` | the task protocol: task records (`tasks`) and SimpleTES cross-references (`cross_referenced_tasks`) read by `inner.tasks.eft_task` | `config/dataset/MANIFEST_eft.json` |
| `raw/eft-aac2e79/benchmarks/{math,ADRS}/` | four math tasks and four ADRS tasks (evaluator, initial program, `config.yaml` with the task text) | evolution-fine-tuning `aac2e79` |
| `raw/eft-aac2e79/benchmarks/ADRS/*/evaluator/{expert-load.json,datasets/*.csv}` | EPLB workload and LLM-SQL tables | downloaded, sha256-checked (`DATA_PROVENANCE.json`) |
| `raw/simpletes-b7e0367/datasets/` | Hadamard maximal determinant (n=29), AHC039, AHC058 with the official testers and 150 cached inputs each | SimpleTES `b7e0367` |
| `raw/skydiscover-0d932b6/benchmarks/gpu_mode/` | TriMul and MLA-decode evaluators; `trimul_lb/` = TriMul under the GPU MODE leaderboard protocol (`shared_eval_lb.py`) | skydiscover `0d932b6` + overlay |
| `raw/ttt-discover-6c40e82/results/kernel-engineering/` | the kernels used as initial programs of the GPU tasks | TTT-Discover `6c40e82` |
| `raw/simpletes-47d3413/datasets/` | astrodynamics, scRNA denoising and ZAPBench task sources (optional `extra` tier) | SimpleTES `47d3413` |
| `runtime/skydiscover_min/`, `runtime/simpletes_min/` | import shims the frozen evaluators need (`EvaluationResult`; `simpletes.construction`, `simpletes.utils.text`) | overlay / SimpleTES |
| `runtime/ahc_native/bin/` | narrow `docker`, `time`, `sync`, `prlimit`, `g++-12` compatibility commands so the unmodified SimpleTES AHC runner executes inside a Slurm container without a Docker daemon (`PROVENANCE.md`) | overlay |

Scores: every evaluator returns `combined_score` (higher is better) and `validity`; `src/inner/tasks/aliases.py` `DISPLAY` maps the
score to the metric the paper reports (sum of radii, determinant ratio, microseconds, ...).

Paths in the manifest may contain `${VARIABLES}` (`SAH_KERNEL_EVAL_PYTHON`, `SAH_NEWTASK_OVERLAY`, see
`config/local.env.example`); the loader expands them from the environment and drops an entry whose variable is unset.

The `extra` tier copies the SimpleTES sources only; their data is prepared by the tasks' own scripts
(`open_problems_bio/setup.sh`, `zapbench/whole_brain_forecasting/setup.sh`; the ephemerides are downloaded by the builder).
The GPU tasks need a CUDA GPU that is not shared with vLLM (pipeline config `eval_gpus`); `mla_decode`'s cases were sized for
an H200 (141 GB).

Verify a snapshot at any time: `python3 scripts/setup/setup_dataset.py --verify --tiers core,gpu`.
