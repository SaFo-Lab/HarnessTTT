# HarnessTTT

**Self-Adapt Harness (SAH) with an online-trained proposer, packaged as one tree.**  A frozen executor `M0`
(Qwen3.5-9B) runs discovery harnesses `H2` on hard optimisation tasks (circle packing, Hadamard maximal determinant, ...).
A proposer `M_phi` -- the same base model plus a cumulative LoRA -- writes `K` candidate harnesses per round through the
fixed proposer harness `H1`.  The executor's rollout scores become gap-normalised rewards and group advantages, slime
updates the LoRA, and the next round proposes with the merged checkpoint.  `M0`'s weights are never touched.

```
round r   scripts/pipeline/worker_round.sh   (2 GPUs for K=8, 2-3 h)      train r   scripts/pipeline/train_worker.sh   (8 GPUs, ~15 min)
  vLLM pool <- proposer_r   : H1 proposes K candidate H2      src/outer      grpo_batch.jsonl -> replay.jsonl               src/trainer/grpo_to_replay.py
  vLLM pool <- frozen base  : K executor rollouts, <=20 evals  src/inner      -> bounded to 18k tokens, padded to 8 rows    bound_replay.py, pad_replay.py
  collect: rewards, advantages (SAH_ADV), program ratchet,                    -> slime offline GRPO LoRA r64/a128, KL 0.1    slime/ + slime_replay_rollout.py
           beam-2 parents for the next round                                  -> merged checkpoint = proposer_{r+1}          merge_lora.sh, model_state.json
  chain.sh <run> after_round r  ->  submits train r                           chain.sh <run> after_train r  ->  submits round r+1
```

## Layout

| path | content |
|---|---|
| `config/` | `env.sh` (the one environment file every script sources), `local.env` (site payload paths for your cluster, git-ignored; template `local.env.example`), `dependencies.lock.json` (pins + checksums of everything not in git), `finch_targets.json` (published reference scores used by the gap-normalised reward), `task_text_registry.json` (task-text hashes; rounds fail closed on a mismatch), `pipeline.yaml` (the evolve loop: task, arm, K, GPUs, max_round, budgets), `campaign.yaml` + `examples/` (knob documentation of the older campaign driver) |
| `dataset/` | `self_adapt_harness/` evaluator snapshot: task specs, initial programs, evaluators, AHC shim (1.5 GB, git-ignored; `dataset/README.md`) |
| `scripts/pipeline/` | the online-training loop: `worker_round.sh`, `train_worker.sh`, `chain.sh`, `build_bases.py`, `templates/{round,train}.sbatch`, `monitor.sh`, `status.py`, `restart_from.sh`, `vllm_protocol.py`, `audit_trajectories.py` -- see `scripts/pipeline/README.md` |
| `scripts/setup/` | `setup_slime.sh` (official slime clone, pin, patch), `setup_deps.sh` (NexAU + Megatron-LM), `preflight.sh` |
| `scripts/patches/` | `slime/openclaw_rl_f48ac35_on_slime_58ee7489.patch` (OpenClaw-RL's Qwen3.5 / FSDP-LoRA changes + SAH's seven training patches against official slime), `nexau/nexau_transport.patch` |
| `scripts/tests/` | contract tests (`pytest scripts/tests`); `fake_slurm/` + `fixtures/` run the whole evolve loop without Slurm or GPUs (`test_pipeline.py`) |
| `scripts/tools/` | `build_task_text_registry.py` |
| `slime/` | **official [THUDM/slime](https://github.com/THUDM/slime)** at `58ee7489` (branch `harnessttt-pin`) with the patch applied -- the RL trainer for the proposer LoRA (git-ignored; recreated by `setup_slime.sh`) |
| `src/inner/` | executor side: EFT task registry, program editing, subprocess-isolated evaluation, H2 runtime and initial H2 NexAU package, `inner.cli.run_baseline` |
| `src/outer/` | proposer side: H1 NexAU package, proposal / materialisation / safety gates, rewards and advantages (`outer/reward/rewards.py`), strict program ratchet (`outer/reward/program_ratchet.py`), `outer.rounds.outer_round propose|collect` |
| `src/pipeline.py` | **the evolve loop**: `python3 src/pipeline.py` submits round and training jobs, reports progress live, resumes from the run dir (see Running the loop) |
| `src/trainer/` | proposer update: `grpo_to_replay.py`, `bound_replay.py`, `pad_replay.py`, `qwen35_format.py`, `qwen35_mask.py`, `slime_replay_rollout.py` (slime's `--rollout-function-path`), `direct_ray_driver.py`, `merge_lora.sh`, `tfpatch/` -- see `src/trainer/README.md` |

`PYTHONPATH` for the SAH code is `src` (packages `inner`, `outer`, `trainer`) plus `NEXAU_ROOT`; slime and Megatron-LM are
added only for the training job.  Every hard-coded path of the original repo now resolves relative to this tree
(`dataset/`, `config/`) or comes from `config/env.sh`.

## Setup (fresh clone)

```bash
cp config/local.env.example config/local.env      # edit: model, image, envs, NexAU, Megatron-LM, RUN_ROOT, Slurm account
bash scripts/setup/setup_slime.sh                 # git clone https://github.com/THUDM/slime.git slime; checkout 58ee7489; apply patch; verify needles
bash scripts/setup/setup_deps.sh                  # NexAU 09e747b + transport patch, Megatron-LM core_v0.16.0 (skipped when local.env points at existing checkouts)
rsync -a <snapshot>/self_adapt_harness/ dataset/self_adapt_harness/    # manifest sha256 in config/dependencies.lock.json
bash scripts/setup/preflight.sh                   # HARNESSTTT_HOST_PYTHON=<python with PyYAML> enables the import probe on a login node
git commit                                        # jobs refuse to start from a dirty tree and record the commit in every round
```

`config/local.env` is git-ignored (copy it from `local.env.example`); when it points at existing checkouts, `setup_*` only verify.  The container image, the three Python
environments (SAH runtime, vLLM 0.17.1, torch 2.9.1 + transformers 5.12.1 training env) and the Qwen3.5-9B checkpoint
are site payloads referenced from `local.env`; they are not copied into the tree.

## Running the loop (one lineage = one run dir)

```bash
# 1. describe the lineage (task, advantage arm, K, GPUs, max_round, ...): copy one of config/pipelines/*.yaml or edit config/pipeline.yaml
cp config/pipelines/hadamard_v3.yaml config/pipelines/my_run.yaml
# 2. start (or resume) the evolve loop; progress is printed live and written to <run>/pipeline.log + progress.json
python3 src/pipeline.py config/pipelines/my_run.yaml                 # blocking; use tmux/nohup for long campaigns
python3 src/pipeline.py config/pipelines/my_run.yaml status          # table: per round valid/8, best candidate, incumbent, accepted, update, durations
python3 src/pipeline.py config/pipelines/my_run.yaml stop            # ABL_STOP: stop between jobs (a running job finishes); re-run to resume
python3 src/pipeline.py config/pipelines/my_run.yaml --dry-run       # write the next sbatch into <run>/gen without submitting
python3 src/pipeline.py                                              # no positional config -> config/pipeline.yaml; any key: --max-round 3 --name test
```

`src/pipeline.py` drives one lineage from the login node: it submits a round job (`scripts/pipeline/worker_round.sh` inside
the container: proposer pool -> propose -> pool swap to the frozen base -> rollouts -> collect), waits while reporting the
phase (vLLM up, replicas probed, candidates materialised, rollouts k/n, collect), builds the beam-2 parents and the training
replay, submits the 8-GPU training job (`train_worker.sh`: bound -> pad -> slime LoRA -> merge), reads
`training/roundNNN/model_state.json` and submits the next round with the merged checkpoint, until `max_round` rounds are
collected (`<run>/DONE`).  Fewer than 4 trainable rows skip the update (proposer unchanged); a round job that ends without
collecting is archived (`roundNNN_failed_<stamp>`) and re-submitted up to `max_round_retries` times.  Before every
submission it enforces `HARNESSTTT_GPU_CAP` (all of the user's jobs, pending included) and `HARNESSTTT_MAX_JOBS`, waiting
for capacity instead of exceeding it.  State lives in the run dir, so a killed runner resumes and re-attaches to a running
job; `pipeline.lock` prevents two runners on one lineage.

By default (`freeze: true`) the jobs execute from an immutable `git worktree` of the committed HEAD under
`$RUN_ROOT/<campaign>/worktrees/htt_<sha>` (with `slime` and `dataset/self_adapt_harness` linked in), so this tree stays
editable while a campaign runs; the tree must be committed and clean when the loop starts.  The job-side alternative
(`scripts/pipeline/chain.sh`, sbatch epilogues chaining round -> train -> round without a login-node process) remains
available and is documented in `scripts/pipeline/README.md`.

## Protocol constants (what the pipeline runs today)

| stage | setting |
|---|---|
| proposer | Qwen3.5-9B + cumulative LoRA, `K=8` candidates per round, H1 max 36 iterations, seed `7311 + round - 1`; parents = beam of 2 dealt round-robin (round 1: the initial H2 package) |
| executor | frozen Qwen3.5-9B; **20 evaluator calls per trajectory**, 600 s per evaluation, 3 h wall per rollout |
| reward | `task_reward_v2`: gap to the incumbent, normalised by the distance to the published reference (`config/finch_targets.json`, `sota_combined`); `3 tanh(g/3)` above the incumbent, linear down to -1 below; invalid / failed / over-cap candidates -1 |
| advantage | proposal groups (same parent + prompt): `K_g = 8` in round 1, 4 afterwards.  `v3`: RLOO + 0.3 centred softmax (T = valid range / 4) + tie collapse (1e-3) + historical-frontier rescue (0.3).  `grpo`: `(r - mean)/(std + 1e-6)`, zero-variance group -> 0 |
| ratchet | strict single-program registry; a program is adopted only if its evaluation time stays under 0.8 x limit (Hadamard 350 s -> 280 s, others 600 s -> 480 s); with `SAH_OVER_CAP_AS_INVALID=1` (default) an over-cap program is an invalid proposal (-1) instead of a refused round |
| update | slime offline GRPO on FSDP, 8 GPUs: LoRA r64 / alpha 128 on q,k,v,o,gate,up,down; lr 1e-5 constant; 3 epochs; batch = all rows; KL 0.1 (`low_var_kl`) to the frozen base; episodes bounded to 18k tokens (terminal-preserving) and padded to a multiple of 8 with zero-advantage rows; adapter resumed with `--load-weights-only`; merged HF checkpoint served by vLLM |

## Rules that keep results valid

- **Clean tree.** Jobs execute from `SAH9B_SAH_ROOT` (default: this tree) and refuse tracked modifications or untracked
  files; `propose` records the commit.  Never edit `src/` while jobs run -- use `git worktree add` per campaign.
- **Fixed ports are forbidden.** vLLM binds with `SO_REUSEPORT`; two jobs on one node sharing `127.0.0.1:8800` silently
  split one proposer's traffic across two models.  `worker_round.sh` picks a per-job port range and accepts a replica only
  when its pid is alive and `GET /v1/models` reports the model root this job launched.
- **GPU cap and queue depth** are enforced in `chain.sh`; canonical jobs use the low QOS (`preflight.sh` fails otherwise).
- **Executor stays frozen**: rollouts always run on `SAH_MODEL_PATH`; only the proposer pool changes.

## Verification performed at migration (2026-09-24/25)

`scripts/setup/preflight.sh` passed (slime pin + 8 needles, dataset manifest, NexAU pin + patch, Megatron-LM, payloads,
QOS, imports of `inner`/`outer`/`trainer` with 15 tasks from `dataset/`); `outer.rounds.outer_round collect` replayed on
recorded ablation rounds through this tree reproduced `grpo_batch.jsonl`, `round_summary.json`, `best_programs_after.json`
and the ratchet audit exactly; `trainer/grpo_to_replay.py` reproduced the recorded training inputs byte-for-byte;
`chain.sh` dry runs (stub `sbatch`) generated valid round/train scripts; `train_worker.sh --dry-run` reproduced the
recorded slime command up to the intended path/module changes.  Details in the migration report of 2026-09-25.

## Origin

Code from the SAH research repository: `src/inner`, `src/outer`, `src/training` (-> `trainer`) at the 2026-09-24 state (v11 + OVERCAP-001); the pipeline from the online-training branch with the reward-ablation launcher refinements (OVERCAP handling, monitor v2, status, restart).  slime: official THUDM/slime `58ee7489`; the patch was derived from Gen-Verse/OpenClaw-RL `f48ac35` (which is that commit plus OpenClaw's modifications) and SAH's seven training patches.
