# scripts/pipeline -- the online-training loop

`python3 src/pipeline.py` is the entry point: it fills the templates below, submits the jobs and reports progress (see
`readme.md`).  `chain.sh` is the job-side alternative that chains the same jobs through sbatch epilogues.

After every collected round the proposer `M_phi` receives one GRPO LoRA update on that round's H1 trajectories and the
next round proposes with the updated (merged) checkpoint while the executor `M0` stays frozen.

```
round N  (worker_round.sh, 2 GPUs for K=8)                     train N  (train_worker.sh, 8 GPUs, ~15 min)
  vLLM pool <- PROPOSER_MODEL : propose K candidate H2           replay.jsonl (src/trainer/grpo_to_replay.py, advantages from collect)
  vLLM pool <- frozen base     : K executor rollouts               -> bounded to 18k tokens (terminal-preserving), padded to 8 rows
  collect (SAH_ADV=v3|grpo|..) : rewards, advantages, ratchet,     -> slime FSDP offline GRPO LoRA r64/a128, KL 0.1 vs base, lr 1e-5, 3 epochs
                                 beam-2 parents                    -> merged HF checkpoint + model_state.json
  chain.sh <run> after_round N -> submits train N                chain.sh <run> after_train N -> submits round N+1
```

| file | role |
|---|---|
| `worker_round.sh` | one outer round inside the SAH container: per-job vLLM port range; two protocol-probed pools (proposer checkpoint for propose, frozen base for rollouts); tri-state readiness (own / foreign / not ready); replica liveness guard with in-job redo; stall watchdog; `MPHI_PATH` provenance in `round.json`; `SAH_OVER_CAP_AS_INVALID` default 1 |
| `train_worker.sh` | one LoRA update: replay -> `bound_replay.py` -> `pad_replay.py` -> Ray + `direct_ray_driver.py` -> `slime/train.py` -> `merge_lora.sh` -> `model_state.json` (`model_path`, `adapter_path`, `weights_updated`); skips when < 4 rows or zero signal; `SAH9B_TRAIN_DRY_RUN=1` |
| `chain.sh` | job-side driver: `start 1` (round 1, frozen proposer), `after_round N`, `after_train N`; honours `ABL_STOP`, `GPU_CAP` / `HARNESSTTT_GPU_CAP`, `HARNESSTTT_MAX_JOBS`, `SAH9B_MAX_ROUND`; fills `templates/*.sbatch` into `<run>/gen/` |
| `build_bases.py` | beam-2 parents for the next round (collector beam when published, else head + best component-carrying candidate within 0.02 + by score); sibling-lineage harness/program migration (`<tag>_b`); eval-time guard; operator registry rollback marker |
| `templates/round.sbatch`, `templates/train.sbatch` | Slurm templates (`@@PLACEHOLDERS@@`, account/partition/QOS/mounts from `config/env.sh`) |
| `monitor.sh` | one line per state change per arm; infra warnings once per job |
| `status.py` | table: best accepted score after each round, updates, current job (`--root`, `--arms`, `--ref label=dir`) |
| `restart_from.sh` | clean restart of an arm from round N (archives, never deletes; restores the registry snapshot) |
| `vllm_protocol.py`, `audit_trajectories.py` | replica protocol probe (plain turn + parsed tool call + post-tool turn); trajectory retention audit |

Advantage implementations live in `src/outer/reward/rewards.py` and are selected with the run dir's `ARM` -> `SAH_ADV`:
`v3` (RLOO baseline + 0.3 softmax sharpening + plateau rescue), `grpo` (standard `(r - mean)/std` within the proposal
group), `v2`, `legacy`, `anchored`.

## Run-dir contract (one arm)

```
<run>/TASK                       cp26 | hadamard | <eft task id>
<run>/ARM                        v3 | grpo | v2 | legacy | anchored     -> SAH_ADV of the collect step
<run>/GPUS                       GPUs of a round job (2 for K=8);  <run>/K (8), <run>/H1_SEED (7311) optional
<run>/rounds/roundNNN/           collected rounds (round.json, grpo_batch.jsonl, round_summary.json, best_programs_after.json, next_bases*.json)
<run>/rounds/best_programs.json  strict-ratchet program registry;  rounds/task_feedback.json;  rounds/component_ledger.jsonl
<run>/training/roundNNN/         replay.jsonl, replay_bounded.jsonl, ckpt/, merged/, train_command.txt, model_state.json, TRAIN_COMPLETE | FAILED
<run>/out/  <run>/gen/  <run>/chain.log  <run>/JOB_ID  <run>/ABL_STOP (stop marker)  <run>/DONE  <run>/OVERCAP001_FROM_ROUND
```

Environment for `chain.sh` (after `config/env.sh`): `SAH9B_RUN_ROOT` (dir holding `GPU_CAP`; default `RUN_ROOT`),
`SAH9B_SAH_ROOT` (committed, clean tree the jobs execute from; default this tree), `SAH9B_ABL_ROOT` (default: parent of
`<run>`), `SAH9B_MAX_ROUND` (15), `SAH9B_TOKEN_BUDGET` (18000), `SAH9B_TRAIN_LR` (1e-5), `SAH9B_TRAIN_KL` (0.1),
`SAH9B_OVER_CAP_AS_INVALID` (1), `SAH9B_BRIEF_DIR`, `SAH9B_JOB_PREFIX` (`htt`), `SAH9B_SBATCH` (stub for dry runs),
`HARNESSTTT_HOST_PYTHON` (a Python with PyYAML for the replay conversion on the submitting node).

## Starting an arm

```bash
source config/env.sh; camp=$RUN_ROOT/<campaign>; run=$camp/hadamard_v3
mkdir -p "$run"; echo hadamard > $run/TASK; echo v3 > $run/ARM; echo 2 > $run/GPUS; echo 48 > $camp/GPU_CAP
SAH9B_RUN_ROOT=$camp bash scripts/pipeline/chain.sh "$run" start 1
```

For a paired ablation seed every arm with the same recorded round 1 (`rounds/round001` + `rounds/best_programs.json` =
`round001/best_programs_after.json` + `rounds/task_feedback.json`) and enter with `chain.sh <run> after_round 1`, so the
arms differ only in the credit assignment of the first update.

## Caveats (learned 2026-09-22 .. 24)

- **Never serve vLLM on fixed ports in a multi-job campaign.**  vLLM 0.17.1 binds with `SO_REUSEPORT`; two round jobs on
  one node using `127.0.0.1:8800/8801` silently shared the ports and one job's H1 turns were answered by the other job's
  proposer (4 of ~138 turns served by its own model).  The worker picks a free per-job range and requires pid alive +
  `GET /v1/models` root == the launched model (three consecutive foreign verdicts abort with exit 66).
- **OVERCAP-001.**  The strict ratchet examines only the top candidate; when its program exceeds the eval-time cap
  (`SAH_RATCHET_EVAL_TIME_FRAC` x task limit) the round used to be refused and every advantage zeroed, discarding a
  slower-but-under-cap improvement.  With `SAH_OVER_CAP_AS_INVALID=1` (default here) the collector re-labels over-cap
  candidates as invalid proposals before rewards (reward -1, excluded from `best_k` and the beam; recorded in
  `round_summary.json` `over_cap_candidates` / `rows[].over_cap`).  `chain.sh` records the first round it applied to in
  `<run>/OVERCAP001_FROM_ROUND`.
- Standard GRPO advantages are unit-variance (|A| ~ 0.4-1.8) while `v3` keeps the reward scale (|A| ~ 0.03-0.4): equal
  learning rates mean 4-10x larger steps for the GRPO arm.
- `grpo_to_replay.py` refuses trajectories that call a tool outside their recorded H1 surface unless
  `SAH_REPLAY_UNKNOWN_TOOL_CALLS=keep`; `chain.sh` keeps them (on-policy behaviour; dropping a row would change the group
  the advantage was normalised over).
- H1 trajectories are 14k-32k tokens; the 18k terminal-preserving bound makes the 9B LoRA update fit on 8 GPUs (4 GPUs
  with a CPU-offloaded reference failed with OOM / NCCL timeouts).
- slime injects metadata-less dummy samples when rows < GPUs (`KeyError: 'raw_reward'` on 7 rows / 8 GPUs);
  `pad_replay.py` pads with zero-advantage duplicates; a failed attempt's artefacts are archived to `attempt_failed_<stamp>/`.
- A round whose training the collector suppressed (0 replay rows) publishes a skip `model_state.json`
  (`weights_updated: false`) and the chain moves on; such rounds count as "no update" in the curves.
- The tree the jobs run from is frozen including its git HEAD (`propose` re-checks the revision); never commit into it while
  jobs are running -- use a worktree per campaign.
- `scancel` from an automated session may be denied; `ABL_STOP` in the run dir (or its parent) stops the chain between
  phases.
