# src/trainer -- the proposer update (the only trainable parameters)

The proposer LoRA `phi` is trained with **slime's offline-GRPO stack** (official THUDM/slime pinned in `slime/`, FSDP
backend, Qwen3.5-9B LoRA).  `M0`, the executor, is never touched.  HarnessTTT owns the data contract; slime owns the
training loop.  The coupling is the on-disk replay format plus one rollout function:

| file | role |
|---|---|
| `grpo_to_replay.py` | `round/grpo_batch.jsonl` -> slime replay rows `{"messages", "tools", "metadata": {"advantage", "reward", "seed" (= proposal group), ...}}`; keeps only on-policy, training-eligible rows with a non-zero advantage; replays the exact H1 tool surface recorded at sampling; appends a terminal user turn so the last assistant action is inside the loss mask |
| `bound_replay.py` | mask-aware bounding to the token budget (18k): clips tool observations, then keeps the longest fitting *terminal* suffix (system + user + final actions) so the rewarded `submit_harness` action survives; uses the same mask generator slime trains with |
| `pad_replay.py` | pads the bounded replay to a multiple of the GPU count with zero-advantage duplicates (`metadata.padding = true`); slime otherwise injects metadata-less dummy samples |
| `qwen35_format.py` | renders parsed tool calls as the Qwen3.5 output XML, so training text equals eval text byte-for-byte |
| `qwen35_mask.py` | the multi-turn loss mask: only assistant turns closed by a later user/tool message carry gradient; the `<think>` wrapper is excluded |
| `slime_replay_rollout.py` | slime's `--rollout-function-path trainer.slime_replay_rollout.generate_rollout`: tokenises a replayed row, attaches its precomputed advantage as `sample.reward`, refuses rows without trainable tokens |
| `direct_ray_driver.py` | runs `slime/train.py` on the job-local Ray head (`train_worker.sh` starts it) |
| `merge_lora.sh` | merges the latest adapter into the frozen base with slime's `tools/merge_lora_adapter.py`; refuses an all-zero adapter |
| `tfpatch/sitecustomize.py` | transformers / accelerate compatibility shims for the training env (first on `PYTHONPATH`) |

Nothing here imports another training repository: slime is reached through its documented API
(`slime.utils.types.Sample`, `slime.utils.processing_utils`), and the slime patch in `scripts/patches/slime/` supplies the
seven behaviours the launcher checks for (`qwen3_5` loss-mask type, `--load-weights-only`, `--debug-train-only` without
sglang, three-value `create_training_models`, SFT-loss metric guard, collective LoRA resume, chunked log-probs) plus
`tools/merge_lora_adapter.py`.

## Per round

```
round r/grpo_batch.jsonl  --grpo_to_replay.py-->  training/round r/replay.jsonl        (chain.sh after_round, login/compute node, PyYAML needed)
replay.jsonl              --bound_replay.py---->  replay_bounded.jsonl                  (train_worker.sh, training env: transformers tokenizer)
                          --pad_replay.py------>  (in place, multiple of 8 rows)
                          --slime train.py----->  ckpt/iter_NNNNNNN/model/adapter_weights.pt
                          --merge_lora.sh------>  merged/  (full HF checkpoint + tokenizer files)  -> model_state.json -> next round's proposer
```

`scripts/pipeline/train_worker.sh` is the launcher (env contract in its header; `SAH9B_TRAIN_DRY_RUN=1` writes
`train_command.txt` without touching a GPU).  The recipe it runs, validated on the cluster (a canary job and the reward
ablation, 2026-09-23/24):

```
train.py --hf-checkpoint $BASE --load <previous adapter | $BASE> --ref-load $BASE --save ckpt --save-interval 1 --no-save-optim
  --rollout-function-path trainer.slime_replay_rollout.generate_rollout --prompt-data replay_bounded.jsonl
  --input-key messages --tool-key tools --metadata-key metadata --loss-mask-type qwen3_5 --rollout-shuffle
  --num-epoch 3 --rollout-batch-size <rows> --global-batch-size <rows> --n-samples-per-prompt 1 --micro-batch-size 1
  --loss-type policy_loss --advantage-estimator grpo --disable-rewards-normalization --debug-train-only
  --use-kl-loss --kl-loss-coef 0.1 --kl-loss-type low_var_kl
  --optimizer adam --lr 1e-5 --lr-decay-style constant --min-lr 0.0 --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --use-lora --lora-rank 64 --lora-alpha 128 --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --train-backend fsdp --gradient-checkpointing --attn-implementation flash_attention_2 --log-probs-chunk-size 128
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --seed 91000+round   [--load-weights-only --start-rollout-id 0 when resuming the adapter]
```

## Objective

slime's clipped policy-gradient surrogate (default clip 0.2) over the masked assistant tokens, 3 epochs; the per-sample
advantage from `outer/reward/rewards.py` is broadcast over the tokens (`--advantage-estimator grpo
--disable-rewards-normalization` = use it as is); a k3 KL penalty (`low_var_kl`, coefficient 0.1) to the frozen base;
LoRA-only parameters.  Zero-advantage padding rows contribute the KL term only.

## Caveats

- Rows with `|advantage| < 1e-6` are dropped by the converter; a round with fewer than 4 rows or no signal is skipped
  (`model_state.json` with `weights_updated: false`, proposer unchanged).
- H1 trajectories are 14k-32k tokens; the 18k terminal-preserving bound is what makes the 9B LoRA update fit on 8 GPUs.
- Standard GRPO advantages are unit-variance (|A| ~ 0.4-1.8) while `v3` keeps the reward scale (|A| ~ 0.03-0.4): at equal
  learning rate the GRPO arm takes 4-10x larger steps.  Report this when comparing arms.
