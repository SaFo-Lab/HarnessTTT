# src/outer — the outer loop (M_phi + H1 → new H2)

> RESTRUCTURE-001: files now live in themed subpackages — see `../README.md` for the map; old import paths still work via shims.

**Outer loop:** `M_phi + H1 → K candidate H2`. **Inner loop** (src/inner):
`M0 + H2 → solution + reward`. RL (GRPO) updates **only** `phi` — M0 stays
frozen forever.

## One round

```
1. propose   M_phi (Qwen3.5-9B ⊕ LoRA phi) + fixed H1 prompt
             --K samples--> K=8 private H2 filesystem edits
             (cat agent.yaml → inspect mounted files → edit → validate → submit)
             → fail-closed validation (must differ from current best H2;
               duplicates/invalid kept as records with fixed reward -1)
             → recompile each valid workspace into a FULL NexAU package:
               candNN/{agent.yaml, prompt.md, tools/, skills/, middlewares/}
2. rollouts  frozen M0 + each candidate H2 → one task × ≤budget evals.
             With `reward.impl=paired`, also run the incoming parent H2 with
             the exact same task/program/budget/model/decode seed.
3. collect   paired proposer reward = normalized(candidate_score -
             parent_control_score). This prevents an executor discovery already
             available from the public task or sampling seed from being credited
             to H1. GRPO advantages remain task-local over the K candidates.
             → grpo_batch.jsonl (+ round_summary.json)
4. train     src/trainer: convert → bound/pad → slime offline-GRPO → merged M_phi (scripts/pipeline/train_worker.sh)
5. iterate   best candidate = next round's BASE_HARNESS; merged ckpt = next
             round's proposer
```

## The H2 genome

M_phi edits a complete private copy of H2. `agent.yaml` is the mount graph;
`prompt.md` is the proposer-owned executor system prompt; generated tool,
skill, and middleware files are inherited and editable in place. Adding or
deleting a component requires a consistent file, mount, and prompt change.
The workspace validator parses these files back into a typed semantic spec,
checks safety and inheritance, then `materialize.py` deterministically rebuilds
the runnable package. Core runtime bindings and the evaluation budget remain
externally fixed.

## Files

| File | Role |
|---|---|
| **`harness/`** | **H1 — the FIXED proposer harness**: read-only shell-shaped inspection plus H2 file edit/delete/validate/submit tools. Never mutated during training; hashed for provenance. |
| `h2_workspace.py` | candidate-isolated H2 filesystem, safe path handling, mount/file/prompt checks, semantic extraction, canonical compile validation |
| `harness_spec.py` | spec schema, fail-closed validation, canonical hash, base-spec extraction, diff-vs-base |
| `h1.py` | round-context builder (user message = base spec + per-task baseline) + H1 package hash |
| `propose_session.py` | ProposeSession state + contextvar bridge behind H1's filesystem tools |
| `propose.py` | run the H1 agent K times (threaded across replicas) → CandidateRecords (+ full trajectories for GRPO) |
| `materialize.py` | effective spec → full candidate package (matches `src/inner/harness_candidate/candNN` scaffold) |
| `rewards.py` | per-task normalized rewards vs `results/baseline_h2_20ev.json`, GRPO group advantages |
| `outer_round.py` | CLI: `propose` / `collect` |

Both loops' harnesses are declarative NexAU packages: H1 = `src/outer/harness/`
(fixed), H2 = `src/inner/harness/` (round-1 base) and `round00r/candNN/`
(generated candidates). GRPO trains on the proposer's full H1 trajectories
(inspect→edit→validate→submit tool calls), which is exactly the multi-turn format
slime masks and trains on.

Round artifacts land in `$RUN_ROOT/self_adapt_harness/outer/round00r/`:
`round.json, prompt.json, responses.json, candNN/…, rollouts/candNN/…,
grpo_batch.jsonl, round_summary.json`.

## Run

```bash
source config/env.sh   # HarnessTTT root
mkdir -p "$LOG_ROOT/slurm"
bash scripts/pipeline/chain.sh <run> start 1      # round 1 (frozen proposer); see scripts/pipeline/README.md
# then: src/trainer/README.md  (GRPO on phi via slime, merge, round 2 -- automated by chain.sh)
```

Default rollout task set (8, diverse + CPU-cheap + non-saturated):
circle_packing, hadamard, erdos, prism, txn_scheduling, eplb,
convolve2d_full_fill, psd_cone_projection.

## Round-1 note

In round 1 phi = 0, so M_phi ≡ M0 weights and one served checkpoint plays both
roles. From round 2 (trained phi) the worker must serve the merged M_phi for
`propose` and the frozen base for the inner loop — flagged TODO(round2) in
`scripts/_outer_round_worker.sh`.
