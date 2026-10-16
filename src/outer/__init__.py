"""Outer loop (M_phi + H1 -> new H2) for the self_adapt_harness project.

Terminology (plan.md §2):
  * Inner loop:  M0 + H2 -> solution + reward       (src/inner)
  * Outer loop:  M_phi + H1 -> candidate H2 packages (this package)

Per round:
  1. M_phi (Qwen3.5-9B + LoRA phi) under the FIXED proposer harness H1 samples
     K=8 candidate H2 specs (typed genome: prompts, skill, tool descriptions,
     sampling, iteration/middleware params — no arbitrary code).
  2. Each spec is validated fail-closed (must differ from the current best H2)
     and materialized into a full NexAU package:
     agent.yaml + prompt.md + tools/ + skills/ + middlewares/.
  3. Inner loop: frozen M0 runs each candidate on N=8 tasks, <=20 evals each
     -> 8x8 rewards.
  4. Rewards are normalized per task against the current best H2 baseline and
     group-normalized over the K candidates (GRPO advantages).
  5. GRPO updates ONLY phi (src/training). Best candidate becomes the next
     round's H2 baseline.

M0 is never updated (plan.md §0).
"""
