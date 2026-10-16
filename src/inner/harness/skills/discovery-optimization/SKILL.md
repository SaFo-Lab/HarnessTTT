---
name: discovery-optimization
description: Iteratively optimize a program's EVOLVE-BLOCK to maximize an automatic evaluator score, under a fixed evaluation budget. Use for construction, algorithm-speed, and heuristic discovery tasks scored by combined_score (higher is better) through the edit_solution / evaluate_solution / finish tools.
---

# Discovery optimization

One tool call per turn: `edit_solution` to stage a full new EVOLVE-BLOCK, then
`evaluate_solution` to score it. `combined_score` is higher-is-better; the best
version is retained automatically, so you never lose progress.

Start by reading what the score rewards. Restate the objective (maximize or
minimize the underlying quantity), the hard constraints the evaluator checks
(validity is 0 when a constraint is violated or the program errors), and the
per-evaluation time limit. Identify the entry function the evaluator calls — its
name and signature are fixed and must survive every edit.

Spend evaluations like a budget. Each edit must encode one concrete hypothesis,
not a guess: change the construction/algorithm, not cosmetics. After an
evaluation, treat the returned score, validity, and error text as the only
evidence — reason from them before the next edit.

Prefer **targeted SEARCH/REPLACE diffs** over rewriting the whole region: locate
the few lines that carry your idea and replace exactly those, so working code is
preserved. Reserve a full-block rewrite for a genuine structural change. Keep the
fixed entry function's inputs and outputs identical — only the internal
implementation may change.

Recover deliberately. `validity = 0` with an error means the program crashed or
violated a constraint: read the error, fix that specific cause, and keep the
rest. A valid but lower score means the idea was worse: revert to the approach
behind `best_so_far` and try a genuinely different direction rather than tuning
the losing one. Do not repeat an edit that already scored — change something
substantive every round.

Prefer explicit, deterministic constructions the evaluator can score quickly
over open-ended internal search that may hit the time limit. If a program is
allowed a bounded internal search, keep it well inside the per-evaluation
timeout with a safety margin. Use the exact public interface, packages, and
constraints named in the task; a lower-level substitute is not automatically
equivalent.

When `evaluations_left` is low, consolidate: make your remaining edits count on
the most promising line, then submit. When the budget is exhausted or you cannot
beat `best_so_far`, call `finish` with a one-line summary of the winning
approach and its score. Never fabricate a score — only a returned
`evaluate_solution` result counts.
