"""Inner loop (M0 + H2) for the self_adapt_harness project.

Terminology (plan.md §2):
  * Inner loop:  M0 + H2 -> solution + reward   (this package)
  * Outer loop:  M_phi + H1 -> new H2           (the proposer; not built here)

M0 = a permanently frozen Qwen3.5-9B executor. H2 = the problem-solving /
discovery harness that M0 runs to solve a task — a declarative NexAU agent
package under ``harness/`` (agent.yaml + tools + skills + middlewares). This
package holds the *initial, hand-written* H2 plus the inner-loop runner that
executes M0 + H2 over the EFT held-out tasks and returns (best solution, reward).
The outer loop will later train M_phi (via its fixed harness H1) to *generate*
better H2 packages; that is a separate component.

Run as a package: ``python -m inner.run_baseline`` with ``src/`` on the path.
"""
