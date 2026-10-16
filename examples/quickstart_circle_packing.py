#!/usr/bin/env python3
"""README Quick Start as a script: optimize a harness for circle packing (n=26) with the frozen executor from config/local.env.

    python examples/quickstart_circle_packing.py                      # paper-style: 15 rounds, 8 proposals, 20 evaluator calls
    python examples/quickstart_circle_packing.py --rounds 1 --proposals 2 --evals 3 --name smoke   # smallest run (one Slurm round job)

The loop runs as Slurm jobs (round job: `--round-gpus`, default 2; proposer update: 8 GPUs) and needs a committed, clean
tree (jobs execute from a frozen worktree of HEAD) plus the site paths of config/local.env.
"""
import argparse
from pathlib import Path

import harnessttt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--proposals", type=int, default=8, help="harness proposals per round (K)")
    ap.add_argument("--evals", type=int, default=20, help="evaluator calls per rollout (B)")
    ap.add_argument("--model", default=None, help="local checkpoint dir of the frozen executor = proposer base (default: SAH_MODEL_PATH)")
    ap.add_argument("--campaign", default="quickstart")
    ap.add_argument("--name", default="cp26")
    ap.add_argument("--round-gpus", type=int, default=2)
    ap.add_argument("--job-prefix", default="htt", help="Slurm job-name prefix")
    ap.add_argument("--dry-run", action="store_true", help="write the first sbatch script without submitting")
    args = ap.parse_args()

    task = harnessttt.examples.circle_packing.init_task()
    print(f"task {task.id}: {task.description[:70]!r}... seed {task.initial_program_path.name}, evaluator {task.evaluator_path.name}")

    result = harnessttt.optimize(
        task=task,
        seed_harness=harnessttt.INITIAL_HARNESS,
        executor_lm=args.model,
        proposer_lm=args.model,
        rounds=args.rounds,
        proposals_per_round=args.proposals,
        max_metric_calls=args.evals,
        campaign=args.campaign,
        name=args.name,
        round_gpus=args.round_gpus,
        job_prefix=args.job_prefix,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("dry run; next job script:", result.dry_run_script)
        return
    print("Evolved harness:", result.best_harness_path)
    print(f"Best score: {result.best_metric} {task.metric} (combined_score {result.best_score})")
    if result.best_program:
        out = result.write_best_program(Path(result.run_dir) / "best_program.py")
        print("Best program:", out)


if __name__ == "__main__":
    main()
