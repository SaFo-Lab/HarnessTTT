"""Circle packing, n=26: place 26 circles in the unit square and maximize the sum of their radii.

The cheapest built-in task: the evaluator (``dataset/self_adapt_harness/raw/eft-aac2e79/benchmarks/math/circle_packing/
evaluator.py``) runs on CPU in seconds per call and needs only numpy.  Registry id ``eft__math__circle_packing``;
paper name CP26; native metric = sum of radii (combined_score x 2.635).
"""
from ..api import Task, load_task

TASK_ID = "eft__math__circle_packing"


def init_task() -> Task:
    """The task as the registry defines it: description, the seed program (with ``# EVOLVE-BLOCK-START/END`` markers),
    the evaluator path and the per-evaluation timeout."""
    return load_task(TASK_ID)
