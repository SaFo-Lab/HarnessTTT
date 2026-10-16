"""HarnessTTT: learn to optimize the harness of a frozen executor at test time.

Thin Python entry points over the existing pipeline (``src/pipeline.py``) and executor runtime
(``inner.cli.run_baseline``).  Nothing here changes how a campaign runs or what it writes.

    import harnessttt
    task = harnessttt.examples.circle_packing.init_task()
    result = harnessttt.optimize(task=task, seed_harness=harnessttt.INITIAL_HARNESS, ...)
"""
from .api import (  # noqa: F401
    DEFAULT_ROUNDS,
    INITIAL_HARNESS,
    ROOT,
    Result,
    Task,
    load_task,
    optimize,
    run_harness,
    run_harness_argv,
)
from . import examples  # noqa: F401

__version__ = "0.1.0"
__all__ = ["DEFAULT_ROUNDS", "INITIAL_HARNESS", "ROOT", "Result", "Task", "load_task", "optimize",
           "run_harness", "run_harness_argv", "examples", "__version__"]
