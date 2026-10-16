"""Single-call entry points over the HarnessTTT pipeline.

``optimize``      builds a ``pipeline.Config`` from keyword arguments and runs ``pipeline.Runner.run_loop`` -- exactly what
                  ``python3 src/pipeline.py <config.yaml>`` does -- then reads the run directory back into a ``Result``.
``run_harness``   runs one harness package (the built-in initial harness or an evolved package) on one task with any
                  OpenAI-compatible executor, by invoking ``python -m inner.cli.run_baseline`` with the matching flags.
``load_task``     returns the registry's view of a task: description, seed program, evaluator, budget hint.

The pipeline is a Slurm loop (round jobs serve the models with vLLM; training jobs update the proposer's LoRA with
slime).  Site paths come from ``config/env.sh`` + ``config/local.env`` as for the command-line entry point.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

ROOT = Path(__file__).resolve().parents[2]            # the repository (src/harnessttt/api.py -> ROOT)
SRC = ROOT / "src"
INITIAL_HARNESS = SRC / "inner" / "harness"            # the initial executor harness (agent.yaml, system.md, skills/, tools/, middleware/)
DEFAULT_ROUNDS = 15

if str(SRC) not in sys.path:                           # the inner/outer/trainer packages and pipeline.py live under src/
    sys.path.insert(0, str(SRC))


def _pipeline():
    """Import ``src/pipeline.py`` (the evolve loop) once, as the tests do."""
    mod = sys.modules.get("pipeline")
    if mod is None:
        spec = importlib.util.spec_from_file_location("pipeline", SRC / "pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pipeline"] = mod
        spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Task:
    """A discovery task as the registry defines it: description, ONE seed program with a delimited editable region,
    a black-box evaluator, the optimization direction and a per-evaluation timeout.  Built from ``inner.tasks.eft_task``."""
    id: str
    description: str
    initial_program_path: Path
    evaluator_path: Path
    task_dir: Path
    domain: str
    direction: str = "maximize"            # the evaluator's combined_score is higher-is-better for every registered task
    per_eval_timeout_s: float = 360.0
    metric: str = "combined_score"
    _eft: Any = field(default=None, repr=False, compare=False)

    @property
    def initial_program(self) -> str:
        return self.initial_program_path.read_text()

    def native_metric(self, combined_score: float) -> float:
        """Map the evaluator's combined score to the metric the paper reports (sum of radii, µs, ...)."""
        p = _pipeline()
        kind, c, _unit = p.DISPLAY.get(self.id, ("mul", 1.0, "score"))
        return p.fmt_metric(kind, c, combined_score)

    def evaluate(self, program: str, *, timeout_s: Optional[float] = None, python_exe: str = sys.executable):
        """Score ``program`` with the task's evaluator in a subprocess (same path as an executor rollout).
        Returns ``inner.evaluation.eval_runner.EvalOutcome`` (``combined_score``, ``validity``, ``error``, ``wall_s``)."""
        from inner.evaluation.eval_runner import evaluate_program
        return evaluate_program(self._eft, program, timeout_s=timeout_s, python_exe=python_exe)


def load_task(name: str) -> Task:
    """``name`` is a pipeline alias (``cp26``, ``hadamard``, ``erdos``, ``ac1``, ``ac2``, ``trimul_lb`` ...) or a full task id."""
    p = _pipeline()
    from inner.tasks.eft_task import get_task
    task_id = p.TASK_IDS.get(name, name)
    t = get_task(task_id)
    _kind, _c, unit = p.DISPLAY.get(task_id, ("mul", 1.0, "combined_score"))
    return Task(id=t.task_id, description=t.spec, initial_program_path=Path(t.initial_program_path),
                evaluator_path=Path(t.evaluator_path), task_dir=Path(t.task_dir), domain=t.domain,
                per_eval_timeout_s=float(t.per_eval_timeout_s), metric=unit, _eft=t)


# ----------------------------------------------------------------------------------------------------------------------
# optimize = the evolve loop
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    run_dir: Path
    task_id: str
    rounds_completed: int
    done: bool
    best_score: Optional[float]                 # evaluator combined_score of the retained program
    best_metric: Optional[float]                # the same in the paper's native metric (sum of radii, µs, ...)
    best_harness_path: Optional[Path]           # the evolved harness package the loop retained (agent.yaml, prompt.md, skills/, tools/ ...)
    best_program: Optional[str]                 # text of the retained program
    best_programs_json: Path                    # strict-ratchet program registry (rounds/best_programs.json)
    progress_json: Path                         # per-round log the runner maintains
    dry_run_script: Optional[Path] = None       # set when optimize(..., dry_run=True): the next sbatch script, not submitted

    def write_best_program(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if self.best_program is None:
            raise ValueError("no retained program yet")
        path.write_text(self.best_program)
        return path


def _result(runner, task_id: str, dry_run_script: Optional[Path] = None) -> Result:
    run = Path(runner.run)
    progress = json.loads((run / "progress.json").read_text()) if (run / "progress.json").is_file() else {}
    rounds = sorted(progress.get("rounds", []), key=lambda r: r["round"])
    n = rounds[-1]["round"] if rounds else 0
    best_score = rounds[-1].get("incumbent") if rounds else None
    reg = run / "rounds" / "best_programs.json"
    best_program = None
    if reg.is_file():
        ent = (json.loads(reg.read_text()) or {}).get(task_id) or {}
        best_program = ent.get("program")
        if best_score is None and ent.get("score") is not None:
            best_score = float(ent["score"])
    pkg = None
    for k in range(n, 0, -1):
        nb = run / "rounds" / f"round{k:03d}" / "next_bases.json"
        if nb.is_file():
            ent = (json.loads(nb.read_text()) or {}).get(task_id) or {}
            if ent.get("package"):
                pkg = Path(ent["package"])
                break
    p = _pipeline()
    kind, c, _unit = p.DISPLAY.get(task_id, ("mul", 1.0, "score"))
    return Result(run_dir=run, task_id=task_id, rounds_completed=n, done=bool(progress.get("done")),
                  best_score=best_score, best_metric=(p.fmt_metric(kind, c, best_score) if best_score else None),
                  best_harness_path=pkg, best_program=best_program, best_programs_json=reg,
                  progress_json=run / "progress.json", dry_run_script=dry_run_script)


def optimize(task: Union[Task, str], *, seed_harness: Union[str, Path] = INITIAL_HARNESS,
             executor_lm: Optional[str] = None, proposer_lm: Optional[str] = None,
             rounds: int = DEFAULT_ROUNDS, proposals_per_round: int = 8, max_metric_calls: int = 20,
             campaign: str = "harnessttt", name: Optional[str] = None, served_model: Optional[str] = None,
             dry_run: bool = False, quiet: bool = False, **pipeline_overrides: Any) -> Result:
    """Run the HarnessTTT evolve loop on ``task`` and return the evolved harness and its score.

    task                 a ``Task`` from ``load_task`` / ``harnessttt.examples.<task>.init_task()``, or its alias / id
    seed_harness         path of the initial executor harness.  The loop starts from the built-in initial harness
                         (``harnessttt.INITIAL_HARNESS``); any other path raises ``NotImplementedError`` (see notes).
    executor_lm          local checkpoint directory of the frozen executor (HF snapshot).  ``None`` = ``SAH_MODEL_PATH``
                         from ``config/local.env``.  The executor is never updated.
    proposer_lm          checkpoint directory of the proposer's base.  The pipeline serves proposer and executor from the
                         SAME base checkpoint and trains a LoRA on the proposer only, so this must equal ``executor_lm``.
    rounds               outer rounds (``max_round``); proposals_per_round = ``k``; max_metric_calls = evaluator calls per
                         rollout (``max_evals``).
    pipeline_overrides   any other ``pipeline.Config`` field (``round_gpus``, ``train_gpus``, ``eval_timeout``, ``adv``,
                         ``job_prefix``, ``run_root``, ``freeze`` ...).

    Notes: this is a Slurm pipeline; a round job uses ``round_gpus`` GPUs (vLLM replicas + evaluator GPUs) and a training
    job ``train_gpus`` GPUs.  ``freeze=True`` (default) runs jobs from an immutable worktree of the committed HEAD, so the
    tree must be committed and clean.  A custom seed harness is not a pipeline parameter today: round 1 always proposes
    from ``src/inner/harness`` (the package released as ``harnesses/initial``).
    """
    p = _pipeline()
    t = task if isinstance(task, Task) else load_task(task)
    seed = Path(seed_harness).resolve()
    if seed != INITIAL_HARNESS.resolve():
        raise NotImplementedError(
            f"optimize() can only start from the built-in initial harness {INITIAL_HARNESS}; got {seed}. "
            "Round 1 of the pipeline always proposes from src/inner/harness (pipeline.py: do_round(1, ..., bases=None)).")
    if executor_lm is not None and proposer_lm is not None and str(executor_lm) != str(proposer_lm):
        raise ValueError("the pipeline serves the proposer and the frozen executor from one base checkpoint (model_path); "
                         "pass the same path for executor_lm and proposer_lm (the LoRA is applied to the proposer only)")
    model_path = executor_lm or proposer_lm
    alias = next((a for a, tid in p.TASK_IDS.items() if tid == t.id), t.id)
    data: Dict[str, Any] = dict(campaign=campaign, name=name or f"{alias}_harnessttt", task=alias,
                                k=int(proposals_per_round), max_round=int(rounds), max_evals=int(max_metric_calls))
    if model_path:
        data["model_path"] = str(model_path)
        data["served_model"] = served_model or Path(str(model_path)).name.lower()
    elif served_model:
        data["served_model"] = served_model
    data.update({k: v for k, v in pipeline_overrides.items() if v is not None})
    cfg = p.Config.load(None, data)
    runner = p.Runner(cfg, quiet=quiet)
    if dry_run:
        script = runner.dry_run()
        return _result(runner, t.id, dry_run_script=Path(script))
    runner.run_loop()
    return _result(runner, t.id)


# ----------------------------------------------------------------------------------------------------------------------
# run_harness = one harness package on one task with any OpenAI-compatible executor
# ----------------------------------------------------------------------------------------------------------------------
def run_harness_argv(harness_dir: Union[str, Path, None], task: Union[Task, str], *, base_url: str, model: str,
                     api_key: str = "EMPTY", max_evals: int = 20, out_dir: Union[str, Path] = "runs/transfer",
                     eval_python: str = sys.executable, seed_programs_file: Union[str, Path, None] = None,
                     extra_args: Sequence[str] = ()) -> List[str]:
    """The ``python -m inner.cli.run_baseline`` command line that ``run_harness`` executes."""
    t = task if isinstance(task, Task) else load_task(task)
    argv = [sys.executable, "-m", "inner.cli.run_baseline", "--ids", t.id, "--base-url", base_url, "--model", model,
            "--api-key", api_key, "--max-evals", str(int(max_evals)), "--eval-python", eval_python, "--out", str(out_dir),
            "--require-trajectory"]
    if harness_dir is not None:                       # None = the built-in initial harness
        argv += ["--harness-dir", str(Path(harness_dir).resolve())]
    if seed_programs_file is not None:
        argv += ["--seed-programs-file", str(seed_programs_file)]
    argv += list(extra_args)
    return argv


def run_harness(harness_dir: Union[str, Path, None], task: Union[Task, str], *, base_url: str, model: str,
                api_key: str = "EMPTY", max_evals: int = 20, out_dir: Union[str, Path] = "runs/transfer",
                eval_python: str = sys.executable, seed_programs_file: Union[str, Path, None] = None,
                extra_args: Sequence[str] = (), env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run ``harness_dir`` (a package with ``agent.yaml``; ``None`` = the initial harness) on ``task`` with the executor
    served at ``base_url`` (vLLM or any OpenAI-compatible API) and return the run's summary row.

    The executor runtime needs the pinned NexAU checkout importable: set ``NEXAU_ROOT`` (``config/local.env``) or put it on
    ``PYTHONPATH``.  ``eval_python`` is the interpreter that runs the task's evaluator (numpy/scipy ... for the math tasks).
    Writes ``<out_dir>/<timestamp>/{summary.json, summary.csv, results/<task>.json, provenance.json}``.
    """
    argv = run_harness_argv(harness_dir, task, base_url=base_url, model=model, api_key=api_key, max_evals=max_evals,
                            out_dir=out_dir, eval_python=eval_python, seed_programs_file=seed_programs_file, extra_args=extra_args)
    run_env = dict(os.environ)
    paths = [str(SRC)] + ([run_env["NEXAU_ROOT"]] if run_env.get("NEXAU_ROOT") else []) + \
            ([run_env["PYTHONPATH"]] if run_env.get("PYTHONPATH") else [])
    run_env["PYTHONPATH"] = os.pathsep.join(paths)
    if env:
        run_env.update(env)
    before = {d.name for d in Path(out_dir).glob("*")} if Path(out_dir).is_dir() else set()
    subprocess.run(argv, check=True, env=run_env, cwd=str(ROOT))
    new = sorted(d for d in Path(out_dir).glob("*") if d.is_dir() and d.name not in before)
    if not new:
        raise RuntimeError(f"run_baseline left no output under {out_dir}")
    rows = json.loads((new[-1] / "summary.json").read_text())
    row = rows[0] if rows else {}
    row["out_dir"] = str(new[-1])
    return row
