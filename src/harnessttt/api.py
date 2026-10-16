"""Single-call entry points over the HarnessTTT pipeline.

``optimize``      builds a ``pipeline.Config`` from keyword arguments and runs ``pipeline.Runner.run_loop`` -- exactly what
                  ``python3 src/pipeline.py <config.yaml>`` does -- then reads the run directory back into a ``Result``.
``run_harness``   runs one harness package (the built-in initial harness or an evolved package) on one task with any
                  OpenAI-compatible executor, by invoking ``python -m inner.cli.run_baseline`` with the matching flags.
``load_task``     returns the registry's view of a task: description, seed program, evaluator, budget hint.

The pipeline is a Slurm loop (round jobs serve the models with vLLM; training jobs update the proposer's LoRA with
slime).  Site paths come from ``config/env.sh`` + ``config/local.env`` as for the command-line entry point.  The loop
needs a source checkout of the repository (templates, workers, config): install with ``pip install -e .`` from a clone,
or point ``HARNESSTTT_ROOT`` at one.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlsplit

import inner
from inner.tasks.aliases import DISPLAY, TASK_IDS, fmt_metric

DEFAULT_ROUNDS = 15
INITIAL_HARNESS = Path(inner.__file__).resolve().parent / "harness"   # the initial executor harness (agent.yaml, system.md, skills/, tools/, middleware/)


def _find_root() -> Optional[Path]:
    """The repository checkout: ``HARNESSTTT_ROOT``, else the tree this file lives in (editable install)."""
    env = os.environ.get("HARNESSTTT_ROOT")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2])
    for root in candidates:
        if (root / "src" / "pipeline.py").is_file() and (root / "scripts" / "pipeline" / "templates").is_dir():
            return root.resolve()
    return None


ROOT = _find_root()                                    # None for a non-editable install without HARNESSTTT_ROOT
SRC = (ROOT / "src") if ROOT else Path(inner.__file__).resolve().parents[1]
if str(SRC) not in sys.path:                           # the inner/outer/trainer packages and pipeline.py live under src/
    sys.path.insert(0, str(SRC))


def _pipeline():
    """Import ``src/pipeline.py`` (the evolve loop) once, as the tests do."""
    mod = sys.modules.get("pipeline")
    if mod is None:
        if ROOT is None:
            raise RuntimeError("optimize()/load_task() need a source checkout of HarnessTTT: install with `pip install -e .` "
                               "from a clone, or set HARNESSTTT_ROOT to the checkout")
        spec = importlib.util.spec_from_file_location("pipeline", ROOT / "src" / "pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pipeline"] = mod
        spec.loader.exec_module(mod)
    return mod


def _served_name(model_path: Union[str, Path]) -> str:
    """The name vLLM serves a checkpoint under: the directory name, or its parent's when the leaf is a HF revision hash
    (``.../Qwen3.5-9B/c2022362...`` -> ``qwen3.5-9b``)."""
    path = Path(str(model_path))
    name = path.name
    if re.fullmatch(r"[0-9a-f]{40}", name) and path.parent.name:
        name = path.parent.name
    return name.lower()


def _unwrap_exit(exc: BaseException) -> RuntimeError:
    """The pipeline reports configuration problems with ``SystemExit(message)``; a library call raises instead of exiting."""
    code = getattr(exc, "code", None)
    return RuntimeError(str(code) if isinstance(code, str) else f"pipeline stopped with exit code {code}")


def _site_env() -> Dict[str, str]:
    """config/env.sh + config/local.env as a dict (empty when there is no checkout or sourcing fails)."""
    if ROOT is None or not (ROOT / "config" / "env.sh").is_file():
        return {}
    try:
        return _pipeline().load_env(ROOT)
    except SystemExit:
        return {}
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("config/env.sh + config/local.env failed to source: " + (exc.stderr or b"").decode(errors="replace")[-400:]) from None


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
        kind, c, _unit = DISPLAY.get(self.id, ("mul", 1.0, "score"))
        return fmt_metric(kind, c, combined_score)

    def evaluate(self, program: str, *, timeout_s: Optional[float] = None, python_exe: str = sys.executable):
        """Score ``program`` with the task's evaluator in a subprocess (same path as an executor rollout).
        Returns ``inner.evaluation.eval_runner.EvalOutcome`` (``combined_score``, ``validity``, ``error``, ``wall_s``)."""
        if self._eft is None:
            raise ValueError("Task.evaluate needs a registry task: build it with load_task() / examples.<task>.init_task()")
        from inner.evaluation.eval_runner import evaluate_program
        return evaluate_program(self._eft, program, timeout_s=timeout_s, python_exe=python_exe)


def load_task(name: str) -> Task:
    """``name`` is a pipeline alias (``cp26``, ``hadamard``, ``erdos``, ``ac1``, ``ac2``, ``trimul_lb`` ...) or a full task id.

    Raises ``ValueError`` for an unknown task, with a hint when the dataset snapshot itself is missing
    (``python3 scripts/setup/setup_dataset.py`` builds it)."""
    from inner.tasks import eft_task
    task_id = TASK_IDS.get(name, name)
    manifest = eft_task.DATASET_ROOT / "MANIFEST_eft.json"
    if not manifest.is_file():
        raise ValueError(f"task snapshot not found: {manifest} (build it with `python3 scripts/setup/setup_dataset.py` or set "
                         f"SAH_DATASET_ROOT); cannot load task {name!r}")
    try:
        t = eft_task.get_task(task_id)
    except KeyError:
        known = sorted({*TASK_IDS, *(x.task_id for x in eft_task.load_tasks())})
        raise ValueError(f"unknown task {name!r}; known aliases and ids: {known}") from None
    _kind, _c, unit = DISPLAY.get(task_id, ("mul", 1.0, "combined_score"))
    return Task(id=t.task_id, description=t.spec, initial_program_path=Path(t.initial_program_path),
                evaluator_path=Path(t.evaluator_path), task_dir=Path(t.task_dir), domain=t.domain,
                per_eval_timeout_s=float(t.per_eval_timeout_s), metric=unit, _eft=t)


# ----------------------------------------------------------------------------------------------------------------------
# optimize = the evolve loop
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    """What the loop left in its run directory.  With ``rounds_completed == 0`` (dry run, or a loop stopped before its
    first round collected) the ``best_*`` fields are ``None``."""
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
    kind, c, _unit = DISPLAY.get(task_id, ("mul", 1.0, "score"))
    return Result(run_dir=run, task_id=task_id, rounds_completed=n, done=bool(progress.get("done")),
                  best_score=best_score, best_metric=(fmt_metric(kind, c, best_score) if best_score is not None else None),
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
    pipeline_overrides   any other ``pipeline.Config`` field (``round_gpus``, ``train_gpus``, ``train_mem``, ``eval_timeout``,
                         ``adv``, ``job_prefix``, ``run_root``, ``freeze`` ...).

    Raises ``RuntimeError`` with the pipeline's message when the site settings are incomplete (``config/local.env``), the
    tree is dirty under ``freeze=True``, or another runner drives the same run directory.

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
    alias = next((a for a, tid in TASK_IDS.items() if tid == t.id), t.id)
    data: Dict[str, Any] = dict(campaign=campaign, name=name or f"{alias}_harnessttt", task=alias,
                                k=int(proposals_per_round), max_round=int(rounds), max_evals=int(max_metric_calls))
    if model_path:
        data["model_path"] = str(model_path)
        data["served_model"] = served_model or _served_name(model_path)
    elif served_model:
        data["served_model"] = served_model
    data.update({k: v for k, v in pipeline_overrides.items() if v is not None})
    try:
        cfg = p.Config.load(None, data)
        runner = p.Runner(cfg, quiet=quiet)
        if dry_run:
            script = runner.dry_run()
            return _result(runner, t.id, dry_run_script=Path(script))
        runner.run_loop()
    except SystemExit as exc:
        raise _unwrap_exit(exc) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("config/env.sh + config/local.env failed to source: " + (exc.stderr or b"").decode(errors="replace")[-400:]) from None
    return _result(runner, t.id)


# ----------------------------------------------------------------------------------------------------------------------
# run_harness = one harness package on one task with any OpenAI-compatible executor
# ----------------------------------------------------------------------------------------------------------------------
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def _looks_like_vllm(base_url: str) -> bool:
    """vLLM is what the pipeline serves locally; a hosted API (api.openai.com, ...) rejects vLLM-only request fields."""
    host = (urlsplit(base_url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".local") or host.replace(".", "").isdigit() or "." not in host


def run_harness_argv(harness_dir: Union[str, Path, None], task: Union[Task, str], *, base_url: str, model: str,
                     api_key: Optional[str] = None, max_evals: int = 20, out_dir: Union[str, Path] = "runs/transfer",
                     eval_python: str = sys.executable, seed_programs_file: Union[str, Path, None] = None,
                     vllm_extras: Optional[bool] = None, python: Optional[str] = None,
                     extra_args: Sequence[str] = ()) -> List[str]:
    """The ``python -m inner.cli.run_baseline`` command line that ``run_harness`` executes.

    python: the interpreter that runs the executor runtime (``None`` = this one; ``run_harness`` resolves ``SAH_PYTHON``).
    api_key is NOT part of the command line (it travels as ``OPENAI_API_KEY`` in the environment, see ``run_harness``).
    vllm_extras: send vLLM-only sampling fields (top_k, chat_template_kwargs).  ``None`` = yes for a local/vLLM endpoint,
    no for a hosted API host (``--no-vllm-extras``, which also moves the token cap to ``max_completion_tokens``)."""
    t = task if isinstance(task, Task) else load_task(task)
    if vllm_extras is None:
        vllm_extras = _looks_like_vllm(base_url)
    argv = [python or sys.executable, "-m", "inner.cli.run_baseline", "--ids", t.id, "--base-url", base_url, "--model", model,
            "--max-evals", str(int(max_evals)), "--eval-python", eval_python,
            "--out", str(Path(out_dir).resolve()), "--require-trajectory"]
    if harness_dir is not None:                       # None = the built-in initial harness
        argv += ["--harness-dir", str(Path(harness_dir).resolve())]
    if seed_programs_file is not None:
        argv += ["--seed-programs-file", str(Path(seed_programs_file).resolve())]
    if not vllm_extras:
        argv.append("--no-vllm-extras")
    argv += list(extra_args)
    return argv


def _executor_python(env: Dict[str, str], explicit: Optional[str]) -> str:
    """The interpreter with the NexAU runtime: an explicit ``python=``, else ``SAH_PYTHON`` (process env, then
    config/local.env), else this interpreter.  Probed with ``import nexau`` before anything is launched."""
    candidate = explicit or env.get("SAH_PYTHON") or _site_env().get("SAH_PYTHON") or sys.executable
    if not os.access(candidate, os.X_OK):
        if explicit:
            raise RuntimeError(f"python={candidate!r} is not executable")
        candidate = sys.executable
    probe_env = dict(env)
    nexau = _nexau_root(env)
    probe_env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + ([nexau] if nexau else []) + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        probe = subprocess.run([candidate, "-c", "import nexau, inner.cli.run_baseline"], env=probe_env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot run the executor interpreter {candidate}: {exc}") from None
    if probe.returncode != 0:
        tail = (probe.stderr.strip().splitlines() or ["?"])[-1][:200]
        raise RuntimeError(f"{candidate} cannot import the executor runtime ({tail}). Running a harness needs the Python 3.12 "
                           f"environment with NexAU: run scripts/setup/setup_deps.sh and scripts/setup/setup_runtime_env.sh, then set "
                           f"SAH_PYTHON in config/local.env or pass python=... to run_harness")
    return candidate


def _nexau_root(env: Dict[str, str]) -> Optional[str]:
    """``NEXAU_ROOT`` from the process environment, else from config/env.sh + config/local.env (where setup_deps.sh puts it)."""
    root = env.get("NEXAU_ROOT")
    if not root and ROOT is not None and (ROOT / "config" / "env.sh").is_file():
        try:
            root = _pipeline().load_env(ROOT).get("NEXAU_ROOT")
        except (SystemExit, subprocess.CalledProcessError):
            root = None
    return root if root and Path(root).is_dir() else None


def run_harness(harness_dir: Union[str, Path, None], task: Union[Task, str], *, base_url: str, model: str,
                api_key: str = "EMPTY", max_evals: int = 20, out_dir: Union[str, Path] = "runs/transfer",
                eval_python: Optional[str] = None, seed_programs_file: Union[str, Path, None] = None,
                vllm_extras: Optional[bool] = None, python: Optional[str] = None, extra_args: Sequence[str] = (),
                env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run ``harness_dir`` (a package with ``agent.yaml``; ``None`` = the initial harness) on ``task`` with the executor
    served at ``base_url`` (vLLM or any OpenAI-compatible API) and return the run's summary row.

    The executor runtime is the pinned NexAU checkout under Python 3.12: ``python`` (default ``SAH_PYTHON`` from the
    environment or ``config/local.env``, built by ``scripts/setup/setup_runtime_env.sh``) must import it, and ``NEXAU_ROOT``
    is put on its ``PYTHONPATH`` when the checkout is not installed.  ``eval_python`` runs the task's evaluator (default:
    the same interpreter).  ``api_key`` is passed through the environment (``OPENAI_API_KEY``), never on the command line.
    ``out_dir`` is resolved against the current directory; each call writes into its own
    ``<out_dir>/<stamp>/<timestamp>/{summary.json, summary.csv, results/<task>.json, provenance.json}``.

    Raises ``RuntimeError`` with the executor's own error text (from ``summary.json``) when the run fails.
    """
    out_root = Path(out_dir).resolve()
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    executor = _executor_python(run_env, python)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out_run = out_root / stamp
    argv = run_harness_argv(harness_dir, task, base_url=base_url, model=model, max_evals=max_evals, out_dir=out_run,
                            eval_python=eval_python or executor, seed_programs_file=seed_programs_file,
                            vllm_extras=vllm_extras, python=executor, extra_args=extra_args)
    nexau = _nexau_root(run_env)
    paths = [str(SRC)] + ([nexau] if nexau else []) + ([run_env["PYTHONPATH"]] if run_env.get("PYTHONPATH") else [])
    run_env["PYTHONPATH"] = os.pathsep.join(paths)
    run_env["OPENAI_API_KEY"] = api_key
    proc = subprocess.run(argv, env=run_env, cwd=str(ROOT) if ROOT else None, capture_output=True, text=True)
    summaries = sorted(out_run.glob("*/summary.json")) if out_run.is_dir() else []
    if not summaries:
        tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()[-5:]
        raise RuntimeError(f"run_baseline exited with {proc.returncode} and left no summary under {out_run}:\n" + "\n".join(tail))
    rows = json.loads(summaries[-1].read_text())
    row = rows[0] if rows else {}
    row["out_dir"] = str(summaries[-1].parent)
    if proc.returncode != 0 or row.get("mode") == "error" or row.get("error"):
        raise RuntimeError(f"run_baseline failed for {row.get('task_id', task)}: {row.get('error') or 'exit ' + str(proc.returncode)} "
                           f"(details: {row['out_dir']})")
    return row
