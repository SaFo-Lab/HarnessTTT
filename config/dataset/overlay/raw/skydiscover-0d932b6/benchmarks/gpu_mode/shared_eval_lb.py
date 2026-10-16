"""GPU MODE *leaderboard* evaluation protocol for the gpu_mode kernel tasks.

This is a faithful port of the kernelbot/popcorn `eval.py` that TTT-discover ran on Modal H100s
(test-time-training/discover @6c40e82: examples/gpu_mode/lib/bioml/trimul/eval.py, mode="leaderboard", and
examples/gpu_mode/lib/libkernelbot/submission.py::compute_score with ranking_by="geom"):

  1. every case in reference.TEST_CASES must pass check_implementation (the kernelbot "test" run);
  2. warm-up: the first benchmark case, recheck=False, <=100 repeats, 1e7 ns budget;
  3. for every benchmark case: recheck=True, <=100 repeats, 3e10 ns budget, 120 s wall - i.e. the inputs are regenerated
     with seed+13 before every timed call and the output is re-checked after it; timing = CUDA events around the kernel call
     only; the loop stops when the standard error of the mean is < 0.1% of the mean, or the budget/wall limit is hit;
  4. score = geometric mean of the per-case mean runtimes (seconds), reported in microseconds.

Differences to skydiscover's shared_eval (the protocol used by the trimul_ttt lineage): recheck (fresh inputs + a
correctness check every iteration), 30 s instead of 10 s per-case budget. Kernelbot also folds a secret POPCORN_SEED into
every case seed (Cantor pairing); set POPCORN_SEED in the environment to reproduce that, the default uses the public seeds.

combined_score = SCORE_SCALE / geom_mean_us with SCORE_SCALE = 3000 (the display scale of this repo). TTT-discover's reward
is 1500 / geom_mean_us - the microseconds are identical, only the constant differs.
"""

import contextlib
import copy
import importlib.util
import math
import os
import sys
import time
import traceback

import reference
import torch
import torch.cuda

from skydiscover.optimize.evaluation.evaluation_result import EvaluationResult

SCORE_SCALE = getattr(reference, "SCORE_SCALE", 3000.0)
LB_MAX_REPEATS = 100
LB_MAX_TIME_NS = 30e9        # leaderboard mode: run_single_benchmark(pool, tests[i], True, 100, 30e9)
LB_WALL_LIMIT_NS = 120e9
LB_REL_ERROR = 0.001
WARMUP_MAX_TIME_NS = 1e7     # leaderboard mode warm-up: run_single_benchmark(pool, tests[0], False, 100, 1e7)
_seed_env = os.environ.get("POPCORN_SEED")
POPCORN_SEED = int(_seed_env) if _seed_env else None
PROTOCOL = "gpu_mode_leaderboard/kernelbot-eval.py (recheck, 100x30s, geom) as in test-time-training/discover@6c40e82"


@contextlib.contextmanager
def _bench_lock():
    """BENCHLOCK-001: hold the pool GPU's flock only while timing (SAH_EVAL_BENCH_LOCK_FILE is set by SAH's eval_runner when
    SAH_EVAL_LOCK_PHASE=benchmark); compilation and correctness tests before this point run concurrently with other evaluations."""
    path = os.environ.get("SAH_EVAL_BENCH_LOCK_FILE")
    if not path:
        yield
        return
    import fcntl
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _combine(a, b):
    """kernelbot's Cantor pairing of the public case seed with the secret leaderboard seed."""
    return int(a + (a + b) * (a + b + 1) // 2)


def _with_seed(args):
    args = dict(args)
    if POPCORN_SEED is not None and "seed" in args:
        args["seed"] = _combine(args["seed"], POPCORN_SEED)
    return args


def _clone(data):
    if isinstance(data, tuple):
        return tuple(_clone(x) for x in data)
    if isinstance(data, list):
        return [_clone(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return copy.deepcopy(data)


def _check(data, output):
    result = reference.check_implementation(data, output)
    if isinstance(result, tuple):
        return result
    return not bool(result), result


def _stats(durations_ns):
    n = len(durations_ns)
    avg = sum(durations_ns) / n
    if n > 1:
        std = math.sqrt(sum((x - avg) ** 2 for x in durations_ns) / (n - 1))
        err = std / math.sqrt(n)
    else:
        std, err = 0.0, 0.0
    return {"runs": n, "mean": avg, "std": std, "err": err, "best": float(min(durations_ns)), "worst": float(max(durations_ns))}


def _bench_single(kernel_fn, case_args, recheck, max_repeats, max_time_ns):
    """Port of kernelbot eval.py::_run_single_benchmark. Returns (stats, None) or (None, error message)."""
    args = _with_seed(case_args)
    data = reference.generate_input(**args)
    check_copy = _clone(data)
    output = kernel_fn(data)
    good, message = _check(check_copy, output)
    if not good:
        return None, f"Benchmark correctness: {message}"
    durations = []
    bm_start = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            if "seed" in args:
                args["seed"] += 13
            data = reference.generate_input(**args)
            check_copy = _clone(data)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = kernel_fn(data)
        end_event.record()
        torch.cuda.synchronize()
        duration_ns = start_event.elapsed_time(end_event) * 1e6
        if recheck:
            good, message = _check(check_copy, output)
            if not good:
                return None, f"Benchmark recheck failed at repeat {i}: {message}"
        del output
        durations.append(duration_ns)
        if i > 1:
            st = _stats(durations)
            total = time.perf_counter_ns() - bm_start
            if st["err"] / st["mean"] < LB_REL_ERROR or st["mean"] * st["runs"] > max_time_ns or total > LB_WALL_LIMIT_NS:
                break
    return _stats(durations), None


def _load(program_path):
    spec = importlib.util.spec_from_file_location("submission", program_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = mod
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _env_info():
    info = {"protocol": PROTOCOL, "torch": torch.__version__, "popcorn_seed": str(POPCORN_SEED)}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = "n/a"
    try:
        info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["gpu"] = "n/a"
    return info


def _evaluate_local(program_path):
    try:
        custom_kernel = _load(program_path)
    except Exception as exc:
        return EvaluationResult(metrics={"combined_score": 0.0, "correctness": 0.0},
                                artifacts={"error": f"Failed to load submission: {exc}", "traceback": traceback.format_exc(),
                                           "failure_stage": "import", **_env_info()})
    # 1. correctness on every test case (kernelbot "test" run: kernel on a clone, check against the original)
    for i, tc in enumerate(reference.TEST_CASES):
        try:
            data = reference.generate_input(**_with_seed(tc))
            torch.cuda.synchronize()
            output = custom_kernel(_clone(data))
            torch.cuda.synchronize()
            good, message = _check(data, output)
            if not good:
                return EvaluationResult(metrics={"combined_score": 0.0, "correctness": 0.0},
                                        artifacts={"error": f"Test {i} failed: {message}", "failure_stage": "correctness",
                                                   "test_index": str(i), **_env_info()})
        except Exception as exc:
            return EvaluationResult(metrics={"combined_score": 0.0, "correctness": 0.0},
                                    artifacts={"error": f"Test {i} error: {exc}", "traceback": traceback.format_exc(),
                                               "failure_stage": "correctness", "test_index": str(i), **_env_info()})
    # 2. warm-up, 3. leaderboard benchmarks with recheck (GPU lock held only here)
    try:
      with _bench_lock():
        _bench_single(custom_kernel, reference.BENCHMARK_CASES[0], False, LB_MAX_REPEATS, WARMUP_MAX_TIME_NS)
        means_ns, runs = [], []
        for case in reference.BENCHMARK_CASES:
            st, err = _bench_single(custom_kernel, case, True, LB_MAX_REPEATS, LB_MAX_TIME_NS)
            if err:
                return EvaluationResult(metrics={"combined_score": 0.0, "correctness": 1.0},
                                        artifacts={"error": err, "failure_stage": "benchmark", **_env_info()})
            means_ns.append(st["mean"])
            runs.append(st["runs"])
    except Exception as exc:
        return EvaluationResult(metrics={"combined_score": 0.0, "correctness": 1.0},
                                artifacts={"error": f"Benchmark error: {exc}", "traceback": traceback.format_exc(),
                                           "failure_stage": "benchmark", **_env_info()})
    # 4. geometric mean of the per-case means (seconds) -> microseconds; compute_score(ranking_by="geom")
    means_s = [ns / 1e9 for ns in means_ns]
    geom_mean_us = math.pow(math.prod(means_s), 1.0 / len(means_s)) * 1e6
    artifacts = {"hardware": "local", **_env_info()}
    for i, (ns, n) in enumerate(zip(means_ns, runs)):
        artifacts[f"bench_{i}_mean_us"] = f"{ns / 1e3:.2f}"
        artifacts[f"bench_{i}_runs"] = str(n)
    return EvaluationResult(metrics={"combined_score": SCORE_SCALE / geom_mean_us, "correctness": 1.0, "geom_mean_us": geom_mean_us},
                            artifacts=artifacts)


def evaluate(program_path):
    try:
        with open(program_path, "r") as f:
            f.read()
    except Exception as exc:
        return EvaluationResult(metrics={"combined_score": 0.0, "correctness": 0.0},
                                artifacts={"error": f"Failed to read file: {exc}", "failure_stage": "file_read"})
    return _evaluate_local(program_path)


evaluate_stage1 = evaluate
evaluate_stage2 = evaluate
