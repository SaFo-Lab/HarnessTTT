"""Evaluator for TriMul under the GPU MODE *leaderboard* protocol (the setting TTT-discover scored against).

Delegates to shared_eval_lb (same directory tree as skydiscover's shared_eval, different timing protocol).
"""
import os
import sys

_problem_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_problem_dir)
for p in (_problem_dir, _parent_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared_eval_lb import evaluate, evaluate_stage1, evaluate_stage2  # noqa: E402,F401
