"""Minimal stand-in for skydiscover.optimize.evaluation.evaluation_result (SAH runtime shim).

The frozen GPU-mode evaluators (benchmarks/gpu_mode/shared_eval.py) return ``EvaluationResult(metrics=..., artifacts=...)``;
SAH's _eval_worker flattens ``.metrics`` into {combined_score, validity, ...}.  Nothing else of skydiscover is needed.
"""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EvaluationResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": dict(self.metrics), "artifacts": dict(self.artifacts)}
