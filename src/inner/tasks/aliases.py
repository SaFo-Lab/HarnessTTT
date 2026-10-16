"""Task aliases and display metrics shared by the pipeline driver, the status tool and the public API.

An alias is the short name a pipeline config uses for a task (``task: cp26``); DISPLAY maps a task's evaluator
``combined_score`` to the metric the paper reports: ``("mul", c, unit)`` shows ``score * c``, ``("div", c, unit)`` shows ``c / score``.
"""
from __future__ import annotations

from typing import Dict, Tuple

TASK_IDS: Dict[str, str] = {
    "cp26": "eft__math__circle_packing", "hadamard": "eft__math__hadamard_maximal_det",
    "ac1": "eft__math__first_autocorr_ineq", "ac2": "eft__math__second_autocorr_ineq",
    "erdos": "eft__math__erdos_min_overlap", "trimul": "gpu_mode__trimul", "trimul_lb": "gpu_mode__trimul_lb",
    "mla_decode": "gpu_mode__mla_decode",
}
DISPLAY: Dict[str, Tuple[str, float, str]] = {
    "eft__math__circle_packing": ("mul", 2.635, "sum_radii"), "eft__math__hadamard_maximal_det": ("mul", 1.0, "det_ratio"),
    "eft__math__second_autocorr_ineq": ("mul", 0.896280, "c2"), "eft__math__first_autocorr_ineq": ("div", 1.505293, "c1"),
    "eft__math__erdos_min_overlap": ("div", 0.380922, "c5"), "gpu_mode__trimul": ("div", 3000.0, "us"),
    "gpu_mode__trimul_lb": ("div", 3000.0, "us"), "gpu_mode__mla_decode": ("div", 3000.0, "us"),
}


def fmt_metric(kind: str, c: float, score: float) -> float:
    """combined_score -> display metric (see DISPLAY)."""
    return (c / score) if (kind == "div" and score) else (score * c)
