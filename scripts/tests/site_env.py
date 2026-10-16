"""Site settings for tests that drive the pipeline without a config/local.env (fresh clone, CI).

config/env.sh and config/local.env.example keep every value a caller already exported, so exporting these before the
runner sources the environment stands in for a site file.  A real config/local.env (unconditional exports) still wins.
The runner stats the payload paths before submitting, so the fake site is a small directory tree with the expected entries.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SITE: Path = Path()


def fake_site() -> Path:
    global _SITE
    if _SITE == Path():
        _SITE = Path(tempfile.mkdtemp(prefix="htt_fake_site_"))
        for rel in ("vllm/bin/vllm", "runtime/bin/python", "train/bin/python", "image.sqsh", "dataset/runtime/ahc_native/bin/docker"):
            (_SITE / rel).parent.mkdir(parents=True, exist_ok=True)
            (_SITE / rel).write_text("#!/bin/sh\n")
        for rel in ("model", "nexau", "slime", "megatron", "dataset/runtime/skydiscover_min", "dataset/runtime/simpletes_min"):
            (_SITE / rel).mkdir(parents=True, exist_ok=True)
        # a one-task snapshot (the hadamard cross-reference the fake-Slurm tests drive) in the real manifest layout
        task = _SITE / "dataset" / "raw" / "simpletes-b7e0367" / "datasets" / "hadamard_maximal_det" / "hadamard_maximal_det_29"
        task.mkdir(parents=True, exist_ok=True)
        (task / "init_program.py").write_text("# EVOLVE-BLOCK-START\nimport numpy as np\n\ndef construct():\n    return np.eye(29)\n# EVOLVE-BLOCK-END\n")
        (task / "evaluator.py").write_text("def evaluate(path):\n    return {'combined_score': 0.0, 'validity': 1.0}\n")
        (task / "hadamard_maximal_det_29.txt").write_text("Maximise the determinant of a 29x29 {-1,+1} matrix (test fixture).\n")
        import json
        (_SITE / "dataset" / "MANIFEST_eft.json").write_text(json.dumps({
            "schema": "sah-evaluator-manifest/1.0", "tasks": [],
            "cross_referenced_tasks": [{"task_id": "eft__math__hadamard_maximal_det",
                                        "use_path": "raw/simpletes-b7e0367/datasets/hadamard_maximal_det/hadamard_maximal_det_29"}]}, indent=1))
    return _SITE


def ensure_site_env() -> None:
    site = fake_site()
    real_dataset = ROOT / "dataset" / "self_adapt_harness"
    dataset = real_dataset if (real_dataset / "MANIFEST_eft.json").is_file() else site / "dataset"
    defaults = {
        "HTT_SLURM_ACCOUNT": "test-account", "HTT_SLURM_PARTITION": "test-partition", "HTT_SLURM_QOS": "test_low",
        "CONTAINER_MOUNTS": "/:/", "SAH_MODEL_PATH": str(site / "model"), "SAH_SERVED_MODEL": "test-model",
        "SAH_IMAGE": str(site / "image.sqsh"), "VLLM_ENV": str(site / "vllm"), "SAH_PYTHON": str(site / "runtime" / "bin" / "python"),
        "SAH_DATA_ROOT": str(dataset), "NEXAU_ROOT": str(site / "nexau"),
        "TRAIN_ENV": str(site / "train"), "SLIME_ROOT": str(site / "slime"), "MEGATRON_ROOT": str(site / "megatron"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if dataset != real_dataset:   # the task loader binds SAH_DATASET_ROOT at import: point it at the fixture snapshot
        os.environ.setdefault("SAH_DATASET_ROOT", str(dataset))
