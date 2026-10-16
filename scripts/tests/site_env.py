"""Site settings for tests that drive the pipeline without a config/local.env (fresh clone, CI).

config/env.sh and config/local.env.example keep every value a caller already exported, so exporting these before the
runner sources the environment stands in for a site file.  A real config/local.env (unconditional exports) still wins.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ensure_site_env() -> None:
    defaults = {
        "HTT_SLURM_ACCOUNT": "test-account", "HTT_SLURM_PARTITION": "test-partition", "HTT_SLURM_QOS": "test_low",
        "CONTAINER_MOUNTS": "/:/", "SAH_MODEL_PATH": str(ROOT), "SAH_SERVED_MODEL": "test-model",
        "SAH_IMAGE": str(ROOT / "pyproject.toml"), "VLLM_ENV": str(ROOT), "SAH_PYTHON": str(ROOT / "pyproject.toml"),
        "TRAIN_ENV": str(ROOT), "SLIME_ROOT": str(ROOT), "MEGATRON_ROOT": str(ROOT),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
