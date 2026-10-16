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
        for rel in ("vllm/bin/vllm", "runtime/bin/python", "train/bin/python", "image.sqsh", "dataset/MANIFEST_eft.json",
                    "dataset/runtime/ahc_native/bin/docker"):
            (_SITE / rel).parent.mkdir(parents=True, exist_ok=True)
            (_SITE / rel).write_text("{}\n" if rel.endswith(".json") else "#!/bin/sh\n")
        for rel in ("model", "nexau", "slime", "megatron"):
            (_SITE / rel).mkdir(parents=True, exist_ok=True)
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
