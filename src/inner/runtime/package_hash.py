"""Canonical, location-independent content hash for an H2 package."""
from __future__ import annotations

import hashlib
from pathlib import Path


def h2_sha256(root: Path) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts \
                or path.name.endswith((".pyc", ".pyo")):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
