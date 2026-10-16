"""Contract tests for HarnessTTT: put <root>/src on sys.path (inner, outer, trainer packages)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
TESTS = Path(__file__).resolve().parent
for path in (SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
