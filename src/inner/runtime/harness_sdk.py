"""Capability surface for M_phi-generated custom tools (h2spec/1.0).

A generated tool is a function ``run(ctx, args)``. ``ctx`` is the ONLY door to
the outside world: it exposes a small, audited set of operations bridged to the
inner session. Generated code passes static gates that forbid imports beyond a
whitelist and any direct I/O, so this context IS the tool's capability set —
the evaluator internals, benchmark answers, network, and the filesystem outside
the sandbox scratch dir are all unreachable by construction.

Anti-reward-hacking invariants:
  * no access to evaluator source or task answer files;
  * ``probe``/``evaluate`` go through the session's official pipeline and
    budgets (probes capped, evaluations debit the real budget);
  * ``read_input_sample`` serves at most SAMPLE_CAP rows of declared task
    input files, read-only.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SAMPLE_CAP = 5000
SCRATCH_CAP_BYTES = 8 * 1024 * 1024


class ToolContext:
    """Capability object handed to generated tools."""

    __slots__ = (
        "_get_program_fn", "_get_best_program_fn", "_best_score_fn",
        "_stage_edit_fn", "_probe_fn", "_evaluate_fn", "_ledger_fn",
        "_candidate_geometry_fn", "_history_note_fn", "_task_dir", "_scratch",
    )

    def __init__(self, session, scratch_dir: Path):
        # Do not retain the live InnerSession as a context attribute.  Only
        # bound, capability-specific callbacks are kept; the static gate rejects
        # every private-attribute/introspection route from generated code.
        self._get_program_fn = lambda: session.current_program
        self._get_best_program_fn = lambda: session.best_program or session.current_program
        self._best_score_fn = lambda: float(session.best_score)
        self._stage_edit_fn = session.apply_edit
        self._probe_fn = session.probe
        self._evaluate_fn = session.evaluate
        self._candidate_geometry_fn = getattr(
            session,
            "candidate_geometry",
            lambda: {
                "ok": False,
                "error": "candidate geometry bridge is unavailable in this runtime",
            },
        )
        self._ledger_fn = lambda: session.ledger
        self._history_note_fn = session.history_note
        self._task_dir = Path(session.task.task_dir).resolve()
        self._scratch = Path(scratch_dir)
        self._scratch.mkdir(parents=True, exist_ok=True)

    # ---- program access ---------------------------------------------------
    def get_program(self) -> str:
        """Current working program text."""
        return self._get_program_fn()

    def get_best_program(self) -> str:
        return self._get_best_program_fn()

    def best_score(self) -> float:
        return self._best_score_fn()

    def stage_edit(self, code: str) -> str:
        """Stage a SEARCH/REPLACE diff or full EVOLVE-BLOCK replacement
        (identical semantics to the edit_solution tool)."""
        return self._stage_edit_fn(code)

    def get_candidate_geometry(self) -> Dict[str, Any]:
        """Execute the current CP26 candidate in the trusted sandbox.

        This debits one probe call and records sandbox wall time.  It exposes
        only normalized centers/radii from the public candidate entry point;
        evaluator internals and answers remain inaccessible.
        """
        return self._candidate_geometry_fn()

    # ---- evaluation -------------------------------------------------------
    def probe(self, subsample: int = 2000) -> Dict[str, Any]:
        """Cheap approximate evaluation (separate probe budget)."""
        ledger = self._ledger_fn()
        if ledger.probe_calls >= ledger.max_probe_calls:
            return {"error": "probe budget exhausted"}
        out = self._probe_fn(subsample=subsample)
        return {"combined_score": out.combined_score, "validity": out.validity,
                "error": out.error}

    def evaluate(self) -> Dict[str, Any]:
        """Full official evaluation — debits the REAL evaluation budget."""
        if self._ledger_fn().exhausted():
            return {"error": "evaluation budget exhausted"}
        out = self._evaluate_fn()
        return {"combined_score": out.combined_score, "validity": out.validity,
                "error": out.error}

    def budget_left(self) -> Dict[str, int]:
        ledger = self._ledger_fn()
        return {"evaluations": ledger.evaluator_budget_left(),
                "probes": ledger.max_probe_calls - ledger.probe_calls}

    # ---- task data (read-only, capped) ------------------------------------
    def list_task_inputs(self) -> List[str]:
        """Names of data files in the task directory (read-only view)."""
        td = self._task_dir
        out = []
        for p in sorted(td.rglob("*")):
            rel = str(p.relative_to(td)).lower() if p.is_file() else ""
            blocked = ("evaluator", "answer", "ground_truth", "expected", "solution")
            if p.is_file() and p.suffix in (".csv", ".txt", ".json") \
                    and not any(marker in rel for marker in blocked):
                out.append(str(p.relative_to(td)))
        return out[:50]

    def _safe_path(self, name: str):
        td = self._task_dir
        raw = Path(str(name))
        if raw.is_absolute():
            return None
        p = (td / raw).resolve()
        try:
            p.relative_to(td)
        except ValueError:
            return None
        rel = str(p.relative_to(td)).lower()
        blocked = ("evaluator", "answer", "ground_truth", "expected", "solution")
        if any(marker in rel for marker in blocked) or not p.is_file():
            return None
        return p

    def read_input_sample(self, name: str, nrows: int = 1000) -> str:
        """First ``nrows`` lines of a task input file (never evaluator code)."""
        p = self._safe_path(name)
        if p is None:
            return "ERROR: path not allowed or missing"
        nrows = min(int(nrows), SAMPLE_CAP)
        buf = io.StringIO()
        with open(p, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= nrows:
                    break
                buf.write(line)
        return buf.getvalue()

    def read_input_df(self, name: str, nrows: int = 2000):
        """Parse a CSV task input into a pandas DataFrame (<=SAMPLE_CAP rows).
        Lets generated tools analyze data without importing io/csv themselves."""
        import pandas as pd
        p = self._safe_path(name)
        if p is None:
            raise ValueError(f"input not allowed or missing: {name}")
        return pd.read_csv(p, nrows=min(int(nrows), SAMPLE_CAP))

    # ---- scratch ----------------------------------------------------------
    def scratch_write(self, name: str, text: str) -> str:
        p = (self._scratch / os.path.basename(name)).resolve()
        if len(text) > SCRATCH_CAP_BYTES:
            return "ERROR: too large"
        p.write_text(text)
        return str(p)

    def scratch_read(self, name: str) -> str:
        p = (self._scratch / os.path.basename(name)).resolve()
        return p.read_text() if p.exists() else "ERROR: no such file"

    def log(self, message: str) -> None:
        self._history_note_fn(f"[custom-tool] {str(message)[:400]}")


class MockContext(ToolContext):
    """Self-test double: no session needed; canned responses."""

    def __init__(self, scratch_dir: Path, program: str = "# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END",
                 sample: str = "a,b\n1,2\n3,4\n"):
        self._scratch = Path(scratch_dir)
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._program = program
        self._sample = sample
        self.calls: List[str] = []

    def get_program(self): self.calls.append("get_program"); return self._program
    def get_best_program(self): self.calls.append("get_best_program"); return self._program
    def get_candidate_geometry(self):
        self.calls.append("get_candidate_geometry")
        centers = [
            [0.05 + 0.18 * (index % 5), 0.05 + 0.15 * (index // 5)]
            for index in range(26)
        ]
        return {
            "ok": True,
            "source": "mock_current_program",
            "centers": centers,
            "radii": [0.01] * 26,
            "sum_radii": 0.26,
            "probes_left": 9,
        }
    def best_score(self): self.calls.append("best_score"); return 0.5
    def stage_edit(self, code):
        self.calls.append("stage_edit")
        return "edit staged"
    def probe(self, subsample=2000):
        self.calls.append("probe")
        return {"combined_score": 0.5, "validity": 1.0, "error": None}
    def evaluate(self):
        self.calls.append("evaluate")
        return {"combined_score": 0.5, "validity": 1.0, "error": None}
    def budget_left(self): return {"evaluations": 10, "probes": 10}
    def list_task_inputs(self): return ["input.csv"]
    def read_input_sample(self, name, nrows=1000): return self._sample
    def read_input_df(self, name, nrows=2000):
        import pandas as pd, io as _io
        self.calls.append("read_input_df")
        return pd.read_csv(_io.StringIO(self._sample))
