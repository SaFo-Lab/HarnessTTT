from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outer.reward.program_ratchet import suppress_task_training, update_strict_single  # noqa: E402


class ProgramRatchetTest(unittest.TestCase):
    TASK = "task"

    def test_unattributed_gain_cannot_enter_proposer_replay(self) -> None:
        rows = [
            {"task_id": self.TASK, "advantage": 0.7},
            {"task_id": self.TASK, "advantage": -0.2},
            {"task_id": "other", "advantage": 0.4},
        ]
        self.assertEqual(suppress_task_training(rows, self.TASK, "no_program_change"), 2)
        self.assertEqual([row["advantage"] for row in rows], [0.0, 0.0, 0.4])
        self.assertTrue(rows[0]["training_suppressed"])
        self.assertEqual(
            rows[0]["training_suppression_reason"], "no_program_change"
        )

    def _result(
        self, root: Path, k: int, score: float, program: str, *,
        seed_program: str = "initial", seed_score: float = 1.0,
        seed_mode: str = "task_initial",
    ) -> None:
        seed_snapshot = root / "seed_programs_in.json"
        if not seed_snapshot.exists():
            seed_snapshot.write_text("{}\n")
        registry_sha = hashlib.sha256(seed_snapshot.read_bytes()).hexdigest()
        (root / "round.json").write_text(json.dumps({
            "cross_round_inputs": {"seed_programs": {
                "snapshot_sha256": registry_sha,
            }},
        }))
        package = root / "tasks" / self.TASK / f"cand{k:02d}"
        package.mkdir(parents=True, exist_ok=True)
        (package / "agent.yaml").write_text("name: test-h2\n")
        package_sha = hashlib.sha256(
            b"agent.yaml\0" + (package / "agent.yaml").read_bytes()
        ).hexdigest()
        path = root / "rollouts" / self.TASK / f"cand{k:02d}" / "run" / "results"
        path.mkdir(parents=True)
        (path / f"{self.TASK}.json").write_text(json.dumps({
            "task_id": self.TASK,
            "best_score": score,
            "best_program": program,
            "seed_score": seed_score,
            "seed_program_provenance": {
                "mode": seed_mode,
                "program_sha256": __import__("hashlib").sha256(
                    seed_program.encode()
                ).hexdigest(),
                "registry_sha256": registry_sha,
            },
            "h2_package_provenance": {
                "path": str(package), "sha256": package_sha,
                "hash_scheme": "canonical-h2-v1",
                "stable_during_rollout": True,
            },
            "score_eligible": True,
        }))

    def test_same_transition_is_route_independent_and_strips_qd_memory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            self._result(
                round_dir, 2, 1.2, "winner", seed_program="old",
                seed_score=1.1, seed_mode="inherited_registry",
            )
            kwargs = dict(
                round_dir=round_dir,
                groups={self.TASK: {
                    "base_score": 1.0, "best_score": 1.2,
                    "best_k": 2, "improved": True,
                }},
                bases_in={self.TASK: {"score": 1.0}},
                previous={self.TASK: {
                    "score": 1.1, "program": "old", "parents": [{"program": "x"}],
                    "elites": [{"program": "y"}],
                }},
                round_id=7,
            )
            proposer, pa = update_strict_single(**kwargs)
            context, ca = update_strict_single(**kwargs)
            self.assertEqual(proposer, context)
            self.assertEqual(pa, ca)
            self.assertEqual(proposer[self.TASK]["program"], "winner")
            self.assertNotIn("parents", proposer[self.TASK])
            self.assertNotIn("elites", proposer[self.TASK])

    def test_candidate_must_beat_h2_and_program_incumbents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            self._result(
                round_dir, 0, 1.15, "candidate", seed_program="old",
                seed_score=1.0, seed_mode="inherited_registry",
            )
            state, audit = update_strict_single(
                round_dir=round_dir,
                groups={self.TASK: {
                    "base_score": 1.0, "best_score": 1.15,
                    "best_k": 0, "improved": True,
                }},
                bases_in={self.TASK: {"score": 1.0}},
                previous={self.TASK: {"score": 1.2, "program": "old"}},
                round_id=2,
            )
            self.assertEqual(state[self.TASK]["program"], "old")
            self.assertFalse(audit["tasks"][self.TASK]["promoted"])
            self.assertEqual(
                audit["tasks"][self.TASK]["reason"],
                "did_not_beat_program_incumbent",
            )

    def test_score_program_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            self._result(round_dir, 1, 1.1, "wrong-score-program")
            state, audit = update_strict_single(
                round_dir=round_dir,
                groups={self.TASK: {
                    "base_score": 1.0, "best_score": 1.2,
                    "best_k": 1, "improved": True,
                }},
                bases_in={self.TASK: {"score": 1.0}},
                previous={}, round_id=3,
            )
            self.assertNotIn(self.TASK, state)
            self.assertEqual(
                audit["tasks"][self.TASK]["reason"], "program_score_mismatch"
            )

    def test_inherited_seed_cannot_be_credited_as_a_new_gain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            self._result(
                round_dir, 0, 1.3, "old", seed_program="old",
                seed_score=1.3, seed_mode="inherited_registry",
            )
            state, audit = update_strict_single(
                round_dir=round_dir,
                groups={self.TASK: {
                    "base_score": 1.2, "best_score": 1.3,
                    "best_k": 0, "improved": True,
                }},
                bases_in={self.TASK: {"score": 1.2}},
                previous={self.TASK: {"score": 1.2, "program": "old"}},
                round_id=4,
            )
            self.assertEqual(state[self.TASK]["program"], "old")
            self.assertFalse(audit["tasks"][self.TASK]["promoted"])
            self.assertEqual(audit["tasks"][self.TASK]["reason"], "no_program_change")

    def test_candidate_h2_bytes_must_match_the_rollout_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            self._result(round_dir, 0, 1.3, "winner", seed_score=1.0)
            package = round_dir / "tasks" / self.TASK / "cand00" / "agent.yaml"
            package.write_text("name: changed-after-rollout\n")
            state, audit = update_strict_single(
                round_dir=round_dir,
                groups={self.TASK: {
                    "base_score": 1.0, "best_score": 1.3,
                    "best_k": 0, "improved": True,
                }},
                bases_in={self.TASK: {"score": 1.0}},
                previous={}, round_id=5,
            )
            self.assertNotIn(self.TASK, state)
            self.assertEqual(
                audit["tasks"][self.TASK]["reason"],
                "h2_package_provenance_mismatch",
            )


if __name__ == "__main__":
    unittest.main()
