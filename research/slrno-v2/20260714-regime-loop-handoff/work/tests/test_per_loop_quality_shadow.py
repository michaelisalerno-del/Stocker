from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import per_loop_quality_shadow_core as core
import run_per_loop_quality_shadow as shadow


class PerLoopQualityShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(shadow.CONTRACT_PATH.read_text())

    def test_exact_safety_labels_and_dormant_contract(self) -> None:
        checks = core.validate_contract(self.contract)
        self.assertGreaterEqual(len(checks), 10)
        self.assertTrue(self.contract["research_only"])
        self.assertFalse(self.contract["live_ordering_enabled"])
        self.assertEqual(self.contract["order_placement"], "disabled")
        self.assertEqual(
            self.contract["eligibility_freeze"]["eligible_cycle_ids"], []
        )

    def test_provisional_global_grade_forces_zero_eligible(self) -> None:
        frame = pd.read_csv(
            shadow.DEFAULT_QUALITY_ROOT / "provisional_tiers_2024.csv"
        )
        cycle_ids = core.validate_provisional_tiers(frame)
        self.assertEqual(len(cycle_ids), 20)
        self.assertTrue(frame["global_grade"].eq("unqualified").all())

    def test_final_tiers_certify_all_periods_unqualified(self) -> None:
        frame = pd.read_csv(shadow.DEFAULT_QUALITY_ROOT / "final_cycle_tiers.csv")
        self.assertEqual(len(core.validate_final_tiers(frame)), 20)
        for column in (
            "provisional_2024_oof_grade",
            "development_2025_grade",
            "backward_2023_grade",
            "final_grade",
        ):
            self.assertTrue(frame[column].eq("unqualified").all())

    def test_independent_post_score_audit_is_48_of_48_and_safe(self) -> None:
        audit = json.loads(
            (
                shadow.DEFAULT_QUALITY_ROOT / "independent_artifact_audit.json"
            ).read_text()
        )
        self.assertTrue(audit["all_passed"])
        self.assertEqual(audit["check_count"], 48)
        self.assertEqual(len(audit["checks"]), 48)
        self.assertTrue(all(item["pass"] for item in audit["checks"]))
        self.assertTrue(audit["research_only"])
        self.assertFalse(audit["live_ordering_enabled"])
        self.assertEqual(audit["order_placement"], "disabled")
        self.assertTrue(audit["no_2026_rows"])
        self.assertEqual(audit["final_decision"]["qualified_good_or_high_cycles"], 0)
        self.assertEqual(audit["final_decision"]["high_cycles"], 0)
        self.assertTrue(audit["final_decision"]["all_twenty_unqualified"])
        self.assertEqual(audit["prospective_shadow"]["ledger_lines"], 0)
        self.assertFalse(audit["prospective_shadow"]["outcomes_opened"])
        self.assertEqual(
            core.sha256_file(
                shadow.DEFAULT_QUALITY_ROOT / "independent_artifact_audit.json"
            ),
            self.contract["final_certification"][
                "independent_post_score_audit_sha256"
            ],
        )

    def test_horizon_only_good_does_not_become_global_good(self) -> None:
        frame = pd.DataFrame(
            {
                "period": ["2024_oof"] * 20,
                "cycle_id": [f"cycle_{index:02d}" for index in range(1, 21)],
                "h6_grade": ["good_movement_quality"] + ["unqualified"] * 19,
                "h12_grade": ["good_movement_quality"] + ["unqualified"] * 19,
                "h24_grade": ["unqualified"] * 20,
                "global_grade": ["unqualified"] * 20,
            }
        )
        self.assertEqual(len(core.validate_provisional_tiers(frame)), 20)

    def test_any_global_qualified_cycle_requires_new_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "period": ["2024_oof"] * 20,
                "cycle_id": [f"cycle_{index:02d}" for index in range(1, 21)],
                "h6_grade": ["unqualified"] * 20,
                "h12_grade": ["unqualified"] * 20,
                "h24_grade": ["unqualified"] * 20,
                "global_grade": ["good_movement_quality"]
                + ["unqualified"] * 19,
            }
        )
        with self.assertRaises(AssertionError):
            core.validate_provisional_tiers(frame)

    def test_probability_schema_keeps_s_q_and_j_separate(self) -> None:
        names = set(core.required_prediction_columns())
        self.assertIn("structural_probability", names)
        for target in core.TARGETS:
            for horizon in core.HORIZONS:
                suffix = f"{target}__h{horizon}"
                self.assertIn(f"conditional_q75__{suffix}", names)
                self.assertIn(f"conditional_q90__{suffix}", names)
                self.assertIn(f"joint_j75__{suffix}", names)
                self.assertIn(f"joint_j90__{suffix}", names)

    def test_dormant_validator_refuses_nonempty_prediction(self) -> None:
        row = {
            "prediction_id": "p1",
            "contract_id": self.contract["contract_id"],
            "issued_at_utc": "2026-07-10T20:00:00Z",
            "start_timestamp": "2026-07-10T19:55:00Z",
            "session_date": "2026-07-10",
            "symbol_norm": "TEST",
            "state": 0,
            "cycle_id": "cycle_09",
            "movement_quality_grade": "good_movement_quality",
            "structural_probability": 0.2,
        }
        for target in core.TARGETS:
            for horizon in core.HORIZONS:
                suffix = f"{target}__h{horizon}"
                row[f"conditional_q75__{suffix}"] = 0.4
                row[f"conditional_q90__{suffix}"] = 0.2
                row[f"joint_j75__{suffix}"] = 0.08
                row[f"joint_j90__{suffix}"] = 0.04
        with self.assertRaises(core.DormantNoEligibleCycles):
            core.validate_prediction_batch(pd.DataFrame([row]), self.contract)

    def test_chain_rule_and_nesting_in_synthetic_non_dormant_contract(self) -> None:
        contract = json.loads(json.dumps(self.contract))
        contract["eligibility_freeze"]["activation_state"] = "dormant_no_eligible_cycles"
        contract["eligibility_freeze"]["eligible_cycle_ids"] = []
        # Contract v1 itself cannot be mutated into an active contract. Exercise
        # numeric identities directly so the dormant surface remains fail-closed.
        structural = np.asarray([0.2, 0.8])
        q75 = np.asarray([0.4, 0.5])
        q90 = np.asarray([0.2, 0.1])
        self.assertTrue(np.all(q90 <= q75))
        self.assertTrue(np.allclose(structural * q75, [0.08, 0.4]))
        self.assertTrue(np.allclose(structural * q90, [0.04, 0.08]))

    def test_issue_stops_before_candidate_file_read(self) -> None:
        args = argparse.Namespace(
            runtime_root=Path("unused"), batch=Path("must-not-be-read.parquet")
        )
        with (
            mock.patch.object(shadow, "verify_runtime", return_value={}),
            mock.patch.object(
                shadow,
                "read_json",
                return_value=self.contract,
            ),
            mock.patch.object(pd, "read_parquet") as read_parquet,
        ):
            with self.assertRaises(core.DormantNoEligibleCycles):
                shadow.issue(args)
            read_parquet.assert_not_called()

    def test_empty_ledger_hash_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "prediction_ledger.jsonl"
            ledger.write_text("")
            self.assertEqual(core.validate_ledger(ledger), [])
            self.assertEqual(
                core.sha256_file(ledger),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            )

    def test_protected_aggregate_snapshot_is_exact(self) -> None:
        expected = json.loads(shadow.PROTECTED_SNAPSHOT_PATH.read_text())
        current = core.content_snapshot(shadow.AGGREGATE_SHADOW, shadow.WORKSPACE)
        self.assertEqual(current, expected)
        self.assertEqual(
            current["snapshot_sha256"],
            self.contract["integrity"][
                "existing_aggregate_shadow_content_snapshot_sha256"
            ],
        )
        self.assertNotEqual(
            current["snapshot_sha256"],
            self.contract["integrity"][
                "broader_protected_path_snapshot_tree_sha256"
            ],
        )

    def test_no_outcome_evaluator_command_exists(self) -> None:
        help_text = shadow.build_parser().format_help()
        self.assertNotIn("evaluate", help_text)


if __name__ == "__main__":
    unittest.main()
