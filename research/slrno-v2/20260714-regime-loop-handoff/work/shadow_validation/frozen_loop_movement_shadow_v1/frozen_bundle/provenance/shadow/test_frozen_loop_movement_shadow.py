from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


WORK = Path(__file__).resolve().parents[1]
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import frozen_loop_movement_shadow_core as core
import run_frozen_loop_movement_shadow as shadow


class FrozenLoopMovementShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(shadow.CONTRACT_PATH.read_text())

    def test_issue_window_is_post_freeze_and_pre_outcome(self) -> None:
        anchor = pd.Timestamp("2026-07-13T14:00:00Z")
        session = shadow.validate_issue_time(
            anchor, pd.Timestamp("2026-07-13T14:07:00Z"), self.contract
        )
        self.assertEqual(session, "2026-07-13")
        with self.assertRaises(AssertionError):
            shadow.validate_issue_time(
                anchor, pd.Timestamp("2026-07-13T14:30:00Z"), self.contract
            )
        with self.assertRaises(AssertionError):
            shadow.validate_issue_time(
                pd.Timestamp("2026-07-10T14:00:00Z"),
                pd.Timestamp("2026-07-10T14:07:00Z"),
                self.contract,
            )

    def test_ledger_chain_and_cumulative_support_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = shadow.runtime_paths(root)
            paths["predictions"].mkdir(parents=True)
            paths["ledger"].write_text("")
            batch = pd.DataFrame(
                {
                    "prediction_id": ["p1", "p2"],
                    "contract_id": ["test", "test"],
                    "session_date": ["2026-07-13", "2026-07-13"],
                    "symbol_norm": ["A", "B"],
                    "quarter": ["2026_q3", "2026_q3"],
                    "state": [0, 1],
                }
            )
            relative = Path("prediction_batches/batch.parquet")
            batch_path = root / relative
            batch.to_parquet(batch_path, index=False)
            record = shadow.append_record(
                root,
                {
                    "contract_id": "test",
                    "batch_file": str(relative),
                    "batch_sha256": core.sha256_file(batch_path),
                    "anchor_count": 2,
                    "session_date": "2026-07-13",
                    "symbols": ["A", "B"],
                    "states": [0, 1],
                },
            )
            self.assertEqual(record["cumulative_support"]["issued_anchors"], 2)
            loaded, predictions = shadow.load_prediction_batches(root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(predictions), 2)
            small_contract = {
                "cohort_close_rule": {
                    "minimum_issued_anchors": 2,
                    "minimum_distinct_session_dates": 1,
                    "minimum_distinct_symbols": 2,
                    "minimum_distinct_calendar_quarters": 1,
                    "required_states": [0, 1],
                }
            }
            sequence, support = shadow.first_closing_prefix(root, small_contract)
            self.assertEqual(sequence, 1)
            self.assertTrue(support["pass"])
            text = paths["ledger"].read_text()
            paths["ledger"].write_text(text.replace('"anchor_count": 2', '"anchor_count": 3'))
            with self.assertRaises(AssertionError):
                shadow.read_ledger(root)

    def test_evaluator_stops_before_provider_access_when_issuance_support_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = shadow.runtime_paths(root)
            paths["root"].mkdir(parents=True, exist_ok=True)
            paths["contract"].write_text(json.dumps(self.contract))
            args = argparse.Namespace(
                runtime_root=root,
                provider_root=Path("/path/that/must/not/be/read"),
            )
            with (
                mock.patch.object(shadow, "verify_runtime", return_value={}),
                mock.patch.object(
                    shadow,
                    "first_closing_prefix",
                    return_value=(None, {"pass": False, "issued_anchors": 0}),
                ),
                mock.patch.object(shadow, "_read_symbol_window") as provider_read,
            ):
                with self.assertRaises(shadow.SupportNotMet):
                    shadow.evaluate(args)
                provider_read.assert_not_called()

    def test_timestamp_support_failure_does_not_open_ohlc_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = shadow.runtime_paths(root)
            paths["evaluation"].mkdir(parents=True)
            paths["contract"].write_text(json.dumps(self.contract))
            predictions = pd.DataFrame(
                {
                    "prediction_id": ["p1"],
                    "start_timestamp": [pd.Timestamp("2025-01-02T14:30:00Z")],
                    "session_date": ["2025-01-02"],
                    "symbol_norm": ["A"],
                    "quarter": ["2025_q1"],
                    "state": [0],
                }
            )
            args = argparse.Namespace(runtime_root=root, provider_root=Path("unused"))
            with (
                mock.patch.object(shadow, "verify_runtime", return_value={}),
                mock.patch.object(
                    shadow,
                    "first_closing_prefix",
                    return_value=(1, {"pass": True}),
                ),
                mock.patch.object(
                    shadow,
                    "load_prediction_batches",
                    return_value=([{"record_sha256": "x"}], predictions),
                ),
                mock.patch.object(
                    shadow,
                    "exact_outcome_mask",
                    return_value=pd.Series([False]).to_numpy(),
                ),
                mock.patch.object(shadow, "build_outcome_panel") as outcome_builder,
            ):
                result = shadow.evaluate(args)
                self.assertFalse(result["outcome_values_opened"])
                self.assertFalse(result["performance_metrics_calculated"])
                outcome_builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
