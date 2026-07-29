from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER = WORKSPACE / "work/run_frozen_regime_loop_pnl_sanity_v1.py"
ROOT = Path("/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712")


def load_runner():
    spec = importlib.util.spec_from_file_location("pnl_sanity_v1", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_hashes_common_universe_and_safety_are_frozen() -> None:
    module = load_runner()
    contract = json.loads(module.CONTRACT_PATH.read_text())
    symbols = contract["overlap_and_portfolio"]["fixed_universe"]
    verified, pre_score = module.load_contract_and_verify_hashes(symbols)
    assert len(symbols) == 20
    assert verified["research_only"] is True
    assert verified["live_ordering_enabled"] is False
    assert verified["order_placement"] == "disabled"
    assert verified["broker_connection_enabled"] is False
    assert verified["paper_or_demo_execution_enabled"] is False
    assert pre_score["pnl_scored_before_freeze"] is False


def test_cash_consistent_short_return_is_simple_not_inverse_or_signed_log() -> None:
    entry = 100.0
    exit_price = 90.0
    direction = -1
    gross = direction * (exit_price / entry - 1.0)
    assert np.isclose(gross, 0.10)
    assert not np.isclose(gross, entry / exit_price - 1.0)
    assert not np.isclose(gross, -np.log(exit_price / entry))


def test_breakout_open_ordering_and_intrabar_ambiguity_are_exact() -> None:
    module = load_runner()
    tape = pd.DataFrame(
        {
            "open": [100.0, 102.0, 100.0, 100.0],
            "high": [101.0, 103.0, 102.0, 100.0],
            "low": [99.0, 98.0, 98.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    )
    gap_known = module.breakout_execution(
        tape,
        np.asarray([0]),
        1,
        np.asarray([101.0]),
        np.asarray([99.0]),
        np.asarray([100.0]),
    )
    assert gap_known["status"].tolist() == ["filled"]
    assert gap_known["direction"].tolist() == [1]
    assert gap_known["entry_price"].tolist() == [102.0]

    tape.loc[1, "open"] = 100.0
    ambiguous = module.breakout_execution(
        tape,
        np.asarray([0]),
        1,
        np.asarray([101.0]),
        np.asarray([99.0]),
        np.asarray([100.0]),
    )
    assert ambiguous["status"].tolist() == ["ambiguous_same_bar"]
    assert np.isnan(ambiguous["gross_return"][0])


def test_nonoverlap_allows_rearming_at_prior_exit_close() -> None:
    module = load_runner()
    frame = pd.DataFrame(
        {
            "symbol_norm": ["X"] * 5,
            "session_date": ["2025-01-02"] * 5,
            "bar_ordinal": [0, 3, 6, 7, 12],
        }
    )
    accepted = module.greedy_nonoverlap(frame, np.ones(len(frame), dtype=bool), 6)
    assert np.flatnonzero(accepted).tolist() == [0, 2, 4]


def test_artifact_decision_and_independent_audit_are_negative_and_exact() -> None:
    decision = json.loads((ROOT / "decision.json").read_text())
    audit = json.loads((ROOT / "independent_audit.json").read_text())
    assert decision["decision"] == "pnl_translation_not_supported"
    assert decision["checks"]["pass"] is False
    assert not any(
        value
        for name, value in decision["checks"].items()
        if name != "pass"
    )
    assert decision["economic_edge_claim"] is False
    assert decision["strategy_promotion"] is False
    assert audit["all_passed"] is True
    assert audit["passed"] == 13
    assert audit["failed"] == 0
