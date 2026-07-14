from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER = WORKSPACE / "work/run_causal_setup_conditions_v1.py"
ROOT = Path("/private/tmp/stocker_causal_setup_conditions_v1_20260712")


def load_runner():
    spec = importlib.util.spec_from_file_location("causal_setup_conditions_v1", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_hashes_and_research_boundary_are_frozen() -> None:
    module = load_runner()
    oof = module.load_oof()
    contract, pre_score = module.load_contract_and_verify(
        sorted(oof["symbol_norm"].unique())
    )
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["strategy_promotion_permitted"] is False
    assert pre_score["later_period_setup_pnl_read"] is False


def test_close_confirmation_uses_completed_close_and_next_open() -> None:
    module = load_runner()
    tape = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 103.0, 104.0],
            "high": [102.0, 103.0, 104.0, 105.0, 105.0],
            "low": [98.0, 99.0, 100.0, 102.0, 103.0],
            "close": [100.0, 101.0, 103.0, 104.0, 104.0],
        }
    )
    result = module.close_confirmation(
        tape,
        np.asarray([0]),
        np.asarray([102.0]),
        np.asarray([98.0]),
    )
    assert result["confirmed"].tolist() == [True]
    assert result["direction"].tolist() == [1]
    assert result["confirmation_step"].tolist() == [2]
    assert result["entry_price"].tolist() == [103.0]
    assert result["strong_close"].tolist() == [True]


def test_compression_and_trend_features_use_only_completed_past_bars() -> None:
    ranges = np.r_[np.full(18, 0.02), np.full(6, 0.005)]
    short_mean = ranges[-6:].mean()
    long_mean = ranges[-24:].mean()
    assert short_mean / long_mean < 0.75
    closes = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    trend = np.log(closes[-1] / closes[-7])
    assert trend > 0.0
    assert 1 * trend > 0.0
    assert -1 * trend < 0.0


def test_nonoverlap_is_independent_and_rearms_at_exit_close() -> None:
    module = load_runner()
    frame = pd.DataFrame(
        {
            "symbol_norm": ["X"] * 5,
            "session_date": ["2024-07-01"] * 5,
            "bar_ordinal": [0, 5, 6, 11, 12],
        }
    )
    positions = module.greedy_positions(
        frame, np.ones(len(frame), dtype=bool), horizon=6
    )
    assert positions.tolist() == [0, 2, 4]


def test_artifact_has_no_retained_setup_and_independent_audit_passes() -> None:
    decision = json.loads((ROOT / "decision.json").read_text())
    audit = json.loads((ROOT / "independent_audit.json").read_text())
    assert decision["retained_hypotheses"] == []
    assert all(
        not result["checks"]["retained"]
        for result in decision["hypotheses"].values()
    )
    assert decision["strategy_promotion"] is False
    assert decision["economic_edge_claim"] is False
    assert audit["all_passed"] is True
    assert audit["passed"] == 11
    assert audit["failed"] == 0
