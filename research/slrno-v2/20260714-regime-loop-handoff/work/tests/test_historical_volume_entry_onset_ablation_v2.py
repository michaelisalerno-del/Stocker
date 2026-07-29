from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


WORK = Path(__file__).resolve().parents[1]
RUNNER_PATH = WORK / "run_historical_volume_entry_onset_ablation_v2.py"
CONTRACT_PATH = WORK / "contracts/20260712-historical-volume-entry-onset-ablation-v2.json"
ARTIFACT_ROOT = Path(
    "/private/tmp/stocker_historical_volume_entry_onset_ablation_v2_20260712"
)


def load_runner():
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    spec = importlib.util.spec_from_file_location("historical_volume_entry_v2_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_tape(n: int = 24) -> pd.DataFrame:
    timestamp = pd.date_range("2024-07-01 13:30:00+00:00", periods=n, freq="5min")
    close = 10.0 * np.exp(np.linspace(0.0, 0.015, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "symbol_norm": "TEST",
            "session_date": "2024-07-01",
            "month_key": "2024-07",
            "bar_ordinal": np.arange(n, dtype=np.int16),
            "segment_index": np.zeros(n, dtype=np.int16),
            "source_position": np.arange(n, dtype=np.int64),
            "segment_position": np.arange(n, dtype=np.int16),
            "segment_size": np.full(n, n, dtype=np.int16),
            "historical_volume": np.exp(np.linspace(6.0, 8.0, n)),
        }
    )


def test_contract_labels_volume_precisely_and_preserves_safety() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["sources"]["provider_volume_label"] == "historical_volume"
    assert contract["interpretation"]["no_order_flow_claim"] is True
    assert contract["interpretation"]["prior_price_only_outcomes_known_before_this_contract"]
    assert contract["periods"]["same_score_month_outcomes_read_before_that_month_probability"] is False
    assert contract["pnl_evaluation_permitted"] is False


def test_candidate_differs_from_baseline_only_by_volume_features() -> None:
    module = load_runner()
    assert module.FEATURES_BY_ALGORITHM[module.BASELINE] == module.PRICE_FEATURES
    candidate = module.FEATURES_BY_ALGORITHM[module.CANDIDATE]
    assert candidate[: len(module.PRICE_FEATURES)] == module.PRICE_FEATURES
    assert candidate[len(module.PRICE_FEATURES) :] == module.VOLUME_FEATURES
    assert len(module.VOLUME_FEATURES) == 11
    assert module.HGB_PARAMETERS == module.base.HGB_PARAMETERS


def test_volume_feature_names_cannot_be_mistaken_for_order_flow() -> None:
    module = load_runner()
    assert all(name.startswith("historical_volume_") for name in module.VOLUME_FEATURES)
    assert all("order_flow" not in name and "buy_volume" not in name for name in module.VOLUME_FEATURES)


def test_historical_volume_features_are_causal_under_future_perturbation() -> None:
    module = load_runner()
    left = synthetic_tape()
    right = left.copy()
    right.loc[12:, "historical_volume"] *= 1_000_000.0
    left_surface = module.build_historical_volume_feature_surface(left)
    right_surface = module.build_historical_volume_feature_surface(right)
    np.testing.assert_allclose(
        left_surface.loc[:11, module.VOLUME_FEATURES].to_numpy(float),
        right_surface.loc[:11, module.VOLUME_FEATURES].to_numpy(float),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )


def test_exact_log_volume_ratios_and_interactions() -> None:
    module = load_runner()
    tape = synthetic_tape()
    surface = module.build_historical_volume_feature_surface(tape)
    lv = np.log1p(tape["historical_volume"].to_numpy(float))
    row = 12
    expected_3 = lv[row] - lv[row - 3 : row].mean()
    expected_6 = lv[row] - lv[row - 6 : row].mean()
    expected_12 = lv[row] - lv[:row].mean()
    assert surface.loc[row, "historical_volume_log_ratio_prior_3"] == pytest.approx(expected_3)
    assert surface.loc[row, "historical_volume_log_ratio_prior_6"] == pytest.approx(expected_6)
    assert surface.loc[row, "historical_volume_log_ratio_prior_12"] == pytest.approx(expected_12)
    assert surface.loc[row, "historical_volume_range_interaction_6"] == pytest.approx(
        expected_6 * surface.loc[row, "log_range_ratio_6"]
    )
    assert surface.loc[row, "historical_volume_body_interaction_6"] == pytest.approx(
        expected_6 * surface.loc[row, "signed_body_fraction"]
    )


def test_recent_three_minus_disjoint_older_twelve() -> None:
    module = load_runner()
    tape = synthetic_tape()
    surface = module.build_historical_volume_feature_surface(tape)
    lv = np.log1p(tape["historical_volume"].to_numpy(float))
    row = 14
    expected = lv[row - 2 : row + 1].mean() - lv[row - 14 : row - 2].mean()
    assert surface.loc[row, "historical_volume_recent_3_minus_older_12"] == pytest.approx(
        expected
    )


def test_missing_volume_is_explicit_and_never_backfilled() -> None:
    module = load_runner()
    tape = synthetic_tape()
    tape.loc[8, "historical_volume"] = np.nan
    surface = module.build_historical_volume_feature_surface(tape)
    assert surface.loc[8, "historical_volume_missing_current"] == 1.0
    assert np.isnan(surface.loc[8, "historical_volume_log_change_1"])
    assert np.isnan(surface.loc[9, "historical_volume_log_change_1"])
    assert surface.loc[8, "historical_volume_availability_12"] < 1.0
    assert not np.isinf(surface.loc[:, module.VOLUME_FEATURES].to_numpy(float)).any()


def test_gap_resets_every_volume_window() -> None:
    module = load_runner()
    first = synthetic_tape(12)
    second = synthetic_tape(12)
    second["timestamp"] = pd.date_range("2024-07-01 15:00:00+00:00", periods=12, freq="5min")
    second["bar_ordinal"] = np.arange(18, 30, dtype=np.int16)
    second["segment_index"] = 1
    second["source_position"] = np.arange(12, 24, dtype=np.int64)
    second["segment_position"] = np.arange(12, dtype=np.int16)
    tape = pd.concat([first, second], ignore_index=True)
    surface = module.build_historical_volume_feature_surface(tape)
    assert np.isnan(surface.loc[12, "historical_volume_log_change_1"])
    assert np.isnan(surface.loc[14, "historical_volume_log_ratio_prior_3"])
    assert np.isfinite(surface.loc[15, "historical_volume_log_ratio_prior_3"])


def test_weighted_quantile_and_hysteresis_constants_are_frozen() -> None:
    module = load_runner()
    assert module.FIRE_QUANTILE == 0.95
    assert module.REARM_QUANTILE == 0.75
    value = module.base.weighted_quantile([0.1, 0.2, 0.3], [1.0, 1.0, 8.0], 0.5)
    assert value == 0.3


def test_completed_artifacts_are_labelled_and_not_audit_passed_implicitly() -> None:
    if not ARTIFACT_ROOT.is_dir():
        pytest.skip("historical-volume artifacts have not been created")
    decision = json.loads((ARTIFACT_ROOT / "decision.json").read_text())
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    audit = json.loads((ARTIFACT_ROOT / "independent_audit.json").read_text())
    assert decision["research_only"] is True
    assert decision["live_ordering_enabled"] is False
    assert decision["order_placement"] == "disabled"
    assert decision["provider_volume_label"] == "historical_volume"
    assert decision["volume_is_order_flow"] is False
    assert manifest["stage"] == "independent_audit_complete_artifact_manifest"
    assert "independent_audit.json" in manifest["files_excluding_this_manifest"]
    assert audit["audit_passed"] is True
    assert audit["checks_passed"] == 13
