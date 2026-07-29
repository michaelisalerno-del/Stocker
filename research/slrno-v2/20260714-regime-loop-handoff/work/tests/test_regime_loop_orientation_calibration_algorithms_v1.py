from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/run_regime_loop_orientation_calibration_algorithms_v1.py"
SPEC = importlib.util.spec_from_file_location("orientation_algorithms", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen_research_only_and_no_promotion() -> None:
    contract = MODULE.load_contract()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["population_and_causality"]["later_period_paths_permitted"] is False
    assert contract["decision"]["named_loop_good_or_high_promotion_permitted"] is False
    assert contract["decision"]["same_experiment_refinement_permitted"] is False


def test_source_is_bound_to_passing_independent_linkage_audit() -> None:
    observed = MODULE.verify_sources()
    assert observed == MODULE.EXPECTED_HASHES


def test_logit_and_probability_clip_are_stable() -> None:
    values = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
    transformed = MODULE.logit(values)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed, -MODULE.logit(1 - values))
    assert MODULE.clip_probability(values).min() == MODULE.EPSILON
    assert MODULE.clip_probability(values).max() == 1 - MODULE.EPSILON


def test_categorical_matrix_has_exact_width_and_values() -> None:
    matrix = MODULE.categorical_matrix(
        np.asarray([0, 2, 1]), 3, np.asarray([0.25, 0.5, 0.75])
    )
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1).A1, [0.25, 0.5, 0.75])


def test_orientation_feature_width_matches_declared_blocks() -> None:
    frame = pd.DataFrame(
        {
            "orientation_index": [0, 1],
            "orientation_clock_index": [0, 2],
            "link__baseline__absolute_return_bps__h6__p75": [0.05, 0.1],
            "link__raw_full_link__absolute_return_bps__h6__p75": [0.1, 0.2],
        }
    )
    raw, _ = MODULE.global_residual_features(
        frame, "absolute_return_bps", 6, "p75"
    )
    scaler = StandardScaler().fit(raw)
    orientation = MODULE.orientation_features(
        frame, "absolute_return_bps", 6, "p75", scaler, 2, None
    )
    clock = MODULE.orientation_features(
        frame, "absolute_return_bps", 6, "p75", scaler, 2, 3
    )
    assert orientation.shape == (2, 6)
    assert clock.shape == (2, 12)


def test_binary_losses_and_calibration_are_exact_on_toy_data() -> None:
    y = np.asarray([0, 1])
    p = np.asarray([0.25, 0.75])
    log_loss, brier = MODULE.binary_losses(y, p)
    assert np.allclose(log_loss, -np.log(0.75))
    assert np.allclose(brier, 0.0625)
    ece, maximum, bins = MODULE.calibration_summary(y, p, np.ones(2), 1)
    assert np.isclose(ece, 0.25)
    assert np.isclose(maximum, 0.25)
    assert bins == 2


def test_holm_adjusts_within_comparison_endpoint_family() -> None:
    frame = pd.DataFrame(
        {
            "comparison": ["base", "base", "raw"],
            "endpoint": ["ll", "ll", "ll"],
            "p_value": [0.01, 0.04, 0.03],
        }
    )
    result = MODULE.holm_adjust(frame, ["comparison", "endpoint"])
    assert result.loc[result["comparison"].eq("base"), "family_size"].eq(2).all()
    assert result.loc[result["comparison"].eq("raw"), "family_size"].eq(1).all()


def test_runner_has_no_later_or_shadow_input_constants() -> None:
    source = SOURCE.read_text()
    assert "PREDICTIONS_2025" not in source
    assert "PREDICTIONS_2023" not in source
    assert "ANCHOR_2026" not in source
    assert "prediction_ledger" not in source

