from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocker_research.broad_conflict_options_iv_screen_v0 import (
    OPTIONS_PRIMARY_FEATURES,
    ROUTE_FEATURES,
    add_iv_relative_outcomes,
    assert_protected_boundary,
    assign_development_frozen_iv_deciles,
    broad_conflict_iv_gate_passes,
    build_matched_control_relations,
    choose_options_movement_decision,
    coverage_gates_pass,
    fit_options_linear_model,
    fixed_session_bootstrap_multiplicities,
    o1_model_gate_passes,
    permute_intact_route_bundle,
)


def test_iv_relative_outcomes_are_underlying_movement_not_option_pnl() -> None:
    frame = pd.DataFrame(
        {
            "absolute_log_return_15m": [0.006, 0.002],
            "iv_sigma_15m": [0.005, 0.005],
            "iv_expected_absolute_15m": [0.004, 0.004],
        }
    )

    result = add_iv_relative_outcomes(frame)

    assert result["iv_absolute_residual_15m"].tolist() == pytest.approx([0.002, -0.002])
    assert result["iv_sigma_ratio_15m"].tolist() == pytest.approx([1.2, 0.4])
    assert result["movement_exceeds_iv_expected_absolute"].tolist() == [1, 0]
    assert result["movement_exceeds_one_iv_sigma"].tolist() == [1, 0]
    assert not any("pnl" in column.casefold() for column in result.columns)


def test_development_frozen_iv_deciles_do_not_use_assessment_values() -> None:
    development = pd.Series(np.arange(1.0, 101.0))
    assessment = pd.Series([1_000_000.0, 50.0])

    first, edges = assign_development_frozen_iv_deciles(development, assessment)
    second, repeated_edges = assign_development_frozen_iv_deciles(
        development, pd.Series([10_000_000.0, 50.0])
    )

    assert edges == repeated_edges
    assert first.iloc[0] == second.iloc[0] == 9
    assert first.iloc[1] == second.iloc[1]


def _matching_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "row_id": "treated",
            "symbol": "AAPL",
            "session": "2025-01-15",
            "year_month": "2025-01",
            "checkpoint": 6,
            "route_resolution_state": "BROAD_CONFLICT",
            "atm_iv_decile": 4,
            "front_dte": 20,
            "any_prefix_one_transition_from_completion": 0,
            "advance_eligible": 1,
        }
    ]
    for index in range(6):
        rows.append(
            {
                "row_id": f"control-{index}",
                "symbol": "AAPL",
                "session": f"2025-01-{index + 2:02d}",
                "year_month": "2025-01",
                "checkpoint": 6,
                "route_resolution_state": "OTHER",
                "atm_iv_decile": 4,
                "front_dte": 20,
                "any_prefix_one_transition_from_completion": 0,
                "advance_eligible": 1,
            }
        )
    return pd.DataFrame(rows)


def test_matched_controls_require_five_and_are_equal_weighted() -> None:
    relations = build_matched_control_relations(_matching_panel())

    assert set(relations["treated_row_id"]) == {"treated"}
    assert len(relations) == 6
    assert relations["match_weight"].sum() == pytest.approx(1.0)
    assert relations["different_session"].all()

    insufficient = _matching_panel().loc[
        lambda frame: ~frame["row_id"].isin(["control-4", "control-5"])
    ]
    assert build_matched_control_relations(insufficient).empty


def _model_panel(*, period: str, values: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(values):
        row: dict[str, object] = {
            "row_id": f"{period}-{index}",
            "period": period,
            "symbol": "AAPL" if index % 2 == 0 else "MSFT",
            "session": f"2024-01-{index + 2:02d}"
            if period == "development"
            else f"2025-01-{index + 2:02d}",
            "checkpoint": 6 if index % 2 == 0 else 8,
            "year_month": "2024-01" if period == "development" else "2025-01",
            "row_weight": 1.0,
            "movement_exceeds_iv_expected_absolute": index % 2,
        }
        for feature_index, feature in enumerate(OPTIONS_PRIMARY_FEATURES):
            row[feature] = value + feature_index / 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_model_preprocessing_is_fit_on_development_only() -> None:
    development = _model_panel(period="development", values=[1.0, 2.0, 3.0, 4.0])
    assessment = _model_panel(period="assessment", values=[1000.0, 2000.0])

    model = fit_options_linear_model(
        development,
        numeric_features=OPTIONS_PRIMARY_FEATURES,
        model_id="O0",
        kind="logistic",
    )
    probabilities = model.predict(assessment)

    assert model.preprocessing_fitted_period == "development_2024_only"
    assert model.numeric_means[0] == pytest.approx(2.5)
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()


def test_session_bootstrap_is_exactly_25_whole_session_draws() -> None:
    frame = pd.DataFrame({"session": ["a", "a", "b", "b"], "row_id": range(4)})
    draws = fixed_session_bootstrap_multiplicities(frame, draws=25, seed=20260722)

    assert draws.shape == (25, 4)
    assert np.array_equal(draws[:, 0], draws[:, 1])
    assert np.array_equal(draws[:, 2], draws[:, 3])
    with pytest.raises(ValueError, match="exactly 25"):
        fixed_session_bootstrap_multiplicities(frame, draws=24, seed=20260722)


def test_route_null_permutates_complete_bundle_within_each_slate() -> None:
    rows: list[dict[str, object]] = []
    for period in ("development", "assessment"):
        for symbol_index, symbol in enumerate(("A", "B", "C")):
            row: dict[str, object] = {
                "period": period,
                "session": "2025-01-02",
                "checkpoint": 6,
                "symbol": symbol,
            }
            for feature_index, feature in enumerate(ROUTE_FEATURES):
                row[feature] = symbol_index * 100 + feature_index
            rows.append(row)
    frame = pd.DataFrame(rows)

    permuted = permute_intact_route_bundle(frame, seed=7)

    for period in ("development", "assessment"):
        original = frame.loc[frame["period"].eq(period), list(ROUTE_FEATURES)].to_numpy()
        changed = permuted.loc[permuted["period"].eq(period), list(ROUTE_FEATURES)].to_numpy()
        assert sorted(map(tuple, original)) == sorted(map(tuple, changed))
        assert np.all(np.diff(changed, axis=1) == 1)


def test_protected_boundary_rejects_signal_or_option_dates() -> None:
    assert_protected_boundary(
        signal_dates=pd.Series([date(2025, 8, 22)]),
        options_dates=pd.Series([date(2025, 8, 21)]),
    )
    with pytest.raises(ValueError, match="protected"):
        assert_protected_boundary(
            signal_dates=pd.Series([date(2025, 8, 23)]),
            options_dates=pd.Series([date(2025, 8, 22)]),
        )


def test_coverage_gate_checks_every_frozen_threshold() -> None:
    evidence = {
        "historical_symbols": 15,
        "paired_symbols_development": 12,
        "paired_symbols_assessment": 12,
        "development_row_coverage": 0.70,
        "assessment_row_coverage": 0.70,
        "assessment_rows": 20_000,
        "assessment_sessions": 130,
        "assessment_months": 7,
        "assessment_broad_conflict_rows": 250,
        "assessment_low_route_support_rows": 250,
        "maximum_stock_weight_share": 0.12,
        "download_integrity_passed": True,
    }
    assert coverage_gates_pass(evidence)
    evidence["assessment_rows"] = 19_999
    assert not coverage_gates_pass(evidence)


def test_exact_decision_ladder_and_gates() -> None:
    model_gates = {
        "log_loss_improvement": 0.01,
        "brier_improvement": 0.01,
        "auc_improvement": 0.0,
        "average_precision_improvement": 0.01,
        "bootstrap_80_log_loss_lower": 0.0,
        "bootstrap_80_brier_lower": 0.0,
        "bootstrap_80_average_precision_lower": 0.0,
        "positive_months": 5,
        "materially_adverse_checkpoint_groups": 0,
        "real_exceeds_matching_nulls": 4,
        "coverage_and_concentration_passed": True,
    }
    broad_gates = {
        "mean_residual": 0.001,
        "minus_low_route_support_residual": 0.001,
        "minus_matched_residual": 0.001,
        "minus_matched_exceed_rate": 0.01,
        "bootstrap_80_minus_low_residual_lower": 0.0,
        "bootstrap_80_minus_matched_residual_lower": 0.0,
        "bootstrap_80_minus_matched_exceed_lower": 0.0,
        "positive_months": 5,
        "materially_adverse_checkpoint_groups": 0,
        "support_and_concentration_passed": True,
    }

    assert o1_model_gate_passes(model_gates)
    assert broad_conflict_iv_gate_passes(broad_gates)
    assert (
        choose_options_movement_decision(
            blocker=None,
            o1_passed=True,
            broad_conflict_passed=True,
            descriptive_only=False,
        )
        == "broad_conflict_predicts_iv_excess_movement"
    )
    assert (
        choose_options_movement_decision(
            blocker="blocked_missing_eodhd_api_token",
            o1_passed=False,
            broad_conflict_passed=False,
            descriptive_only=False,
        )
        == "blocked_missing_eodhd_api_token"
    )
