from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.dense_signed_pressure_v0 import (
    DENSE_CHECKPOINTS,
    SPARSE_TOLERANCE,
    apply_component_scale,
    assert_phase2_authorized,
    build_dense_bar_grid,
    causal_progress_surface,
    classify_cross_sectional_dependency,
    compare_causal_candidate_to_sparse,
    discover_signed_pressure_lineage,
    evaluate_coverage_gate,
    exact_pressure_raw_components,
    invalid_dense_pressure_surface,
    phase1_decision,
    pressure_window_audit,
    sparse_compatibility_summary,
    standardized_signed_pressure,
    validate_authorized_sessions,
)
from stocker_research.pretrigger_quiet_accumulation_v0 import (
    PRIMARY_WINDOW_BARS,
    QUIET_SIGNED_COMPONENTS,
    activity_without_displacement,
    apply_quiet_score_parameters,
    build_pretrigger_feature_rows,
    fit_quiet_score_parameters,
    freeze_confidence_boundary,
    pressure_persistence,
    pressure_slope,
    score_sign_persistence,
    selective_actions,
    signed_absorption_divergence,
)

ROOT = Path(__file__).resolve().parents[4]


def _trace(*, remove_bbb_bar: int | None = None) -> pd.DataFrame:
    start = pd.Timestamp("2024-06-03 13:30:00", tz="UTC")
    rows: list[dict[str, object]] = []
    for stock, count, drift in (("AAA", 9, 0.001), ("BBB", 6, -0.001)):
        previous = 100.0
        for ordinal in range(count):
            if stock == "BBB" and ordinal == remove_bbb_bar:
                continue
            open_price = previous
            close = open_price * (1.0 + drift)
            rows.append(
                {
                    "symbol": stock,
                    "session": "2024-06-03",
                    "bar_ordinal": ordinal,
                    "bar_start_timestamp": start + pd.Timedelta(minutes=5 * ordinal),
                    "bar_complete_timestamp": start + pd.Timedelta(minutes=5 * (ordinal + 1)),
                    "open": open_price,
                    "high": max(open_price, close) + 0.1,
                    "low": min(open_price, close) - 0.1,
                    "close": close,
                    "historical_relative_activity": 1.0 + ordinal / 10.0,
                }
            )
            previous = close
    return pd.DataFrame(rows)


def _pretrigger_bars() -> pd.DataFrame:
    trace = _trace()
    bars = trace.loc[trace["symbol"].eq("AAA")].copy().rename(columns={"symbol": "stock"})
    bars["vti__bar_log_return"] = 0.0001
    bars["signed_pressure"] = np.linspace(-0.2, 0.6, len(bars))
    return bars


def _raw_score_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        side = -1.0 if index < 6 else 1.0
        row: dict[str, object] = {
            "partition": "development",
            "net_return_25": side * (0.001 + index / 100_000),
            "path_length_25": 0.01 + index / 1_000,
            "range_sum_25": 0.02 + index / 1_000,
            "pressure_sum_25": side * (0.2 + index / 20),
            "pressure_persistence_25": side * 0.6,
            "pressure_slope_25": side * 0.1,
            "activity_without_displacement_25": side * 0.7,
            "relative_resilience_25": side * 0.002,
            "mean_clv_25": side * 0.4,
            "mean_wick_asymmetry_25": side * 0.3,
            "break_failure_asymmetry_25": side * 0.2,
            "mean_vwap_distance_25": side * 0.5,
            "vwap_side_balance_25": side * 0.6,
            "vwap_reclaim_balance_25": side * 0.25,
        }
        for position in range(PRIMARY_WINDOW_BARS):
            row[f"_pressure_sum_3bar_position_{position}"] = side * (
                0.2 + index / 20 + position / 100
            )
            row[f"_net_return_3bar_position_{position}"] = side * (
                0.001 + index / 100_000 + position / 1_000_000
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_exact_pressure_lineage_discovery() -> None:
    lineage = discover_signed_pressure_lineage(
        ROOT / "packages/stocker_research/src/stocker_research/behavioural_state_dimensions_v0.py",
        ROOT / "research/route-competition/"
        "20260722-route-competition-hazard-quick-v0/run_screen_v0.py",
    )
    assert lineage.function_name == "_signed_pressure"
    assert lineage.route_call_site_function == "add_development_frozen_baseline_features"
    assert lineage.input_columns == (
        "z_signed_progress",
        "z_signed_efficiency",
        "z_mean_close_location",
        "z_boundary_slope",
    )
    assert lineage.component_weights == "equal_arithmetic_mean"


def test_future_filtered_cross_sectional_membership_is_class_d() -> None:
    current = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "session": ["2024-06-03", "2024-06-03"],
            "checkpoint": [6, 6],
            "current_history_available": [True, True],
            "three_future_bars_available": [True, False],
        }
    )

    result = classify_cross_sectional_dependency(current)

    assert result.classification == "D"
    assert result.causality_status == "future_dependent_population_membership"
    assert result.changed_slates == 1


def test_dense_checkpoint_indexing_and_odd_checkpoint_generation() -> None:
    grid = build_dense_bar_grid(_trace())
    assert tuple(sorted(grid["checkpoint"].unique())) == DENSE_CHECKPOINTS
    aaa_five = grid.loc[grid["symbol"].eq("AAA") & grid["checkpoint"].eq(5)].iloc[0]
    assert bool(aaa_five["bar_present"])
    assert bool(aaa_five["current_history_available"])
    assert pd.Timestamp(aaa_five["bar_complete_timestamp"]) == pd.Timestamp(
        "2024-06-03 13:55:00+00:00"
    )


def test_completed_bar_and_rolling_lookback_causality() -> None:
    bars = _trace().loc[lambda frame: frame["symbol"].eq("AAA")].iloc[:5].copy()
    centered = 12.5
    original = exact_pressure_raw_components(bars, centered_signed_progress_bps=centered)
    mutated = _trace().loc[lambda frame: frame["symbol"].eq("AAA")].copy()
    mutated.loc[mutated["bar_ordinal"].ge(5), ["open", "high", "low", "close"]] *= 10.0
    recalculated = exact_pressure_raw_components(
        mutated.iloc[:5], centered_signed_progress_bps=centered
    )
    assert original == recalculated


def test_missing_bars_are_not_interpolated_or_filled() -> None:
    grid = build_dense_bar_grid(_trace(remove_bbb_bar=2))
    missing = grid.loc[grid["symbol"].eq("BBB") & grid["checkpoint"].eq(3)].iloc[0]
    later = grid.loc[grid["symbol"].eq("BBB") & grid["checkpoint"].eq(4)].iloc[0]
    assert not bool(missing["bar_present"])
    assert not bool(later["prefix_contiguous"])
    assert not bool(later["current_history_available"])


def test_causal_and_future_filtered_progress_slates_differ() -> None:
    progress = causal_progress_surface(build_dense_bar_grid(_trace()))
    aaa = progress.loc[progress["symbol"].eq("AAA") & progress["checkpoint"].eq(6)].iloc[0]
    assert aaa["causal_signed_progress_bps"] > 0.0
    assert aaa["archived_future_filtered_signed_progress_bps"] == pytest.approx(0.0)


def test_sparse_to_dense_compatibility_fails_at_binding_tolerance() -> None:
    progress = causal_progress_surface(build_dense_bar_grid(_trace()))
    sparse = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "session": ["2024-06-03"],
            "checkpoint": [6],
            "signed_pressure": [0.25],
            "z_component__signed_progress": [0.0],
            "raw_component__signed_progress": [0.0],
        }
    )
    scaling = {
        "6": {
            "signed_progress": {
                "center": 0.0,
                "scale": 100.0,
                "clip_lower": -5.0,
                "clip_upper": 5.0,
            }
        }
    }
    comparison = compare_causal_candidate_to_sparse(sparse, progress, scaling)
    summary = sparse_compatibility_summary(comparison)
    assert comparison.loc[0, "absolute_difference"] > SPARSE_TOLERANCE
    assert summary["rows_exceeding_1e_12"] == 1
    assert summary["passed"] is False


def test_exact_raw_component_formula_and_equal_pressure_weights() -> None:
    bars = _trace().loc[lambda frame: frame["symbol"].eq("AAA")].iloc[:3]
    raw = exact_pressure_raw_components(bars, centered_signed_progress_bps=20.0)
    assert set(raw) == {
        "signed_progress",
        "signed_efficiency",
        "mean_close_location",
        "boundary_slope",
    }
    assert raw["signed_progress"] == 20.0
    assert raw["signed_efficiency"] > 0.0
    assert (
        standardized_signed_pressure(
            {
                "signed_progress": 1.0,
                "signed_efficiency": 2.0,
                "mean_close_location": 3.0,
                "boundary_slope": 4.0,
            }
        )
        == 2.5
    )
    assert (
        apply_component_scale(
            1_000.0,
            {"center": 0.0, "scale": 10.0, "clip_lower": -5.0, "clip_upper": 5.0},
        )
        == 5.0
    )


def test_invalid_surface_does_not_impute_or_forward_fill_pressure() -> None:
    grid = build_dense_bar_grid(_trace())
    surface = invalid_dense_pressure_surface(
        grid,
        formula_hash="formula",
        preprocessing_hash="preprocessing",
        source_lineage_version="lineage",
    )
    assert surface["signed_pressure"].isna().all()
    assert not surface["pressure_valid"].any()
    assert (
        surface["missing_reason_code"] == "blocked_future_dependent_cross_sectional_population"
    ).all()


def test_five_bar_pressure_window_validity_and_coverage_gate() -> None:
    episode = pd.DataFrame(
        {
            "stock": ["AAA"],
            "session": ["2024-06-03"],
            "checkpoint": [6],
            "partition": ["development"],
        }
    )
    dense = pd.DataFrame(
        {
            "stock": ["AAA"] * 5,
            "session": ["2024-06-03"] * 5,
            "checkpoint_index": [1, 2, 3, 4, 5],
            "pressure_valid": [True] * 5,
        }
    )
    complete = pressure_window_audit(episode, dense)
    assert complete.loc[0, "required_pressure_checkpoints"] == "1,2,3,4,5"
    assert bool(complete.loc[0, "complete_five_bar_pressure_window"])
    incomplete = pressure_window_audit(episode, dense.iloc[:4])
    assert incomplete.loc[0, "valid_pressure_bars"] == 4
    assert not bool(incomplete.loc[0, "complete_five_bar_pressure_window"])
    assert not evaluate_coverage_gate(incomplete).passed


def test_coverage_gate_passes_only_at_frozen_support() -> None:
    rows: list[dict[str, object]] = []
    development_months = [f"2024-{month:02d}" for month in range(1, 11)]
    assessment_months = [f"2025-{month:02d}" for month in range(1, 9)]
    for index in range(220):
        month = development_months[index % len(development_months)]
        rows.append(
            {
                "stock": f"S{index % 15:02d}",
                "session": f"{month}-{index % 22 + 1:02d}",
                "month": month,
                "partition": "development",
                "complete_five_bar_pressure_window": True,
            }
        )
    for index in range(180):
        month = assessment_months[index % len(assessment_months)]
        rows.append(
            {
                "stock": f"S{index % 15:02d}",
                "session": f"{month}-{index % 22 + 1:02d}",
                "month": month,
                "partition": "assessment",
                "complete_five_bar_pressure_window": True,
            }
        )
    assert evaluate_coverage_gate(pd.DataFrame(rows)).passed


def test_trigger_bar_exclusion_pressure_aggregates_and_absorption_signs() -> None:
    bars = _pretrigger_bars()
    trigger = bars.loc[bars["bar_ordinal"].eq(5)].iloc[0]
    episode = pd.DataFrame(
        {
            "stock": ["AAA"],
            "session": ["2024-06-03"],
            "checkpoint": [6],
            "signal_timestamp": [trigger["bar_complete_timestamp"]],
            "prospective_entry_timestamp": [trigger["bar_complete_timestamp"]],
            "m1_probability": [0.6],
            "partition": ["development"],
        }
    )
    features = build_pretrigger_feature_rows(episode, bars)
    pressure = bars.loc[bars["bar_ordinal"].between(0, 4), "signed_pressure"].to_numpy()
    assert features.loc[0, "primary_window_bar_ordinals"] == "0,1,2,3,4"
    assert bool(features.loc[0, "trigger_bar_excluded"])
    assert features.loc[0, "pressure_sum_25"] == pytest.approx(float(pressure.sum()))
    assert features.loc[0, "pressure_persistence_25"] == pytest.approx(
        pressure_persistence(pressure)
    )
    assert features.loc[0, "pressure_slope_25"] == pytest.approx(pressure_slope(pressure))
    assert signed_absorption_divergence(
        pressure_sum=2.0, pressure_z=2.0, price_z=0.5
    ) == pytest.approx(1.5)
    assert signed_absorption_divergence(
        pressure_sum=-2.0, pressure_z=-2.0, price_z=-0.5
    ) == pytest.approx(-1.5)
    assert activity_without_displacement(
        pressure_sum=1.0,
        activity=np.ones(5),
        net_return=0.0,
        path_length=0.01,
    ) == pytest.approx(1.0)
    assert activity_without_displacement(
        pressure_sum=-1.0,
        activity=np.ones(5),
        net_return=0.0,
        path_length=0.01,
    ) == pytest.approx(-1.0)
    assert score_sign_persistence(np.asarray([-1.0, 0.0, 1.0, 1.0, 1.0])) == 0.4


def test_composite_score_and_development_only_preprocessing_remain_frozen() -> None:
    development = _raw_score_rows()
    parameters = fit_quiet_score_parameters(development)
    transformed = apply_quiet_score_parameters(development, parameters)
    columns = [f"{name}__clipped_z" for name in QUIET_SIGNED_COMPONENTS]
    expected = transformed.loc[:, columns].mean(axis=1)
    assert np.allclose(transformed["signed_accumulation_core_25"], expected)
    assert np.allclose(
        transformed["quiet_absorption_score_25"],
        transformed["quietness_25"] * expected,
    )
    contaminated = development.copy()
    contaminated.loc[0, "partition"] = "assessment"
    with pytest.raises(ValueError, match="development rows only"):
        fit_quiet_score_parameters(contaminated)


def test_frozen_call_put_abstain_policy_is_unchanged() -> None:
    probabilities = np.asarray([0.0, 0.48, 0.52, 1.0])
    boundary = freeze_confidence_boundary(probabilities, target_coverage=0.5, minimum_actions=2)
    actions = selective_actions(probabilities, boundary)
    assert actions.tolist() == ["PUT", "ABSTAIN", "ABSTAIN", "CALL"]


def test_missing_dependency_and_phase_decision_fail_closed() -> None:
    decision = phase1_decision(
        lineage_found=True,
        binding_dependency_class="D",
        compatibility_passed=False,
        causality_passed=False,
        coverage_passed=False,
        reproducibility_passed=True,
    )
    assert decision == "blocked_dense_pressure_upstream_dependency"
    with pytest.raises(RuntimeError, match="not authorized"):
        assert_phase2_authorized(decision)


def test_protected_and_opened_holdout_boundaries() -> None:
    valid = pd.DataFrame({"session": ["2024-01-02", "2025-08-22"]})
    assert validate_authorized_sessions(valid)["passed"] is True
    for forbidden in ("2025-09-01", "2026-01-02"):
        with pytest.raises(ValueError, match="protected|holdout"):
            validate_authorized_sessions(pd.DataFrame({"session": [forbidden]}))


def test_scaling_refuses_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_component_scale(
            1.0,
            {"center": 0.0, "scale": 0.0, "clip_lower": -5.0, "clip_upper": 5.0},
        )
    assert math.isnan(
        standardized_signed_pressure(
            {
                "signed_progress": math.nan,
                "signed_efficiency": 0.0,
                "mean_close_location": 0.0,
                "boundary_slope": 0.0,
            }
        )
    )
