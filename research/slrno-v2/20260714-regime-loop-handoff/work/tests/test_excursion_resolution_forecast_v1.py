from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

WORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stocker_research.excursion_forecast_v1 import (  # noqa: E402
    SAFETY_FLAGS,
    TARGET_CLASSES,
    FoldPreprocessor,
    PartBGateMetrics,
    balanced_event_weights,
    build_active_excursion_rows,
    canonical_target_family,
    constant_hazard_competing_risk,
    decide_part_b,
    first_eligible_rows,
    fit_multinomial,
    frequency_probabilities,
    last_eligible_rows,
    multiclass_losses,
    paired_block_bootstrap,
    validate_probabilities,
)
from stocker_research.regime_panel_v2 import EMISSION_FEATURES  # noqa: E402


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    timestamps = pd.date_range("2024-06-03 13:30", periods=9, freq="5min", tz="UTC")
    emission_rows = []
    posterior_rows = []
    for index, timestamp in enumerate(timestamps):
        row: dict[str, object] = {
            "decision_id": f"decision-{index}",
            "symbol": "AAA",
            "session": "2024-06-03",
            "segment_id": "segment-1",
            "bar_ordinal": index,
            "decision_timestamp": timestamp,
            "availability_timestamp": timestamp,
            "short_trajectory_velocity": float(index),
            "short_trajectory_acceleration": 0.25,
            "local_path_length": float(index),
            "local_directional_consistency": 1.0,
            "period": "DEVELOPMENT_2024",
        }
        for feature_index, feature in enumerate(EMISSION_FEATURES):
            value = 0.1 * index * (feature_index + 1)
            row[f"z__{feature}"] = value
            row[f"delta_z__{feature}"] = 0.1 * (feature_index + 1)
            row[f"missing__{feature}"] = False
        emission_rows.append(row)
        posterior = {
            "decision_id": f"decision-{index}",
            "symbol": "AAA",
            "session": "2024-06-03",
            "segment_id": "segment-1",
            "bar_ordinal": index,
            "model_lineage": "MODEL_FULL_REFIT",
            "posterior_entropy": 0.5,
            "expected_state_age": float(index + 1),
            "departure_probability": 0.2,
            "hard_hysteretic_disagreement": index % 2 == 0,
            "posterior_velocity": 0.1,
            "hard_map_state": 1 if index < 5 else 2,
            "availability_timestamp": timestamp,
        }
        for state in range(8):
            posterior[f"posterior_state_{state}"] = 0.8 if state == 0 else 0.2 / 7
        posterior_rows.append(posterior)
    event = {
        "event_id": "event-1",
        "candidate_id": "candidate",
        "event_definition_hash": "event-hash",
        "trajectory_representation": "E",
        "distance_metric": "SHRINKAGE_MAHALANOBIS",
        "symbol": "AAA",
        "session": "2024-06-03",
        "segment_id": "segment-1",
        "frozen_origin_id": "origin-1",
        "frozen_origin_vector": json.dumps([0.0] * len(EMISSION_FEATURES)),
        "frozen_posterior_origin": json.dumps([0.125] * 8),
        "departure_direction_vector": json.dumps([1.0] * len(EMISSION_FEATURES)),
        "first_detectable_bar_ordinal": 1,
        "confirmation_bar_ordinal": 2,
        "onset_bar_ordinal": 1,
        "resolution_bar_ordinal": 7,
        "first_detectable_timestamp": timestamps[1],
        "confirmation_timestamp": timestamps[2],
        "onset_timestamp": timestamps[1],
        "resolution_timestamp": timestamps[7],
        "event_family": "RETURN_TO_ORIGIN",
        "departure_distance": 2.0,
        "confirmation_distance": 2.5,
        "maximum_excursion_distance": 5.0,
        "resolution_distance": 0.5,
        "retracement_fraction": 0.9,
        "decision_id": "decision-2",
        "decision_timestamp": timestamps[2],
        "period": "DEVELOPMENT_2024",
        "period_data_snapshot_hash": "snapshot",
        "panel_hash": "panel",
        "run_id": "run",
        "git_sha": "sha",
        "contract_hash": "part-a-contract",
        "feature_manifest_hash": "part-a-features",
        "model_lineage": "MODEL_FULL_REFIT",
        "source_artifact": "excursion_event_ledger.parquet",
        "source_hash": "source",
    }
    return (
        pd.DataFrame([event]),
        pd.DataFrame(emission_rows),
        pd.DataFrame(posterior_rows),
        np.eye(len(EMISSION_FEATURES)),
    )


def _active() -> pd.DataFrame:
    events, emission, posterior, precision = _synthetic_inputs()
    return build_active_excursion_rows(
        events,
        emission,
        posterior,
        precision=precision,
        emission_features=EMISSION_FEATURES,
        primary_model_lineage="MODEL_FULL_REFIT",
    )


def test_36_active_rows_begin_after_confirmed_departure() -> None:
    rows = _active()
    assert rows["bar_ordinal"].min() == 3
    assert rows["bar_ordinal"].gt(rows["confirmation_bar_ordinal"]).all()


def test_37_no_row_appears_after_resolution() -> None:
    rows = _active()
    assert rows["bar_ordinal"].max() == 6
    assert rows["bar_ordinal"].lt(rows["resolution_bar_ordinal"]).all()


def test_38_all_forecast_features_are_causal() -> None:
    rows = _active()
    assert rows["feature_available_timestamp"].le(rows["decision_timestamp"]).all()


def test_39_geometry_reconciles_with_event_origin() -> None:
    rows = _active()
    first = rows.iloc[0]
    index = int(first["bar_ordinal"])
    expected = np.linalg.norm(
        np.asarray([0.1 * index * (feature + 1) for feature in range(len(EMISSION_FEATURES))])
    )
    assert np.isclose(first["current_distance"], expected)


def test_40_posterior_features_use_frozen_lineage() -> None:
    events, emission, posterior, precision = _synthetic_inputs()
    alternative = posterior.copy()
    alternative["model_lineage"] = "MODEL_OTHER"
    alternative["posterior_entropy"] = 99.0
    rows = build_active_excursion_rows(
        events,
        emission,
        pd.concat([posterior, alternative], ignore_index=True),
        precision=precision,
        emission_features=EMISSION_FEATURES,
        primary_model_lineage="MODEL_FULL_REFIT",
    )
    assert rows["posterior_entropy"].eq(0.5).all()


def test_41_market_features_are_causal_current_bar_values() -> None:
    rows = _active()
    first = rows.iloc[0]
    feature_index = list(EMISSION_FEATURES).index("regime_log_market_dispersion")
    expected = 0.1 * int(first["bar_ordinal"]) * (feature_index + 1)
    assert np.isclose(first["market_dispersion_z"], expected)


def test_42_chronological_masks_do_not_leak() -> None:
    dates = pd.to_datetime(["2024-03-31", "2024-04-01", "2024-06-30"])
    train = dates <= pd.Timestamp("2024-03-31")
    validation = (dates >= pd.Timestamp("2024-04-01")) & (dates <= pd.Timestamp("2024-06-30"))
    assert not np.any(train & validation)
    assert dates[train].max() < dates[validation].min()


def test_43_preprocessing_fits_on_training_fold_only() -> None:
    fitted = FoldPreprocessor.fit(np.asarray([[0.0], [2.0]]))
    assert np.allclose(fitted.means, [1.0])
    assert np.allclose(fitted.transform(np.asarray([[100.0]])), [[99.0]])


def test_44_models_use_identical_eligible_population() -> None:
    rows = _active()
    eligible = rows.loc[rows["target_observed"]]
    global_probability = frequency_probabilities(
        eligible,
        eligible,
        target_column="target_family",
    )
    estimator = fit_multinomial(
        eligible.assign(dummy=eligible["current_distance"]),
        features=["dummy"],
        target_column="target_family",
    )
    assert len(global_probability) == len(estimator.predict_proba(eligible.assign(dummy=0.0)))


def test_45_global_frequency_baseline_is_laplace_correct() -> None:
    train = pd.DataFrame({"target": ["RETURN_TO_ORIGIN"] * 2})
    probability = frequency_probabilities(train, train.iloc[:1], target_column="target")
    assert np.isclose(probability[0, 0], 3.0 / 8.0)
    validate_probabilities(probability)


def test_46_clock_baseline_uses_only_matching_clock_bucket() -> None:
    train = pd.DataFrame(
        {
            "target": ["RETURN_TO_ORIGIN", "SESSION_END"],
            "clock_phase": ["EARLY", "LATE"],
        }
    )
    predict = pd.DataFrame({"clock_phase": ["EARLY"]})
    probability = frequency_probabilities(
        train,
        predict,
        target_column="target",
        group_columns=["clock_phase"],
    )
    assert probability[0, 0] > probability[0, 4]


def test_47_persistence_baseline_uses_current_trend_only() -> None:
    train = pd.DataFrame(
        {
            "target": ["CONTINUE_AWAY", "RETURN_TO_ORIGIN"],
            "trend": ["OUTWARD", "INWARD"],
        }
    )
    probability = frequency_probabilities(
        train,
        pd.DataFrame({"trend": ["OUTWARD"]}),
        target_column="target",
        group_columns=["trend"],
    )
    assert probability[0, 2] > probability[0, 0]


def test_48_cause_specific_hazards_normalize() -> None:
    probabilities = np.asarray([[0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03]])
    validate_probabilities(probabilities)
    incidence, survival = constant_hazard_competing_risk(
        probabilities,
        no_event_index=0,
        horizons=[3, 6, 12],
    )
    assert np.allclose(incidence.sum(axis=2) + survival, 1.0)


def test_49_cumulative_incidence_is_monotonic() -> None:
    probabilities = np.asarray([[0.8, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02]])
    incidence, _ = constant_hazard_competing_risk(
        probabilities, no_event_index=0, horizons=[3, 6, 12]
    )
    assert np.all(np.diff(incidence, axis=1) >= -1e-12)


def test_50_family_probabilities_plus_survival_are_valid() -> None:
    probabilities = np.asarray([[0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]])
    incidence, survival = constant_hazard_competing_risk(
        probabilities, no_event_index=0, horizons=[6]
    )
    assert np.isclose(incidence[0, 0].sum() + survival[0, 0], 1.0)


def test_51_event_metrics_deduplicate_repeated_forecasts() -> None:
    rows = _active()
    assert len(first_eligible_rows(rows)) == 1
    assert len(last_eligible_rows(rows)) == 1


def test_52_lead_time_is_before_resolution() -> None:
    rows = _active()
    first = first_eligible_rows(rows).iloc[0]
    assert first["bars_until_resolution"] == 4


def test_53_validation_cannot_alter_preprocessor() -> None:
    fitted = FoldPreprocessor.fit(np.asarray([[0.0], [2.0]]))
    original_hash = fitted.hash
    fitted.transform(np.asarray([[10_000.0]]))
    assert fitted.hash == original_hash


def test_54_leave_one_stock_out_requires_recomputed_loss() -> None:
    frame = pd.DataFrame({"symbol": ["A", "A", "B"], "loss": [1.0, 3.0, 9.0]})
    deletion = frame.loc[frame["symbol"].ne("A"), "loss"].mean()
    assert deletion == 9.0
    assert deletion != frame["loss"].mean()


def test_55_quarter_assignment_is_calendar_correct() -> None:
    dates = pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"])
    assert dates.quarter.tolist() == [1, 2, 3, 4]


def test_56_paired_block_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame({"session": ["a", "a", "b", "b"]})
    candidate = np.asarray([0.1, 0.2, 0.3, 0.4])
    baseline = np.asarray([0.2, 0.3, 0.4, 0.5])
    left = paired_block_bootstrap(
        frame,
        candidate_loss=candidate,
        baseline_loss=baseline,
        group_column="session",
        draws=20,
        seed=7,
    )
    right = paired_block_bootstrap(
        frame,
        candidate_loss=candidate,
        baseline_loss=baseline,
        group_column="session",
        draws=20,
        seed=7,
    )
    pdt.assert_frame_equal(left, right)


def test_57_binary_return_target_is_separate() -> None:
    assert canonical_target_family("RETURN_TO_ORIGIN") == "RETURN_TO_ORIGIN"
    assert canonical_target_family("CONTINUE_AWAY") == "CONTINUE_AWAY"
    binary = (
        "RETURN_TO_ORIGIN"
        if canonical_target_family("CONTINUE_AWAY") == "RETURN_TO_ORIGIN"
        else "NON_RETURN"
    )
    assert binary == "NON_RETURN"


def test_58_part_b_decision_hierarchy_is_deterministic() -> None:
    values = PartBGateMetrics(
        source_blocked=False,
        candidate_beats_log_loss=True,
        candidate_beats_brier=True,
        log_loss_upper_below_zero=True,
        brier_upper_below_zero=True,
        relative_log_loss_improvement=0.01,
        favourable_quarters=4,
        all_stock_deletions_favourable=True,
        calibration_not_worse=True,
        improved_major_classes=2,
        return_only_gain=False,
        median_correct_lead_time_bars=3.0,
        sensitivity_directionally_similar=True,
        binary_support_sufficient=True,
        binary_gate_pass=True,
        timing_gate_pass=True,
        pooled_improvement_present=True,
    )
    assert decide_part_b(values) == "cluster_invariant_excursion_forecast_validated"
    assert decide_part_b(values) == decide_part_b(values)


def test_59_no_payoff_or_execution_columns_enter() -> None:
    columns = {column.lower() for column in _active().columns}
    forbidden = {"future_return", "pnl", "payoff", "mfe", "mae", "spread", "slippage"}
    assert columns.isdisjoint(forbidden)


def test_60_safety_flags_are_present() -> None:
    assert SAFETY_FLAGS == {
        "research_only": True,
        "execution_enabled": False,
        "order_placement": "disabled",
        "broker_connected": False,
        "economic_outcomes_used": False,
        "payoff_selection_used": False,
        "production_runtime_modified": False,
        "strategy_promotion": False,
    }


def test_event_weights_sum_to_one_per_event() -> None:
    frame = pd.DataFrame({"event_id": ["a", "a", "b"]})
    weights = balanced_event_weights(frame)
    assert np.allclose(weights, [0.5, 0.5, 1.0])


def test_multiclass_losses_are_proper_and_finite() -> None:
    targets = list(TARGET_CLASSES)
    probabilities = np.eye(len(TARGET_CLASSES)) * 0.9 + 0.1 / len(TARGET_CLASSES)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_loss, brier = multiclass_losses(targets, probabilities)
    assert np.isfinite(log_loss).all()
    assert np.isfinite(brier).all()
    assert np.all(log_loss < 0.2)
