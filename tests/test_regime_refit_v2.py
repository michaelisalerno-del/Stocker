from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.regime_refit_v2 import (
    DURATION_ONLY_MODEL_ID,
    FULL_REFIT_MODEL_ID,
    RefitConfig,
    build_duration_only_repair,
    deterministic_parameter_hash,
    deterministic_stride_indices,
    fit_full_right_censored_refit,
)
from stocker_research.regime_validity_v2 import (
    CleaningVariant,
    SemiMarkovParameters,
)


def _parameters() -> SemiMarkovParameters:
    return SemiMarkovParameters(
        means=np.asarray([[-1.0], [1.0]]),
        variances=np.asarray([[0.5], [0.5]]),
        duration_hazard=np.full((2, 4), 0.2),
        transitions=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        initial=np.asarray([0.5, 0.5]),
        occupancy=np.asarray([0.5, 0.5]),
    )


def _panel(rows: int = 24) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    state_signal = np.tile(np.asarray([-1.5, -1.0, 1.0, 1.5]), rows // 4 + 1)[:rows]
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "session": "2024-01-02",
            "segment_id": "TEST::2024-01-02::segment_00",
            "segment_index": 0,
            "bar_ordinal": np.arange(rows),
            "bar_start_timestamp": [
                start + pd.Timedelta(minutes=5 * index) for index in range(rows)
            ],
            "bar_complete_timestamp": [
                start + pd.Timedelta(minutes=5 * (index + 1)) for index in range(rows)
            ],
            "session_source_complete": True,
            "expected_session_bars": rows,
            "regime_log_activity_12": state_signal + 2.0,
            "signed_efficiency_12": state_signal,
            "f0": state_signal,
        }
    )


def test_historical_stride_policy_is_deterministic_and_not_random() -> None:
    first = deterministic_stride_indices(424_583, nominal_maximum_rows=200_000)
    second = deterministic_stride_indices(424_583, nominal_maximum_rows=200_000)

    np.testing.assert_array_equal(first, second)
    assert first[0] == 0
    assert np.unique(np.diff(first)).tolist() == [2]
    assert len(first) == 212_292


def test_duration_only_repair_preserves_every_non_duration_parameter() -> None:
    frozen = _parameters()
    panel = _panel(4)
    labels = np.asarray([0, 0, 1, 1])

    repaired = build_duration_only_repair(
        frozen_parameters=frozen,
        panel=panel,
        frozen_training_labels=labels,
        maximum_age=4,
    )

    assert repaired.model_id == DURATION_ONLY_MODEL_ID
    np.testing.assert_array_equal(repaired.parameters.means, frozen.means)
    np.testing.assert_array_equal(repaired.parameters.variances, frozen.variances)
    np.testing.assert_array_equal(repaired.parameters.transitions, frozen.transitions)
    np.testing.assert_array_equal(repaired.parameters.initial, frozen.initial)
    np.testing.assert_array_equal(repaired.parameters.occupancy, frozen.occupancy)
    assert repaired.parameters.duration_hazard.shape == (2, 4)
    assert not np.any(repaired.parameters.duration_hazard == 1.0)


def test_parameter_hash_changes_when_one_float_changes() -> None:
    original = _parameters()
    changed = _parameters()
    changed.means[0, 0] = np.nextafter(changed.means[0, 0], 0.0)

    assert deterministic_parameter_hash(original) != deterministic_parameter_hash(changed)


def test_full_refit_is_deterministic_and_records_configuration() -> None:
    panel = _panel()
    config = RefitConfig(
        state_count=2,
        seed=20260710,
        nominal_maximum_rows=20,
        maximum_age=24,
        cleaning_variant=CleaningVariant.CLEANING_1,
        batch_size=8,
        n_init=2,
        max_iter=50,
    )

    first = fit_full_right_censored_refit(
        panel,
        feature_names=("f0",),
        config=config,
    )
    second = fit_full_right_censored_refit(
        panel,
        feature_names=("f0",),
        config=config,
    )

    assert first.model_id == FULL_REFIT_MODEL_ID
    assert first.training_row_hash == second.training_row_hash
    assert first.parameter_hash == second.parameter_hash
    np.testing.assert_array_equal(first.raw_labels, second.raw_labels)
    np.testing.assert_array_equal(first.cleaned_labels, second.cleaned_labels)
    np.testing.assert_array_equal(first.semantic_labels, second.semantic_labels)
    assert first.effective_configuration["seed"] == 20260710
    assert first.effective_configuration["cleaning_variant"] == "CLEANING_1"
    assert first.kmeans_iterations >= 1
    assert first.kmeans_converged


def test_full_refit_preprocessing_fits_on_recorded_training_rows_only() -> None:
    panel = _panel()
    panel.loc[12:, "f0"] = 1_000_000.0
    training_indices = np.arange(12, dtype=np.int64)
    config = RefitConfig(
        state_count=2,
        seed=20260710,
        nominal_maximum_rows=20,
        maximum_age=24,
        batch_size=8,
        n_init=2,
        max_iter=50,
    )

    result = fit_full_right_censored_refit(
        panel,
        feature_names=("f0",),
        config=config,
        training_indices=training_indices,
    )

    expected_median = float(panel.iloc[training_indices]["f0"].median())
    assert result.preprocessing.medians[0] == expected_median
    assert result.preprocessing.medians[0] != float(panel["f0"].median())


def test_full_refit_rejects_outcome_columns() -> None:
    panel = _panel()
    panel["future_return"] = 0.0
    config = RefitConfig(
        state_count=2,
        seed=1,
        nominal_maximum_rows=20,
        maximum_age=24,
        batch_size=8,
        n_init=1,
        max_iter=10,
    )

    with pytest.raises(ValueError, match="outcome"):
        fit_full_right_censored_refit(
            panel,
            feature_names=("f0", "future_return"),
            config=config,
        )
