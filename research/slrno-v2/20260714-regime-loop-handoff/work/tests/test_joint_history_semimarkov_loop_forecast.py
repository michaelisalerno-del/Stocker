from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_joint_history_semimarkov_loop_forecast.py"
)
SPEC = importlib.util.spec_from_file_location(
    "joint_history_semimarkov_loop_forecast_under_test", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
joint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = joint
SPEC.loader.exec_module(joint)


def _delta_zero() -> np.ndarray:
    distribution = np.zeros((1, joint.MAX_HORIZON + 1), dtype=float)
    distribution[0, 0] = 1.0
    return distribution


def _transition(duration_probabilities: dict[int, float]) -> np.ndarray:
    transition = np.zeros((1, joint.DURATION_BUCKETS), dtype=float)
    for duration, probability in duration_probabilities.items():
        transition[0, duration - 1] = probability
    return transition


def _brute_force_distribution(
    transitions: list[dict[int, float]], horizon: int
) -> np.ndarray:
    expected = np.zeros(horizon + 1, dtype=float)
    choices = [list(transition.items()) for transition in transitions]
    for route in itertools.product(*choices):
        total_duration = sum(duration for duration, _ in route)
        probability = np.prod([probability for _, probability in route])
        if total_duration <= horizon:
            expected[total_duration] += probability
    return expected


def _uniform_destination_parameters() -> dict[str, np.ndarray]:
    token_count = joint.HISTORY_VALUES * joint.HISTORY_VALUES * joint.K
    return {
        "history_intercept": np.zeros(joint.DESTINATIONS, dtype=float),
        "history_coef": np.zeros(
            (joint.DESTINATIONS, token_count), dtype=float
        ),
    }


def _constant_kernel(pmf: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "state_dest_pmf": np.broadcast_to(
            pmf, (joint.K, joint.DESTINATIONS, joint.DURATION_BUCKETS)
        ).copy(),
        "order2_pmf": np.broadcast_to(
            pmf,
            (
                joint.HISTORY_VALUES,
                joint.K,
                joint.DESTINATIONS,
                joint.DURATION_BUCKETS,
            ),
        ).copy(),
        "order3_pmf": np.broadcast_to(
            pmf,
            (
                joint.HISTORY_VALUES,
                joint.HISTORY_VALUES,
                joint.K,
                joint.DESTINATIONS,
                joint.DURATION_BUCKETS,
            ),
        ).copy(),
    }


def _empty_count_tensors() -> dict[str, np.ndarray]:
    return {
        "state_dest_counts": np.zeros(
            (joint.K, joint.DESTINATIONS, joint.DURATION_BUCKETS),
            dtype=np.int64,
        ),
        "order2_counts": np.zeros(
            (
                joint.HISTORY_VALUES,
                joint.K,
                joint.DESTINATIONS,
                joint.DURATION_BUCKETS,
            ),
            dtype=np.int64,
        ),
        "order3_counts": np.zeros(
            (
                joint.HISTORY_VALUES,
                joint.HISTORY_VALUES,
                joint.K,
                joint.DESTINATIONS,
                joint.DURATION_BUCKETS,
            ),
            dtype=np.int64,
        ),
        "fit_rows": np.asarray([0], dtype=np.int64),
        "terminal_rows_excluded": np.asarray([0], dtype=np.int64),
    }


def test_advance_distribution_matches_hand_calculation() -> None:
    distribution = np.zeros((1, joint.MAX_HORIZON + 1), dtype=float)
    distribution[0, 0] = 0.5
    distribution[0, 2] = 0.5
    transition = _transition({1: 0.25, 3: 0.75})

    observed = joint.advance_distribution(distribution, transition)[0]
    expected = np.zeros(joint.MAX_HORIZON + 1, dtype=float)
    expected[1] = 0.125
    expected[3] = 0.5
    expected[5] = 0.375

    np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0.0)
    assert observed.sum() == 1.0


def test_dynamic_programming_matches_brute_force_enumeration() -> None:
    transitions = [
        {1: 0.2, 2: 0.8},
        {2: 0.6, 4: 0.4},
        {1: 0.3, 3: 0.7},
    ]
    observed = _delta_zero()
    for transition in transitions:
        observed = joint.advance_distribution(
            observed, _transition(transition)
        )

    expected = _brute_force_distribution(transitions, joint.MAX_HORIZON)
    np.testing.assert_allclose(observed[0], expected, atol=1e-15, rtol=0.0)
    assert observed.sum() == 1.0


def test_duration_24_and_above_share_overflow_bucket_and_are_excluded() -> None:
    buckets = joint.dwell_bucket(np.asarray([1, 23, 24, 25, 100]))
    np.testing.assert_array_equal(buckets, np.asarray([0, 22, 23, 23, 23]))

    transition = _transition({1: 0.4, 24: 0.6})
    observed = joint.advance_distribution(_delta_zero(), transition)[0]

    assert observed[1] == 0.4
    assert observed[24] == 0.0
    assert observed.sum() == 0.4


def test_route_probabilities_are_monotone_and_bounded_by_path_mass() -> None:
    pmf = np.zeros(joint.DURATION_BUCKETS, dtype=float)
    pmf[3] = 0.5  # four bars
    pmf[7] = 0.5  # eight bars
    frozen_pmf = np.broadcast_to(pmf, (joint.K, joint.DURATION_BUCKETS)).copy()
    anchors = pd.DataFrame(
        {
            "previous_state_2": [joint.END_STATE],
            "previous_state_1": [joint.END_STATE],
        }
    )

    forecast = joint.route_probabilities(
        anchors,
        (0, 1, 0),
        _uniform_destination_parameters(),
        _constant_kernel(pmf),
        frozen_pmf,
    )

    path_probability = 1.0 / joint.DESTINATIONS**2
    assert forecast["path_probability"][0] == path_probability
    for model in (
        "history_frozen_state_timed",
        "history_destination_timed",
        "history_order2_timed",
        "history_joint_timed",
    ):
        by_horizon = [
            forecast["probabilities"][horizon][model][0]
            for horizon in joint.HORIZONS
        ]
        np.testing.assert_allclose(
            by_horizon,
            [0.0, 0.75 * path_probability, path_probability],
            atol=1e-15,
            rtol=0.0,
        )
        assert 0.0 <= by_horizon[0] <= by_horizon[1] <= by_horizon[2]
        assert by_horizon[2] <= forecast["path_probability"][0]
        assert (
            forecast["distributions"][model].sum()
            <= forecast["path_probability"][0] + 1e-15
        )


def test_labels_and_completion_times_respect_route_and_horizon_boundaries() -> None:
    anchors = pd.DataFrame(
        {
            "future_state_1": [2, 2, 2],
            "future_state_2": [3, 3, 3],
            "future_state_3": [1, 1, 0],
            "duration": [2, 2, 2],
            "future_duration_1": [1, 2, 1],
            "future_duration_2": [3, 3, 3],
            # The closing state's duration is not part of time-to-completion.
            "future_duration_3": [99, 99, 99],
        }
    )

    label, completion = joint.oriented_actual_completion(
        anchors, (1, 2, 3, 1)
    )

    np.testing.assert_array_equal(label, np.asarray([True, True, False]))
    np.testing.assert_array_equal(completion, np.asarray([6, 7, 6]))
    np.testing.assert_array_equal(
        label & (completion <= 6), np.asarray([True, False, False])
    )
    np.testing.assert_array_equal(
        label & (completion <= 7), np.asarray([True, True, False])
    )


def test_scoring_run_loader_filters_off_grid_and_after_bar_53(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_runs.csv"
    frame = pd.DataFrame(
        {
            "symbol_norm": ["AAA"] * 4,
            "session_date": ["2024-07-01"] * 4,
            "state": [0, 1, 0, 2],
            "duration": [1, 1, 1, 1],
            "start_pos": [0, 1, 2, 3],
            "start_timestamp": [
                "2024-07-01T13:30:00Z",  # 09:30 ET, bar 0
                "2024-07-01T13:31:00Z",  # off the five-minute grid
                "2024-07-01T17:55:00Z",  # 13:55 ET, bar 53
                "2024-07-01T18:00:00Z",  # 14:00 ET, bar 54
            ],
            "previous_state_1": [joint.END_STATE, 0, 1, 0],
            "previous_state_2": [joint.END_STATE, joint.END_STATE, 0, 1],
            "next_state": [1, 0, 2, np.nan],
            "has_next_state": [True, True, True, False],
        }
    )
    frame.to_csv(path, index=False)

    observed = joint.load_runs(path, 2024, "synthetic", scoring=True)

    np.testing.assert_array_equal(
        observed["bar_index_in_session"].to_numpy(), np.asarray([0, 53])
    )
    np.testing.assert_array_equal(observed["state"].to_numpy(), np.asarray([0, 0]))
    np.testing.assert_array_equal(observed["anchor_id"].to_numpy(), np.asarray([0, 1]))
    assert observed["clock_grid_valid"].all()


def test_hierarchical_smoothing_normalizes_and_backs_off_exactly() -> None:
    frozen_pmf = np.zeros((joint.K, joint.DURATION_BUCKETS), dtype=float)
    frozen_pmf[:, 0] = 0.25
    frozen_pmf[:, 2] = 0.75
    counts = _empty_count_tensors()
    counts["state_dest_counts"][0, 1, 1] = 3
    counts["order2_counts"][2, 0, 1, 4] = 2
    counts["order3_counts"][7, 2, 0, 1, 6] = 1

    arrays = joint.smooth_dwell_counts(
        counts, frozen_pmf, strengths=(2.0, 3.0, 4.0)
    )
    joint.verify_kernel_arrays(arrays)

    state_expected = (
        counts["state_dest_counts"][0, 1] + 2.0 * frozen_pmf[0]
    ) / 5.0
    np.testing.assert_allclose(arrays["state_dest_pmf"][0, 1], state_expected)
    np.testing.assert_allclose(arrays["state_dest_pmf"][0, 2], frozen_pmf[0])
    np.testing.assert_allclose(arrays["order2_pmf"][8, 0, 1], state_expected)

    order2_expected = (
        counts["order2_counts"][2, 0, 1] + 3.0 * state_expected
    ) / 5.0
    np.testing.assert_allclose(arrays["order2_pmf"][2, 0, 1], order2_expected)
    np.testing.assert_allclose(arrays["order3_pmf"][8, 2, 0, 1], order2_expected)

    order3_expected = (
        counts["order3_counts"][7, 2, 0, 1] + 4.0 * order2_expected
    ) / 5.0
    np.testing.assert_allclose(arrays["order3_pmf"][7, 2, 0, 1], order3_expected)
    for name in ("state_pmf", "state_dest_pmf", "order2_pmf", "order3_pmf"):
        np.testing.assert_allclose(arrays[name].sum(axis=-1), 1.0, atol=1e-15)


def test_dwell_count_tensors_exclude_terminal_session_runs() -> None:
    train = pd.DataFrame(
        {
            "previous_state_2": [joint.END_STATE, joint.END_STATE],
            "previous_state_1": [joint.END_STATE, 0],
            "state": [0, 1],
            "next_outcome": [1, joint.END_STATE],
            "duration": [2, 5],
        }
    )

    counts = joint.dwell_count_tensors(train)

    assert counts["fit_rows"].item() == 1
    assert counts["terminal_rows_excluded"].item() == 1
    assert counts["state_dest_counts"].sum() == 1
    assert counts["state_dest_counts"][0, 1, 1] == 1
    assert counts["state_dest_counts"][:, joint.END_STATE].sum() == 0
