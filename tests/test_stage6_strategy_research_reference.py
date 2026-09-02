"""STAGE 6 / STRATEGY REFERENCE for frozen cohort percentile and band semantics.

These helpers are test-only lineage evidence. Stage 5 runtime does not import or apply them.
"""

from __future__ import annotations

from math import isfinite

import pytest

MINIMUM_PRIOR_OBSERVATIONS = 30


def reference_cohort_percentile(
    current_pre_move_m: float, prior_qualifying_pre_move_m: list[float]
) -> float | None:
    assert isfinite(current_pre_move_m) and current_pre_move_m >= 0.0
    assert all(isfinite(value) and value >= 0.0 for value in prior_qualifying_pre_move_m)
    if len(prior_qualifying_pre_move_m) < MINIMUM_PRIOR_OBSERVATIONS:
        return None
    count = sum(value <= current_pre_move_m for value in prior_qualifying_pre_move_m)
    return 100.0 * count / len(prior_qualifying_pre_move_m)


def reference_band(percentile: float) -> str:
    if percentile <= 33.33:
        return "LOW"
    if percentile <= 66.67:
        return "MID"
    return "HIGH"


def test_stage6_reference_percentile_uses_less_than_or_equal_ties() -> None:
    history = [1.0] * 15 + [2.0] * 15

    assert reference_cohort_percentile(1.0, history) == 50.0


def test_stage6_reference_percentile_requires_30_prior_observations() -> None:
    assert reference_cohort_percentile(1.0, [1.0] * 29) is None


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [
        (33.33, "LOW"),
        (33.3300000001, "MID"),
        (66.67, "MID"),
        (66.6700000001, "HIGH"),
    ],
)
def test_stage6_reference_band_boundaries_are_exact(percentile: float, expected: str) -> None:
    assert reference_band(percentile) == expected


def test_stage6_reference_does_not_round_one_third_percentile() -> None:
    percentile = reference_cohort_percentile(10.0, [float(value) for value in range(1, 31)])

    assert percentile == pytest.approx(100.0 / 3.0)
    assert reference_band(percentile) == "MID"


def test_stage6_reference_unseen49_hood_rank_matches_frozen_ledger() -> None:
    # HOOD|2025-04-22|6 and its exact prior-20-session qualifying population.
    history = [
        1.13307437950998,
        1.01638600994221,
        0.520139320403003,
        1.05290217408396,
        0.653603476926764,
        1.11373459162757,
        1.19819684673582,
        1.02624385971943,
        0.714653207954931,
        0.6466291343547,
        0.559767962323959,
        1.64639951326801,
        1.46564328277018,
        1.0116985052533,
        0.867104005963024,
        0.733777192523018,
        0.501895931066927,
        1.21719454641158,
        1.35691004079688,
        1.25810973323133,
        0.521349556474786,
        0.564017288969056,
        0.955582054581471,
        1.12212665421029,
        0.994529262286479,
        0.821932221300254,
        0.547724099764726,
        2.08020245737097,
        1.19497626428145,
        0.571842409121946,
        0.681553698006495,
        0.517062567837534,
        0.812194056764132,
        0.81896515840008,
        0.865079403128148,
        1.04477944085885,
        0.924755484420779,
        0.913129871699216,
        0.558624762706318,
        1.04999576955118,
        0.933354802446138,
        0.982243889734966,
        0.983948377863543,
        0.596856827982767,
        0.485588060171765,
        1.95962662660837,
        1.18532362930428,
        0.82524187326742,
    ]

    percentile = reference_cohort_percentile(0.784827092404623, history)

    assert percentile == pytest.approx(33.3333333333333, abs=5e-14)
    assert reference_band(percentile) == "MID"
