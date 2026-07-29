from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.m1c_opening_market_transition_v1 import (
    M1C_HIGH_MOVEMENT_THRESHOLD_V1,
    assign_opening_response_quintile_v1,
    freeze_opening_response_quintiles_v1,
    freeze_opening_thresholds_v1,
    validate_prior_population_reconciliation_v1,
)


def _market_predictors() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(25):
        rows.append(
            {
                "session": f"2024-02-{index + 1:02d}",
                "checkpoint": 6,
                "complete_v1": True,
                "market_opening_return_v1": float(index - 12) / 1_000.0,
                "market_opening_range_v1": float(index + 10) / 1_000.0,
                "market_overnight_gap_v1": float(index - 8) / 2_000.0,
                "market_total_transition_v1": float(index - 10) / 800.0,
            }
        )
    rows.append(
        {
            "session": "2025-01-02",
            "checkpoint": 6,
            "complete_v1": True,
            "market_opening_return_v1": 999.0,
            "market_opening_range_v1": 999.0,
            "market_overnight_gap_v1": 999.0,
            "market_total_transition_v1": 999.0,
        }
    )
    return pd.DataFrame(rows)


def test_opening_thresholds_use_only_fixed_2024_predictors() -> None:
    source = _market_predictors()
    frozen = freeze_opening_thresholds_v1(source)
    development = source.loc[source["session"].astype(str).str.startswith("2024")]

    assert frozen.calibration_complete_v1
    assert frozen.market_opening_return_support_v1 == 25
    assert frozen.market_opening_return_q10_v1 == pytest.approx(
        np.quantile(
            development["market_opening_return_v1"],
            0.10,
            method="linear",
        )
    )
    assert frozen.market_opening_return_q90_v1 == pytest.approx(
        np.quantile(
            development["market_opening_return_v1"],
            0.90,
            method="linear",
        )
    )
    assert frozen.market_opening_range_q75_v1 == pytest.approx(
        np.quantile(
            development["market_opening_range_v1"],
            0.75,
            method="linear",
        )
    )

    changed = source.copy()
    changed.loc[changed["session"].eq("2025-01-02"), "future_outcome"] = -1e12
    assert freeze_opening_thresholds_v1(changed) == frozen


def test_opening_threshold_calibration_rejects_protected_sessions() -> None:
    source = pd.concat(
        [
            _market_predictors(),
            pd.DataFrame(
                [
                    {
                        "session": "2026-01-02",
                        "checkpoint": 6,
                        "complete_v1": True,
                        "market_opening_return_v1": 0.0,
                        "market_opening_range_v1": 0.01,
                        "market_overnight_gap_v1": 0.0,
                        "market_total_transition_v1": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="protected 2026"):
        freeze_opening_thresholds_v1(source)


def test_response_quintiles_use_only_valid_development_severe_predictors() -> None:
    rows = [
        {
            "session": f"2024-03-{index + 1:02d}",
            "checkpoint": 6,
            "tail_phase_v1": "FIRST_ENTRY",
            "M1C_probability": M1C_HIGH_MOVEMENT_THRESHOLD_V1,
            "opening_market_transition_state_v1": (
                "NEGATIVE_SEVERE_OPENING_TRANSITION"
                if index % 2
                else "POSITIVE_SEVERE_OPENING_TRANSITION"
            ),
            "stock_opening_response_complete_v1": True,
            "stock_relative_opening_response_v1": float(index),
        }
        for index in range(10)
    ]
    rows.extend(
        [
            {
                **rows[0],
                "stock_relative_opening_response_v1": 999.0,
                "opening_market_transition_state_v1": "NORMAL_OPENING",
            },
            {
                **rows[0],
                "stock_relative_opening_response_v1": 999.0,
                "M1C_probability": M1C_HIGH_MOVEMENT_THRESHOLD_V1 - 0.01,
            },
            {
                **rows[0],
                "session": "2025-01-02",
                "stock_relative_opening_response_v1": -999.0,
            },
        ]
    )
    frozen = freeze_opening_response_quintiles_v1(pd.DataFrame(rows))

    assert frozen.support_v1 == 10
    assert (
        frozen.q20_v1,
        frozen.q40_v1,
        frozen.q60_v1,
        frozen.q80_v1,
    ) == pytest.approx(np.quantile(np.arange(10.0), [0.2, 0.4, 0.6, 0.8]))
    assert assign_opening_response_quintile_v1(frozen.q20_v1, frozen) == "Q1"
    assert assign_opening_response_quintile_v1(9.0, frozen) == "Q5"


def test_prior_population_reconciliation_requires_every_episode_and_reason() -> None:
    reconciliation = pd.DataFrame(
        [
            {
                "period": "assessment",
                "stock": "AAA",
                "session": "2025-01-02",
                "checkpoint": 6,
                "fresh_episode_id": "fresh-1",
                "tail_phase_v1": "FIRST_ENTRY",
                "market_shock_state_v1": "POSITIVE_SHOCK_ONSET",
                "included_in_primary_signed_shock_population_v1": True,
                "included_in_tail_phase_diagnostics_v1": True,
                "inclusion_exclusion_reason_v1": "included:canonical_fresh_episode",
            },
            {
                "period": "assessment",
                "stock": "BBB",
                "session": "2025-01-03",
                "checkpoint": 10,
                "fresh_episode_id": None,
                "tail_phase_v1": "RE_ENTRY",
                "market_shock_state_v1": "NEGATIVE_SHOCK_ONSET",
                "included_in_primary_signed_shock_population_v1": False,
                "included_in_tail_phase_diagnostics_v1": True,
                "inclusion_exclusion_reason_v1": (
                    "different_population_definition:"
                    "minimum_episode_spacing_not_met:20<30"
                ),
            },
        ]
    )
    expected_tail_keys = {
        ("assessment", "AAA", "2025-01-02", 6),
        ("assessment", "BBB", "2025-01-03", 10),
    }
    expected_primary_ids = {"fresh-1"}

    validate_prior_population_reconciliation_v1(
        reconciliation,
        expected_tail_keys=expected_tail_keys,
        expected_primary_episode_ids=expected_primary_ids,
    )

    with pytest.raises(ValueError, match="tail diagnostic keys"):
        validate_prior_population_reconciliation_v1(
            reconciliation.iloc[:1],
            expected_tail_keys=expected_tail_keys,
            expected_primary_episode_ids=expected_primary_ids,
        )
