from __future__ import annotations

import pandas as pd

from stocker_research.directed_economic_rotation import (
    FamilyTaxonomy,
    build_family_episode_intervals,
    build_family_payoff_support,
)


def _taxonomy() -> FamilyTaxonomy:
    return FamilyTaxonomy(
        mapping_id="test-taxonomy",
        destination_families=("family-1", "family-2"),
        pair_to_family={
            ("cycle_01", "state_1"): "family-1",
            ("cycle_02", "state_1"): "family-1",
            ("cycle_03", "state_2"): "family-2",
        },
        unknown_family="unknown",
    )


def _calendar() -> pd.DataFrame:
    sessions = ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    return pd.DataFrame(
        {
            "period": 2025,
            "score_session": sessions,
            "forecast_freeze_timestamp": pd.to_datetime(
                [f"{session}T14:30:00Z" for session in sessions], utc=True
            ),
        }
    )


def test_adjacent_pair_episodes_union_into_one_family_episode() -> None:
    diagnostics = pd.DataFrame(
        {
            "episode_id": ["pair-a", "pair-b"],
            "period": [2025, 2025],
            "loop_id": ["cycle_01", "cycle_02"],
            "orientation": ["state_1", "state_1"],
            "hindsight_estimated_onset": ["2025-01-03", "2025-01-07"],
            "hindsight_estimated_end": ["2025-01-06", "2025-01-08"],
            "total_episode_payoff_bps": [10.0, 20.0],
        }
    )
    support = pd.DataFrame(
        {
            "period": [2025, 2025],
            "session": ["2025-01-06", "2025-01-08"],
            "destination_family": ["family-1", "family-1"],
            "data_availability_timestamp": pd.to_datetime(
                ["2025-01-06T20:00Z", "2025-01-08T20:00Z"], utc=True
            ),
            "robust_net_payoff_bps": [10.0, 20.0],
            "independent_stock_count": [2, 3],
            "effective_sample_size": [2.0, 3.0],
        }
    )

    result = build_family_episode_intervals(diagnostics, _taxonomy(), _calendar(), support)

    assert len(result) == 1
    assert result.loc[0, "episode_onset_session"] == "2025-01-03"
    assert result.loc[0, "episode_end_session"] == "2025-01-08"
    assert result.loc[0, "source_pair_episode_ids"] == "pair-a|pair-b"
    assert result.loc[0, "total_episode_payoff_bps"] == 30.0
    assert result.loc[0, "label_availability_timestamp"] > pd.Timestamp("2025-01-08T14:30Z")


def test_family_payoff_support_never_turns_absence_into_zero() -> None:
    panel = pd.DataFrame(
        {
            "period": [2025, 2025],
            "session": ["2025-01-03", "2025-01-03"],
            "loop_id": ["cycle_01", "cycle_02"],
            "orientation": ["state_1", "state_1"],
            "data_availability_timestamp": pd.to_datetime(
                ["2025-01-03T19:00Z", "2025-01-03T20:00Z"], utc=True
            ),
            "robust_net_payoff_bps": [10.0, 30.0],
            "independent_stock_count": [1, 2],
            "effective_sample_size": [1.0, 2.0],
        }
    )

    result = build_family_payoff_support(panel, _taxonomy())

    assert len(result) == 1
    assert result.loc[0, "robust_net_payoff_bps"] == 20.0
    assert result.loc[0, "independent_stock_count"] == 3
    assert "2025-01-06" not in set(result["session"])


def test_episode_mapping_fails_closed_for_unmapped_pairs() -> None:
    diagnostics = pd.DataFrame(
        {
            "episode_id": ["unmapped"],
            "period": [2025],
            "loop_id": ["cycle_99"],
            "orientation": ["state_9"],
            "hindsight_estimated_onset": ["2025-01-03"],
            "hindsight_estimated_end": ["2025-01-06"],
            "total_episode_payoff_bps": [100.0],
        }
    )

    result = build_family_episode_intervals(
        diagnostics,
        _taxonomy(),
        _calendar(),
        pd.DataFrame(
            columns=[
                "period",
                "session",
                "destination_family",
                "data_availability_timestamp",
            ]
        ),
    )

    assert result.empty
