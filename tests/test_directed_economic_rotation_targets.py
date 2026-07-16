from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocker_research.directed_economic_rotation import (
    ActivationRegistration,
    FamilyTaxonomy,
    aggregate_family_states,
    build_activation_targets,
    derive_source_events,
)

ROOT = Path(__file__).resolve().parents[1]
MAPPING = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/contracts"
    / "20260716-directed-economic-loop-regime-rotation-v1-family-mapping.json"
)


def _pair_state(
    session: str,
    loop_id: str,
    orientation: str,
    state: str,
    *,
    p_active: float,
    mean_bps: float,
) -> dict[str, object]:
    freeze = pd.Timestamp(f"{session} 14:30", tz="UTC")
    return {
        "period": int(session[:4]),
        "score_session": session,
        "decision_timestamp": freeze,
        "prediction_frozen_at": freeze,
        "feature_max_availability_timestamp": freeze - pd.Timedelta(days=1),
        "training_latest_availability_timestamp": freeze - pd.Timedelta(hours=1),
        "loop_id": loop_id,
        "orientation": orientation,
        "horizon": 24,
        "model_name": "hierarchical_payoff_history_change_point",
        "edge_state": state,
        "p_edge_positive": p_active,
        "p_edge_active": p_active,
        "p_change_now": 0.1,
        "p_on_next": 1.0 - p_active,
        "p_off_next": 0.2,
        "p_survive_horizon": 0.7,
        "posterior_mean_net_bps": mean_bps,
        "posterior_lower_bound_net_bps": mean_bps - 5.0,
        "posterior_std_net_bps": 10.0,
        "posterior_run_length_mean": 4.0,
        "effective_sessions": 12.0,
        "independent_stocks": 5,
        "effective_sample_size": 10.0,
        "reason_codes": "",
    }


def test_frozen_family_mapping_is_topology_only_and_total_for_scored_pairs() -> None:
    taxonomy = FamilyTaxonomy.from_json(MAPPING)
    raw = json.loads(MAPPING.read_text(encoding="utf-8"))

    assert taxonomy.destination_families == tuple(
        f"two_transition_return_cycle__state_{state}" for state in range(8)
    )
    assert not any(
        token in json.dumps(raw).lower()
        for token in ("net_payoff", "episode_result", "activation_result")
    )
    assert taxonomy.family_for("cycle_04", "state_4").endswith("state_4")
    assert taxonomy.family_for("cycle_13", "state_7").endswith("state_7")
    assert taxonomy.family_for("cycle_missing", "state_0") == "unknown_topology"


def test_family_state_aggregation_is_causal_and_fails_closed_on_unknown() -> None:
    taxonomy = FamilyTaxonomy.from_json(MAPPING)
    rows = [
        _pair_state("2025-01-03", "cycle_04", "state_4", "active", p_active=0.8, mean_bps=9),
        _pair_state("2025-01-03", "cycle_06", "state_4", "unknown", p_active=0.2, mean_bps=-2),
        _pair_state("2025-01-06", "cycle_04", "state_4", "retired", p_active=0.1, mean_bps=-4),
        _pair_state("2025-01-06", "cycle_06", "state_4", "unknown", p_active=0.2, mean_bps=-1),
    ]

    family = aggregate_family_states(pd.DataFrame(rows), taxonomy)

    assert (
        family.loc[family["score_session"].eq("2025-01-03"), "operational_state"].item() == "active"
    )
    assert (
        family.loc[family["score_session"].eq("2025-01-06"), "operational_state"].item()
        == "unknown"
    )
    assert family["feature_availability_timestamp"].le(family["forecast_freeze_timestamp"]).all()
    assert "hindsight_episode_state" not in family.columns


def test_newly_decaying_and_retired_events_are_timestamped_at_observation() -> None:
    sessions = ["2025-01-03", "2025-01-06", "2025-01-07"]
    states = ["active", "decaying", "retired"]
    frame = pd.DataFrame(
        {
            "period": 2025,
            "score_session": sessions,
            "destination_family": "two_transition_return_cycle__state_4",
            "operational_state": states,
            "forecast_freeze_timestamp": pd.to_datetime(
                [f"{session} 14:30Z" for session in sessions], utc=True
            ),
        }
    )

    events = derive_source_events(frame)

    assert events.loc[events["score_session"].eq("2025-01-06"), "newly_decaying"].item()
    assert events.loc[events["score_session"].eq("2025-01-07"), "newly_retired"].item()
    assert events["source_event_timestamp"].eq(events["forecast_freeze_timestamp"]).all()


def test_activation_windows_follow_explicit_sessions_and_preserve_multiple_events() -> None:
    registration = ActivationRegistration()
    families = [
        "two_transition_return_cycle__state_1",
        "two_transition_return_cycle__state_2",
    ]
    calendar = pd.DataFrame(
        {
            "period": 2025,
            "score_session": ["2025-01-03", "2025-01-06", "2025-01-08", "2025-01-09"],
        }
    )
    forecast_states = pd.DataFrame(
        [
            {
                "period": 2025,
                "score_session": session,
                "destination_family": family,
                "operational_state": "retired",
                "forecast_freeze_timestamp": pd.Timestamp(f"{session} 14:30Z"),
            }
            for session in calendar["score_session"]
            for family in families
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                "period": 2025,
                "destination_family": families[0],
                "episode_id": "episode-a",
                "episode_onset_session": "2025-01-06",
                "episode_end_session": "2025-01-08",
                "label_availability_timestamp": pd.Timestamp("2025-01-08 20:00Z"),
            },
            {
                "period": 2025,
                "destination_family": families[1],
                "episode_id": "episode-b",
                "episode_onset_session": "2025-01-08",
                "episode_end_session": "2025-01-09",
                "label_availability_timestamp": pd.Timestamp("2025-01-09 20:00Z"),
            },
        ]
    )
    support = pd.DataFrame(
        [
            {
                "period": 2025,
                "session": session,
                "destination_family": family,
                "data_availability_timestamp": pd.Timestamp(f"{session} 20:00Z"),
            }
            for session in calendar["score_session"]
            for family in families
        ]
    )

    targets = build_activation_targets(
        forecast_states=forecast_states,
        calendar=calendar,
        episode_intervals=intervals,
        payoff_support=support,
        registration=registration,
    )
    lead_one = targets.loc[
        targets["forecast_session"].eq("2025-01-03") & targets["target_window_sessions"].eq(1)
    ]
    lead_three = targets.loc[
        targets["forecast_session"].eq("2025-01-03") & targets["target_window_sessions"].eq(3)
    ]

    assert lead_one.loc[lead_one["destination_family"].eq(families[0]), "activation_target"].item()
    assert not lead_one.loc[
        lead_one["destination_family"].eq(families[1]), "activation_target"
    ].item()
    assert lead_three["observed_activation_count"].eq(2).all()
    assert lead_three["multiple_activation_flag"].all()
    assert lead_three["target_start_session"].eq("2025-01-06").all()
    assert lead_three["target_end_session"].eq("2025-01-09").all()


def test_missing_support_and_period_boundaries_are_not_no_activation() -> None:
    registration = ActivationRegistration()
    family = "two_transition_return_cycle__state_1"
    calendar = pd.DataFrame(
        {
            "period": [2023, 2023, 2025, 2025],
            "score_session": ["2023-12-28", "2023-12-29", "2025-01-02", "2025-01-03"],
        }
    )
    forecast_states = pd.DataFrame(
        {
            "period": [2023, 2025],
            "score_session": ["2023-12-29", "2025-01-02"],
            "destination_family": family,
            "operational_state": "retired",
            "forecast_freeze_timestamp": pd.to_datetime(
                ["2023-12-29 14:30Z", "2025-01-02 14:30Z"], utc=True
            ),
        }
    )
    targets = build_activation_targets(
        forecast_states=forecast_states,
        calendar=calendar,
        episode_intervals=pd.DataFrame(
            columns=[
                "period",
                "destination_family",
                "episode_id",
                "episode_onset_session",
                "episode_end_session",
                "label_availability_timestamp",
            ]
        ),
        payoff_support=pd.DataFrame(
            columns=[
                "period",
                "session",
                "destination_family",
                "data_availability_timestamp",
            ]
        ),
        registration=registration,
    )

    boundary = targets.loc[
        targets["forecast_session"].eq("2023-12-29") & targets["target_window_sessions"].eq(1)
    ].iloc[0]
    missing = targets.loc[
        targets["forecast_session"].eq("2025-01-02") & targets["target_window_sessions"].eq(1)
    ].iloc[0]
    assert boundary.target_status == "period_boundary"
    assert pd.isna(boundary.activation_target)
    assert missing.target_status == "insufficient_future_support"
    assert pd.isna(missing.activation_target)


def test_current_active_family_is_not_relabeled_as_a_new_activation() -> None:
    family = "two_transition_return_cycle__state_1"
    targets = build_activation_targets(
        forecast_states=pd.DataFrame(
            {
                "period": [2025],
                "score_session": ["2025-01-03"],
                "destination_family": [family],
                "operational_state": ["active"],
                "forecast_freeze_timestamp": pd.to_datetime(["2025-01-03 14:30Z"], utc=True),
            }
        ),
        calendar=pd.DataFrame(
            {"period": [2025, 2025], "score_session": ["2025-01-03", "2025-01-06"]}
        ),
        episode_intervals=pd.DataFrame(
            {
                "period": [2025],
                "destination_family": [family],
                "episode_id": ["episode-a"],
                "episode_onset_session": ["2025-01-06"],
                "episode_end_session": ["2025-01-06"],
                "label_availability_timestamp": pd.to_datetime(["2025-01-06 20:00Z"], utc=True),
            }
        ),
        payoff_support=pd.DataFrame(
            {
                "period": [2025],
                "session": ["2025-01-06"],
                "destination_family": [family],
                "data_availability_timestamp": pd.to_datetime(["2025-01-06 20:00Z"], utc=True),
            }
        ),
        registration=ActivationRegistration(),
    )

    row = targets.loc[targets["target_window_sessions"].eq(1)].iloc[0]
    assert row.target_status == "current_active_not_candidate"
    assert pd.isna(row.activation_target)
