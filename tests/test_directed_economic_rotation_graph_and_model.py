from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stocker_research.directed_economic_rotation import (
    GraphSettings,
    MaturedRotationExample,
    PastOnlyRotationGraph,
    PrequentialSettings,
    permute_source_events,
    run_prequential_rotation,
    shift_source_events,
)

FAMILY_A = "two_transition_return_cycle__state_1"
FAMILY_B = "two_transition_return_cycle__state_2"


def _example(*, available: str, target: bool, source: str = FAMILY_A) -> MaturedRotationExample:
    return MaturedRotationExample(
        example_id=f"example-{available}-{target}-{source}",
        period=2025,
        forecast_session="2025-01-03",
        destination_family=FAMILY_B,
        activation_target=target,
        label_availability_timestamp=pd.Timestamp(available),
        source_events={source: frozenset({"newly_decaying"})},
    )


def test_graph_uses_only_matured_past_labels_and_smooths_unseen_edges() -> None:
    graph = PastOnlyRotationGraph(GraphSettings())
    past = _example(available="2025-01-05T20:00Z", target=True)
    future = _example(available="2025-01-10T20:00Z", target=False, source="family-c")

    updated = graph.update_matured([past, future], as_of=pd.Timestamp("2025-01-06T14:30Z"))
    unseen = graph.edge_summary("family-unseen", "newly_retired", FAMILY_B)
    observed = graph.edge_summary(FAMILY_A, "newly_decaying", FAMILY_B)

    assert updated == 1
    assert observed.support == 1
    assert observed.raw_transition_probability > 0.0
    assert unseen.raw_transition_probability > 0.0
    assert unseen.support_status == "unknown"
    assert graph.update_matured([past], as_of=pd.Timestamp("2025-01-07T14:30Z")) == 0


def test_same_family_persistence_is_not_counted_as_cross_family_rotation() -> None:
    graph = PastOnlyRotationGraph(GraphSettings(minimum_source_event_sessions=1))
    graph.update_matured(
        [_example(available="2025-01-05T20:00Z", target=True, source=FAMILY_B)],
        as_of=pd.Timestamp("2025-01-06T14:30Z"),
    )

    edge = graph.edge_summary(FAMILY_B, "newly_decaying", FAMILY_B)

    assert edge.support == 0
    assert edge.support_status == "same_family_excluded"


def _synthetic_rotation(
    *, causal_rotation: bool, sessions: int = 120
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=sessions)
    state_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        session = date.strftime("%Y-%m-%d")
        freeze = pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=14, minutes=30)
        source_fires = index % 5 == 0
        target = source_fires if causal_rotation else index % 7 == 0
        for family in (FAMILY_A, FAMILY_B):
            state_rows.append(
                {
                    "period": 2025,
                    "score_session": session,
                    "destination_family": family,
                    "horizon": 24,
                    "forecast_freeze_timestamp": freeze,
                    "feature_availability_timestamp": freeze - pd.Timedelta(days=1),
                    "training_cutoff": freeze - pd.Timedelta(days=1),
                    "operational_state": "decaying"
                    if family == FAMILY_A and source_fires
                    else "retired",
                    "max_p_edge_active": 0.1,
                    "mean_p_edge_active": 0.1,
                    "max_p_on_next": 0.1,
                    "mean_p_off_next": 0.2,
                    "posterior_mean_net_bps": -2.0,
                    "posterior_std_net_bps": 20.0,
                    "active_probability_change": 0.0,
                    "posterior_mean_change_bps": 0.0,
                    "state_age_sessions": 3,
                    "sessions_since_active": 8,
                    "effective_sessions": 12.0,
                    "independent_stocks": 5,
                    "effective_sample_size": 15.0,
                    "reason_codes": "",
                }
            )
            event_rows.append(
                {
                    "period": 2025,
                    "score_session": session,
                    "destination_family": family,
                    "forecast_freeze_timestamp": freeze,
                    "source_active": False,
                    "newly_active": False,
                    "newly_decaying": bool(family == FAMILY_A and source_fires),
                    "newly_retired": False,
                    "source_state_transition": bool(family == FAMILY_A and source_fires),
                }
            )
            activation = bool(family == FAMILY_B and target)
            target_rows.append(
                {
                    "period": 2025,
                    "forecast_session": session,
                    "destination_family": family,
                    "target_window_sessions": 3,
                    "target_status": "activation_observed" if activation else "no_activation",
                    "target_available": True,
                    "activation_target": activation,
                    "label_availability_timestamp": freeze + pd.Timedelta(days=4),
                    "first_activation_session": session if activation else pd.NA,
                    "target_episode_ids": f"episode-{index}" if activation else "",
                    "observed_activation_count": int(activation),
                    "multiple_activation_flag": False,
                    "no_activation_flag": not activation,
                }
            )
    return pd.DataFrame(state_rows), pd.DataFrame(event_rows), pd.DataFrame(target_rows)


def test_synthetic_cross_family_rotation_improves_m3_but_wrong_lag_destroys_it() -> None:
    states, events, targets = _synthetic_rotation(causal_rotation=True)
    settings = PrequentialSettings(
        run_id="synthetic",
        minimum_training_rows=10,
        minimum_training_activations=2,
        graph=GraphSettings(minimum_source_event_sessions=2, pooling_strength=5.0),
    )

    real = run_prequential_rotation(states, events, targets, settings=settings)
    shifted = run_prequential_rotation(
        states,
        shift_source_events(events, sessions=2),
        targets,
        settings=settings,
    )
    eligible = real.loc[
        real["destination_family"].eq(FAMILY_B)
        & real["target_available"]
        & real["training_rows"].ge(20)
    ]
    shifted_eligible = shifted.loc[
        shifted["destination_family"].eq(FAMILY_B)
        & shifted["target_available"]
        & shifted["training_rows"].ge(20)
    ]
    real_pivot = eligible.pivot(
        index="forecast_session", columns="model_name", values="predicted_activation_probability"
    )
    shifted_pivot = shifted_eligible.pivot(
        index="forecast_session", columns="model_name", values="predicted_activation_probability"
    )
    y = (
        eligible.loc[eligible["model_name"].eq("M1_destination_own_history"), "activation_target"]
        .astype(float)
        .to_numpy()
    )
    brier_m1 = float(np.mean((real_pivot["M1_destination_own_history"].to_numpy() - y) ** 2))
    brier_m3 = float(np.mean((real_pivot["M3_directed_family_rotation"].to_numpy() - y) ** 2))
    shifted_y = (
        shifted_eligible.loc[
            shifted_eligible["model_name"].eq("M1_destination_own_history"), "activation_target"
        ]
        .astype(float)
        .to_numpy()
    )
    shifted_m1 = float(
        np.mean((shifted_pivot["M1_destination_own_history"].to_numpy() - shifted_y) ** 2)
    )
    shifted_m3 = float(
        np.mean((shifted_pivot["M3_directed_family_rotation"].to_numpy() - shifted_y) ** 2)
    )

    assert brier_m3 < brier_m1
    assert (brier_m1 - brier_m3) > (shifted_m1 - shifted_m3)


def test_synthetic_null_does_not_manufacture_directed_improvement() -> None:
    states, events, targets = _synthetic_rotation(causal_rotation=False)
    settings = PrequentialSettings(
        run_id="synthetic-null",
        minimum_training_rows=10,
        minimum_training_activations=2,
        graph=GraphSettings(minimum_source_event_sessions=2, pooling_strength=5.0),
    )
    forecasts = run_prequential_rotation(states, events, targets, settings=settings)
    eligible = forecasts.loc[
        forecasts["destination_family"].eq(FAMILY_B)
        & forecasts["target_available"]
        & forecasts["training_rows"].ge(30)
    ]
    pivot = eligible.pivot(
        index="forecast_session", columns="model_name", values="predicted_activation_probability"
    )
    y = (
        eligible.loc[eligible["model_name"].eq("M1_destination_own_history"), "activation_target"]
        .astype(float)
        .to_numpy()
    )
    m1 = float(np.mean((pivot["M1_destination_own_history"].to_numpy() - y) ** 2))
    m3 = float(np.mean((pivot["M3_directed_family_rotation"].to_numpy() - y) ** 2))

    assert m3 >= m1 - 0.002


def test_appending_future_rows_cannot_change_frozen_historical_forecasts() -> None:
    states, events, targets = _synthetic_rotation(causal_rotation=True, sessions=80)
    settings = PrequentialSettings(
        run_id="append-invariant",
        minimum_training_rows=5,
        minimum_training_activations=1,
        graph=GraphSettings(minimum_source_event_sessions=1, pooling_strength=5.0),
    )
    cutoff = sorted(states["score_session"].unique())[49]
    short_states = states.loc[states["score_session"].le(cutoff)]
    short_events = events.loc[events["score_session"].le(cutoff)]
    short_targets = targets.loc[targets["forecast_session"].le(cutoff)]

    short = run_prequential_rotation(short_states, short_events, short_targets, settings=settings)
    full = run_prequential_rotation(states, events, targets, settings=settings)
    columns = [
        "forecast_id",
        "predicted_activation_probability",
        "activation_base_rate",
        "training_rows",
        "training_activations",
        "frozen_feature_values_json",
    ]

    pd.testing.assert_frame_equal(
        short.loc[:, columns].reset_index(drop=True),
        full.loc[full["forecast_session"].le(cutoff), columns].reset_index(drop=True),
    )


def test_m1_m2_m3_populations_are_identical_and_m3_adds_only_registered_features() -> None:
    states, events, targets = _synthetic_rotation(causal_rotation=True, sessions=30)
    forecasts = run_prequential_rotation(
        states,
        events,
        targets,
        settings=PrequentialSettings(run_id="paired-population"),
    )
    compared = forecasts.loc[forecasts["model_name"].str.startswith(("M1", "M2", "M3"))]
    keys = ["period", "forecast_session", "destination_family", "target_window_sessions"]
    counts = compared.groupby(keys, sort=True)["model_name"].nunique()

    assert counts.eq(3).all()
    schemas = {
        row.model_name: set(json.loads(row.feature_schema_json))
        for row in compared.drop_duplicates("model_name").itertuples()
    }
    assert schemas["M1_destination_own_history"] < schemas["M2_undirected_system_state"]
    assert schemas["M2_undirected_system_state"] < schemas["M3_directed_family_rotation"]


def test_source_permutation_is_deterministic_and_target_free() -> None:
    _, events, _ = _synthetic_rotation(causal_rotation=True, sessions=20)
    first = permute_source_events(events, seed=20260716)
    second = permute_source_events(events, seed=20260716)

    pd.testing.assert_frame_equal(first, second)
    assert not any("target" in column or "payoff" in column for column in first.columns)
