from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.fixed_one_bar_entry_latency.metrics import (
    build_exact_paired_population,
    paired_summary,
    session_block_bootstrap,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "opportunity_id": ["one", "two", "three"],
            "anchor_id": ["a1", "a2", "a3"],
            "event_lineage_id": ["e1", "e2", "e3"],
            "period": [2025, 2025, 2025],
            "session_date": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "symbol": ["AAA", "BBB", "CCC"],
            "loop_id": ["cycle_04", "cycle_04", "cycle_07"],
            "orientation": ["state_4", "state_4", "state_5"],
            "direction": [1, -1, 1],
            "original_entry_timestamp": pd.to_datetime(
                ["2025-01-02 15:00Z", "2025-01-03 15:00Z", "2025-01-06 15:00Z"]
            ),
            "original_terminal_timestamp": pd.to_datetime(
                ["2025-01-02 17:00Z", "2025-01-03 17:00Z", "2025-01-06 17:00Z"]
            ),
            "original_gross_payoff_bps": [30.0, -10.0, 100.0],
            "original_total_cost_bps": [10.0, 10.0, 10.0],
            "original_net_payoff_bps": [20.0, -20.0, 90.0],
        }
    )


def _latency() -> pd.DataFrame:
    frame = (
        _source()
        .loc[
            :,
            [
                "opportunity_id",
                "anchor_id",
                "event_lineage_id",
                "period",
                "session_date",
                "symbol",
                "loop_id",
                "orientation",
                "direction",
                "original_terminal_timestamp",
            ],
        ]
        .copy()
    )
    frame["t1_status"] = ["available", "missing_exact_t1_open", "available"]
    frame["t1_entry_timestamp"] = pd.to_datetime(["2025-01-02 15:05Z", None, "2025-01-06 15:05Z"])
    frame["t1_gross_return_bps"] = [40.0, float("nan"), 70.0]
    frame["t1_total_cost_bps"] = [10.0, float("nan"), 10.0]
    frame["t1_net_return_bps"] = [30.0, float("nan"), 60.0]
    frame["paired_difference_bps"] = [10.0, float("nan"), -30.0]
    return frame


def test_exact_pairing_keeps_missing_t1_explicit_and_never_replaces_it() -> None:
    all_rows, paired, unavailable = build_exact_paired_population(_source(), _latency())

    assert len(all_rows) == 3
    assert set(paired["opportunity_id"]) == {"one", "three"}
    assert unavailable[["opportunity_id", "t1_status"]].to_dict("records") == [
        {"opportunity_id": "two", "t1_status": "missing_exact_t1_open"}
    ]
    assert all_rows["replacement_opportunity_id"].isna().all()
    assert not all_rows["overlap_or_capacity_refilled"].any()


def test_pairing_rejects_identity_drift_even_when_opportunity_id_matches() -> None:
    latency = _latency()
    latency.loc[0, "loop_id"] = "cycle_07"

    with pytest.raises(ValueError, match="identity mismatch"):
        build_exact_paired_population(_source(), latency)


def test_primary_t0_levels_are_recalculated_only_on_exact_paired_rows() -> None:
    _, paired, _ = build_exact_paired_population(_source(), _latency())

    summary = paired_summary(paired)

    assert summary["paired_opportunities"] == 2
    assert summary["t0_net_payoff_bps"] == 110.0
    assert summary["t1_net_payoff_bps"] == 90.0
    assert summary["paired_total_difference_bps"] == -20.0
    assert summary["paired_mean_difference_bps"] == -10.0
    assert summary["opportunities_improved_fraction"] == 0.5


def test_appending_future_unavailable_source_does_not_change_historical_pairs() -> None:
    _, paired, _ = build_exact_paired_population(_source(), _latency())
    source = pd.concat(
        [
            _source(),
            pd.DataFrame(
                [
                    {
                        **_source().iloc[-1].to_dict(),
                        "opportunity_id": "future",
                        "anchor_id": "future-anchor",
                        "event_lineage_id": "future-event",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    latency = pd.concat(
        [
            _latency(),
            pd.DataFrame(
                [
                    {
                        **_latency().iloc[-1].to_dict(),
                        "opportunity_id": "future",
                        "anchor_id": "future-anchor",
                        "event_lineage_id": "future-event",
                        "t1_status": "missing_provider_data",
                        "t1_net_return_bps": float("nan"),
                        "paired_difference_bps": float("nan"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    _, appended_pairs, _ = build_exact_paired_population(source, latency)

    assert appended_pairs["opportunity_id"].tolist() == paired["opportunity_id"].tolist()


def test_session_block_interval_is_deterministic_under_frozen_seed() -> None:
    _, paired, _ = build_exact_paired_population(_source(), _latency())

    first = session_block_bootstrap(paired, resamples=200, block_length=5, seed=20260716)
    second = session_block_bootstrap(paired, resamples=200, block_length=5, seed=20260716)

    assert first == second
    assert first["bootstrap_lower_95_bps"] <= first["observed_session_mean_delta_bps"]
    assert first["bootstrap_upper_95_bps"] >= first["observed_session_mean_delta_bps"]
