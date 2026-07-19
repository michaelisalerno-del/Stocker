from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.observable_event_ranking_v1.events import (
    build_relative_context,
    calibrate_q90_threshold,
    causal_robust_scale,
    deduplicate_and_assign_events,
    trigger_e1_events,
)
from stocker_research.observable_event_ranking_v1.provenance import (
    ProvenanceViolation,
    assert_outcome_free_event_ledger,
    audit_primary_imports,
)
from stocker_research.observable_event_ranking_v1.sector_context import (
    validate_sector_membership_ledger,
)
from stocker_research.observable_event_ranking_v1.universe import (
    UniverseRules,
    build_monthly_universe_ledger,
)


def _bar_panel(*, symbols: int = 20, broken_symbol: str | None = None) -> pd.DataFrame:
    session = pd.Timestamp("2025-07-07", tz="UTC")
    rows: list[dict[str, object]] = []
    for symbol_number in range(symbols):
        symbol = f"S{symbol_number:02d}"
        closes = [100.0 + symbol_number + offset for offset in (0, 1, 2, 3, 5, 8, 12)]
        for bar_number, close in enumerate(closes):
            bar_end = session + pd.Timedelta(hours=13, minutes=35 + 5 * bar_number)
            rows.append(
                {
                    "source_provider": "EODHD",
                    "source_dataset_id": f"fixture:{symbol}",
                    "source_hash": f"hash-{symbol}",
                    "symbol": symbol,
                    "sector": "Technology" if symbol_number < 10 else "Industrials",
                    "session": session,
                    "bar_start": bar_end - pd.Timedelta(minutes=5),
                    "bar_end": bar_end,
                    "feature_availability_time": bar_end + pd.Timedelta(seconds=1),
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.4,
                    "close": close,
                    "volume": 10_000.0,
                    "timezone": "UTC",
                    "adjustment_status": "raw_unadjusted",
                    "corporate_action_status": "clear",
                    "gap_status": "source_gap"
                    if symbol == broken_symbol and bar_number == 4
                    else "complete",
                    "volume_is_provider_activity_proxy": True,
                    "fully_completed": not (symbol == broken_symbol and bar_number == 6),
                    "universe_eligible": True,
                    "session_close": session + pd.Timedelta(hours=20),
                }
            )
    return pd.DataFrame(rows)


def test_relative_context_uses_exact_non_overlapping_windows_and_leave_one_out_peers() -> None:
    panel = _bar_panel()

    context = build_relative_context(panel)
    row = context.loc[context["symbol"] == "S00"].iloc[-1]

    assert row["recent_15m_return"] == pytest.approx(112.0 / 103.0 - 1.0)
    assert row["preceding_15m_return"] == pytest.approx(103.0 / 100.0 - 1.0)
    assert row["market_peer_count"] == 19
    assert row["sector_peer_count"] == 9
    assert bool(row["context_valid"])


def test_source_gap_or_incomplete_bar_invalidates_crossing_event_window() -> None:
    panel = _bar_panel(broken_symbol="S00")

    context = build_relative_context(panel)
    row = context.loc[context["symbol"] == "S00"].iloc[-1]

    assert not bool(row["context_valid"])
    assert "incomplete_bar" in str(row["context_unavailable_reason"])
    assert "source_gap_crossing_window" in str(row["context_unavailable_reason"])


def test_mutating_bars_after_confirmation_cannot_change_earlier_event_context() -> None:
    panel = _bar_panel(symbols=20)
    original = build_relative_context(panel)
    cutoff = pd.Timestamp("2025-07-07T14:05:00Z")
    appended = panel.groupby("symbol", sort=True).tail(1).copy()
    appended["bar_start"] = pd.Timestamp("2025-07-07T14:30:00Z")
    appended["bar_end"] = pd.Timestamp("2025-07-07T14:35:00Z")
    appended["feature_availability_time"] = pd.Timestamp("2025-07-07T14:35:01Z")
    appended["close"] = 1_000_000.0
    mutated = build_relative_context(pd.concat([panel, appended], ignore_index=True))
    columns = [
        "symbol",
        "bar_end",
        "recent_market_relative",
        "preceding_market_relative",
        "recent_sector_relative",
        "preceding_sector_relative",
        "market_relative_acceleration",
        "sector_relative_acceleration",
        "context_valid",
    ]
    expected = original.loc[original["bar_end"].le(cutoff), columns].reset_index(drop=True)
    actual = mutated.loc[mutated["bar_end"].le(cutoff), columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_minimum_market_and_sector_peer_counts_fail_closed() -> None:
    context = build_relative_context(_bar_panel(symbols=19))
    row = context.loc[context["symbol"] == "S00"].iloc[-1]

    assert not bool(row["context_valid"])
    assert "insufficient_market_peers" in str(row["context_unavailable_reason"])


def test_trailing_robust_scale_excludes_current_and_later_sessions() -> None:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-02", tz="UTC")
    for session_number in range(22):
        for stock_number in range(2):
            rows.append(
                {
                    "symbol": f"S{stock_number}",
                    "session": start + pd.Timedelta(days=session_number),
                    "bar_end": start + pd.Timedelta(days=session_number, hours=15),
                    "market_relative_acceleration": session_number + stock_number / 10,
                    "sector_relative_acceleration": session_number + stock_number / 5,
                    "context_valid": True,
                }
            )
    frame = pd.DataFrame(rows)
    scaled = causal_robust_scale(frame, min_observations=20)
    target_index = scaled.index[scaled["session"] == start + pd.Timedelta(days=20)][0]
    original = scaled.loc[
        target_index,
        ["market_relative_acceleration_z", "sector_relative_acceleration_z"],
    ].to_numpy(dtype=float)

    mutated = frame.copy()
    mutation_mask = (mutated["session"] > start + pd.Timedelta(days=20)) | (
        (mutated["session"] == start + pd.Timedelta(days=20)) & (mutated.index != target_index)
    )
    mutated.loc[
        mutation_mask,
        ["market_relative_acceleration", "sector_relative_acceleration"],
    ] = 1_000_000.0
    rescored = causal_robust_scale(mutated, min_observations=20)

    assert rescored.loc[
        target_index,
        ["market_relative_acceleration_z", "sector_relative_acceleration_z"],
    ].to_numpy(dtype=float) == pytest.approx(original)


def test_q90_calibration_is_outcome_free_and_threshold_is_inclusive() -> None:
    months = pd.date_range("2025-01-01", periods=6, freq="MS", tz="UTC")
    rows: list[dict[str, object]] = []
    for month_number, month in enumerate(months):
        for observation in range(10):
            strength = float(month_number * 10 + observation)
            rows.append(
                {
                    "session": month + pd.Timedelta(days=observation),
                    "bar_end": month + pd.Timedelta(days=observation, hours=15),
                    "symbol": f"S{observation:02d}",
                    "event_strength": strength,
                    "recent_market_relative": 0.01,
                    "recent_sector_relative": 0.01,
                    "context_valid": True,
                }
            )
    frame = pd.DataFrame(rows)
    calibration = calibrate_q90_threshold(frame)
    equality = frame.iloc[[0]].copy()
    equality["event_strength"] = calibration.threshold

    triggered = trigger_e1_events(equality, calibration.threshold)

    assert calibration.threshold == pytest.approx(np.quantile(np.arange(60), 0.90))
    assert len(calibration.rows) == 60
    assert len(triggered) == 1
    assert not any("future" in column for column in calibration.rows.columns)


def test_first_event_owns_stock_session_and_later_triggers_are_diagnostic_only() -> None:
    session = pd.Timestamp("2025-07-07", tz="UTC")
    triggers = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session": session,
                "bar_end": session + pd.Timedelta(hours=14, minutes=1),
                "feature_availability_time": session + pd.Timedelta(hours=14, minutes=1, seconds=1),
                "event_strength": 2.0,
                "session_close": session + pd.Timedelta(hours=20),
            },
            {
                "symbol": "AAA",
                "session": session,
                "bar_end": session + pd.Timedelta(hours=14, minutes=8),
                "feature_availability_time": session + pd.Timedelta(hours=14, minutes=8, seconds=1),
                "event_strength": 3.0,
                "session_close": session + pd.Timedelta(hours=20),
            },
        ]
    )

    result = deduplicate_and_assign_events(triggers)

    assert len(result.primary_events) == 1
    assert result.raw_trigger_count == 2
    assert result.first_event_count == 1
    assert result.later_trigger_count == 1
    assert result.primary_events.iloc[0]["assigned_decision_time"] == pd.Timestamp(
        "2025-07-07T14:30:00Z"
    )
    rerun = deduplicate_and_assign_events(triggers)
    assert result.primary_events.iloc[0]["event_id"] == rerun.primary_events.iloc[0]["event_id"]
    assert result.primary_events.iloc[0]["slate_id"] == rerun.primary_events.iloc[0]["slate_id"]


def test_first_event_ownership_uses_causal_availability_not_bar_end_alone() -> None:
    session = pd.Timestamp("2025-07-07", tz="UTC")
    triggers = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session": session,
                "bar_end": pd.Timestamp("2025-07-07T14:01:00Z"),
                "feature_availability_time": pd.Timestamp("2025-07-07T14:20:00Z"),
                "event_strength": 2.0,
                "session_close": pd.Timestamp("2025-07-07T20:00:00Z"),
            },
            {
                "symbol": "AAA",
                "session": session,
                "bar_end": pd.Timestamp("2025-07-07T14:08:00Z"),
                "feature_availability_time": pd.Timestamp("2025-07-07T14:09:00Z"),
                "event_strength": 3.0,
                "session_close": pd.Timestamp("2025-07-07T20:00:00Z"),
            },
        ]
    )

    result = deduplicate_and_assign_events(triggers)

    assert result.primary_events.iloc[0]["event_strength"] == 3.0
    assert set(result.diagnostics["deduplication_status"]) == {
        "first_event",
        "later_trigger_diagnostic",
    }
    assert "grid_assignment_rejection_reason" in result.diagnostics


def test_early_close_rejects_event_when_full_primary_interval_does_not_fit() -> None:
    session = pd.Timestamp("2025-11-28", tz="UTC")
    triggers = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session": session,
                "bar_end": pd.Timestamp("2025-11-28T17:01:00Z"),
                "feature_availability_time": pd.Timestamp("2025-11-28T17:01:01Z"),
                "event_strength": 2.0,
                "session_close": pd.Timestamp("2025-11-28T18:00:00Z"),
            }
        ]
    )

    result = deduplicate_and_assign_events(triggers)

    assert result.primary_events.empty
    assert result.rejected_count == 1
    assert result.diagnostics.iloc[0]["grid_assignment_rejection_reason"] == (
        "primary_outcome_interval_exceeds_session"
    )


def test_sector_ledger_rejects_current_snapshot_without_effective_dates() -> None:
    current_snapshot = pd.DataFrame(
        [{"symbol": "AAA", "sector": "Technology", "source": "current_screener"}]
    )

    issues = validate_sector_membership_ledger(current_snapshot)

    assert "missing_effective_from" in issues
    assert "missing_known_at" in issues
    assert "missing_stable_source_id" in issues


def test_monthly_universe_eligibility_cannot_change_from_later_sessions() -> None:
    sessions = pd.bdate_range("2025-01-02", periods=65, tz="UTC")
    stats = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session": session,
                "close": 10.0,
                "daily_dollar_activity": 25_000_000.0,
                "bar_coverage": 1.0,
                "valid_session": True,
                "unresolved_problem": False,
                "source_provider": "EODHD",
                "source_dataset_id": "raw/eodhd/AAA.US/5m",
                "source_hash": "bars-hash",
            }
            for session in sessions
        ]
    )
    known_at = pd.Timestamp("2024-12-01", tz="UTC")
    security_master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "effective_from": pd.Timestamp("2020-01-01", tz="UTC"),
                "effective_to": pd.NaT,
                "known_at": known_at,
                "security_type": "common_stock",
                "currency": "USD",
                "country": "US",
                "stable_source_id": "eodhd:AAA.US",
                "source_hash": "security-hash",
            }
        ]
    )
    sectors = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "Technology",
                "effective_from": pd.Timestamp("2020-01-01", tz="UTC"),
                "effective_to": pd.NaT,
                "known_at": known_at,
                "stable_source_id": "sector:AAA:2020",
                "source_provider": "trusted-pit-sector-source",
                "source_dataset_id": "sector-history-v1",
                "source_hash": "sector-hash",
            }
        ]
    )
    month = pd.Timestamp("2025-04-01", tz="UTC")
    original = build_monthly_universe_ledger(
        month_starts=[month],
        daily_stats=stats,
        security_master=security_master,
        sector_membership=sectors,
        rules=UniverseRules(min_same_sector_total=1),
    )
    assert original.iloc[0]["source_provider"] == "EODHD"
    assert original.iloc[0]["source_dataset_id"] == "raw/eodhd/AAA.US/5m"
    assert original.iloc[0]["source_hash"] == "bars-hash"
    invalid_previous = stats.copy()
    invalid_previous.loc[
        invalid_previous["session"].eq(pd.Timestamp("2025-03-31", tz="UTC")),
        "valid_session",
    ] = False
    invalid_previous_result = build_monthly_universe_ledger(
        month_starts=[month],
        daily_stats=invalid_previous,
        security_master=security_master,
        sector_membership=sectors,
        rules=UniverseRules(min_same_sector_total=1),
    )
    assert (
        "missing_or_invalid_previous_official_session"
        in invalid_previous_result.iloc[0]["qualification_reasons"]
    )
    unresolved = stats.copy()
    unresolved.loc[unresolved.index[0], "unresolved_problem"] = True
    unresolved_result = build_monthly_universe_ledger(
        month_starts=[month],
        daily_stats=unresolved,
        security_master=security_master,
        sector_membership=sectors,
        rules=UniverseRules(min_same_sector_total=1),
    )
    assert (
        "unresolved_data_or_corporate_action_problem"
        in unresolved_result.iloc[0]["qualification_reasons"]
    )
    future = stats.copy()
    future.loc[len(future)] = {
        "symbol": "AAA",
        "session": pd.Timestamp("2025-05-01", tz="UTC"),
        "close": 0.01,
        "daily_dollar_activity": 0.0,
        "bar_coverage": 0.0,
        "valid_session": False,
        "unresolved_problem": True,
        "source_provider": "EODHD",
        "source_dataset_id": "raw/eodhd/AAA.US/5m",
        "source_hash": "future-bars-hash",
    }
    mutated = build_monthly_universe_ledger(
        month_starts=[month],
        daily_stats=future,
        security_master=security_master,
        sector_membership=sectors,
        rules=UniverseRules(min_same_sector_total=1),
    )

    pd.testing.assert_frame_equal(original, mutated)


def test_event_ledger_rejects_future_and_old_line_columns() -> None:
    with pytest.raises(ProvenanceViolation):
        assert_outcome_free_event_ledger(
            pd.DataFrame([{"event_id": "x", "future_return_60m": 0.1}])
        )
    with pytest.raises(ProvenanceViolation):
        assert_outcome_free_event_ledger(pd.DataFrame([{"event_id": "x", "regime_id": 2}]))


def test_static_provenance_audit_detects_a_retired_primary_import(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text("import pandas\n", encoding="utf-8")
    forbidden = tmp_path / "forbidden.py"
    forbidden.write_text("from stocker_research.slrno.regime import State\n", encoding="utf-8")

    assert audit_primary_imports([clean]) == []
    assert audit_primary_imports([clean, forbidden]) == [
        "forbidden.py:stocker_research.slrno.regime"
    ]
