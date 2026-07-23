from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from stocker_research.eodhd_fixed_options_strategy_v0 import (
    DirectionMappingUnavailable,
    OptionSelectionError,
    apply_hidden_232_veto,
    assert_no_daily_option_high_low,
    build_matched_controls,
    choose_overall_decision,
    expiry_intrinsic_values,
    frozen_route_state_labels,
    option_position_pnl,
    quote_integrity_reason,
    reject_protected_dates,
    resolve_price_direction,
    select_atm_straddle,
    select_directional_debit_spreads,
    session_bootstrap_intervals,
    standard_contract_multiplier,
    validate_checkpoint_timing,
    validate_expiration_session,
    validate_preselection_date,
)


def _row(
    contract_id: str,
    option_type: str,
    strike: float,
    *,
    expiry: str = "2025-01-17",
    delta: float | None = None,
    bid: float = 1.0,
    ask: float = 1.2,
    open_interest: int = 100,
) -> dict[str, object]:
    return {
        "trade_date": date(2025, 1, 3),
        "contract_id": contract_id,
        "underlying_symbol": "AAL",
        "expiration_date": date.fromisoformat(expiry),
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "midpoint": (bid + ask) / 2.0,
        "open_interest": open_interest,
        "implied_volatility": 0.40,
        "delta": delta,
        "contract_multiplier": 100,
        "adjusted_contract": False,
        "deliverable_resolved": True,
    }


def test_ordinal_72_signal_is_available_at_1530_with_six_bars_left() -> None:
    bars_left = validate_checkpoint_timing(
        checkpoint=72,
        zero_based_bar_ordinal=71,
        feature_available_timestamp=pd.Timestamp("2025-01-02T20:30:00Z"),
        scheduled_close_timestamp=pd.Timestamp("2025-01-02T21:00:00Z"),
    )
    assert bars_left == 6


def test_frozen_route_state_labels_keep_precedence() -> None:
    frame = pd.DataFrame(
        [
            {
                "active_prefix_count": 8,
                "active_prefix_count_change_last_3_bars": 0,
                "top_prefix_depth_fraction": 0.40,
                "top_minus_second_prefix_depth": 0.05,
                "prefix_family_entropy": 0.90,
                "depth_margin_change_last_3_bars": 0.0,
            },
            {
                "active_prefix_count": 5,
                "active_prefix_count_change_last_3_bars": -1,
                "top_prefix_depth_fraction": 0.60,
                "top_minus_second_prefix_depth": 0.30,
                "prefix_family_entropy": 0.20,
                "depth_margin_change_last_3_bars": 0.1,
            },
            {
                "active_prefix_count": 4,
                "active_prefix_count_change_last_3_bars": 0,
                "top_prefix_depth_fraction": 0.90,
                "top_minus_second_prefix_depth": 0.50,
                "prefix_family_entropy": 0.20,
                "depth_margin_change_last_3_bars": 0.0,
            },
            {
                "active_prefix_count": 2,
                "active_prefix_count_change_last_3_bars": 0,
                "top_prefix_depth_fraction": 0.20,
                "top_minus_second_prefix_depth": 0.10,
                "prefix_family_entropy": 0.20,
                "depth_margin_change_last_3_bars": 0.0,
            },
        ]
    )
    thresholds = {
        "top_prefix_depth_fraction": (0.25, 0.50, 0.75),
        "top_minus_second_prefix_depth": (0.10, 0.25, 0.40),
        "prefix_family_entropy": (0.20, 0.50, 0.75),
    }
    assert frozen_route_state_labels(frame, thresholds).tolist() == [
        "BROAD_CONFLICT",
        "NARROWING",
        "DOMINANT_ROUTE",
        "LOW_ROUTE_SUPPORT",
    ]


def test_direction_mapping_rejects_unregistered_orientation() -> None:
    with pytest.raises(DirectionMappingUnavailable):
        resolve_price_direction("loop_p_0-1-0__o_0-1-0", {})


def test_contract_selection_uses_exact_previous_us_session() -> None:
    assert validate_preselection_date(date(2025, 1, 3), date(2025, 1, 6)) is None


def test_same_close_contract_selection_is_prohibited() -> None:
    with pytest.raises(OptionSelectionError, match="exact previous"):
        validate_preselection_date(date(2025, 1, 6), date(2025, 1, 6))


def test_atm_straddle_uses_frozen_common_strike_tie_break() -> None:
    chain = pd.DataFrame(
        [
            _row("AAL250117C00099000", "call", 99.0, open_interest=500),
            _row("AAL250117P00099000", "put", 99.0, open_interest=500),
            _row("AAL250117C00100000", "call", 100.0, open_interest=50),
            _row("AAL250117P00100000", "put", 100.0, open_interest=50),
        ]
    )
    selected = select_atm_straddle(
        chain,
        selection_date=date(2025, 1, 3),
        entry_date=date(2025, 1, 6),
        underlying_close=100.2,
        entry_dte_min=7,
        entry_dte_max=14,
    )
    assert selected.available
    assert selected.strike == 100.0
    assert selected.call_contract_id == "AAL250117C00100000"
    assert selected.put_contract_id == "AAL250117P00100000"


def test_call_debit_spread_selection_uses_frozen_rank() -> None:
    chain = pd.DataFrame(
        [
            _row("AAL250117C00100000", "call", 100.0, delta=0.50, open_interest=100),
            _row("AAL250117C00105000", "call", 105.0, delta=0.25, open_interest=80),
            _row("AAL250117P00100000", "put", 100.0, delta=-0.50, open_interest=100),
            _row("AAL250117P00095000", "put", 95.0, delta=-0.25, open_interest=80),
        ]
    )
    selected = select_directional_debit_spreads(
        chain,
        selection_date=date(2025, 1, 3),
        entry_date=date(2025, 1, 6),
    )
    assert selected.available
    assert selected.bullish_long_contract_id == "AAL250117C00100000"
    assert selected.bullish_short_contract_id == "AAL250117C00105000"


def test_put_debit_spread_selection_uses_lower_short_strike() -> None:
    chain = pd.DataFrame(
        [
            _row("AAL250117C00100000", "call", 100.0, delta=0.50),
            _row("AAL250117C00105000", "call", 105.0, delta=0.25),
            _row("AAL250117P00100000", "put", 100.0, delta=-0.50),
            _row("AAL250117P00095000", "put", 95.0, delta=-0.25),
        ]
    )
    selected = select_directional_debit_spreads(
        chain,
        selection_date=date(2025, 1, 3),
        entry_date=date(2025, 1, 6),
    )
    assert selected.bearish_long_contract_id == "AAL250117P00100000"
    assert selected.bearish_short_contract_id == "AAL250117P00095000"


def test_dte1_atm_selection_requires_expiry_on_next_calendar_day() -> None:
    chain = pd.DataFrame(
        [
            _row("AAL250107C00100000", "call", 100.0, expiry="2025-01-07"),
            _row("AAL250107P00100000", "put", 100.0, expiry="2025-01-07"),
        ]
    )
    selected = select_atm_straddle(
        chain,
        selection_date=date(2025, 1, 3),
        entry_date=date(2025, 1, 6),
        underlying_close=100.0,
        entry_dte_min=1,
        entry_dte_max=1,
    )
    assert selected.available
    assert selected.entry_dte == 1


def test_bid_ask_entry_exit_and_primary_commission() -> None:
    result = option_position_pnl(
        structure="long_straddle",
        entry_quotes={"call_ask": 3.0, "put_ask": 2.0},
        exit_quotes={"call_bid": 4.0, "put_bid": 1.0},
        multiplier=100,
        commission_per_contract_side=0.75,
    )
    assert result["entry_debit"] == 5.0
    assert result["exit_credit"] == 5.0
    assert result["commissions"] == 3.0
    assert result["net_pnl"] == -3.0
    assert result["total_initial_cash_debit"] == 501.5


def test_debit_spread_uses_long_bid_and_short_ask_at_exit() -> None:
    result = option_position_pnl(
        structure="debit_spread",
        entry_quotes={"long_ask": 2.0, "short_bid": 0.5},
        exit_quotes={"long_bid": 2.4, "short_ask": 0.7},
        multiplier=100,
        commission_per_contract_side=0.75,
    )
    assert result["entry_debit"] == 1.5
    assert result["exit_credit"] == pytest.approx(1.7)
    assert result["net_pnl"] == pytest.approx(17.0)


def test_expiry_intrinsic_values_are_secondary_diagnostic() -> None:
    assert expiry_intrinsic_values(underlying_close=103.0, strike=100.0) == {
        "call_intrinsic": 3.0,
        "put_intrinsic": 0.0,
    }


def test_expiration_session_accepts_early_close_and_rejects_ambiguous_settlement() -> None:
    assert (
        validate_expiration_session(
            expiration_date=date(2025, 7, 3),
            exit_session=date(2025, 7, 3),
            settlement_style="standard_equity_pm",
            scheduled_close_timestamp=pd.Timestamp("2025-07-03T17:00:00Z"),
            adjusted_contract=False,
            deliverable_resolved=True,
        )
        == "early_close"
    )
    with pytest.raises(OptionSelectionError, match="settlement"):
        validate_expiration_session(
            expiration_date=date(2025, 1, 17),
            exit_session=date(2025, 1, 17),
            settlement_style=None,
            scheduled_close_timestamp=pd.Timestamp("2025-01-17T21:00:00Z"),
            adjusted_contract=False,
            deliverable_resolved=True,
        )


def test_adjusted_or_nonstandard_contract_is_rejected() -> None:
    with pytest.raises(OptionSelectionError, match="adjusted"):
        standard_contract_multiplier(
            "AAL250117C00100000",
            underlying_symbol="AAL",
            strike=100.0,
            adjusted_contract=True,
            deliverable_resolved=False,
        )


def test_unknown_contract_safety_metadata_is_rejected() -> None:
    row = _row("AAL250117C00100000", "call", 100.0)
    row.pop("deliverable_resolved")
    assert quote_integrity_reason(row, require_open_interest=True) == "deliverable_metadata_unknown"
    row["deliverable_resolved"] = True
    row["adjusted_contract"] = "False"
    assert (
        quote_integrity_reason(row, require_open_interest=True)
        == "adjusted_contract_metadata_unknown"
    )


def test_hidden_232_veto_excludes_only_frozen_family() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["a", "b", "c"],
            "hidden_2_3_2_prior_6": [False, True, False],
            "any_hidden_event_prior_6": [False, True, True],
        }
    )
    assert apply_hidden_232_veto(frame)["trade_id"].tolist() == ["a", "c"]


def test_matched_controls_require_three_and_cap_at_five() -> None:
    rows = []
    for index in range(7):
        rows.append(
            {
                "trade_id": f"t{index}",
                "strategy": "S1",
                "symbol": "AAL",
                "session": date(2025, 1, 6 + index),
                "calendar_month": "2025-01",
                "weekday": 0,
                "entry_dte_bin": "7-9",
                "previous_close_atm_iv_quartile": "Q1",
                "valid_strategy_construction": True,
                "qualifying_signal": index == 0,
                "return_on_entry_debit": 0.1 * index,
            }
        )
    matched = build_matched_controls(pd.DataFrame(rows), treated_trade_ids=["t0"])
    assert bool(matched.iloc[0]["matched"])
    assert int(matched.iloc[0]["control_count"]) == 5


def test_bootstrap_uses_exactly_ten_whole_session_draws() -> None:
    trades = pd.DataFrame(
        {
            "strategy": ["S1", "S1", "S2_ALL", "S2_VETO", "S3"],
            "session": ["2025-01-02", "2025-01-03", "2025-01-02", "2025-01-02", "2025-01-03"],
            "return_on_entry_debit": [0.1, -0.1, 0.2, 0.3, 0.05],
            "win": [True, False, True, True, True],
            "matched_control_excess": [0.02, -0.02, float("nan"), float("nan"), 0.01],
        }
    )
    intervals = session_bootstrap_intervals(trades, draws=10, seed=20260723)
    assert set(intervals["draws"]) == {10}
    assert set(intervals["level"]) == {0.80, 0.90, 0.95}
    with pytest.raises(ValueError, match="exactly 10"):
        session_bootstrap_intervals(trades, draws=9, seed=20260723)


def test_protected_dates_and_option_high_low_are_rejected() -> None:
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"trade_date": ["2025-08-23"]}), ["trade_date"])
    with pytest.raises(ValueError, match="high/low"):
        assert_no_daily_option_high_low(pd.DataFrame({"option_high": [2.0]}))


def test_decision_logic_prefers_exact_single_strategy_category() -> None:
    assert (
        choose_overall_decision(
            {"S1": True, "S2": False, "S3": False},
            hidden_veto_positive=False,
            any_supported=True,
        )
        == "overnight_straddle_feasible_only"
    )
    assert (
        choose_overall_decision(
            {"S1": False, "S2": False, "S3": False},
            hidden_veto_positive=False,
            any_supported=False,
        )
        == "descriptive_options_strategy_results_only"
    )


def _load_experiment_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "run_screen_v0.py"
    specification = importlib.util.spec_from_file_location("test_fixed_options_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_complete_cache_pipeline_constructs_causal_s1_and_s3_trades() -> None:
    runner = _load_experiment_runner()
    signal_rows = []
    option_rows = []
    fixtures = (
        ("2024-01-04", "2024-01-03", "2024-01-05", "2024-01-12", 0.30),
        ("2024-01-08", "2024-01-05", "2024-01-09", "2024-01-19", 0.50),
    )
    for session, selection_date, exit_date, s1_expiry, iv in fixtures:
        signal_rows.append(
            {
                "row_id": f"AAL|{session}|72",
                "symbol": "AAL",
                "session": session,
                "period": "development",
                "contract_selection_date": selection_date,
                "exit_session": exit_date,
                "ordinal_72_structural_available": True,
                "chronology_eligible": True,
                "underlying_source_available": True,
                "split_boundary_ambiguous": False,
                "previous_close_underlying_price": 10.0,
                "entry_underlying_close": 10.0,
                "exit_underlying_close": 10.5,
                "route_resolution_state": "BROAD_CONFLICT",
                "BROAD_CONFLICT": True,
                "recent_registered_completion_prior_6": False,
                "hidden_2_3_2_prior_6": False,
                "any_hidden_event_prior_6": False,
            }
        )
        for expiry in (s1_expiry, exit_date):
            expiry_token = date.fromisoformat(expiry).strftime("%y%m%d")
            for option_type, code in (("call", "C"), ("put", "P")):
                contract_id = f"AAL{expiry_token}{code}00010000"
                for trade_date, bid, ask in (
                    (selection_date, 1.0, 1.2),
                    (session, 1.0, 1.2),
                    (exit_date, 1.4, 1.6),
                ):
                    option_rows.append(
                        {
                            "underlying_symbol": "AAL",
                            "contract_id": contract_id,
                            "option_type": option_type,
                            "expiration_date": date.fromisoformat(expiry),
                            "strike": 10.0,
                            "trade_date": date.fromisoformat(trade_date),
                            "bid": bid,
                            "ask": ask,
                            "midpoint": (bid + ask) / 2.0,
                            "open_interest": 100,
                            "implied_volatility": iv,
                            "delta": 0.50 if option_type == "call" else -0.50,
                            "raw_record_hash": (f"{contract_id}|{trade_date}".encode().hex()[:64]),
                            "adjusted_contract": False,
                            "deliverable_resolved": True,
                            "contract_multiplier": 100,
                            "settlement_style": "standard_equity_pm",
                            "chain_complete": True,
                            "cache_source": "test",
                        }
                    )
    options = pd.DataFrame(option_rows).drop_duplicates(["contract_id", "trade_date"], keep="first")
    selected, trades, _candidates, quote_audit = runner.construct_straddle_economics(
        pd.DataFrame(signal_rows), options
    )
    assert set(selected.loc[selected["selection_status"].eq("selected"), "strategy"]) == {
        "S1",
        "S3",
    }
    assert set(trades["strategy"]) == {"S1", "S3"}
    assert trades["session"].eq("2024-01-04").all()
    summaries = quote_audit.loc[quote_audit["quote_role"].eq("selection_summary")]
    assert len(summaries) == 4
    assert summaries["passed"].all()

    invalid_options = options.copy()
    invalid_options.loc[
        invalid_options["contract_id"].eq("AAL240105P00010000")
        & invalid_options["trade_date"].eq(date(2024, 1, 4)),
        "settlement_style",
    ] = "ambiguous"
    invalid_options = invalid_options.loc[
        ~(
            invalid_options["contract_id"].eq("AAL240112C00010000")
            & invalid_options["trade_date"].eq(date(2024, 1, 5))
        )
    ]
    invalid_options.loc[
        invalid_options["contract_id"].eq("AAL240112P00010000")
        & invalid_options["trade_date"].eq(date(2024, 1, 5)),
        "ask",
    ] = None
    _selected, invalid_trades, _candidates, invalid_audit = runner.construct_straddle_economics(
        pd.DataFrame(signal_rows), invalid_options
    )
    rejected = invalid_audit.loc[
        invalid_audit["quote_role"].eq("selection_summary")
        & invalid_audit["signal_session"].eq("2024-01-04")
    ]
    assert set(rejected["strategy"]) == {"S1", "S3"}
    assert rejected["passed"].eq(False).all()  # noqa: E712 - explicit audit value
    assert (
        rejected.loc[rejected["strategy"].eq("S1"), "detail"]
        .str.contains("call_exit:missing_quote")
        .all()
    )
    assert (
        rejected.loc[rejected["strategy"].eq("S1"), "detail"]
        .str.contains("put_exit:ask_invalid_or_crossed")
        .all()
    )
    assert rejected.loc[rejected["strategy"].eq("S3"), "detail"].str.contains("settlement").all()
    assert invalid_trades.empty

    unknown_chain_metadata = options.assign(chain_complete="False")
    metadata_selected, _trades, _candidates, _audit = runner.construct_straddle_economics(
        pd.DataFrame(signal_rows), unknown_chain_metadata
    )
    assert metadata_selected["selection_status"].eq("rejected").all()
    assert (
        metadata_selected["rejection_reason"]
        .str.startswith("chain_complete_metadata_invalid")
        .all()
    )


def test_exit_session_corporate_action_boundary_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_experiment_runner()
    monkeypatch.setattr(runner, "FROZEN_COHORT", ("AAL",))
    provider_root = tmp_path / "processed"
    path = provider_root / "symbol=AAL" / "timeframe=5m" / "data.parquet"
    path.parent.mkdir(parents=True)
    source_rows = pd.DataFrame(
        [
            {"timestamp": "2024-01-02T14:30:00Z", "open": 10.0, "close": 10.0},
            {"timestamp": "2024-01-02T20:55:00Z", "open": 10.0, "close": 10.0},
            {"timestamp": "2024-01-03T14:30:00Z", "open": 10.0, "close": 10.0},
            {"timestamp": "2024-01-03T20:55:00Z", "open": 10.0, "close": 10.0},
            {"timestamp": "2024-01-04T14:30:00Z", "open": 20.0, "close": 20.0},
            {"timestamp": "2024-01-04T20:55:00Z", "open": 20.0, "close": 20.0},
        ]
    )
    source_rows["timestamp"] = pd.to_datetime(source_rows["timestamp"], utc=True)
    source_rows.to_parquet(path, index=False)
    skeleton = pd.DataFrame(
        [
            {
                "symbol": "AAL",
                "session": "2024-01-03",
                "contract_selection_date": "2024-01-02",
                "unprotected_next_session": "2024-01-04",
            }
        ]
    )
    audit, _hashes = runner.underlying_price_audit(skeleton, provider_root=provider_root)
    assert bool(audit.iloc[0]["inferred_split_on_exit_date"])
    assert bool(audit.iloc[0]["split_boundary_ambiguous"])
    assert bool(audit.iloc[0]["underlying_source_available"])


def test_cache_completeness_is_strategy_specific(tmp_path: Path) -> None:
    runner = _load_experiment_runner()
    rows: list[dict[str, object]] = []
    for option_type, code in (("call", "C"), ("put", "P")):
        row = {column: None for column in runner.CANONICAL_OPTION_COLUMNS}
        row.update(
            {
                "provider": "EODHD",
                "provider_schema_version": "test",
                "request_id": "test-s1",
                "underlying_symbol": "AAL",
                "contract_id": f"AAL250117{code}00100000",
                "option_type": option_type,
                "expiration_date": date(2025, 1, 17),
                "strike": 100.0,
                "trade_date": date(2025, 1, 3),
                "bid": 1.0,
                "ask": 1.2,
                "midpoint": 1.1,
                "open_interest": 100,
                "implied_volatility": 0.4,
                "raw_record_hash": code * 64,
                "adjusted_contract": False,
                "deliverable_resolved": True,
                "contract_multiplier": 100,
                "settlement_style": "standard_equity_pm",
                "chain_complete": True,
                "cache_source": "test",
                "request_strategy": "S1",
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp_path / "s1.parquet", index=False)
    required = pd.DataFrame(
        [
            {
                "symbol": "AAL",
                "option_date": "2025-01-03",
                "strategy": strategy,
                "role": "contract_preselection",
                "signal_session": "2025-01-06",
            }
            for strategy in ("S1", "S3")
        ]
    )
    gaps, _summary, _quote_integrity, _canonical = runner.inspect_existing_option_cache(
        required, tmp_path
    )
    assert gaps["strategies"].tolist() == ["S3"]
    assert gaps.iloc[0]["gap_reason"] == "S3_bounded_chain_coverage_absent"


def test_month_concentration_gate_cannot_be_reported_as_feasible() -> None:
    runner = _load_experiment_runner()
    rows: list[dict[str, object]] = []
    assessment_months = ["2025-01"] * 19 + [
        f"2025-{month:02d}" for month in (2, 3, 4, 5, 6) for _ in range(9)
    ][:41]
    for index, month in enumerate(assessment_months):
        rows.append(
            {
                "trade_id": f"a{index}",
                "strategy": "S1",
                "period": "assessment",
                "symbol": f"S{index % 10}",
                "session": f"{month}-{(index % 20) + 1:02d}",
                "calendar_month": month,
                "total_initial_cash_debit": 100.0,
                "net_pnl": 1.0,
                "return_on_entry_debit": 0.01,
                "matched": False,
                "matched_control_excess": float("nan"),
            }
        )
    for index in range(10):
        rows.append(
            {
                "trade_id": f"d{index}",
                "strategy": "S1",
                "period": "development",
                "symbol": f"S{index}",
                "session": f"2024-01-{index + 2:02d}",
                "calendar_month": "2024-01",
                "total_initial_cash_debit": 100.0,
                "net_pnl": 1.0,
                "return_on_entry_debit": 0.01,
                "matched": False,
                "matched_control_excess": float("nan"),
            }
        )
    trades = pd.DataFrame(rows)
    bootstrap = pd.DataFrame(
        [
            {
                "statistic": "s1_mean_return_on_debit",
                "level": 0.80,
                "lower": 0.01,
                "upper": 0.01,
            }
        ]
    )
    support, positive, _reasons = runner._support_and_positive(trades, bootstrap, "S1")
    assert support
    assert not positive
