from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.event_ingest import IBKRCallbackNormalizer, StreamKind, StreamOwner
from stocker_prospective.events import OptionQuoteEvent, UnderlyingLevel1QuoteEvent
from stocker_prospective.frozen_m1c import FrozenM1CScore
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.option_recorder import BoundedOptionRecorder
from stocker_prospective.option_risk_accounting import (
    MARGIN_UNAVAILABLE,
    THETA_ATTRIBUTION_INCOMPLETE,
    GreekSourceSnapshot,
    MarginEstimate,
    OptionLeg,
    OptionLegQuote,
    OptionRiskAccountingRecord,
    OptionStrategy,
    OptionStrategySnapshot,
    StrategyType,
    TransactionCosts,
    UnderlyingQuoteSnapshot,
    UnderlyingStrategy,
    build_strategy_comparison_report,
    calculate_option_strategy_path,
    calculate_underlying_strategy_path,
    option_leg_quote_from_event,
)
from stocker_prospective.options import DteBucket
from stocker_prospective.quiet_state import QuietEpisodeTracker, classify_quiet_state
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository

ENTRY = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)


def _quiet_repository(
    tmp_path: Path,
) -> tuple[FrozenRecorderRepository, ProspectiveReadStore, EvidenceMetadata, str]:
    database = ProspectiveRepository(tmp_path / "risk-accounting.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="risk-accounting",
        prospective_start_utc=ENTRY,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="b" * 64,
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[ENTRY.isoformat()],
        recorded_at_utc=ENTRY,
    )
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    score = FrozenM1CScore(
        model_hash="b" * 64,
        probability=0.13,
        threshold=0.488333710794033,
        threshold_passed=False,
        feature_order=("x",),
        feature_values=(1.0,),
        transformed_values=(1.0,),
        feature_hash="c" * 64,
        missing_feature_count=0,
    )
    checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="TEST",
        session=ENTRY.date(),
        checkpoint=6,
        bar_start_utc=ENTRY - timedelta(minutes=5),
        bar_end_utc=ENTRY,
        score=score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
    )
    snapshot = classify_quiet_state(
        probability=score.probability,
        previous_probability=None,
        model_hash=score.model_hash,
        feature_hash=score.feature_hash,
        data_quality_status="valid",
    )
    quiet_checkpoint_id = repository.record_quiet_checkpoint(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="TEST",
        session=ENTRY.date(),
        checkpoint=6,
        snapshot=snapshot,
        eligible=True,
    )
    decision = QuietEpisodeTracker().evaluate(
        symbol="TEST",
        session=ENTRY.date(),
        checkpoint=6,
        trigger_bar_end=ENTRY,
        probability=score.probability,
    )
    observation_id = repository.record_quiet_episode(
        metadata,
        quiet_checkpoint_id=quiet_checkpoint_id,
        decision=decision,
        scientific_recording_valid=True,
    )
    return repository, ProspectiveReadStore(database.database_path), metadata, observation_id


def _option_event(
    contract: OptionContract,
    *,
    observed_at: datetime,
    bid: float,
    ask: float,
    delta: float,
    sequence: int,
) -> OptionQuoteEvent:
    assert contract.con_id is not None
    return OptionQuoteEvent(
        event_id=f"option-{contract.con_id}-{sequence}",
        received_timestamp_utc=observed_at,
        received_monotonic_ns=sequence,
        provider_timestamp_utc=observed_at,
        source_sequence=sequence,
        session=ENTRY.date(),
        symbol="TEST",
        con_id=contract.con_id,
        request_id=contract.con_id,
        episode_id="quiet-accounting",
        expiry=contract.expiry,
        dte=contract.dte,
        dte_bucket=contract.dte_bucket,
        strike=contract.strike,
        right=contract.right,
        multiplier=contract.multiplier,
        exchange=contract.exchange,
        trading_class=contract.trading_class,
        bid=bid,
        bid_size=10.0,
        ask=ask,
        ask_size=10.0,
        last=(bid + ask) / 2.0,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        option_computation_by_source={
            "model": {
                "implied_volatility": 0.30,
                "delta": delta,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.10,
                "option_price": (bid + ask) / 2.0,
                "underlying_reference_price": 100.0,
                "greek_timestamp_utc": observed_at.isoformat(),
                "market_data_status": "live",
            }
        },
    )


def _underlying_event(
    *,
    observed_at: datetime,
    bid: float,
    ask: float,
    sequence: int,
) -> UnderlyingLevel1QuoteEvent:
    return UnderlyingLevel1QuoteEvent(
        event_id=f"underlying-{sequence}",
        received_timestamp_utc=observed_at,
        received_monotonic_ns=sequence,
        provider_timestamp_utc=observed_at,
        source_sequence=sequence,
        session=ENTRY.date(),
        symbol="TEST",
        con_id=1,
        request_id=1,
        bid=bid,
        bid_size=100.0,
        ask=ask,
        ask_size=100.0,
        last=(bid + ask) / 2.0,
        last_size=10.0,
        market_data_type=MarketDataType.LIVE,
        source="ibkr",
        quote_valid=True,
        tick_type="level1",
        exchange="SMART",
    )


def test_short_put_uses_bid_entry_ask_exit_and_signed_contract_quantity() -> None:
    strategy = OptionStrategy(
        strategy_id="short-put-50",
        strategy_type=StrategyType.SHORT_PUT,
        legs=(
            OptionLeg(
                leg_id="put-50",
                right="P",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=-1,
            ),
        ),
        costs=TransactionCosts(
            commissions=1.30,
            regulatory_fees=0.10,
            exchange_fees=0.20,
        ),
    )
    records = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="put-50",
                        quote_timestamp=ENTRY,
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                    ),
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(hours=1),
                legs=(
                    OptionLegQuote(
                        leg_id="put-50",
                        quote_timestamp=ENTRY + timedelta(hours=1),
                        bid=0.90,
                        ask=1.00,
                        last=0.95,
                        market_data_status="live",
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(hours=2),
    )

    result = records[-1]
    assert result.option_multiplier == 100
    assert result.gross_entry_credit == pytest.approx(200.0)
    assert result.gross_entry_debit == 0.0
    assert result.gross_executable_pnl == pytest.approx(100.0)
    assert result.net_option_pnl == pytest.approx(98.40)
    assert result.commissions == pytest.approx(1.30)
    assert result.regulatory_fees == pytest.approx(0.10)
    assert result.exchange_fees == pytest.approx(0.20)
    assert result.cash_secured_capital == pytest.approx(4_800.0)
    assert result.theoretical_maximum_loss == pytest.approx(4_801.60)
    assert result.short_put_max_loss == pytest.approx(4_801.60)
    assert result.cash_secured_roi == pytest.approx(98.40 / 4_800.0)
    assert result.full_risk_roi == pytest.approx(98.40 / 4_801.60)
    assert result.gross_premium_yield == pytest.approx(200.0 / 4_800.0)
    assert result.net_premium_yield == pytest.approx(198.40 / 4_800.0)
    assert result.capital_hours_employed == pytest.approx(4_800.0)
    assert result.return_per_1000_reserved_capital == pytest.approx(20.5)
    assert result.ibkr_initial_margin == MARGIN_UNAVAILABLE
    assert result.ibkr_maintenance_margin == MARGIN_UNAVAILABLE
    assert result.maximum_observed_initial_margin == MARGIN_UNAVAILABLE
    assert result.maximum_observed_maintenance_margin == MARGIN_UNAVAILABLE


def test_long_option_uses_ask_entry_bid_exit_and_premium_specific_roi() -> None:
    strategy = OptionStrategy(
        strategy_id="long-call-55",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="call-55",
                right="C",
                strike=55.0,
                multiplier=50,
                signed_contract_quantity=1,
            ),
        ),
        costs=TransactionCosts(commissions=0.70, regulatory_fees=0.10, exchange_fees=0.20),
    )
    records = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="call-55",
                        quote_timestamp=ENTRY,
                        bid=1.80,
                        ask=2.00,
                        last=1.90,
                        market_data_status="live",
                    ),
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(minutes=30),
                legs=(
                    OptionLegQuote(
                        leg_id="call-55",
                        quote_timestamp=ENTRY + timedelta(minutes=30),
                        bid=3.00,
                        ask=3.20,
                        last=3.10,
                        market_data_status="live",
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(hours=1),
    )

    result = records[-1]
    assert result.option_multiplier == 50
    assert result.gross_entry_debit == pytest.approx(100.0)
    assert result.gross_entry_credit == 0.0
    assert result.gross_executable_pnl == pytest.approx(50.0)
    assert result.net_option_pnl == pytest.approx(49.0)
    assert result.total_premium_paid == pytest.approx(100.0)
    assert result.premium_roi == pytest.approx(49.0 / 100.0)
    assert result.theoretical_maximum_loss == pytest.approx(101.0)
    assert result.midpoint_pnl == pytest.approx(60.0)
    assert result.bid_ask_cost == pytest.approx(10.0)
    assert result.cash_secured_roi is None
    assert result.full_risk_roi is None
    assert "roi" not in result.model_dump()
    assert not any("annual" in field for field in type(result).model_fields)


def test_missing_executable_entry_mark_is_not_turned_into_zero_capital() -> None:
    strategy = OptionStrategy(
        strategy_id="missing-entry-call",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="call",
                right="C",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="executable entry quote"):
        calculate_option_strategy_path(
            strategy=strategy,
            snapshots=(
                OptionStrategySnapshot(
                    observed_at=ENTRY,
                    legs=(
                        OptionLegQuote(
                            leg_id="call",
                            quote_timestamp=ENTRY,
                            bid=1.00,
                            ask=None,
                            last=1.10,
                            market_data_status="live",
                        ),
                    ),
                ),
            ),
            maximum_attribution_gap=timedelta(minutes=5),
        )


def test_bull_put_spread_uses_each_executable_side_and_defined_risk_capital() -> None:
    strategy = OptionStrategy(
        strategy_id="bull-put-50-45",
        strategy_type=StrategyType.BULL_PUT_SPREAD,
        legs=(
            OptionLeg(
                leg_id="short-put-50",
                right="P",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=-1,
            ),
            OptionLeg(
                leg_id="long-put-45",
                right="P",
                strike=45.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
        costs=TransactionCosts(commissions=2.60, regulatory_fees=0.10, exchange_fees=0.10),
    )
    records = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="short-put-50",
                        quote_timestamp=ENTRY,
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                    ),
                    OptionLegQuote(
                        leg_id="long-put-45",
                        quote_timestamp=ENTRY,
                        bid=0.40,
                        ask=0.50,
                        last=0.45,
                        market_data_status="live",
                    ),
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(hours=2),
                legs=(
                    OptionLegQuote(
                        leg_id="short-put-50",
                        quote_timestamp=ENTRY + timedelta(hours=2),
                        bid=1.10,
                        ask=1.20,
                        last=1.15,
                        market_data_status="live",
                    ),
                    OptionLegQuote(
                        leg_id="long-put-45",
                        quote_timestamp=ENTRY + timedelta(hours=2),
                        bid=0.30,
                        ask=0.40,
                        last=0.35,
                        market_data_status="live",
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(hours=3),
    )

    result = records[-1]
    assert result.gross_entry_credit == pytest.approx(150.0)
    assert result.gross_executable_pnl == pytest.approx(60.0)
    assert result.net_option_pnl == pytest.approx(57.20)
    assert result.spread_max_loss == pytest.approx(352.80)
    assert result.theoretical_maximum_loss == pytest.approx(352.80)
    assert result.defined_risk_capital == pytest.approx(352.80)
    assert result.defined_risk_roi == pytest.approx(57.20 / 352.80)
    assert result.midpoint_pnl == pytest.approx(85.0)
    assert result.bid_ask_cost == pytest.approx(25.0)
    assert result.capital_hours_employed == pytest.approx(705.60)


def test_short_put_uses_reliable_entry_and_peak_initial_margin_for_distinct_rois() -> None:
    strategy = OptionStrategy(
        strategy_id="margin-short-put",
        strategy_type=StrategyType.SHORT_PUT,
        legs=(
            OptionLeg(
                leg_id="put-50",
                right="P",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=-1,
            ),
        ),
    )
    records = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="put-50",
                        quote_timestamp=ENTRY,
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                    ),
                ),
                margin_estimate=MarginEstimate(
                    initial_margin=1_000.0,
                    maintenance_margin=800.0,
                    reliable=True,
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(hours=1),
                legs=(
                    OptionLegQuote(
                        leg_id="put-50",
                        quote_timestamp=ENTRY + timedelta(hours=1),
                        bid=0.90,
                        ask=1.00,
                        last=0.95,
                        market_data_status="live",
                    ),
                ),
                margin_estimate=MarginEstimate(
                    initial_margin=1_200.0,
                    maintenance_margin=950.0,
                    reliable=True,
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(hours=2),
    )

    result = records[-1]
    assert result.entry_initial_margin == pytest.approx(1_000.0)
    assert result.entry_maintenance_margin == pytest.approx(800.0)
    assert result.ibkr_initial_margin == pytest.approx(1_200.0)
    assert result.ibkr_maintenance_margin == pytest.approx(950.0)
    assert result.maximum_observed_initial_margin == pytest.approx(1_200.0)
    assert result.maximum_observed_maintenance_margin == pytest.approx(950.0)
    assert result.entry_margin_roi == pytest.approx(100.0 / 1_000.0)
    assert result.peak_margin_roi == pytest.approx(100.0 / 1_200.0)


def test_spread_position_greeks_preserve_sources_and_do_not_substitute_model_values() -> None:
    strategy = OptionStrategy(
        strategy_id="greek-spread",
        strategy_type=StrategyType.BULL_PUT_SPREAD,
        legs=(
            OptionLeg(
                leg_id="short-put",
                right="P",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=-1,
            ),
            OptionLeg(
                leg_id="long-put",
                right="P",
                strike=45.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )
    greek_time = ENTRY - timedelta(seconds=1)
    source = GreekSourceSnapshot(
        implied_volatility=0.40,
        delta=-0.40,
        gamma=0.02,
        theta=-0.05,
        vega=0.10,
        option_model_price=2.10,
        underlying_model_reference_price=50.0,
        greek_timestamp=greek_time,
        market_data_status="live",
    )
    records = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="short-put",
                        quote_timestamp=ENTRY - timedelta(seconds=2),
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                        greeks_by_source={
                            "bid": source.model_copy(update={"delta": -0.41}),
                            "ask": source.model_copy(update={"delta": -0.39}),
                            "last": source.model_copy(update={"delta": -0.40}),
                            "model": source,
                        },
                    ),
                    OptionLegQuote(
                        leg_id="long-put",
                        quote_timestamp=ENTRY - timedelta(seconds=2),
                        bid=0.40,
                        ask=0.50,
                        last=0.45,
                        market_data_status="live",
                        greeks_by_source={
                            "model": GreekSourceSnapshot(
                                implied_volatility=0.35,
                                delta=-0.20,
                                gamma=0.01,
                                theta=-0.02,
                                vega=0.05,
                                option_model_price=0.45,
                                underlying_model_reference_price=49.5,
                                greek_timestamp=greek_time,
                                market_data_status="live",
                            )
                        },
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(minutes=1),
    )

    result = records[0]
    model = result.position_greeks_by_source["model"]
    assert model.delta == pytest.approx(20.0)
    assert model.gamma == pytest.approx(-1.0)
    assert model.theta == pytest.approx(3.0)
    assert model.vega == pytest.approx(-5.0)
    assert result.position_greeks_by_source["bid"].delta is None
    assert set(result.greek_observations_by_leg["short-put"]) == {
        "bid",
        "ask",
        "last",
        "model",
    }
    assert result.greek_observations_by_leg["short-put"]["model"].greek_timestamp == greek_time
    assert result.quote_age_seconds_by_leg == {"short-put": 2.0, "long-put": 2.0}
    assert result.market_data_status_by_leg == {"short-put": "live", "long-put": "live"}
    assert result.net_delta_exposure == pytest.approx(20.0)
    assert result.net_gamma_exposure == pytest.approx(-1.0)
    assert result.net_vega_exposure == pytest.approx(-5.0)
    assert result.delta_equivalent_underlying_exposure == pytest.approx(1_010.0)


def test_theta_uses_trapezoidal_calendar_day_integration_at_irregular_intervals() -> None:
    strategy = OptionStrategy(
        strategy_id="theta-long-call",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="call",
                right="C",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )

    def snapshot(hours: int, theta: float) -> OptionStrategySnapshot:
        observed = ENTRY + timedelta(hours=hours)
        return OptionStrategySnapshot(
            observed_at=observed,
            legs=(
                OptionLegQuote(
                    leg_id="call",
                    quote_timestamp=observed,
                    bid=2.00,
                    ask=2.10,
                    last=2.05,
                    market_data_status="live",
                    greeks_by_source={
                        "model": GreekSourceSnapshot(
                            theta=theta,
                            greek_timestamp=observed,
                            market_data_status="live",
                        )
                    },
                ),
            ),
        )

    records = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(snapshot(0, -0.10), snapshot(6, -0.06), snapshot(18, -0.02)),
        maximum_attribution_gap=timedelta(days=1),
    )

    assert records[1].theta_interval_contribution == pytest.approx(-2.0)
    assert records[2].theta_interval_contribution == pytest.approx(-2.0)
    assert records[2].estimated_theta_contribution == pytest.approx(-4.0)
    assert records[2].theta_attribution_status == "COMPLETE"


@pytest.mark.parametrize(
    ("middle_theta", "maximum_gap"),
    ((None, timedelta(days=1)), (-0.06, timedelta(hours=5))),
)
def test_theta_records_incomplete_for_missing_observations_or_long_quote_gaps(
    middle_theta: float | None,
    maximum_gap: timedelta,
) -> None:
    strategy = OptionStrategy(
        strategy_id="incomplete-theta",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="call",
                right="C",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )

    def snapshot(hours: int, theta: float | None) -> OptionStrategySnapshot:
        observed = ENTRY + timedelta(hours=hours)
        return OptionStrategySnapshot(
            observed_at=observed,
            legs=(
                OptionLegQuote(
                    leg_id="call",
                    quote_timestamp=observed,
                    bid=2.00,
                    ask=2.10,
                    last=2.05,
                    market_data_status="live",
                    greeks_by_source={
                        "model": GreekSourceSnapshot(
                            theta=theta,
                            greek_timestamp=observed,
                            market_data_status="live",
                        )
                    },
                ),
            ),
        )

    result = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(snapshot(0, -0.10), snapshot(6, middle_theta)),
        maximum_attribution_gap=maximum_gap,
    )[-1]

    assert result.estimated_theta_contribution is None
    assert result.theta_interval_contribution is None
    assert result.theta_attribution_status == THETA_ATTRIBUTION_INCOMPLETE


def test_theta_records_incomplete_when_sticky_greek_timestamp_is_stale() -> None:
    strategy = OptionStrategy(
        strategy_id="stale-theta",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="call",
                right="C",
                strike=50.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )

    def snapshot(observed: datetime, greek_timestamp: datetime) -> OptionStrategySnapshot:
        return OptionStrategySnapshot(
            observed_at=observed,
            legs=(
                OptionLegQuote(
                    leg_id="call",
                    quote_timestamp=observed,
                    bid=2.00,
                    ask=2.10,
                    last=2.05,
                    market_data_status="live",
                    greeks_by_source={
                        "model": GreekSourceSnapshot(
                            theta=-0.10,
                            greek_timestamp=greek_timestamp,
                            market_data_status="live",
                        )
                    },
                ),
            ),
        )

    result = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            snapshot(ENTRY, ENTRY - timedelta(minutes=5)),
            snapshot(ENTRY + timedelta(minutes=1), ENTRY - timedelta(minutes=5)),
        ),
        maximum_attribution_gap=timedelta(minutes=2),
    )[-1]

    assert result.estimated_theta_contribution is None
    assert result.theta_interval_contribution is None
    assert result.theta_attribution_status == THETA_ATTRIBUTION_INCOMPLETE


def test_greek_attribution_uses_frozen_units_and_records_model_price_residual() -> None:
    strategy = OptionStrategy(
        strategy_id="attribution-call",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="call",
                right="C",
                strike=100.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )

    def snapshot(
        observed: datetime,
        *,
        bid: float,
        ask: float,
        implied_volatility: float,
        model_price: float,
        underlying: float,
    ) -> OptionStrategySnapshot:
        return OptionStrategySnapshot(
            observed_at=observed,
            legs=(
                OptionLegQuote(
                    leg_id="call",
                    quote_timestamp=observed,
                    bid=bid,
                    ask=ask,
                    last=(bid + ask) / 2.0,
                    market_data_status="live",
                    greeks_by_source={
                        "model": GreekSourceSnapshot(
                            implied_volatility=implied_volatility,
                            delta=0.50,
                            gamma=0.02,
                            theta=-0.10,
                            vega=0.05,
                            option_model_price=model_price,
                            underlying_model_reference_price=underlying,
                            greek_timestamp=observed,
                            market_data_status="live",
                        )
                    },
                ),
            ),
        )

    result = calculate_option_strategy_path(
        strategy=strategy,
        snapshots=(
            snapshot(
                ENTRY,
                bid=1.90,
                ask=2.10,
                implied_volatility=0.20,
                model_price=2.00,
                underlying=100.0,
            ),
            snapshot(
                ENTRY + timedelta(days=1),
                bid=3.00,
                ask=3.20,
                implied_volatility=0.21,
                model_price=3.20,
                underlying=102.0,
            ),
        ),
        maximum_attribution_gap=timedelta(days=2),
    )[-1]

    attribution = result.greek_attribution_interval
    assert attribution is not None
    assert attribution.delta_contribution == pytest.approx(100.0)
    assert attribution.gamma_contribution == pytest.approx(4.0)
    assert attribution.theta_contribution == pytest.approx(-10.0)
    assert attribution.vega_contribution == pytest.approx(5.0)
    assert attribution.total_estimated_contribution == pytest.approx(99.0)
    assert attribution.model_or_midpoint_change == pytest.approx(120.0)
    assert attribution.change_basis == "model"
    assert attribution.greek_residual == pytest.approx(21.0)
    assert result.model_price_pnl == pytest.approx(120.0)
    assert result.greek_residual == pytest.approx(21.0)
    assert attribution.diagnostic_only is True


def test_underlying_long_uses_entry_notional_and_executable_bid_ask_marks() -> None:
    strategy = UnderlyingStrategy(
        strategy_id="underlying-long",
        quantity=100,
        costs=TransactionCosts(commissions=1.00, regulatory_fees=0.20, exchange_fees=0.30),
    )
    records = calculate_underlying_strategy_path(
        strategy=strategy,
        snapshots=(
            UnderlyingQuoteSnapshot(
                observed_at=ENTRY,
                quote_timestamp=ENTRY,
                bid=49.90,
                ask=50.10,
                last=50.00,
                market_data_status="live",
            ),
            UnderlyingQuoteSnapshot(
                observed_at=ENTRY + timedelta(minutes=30),
                quote_timestamp=ENTRY + timedelta(minutes=30),
                bid=51.00,
                ask=51.20,
                last=51.10,
                market_data_status="live",
            ),
        ),
    )

    result = records[-1]
    assert result.entry_underlying_notional == pytest.approx(5_010.0)
    assert result.gross_underlying_pnl == pytest.approx(90.0)
    assert result.net_underlying_pnl == pytest.approx(88.50)
    assert result.underlying_roi == pytest.approx(88.50 / 5_010.0)
    assert result.midpoint_pnl == pytest.approx(110.0)
    assert result.bid_ask_cost == pytest.approx(20.0)
    assert result.account_independent_notional_exposure == pytest.approx(5_010.0)
    assert result.capital_hours_employed == pytest.approx(2_505.0)
    assert result.return_per_1000_reserved_capital == pytest.approx(88.50 / 5_010.0 * 1_000.0)
    assert result.can_authorize_trade is False


def test_comparison_report_keeps_strategy_specific_returns_and_tail_metrics() -> None:
    underlying_path = calculate_underlying_strategy_path(
        strategy=UnderlyingStrategy(strategy_id="underlying", quantity=100),
        snapshots=(
            UnderlyingQuoteSnapshot(
                observed_at=ENTRY,
                quote_timestamp=ENTRY,
                bid=49.90,
                ask=50.10,
                last=50.00,
                market_data_status="live",
            ),
            UnderlyingQuoteSnapshot(
                observed_at=ENTRY + timedelta(hours=1),
                quote_timestamp=ENTRY + timedelta(hours=1),
                bid=51.00,
                ask=51.20,
                last=51.10,
                market_data_status="live",
            ),
        ),
    )
    short_put_path = calculate_option_strategy_path(
        strategy=OptionStrategy(
            strategy_id="short-put",
            strategy_type=StrategyType.SHORT_PUT,
            legs=(
                OptionLeg(
                    leg_id="short",
                    right="P",
                    strike=50.0,
                    multiplier=100,
                    signed_contract_quantity=-1,
                ),
            ),
        ),
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="short",
                        quote_timestamp=ENTRY,
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                    ),
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(hours=1),
                legs=(
                    OptionLegQuote(
                        leg_id="short",
                        quote_timestamp=ENTRY + timedelta(hours=1),
                        bid=0.90,
                        ask=1.00,
                        last=0.95,
                        market_data_status="live",
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(hours=2),
    )
    bull_put_path = calculate_option_strategy_path(
        strategy=OptionStrategy(
            strategy_id="bull-put",
            strategy_type=StrategyType.BULL_PUT_SPREAD,
            legs=(
                OptionLeg(
                    leg_id="short",
                    right="P",
                    strike=50.0,
                    multiplier=100,
                    signed_contract_quantity=-1,
                ),
                OptionLeg(
                    leg_id="long",
                    right="P",
                    strike=45.0,
                    multiplier=100,
                    signed_contract_quantity=1,
                ),
            ),
        ),
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="short",
                        quote_timestamp=ENTRY,
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                    ),
                    OptionLegQuote(
                        leg_id="long",
                        quote_timestamp=ENTRY,
                        bid=0.40,
                        ask=0.50,
                        last=0.45,
                        market_data_status="live",
                    ),
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(hours=1),
                legs=(
                    OptionLegQuote(
                        leg_id="short",
                        quote_timestamp=ENTRY + timedelta(hours=1),
                        bid=1.10,
                        ask=1.20,
                        last=1.15,
                        market_data_status="live",
                    ),
                    OptionLegQuote(
                        leg_id="long",
                        quote_timestamp=ENTRY + timedelta(hours=1),
                        bid=0.30,
                        ask=0.40,
                        last=0.35,
                        market_data_status="live",
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(hours=2),
    )

    report = build_strategy_comparison_report(
        underlying_long=underlying_path,
        short_put=short_put_path,
        bull_put_spread=bull_put_path,
    )
    rows = {row.strategy_type: row for row in report.strategies}

    assert set(rows) == {"UNDERLYING_LONG", "SHORT_PUT", "BULL_PUT_SPREAD"}
    assert rows["UNDERLYING_LONG"].underlying_roi == underlying_path[-1].underlying_roi
    assert rows["SHORT_PUT"].cash_secured_roi == short_put_path[-1].cash_secured_roi
    assert rows["BULL_PUT_SPREAD"].defined_risk_roi == bull_put_path[-1].defined_risk_roi
    assert rows["SHORT_PUT"].maximum_theoretical_loss == pytest.approx(4_800.0)
    assert rows["BULL_PUT_SPREAD"].maximum_theoretical_loss == pytest.approx(350.0)
    assert rows["SHORT_PUT"].maximum_observed_margin == MARGIN_UNAVAILABLE
    assert rows["SHORT_PUT"].position_greek_source == "model"
    assert rows["UNDERLYING_LONG"].position_greek_source == "underlying"
    assert rows["UNDERLYING_LONG"].maximum_drawdown >= 0.0
    assert rows["SHORT_PUT"].expected_shortfall is not None
    assert rows["BULL_PUT_SPREAD"].bid_ask_cost == pytest.approx(25.0)
    assert not any("annual" in field for field in type(rows["SHORT_PUT"]).model_fields)
    assert "roi" not in rows["SHORT_PUT"].model_dump()
    assert "trade_roi_metric" not in rows["SHORT_PUT"].model_dump()
    assert "trade_roi_value" not in rows["SHORT_PUT"].model_dump()


def test_ibkr_quote_event_preserves_each_greek_source_timestamp_and_status() -> None:
    contract = OptionContract(
        underlying_con_id=1,
        con_id=2,
        expiry=ENTRY.date() + timedelta(days=1),
        dte=1,
        dte_bucket=DteBucket.ONE_DTE,
        strike=50.0,
        right="P",
        multiplier=100,
        exchange="SMART",
        trading_class="TEST",
    )
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=ENTRY - timedelta(minutes=1))
    normalizer.register(
        StreamOwner(
            request_id=7,
            kind=StreamKind.OPTION_LEVEL1,
            symbol="TEST",
            con_id=2,
            episode_id="episode",
            option_contract=contract,
        )
    )
    event = None
    for sequence, source in enumerate(("bid", "ask", "last", "model"), start=1):
        observed = ENTRY + timedelta(seconds=sequence)
        normalized = normalizer.normalize(
            {
                "kind": "level1_quote_update",
                "request_id": 7,
                "received_timestamp_utc": observed.isoformat(),
                "provider_timestamp_utc": observed.isoformat(),
                "received_monotonic_ns": sequence,
                "source_sequence": sequence,
                "field": "option_computation",
                "computation_source": source,
                "implied_volatility": 0.30 + sequence / 100.0,
                "delta": -0.40 + sequence / 100.0,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.10,
                "option_price": 2.00 + sequence / 10.0,
                "underlying_reference_price": 50.0,
                "market_data_type": "live",
            }
        )
        assert normalized is not None
        event = normalized.raw_event

    assert event is not None
    quote = option_leg_quote_from_event(event, leg_id="put")
    assert set(quote.greeks_by_source) == {"bid", "ask", "last", "model"}
    assert quote.greeks_by_source["bid"].greek_timestamp == ENTRY + timedelta(seconds=1)
    assert quote.greeks_by_source["model"].greek_timestamp == ENTRY + timedelta(seconds=4)
    assert quote.greeks_by_source["ask"].implied_volatility == pytest.approx(0.32)
    assert quote.greeks_by_source["last"].option_model_price == pytest.approx(2.30)
    assert all(item.market_data_status == "live" for item in quote.greeks_by_source.values())
    assert quote.quote_timestamp == ENTRY + timedelta(seconds=4)


def test_quiet_risk_observations_are_append_only_and_readable(tmp_path: Path) -> None:
    repository, read_store, metadata, observation_id = _quiet_repository(tmp_path)
    records = calculate_option_strategy_path(
        strategy=OptionStrategy(
            strategy_id="persisted-short-put",
            strategy_type=StrategyType.SHORT_PUT,
            legs=(
                OptionLeg(
                    leg_id="put",
                    right="P",
                    strike=50.0,
                    multiplier=100,
                    signed_contract_quantity=-1,
                ),
            ),
        ),
        snapshots=(
            OptionStrategySnapshot(
                observed_at=ENTRY,
                legs=(
                    OptionLegQuote(
                        leg_id="put",
                        quote_timestamp=ENTRY,
                        bid=2.00,
                        ask=2.20,
                        last=2.10,
                        market_data_status="live",
                    ),
                ),
            ),
            OptionStrategySnapshot(
                observed_at=ENTRY + timedelta(minutes=5),
                legs=(
                    OptionLegQuote(
                        leg_id="put",
                        quote_timestamp=ENTRY + timedelta(minutes=5),
                        bid=1.70,
                        ask=1.80,
                        last=1.75,
                        market_data_status="live",
                    ),
                ),
            ),
        ),
        maximum_attribution_gap=timedelta(minutes=10),
    )

    ids = tuple(
        repository.record_quiet_option_risk_observation(
            metadata,
            observation_id=observation_id,
            dte_bucket="1DTE",
            horizon_label="5m",
            record=record,
        )
        for record in records
    )
    assert (
        repository.record_quiet_option_risk_observation(
            metadata,
            observation_id=observation_id,
            dte_bucket="1DTE",
            horizon_label="5m",
            record=records[-1],
        )
        == ids[-1]
    )

    projection = read_store.quiet_state_episode_v0(observation_id)
    assert projection is not None
    persisted = projection["risk_observations"]
    assert len(persisted) == 2
    assert persisted[-1]["payload"]["net_option_pnl"] == pytest.approx(20.0)
    assert persisted[-1]["payload"]["ibkr_initial_margin"] == MARGIN_UNAVAILABLE
    assert persisted[-1]["payload"]["can_authorize_trade"] is False
    assert persisted[-1]["policy_gate"] == 0


def test_bounded_recorder_emits_record_only_paths_and_three_strategy_comparison() -> None:
    class RecordingRepository:
        def __init__(self) -> None:
            self.risk_records: list[dict[str, Any]] = []
            self.comparisons: list[dict[str, Any]] = []

        def record_quiet_shadow_structure(self, *_args: object, **_kwargs: object) -> None:
            return None

        def record_quiet_option_risk_observation(
            self,
            *_args: object,
            **kwargs: Any,
        ) -> int:
            self.risk_records.append(kwargs)
            return len(self.risk_records)

        def record_quiet_option_strategy_comparison(
            self,
            *_args: object,
            **kwargs: Any,
        ) -> int:
            self.comparisons.append(kwargs)
            return len(self.comparisons)

    expiry = ENTRY.date() + timedelta(days=1)
    contracts = tuple(
        OptionContract(
            underlying_con_id=1,
            con_id=index,
            expiry=expiry,
            dte=1,
            dte_bucket=DteBucket.ONE_DTE,
            strike=strike,
            right=right,
            multiplier=100,
            exchange="SMART",
            trading_class="TEST",
        )
        for index, (strike, right) in enumerate(
            ((99.0, "C"), (99.0, "P"), (100.0, "C"), (100.0, "P"), (101.0, "C"), (101.0, "P")),
            start=10,
        )
    )
    by_key = {(contract.strike, contract.right): contract for contract in contracts}
    entry_quotes = tuple(
        _option_event(
            contract,
            observed_at=ENTRY + timedelta(seconds=1),
            bid=(2.00 if contract.strike == 100.0 else 0.50),
            ask=(2.20 if contract.strike == 100.0 else 0.60),
            delta=(-0.50 if contract.right == "P" else 0.50),
            sequence=index,
        )
        for index, contract in enumerate(contracts, start=100)
    )
    exit_quotes = tuple(
        _option_event(
            contract,
            observed_at=ENTRY + timedelta(minutes=5),
            bid=(1.10 if contract.strike == 100.0 else 0.30),
            ask=(1.20 if contract.strike == 100.0 else 0.40),
            delta=(-0.40 if contract.right == "P" else 0.40),
            sequence=index,
        )
        for index, contract in enumerate(contracts, start=200)
    )
    underlying_quotes = (
        _underlying_event(
            observed_at=ENTRY + timedelta(seconds=1),
            bid=99.90,
            ask=100.10,
            sequence=1,
        ),
        _underlying_event(
            observed_at=ENTRY + timedelta(minutes=5),
            bid=101.00,
            ask=101.20,
            sequence=2,
        ),
    )
    repository = RecordingRepository()
    recorder = BoundedOptionRecorder(
        adapter=cast(Any, object()),
        subscriptions=cast(Any, object()),
        repository=cast(Any, repository),
        raw_store=cast(Any, object()),
        maximum_quote_age=timedelta(seconds=5),
        underlying_path_provider=lambda _symbol, _start, _end: (100.0, 101.0),
        underlying_quote_provider=lambda _symbol, _start, _end: underlying_quotes,
    )

    recorder._record_quiet_bucket_outcomes(
        cast(Any, object()),
        observation_id="quiet-accounting",
        symbol="TEST",
        entry_timestamp=ENTRY,
        bucket=DteBucket.ONE_DTE,
        planned_bucket_contracts=contracts,
        bucket_contracts=contracts,
        atm_call=by_key[(100.0, "C")],
        atm_put=by_key[(100.0, "P")],
        quotes=(*entry_quotes, *exit_quotes),
        outcomes=(),
        horizon_minutes=(5,),
        subscription_gap_spans_horizon=False,
        plan_capacity_reduced=False,
    )

    strategy_types = {str(row["record"].strategy_type) for row in repository.risk_records}
    assert strategy_types == {
        "UNDERLYING_LONG",
        "LONG_OPTION",
        "SHORT_PUT",
        "BULL_PUT_SPREAD",
    }
    assert len(repository.risk_records) == 10
    assert len(repository.comparisons) == 1
    report = repository.comparisons[0]["report"]
    assert report.can_authorize_trade is False
    assert {row.strategy_type for row in report.strategies} == {
        "UNDERLYING_LONG",
        "SHORT_PUT",
        "BULL_PUT_SPREAD",
    }


def test_bounded_recorder_waits_for_the_first_executable_entry_side() -> None:
    contract = OptionContract(
        underlying_con_id=1,
        con_id=77,
        expiry=ENTRY.date() + timedelta(days=1),
        dte=1,
        dte_bucket=DteBucket.ONE_DTE,
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="TEST",
    )
    bid_only = _option_event(
        contract,
        observed_at=ENTRY + timedelta(seconds=1),
        bid=1.00,
        ask=1.20,
        delta=0.50,
        sequence=1,
    ).model_copy(update={"ask": None, "last": None})
    executable = _option_event(
        contract,
        observed_at=ENTRY + timedelta(seconds=2),
        bid=1.00,
        ask=1.20,
        delta=0.50,
        sequence=2,
    )
    strategy = OptionStrategy(
        strategy_id="long-call",
        strategy_type=StrategyType.LONG_OPTION,
        legs=(
            OptionLeg(
                leg_id="conid:77",
                right="C",
                strike=100.0,
                multiplier=100,
                signed_contract_quantity=1,
            ),
        ),
    )
    recorder = BoundedOptionRecorder(
        adapter=cast(Any, object()),
        subscriptions=cast(Any, object()),
        repository=cast(Any, object()),
        raw_store=cast(Any, object()),
        maximum_quote_age=timedelta(seconds=5),
    )

    snapshots = recorder._accounting_snapshots(
        strategy=strategy,
        quotes=(bid_only, executable),
        entry_timestamp=ENTRY,
        target_timestamp=ENTRY + timedelta(seconds=5),
        subscription_gap_spans_horizon=False,
    )

    assert snapshots[0].observed_at == ENTRY + timedelta(seconds=2)
    assert snapshots[0].legs[0].ask == pytest.approx(1.20)


def test_option_accounting_has_no_dependency_on_opening_leader_selection() -> None:
    package = Path(__file__).parents[1] / "packages/stocker_prospective/src/stocker_prospective"
    accounting_tree = ast.parse((package / "option_risk_accounting.py").read_text(encoding="utf-8"))
    leader_tree = ast.parse(
        (package / "opening_leader_continuation_v0.py").read_text(encoding="utf-8")
    )

    def imported_modules(tree: ast.AST) -> set[str]:
        modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        return modules

    assert not any("opening_leader" in name for name in imported_modules(accounting_tree))
    assert not any("option_risk_accounting" in name for name in imported_modules(leader_tree))


def test_option_accounting_has_no_reachable_order_or_account_path() -> None:
    package = Path(__file__).parents[1] / "packages/stocker_prospective/src/stocker_prospective"
    forbidden_calls = {
        "placeOrder",
        "place_order",
        "cancelOrder",
        "cancel_order",
        "exerciseOptions",
        "exercise_options",
        "reqAccountSummary",
        "reqAccountUpdates",
        "reqPositions",
        "reqExecutions",
        "reqPnL",
    }
    for filename in (
        "option_risk_accounting.py",
        "option_recorder.py",
        "recorder_repository.py",
    ):
        tree = ast.parse((package / filename).read_text(encoding="utf-8"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert forbidden_calls.isdisjoint(called_attributes | called_names), filename

    field_names = set(OptionRiskAccountingRecord.model_fields)
    assert not {"account_balance", "net_liquidation", "buying_power"}.intersection(field_names)
