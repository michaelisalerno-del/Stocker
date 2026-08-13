from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.microstructure import (
    DescriptiveScore,
    MicrostructureWindowSummary,
    PriceImpactSummary,
    QuoteFlowSummary,
    ReplenishmentSummary,
    TradeFlowSummary,
)
from stocker_prospective.microstructure_direction_v0 import (
    DirectionActionV0,
    DirectionMethodV0,
    build_microstructure_direction_v0,
    microstructure_direction_windows_v0,
    select_first_causal_entry_v0,
)

T0 = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)


def _summary(
    *,
    trade_imbalance: float | None = 0.4,
    quote_imbalance: float | None = 0.2,
    microprice_edge: float | None = 0.1,
    trade_valid: bool = True,
    causal_as_of: datetime = T0,
) -> MicrostructureWindowSummary:
    quote_flow = QuoteFlowSummary(
        bid_price_improvements=1,
        bid_price_deteriorations=0,
        ask_price_improvements=0,
        ask_price_deteriorations=0,
        bid_size_additions=1,
        bid_size_removals=0,
        ask_size_additions=0,
        ask_size_removals=1,
        quote_update_count=4,
        quote_update_rate=4.0,
        price_moving_updates=1,
        size_only_updates=3,
        locked_market_count=0,
        crossed_market_count=0,
        invalid_quote_count=0,
        spread_tightening_count=0,
        spread_widening_count=0,
        time_weighted_quote_size_imbalance=quote_imbalance,
        time_weighted_microprice_edge=microprice_edge,
        midpoint_change=0.02,
        best_bid_change=0.01,
        best_ask_change=0.01,
        bid_displayed_size_removal_proxy=0.0,
        ask_displayed_size_removal_proxy=0.2,
    )
    trade_flow = TradeFlowSummary(
        probable_buy_volume=70.0 if trade_imbalance is not None else 0.0,
        probable_sell_volume=30.0 if trade_imbalance is not None else 0.0,
        unknown_volume=10.0,
        probable_buy_trade_count=2 if trade_imbalance is not None else 0,
        probable_sell_trade_count=1 if trade_imbalance is not None else 0,
        unknown_trade_count=1,
        trade_imbalance=trade_imbalance,
        unknown_volume_fraction=10.0 / 110.0,
        classification_valid_fraction=0.75 if trade_imbalance is not None else 0.0,
        mean_trade_size=27.5,
        median_trade_size=25.0,
        trade_arrival_rate=4.0,
        buy_arrival_rate=2.0,
        sell_arrival_rate=1.0,
        trade_features_valid=trade_valid,
    )
    impact = PriceImpactSummary(
        side="probable_buyer_initiated",
        raw_signed_price_impact=0.01,
        price_impact_per_share=0.001,
        price_impact_per_notional=0.00001,
        price_impact_bps=1.0,
        flow_volume=70.0,
        event_count=2,
        window_duration_seconds=1.0,
        support_valid=True,
    )
    replenishment = {
        f"{side}_{seconds}s": ReplenishmentSummary(
            side=side,
            response_interval_seconds=seconds,
            trigger_count=0,
            size_depleted=0.0,
            size_restored=0.0,
            replenishment_ratio=None,
            price_survival_time_seconds=None,
            midpoint_response=None,
        )
        for side in ("bid", "ask")
        for seconds in (1, 3, 5)
    }
    mc_components = {
        "trade_imbalance": trade_imbalance,
        "quote_size_imbalance": quote_imbalance,
        "microprice_edge": microprice_edge,
    }
    md_components = {
        name: None if value is None else -value for name, value in mc_components.items()
    }
    valid_values = [value for value in mc_components.values() if value is not None]
    mc = None if not valid_values else sum(valid_values) / len(valid_values)
    md = None if mc is None else -mc
    return MicrostructureWindowSummary(
        symbol="AAL",
        window_start=T0 - timedelta(seconds=1),
        window_end=T0,
        duration_seconds=1.0,
        quote_flow=quote_flow,
        trade_flow=trade_flow,
        trade_classifications=(),
        buy_impact=impact,
        sell_impact=impact.model_copy(update={"side": "probable_seller_initiated"}),
        replenishment=replenishment,
        scores={
            "MC": DescriptiveScore(
                score_id="MC",
                components=mc_components,
                composite=mc,
                valid_component_count=len(valid_values),
            ),
            "MD": DescriptiveScore(
                score_id="MD",
                components=md_components,
                composite=md,
                valid_component_count=len(valid_values),
            ),
        },
        causal_as_of=causal_as_of,
    )


def _results(
    summary: MicrostructureWindowSummary,
    *,
    tick_last_present: bool = True,
) -> dict[DirectionMethodV0, object]:
    rows = build_microstructure_direction_v0(
        episode_id="episode-1",
        run_id="run-1",
        trigger_timestamp_utc=T0,
        window_name="T0_to_T+1s",
        summary=summary,
        tick_bidask_present=True,
        tick_last_present=tick_last_present,
        depth_present=False,
        depth_valid=False,
        market_data_type=MarketDataType.LIVE.value,
        data_quality_flags=(),
    )
    return {row.direction_method: row for row in rows}


def test_predeclared_sign_methods_and_consensus_are_exact() -> None:
    rows = _results(_summary())

    assert rows[DirectionMethodV0.M01].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M02].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M03].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M04].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M05].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M06].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M01].signed_score == 0.4
    assert rows[DirectionMethodV0.M04].signed_score == 2 * (0.4 + 0.2 + 0.1) / 3


def test_majority_resolves_disagreement_while_strict_consensus_abstains() -> None:
    rows = _results(_summary(trade_imbalance=-0.4))

    assert rows[DirectionMethodV0.M01].action is DirectionActionV0.DOWN
    assert rows[DirectionMethodV0.M05].action is DirectionActionV0.ABSTAIN
    assert rows[DirectionMethodV0.M06].action is DirectionActionV0.UP


def test_missing_or_invalid_trade_evidence_abstains_without_imputation() -> None:
    rows = _results(_summary(trade_imbalance=None, trade_valid=False), tick_last_present=False)

    assert rows[DirectionMethodV0.M01].action is DirectionActionV0.ABSTAIN
    assert rows[DirectionMethodV0.M02].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M03].action is DirectionActionV0.UP
    assert rows[DirectionMethodV0.M05].action is DirectionActionV0.ABSTAIN
    assert rows[DirectionMethodV0.M06].action is DirectionActionV0.ABSTAIN
    assert rows[DirectionMethodV0.M01].component_validity["trade_imbalance"] is False


def test_all_missing_core_evidence_abstains_without_calculating_a_score() -> None:
    rows = _results(
        _summary(
            trade_imbalance=None,
            quote_imbalance=None,
            microprice_edge=None,
            trade_valid=False,
        ),
        tick_last_present=False,
    )

    for method in DirectionMethodV0:
        assert rows[method].action is DirectionActionV0.ABSTAIN
        assert rows[method].signed_score is None


def test_future_information_fails_closed_for_every_method() -> None:
    rows = _results(_summary(causal_as_of=T0 + timedelta(microseconds=1)))

    assert all(row.action is DirectionActionV0.ABSTAIN for row in rows.values())
    assert all(row.causal_valid is False for row in rows.values())
    assert all("information_after_decision" in row.data_quality_flags for row in rows.values())


def test_direction_windows_are_predeclared_and_decide_at_the_window_end() -> None:
    windows = microstructure_direction_windows_v0(T0)

    assert tuple(windows) == (
        "T-60s_to_T0",
        "T-30s_to_T0",
        "T-15s_to_T0",
        "T-5s_to_T0",
        "T0_to_T+1s",
        "T0_to_T+5s",
        "T0_to_T+15s",
        "T0_to_T+30s",
        "T0_to_T+60s",
    )
    assert windows["T-60s_to_T0"] == (T0 - timedelta(seconds=60), T0)
    assert windows["T0_to_T+15s"] == (T0, T0 + timedelta(seconds=15))


def _quote(event_id: str, timestamp: datetime) -> UnderlyingLevel1QuoteEvent:
    return UnderlyingLevel1QuoteEvent(
        event_id=event_id,
        received_timestamp_utc=timestamp,
        received_monotonic_ns=int(timestamp.timestamp() * 1_000_000_000),
        provider_timestamp_utc=timestamp,
        source_sequence=int(timestamp.timestamp() * 1000),
        session=date(2026, 8, 6),
        symbol="AAL",
        con_id=265598,
        request_id=10,
        bid=100.0,
        bid_size=100.0,
        ask=100.2,
        ask_size=100.0,
        last=100.1,
        last_size=10.0,
        market_data_type=MarketDataType.LIVE,
        source="official_ibkr_tick_by_tick_bidask",
        quote_valid=True,
        staleness_ms=0.0,
        tick_type="BidAsk",
        exchange="SMART",
    )


def test_causal_entry_is_first_valid_quote_strictly_after_information_cutoff() -> None:
    at_cutoff = _quote("at", T0)
    after = _quote("after", T0 + timedelta(microseconds=1))

    entry = select_first_causal_entry_v0(
        information_cutoff_utc=T0,
        quotes=(after, at_cutoff),
    )

    assert entry is not None
    assert entry.event_id == "after"
    assert entry.timestamp_utc == T0 + timedelta(microseconds=1)
    assert entry.delay_seconds == 0.000001
    assert entry.midpoint == 100.1


def test_repeated_direction_builds_are_identical_and_permanently_research_labelled() -> None:
    first = build_microstructure_direction_v0(
        episode_id="episode-1",
        run_id="run-1",
        trigger_timestamp_utc=T0,
        window_name="T-5s_to_T0",
        summary=_summary(),
        tick_bidask_present=True,
        tick_last_present=True,
        depth_present=False,
        depth_valid=False,
        market_data_type=MarketDataType.LIVE.value,
        data_quality_flags=(),
    )
    second = build_microstructure_direction_v0(
        episode_id="episode-1",
        run_id="run-1",
        trigger_timestamp_utc=T0,
        window_name="T-5s_to_T0",
        summary=_summary(),
        tick_bidask_present=True,
        tick_last_present=True,
        depth_present=False,
        depth_valid=False,
        market_data_type=MarketDataType.LIVE.value,
        data_quality_flags=(),
    )

    assert first == second
    assert all(
        row.research_label
        == "RESEARCH ONLY — MICROSTRUCTURE DIRECTION V0 — NOT VALIDATED — NO RECOMMENDATION"
        for row in first
    )
