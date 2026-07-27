from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_prospective.events import (
    DepthOperation,
    DepthSide,
    UnderlyingDepthEvent,
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickTradeEvent,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.microstructure import (
    ProbableTradeSide,
    classify_probable_trade_side,
    compare_frozen_archetypes,
    quote_primitives,
    summarise_microstructure_window,
)
from stocker_prospective.order_book import DepthBook

START = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)


def quote(
    second: int,
    *,
    bid: float = 100.0,
    ask: float = 100.2,
    bid_size: float = 100.0,
    ask_size: float = 80.0,
) -> UnderlyingLevel1QuoteEvent:
    timestamp = START + timedelta(seconds=second)
    return UnderlyingLevel1QuoteEvent(
        event_id=f"q-{second}",
        received_timestamp_utc=timestamp,
        received_monotonic_ns=second,
        provider_timestamp_utc=timestamp,
        source_sequence=second,
        session=date(2026, 7, 24),
        symbol="AAL",
        con_id=265598,
        request_id=10,
        bid=bid,
        bid_size=bid_size,
        ask=ask,
        ask_size=ask_size,
        last=100.1,
        last_size=10.0,
        market_data_type=MarketDataType.LIVE,
        source="ibkr_level1",
        quote_valid=True,
        staleness_ms=0.0,
        tick_type="state_change",
        exchange="SMART",
    )


def trade(second: int, price: float, size: float = 20.0) -> UnderlyingTickTradeEvent:
    timestamp = START + timedelta(seconds=second)
    return UnderlyingTickTradeEvent(
        event_id=f"t-{second}",
        received_timestamp_utc=timestamp,
        received_monotonic_ns=100 + second,
        provider_timestamp_utc=timestamp,
        source_sequence=100 + second,
        session=date(2026, 7, 24),
        symbol="AAL",
        con_id=265598,
        request_id=20,
        price=price,
        size=size,
        exchange="NYSE",
        conditions=(),
        market_data_type=MarketDataType.LIVE,
    )


def depth_event(
    sequence: int,
    *,
    operation: DepthOperation,
    side: DepthSide,
    position: int,
    price: float | None,
    size: float | None,
) -> UnderlyingDepthEvent:
    timestamp = START + timedelta(milliseconds=sequence)
    return UnderlyingDepthEvent(
        event_id=f"d-{sequence}",
        received_timestamp_utc=timestamp,
        received_monotonic_ns=sequence,
        provider_timestamp_utc=None,
        source_sequence=sequence,
        session=date(2026, 7, 24),
        symbol="AAL",
        con_id=265598,
        request_id=30,
        operation=operation,
        position=position,
        side=side,
        price=price,
        size=size,
        market_maker_or_exchange="NYSE",
        smart_depth=True,
    )


def test_quote_primitives_are_exact_and_bounded() -> None:
    values = quote_primitives(quote(0))

    assert values.midpoint == pytest.approx(100.1)
    assert values.spread == pytest.approx(0.2)
    assert values.quote_size_imbalance == pytest.approx(20 / 180)
    assert values.microprice == pytest.approx((100.2 * 100 + 100.0 * 80) / 180)
    assert -1.0 <= values.quote_size_imbalance <= 1.0
    assert -1.0 <= values.microprice_edge_half_spread_fraction <= 1.0


def test_probable_trade_side_uses_prevailing_quote_and_never_invents_unknown_side() -> None:
    prevailing = quote(0)

    buy = classify_probable_trade_side(
        trade(1, 100.2),
        prevailing,
        maximum_quote_age=timedelta(seconds=2),
    )
    sell = classify_probable_trade_side(
        trade(1, 100.0),
        prevailing,
        maximum_quote_age=timedelta(seconds=2),
    )
    inside = classify_probable_trade_side(
        trade(1, 100.1),
        prevailing,
        maximum_quote_age=timedelta(seconds=2),
    )
    stale = classify_probable_trade_side(
        trade(5, 100.2),
        prevailing,
        maximum_quote_age=timedelta(seconds=2),
    )

    assert buy.side is ProbableTradeSide.BUY
    assert sell.side is ProbableTradeSide.SELL
    assert inside.side is ProbableTradeSide.UNKNOWN
    assert stale.side is ProbableTradeSide.UNCLASSIFIED
    assert stale.quote_age_ms == 5000.0
    assert all(item.classification_method == "prevailing_quote_v1" for item in (buy, sell))


def test_window_summary_separates_quote_flow_trade_flow_and_replenishment() -> None:
    quotes = (
        quote(0),
        quote(1, bid_size=60.0),
        quote(2, bid_size=110.0),
        quote(3, bid=100.1, ask=100.2, bid_size=120.0, ask_size=50.0),
    )
    trades = (trade(1, 100.0, 40.0), trade(3, 100.2, 25.0), trade(4, 100.15, 10.0))

    summary = summarise_microstructure_window(
        symbol="AAL",
        window_start=START,
        window_end=START + timedelta(seconds=5),
        quotes=quotes,
        trades=trades,
        maximum_quote_age=timedelta(seconds=2),
        minimum_classification_valid_fraction=0.5,
    )

    assert summary.quote_flow.bid_size_removals == 1
    assert summary.quote_flow.bid_size_additions == 1
    assert summary.quote_flow.bid_price_improvements == 1
    assert summary.trade_flow.probable_buy_volume == 25.0
    assert summary.trade_flow.probable_sell_volume == 40.0
    assert summary.trade_flow.unknown_volume == 10.0
    assert summary.trade_flow.trade_features_valid is True
    assert summary.replenishment["bid_1s"].size_restored >= 0.0
    assert set(summary.scores) == {"MC", "MD", "MA", "MB"}
    assert all(
        score.label == "microstructure descriptive score" for score in summary.scores.values()
    )
    relationships = compare_frozen_archetypes(
        actions={"A1": "CALL", "C1": "PUT", "R1": "ABSTAIN"},
        summary=summary,
    )
    assert set(relationships.values()).issubset(
        {
            "same_direction",
            "opposite_direction",
            "microstructure_neutral",
            "insufficient_data",
        }
    )


def test_same_provider_timestamp_never_uses_a_quote_received_after_the_trade() -> None:
    early = quote(0).model_copy(
        update={
            "event_id": "early",
            "received_monotonic_ns": 10,
            "source_sequence": 10,
        }
    )
    observed_trade = trade(0, 100.2).model_copy(
        update={
            "event_id": "tied-trade",
            "received_monotonic_ns": 20,
            "source_sequence": 20,
        }
    )
    later = quote(0, bid=100.2, ask=100.4).model_copy(
        update={
            "event_id": "later",
            "received_timestamp_utc": START + timedelta(milliseconds=1),
            "received_monotonic_ns": 30,
            "source_sequence": 30,
        }
    )

    summary = summarise_microstructure_window(
        symbol="AAL",
        window_start=START - timedelta(milliseconds=1),
        window_end=START + timedelta(seconds=1),
        quotes=(early, later),
        trades=(observed_trade,),
        maximum_quote_age=timedelta(seconds=2),
        minimum_classification_valid_fraction=0.5,
    )

    classification = summary.trade_classifications[0]
    assert classification.trade_event_id == "tied-trade"
    assert classification.side is ProbableTradeSide.BUY
    assert classification.prevailing_bid == 100.0
    assert classification.prevailing_ask == 100.2
    assert classification.quote_age_ms == 0.0


def test_depth_book_reconstructs_rows_and_fails_closed_across_reset() -> None:
    book = DepthBook(symbol="AAL", con_id=265598, rows_per_side=2)
    book.apply(
        depth_event(
            1,
            operation=DepthOperation.INSERT,
            side=DepthSide.BID,
            position=0,
            price=100.0,
            size=50.0,
        )
    )
    book.apply(
        depth_event(
            2,
            operation=DepthOperation.INSERT,
            side=DepthSide.ASK,
            position=0,
            price=100.2,
            size=60.0,
        )
    )
    with pytest.raises(ValueError, match="configured rows"):
        book.mark_complete()
    book.apply(
        depth_event(
            3,
            operation=DepthOperation.INSERT,
            side=DepthSide.BID,
            position=1,
            price=99.9,
            size=40.0,
        )
    )
    book.apply(
        depth_event(
            4,
            operation=DepthOperation.INSERT,
            side=DepthSide.ASK,
            position=1,
            price=100.3,
            size=30.0,
        )
    )
    book.mark_complete()
    snapshot = book.snapshot(
        START + timedelta(seconds=1),
        advance_centroid_baseline=True,
    )

    assert snapshot.book_valid is True
    assert snapshot.total_bid_size == 90.0
    assert snapshot.total_ask_size == 90.0
    book.apply(
        depth_event(
            5,
            operation=DepthOperation.UPDATE,
            side=DepthSide.BID,
            position=0,
            price=100.0,
            size=75.0,
        )
    )
    replenished = book.snapshot(START + timedelta(seconds=1))
    repeated_probe = book.snapshot(START + timedelta(seconds=1))
    assert replenished.bid_depth_additions == 25.0
    assert replenished.book_slope is not None
    assert repeated_probe.depth_centroid_shift == replenished.depth_centroid_shift

    gap = book.reset(START + timedelta(seconds=2), reason="ibkr_depth_reset")
    invalid = book.snapshot(START + timedelta(seconds=2))

    assert gap.reason == "ibkr_depth_reset"
    assert invalid.book_valid is False
    assert invalid.bid_rows == ()
    assert invalid.ask_rows == ()
    assert invalid.reset_count == 1
