"""Preserved provider-independent frozen mathematics and durable-obligation regressions."""

import json
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

import pytest

from saxo_support import AT, fill_record, frozen_strike, record_entry
from stocker_execution.config import MARKETS
from stocker_execution.contracts import grid_price
from stocker_execution.rules import (
    Bar,
    clocks,
    eligibility,
    model_delta,
    opportunity,
    prior_rv,
)
from stocker_execution.store import Store


def test_source_fixture_rv_clocks_and_original_exit_anchor():
    fixture = json.loads(Path("tests/fixtures/futures/frozen.json").read_text())
    for row in fixture["rows"]:
        if row["market"] not in MARKETS:
            continue  # Historical fixture remains immutable; active universe is five markets.
        at = datetime.fromisoformat(row["at"]).astimezone(UTC)
        bars = [
            Bar(at - timedelta(minutes=31 - i), p, p, p, p, 1, p)
            for i, p in enumerate(row["pre31"])
        ]
        assert prior_rv(bars, at) == pytest.approx(row["rv15"], rel=1e-10, abs=1e-14)
        event = opportunity(row["market"], row["con_id"], at)
        assert datetime.fromisoformat(event["exit_at"]) == at + timedelta(minutes=60)
        expiry = at.replace(hour=21)
        strike = frozen_strike(
            row["price"], row["rv15"], at, expiry, event["right"], event["target_delta"]
        )
        assert model_delta(
            row["price"], strike, row["rv15"], at, expiry, event["right"]
        ) == pytest.approx(event["target_delta"], abs=1e-10)
        # An incomplete current bar cannot contaminate the completed prefix.
        assert prior_rv(bars + [Bar(at, 999, 999, 999, 999, 1, 999)], at) == prior_rv(bars, at)
        with pytest.raises(ValueError, match="INCOMPLETE"):
            prior_rv(bars[:-1], at)


def test_dst_weekends_and_only_approved_veto():
    assert clocks(date(2026, 3, 9))[0].hour == 13  # US DST precedes UK
    assert clocks(date(2026, 3, 2))[0].hour == 14
    assert clocks(date(2026, 10, 26))[0].hour == 13  # UK reverts before US
    assert clocks(date(2026, 11, 2))[0].hour == 14
    assert clocks(date(2026, 9, 27)) == []
    for market in MARKETS:
        for at in clocks(date(2026, 9, 28)):
            event = opportunity(market, 1, at)
            assert bool(event["veto"]) == (market == "NG" and at.hour == 17)
    assert opportunity("GC", 1, AT + timedelta(hours=7))["veto"] == ""


def test_cancel_uncertainty_and_late_fill_keep_obligation(tmp_path):
    s = Store(tmp_path / "futures.sqlite")
    ref = record_entry(s)
    assert not s.confirm_closed("x", 0)
    with s.db:
        s.db.execute("UPDATE orders SET status='Cancelled' WHERE reference=?", (ref,))
    assert s.confirm_closed("x", 0)
    assert s.capacity()["reserved_open_trades"] == 0
    s.record_fill(fill_record(ref))
    assert s.capacity()["reserved_open_trades"] == 1
    assert not s.confirm_closed("x", 0)
    with s.db:
        s.db.execute("UPDATE orders SET filled=1 WHERE reference=?", (ref,))
    assert not s.confirm_closed("x", 1)


def test_missing_commission_is_provisional_and_exit_submission_does_not_release(tmp_path):
    s = Store(tmp_path / "futures.sqlite")
    ref = record_entry(s)
    s.record_fill(fill_record(ref))
    with s.db:
        s.db.execute(
            "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?", (ref,)
        )
    exit_ref = s.prepare_order("x", "EXIT", 900, AT.isoformat(), {"con_id": 100})
    assert not s.confirm_closed("x", 1)
    s.record_fill({**fill_record(exit_ref, "exit.1", "SLD"), "price": 0.15})
    with s.db:
        s.db.execute(
            "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?", (exit_ref,)
        )
    assert s.confirm_closed("x", 0)
    assert s.economics()["realised_net_gbp"] is None
    with s.db:
        s.db.execute("UPDATE fills SET commission=1,commission_currency='USD'")
    assert s.economics()["realised_net_gbp"] == pytest.approx(2.4)
    assert s.economics()["closed_with_complete_costs"] == 1
    # A terminal unfilled exit attempt has no commission to await.
    with s.db:
        s.db.execute(
            "INSERT INTO orders"
            "(reference,event_id,role,order_id,status,filled,remaining,deadline,payload) "
            "VALUES('cancelled-exit','x','EXIT',901,'Cancelled',0,0,?,'{}')",
            (AT.isoformat(),),
        )
    assert s.economics()["realised_net_gbp"] == pytest.approx(2.4)
    s.record_fill(fill_record(ref))  # replayed evidence is idempotent
    with pytest.raises(ValueError, match="EXECUTION_ID_REUSED"):
        s.record_fill({**fill_record(ref), "price": 0.11})
    assert s.exposure("x") == 0
    assert len(s.fills("x")) == 2


def test_inherited_feature_gate_rejects_flat_last_five_returns():
    prices = [100 + i * 0.01 for i in range(60)]
    references = [{9: {"rv15": 0.01, "range15": 0.01, "volume15": 150}} for _ in range(5)]

    def bars(values):
        return [
            Bar(AT - timedelta(minutes=60 - i), p, p + 0.01, p - 0.01, p, 10, p)
            for i, p in enumerate(values)
        ]

    assert eligibility(bars(prices), AT, date(2026, 12, 1), references)["rv15"] > 0
    prices[-6:] = [prices[-6]] * 6
    assert prior_rv(bars(prices), AT) > 0
    with pytest.raises(ValueError, match="FEATURE_AVAILABILITY"):
        eligibility(bars(prices), AT, date(2026, 12, 1), references)


@pytest.mark.parametrize(
    ("price", "tick"),
    [(0.3, 0.1), (0.7, 0.1), (0.07, 0.01), (0.145, 0.005), (0.043, 0.001), (2.24, 0.01)],
)
def test_entry_limit_and_exit_price_stay_exactly_one_tick_from_the_quote(price, tick):
    """Float division (0.3 / 0.1 = 2.999…) used to land paper fills two ticks away."""
    entry = grid_price(price, tick, ROUND_CEILING, 1)
    exit_price = grid_price(price, tick, ROUND_FLOOR, -1)
    assert Decimal(str(entry)) == Decimal(str(price)) + Decimal(str(tick))
    assert Decimal(str(exit_price)) == Decimal(str(price)) - Decimal(str(tick))
    with pytest.raises(ValueError, match="INVALID_TICK_SIZE"):
        grid_price(price, 0, ROUND_CEILING, 1)
