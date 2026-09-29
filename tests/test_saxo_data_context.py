"""Additional Saxo data: chart v3, price context, daily range, chain smile, events, P&L."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

from saxo_support import FUTURE, record_entry, setup
from stocker_execution import views
from stocker_execution.saxo_balance import ensure_activity_subscription
from stocker_execution.saxo_client import allowed
from stocker_execution.saxo_history import daily_context, history, history_range
from stocker_execution.saxo_stream import PriceState


def test_bars_use_chart_v3_and_v1_is_no_longer_allowed(tmp_path):
    calls = []

    async def scenario():
        _, data, store = setup(tmp_path)

        async def request(method, path, **kwargs):
            calls.append((method, path, kwargs.get("params")))
            return {"Data": [], "DataVersion": 1}

        data.client.request = request
        await history(data, data.markets["CL"])
        await history_range(
            data, 100, datetime(2026, 9, 28, 12, tzinfo=UTC), datetime(2026, 9, 28, 13, tzinfo=UTC)
        )
        store.db.close()

    asyncio.run(scenario())
    assert {path for _, path, _ in calls} == {"/chart/v3/charts"}
    assert allowed("GET", "/chart/v3/charts")
    assert not allowed("GET", "/chart/v1/charts")


def test_price_context_reports_only_numeric_provider_fields():
    price = PriceState()
    price.snapshot(
        {
            "Quote": {"Bid": 70, "Ask": 70.01, "MarketState": "Open"},
            "PriceInfo": {"High": 71.2, "Low": 69.5, "NetChange": 0.4, "PercentChange": 0.57},
            "PriceInfoDetails": {"Open": 69.9, "LastClose": 69.6, "Volume": 1000},
            "InstrumentPriceDetails": {"OpenInterest": 250000, "IsMarketOpen": True},
        },
        "fixture",
        time.time(),
    )
    state = type("State", (), {"price": price, "daily_range": 1.8, "daily_problem": ""})()
    context = views.price_context(state)
    assert context["high"] == 71.2 and context["low"] == 69.5 and context["open"] == 69.9
    assert context["last_close"] == 69.6 and context["open_interest"] == 250000
    assert context["market_state"] == "Open" and context["daily_range"] == 1.8
    price.value["PriceInfo"]["High"] = "71.2"
    assert views.price_context(state)["high"] is None


def test_daily_range_averages_completed_sessions_once_per_day(tmp_path):
    calls = []

    async def scenario():
        _, data, store = setup(tmp_path)
        state = data.markets["CL"]

        async def request(method, path, params=None, **kwargs):
            calls.append(params)
            start = datetime(2026, 8, 1, tzinfo=UTC)
            return {
                "Data": [
                    {"Time": (start + timedelta(days=i)).isoformat(), "High": 71 + i % 3, "Low": 70}
                    for i in range(21)
                ]
            }

        data.client.request = request
        await daily_context(data, state)
        await daily_context(data, state)
        assert calls[0]["Horizon"] == 1440 and len(calls) == 1
        # The newest (possibly incomplete) sample is excluded from the 20 averaged.
        assert state.daily_range == sum(1 + i % 3 for i in range(20)) / 20
        state.daily_day = ""

        async def short(method, path, params=None, **kwargs):
            return {"Data": [{"Time": "2026-09-01T00:00:00Z", "High": 1, "Low": 0}] * 3}

        data.client.request = short
        await daily_context(data, state)
        assert state.daily_range is None and state.daily_problem == "DAILY_SAMPLES_INSUFFICIENT"
        store.db.close()

    asyncio.run(scenario())


BOARD = {
    "Expiries": [
        {"Index": 0, "Expiry": "2026-09-29T00:00:00Z", "Strikes": []},
        {
            "Index": 1,
            "Expiry": "2026-09-30T00:00:00Z",
            "LastTradeDate": "2026-09-30T18:30:00Z",
            "MidStrikePrice": 70.2,
            "Strikes": [
                {
                    "Index": 2,
                    "Strike": 71,
                    "MidVolatilityPct": 0.31,
                    "Put": {
                        "Uic": 711,
                        "Bid": 0.5,
                        "Ask": 0.6,
                        "OpenInterest": 40,
                        "Greeks": {"Delta": -0.62, "MidVolatility": 0.30},
                    },
                },
                {
                    "Index": 1,
                    "Strike": 69,
                    "MidVolatilityPct": 0.36,
                    "Put": {
                        "Uic": 691,
                        "Bid": 0.1,
                        "Ask": 0.14,
                        "OpenInterest": 900,
                        "Volume": 25,
                        "Greeks": {"Delta": -0.11, "MidVolatility": 0.35},
                    },
                    "Call": "unexpected",
                },
            ],
        },
    ]
}


def test_chain_smile_picks_todays_expiry_sorted_in_provider_units():
    result = views.smile(BOARD, "2026-09-30")
    assert result["expiry"].startswith("2026-09-30") and result["mid_strike_price"] == 70.2
    assert [s["strike"] for s in result["strikes"]] == [69, 71]
    low = result["strikes"][0]
    assert low["put"] == {
        "uic": 691,
        "bid": 0.1,
        "ask": 0.14,
        "delta": -0.11,
        "mid_volatility": 0.35,
        "iv": None,
        "iv_minus_model": None,
        "open_interest": 900,
        "volume": 25,
    }
    assert low["call"] is None and low["mid_volatility_pct"] == 0.36
    assert result["scaling"] == "PROVIDER_NATIVE_UNVERIFIED" and result["executable"] is False
    assert views.smile(BOARD, "2026-12-01")["expiry"].startswith("2026-09-29")
    assert views.smile({}, "2026-09-30") is None
    assert abs(views.model_sigma(0.001) - 0.001 * (525600 / 15) ** 0.5) < 1e-12


def test_sim_activity_events_wake_reconciliation_and_never_reset_the_stream(tmp_path):
    posted = []

    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        data.subscriptions.clear()

        async def request(method, path, body=None, **kwargs):
            posted.append((method, path, body))
            return {"InactivityTimeout": 30}

        data.client.request = request
        await ensure_activity_subscription(data)
        await ensure_activity_subscription(data)  # already subscribed
        assert len(posted) == 1
        method, path, body = posted[0]
        assert (method, path) == ("POST", "/ens/v1/activities/subscriptions")
        assert body["Arguments"]["Activities"] == ["Orders", "Positions"]
        ref = next(r for r, s in data.subscriptions.items() if s["kind"] == "ACTIVITIES")
        assert not broker.reconcile_due()
        await data.receive(
            {
                "reference": ref,
                "message_id": "7",
                "payload": [{"ActivityType": "Orders", "Status": "FinalFill"}],
            }
        )
        assert broker.reconcile_due()
        store.db.close()

    asyncio.run(scenario())


def test_activity_events_are_sim_execution_only(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)  # INTERNAL_PAPER
        data.subscriptions.clear()
        await ensure_activity_subscription(data)
        assert not data.subscriptions and not data.client.calls
        store.db.close()

    asyncio.run(scenario())


def test_saxo_closed_positions_are_kept_as_evidence_in_base_currency(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        reference = record_entry(store, identity="x")
        rows = [
            {
                "ClosedPositionUniqueId": "A-B",
                "ClosedPosition": {
                    "OpeningExternalReferenceId": reference,
                    "ClosingExternalReferenceId": "other",
                    "ClosedProfitLossInBaseCurrency": 12.5,
                    "CostOpeningInBaseCurrency": 0.8,
                    "CostClosingInBaseCurrency": 0.7,
                    "Uic": 100,
                },
            },
            {
                "ClosedPositionUniqueId": "C-D",
                "ClosedPosition": {
                    "OpeningExternalReferenceId": "someone-else",
                    "ClosedProfitLossInBaseCurrency": 99,
                },
            },
        ]
        requests = []

        async def request(method, path, params=None, **kwargs):
            requests.append(path)
            return {"Data": rows}

        data.client.request = request
        await broker.refresh_closed_positions()
        await broker.refresh_closed_positions()  # ten-minute spacing
        assert requests == ["/port/v1/closedpositions"]
        reported = store.economics()["broker_reported"]
        assert reported == {"count": 1, "closed_profit_loss_base": 12.5, "costs_base": 1.5}
        audit = store.db.execute(
            "SELECT reference, detail FROM lifecycle WHERE kind='SAXO_SIM_CLOSED_POSITION'"
        ).fetchall()
        assert [a[0] for a in audit] == [reference]
        assert json.loads(audit[0][1])["ClosedProfitLossInBaseCurrency"] == 12.5
        broker.closed_checked = float("-inf")
        await broker.refresh_closed_positions()  # already recorded: no duplicate
        assert store.economics()["broker_reported"]["count"] == 1
        store.db.close()

    asyncio.run(scenario())


def test_market_view_carries_smile_price_context_and_model_sigma(tmp_path):
    from stocker_execution.config import FuturesConfig
    from stocker_execution.runtime import Runtime
    from stocker_execution.store import Store

    runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
    state = runtime.markets["CL"]
    state.identity = FUTURE
    state.option_board = BOARD
    detail = runtime.market_detail("CL")["markets"][0]
    # Without today's expiry in the chain window, the first listed expiry is shown with its date.
    assert detail["smile"]["expiry"] == "2026-09-29T00:00:00Z"
    assert detail["price_context"]["basis"].startswith("Saxo")
    assert detail["model_sigma"] is None  # no completed bars yet
    runtime.store.db.close()


def test_iv_spread_appears_only_with_an_operator_verified_scale():
    raw = views.smile(BOARD, "2026-09-30", "UNVERIFIED", 0.2)
    put = raw["strikes"][0]["put"]
    assert put["mid_volatility"] == 0.35 and put["iv"] is None and put["iv_minus_model"] is None
    assert raw["scaling"] == "PROVIDER_NATIVE_UNVERIFIED"
    fraction = views.smile(BOARD, "2026-09-30", "FRACTION", 0.2)["strikes"][0]["put"]
    assert fraction["iv"] == 0.35 and abs(fraction["iv_minus_model"] - 0.15) < 1e-12
    board = {
        "Expiries": [
            {
                "Expiry": "2026-09-30",
                "Strikes": [{"Strike": 69, "Put": {"Greeks": {"MidVolatility": 35.0}}}],
            }
        ]
    }
    percent = views.smile(board, "2026-09-30", "PERCENT", 0.2)
    assert percent["scaling"] == "PERCENT" and percent["strikes"][0]["put"]["iv"] == 0.35
    assert (
        views.smile(board, "2026-09-30", "PERCENT", None)["strikes"][0]["put"]["iv_minus_model"]
        is None
    )


def delayed_quote(delay):
    price = PriceState()
    price.snapshot(
        {
            "Quote": {
                "Bid": 90.39,
                "Ask": 90.41,
                "PriceTypeBid": "OldIndicative",
                "PriceTypeAsk": "OldIndicative",
                "DelayedByMinutes": delay,
            }
        },
        "fixture",
        time.time(),
    )
    return price


def test_delayed_saxo_data_is_named_in_gates_setup_and_history(tmp_path):
    from stocker_execution.config import FuturesConfig
    from stocker_execution.runtime import Runtime
    from stocker_execution.store import Store

    runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
    state = runtime.markets["CL"]
    state.identity, state.price = FUTURE, delayed_quote(10)
    quote_gate = next(g for g in runtime.overview()["markets"][0]["gates"] if g["key"] == "quote")
    assert quote_gate == {
        "key": "quote",
        "label": "Quote",
        "ok": False,
        "detail": "QUOTE_DELAYED_OR_DELAY_UNKNOWN",
    }
    realtime = next(i for i in runtime.status()["setup"] if i["key"] == "realtime")
    assert not realtime["done"] and "delayed by 10 min" in realtime["detail"]
    state.price = delayed_quote(0)
    quote_gate = next(g for g in runtime.overview()["markets"][0]["gates"] if g["key"] == "quote")
    assert quote_gate["detail"] == "QUOTE_NOT_USABLE"  # OldIndicative is never usable
    assert next(i for i in runtime.status()["setup"] if i["key"] == "realtime")["done"]
    runtime.store.db.close()


def test_delayed_chart_data_blocks_history_with_a_named_reason(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        state = data.markets["CL"]
        now = datetime.now(UTC).replace(second=0, microsecond=0)

        async def request(method, path, params=None, **kwargs):
            return {
                "Data": [
                    {
                        "Time": (now - timedelta(minutes=12 - i)).isoformat(),
                        "Open": 70,
                        "High": 71,
                        "Low": 69,
                        "Close": 70.5,
                        "Volume": 3,
                    }
                    for i in range(3)
                ],
                "ChartInfo": {"DelayedByMinutes": 10},
                "DataVersion": 1,
            }

        data.client.request = request
        await history(data, state)
        assert state.history_problem == "SAXO_CHART_DATA_DELAYED" and len(state.bars) == 2
        assert state.capabilities["history"]["delayed_by_minutes"] == 10
        store.db.close()

    asyncio.run(scenario())


def test_verified_price_midvol_gives_an_implied_minus_model_spread():
    from stocker_execution import option_context

    fields = option_context.fields({"Greeks": {"MidVol": 0.589}}, 100.0, "REGULAR_PRICE")
    assert fields["Greeks.MidVol"]["scaling"] == "VERIFIED_LIVE"
    assert fields["Greeks.MidVol"]["normalised"] == 0.589
    spread = views.iv_spread({"analytics": {"Greeks.MidVol": {"value": 0.589}}}, 0.002)
    sigma = 0.002 * (525600 / 15) ** 0.5
    assert spread["model_sigma"] == sigma
    assert abs(spread["iv_minus_model_sigma"] - (0.589 - sigma)) < 1e-12
    assert views.iv_spread({}, 0.002)["iv_minus_model_sigma"] is None
    assert (
        views.iv_spread({"analytics": {"Greeks.MidVol": {"value": 0.5}}}, None)[
            "iv_minus_model_sigma"
        ]
        is None
    )
