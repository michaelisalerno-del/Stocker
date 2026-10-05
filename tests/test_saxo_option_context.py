"""Sanitised, synthetic observations. No entitlement or feed availability claims."""

import asyncio
import copy
import json
import time
import zlib
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from saxo_support import FUTURE, OPTION, FakeClient, frozen_strike, quote, setup, signal
from stocker_execution import option_context
from stocker_execution.config import FuturesConfig, OptionApproval, RecorderConfig
from stocker_execution.contracts import (
    cost_estimate,
    deadline_instant,
    executable_quote,
    fill_conversion,
    key,
    option_identity,
    option_price,
    require_one_whole_contract,
    tick_at,
    verified_cutoff,
)
from stocker_execution.recorder import Recorder, read_row
from stocker_execution.rules import opportunity, session_day
from stocker_execution.saxo_auth import SaxoError
from stocker_execution.saxo_client import allowed
from stocker_execution.saxo_data import DataService, chain_identity
from stocker_execution.saxo_stream import PriceState, merge_board


def conditions(uic=101):
    return {
        "Uic": uic,
        "AssetType": "FuturesOption",
        "InstrumentCurrency": "USD",
        "AccountCurrency": "USD",
        "CommissionLimits": [
            {"OrderAction": "ExecuteOrder", "Currency": "USD", "PerUnitRate": 0.1}
        ],
        "ExchangeFeeRules": [
            {"OrderAction": "ExecuteOrder", "Type": "PerLot", "Currency": "USD", "Value": 0.02}
        ],
    }


def test_eto_fields_no_fx_delta_and_unknown_scales_preserve_signs():
    side = {
        "Uic": 101,
        "ContractId": 999,
        "DeltaPct": 80,
        "Greeks": {
            "Delta": -0.2,
            "Gamma": 0.1,
            "Theta": -0.04,
            "Vega": 0.3,
            "BidVolatility": 20,
            "AskVolatility": 22,
            "MidVolatility": 21,
        },
        "Volume": 0,
        "OpenInterest": 120,
        "LastTraded": 1.2,
    }
    fields = option_context.fields(option_context.chain_update(side), 100, "OPTIONS_CHAIN")
    assert fields["Greeks.Delta"]["raw"] == -0.2
    assert fields["Greeks.Theta"]["value"] == -0.04
    assert fields["Greeks.MidVolatility"]["normalised"] is None
    assert fields["InstrumentPriceDetails.OpenInterest"]["unit"] == "contracts"
    assert fields["PriceInfoDetails.Volume"]["value"] == 0
    assert "DeltaPct" not in str(fields)
    assert not option_context.fields(
        option_context.chain_update({"DeltaPct": 80}), 100, "OPTIONS_CHAIN"
    )


def test_missing_stale_optional_fields_do_not_break_quotes_or_fabricate_volume():
    p = quote(at=100)
    p.update({"Greeks": {"Delta": -0.2}, "PriceInfoDetails": {"LastTradedSize": 5}}, "1", 101)
    p.update({"Quote": {"Bid": 1, "Ask": 2}, "PriceInfoDetails": {"LastTradedSize": 5}}, "2", 180)
    assert p.receipt == 180
    assert "PriceInfoDetails.Volume" not in p.analytics
    assert option_context.view(p.analytics, 180)["Greeks.Delta"]["status"] == "STALE"
    p.update({"Greeks": None}, "3", 181)
    assert p.analytics["Greeks.Delta"]["value"] is None
    assert p.value["Quote"]["Bid"] == 1


def test_oi_publication_receipt_is_not_effective_date_or_backfilled_decision():
    p = PriceState()
    p.snapshot({"InstrumentPriceDetails": {"OpenInterest": 100}}, "s", 100)
    earlier = copy.deepcopy(option_context.view(p.analytics, 110))
    p.update(
        {"LastUpdated": "2026-09-27T00:00:00Z", "InstrumentPriceDetails": {"OpenInterest": 150}},
        "m",
        200,
    )
    assert earlier["InstrumentPriceDetails.OpenInterest"]["value"] == 100
    assert option_context.view(p.analytics, 199) == {}
    row = option_context.view(p.analytics, 200)["InstrumentPriceDetails.OpenInterest"]
    assert row["effective_at"] is None and row["received_at"] == 200
    assert row["status"] == "AS_OF_EFFECTIVE_TIME_UNKNOWN"
    assert (
        option_context.view(p.analytics, 90000)["InstrumentPriceDetails.OpenInterest"]["status"]
        == "STALE"
    )


def test_chain_old_delayed_and_stale_prices_cannot_price_execution():
    at = datetime.now(UTC)
    p = quote(at=at.timestamp())
    with pytest.raises(ValueError, match="CHAIN_PRICE"):
        executable_quote(OPTION, {**p.value, "price_source": "OPTIONS_CHAIN"}, p.receipt, at)
    # Saxo's normal real-time price quality is Indicative (Tradable is obsolete).
    p.value["Quote"].update(PriceTypeBid="Indicative", PriceTypeAsk="Indicative")
    assert executable_quote(OPTION, p.value, p.receipt, at)
    for quality in ("OldIndicative", "Pending", "NoMarket", "NoAccess", "None", None):
        stale = {**p.value, "Quote": {**p.value["Quote"], "PriceTypeAsk": quality}}
        with pytest.raises(ValueError, match="QUOTE_NOT_USABLE"):
            executable_quote(OPTION, stale, p.receipt, at)
    delayed = {**p.value, "Quote": {**p.value["Quote"], "DelayedByMinutes": 10}}
    with pytest.raises(ValueError, match="QUOTE_DELAYED"):
        executable_quote(OPTION, delayed, p.receipt, at)
    with pytest.raises(ValueError, match="STALE"):
        executable_quote(OPTION, p.value, p.receipt, at + timedelta(seconds=6))


def test_same_board_index_new_uic_never_carries_old_greeks_or_prices():
    assert merge_board({"Uic": 1, "Bid": 2, "Greeks": {"Delta": 0.2}}, {"Uic": 2}) == {"Uic": 2}


def test_partial_pretrigger_and_metadata_are_shared_not_repeated(tmp_path):
    r = Recorder(
        RecorderConfig(persistent_capture=True, recording_permission_evidence="fixture"), tmp_path
    )
    r.register("future", FUTURE)
    r.ingest("future", "SNAPSHOT", {"Quote": {"Bid": 1}}, 0)
    r.trigger("future", {"id": "first"}, 100)
    r.register("option", OPTION)
    version = r.metadata_version("option", {"reference": {"ContractSize": 1000}}, 110)
    r.ingest("option", "SNAPSHOT", {"Quote": {"Bid": 1}}, 110)
    r.attach("first", "option", 120)
    capture = next(iter(r.active.values()))
    assert capture["events"][0]["prehistory_seconds"]["option"] == 0
    assert list(capture["metadata_versions"]) == [version]
    r.trigger("future", {"id": "second"}, 130, ["option"])
    assert len(r.active) == 1
    assert capture["events"][1]["prehistory_seconds"]["option"] == 20
    r.ingest("option", "UPDATE", {"Quote": {"Bid": 2}}, 131)
    rows = [read_row(b) for _, b, _ in r.windows["option"].rows]
    assert all(row["metadata_version"] == version for row in rows)
    assert "ContractSize" not in json.dumps(rows)
    sequences = [
        json.loads(b).get("local_sequence")
        for item in list(r.queue._queue)
        for batch in item[2]
        for b in zlib.decompress(batch).splitlines()
        if json.loads(b).get("identity", {}).get("asset_type") == "FuturesOption"
        and "local_sequence" in json.loads(b)
    ]
    assert sequences == [1, 2]


@pytest.mark.parametrize("factor", [20, 100, 1000, 5000, 10000, 20000, 300000])
def test_contract_conversion_and_separate_fees_budget(factor):
    result = cost_estimate({**OPTION, "price_factor": factor}, 0.005, conditions(), 0.8)
    assert result["premium_gbp"] == pytest.approx(0.005 * factor * 0.8)
    assert result["entry_costs_gbp"] == pytest.approx(0.096)
    assert result["estimated_exit_costs_gbp"] == pytest.approx(0.096)
    assert result["fees_gbp"] == pytest.approx(0.192)
    assert result["remaining_budget_gbp"] == pytest.approx(1000 - result["total_gbp"])
    assert (result["budget_result"] == "WITHIN_BUDGET") == (factor <= 20000)


def test_cost_totals_not_added_twice_and_unsupported_conventions_block():
    c = conditions()
    c["EstimatedTotalCost"] = 999  # never added to component schedule
    assert cost_estimate(OPTION, 0.001, c, 0.8)["total_gbp"] == 1.0
    c["CommissionLimits"][0]["PerUnitRate"] = 0
    c["ExchangeFeeRules"] = []
    assert cost_estimate(OPTION, 0.001, c, 1)["fees_gbp"] == 0
    for option in ({**OPTION, "lot_size": 2}, {**OPTION, "amount_decimals": 1}):
        with pytest.raises(ValueError, match="WHOLE_CONTRACT"):
            cost_estimate(option, 0.001, c, 1)
    c["CommissionLimits"][0]["RateOnAmount"] = 0.1
    with pytest.raises(ValueError, match="SCALING_UNVERIFIED"):
        cost_estimate(OPTION, 0.001, c, 1)
    with pytest.raises(ValueError, match="TRADING_NOT_ALLOWED"):
        cost_estimate(OPTION, 0.001, {**conditions(), "IsTradable": False}, 1)
    # LIVE returns overnight financing for every option; Saxo applies it to short options only.
    carrying = {"InterbankRate": {"AskRate": 1e-06, "BidRate": 1e-06}, "MarkUpRate": 2.5}
    with_carrying = {**conditions(), "CarryingCost": carrying}
    assert cost_estimate(OPTION, 0.001, with_carrying, 0.8) == cost_estimate(
        OPTION, 0.001, conditions(), 0.8
    )
    with pytest.raises(ValueError, match="COST_RULE_UNVERIFIED_HOLDINGFEE"):
        cost_estimate(OPTION, 0.001, {**with_carrying, "HoldingFee": {"Rate": 1}}, 0.8)


def test_gbp_account_pays_saxo_conversion_markup_both_ways():
    # LIVE GBP account, USD option (2026-10-01): Markup 0.6 per cent on each conversion.
    gbp = {**conditions(), "AccountCurrency": "GBP", "CurrencyConversion": {"Markup": 0.6}}
    plain = cost_estimate(OPTION, 0.005, conditions(), 0.75)
    marked = cost_estimate(OPTION, 0.005, gbp, 0.75)
    assert marked["fx_markup"] == pytest.approx(0.006) and plain["fx_markup"] == 0
    assert marked["premium_gbp"] == pytest.approx(plain["premium_gbp"] * 1.006)
    assert fill_conversion(0.75, 0.006, "ENTRY") == pytest.approx(0.7545)
    assert fill_conversion(0.75, 0.006, "EXIT") == pytest.approx(0.7455)
    with pytest.raises(ValueError, match="MISSING_BROKER_FX_MARKUP"):
        cost_estimate(OPTION, 0.005, {**gbp, "CurrencyConversion": {}}, 0.75)


def test_deadlines_distinct_timezone_dst_and_missing_never_invented():
    assert (
        deadline_instant("16:00:00", "2026-07-01", "America/New_York")
        == "2026-07-01T20:00:00+00:00"
    )
    assert (
        deadline_instant("16:00:00", "2026-12-01", "America/New_York")
        == "2026-12-01T21:00:00+00:00"
    )
    assert deadline_instant("01:30:00", "2026-11-01", "America/New_York") is None
    assert deadline_instant("02:30:00", "2026-03-08", "America/New_York") is None
    assert deadline_instant("16:00:00", "2026-07-01") is None
    assert deadline_instant("0001-01-01T00:00:00Z", "2026-07-01") is None
    exit_at = datetime(2026, 9, 28, 14, tzinfo=UTC)
    option = {
        **OPTION,
        "expiry": "2026-09-28",
        "expiry_instant": "2026-09-28T20:00:00Z",
        "last_trade_at": "2026-09-28T18:00:00Z",
        "exercise_cutoff": "21:00:00",
        "trading_sessions": {
            "Sessions": [
                {
                    "StartTime": "2026-09-28T12:00:00Z",
                    "EndTime": "2026-09-28T19:00:00Z",
                    "State": "AutomatedTrading",
                }
            ]
        },
    }
    assert verified_cutoff(option, exit_at).hour == 18
    with pytest.raises(ValueError, match="LAST_TRADING"):
        verified_cutoff({**option, "last_trade_at": None}, exit_at)
    with pytest.raises(ValueError, match="CUTOFF"):
        verified_cutoff(option, exit_at.replace(hour=18) - timedelta(seconds=60))


def test_config_budgets_and_exact_expiry_evidence_required():
    for n in (0, 17):
        with pytest.raises(ValidationError):
            FuturesConfig(option_subscription_budget=n)
    with pytest.raises(ValidationError):
        OptionApproval(
            environment="SAXO_SIM",
            option_root_ids=(50,),
            delta_tolerance=0.02,
            source="frozen source",
            approval="explicit approval",
            fee_per_side_gbp=1,
            fee_evidence="fixture fee evidence",
            expiry_instants={"2026-09-28": "2026-09-28T20:00:00Z"},
        )


def test_frozen_model_ranking_preserved_and_provider_delta_not_substituted(tmp_path):
    config = FuturesConfig(
        mappings={
            "CL": {
                "environment": "SAXO_SIM",
                "option_root_ids": [50],
                "delta_tolerance": 0.01,
                "source": "frozen source",
                "approval": "explicit approval",
                "fee_per_side_gbp": 0.1,
                "fee_evidence": "fixture evidence",
                "expiry_time_evidence": "verified fixture instant",
                "expiry_instants": {"2026-09-28": "2026-09-28T20:00:00Z"},
            }
        }
    )
    data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
    state = data.markets["CL"]
    state.identity, state.option_roots = FUTURE, (50,)
    at = datetime(2026, 9, 28, 14, tzinfo=UTC)
    expiry = at.replace(hour=20)
    state.expiry_instants = {"2026-09-28": expiry}
    strike = frozen_strike(70, 0.01, at, expiry, "C", 0.1)
    state.option_space = [
        {
            "Uic": 101,
            "UnderlyingUic": 100,
            "PutCall": "Call",
            "StrikePrice": strike,
            "Expiry": "2026-09-28",
        },
        {
            "Uic": 102,
            "UnderlyingUic": 100,
            "PutCall": "Call",
            "StrikePrice": strike + 1,
            "Expiry": "2026-09-28",
        },
    ]
    event = opportunity("CL", 100, at)
    ranked = data.rank_candidates(state, event, {"futures_price": 70, "rv15": 0.01})
    assert ranked[0][2]["Uic"] == 101 and ranked[0][0] < 1e-8
    assert not state.option_board  # Missing provider Greeks are optional to this existing rule.


def test_chain_history_stays_with_contract_and_never_modifies_regular_price(tmp_path):
    _, data, store = setup(tmp_path)
    state = data.markets["CL"]
    data.recorder.register(key(OPTION), OPTION)
    board = {
        "Expiries": [
            {
                "Index": 0,
                "Strikes": [
                    {"Index": 0, "Call": {"Uic": 101, "Bid": 99, "Greeks": {"Delta": 0.2}}}
                ],
            }
        ]
    }
    state.option_board = board
    data.record_board(state, board, 100)
    assert data.options[101][1].value["Quote"]["Bid"] != 99
    row = read_row(data.recorder.windows[key(OPTION)].rows[0][1])
    assert row["kind"] == "CHAIN_CONTEXT" and row["identity"]["uic"] == 101
    assert row["observation_context"]["executable"] is False
    store.db.close()


def test_owned_and_captured_options_not_evicted_at_budget(tmp_path):
    async def run():
        config = FuturesConfig(option_subscription_budget=4)
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_roots = FUTURE, (50,)
        for uic in range(101, 105):
            data.options[uic] = ({**OPTION, "uic": uic}, quote())
        data.owned_options = set(data.options)
        with pytest.raises(ValueError, match="SUBSCRIPTION_CAPACITY"):
            await data.option_subscribe(state, {"Uic": 105})
        assert len(data.options) == 4 and not data.client.calls
        assert not data.option_required_at

    asyncio.run(run())


def test_contract_option_cost_endpoint_only_and_live_orders_still_blocked():
    assert allowed("GET", "/cs/v1/tradingconditions/ContractOptionSpaces/fixture-key==/50")
    assert allowed("GET", "/cs/v1/tradingconditions/ContractOptionSpaces/fixture%2B%2Fkey%3D%3D/50")
    assert not allowed("GET", "/cs/v1/tradingconditions/ContractOptionSpaces/%2E%2E/50")
    assert not allowed("GET", "/cs/v1/tradingconditions/instrument/fixture-key/101/FuturesOption")
    assert not allowed("POST", "/trade/v2/orders")


def test_option_reference_cross_underlying_and_expiry_are_rejected():
    raw = {
        "AssetType": "FuturesOption",
        "UnderlyingAssetType": "ContractFutures",
        "Uic": 101,
        "PutCall": "Call",
        "StrikePrice": 70,
        "ExpiryDate": "2026-09-29",
    }
    space = {
        "Uic": 101,
        "UnderlyingUic": 100,
        "PutCall": "Call",
        "StrikePrice": 70,
        "Expiry": "2026-09-28",
    }
    with pytest.raises(ValueError, match="EXPIRY_MISMATCH"):
        option_identity(FUTURE, raw, 50, space)
    with pytest.raises(ValueError, match="UNDERLYING_RELATIONSHIP"):
        option_identity(FUTURE, raw, 50, {**space, "UnderlyingUic": 999})


def test_live_option_schema_one_contract_and_tiered_ticks():
    # Observed LIVE NQ option details (2026-10-01): no TickSize, a price-tiered scheme only.
    space = {
        "Uic": 101,
        "UnderlyingUic": 100,
        "PutCall": "Call",
        "StrikePrice": 70,
        "Expiry": OPTION["expiry"],
    }
    raw = {k: v for k, v in reference().items() if k != "TickSize"}
    raw["TickSizeScheme"] = {
        "DefaultTickSize": 1.0,
        "Elements": [
            {"HighPrice": 100.0, "TickSize": 0.25},
            {"HighPrice": 5.0, "TickSize": 0.05},
            {"HighPrice": 500.0, "TickSize": 0.5},
        ],
    }
    nq = option_identity(FUTURE, raw, 50, space)
    assert nq["tick_size"] is None
    assert (nq["minimum_quantity"], nq["lot_size"]) == (1, 1)
    require_one_whole_contract(nq)
    # A boundary lies on both grids; only the step away from it changes tier.
    assert option_price(nq, 4.95, ROUND_CEILING, 1) == 5.0
    assert option_price(nq, 5.0, ROUND_CEILING, 1) == 5.25
    assert option_price(nq, 5.03, ROUND_CEILING, 0) == 5.25
    assert option_price(nq, 100.0, ROUND_CEILING, 1) == 100.5
    assert option_price(nq, 5.0, ROUND_FLOOR, -1) == 4.95
    assert option_price(nq, 100.5, ROUND_FLOOR, -1) == 100.0
    assert option_price(nq, 600.0, ROUND_FLOOR, -1) == 599.0
    assert (tick_at(nq, 5.0, 1), tick_at(nq, 5.0, -1)) == (0.25, 0.05)
    cl = option_identity(FUTURE, reference(), 50, space)
    assert cl["tick_size"] == 0.001 and cl["tick_size_scheme"] is None
    assert option_price(cl, 1.2345, ROUND_CEILING, 1) == 1.236
    with pytest.raises(ValueError, match="MISSING_INCREMENT_SIZE"):
        option_identity(FUTURE, {**raw, "IncrementSize": None}, 50, space)
    with pytest.raises(ValueError, match="MISSING_TICK_SIZE"):
        option_identity(FUTURE, {**raw, "TickSizeScheme": None}, 50, space)


def reference():
    return {
        "Uic": 101,
        "AssetType": "FuturesOption",
        "UnderlyingAssetType": "ContractFutures",
        "PutCall": "Call",
        "StrikePrice": 70,
        "ExpiryDate": OPTION["expiry"],
        "CurrencyCode": "USD",
        "ContractSize": 1000,
        "PriceToContractFactor": 1000,
        "TickSize": 0.001,
        # LIVE schema (2026-10-01): IncrementSize is the amount step, MinimumLotSize 0 = none.
        "IncrementSize": 1.0,
        "MinimumLotSize": 0.0,
        "AmountDecimals": 0,
        "IsTradable": True,
        "TradingSessions": OPTION["trading_sessions"],
    }


def test_optional_cost_failure_does_not_stop_regular_observation(tmp_path):
    async def run():
        config = FuturesConfig()
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_roots = FUTURE, (50,)
        data.client.request = AsyncMock(
            side_effect=[
                reference(),
                SaxoError("NoAccess"),
                {"Snapshot": quote().value, "RefreshRate": 1000},
            ]
        )
        await data.option_subscribe(
            state,
            {
                "Uic": 101,
                "UnderlyingUic": 100,
                "PutCall": "Call",
                "StrikePrice": 70,
                "Expiry": OPTION["expiry"],
                "OptionRootId": 50,
            },
        )
        assert data.options[101][1].value["Quote"]["Ask"] == 0.009
        assert data.option_references[101]["problem"] == "NoAccess"
        assert data.option_view(101, time.time())["costs"]["reason"] == "NoAccess"

    asyncio.run(run())


def test_event_pins_one_contract_and_detaches_decision_evidence(tmp_path, monkeypatch):
    async def run():
        broker, data, store = setup(tmp_path)
        data.markets["CL"].option_roots = (50,)
        event = signal(1)
        data.recorder = Recorder(
            RecorderConfig(
                persistent_capture=True, recording_permission_evidence="OFFLINE FIXTURE"
            ),
            tmp_path,
        )
        data.recorder.register(key(OPTION), OPTION)
        data.recorder.register(key(FUTURE), FUTURE)
        data.recorder.trigger(key(FUTURE), event, time.time(), [key(OPTION)])
        monkeypatch.setattr(
            data, "rank_candidates", lambda *_: [(0.001, 70, {"Uic": 101, "OptionRootId": 50})]
        )
        monkeypatch.setattr(
            data,
            "focus_board",
            AsyncMock(side_effect=AssertionError("Optional I/O in decision path")),
        )
        first, _ = await broker.select_option(event, data.markets["CL"], {})
        monkeypatch.setattr(data, "rank_candidates", lambda *_: [(0, 71, {"Uic": 102})])
        second, _ = await broker.select_option(event, data.markets["CL"], {})
        assert first["uic"] == second["uic"] == 101 and 102 not in data.options
        context = data.option_view(101, time.time())
        data.recorder.annotate(event["id"], {"option_context": context})
        context["identity"]["last_trade_at"] = "later publication"
        captured = next(iter(data.recorder.active.values()))["events"][0]
        assert "later publication" not in json.dumps(captured)
        assert "last_trade_at" not in data.options[101][0]
        store.db.close()

    asyncio.run(run())


def test_chain_focus_centres_on_the_money_of_the_selected_expiry(tmp_path):
    async def run():
        _, data, store = setup(tmp_path)
        state = data.markets["CL"]
        strikes = [{"Index": i, "Strike": 60 + i * 0.5} for i in range(80)]  # 60.0 .. 99.5
        state.option_board = {
            "Expiries": [
                {"Index": 0, "Expiry": "2026-09-27", "MidStrikePrice": 50, "Strikes": []},
                {
                    "Index": 3,
                    "Expiry": OPTION["expiry"],
                    "MidStrikePrice": 70.1,
                    "Strikes": strikes,
                },
            ]
        }
        data.subscriptions["chain"] = {
            "kind": "BOARD",
            "target": "CL",
            "path": "/trade/v1/optionschain/subscriptions",
        }
        data.client.request = AsyncMock(return_value={})
        await data.focus_board(state, OPTION)  # the candidate's strike is elsewhere
        await data.focus_board(state, OPTION)
        data.client.request.assert_awaited_once()
        args = data.client.request.call_args
        assert args.args[0] == "PATCH"
        assert args.kwargs["body"] == {
            # Observation-only window: option_chain_strikes centred on the strike nearest
            # the money (70.0, index 20), on the selected option's expiry.
            "Expiries": [{"Index": 3, "StrikeStartIndex": 15}],
            "MaxStrikesPerExpiry": 11,
        }
        state.option_board["Expiries"][1]["MidStrikePrice"] = 71.0  # two strikes: kept
        await data.focus_board(state, OPTION)
        data.client.request.assert_awaited_once()
        state.option_board["Expiries"][1]["MidStrikePrice"] = 72.0  # four strikes: moved
        await data.focus_board(state, OPTION)
        assert data.client.request.call_args.kwargs["body"]["Expiries"] == [
            {"Index": 3, "StrikeStartIndex": 19}
        ]
        store.db.close()

    asyncio.run(run())


def test_chain_window_is_recorded_merged_by_index_and_joins_the_capture(tmp_path):
    _, data, store = setup(tmp_path)
    state = data.markets["CL"]
    data.recorder = Recorder(
        RecorderConfig(persistent_capture=True, recording_permission_evidence="OFFLINE FIXTURE"),
        tmp_path / "chain",
    )
    data.recorder.register(key(FUTURE), FUTURE)
    chain = key(chain_identity(FUTURE))
    snapshot = {
        "Expiries": [
            {
                "Index": 0,
                "MidStrikePrice": 70,
                "Strikes": [
                    {"Index": 0, "Strike": 69, "Call": {"Bid": 1.2, "Ask": 1.3}},
                    {"Index": 1, "Strike": 70, "Call": {"Bid": 0.6, "Ask": 0.7}},
                ],
            }
        ]
    }
    data.record_chain(state, "SNAPSHOT", snapshot, 100)
    data.record_chain(
        state,
        "UPDATE",
        {"Expiries": [{"Index": 0, "Strikes": [{"Index": 1, "Call": {"Bid": 0.65}}]}]},
        101,
    )
    strikes = data.recorder.windows[chain].current["Expiries"][0]["Strikes"]
    assert strikes[0]["Call"] == {"Bid": 1.2, "Ask": 1.3}
    assert strikes[1]["Call"] == {"Bid": 0.65, "Ask": 0.7}
    assert data.recorder.windows[chain].identity["asset_type"] == "OptionsChain"
    # The future's book flow ignores chain rows: they live under their own instrument.
    assert data.recorder.windows[key(FUTURE)].sequence == 0
    capture = data.recorder.trigger(key(FUTURE), signal(1), time.time(), [chain])
    assert chain in data.recorder.active[capture["segment"]]["instruments"]
    assert not state.capabilities["chain_recording_problem"]
    store.db.close()


def test_permanently_disabled_chain_does_not_disable_future_price(tmp_path):
    async def run():
        _, data, store = setup(tmp_path)
        data.subscriptions["chain"] = {"kind": "BOARD", "target": "CL"}
        await data.receive(
            {
                "reference": "_heartbeat",
                "message_id": "1",
                "payload": {
                    "Heartbeats": [
                        {
                            "OriginatingReferenceId": "chain",
                            "Reason": "SubscriptionPermanentlyDisabled",
                        }
                    ]
                },
            }
        )
        assert ("BOARD", "CL") in data.disabled_targets
        assert ("PRICE", "CL") not in data.disabled_targets
        assert data.markets["CL"].price.value["Quote"]["Bid"] == 70
        store.db.close()

    asyncio.run(run())


def test_refreshed_multiplier_change_blocks_costs_and_old_size_is_labelled(tmp_path):
    async def run():
        _, data, store = setup(tmp_path)
        data.options[101] = (dict(OPTION), quote(at=100))
        data.option_references[101] = {"received_at": 100, "conditions": conditions()}
        data.client.request = AsyncMock(return_value={**reference(), "PriceToContractFactor": 10})
        await data.refresh_option_metadata()
        assert (
            data.option_references[101]["problem"]
            == "OPTION_REFERENCE_CONVENTION_CHANGED_PRICE_FACTOR"
        )
        data.options[101][1].update({"Quote": {"Bid": 0.008, "Ask": 0.009}}, "2", 110)
        view = data.option_view(101, 110)
        assert view["size_status"]["Ask"] == "STALE_OR_MISSING"
        assert view["quote_status"] == "OBSERVED"
        store.db.close()

    asyncio.run(run())


def test_transport_failure_during_metadata_refresh_keeps_the_verified_entry(tmp_path):
    """A network blip must not mark an owned option untradable for the metadata age."""

    async def run():
        _, data, store = setup(tmp_path)
        identity = dict(OPTION)
        data.options[101] = (identity, quote(at=100))
        data.option_references[101] = {"received_at": 100, "conditions": conditions()}
        data.client.request = AsyncMock(side_effect=SaxoError("SAXO_TRANSPORT_UNAVAILABLE"))
        await data.refresh_option_metadata()
        assert identity["is_tradable"] is True
        assert data.option_references[101] == {"received_at": 100, "conditions": conditions()}
        data.client.request = AsyncMock(return_value="not-a-reference")
        await data.refresh_option_metadata()
        assert identity["is_tradable"] is False
        assert data.option_references[101]["problem"] == "OPTION_METADATA_SCHEMA_UNVERIFIED"
        store.db.close()

    asyncio.run(run())


def test_retired_strike_history_yields_its_window_slot_to_a_live_contract(tmp_path):
    from stocker_execution.config import SUBSCRIPTION_LIMIT

    _, data, store = setup(tmp_path)
    windows = data.recorder.windows
    retired = [{**OPTION, "uic": uic} for uic in range(1000, 1000 + SUBSCRIPTION_LIMIT - 1)]
    for at, identity in enumerate(retired):
        data.recorder.register(key(identity), identity)
        data.recorder.ingest(key(identity), "SNAPSHOT", {"Quote": {"Bid": 1}}, float(at))
    assert len(windows) == SUBSCRIPTION_LIMIT  # the future plus retired strikes
    data.release_window_slot()
    assert len(windows) == SUBSCRIPTION_LIMIT - 1 and key(retired[0]) not in windows
    data.recorder.register(key(retired[0]), retired[0])
    data.options.update({i["uic"]: (i, quote()) for i in retired})  # every strike is live
    with pytest.raises(SaxoError, match="RECORDER_INSTRUMENT_LIMIT"):
        data.release_window_slot()
    store.db.close()


def test_full_candidate_window_rotates_without_mixed_history(tmp_path, monkeypatch):
    async def run():
        config = FuturesConfig()
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_roots = FUTURE, (50,)
        state.bars = [SimpleNamespace(at=datetime.now(UTC) - timedelta(minutes=1), close=70)]
        data.recorder.register(key(FUTURE), FUTURE)
        for uic in range(101, 113):
            identity = {**OPTION, "uic": uic, "market": "CL" if uic < 104 else "ES"}
            data.options[uic] = (identity, quote())
            data.recorder.register(key(identity), identity)
            data.recorder.ingest(key(identity), "SNAPSHOT", {"Quote": {"Bid": uic}}, time.time())
        monkeypatch.setattr("stocker_execution.saxo_data.prior_rv", lambda *_: 0.01)
        monkeypatch.setattr(
            data, "rank_candidates", lambda *_: [(0, 70, {"Uic": uic}) for uic in (201, 202, 203)]
        )

        async def subscribe(_, selected):
            identity = {**OPTION, "uic": selected["Uic"]}
            data.options[selected["Uic"]] = (identity, quote())
            data.recorder.register(key(identity), identity)
            return identity

        monkeypatch.setattr(data, "option_subscribe", subscribe)
        await data.warm_candidates(state)
        assert len(data.options) == 12 and state.candidate_uic == 201
        assert not {101, 102, 103}.intersection(data.options)
        assert data.recorder.windows[key(OPTION)].identity["uic"] == 101
        assert state.candidate_changes[-1]["reason"] == "NEAREST_FROZEN_MODEL_DELTA"
        assert not data.recorder.event_ids

    asyncio.run(run())


def test_malformed_optional_metadata_does_not_stop_other_market_observation(tmp_path):
    async def run():
        _, data, store = setup(tmp_path)
        data.options[101] = (dict(OPTION), quote())
        data.client.request = AsyncMock(return_value={**reference(), "Exchange": None})
        await data.refresh_option_metadata()
        assert data.option_references[101]["problem"] == "OPTION_METADATA_SCHEMA_UNVERIFIED"
        assert data.options[101][1].value["Quote"]["Ask"] == 0.009
        assert data.markets["CL"].price.value["Quote"]["Bid"] == 70
        store.db.close()

    asyncio.run(run())


def weekly_space(day, stamp, uic):
    return {
        "AssetType": "FuturesOption",
        "ExerciseStyle": "American",
        "OptionSpace": [
            {
                "Expiry": day,
                "LastTradeDate": stamp,
                "SpecificOptions": [
                    {"Uic": uic, "UnderlyingUic": 100, "PutCall": "Call", "StrikePrice": 70}
                ],
            }
        ],
    }


def family_mapping(**extra):
    return {
        "environment": "SAXO_SIM",
        "option_root_ids": [51, 52, 53],
        "delta_tolerance": 0.03,
        "source": "weekly family fixture",
        "approval": "explicit approval",
        "fee_per_side_gbp": 2.7,
        "fee_evidence": "fixture fee evidence",
        "expiry_time_evidence": "exchange rule fixture",
        "expiry_clock_new_york": "14:30",
        **extra,
    }


def ny_stamp(day, hour, minute, second=0):
    from stocker_execution.rules import NY

    y, m, d = map(int, day.split("-"))
    local = datetime(y, m, d, hour, minute, second, tzinfo=NY)
    return local.astimezone(UTC).isoformat().replace("+00:00", "Z")


def family_service(tmp_path, spaces):
    import stocker_execution.saxo_data as module

    module.OPTION_SPACE_GAP_SECONDS = 0
    config = FuturesConfig(mappings={"CL": family_mapping()})
    data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))

    async def request(method, path, params=None, **_):
        return spaces[int(path.rsplit("/", 1)[1])]

    data.client.request = request
    state = data.markets["CL"]
    state.identity = FUTURE
    raw = {
        "RelatedOptionRootsEnhanced": [
            {"AssetType": "FuturesOption", "OptionRootId": r} for r in (51, 52, 53, 54)
        ]
    }
    return data, state, raw


def test_weekly_family_loads_today_root_and_saxo_timestamps_at_the_approved_clock(tmp_path):
    from stocker_execution.rules import NY

    today = session_day(datetime.now(NY))  # the app's day: the CME session's
    d = [(today + timedelta(days=i)).isoformat() for i in range(3)]
    spaces = {
        51: weekly_space(d[0], ny_stamp(d[0], 14, 30), 201),
        52: weekly_space(d[1], ny_stamp(d[1], 14, 30), 202),
        53: weekly_space(d[2], ny_stamp(d[2], 13, 0), 203),  # not the approved clock
    }
    data, state, raw = family_service(tmp_path, spaces)
    asyncio.run(data.load_options(state, raw))
    assert state.option_roots == (51, 52, 53)  # 54 is related but not approved
    assert {o["OptionRootId"] for o in state.option_space} == {51, 52, 53}
    assert sorted(state.expiry_instants) == d[:2]
    assert state.expiry_instants[d[0]].astimezone(NY).strftime("%Y-%m-%d %H:%M") == d[0] + " 14:30"
    assert state.option_root == 51  # today's expiry drives the chain board


def test_conflicting_saxo_timestamps_leave_that_day_unverified(tmp_path):
    from stocker_execution.rules import NY

    day = session_day(datetime.now(NY)).isoformat()
    spaces = {
        51: weekly_space(day, ny_stamp(day, 14, 30), 201),
        52: weekly_space(day, ny_stamp(day, 14, 30, 30), 202),
        53: {"AssetType": "FuturesOption", "OptionSpace": []},
    }
    data, state, raw = family_service(tmp_path, spaces)
    asyncio.run(data.load_options(state, raw))
    assert day not in state.expiry_instants


def test_board_moves_to_a_new_root_by_replacing_its_subscription(tmp_path):
    async def run():
        config = FuturesConfig()
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_root = FUTURE, 51
        calls = []

        async def subscribe(kind, arguments, target, old=None):
            calls.append((arguments["Identifier"], old))
            data.subscriptions["S1"] = {"kind": kind, "arguments": arguments, "target": target}
            return "S1"

        data.subscribe = subscribe
        await data.subscribe_board(state)
        await data.subscribe_board(state)  # same root: nothing to do
        state.option_root = 52
        await data.subscribe_board(state)
        assert calls == [(51, None), (52, "S1")]

    asyncio.run(run())


def test_family_approval_needs_one_time_source_evidence_and_unique_roots():
    OptionApproval(**family_mapping())
    for bad in (
        {"expiry_time_evidence": None},
        {"expiry_instants": {"2026-09-28": "2026-09-28T20:00:00Z"}},
        {"option_root_ids": [51, 51]},
        {"option_root_ids": []},
        {"expiry_clock_new_york": "2:30pm"},
    ):
        with pytest.raises(ValidationError):
            OptionApproval(**family_mapping(**bad))


def test_a_series_expiring_the_same_day_at_another_time_is_never_a_candidate(tmp_path):
    from stocker_execution.rules import NY

    at = datetime(2026, 12, 18, 14, tzinfo=UTC)  # 09:00 New York
    day = at.astimezone(NY).date().isoformat()
    config = FuturesConfig(mappings={"CL": family_mapping()})
    data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
    state = data.markets["CL"]
    state.identity, state.option_roots = FUTURE, (51, 52)
    instant = datetime.fromisoformat(ny_stamp(day, 14, 30).replace("Z", "+00:00"))
    state.expiry_instants = {day: instant}
    strike = frozen_strike(70, 0.01, at, instant, "C", 0.1)
    row = {"UnderlyingUic": 100, "PutCall": "Call", "StrikePrice": strike, "Expiry": day}
    state.option_space = [
        {**row, "Uic": 301, "OptionRootId": 51, "LastTradeDate": ny_stamp(day, 9, 30)},
        {**row, "Uic": 302, "OptionRootId": 52, "LastTradeDate": ny_stamp(day, 14, 30)},
    ]
    ranked = data.rank_candidates(
        state, opportunity("CL", 100, at), {"futures_price": 70, "rv15": 0.01}
    )
    assert [r[2]["Uic"] for r in ranked] == [302]


def test_next_listed_expiry_only_where_approved_and_only_without_a_usable_same_day(tmp_path):
    from stocker_execution.rules import NY

    # Thursday 09:00 New York: silver lists Fridays only; the 14:00 clock exits after a 14:30
    # same-day expiry would be too late (exit + 2 minutes must come before it).
    thursday, friday = "2026-10-01", "2026-10-02"

    def ranked_days(rule, at, days):
        mapping = family_mapping(expiry_rule=rule) if rule else family_mapping()
        config = FuturesConfig(mappings={"CL": mapping})
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_roots = FUTURE, (51, 52)
        state.expiry_instants, state.option_space = {}, []
        for uic, day in enumerate(days, 400):
            instant = datetime.fromisoformat(ny_stamp(day, 14, 30).replace("Z", "+00:00"))
            state.expiry_instants[day] = instant
            strike = frozen_strike(70, 0.01, at, instant, "C", 0.1)
            state.option_space.append(
                {
                    "Uic": uic,
                    "UnderlyingUic": 100,
                    "PutCall": "Call",
                    "StrikePrice": strike,
                    "Expiry": day,
                    "OptionRootId": 51,
                    "LastTradeDate": ny_stamp(day, 14, 30),
                }
            )
        event = opportunity("CL", 100, at)
        ranked = data.rank_candidates(state, event, {"futures_price": 70, "rv15": 0.01})
        return {r[2]["Expiry"] for r in ranked}

    nine = datetime(2026, 10, 1, 13, tzinfo=UTC)  # 09:00 New York
    two = datetime(2026, 10, 1, 18, tzinfo=UTC)  # 14:00 New York, exit 15:00
    with pytest.raises(ValueError, match="NO_VERIFIED_REAL_0DTE_EXPIRY_TIME"):
        ranked_days(None, nine, [friday])  # frozen same-day rule is unchanged
    nxt = "SAME_DAY_OR_NEXT_LISTED"
    assert ranked_days(nxt, nine, [friday]) == {friday}
    assert ranked_days(nxt, nine, [thursday, friday]) == {thursday}  # same day still first
    assert ranked_days(nxt, two, [thursday, friday]) == {friday}  # same day ends before exit
    with pytest.raises(ValueError, match="NO_VERIFIED_LISTED_EXPIRY_TIME"):
        ranked_days(nxt, two, [thursday])
    assert two.astimezone(NY).hour == 14


def test_a_failed_daily_option_refresh_stays_with_the_options_and_waits_to_retry(tmp_path):
    async def run():
        config = FuturesConfig()
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity = FUTURE
        calls = []

        async def request(method, path, **_):
            calls.append(path)
            raise SaxoError("RateLimitExceeded")

        data.client.request = request
        await data.refresh_options(state)  # does not raise into the history pass
        await data.refresh_options(state)  # and does not retry straight away
        assert len(calls) == 1
        assert state.capabilities["options"] == {"problem": "RateLimitExceeded"}

    asyncio.run(run())


def test_an_options_problem_at_connect_does_not_block_the_verified_future(tmp_path):
    async def run():
        config = FuturesConfig(
            contracts={
                "CL": {
                    "environment": "SAXO_SIM",
                    "uic": 100,
                    "symbol": FUTURE["symbol"],
                    "exchange": FUTURE["exchange"],
                    "contract_month": FUTURE["contract_month"],
                    "approval": "fixture pin approval",
                }
            },
            mappings={"CL": family_mapping()},
        )
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        details = {
            "AssetType": "ContractFutures",
            "Uic": 100,
            "Symbol": FUTURE["symbol"],
            "ContractSize": 1000,
            "ExpiryDate": "2026-10-20",
            "CurrencyCode": "USD",
            "Exchange": {"ExchangeId": FUTURE["exchange"]},
            "TickSize": 0.01,
            "PriceToContractFactor": 1000,
            "RelatedOptionRootsEnhanced": [
                {"AssetType": "FuturesOption", "OptionRootId": r} for r in (51, 52, 53)
            ],
        }

        async def request(method, path, **_):
            if path == "/ref/v1/instruments":
                return {"Data": []}
            if path.startswith("/ref/v1/instruments/details/"):
                return details
            raise SaxoError("RateLimitExceeded")

        data.client.request = request
        await data.discover(state)
        assert state.identity["uic"] == 100 and state.problem == ""
        assert state.capabilities["options"] == {"problem": "RateLimitExceeded"}
        assert state.option_space_day == ""  # the daily refresh will retry
        # A reconnect on the same New York day re-verifies the future but keeps today's
        # option spaces, so its price feed is not held back behind ~70 option-space requests.
        from stocker_execution.rules import NY

        calls = []

        async def counted(method, path, **kwargs):
            calls.append(path)
            return await request(method, path, **kwargs)

        data.client.request = counted
        state.option_space_day = session_day(datetime.now(NY)).isoformat()
        await data.discover(state)
        assert calls and not any("contractoptionspaces" in p for p in calls)
        state.option_space_day = "2026-09-30"  # a new day still loads them
        await data.discover(state)
        assert any("contractoptionspaces" in p for p in calls)

    asyncio.run(run())


def test_a_failed_price_feed_at_connect_is_retried_from_the_history_pass(tmp_path):
    """One timeout on a market's PRICE subscribe used to leave the market blocked until an
    unrelated reconnect; it is retried a minute later, and only then."""

    async def run():
        config = FuturesConfig(
            contracts={
                "CL": {
                    "environment": "SAXO_SIM",
                    "uic": 100,
                    "symbol": FUTURE["symbol"],
                    "exchange": FUTURE["exchange"],
                    "contract_month": FUTURE["contract_month"],
                    "approval": "fixture pin approval",
                }
            }
        )
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        details = {
            "AssetType": "ContractFutures",
            "Uic": 100,
            "Symbol": FUTURE["symbol"],
            "ContractSize": 1000,
            "ExpiryDate": "2026-10-20",
            "CurrencyCode": "USD",
            "Exchange": {"ExchangeId": FUTURE["exchange"]},
            "TickSize": 0.01,
            "PriceToContractFactor": 1000,
            "RelatedOptionRootsEnhanced": [],
        }

        async def request(method, path, **_):
            if path.startswith("/ref/v1/instruments/details/"):
                return details
            raise SaxoError("RateLimitExceeded")

        attempts = []

        async def subscribe(kind, arguments, target, old=None):
            attempts.append((kind, target))
            if attempts.count(("PRICE", "CL")) == 1:
                raise SaxoError("SAXO_TRANSPORT_UNAVAILABLE")
            return "ref"

        data.client.request = request
        data.subscribe = subscribe
        data.connected = True
        await data.start_market(state)
        assert state.identity["uic"] == 100 and state.problem == "SAXO_TRANSPORT_UNAVAILABLE"
        await data.retry_failed_startups()
        assert state.problem == "" and attempts.count(("PRICE", "CL")) == 2
        state.problem = "SAXO_TRANSPORT_UNAVAILABLE"
        await data.retry_failed_startups()  # not again inside the minute
        assert attempts.count(("PRICE", "CL")) == 2

    asyncio.run(run())


def test_an_approved_root_saxo_does_not_relate_blocks_the_family(tmp_path):
    """Loading only the roots Saxo lists would move SAME_DAY_OR_NEXT_LISTED to the following
    day's option when a weekday root is missing; the family loads whole or not at all."""
    from stocker_execution.saxo_data import roots_verified

    async def run():
        config = FuturesConfig(
            contracts={
                "CL": {
                    "environment": "SAXO_SIM",
                    "uic": 100,
                    "symbol": FUTURE["symbol"],
                    "exchange": FUTURE["exchange"],
                    "contract_month": FUTURE["contract_month"],
                    "approval": "fixture pin approval",
                }
            },
            mappings={"CL": family_mapping()},
        )
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        details = {
            "AssetType": "ContractFutures",
            "Uic": 100,
            "Symbol": FUTURE["symbol"],
            "ContractSize": 1000,
            "ExpiryDate": "2026-10-20",
            "CurrencyCode": "USD",
            "Exchange": {"ExchangeId": FUTURE["exchange"]},
            "TickSize": 0.01,
            "PriceToContractFactor": 1000,
            "RelatedOptionRootsEnhanced": [
                {"AssetType": "FuturesOption", "OptionRootId": r} for r in (51, 53)
            ],
        }

        async def request(method, path, **_):
            if path.startswith("/ref/v1/instruments/details/"):
                return details
            raise AssertionError(path)  # no option space is loaded for a partial family

        data.client.request = request
        await data.discover(state)
        assert state.problem == ""  # the future itself is verified
        assert state.capabilities["options"] == {"problem": "APPROVED_OPTION_ROOT_NOT_RELATED"}
        assert not roots_verified(state, (51, 52, 53))

    asyncio.run(run())
