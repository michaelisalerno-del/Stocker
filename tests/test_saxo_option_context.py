"""Sanitised, synthetic observations. No entitlement or feed availability claims."""

import asyncio
import copy
import json
import time
import zlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from saxo_support import FUTURE, OPTION, FakeClient, quote, setup, signal
from stocker_execution import option_context
from stocker_execution.config import FuturesConfig, OptionApproval, RecorderConfig
from stocker_execution.contracts import (
    cost_estimate,
    deadline_instant,
    executable_quote,
    key,
    option_identity,
    verified_cutoff,
)
from stocker_execution.recorder import Recorder, read_row
from stocker_execution.rules import frozen_strike, opportunity
from stocker_execution.saxo_auth import SaxoError
from stocker_execution.saxo_client import allowed
from stocker_execution.saxo_data import DataService
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
    rows = [read_row(b) for _, b in r.windows["option"].rows]
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


@pytest.mark.parametrize("factor", [20, 100, 1000, 5000, 10000, 20000])
def test_contract_conversion_and_separate_fees_budget(factor):
    result = cost_estimate({**OPTION, "price_factor": factor}, 0.005, conditions(), 0.8)
    assert result["premium_gbp"] == pytest.approx(0.005 * factor * 0.8)
    assert result["entry_costs_gbp"] == pytest.approx(0.096)
    assert result["estimated_exit_costs_gbp"] == pytest.approx(0.096)
    assert result["fees_gbp"] == pytest.approx(0.192)
    assert result["remaining_budget_gbp"] == pytest.approx(50 - result["total_gbp"])
    assert (result["budget_result"] == "WITHIN_BUDGET") == (factor <= 10000)


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
            option_root_id=50,
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
                "option_root_id": 50,
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
    state.identity, state.option_root = FUTURE, 50
    at = datetime(2026, 9, 28, 14, tzinfo=UTC)
    expiry = at.replace(hour=20)
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
        state.identity, state.option_root = FUTURE, 50
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
        "MinimumTradeSize": 1,
        "LotSize": 1,
        "AmountDecimals": 0,
        "IsTradable": True,
        "TradingSessions": OPTION["trading_sessions"],
    }


def test_optional_cost_failure_does_not_stop_regular_observation(tmp_path):
    async def run():
        config = FuturesConfig()
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_root = FUTURE, 50
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
            },
        )
        assert data.options[101][1].value["Quote"]["Ask"] == 0.009
        assert data.option_references[101]["problem"] == "NoAccess"
        assert data.option_view(101, time.time())["costs"]["reason"] == "NoAccess"

    asyncio.run(run())


def test_event_pins_one_contract_and_detaches_decision_evidence(tmp_path, monkeypatch):
    async def run():
        broker, data, store = setup(tmp_path)
        data.markets["CL"].option_root = 50
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
        monkeypatch.setattr(data, "rank_candidates", lambda *_: [(0.001, 70, {"Uic": 101})])
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


def test_chain_focus_uses_selected_expiry_strike_and_existing_subscription(tmp_path):
    async def run():
        _, data, store = setup(tmp_path)
        state = data.markets["CL"]
        state.option_board = {
            "Expiries": [
                {"Index": 3, "Expiry": OPTION["expiry"], "Strikes": [{"Index": 90, "Strike": 70}]}
            ]
        }
        data.subscriptions["chain"] = {
            "kind": "BOARD",
            "target": "CL",
            "path": "/trade/v1/optionschain/subscriptions",
        }
        data.client.request = AsyncMock(return_value={})
        await data.focus_board(state, OPTION)
        await data.focus_board(state, OPTION)
        data.client.request.assert_awaited_once()
        args = data.client.request.call_args
        assert args.args[0] == "PATCH"
        assert args.kwargs["body"] == {
            # Observation-only chain window: option_chain_strikes centred on the strike.
            "Expiries": [{"Index": 3, "StrikeStartIndex": 85}],
            "MaxStrikesPerExpiry": 11,
        }
        store.db.close()

    asyncio.run(run())


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


def test_full_candidate_window_rotates_without_mixed_history(tmp_path, monkeypatch):
    async def run():
        config = FuturesConfig()
        data = DataService(config, FakeClient(), Recorder(config.recorder, tmp_path))
        state = data.markets["CL"]
        state.identity, state.option_root = FUTURE, 50
        state.bars = [SimpleNamespace(at=datetime.now(UTC) - timedelta(minutes=1), close=70)]
        data.recorder.register(key(FUTURE), FUTURE)
        for uic in range(101, 113):
            identity = {**OPTION, "uic": uic, "market": "CL" if uic < 104 else "NG"}
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
