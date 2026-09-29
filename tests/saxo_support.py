"""Shared offline Saxo fixtures for runtime tests: no network, credentials or orders."""

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from stocker_execution.broker import PaperBroker
from stocker_execution.config import FuturesConfig
from stocker_execution.contracts import key
from stocker_execution.recorder import Recorder
from stocker_execution.rules import opportunity
from stocker_execution.saxo_client import SaxoError
from stocker_execution.saxo_data import DataService
from stocker_execution.saxo_stream import PriceState
from stocker_execution.store import Store

FUTURE = {
    "provider": "SAXO",
    "environment": "SAXO_SIM",
    "market": "CL",
    "uic": 100,
    "asset_type": "ContractFutures",
    "symbol": "CLZ6:NYMEX",
    "expiry": "2026-12-20",
    "contract_month": "2026-12",
    "exchange": "NYMEX",
}
OPTION = {
    **FUTURE,
    "uic": 101,
    "asset_type": "FuturesOption",
    "underlying_uic": 100,
    "option_root_id": 50,
    "right": "Call",
    "strike": 70,
    "currency": "USD",
    "minimum_quantity": 1,
    "lot_size": 1,
    "amount_decimals": 0,
    "tick_size": 0.001,
    "price_factor": 1000,
    "multiplier": 1000,
    "is_tradable": True,
    "trading_sessions": {
        "Sessions": [
            {
                "StartTime": (datetime.now(UTC) - timedelta(hours=24)).isoformat(),
                "EndTime": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
                "State": "Open",
            }
        ]
    },
}


def quote(bid=0.008, ask=0.009, at=None):
    p = PriceState()
    p.snapshot(
        {
            "Quote": {
                "Bid": bid,
                "Ask": ask,
                "PriceTypeBid": "Tradable",
                "PriceTypeAsk": "Tradable",
                "DelayedByMinutes": 0,
            },
            "PriceInfoDetails": {"BidSize": 2, "AskSize": 2},
        },
        "fixture",
        at or time.time(),
    )
    return p


class FakeClient:
    def __init__(self):
        self.calls = []
        self.sim_account_verified = True
        self.oauth = SimpleNamespace(account_key="fixture-account")
        self.positions = []
        self.orders = []

    async def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/root/v2/user":
            return {"UserId": "fixture-user"}
        if path == "/port/v1/accounts/me":
            return {
                "Data": [
                    {"AccountKey": "fixture-account", "AccountId": "fixture-id", "Currency": "GBP"}
                ]
            }
        if path == "/root/v1/sessions/capabilities":
            return {"TradeLevel": "FullTradingAndChat"}
        if path == "/port/v1/positions/me":
            return {"Data": self.positions}
        if path == "/port/v1/orders/me":
            return {"Data": self.orders}
        if path.endswith("precheck"):
            return {
                "PreCheckResult": "Ok",
                "EstimatedCashRequired": 7.9,
                "EstimatedCashRequiredCurrency": "GBP",
                "EstimatedTotalCostInAccountCurrency": 0.1,
            }
        raise SaxoError("AMBIGUOUS_REQUEST")


def setup(tmp_path, mode="INTERNAL_PAPER"):
    config = FuturesConfig(execution_mode=mode)
    client = FakeClient()
    r = Recorder(config.recorder, tmp_path / "events")
    data = DataService(config, client, r)
    data.connected = data.account_verified = True
    data.account_id, data.account_currency = "fixture-id", "GBP"
    data.session = {"TradeLevel": "FullTradingAndChat"}
    data.markets["CL"].identity = FUTURE
    data.markets["CL"].price = quote(70, 71)
    r.register(key(FUTURE), FUTURE)
    data.options[101] = (OPTION, quote())
    data.subscriptions["option"] = {"target": "101"}
    data.fx = quote(1.3, 1.31)
    store = Store(tmp_path / "ledger.sqlite3")
    store.bind("SAXO_SIM", mode)
    broker = PaperBroker(config, store, data)
    broker.armed = broker.reconciled = True
    broker.last_reconcile = time.monotonic()
    broker.problem = ""
    return broker, data, store


def signal(i):
    at = datetime.now(UTC)
    return {
        "id": f"fixture-{i}",
        "market": "CL",
        "rule_version": "fixture",
        "signal_at": at.isoformat(),
        "exit_at": (at + timedelta(hours=1)).isoformat(),
    }


def plan():
    return {
        "quantity": 1,
        "cash_pennies": 800,
        "premium_gbp": 7.69,
        "fees_gbp": 0.2,
        "total_gbp": 8,
        "limit": 0.01,
        "option": OPTION,
        "underlying": FUTURE,
        "fee_per_side_gbp": 0.1,
        "currency": "USD",
        "multiplier": 1000,
        "price_unit_factor": 1,
        "fx": 1 / 1.3,
        "fx_at": time.time(),
        "cutoff": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
    }


AT = datetime(2026, 9, 28, 13, tzinfo=UTC)


def ledger_plan(cid=100):
    return {
        "quantity": 1,
        "cash_pennies": 1000,
        "multiplier": 100,
        "price_unit_factor": 1,
        "currency": "USD",
        "option": {"uic": cid, "asset_type": "FuturesOption"},
    }


def record_entry(store, identity="x", cid=100):
    event = {**opportunity("GC", 1, AT), "id": identity}
    store.observe(event, "", {})
    assert store.reserve(identity, ledger_plan(cid)) == ""
    return store.prepare_order(
        identity,
        "ENTRY",
        10 + cid,
        (AT + timedelta(seconds=20)).isoformat(),
        {"con_id": cid, "quantity": 1, "limit": 0.1},
    )


def fill_record(ref, exec_id="exec.1", side="BOT", at=AT):
    return dict(
        exec_id=exec_id,
        reference=ref,
        con_id=100,
        quantity=1,
        price=0.1,
        side=side,
        at=at.isoformat(),
        fx=0.8,
        fx_at=at.isoformat(),
    )


IDENTITY = {
    "provider": "SAXO",
    "environment": "SAXO_SIM",
    "asset_type": "ContractFutures",
    "market": "CL",
    "uic": 123,
    "tick_size": 0.01,
}


def book(n=10, bid_size=3, ask_size=1, shift=0):
    return {
        "Quote": {
            "Bid": 70 + shift,
            "Ask": 70.01 + shift,
            "BidSize": bid_size,
            "AskSize": ask_size,
            "DelayedByMinutes": 0,
            "PriceTypeBid": "Tradable",
            "PriceTypeAsk": "Tradable",
        },
        "PriceInfoDetails": {"LastTraded": 70, "LastTradedSize": 2, "Volume": 100},
        "MarketDepth": {
            "Bid": [70 + shift - 0.01 * i for i in range(n)],
            "Ask": [70.01 + shift + 0.01 * i for i in range(n)],
            "BidSize": [bid_size] * n,
            "AskSize": [ask_size] * n,
            "BidOrders": [2] * n,
            "AskOrders": [1] * n,
            "NoOfBids": n,
            "NoOfOffers": n,
            "UsingOrders": True,
        },
    }


GC_MAPPING = {
    "environment": "SAXO_SIM",
    "option_root_id": 60,
    "delta_tolerance": 0.01,
    "source": "fixture frozen source",
    "approval": "fixture explicit approval",
    "fee_per_side_gbp": 0.1,
    "fee_evidence": "fixture fee evidence",
}
