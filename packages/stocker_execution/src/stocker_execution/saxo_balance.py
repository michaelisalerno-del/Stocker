"""Account balance and SIM activity subscriptions: an optional, read-only account view."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from stocker_execution.saxo_auth import SaxoError

if TYPE_CHECKING:
    from stocker_execution.saxo_data import DataService

    pass


async def ensure_balance_subscription(data: DataService) -> None:
    """One optional account-scoped subscription; consumers only read the snapshot."""
    if not data.connected or not data.account_verified:
        return
    existing = next(
        ((ref, sub) for ref, sub in data.subscriptions.items() if sub["kind"] == "BALANCE"),
        None,
    )
    if (
        existing
        and existing[1]["arguments"].get("AccountKey") == data.client.oauth.account_key
        # Replacing a live subscription cannot fix what its own data reports.
        and data.balance_problem
        not in {"ACCOUNT_BALANCE_UNAVAILABLE", "BALANCE_SUBSCRIPTION_DISABLED"}
        and time.monotonic() - existing[1]["contact"] <= existing[1]["timeout"]
    ):
        return
    if time.monotonic() - data.balance_attempt < 60:
        return
    data.balance_attempt = time.monotonic()
    try:
        await data.subscribe(
            "BALANCE",
            {
                "AccountKey": data.client.oauth.account_key,
                "FieldGroups": ["CalculateCashForTrading"],
            },
            "BALANCE",
            old=existing[0] if existing else None,
        )
    except (SaxoError, ValueError, KeyError, TypeError):
        data.balance_problem = "ACCOUNT_BALANCE_UNAVAILABLE"


async def ensure_activity_subscription(data: DataService) -> None:
    """SAXO_SIM only: order/position events wake reconciliation; evidence stays audited."""
    if (
        data.config.execution_mode != "SAXO_SIM"
        or not data.connected
        or not data.account_verified
        or any(s["kind"] == "ACTIVITIES" for s in data.subscriptions.values())
        or time.monotonic() - data.activity_attempt < 60
    ):
        return
    data.activity_attempt = time.monotonic()
    try:
        await data.subscribe(
            "ACTIVITIES",
            {
                "AccountKey": data.client.oauth.account_key,
                "Activities": ["Orders", "Positions"],
            },
            "ACTIVITIES",
        )
        data.activity_problem = ""
    except (SaxoError, ValueError, KeyError, TypeError):
        data.activity_problem = "ACTIVITY_EVENTS_UNAVAILABLE"


def receive_balance(
    data: DataService, payload: Any, receipt: float, *, snapshot: bool = False
) -> None:
    if not isinstance(payload, dict):
        data.balance_problem = "BALANCE_SCHEMA_UNAVAILABLE"
        return
    if data.balance_account_key != data.client.oauth.account_key:
        data.balance.clear()
        data.balance_stream.clear()
        data.balance_received_at = None
    data.balance_account_key = data.client.oauth.account_key
    values = {} if snapshot else dict(data.balance_stream)
    # Do not expose account/client keys or retain an unbounded broker payload.
    for name in (
        "TotalValue",
        "CashBalance",
        "CashAvailableForTrading",
        "Currency",
        "CalculationReliability",
        "TransactionsNotBooked",
        "CostToClosePositions",
    ):
        if name in payload:
            value = payload[name]
            if name not in {"Currency", "CalculationReliability"}:
                value = (
                    value
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    )
                    else None
                )
            elif not isinstance(value, str):
                value = None
            values[name] = value
    data.balance_stream = values
    if values.get("CalculationReliability") != "Ok":
        data.balance_problem = "BALANCE_CALCULATION_UNVERIFIED"
        return
    if values.get("Currency") != data.account_currency:
        data.balance_problem = "BALANCE_CURRENCY_UNVERIFIED"
        return
    data.balance = values
    data.balance_received_at = receipt
    data.balance_problem = ""


def balance_view(data: DataService, at: float) -> dict[str, Any]:
    same_account = (
        data.balance_account_key is not None
        and data.balance_account_key == data.client.oauth.account_key
    )
    values = data.balance if same_account else {}
    sub = next((s for s in data.subscriptions.values() if s["kind"] == "BALANCE"), None)
    remaining = sub["timeout"] - (time.monotonic() - sub["contact"]) if sub else 0
    currency = values.get("Currency")
    available = any(
        values.get(k) is not None for k in ("TotalValue", "CashBalance", "CashAvailableForTrading")
    )
    fresh = bool(
        same_account
        and data.account_verified
        and data.connected
        and not data.balance_problem
        and remaining > 0
        and currency
        and currency == data.account_currency
    )
    return {
        "environment": "SIM" if data.config.data_environment == "SAXO_SIM" else "LIVE",
        "label": "SIM · simulated funds"
        if data.config.data_environment == "SAXO_SIM"
        else "LIVE · real-money account",
        "connection_note": "Real-money balances are not connected"
        if data.config.data_environment == "SAXO_SIM"
        else "Selected authenticated LIVE account · ordering disabled",
        "account": "••••" + data.account_id[-4:]
        if same_account and data.account_id
        else "Unavailable",
        "currency": currency or data.account_currency,
        "status": "Unavailable" if not available else "Current" if fresh else "Stale",
        "last_success_at": data.balance_received_at if same_account else None,
        "valid_until": at + remaining if fresh else None,
        "total_value": values.get("TotalValue"),
        "cash_balance": values.get("CashBalance"),
        "cash_available_for_trading": values.get("CashAvailableForTrading"),
        "problem": data.balance_problem,
        "details": {
            k: values.get(k)
            for k in ("CalculationReliability", "TransactionsNotBooked", "CostToClosePositions")
        },
        "basis": "Broker-reported native currency; no conversion or strategy P&L added",
    }
