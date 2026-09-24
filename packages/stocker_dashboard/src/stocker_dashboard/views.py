"""Read-only FIRST4 projections. Display limits never limit fill accounting."""

import json
import math
from typing import Any

from stocker_execution.first4 import Q5
from stocker_execution.first4_store import Store

EVENT_COLUMNS = "session,symbol,information_at,rank,prior15,decision,slot,entry_at,outcome"
TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def opportunities(
    store: Store, session: str, limit: int, offset: int, decision: str, top: bool = False
) -> list[dict[str, Any]]:
    condition = {"all": "", "selected": " AND slot IS NOT NULL", "rejected": " AND slot IS NULL"}[
        decision
    ]
    ordering = "information_at DESC,rank,symbol"
    params: list[Any] = [session]
    if top:
        # Presentation ranking only. These are final first-appearance decisions.
        condition = " AND slot IS NULL AND prior15 IS NOT NULL"
        ordering = "abs(prior15-?),rank,information_at,symbol"
        params.append(Q5)
    params.extend([min(200, max(1, limit)), max(0, offset)])
    rows = [
        dict(r)
        for r in store.db.execute(
            f"SELECT {EVENT_COLUMNS} FROM first4_events WHERE session=?{condition} "
            f"ORDER BY {ordering} LIMIT ? OFFSET ?",
            params,
        )
    ]
    for row in rows:
        value = row["prior15"]
        if value is not None and not math.isfinite(value):
            row["prior15"] = None
        row["basis"] = "FIRST_APPEARANCE_SNAPSHOT_FINAL_DECISION"
    return rows


def allocations(store: Store, session: str) -> list[dict[str, Any]]:
    events = [
        dict(r)
        for r in store.db.execute(
            f"SELECT {EVENT_COLUMNS},json_extract(detail,'$.error') blocker,"
            "json_extract(detail,'$.price') anchor,"
            "json_extract(detail,'$.broker_trade_time') anchor_observed_at "
            "FROM first4_events WHERE session=? AND slot IS NOT NULL ORDER BY slot LIMIT 4",
            (session,),
        )
    ]
    # Four permanent entry reservations at most. Exit orders are aggregated in SQL.
    entries = {
        r["symbol"]: dict(r)
        for r in store.db.execute(
            "SELECT symbol,reference,status,obligation_done,payload FROM first4_orders "
            "WHERE session=? AND role='ENTRY' ORDER BY symbol,reference",
            (session,),
        )
    }
    exits = {
        r["symbol"]: dict(r)
        for r in store.db.execute(
            "SELECT symbol,count(*) orders,"
            "sum(status NOT IN ('Filled','Cancelled','ApiCancelled','Inactive')) working "
            "FROM first4_orders WHERE session=? AND role='EXIT' GROUP BY symbol",
            (session,),
        )
    }
    # Complete effective executions, including individual-leg exits and corrections.
    legs = [
        dict(r)
        for r in store.db.execute(
            "SELECT o.symbol,f.con_id,"
            "SUM(CASE side WHEN 'BOT' THEN quantity ELSE 0 END) bought,"
            "SUM(CASE side WHEN 'SLD' THEN quantity ELSE 0 END) sold,"
            "SUM(CASE side WHEN 'BOT' THEN quantity*price*multiplier ELSE 0 END) cost,"
            "SUM(CASE side WHEN 'SLD' THEN quantity*price*multiplier ELSE 0 END) proceeds,"
            "SUM(COALESCE(commission,0)) fees,SUM(commission IS NULL) pending,"
            "MIN(CASE side WHEN 'BOT' THEN time END) entry_time,"
            "MAX(CASE side WHEN 'SLD' THEN time END) exit_time "
            "FROM first4_orders o JOIN first4_effective_fills f USING(reference) "
            "WHERE o.session=? GROUP BY o.symbol,f.con_id ORDER BY o.symbol,f.con_id",
            (session,),
        )
    ]
    result = []
    for event in events:
        entry = entries.get(event["symbol"], {})
        payload = json.loads(entry.get("payload", "{}"))
        held = [r for r in legs if r["symbol"] == event["symbol"]]
        for leg in held:
            leg["remaining"] = leg["bought"] - leg["sold"]
        paid = sum(r["cost"] for r in held)
        proceeds = sum(r["proceeds"] for r in held)
        fees = sum(r["fees"] for r in held)
        pending = sum(r["pending"] for r in held)
        exposure = any(r["remaining"] != 0 for r in held)
        complete = bool(entry.get("obligation_done")) and not exposure
        closed = complete and bool(held)
        exit_orders = exits.get(event["symbol"], {})
        contracts = [payload[k] for k in ("put", "call") if k in payload]
        expected = {c["conId"] for c in contracts}
        full = bool(expected) and all(
            any(r["con_id"] == con_id and r["bought"] >= payload.get("quantity", 1) for r in held)
            for con_id in expected
        )
        if exposure:
            state = (
                "EXIT PENDING"
                if exit_orders or event["outcome"].startswith("EXIT")
                else ("OPEN" if full else "PARTIALLY FILLED")
            )
        elif closed:
            state = "CLOSED"
        elif held or entry.get("status") == "Filled":
            state = "RECONCILIATION PENDING"
        elif event["outcome"] == "ENTRY_UNFILLED" or complete:
            state = "UNFILLED"
        elif (
            event["outcome"] in {"UNARMED", "EXECUTION_FAILED"} or entry.get("status") == "Inactive"
        ):
            state = "BLOCKED / FAILED"
        elif entry:
            state = "UNFILLED" if entry["status"] in TERMINAL else "ENTRY PENDING"
        else:
            state = "SELECTED"
        net = proceeds - paid - fees
        result.append(
            {
                **event,
                "state": state,
                "entry_status": entry.get("status"),
                "entry_reference": entry.get("reference"),
                "legs": held,
                "contracts": contracts,
                "actual_premium_paid": paid if held else None,
                "proceeds": proceeds,
                "fees": fees,
                "pending_fees": pending,
                "fees_complete": pending == 0,
                "complete": complete,
                "closed": closed,
                "has_exposure": exposure,
                "gross": proceeds - paid if closed else None,
                "realised": net if closed and not pending else None,
                "provisional_result": net if closed else None,
                "return_pct": 100 * net / paid if closed and paid > 0 else None,
                "return_basis": "NET_RESULT_ON_ACTUAL_PREMIUM_PAID",
                "unrealised": None,
                "valuation_basis": "UNAVAILABLE",
                "cash_flow": net,
                "basis": "IBKR_PAPER_SIMULATED_FILLS",
                "actual_entry_at": min(
                    (r["entry_time"] for r in held if r["entry_time"]), default=None
                ),
                "actual_exit_at": max(
                    (r["exit_time"] for r in held if r["exit_time"]), default=None
                ),
                "intended_exit_at": payload.get("exit_at"),
                "anchor": event["anchor"] or payload.get("anchor"),
                "fee_reserve": payload.get("fee_reserve_usd", 10) if entry else 0,
                "allocation_usd": payload.get("allocation_usd", 260) if entry else 0,
                "next_step": event["blocker"]
                or (
                    event["outcome"]
                    if state == "BLOCKED / FAILED"
                    else "Awaiting fill"
                    if state == "ENTRY PENDING"
                    else "Unrealised P&L unavailable"
                    if exposure
                    else "Fees pending"
                    if closed and pending
                    else "Completion verified"
                    if closed
                    else "No filled exposure"
                ),
            }
        )
    return result


def economics(items: list[dict[str, Any]]) -> dict[str, Any]:
    held = [leg for item in items for leg in item["legs"]]
    completed = [item for item in items if item["closed"]]
    known = [item for item in completed if item["fees_complete"]]
    pending = sum(item["pending_fees"] for item in items)
    return {
        "basis": "IBKR_PAPER_SIMULATED_FILLS",
        "currency": "USD",
        "actual_fees_usd": sum(item["fees"] for item in items),
        "fees_complete": pending == 0,
        "pending_fee_executions": pending,
        "completed_gross_usd": sum(item["gross"] for item in completed),
        "completed_pending_fee_allocations": len(completed) - len(known),
        "realised": sum(item["realised"] for item in known) if known else (None if held else 0),
        "realised_basis": "COMPLETED_ALLOCATIONS_WITH_KNOWN_FEES",
        "partial_close_gross_usd": sum(
            leg["proceeds"] - leg["cost"] * leg["sold"] / leg["bought"]
            for item in items
            if not item["closed"]
            for leg in item["legs"]
            if 0 <= leg["sold"] <= leg["bought"] and leg["bought"] > 0
        ),
        "open_owned_legs": [
            {"symbol": r["symbol"], "con_id": r["con_id"], "quantity": r["remaining"]}
            for r in held
            if r["remaining"] != 0
        ],
        "gross_cash_flow": sum(r["proceeds"] - r["cost"] for r in held),
        "net_cash_flow": sum(item["cash_flow"] for item in items),
        "net_cash_flow_basis": "KNOWN_FEES_ONLY" if pending else "ALL_REPORTED_FEES",
        "reserved_fee_allowance_usd": sum(item["fee_reserve"] for item in items),
        "session_allocation_usd": sum(item["allocation_usd"] for item in items),
        "allocations_used": len(items),
        "exposed_allocations": sum(i["has_exposure"] for i in items),
        "unrealised": None,
    }
