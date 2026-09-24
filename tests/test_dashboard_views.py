"""Synthetic read-only dashboard projections, separate from frozen trading fixtures."""

import asyncio
import json
import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from stocker_dashboard.app import create_dashboard_app
from stocker_dashboard.views import allocations, economics, opportunities
from stocker_execution.first4 import Q5
from stocker_execution.first4_config import First4Config
from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store

DAY = "2026-09-24"
CLOCK = datetime(2026, 9, 24, 14, tzinfo=UTC)


def test_prior15_explanation_is_only_for_early_first_appearance(runtime):
    early = CLOCK.replace(hour=13, minute=31)
    event(runtime, "EARLY", prior=None, clock=early)
    event(runtime, "LATE", prior=None, clock=CLOCK)
    rows = {r["symbol"]: r for r in opportunities(runtime.store, DAY, 50, 0, "all")}
    assert (
        rows["EARLY"]["prior15_explanation"]
        == "PRIOR15 unavailable at first appearance — fewer than 15 session minutes"
    )
    assert rows["LATE"]["prior15_explanation"] == "PRIOR15 unavailable"
    assert rows["EARLY"]["decision_explanation"] == "Rejected for this session — not reconsidered"
    with runtime.store.db:
        runtime.store.db.execute(
            "UPDATE first4_events SET decision='CHANGE_BELOW_5_5' WHERE symbol='EARLY'"
        )
    row = next(r for r in opportunities(runtime.store, DAY, 50, 0, "all") if r["symbol"] == "EARLY")
    assert row["decision"] == "CHANGE_BELOW_5_5"


def test_execution_failure_without_fill_has_no_position_label(runtime):
    selected = event(runtime)
    runtime.store.outcome(
        selected, "EXECUTION_FAILED", {"error": "OPTION_QUOTES_INVALID_STALE_OR_UNAVAILABLE"}
    )
    row = allocations(runtime.store, DAY)[0]
    assert row["position_explanation"] == "No position opened"
    assert not row["has_exposure"] and row["state"] == "BLOCKED / FAILED"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Dashboard must never contact a broker")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    value = Runtime(First4Config(), Store(tmp_path / "synthetic.sqlite"))
    value.session = DAY
    yield value
    value.store.db.close()


def get(runtime, path):
    async def read():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            return await client.get(path)

    return asyncio.run(read())


def event(runtime, symbol="AAA", prior=5, rank=1, clock=CLOCK, day=DAY):
    runtime.store.observe(
        day,
        clock,
        clock + timedelta(hours=6),
        [
            {
                "symbol": symbol,
                "con_id": rank,
                "rank": rank,
                "price": 10,
                "change_pct": 8,
                "prior15": prior,
                "detail": {"diagnostic": "x" * 10000},
            }
        ],
    )
    return runtime.store.event(day, symbol)


def order(runtime, admission, role="ENTRY", suffix="", **updates):
    payload = {
        "put": {"conId": 101, "right": "P", "strike": 9.8},
        "call": {"conId": 102, "right": "C", "strike": 10.2},
        "quantity": 1,
        "multiplier": 100,
        "quoted_entry_ask_usd": 9999,
        "exit_at": "2026-09-24T19:59:00+00:00",
        **updates,
    }
    return runtime.store.reserve_order(admission, role, 1, payload, suffix)


def fill(
    runtime,
    reference,
    execution,
    con=101,
    side="BOT",
    quantity=1,
    price=1,
    commission=0.65,
    superseded=0,
):
    runtime.store.db.execute(
        "INSERT INTO first4_fills VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            execution,
            reference,
            con,
            quantity,
            price,
            side,
            100,
            "2026-09-24T14:01:00+00:00",
            commission,
            superseded,
        ),
    )


def test_four_permanent_slots_and_no_live_snapshot_claim(runtime):
    admission = event(runtime)
    event(runtime, "BELOW", Q5, clock=CLOCK + timedelta(minutes=1))
    runtime.store.outcome(admission, "UNARMED")
    data = get(runtime, "/api/overview").json()
    assert [s["slot"] for s in data["slots"]] == [1, 2, 3, 4]
    assert data["slots"][0]["state"] == "BLOCKED / FAILED"
    assert data["slots"][0]["actual_premium_paid"] is None
    assert data["slots"][0]["realised"] is None
    assert all(s["state"] == "WAITING" for s in data["slots"][1:])
    assert data["q5"] == Q5
    assert data["candidates"][0]["decision"] == "NOT_Q5"
    assert data["candidates"][0]["basis"] == "FIRST_APPEARANCE_SNAPSHOT_FINAL_DECISION"
    assert "payload" not in json.dumps(data) and "diagnostic" not in json.dumps(data)
    assert "settings" not in data["system"]
    assert not {"orders", "fills", "positions", "quote_comparisons"} & data.keys()


def test_partial_individual_legs_closed_pending_and_late_fees(runtime):
    admission = event(runtime)
    assert allocations(runtime.store, DAY)[0]["state"] == "SELECTED"
    entry = order(runtime, admission)
    assert allocations(runtime.store, DAY)[0]["state"] == "ENTRY PENDING"
    fill(runtime, entry, "buy-put", commission=None)
    item = allocations(runtime.store, DAY)[0]
    assert item["state"] == "PARTIALLY FILLED" and item["actual_premium_paid"] == 100
    assert item["unrealised"] is None and item["realised"] is None
    fill(runtime, entry, "buy-call", con=102, commission=None)
    assert allocations(runtime.store, DAY)[0]["state"] == "OPEN"
    exit_ref = order(runtime, admission, "EXIT", "101")
    fill(runtime, exit_ref, "sell-put-half", side="SLD", quantity=0.5, price=1.5, commission=None)
    item = allocations(runtime.store, DAY)[0]
    assert item["state"] == "EXIT PENDING"
    assert [leg["remaining"] for leg in item["legs"]] == [0.5, 1]
    assert economics([item])["partial_close_gross_usd"] == 25
    fill(runtime, exit_ref, "sell-put-rest", side="SLD", quantity=0.5, price=1.5, commission=None)
    fill(runtime, exit_ref, "sell-call", con=102, side="SLD", price=1.2, commission=None)
    assert allocations(runtime.store, DAY)[0]["state"] == "RECONCILIATION PENDING"
    runtime.store.db.execute("UPDATE first4_orders SET obligation_done=1 WHERE role='ENTRY'")
    item = allocations(runtime.store, DAY)[0]
    assert item["state"] == "CLOSED" and item["realised"] is None
    assert item["provisional_result"] == 70 and item["return_pct"] == 35
    runtime.store.db.execute("UPDATE first4_fills SET commission=0.65")
    item = allocations(runtime.store, DAY)[0]
    assert item["realised"] == pytest.approx(66.75)
    assert item["return_pct"] == pytest.approx(33.375)
    assert item["return_basis"] == "NET_RESULT_ON_ACTUAL_PREMIUM_PAID"
    assert item["slot"] == 1 and item["symbol"] == "AAA"


def test_accounting_complete_beyond_evidence_pages_and_corrections(runtime):
    admission = event(runtime)
    reference = order(runtime, admission)
    for i in range(250):
        fill(runtime, reference, str(i), quantity=0.004, commission=0.01)
    fill(runtime, reference, "superseded", quantity=9999, superseded=1)
    item = allocations(runtime.store, DAY)[0]
    assert item["actual_premium_paid"] == pytest.approx(100)
    assert item["fees"] == pytest.approx(2.5)
    first = get(runtime, f"/api/detail?session={DAY}&symbol=AAA&limit=100").json()
    second = get(runtime, f"/api/detail?session={DAY}&symbol=AAA&limit=100&offset=100").json()
    assert len(first["fills"]) == len(second["fills"]) == 100
    assert not {f["exec_id"] for f in first["fills"]} & {f["exec_id"] for f in second["fills"]}
    assert allocations(runtime.store, DAY)[0] == item


def test_ordering_pagination_filters_and_scoped_linkage(runtime):
    for i, symbol in enumerate(["BBB", "AAA", "CCC"]):
        event(runtime, symbol, prior=4, clock=CLOCK + timedelta(minutes=i))
    runtime.store.db.execute(
        "UPDATE first4_events SET information_at=?,rank=1", (CLOCK.isoformat(),)
    )
    rows = opportunities(runtime.store, DAY, 2, 0, "all")
    assert [r["symbol"] for r in rows] == ["AAA", "BBB"]
    assert [r["symbol"] for r in opportunities(runtime.store, DAY, 2, 2, "all")] == ["CCC"]
    event(runtime, "AAA", day="2026-09-23", clock=CLOCK + timedelta(days=-1))
    historical = get(runtime, "/api/opportunities?session=2026-09-23").json()
    assert len(historical["rows"]) == 1 and historical["rows"][0]["slot"] == 1
    assert get(runtime, "/api/opportunities?decision=selected").json()["rows"] == []
    assert len(get(runtime, "/api/opportunities?decision=rejected").json()["rows"]) == 3
    assert get(runtime, "/api/opportunities?limit=201").status_code == 422
    assert get(runtime, "/api/opportunities?offset=-1").status_code == 422
    assert get(runtime, "/api/detail?session=2026-09-24&symbol=missing").status_code == 404
    assert get(runtime, "/api/overview").json()["pnl"]["allocations_used"] == 0


def test_same_contract_different_allocations_do_not_mix(runtime):
    first = event(runtime)
    second = event(runtime, "BBB", clock=CLOCK + timedelta(minutes=1))
    fill(runtime, order(runtime, first), "first", price=1)
    fill(runtime, order(runtime, second), "second", price=2)
    items = allocations(runtime.store, DAY)
    assert [i["actual_premium_paid"] for i in items] == [100, 200]
    assert economics(items)["exposed_allocations"] == 2
    assert len(economics(items)["open_owned_legs"]) == 2


def test_unfilled_failed_cap_and_zero_denominator(runtime):
    for i in range(5):
        admission = event(runtime, str(i), clock=CLOCK + timedelta(minutes=i))
        if i < 4:
            reference = order(runtime, admission)
            runtime.store.db.execute(
                "UPDATE first4_orders SET status='Cancelled',obligation_done=1 WHERE reference=?",
                (reference,),
            )
    data = get(runtime, "/api/overview").json()
    assert all(s["state"] == "UNFILLED" and s["return_pct"] is None for s in data["slots"])
    assert data["candidates"][0]["decision"] == "DAILY_CAP"
    assert data["candidates"][0]["slot"] is None
    assert all(s["symbol"] != "4" for s in data["slots"])


def test_queries_no_full_history_or_duplicate_fill_reads(runtime):
    event(runtime)
    queries = []
    runtime.store.db.set_trace_callback(queries.append)
    response = get(runtime, "/api/overview")
    runtime.store.db.set_trace_callback(None)
    assert response.status_code == 200 and len(response.content) < 8000
    assert sum("JOIN first4_effective_fills" in sql for sql in queries) == 1
    assert not any(sql == "SELECT * FROM first4_orders" for sql in queries)
    assert not any(sql == "SELECT * FROM first4_fills" for sql in queries)
    assert runtime.store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert runtime.store.db.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_rebrand_redirects_and_no_old_runtime_pages(runtime):
    assert "SLRNO" in get(runtime, "/").text
    assert get(runtime, "/orders").headers["location"] == "/execution"
    assert get(runtime, "/settings").headers["location"] == "/system"
    assert get(runtime, "/api/orders").status_code == 404
    assert get(runtime, "/api/settings").status_code == 404
