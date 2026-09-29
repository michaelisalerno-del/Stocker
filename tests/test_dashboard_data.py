"""Display data (gates, setup, timeline, book history), alerts and the event calendar."""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from saxo_support import GC_MAPPING, IDENTITY, book, plan, setup, signal
from stocker_dashboard.app import create_dashboard_app
from stocker_execution import views
from stocker_execution.alerts import Alerts
from stocker_execution.config import AlertSettings, FuturesConfig
from stocker_execution.event_calendar import EventCalendar, load_calendar
from stocker_execution.recorder import Recorder
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store


def runtime_for(tmp_path, **config):
    return Runtime(FuturesConfig(**config), Store(tmp_path / "ledger.sqlite3"))


def test_gates_follow_the_decision_order_and_explain_the_first_block(tmp_path):
    runtime = runtime_for(tmp_path)
    gates = runtime.overview()["markets"][1]["gates"]
    assert [g["key"] for g in gates] == [
        "saxo",
        "contract",
        "quote",
        "fx",
        "history",
        "approval",
        "strike",
        "cost",
        "execution",
    ]
    assert not any(g["ok"] for g in gates)
    by_key = {g["key"]: g for g in gates}
    assert by_key["approval"]["detail"] == "GC_MONITOR_ONLY_UNTIL_APPROVED"
    assert by_key["execution"]["detail"] == "EXECUTION_DISABLED"
    assert by_key["strike"]["detail"] == "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED"
    assert by_key["cost"]["detail"] == "NO_CANDIDATE"
    approved = runtime_for(tmp_path / "approved", mappings={"GC": GC_MAPPING})
    gc = {g["key"]: g for g in approved.market_detail("GC")["markets"][0]["gates"]}
    assert gc["approval"]["ok"] and gc["approval"]["detail"] == ""
    runtime.store.db.close()
    approved.store.db.close()


def test_setup_checklist_and_server_clock_are_in_every_status(tmp_path):
    runtime = runtime_for(tmp_path)
    status = runtime.status()
    items = {i["key"]: i for i in status["setup"]}
    assert items["contracts"]["label"] == "Futures contracts verified 0/5"
    assert not items["contracts"]["done"] and not items["mode"]["done"]
    assert items["recording"]["optional"] and items["alerts"]["optional"]
    assert items["references"]["detail"] == "Set reference_selections_file"
    assert abs(status["server_time"] - datetime.now(UTC).timestamp()) < 5
    assert datetime.fromisoformat(status["next_clock"]) > datetime.now(UTC)
    runtime.store.db.close()


def test_opportunity_timeline_orders_clock_checks_admission_orders_and_fill(tmp_path):
    async def scenario():
        broker, _, store = setup(tmp_path)
        event = {**signal(1), "skip_reason": ""}
        store.observe(event, "", {"rv15": 0.0012, "futures_price": 70.5})
        with store.db:
            row = store.db.execute(
                "SELECT detail FROM signals WHERE id=?", (event["id"],)
            ).fetchone()
            detail = {
                **json.loads(row[0]),
                "option_context": {
                    "selection_status": "SELECTED",
                    "identity": {"right": "Call", "strike": 70, "uic": 101},
                },
            }
            store.db.execute(
                "UPDATE signals SET detail=? WHERE id=?", (json.dumps(detail), event["id"])
            )
        assert await broker.enter(event, plan()) == ""
        signal_row = dict(
            store.db.execute("SELECT * FROM signals WHERE id=?", (event["id"],)).fetchone()
        )
        lifecycle = [
            dict(r)
            for r in store.db.execute(
                "SELECT * FROM lifecycle WHERE reference=? OR reference IN "
                "(SELECT reference FROM orders WHERE event_id=?)",
                (event["id"], event["id"]),
            )
        ]
        steps = views.timeline(signal_row, lifecycle, store.fills(event["id"]))
        kinds = [s["kind"] for s in steps]
        assert kinds[:3] == ["CLOCK", "CHECKS", "OPTION"]
        assert (
            kinds.index("ADMISSION")
            < kinds.index("DURABLE_SUBMISSION_INTENT")
            < kinds.index("INTERNAL_SIMULATED_FILL")
        )
        assert kinds[-1] == "OUTCOME"
        assert "rv15 0.0012" in steps[1]["detail"] and "Call 70" in steps[2]["detail"]
        assert "1 fill(s)" in steps[-1]["detail"]
        store.db.close()

    asyncio.run(scenario())


def test_book_series_buckets_recorded_depth_without_inferring_anything(tmp_path):
    recorder = Recorder(FuturesConfig().recorder, tmp_path / "events")
    recorder.register("future", IDENTITY)
    recorder.ingest("future", "SNAPSHOT", book(), 1000)
    recorder.ingest("future", "UPDATE", {"MarketDepth": {"BidSize": [9] * 10}}, 1001)
    recorder.ingest("future", "UPDATE", {"MarketDepth": {"AskSize": [4] * 10}}, 1007)
    sequence, tick, blobs = views.book_rows(recorder, "future")
    result = views.book_series(sequence, tick, blobs, "future")
    assert result["tick_size"] == 0.01 and [r["at"] for r in result["series"]] == [1000, 1005]
    first, second = result["series"]
    assert first["bid"][0] == [7000, 9] and first["ask"][0] == [7001, 1]  # last obs in bucket
    assert second["ask"][0] == [7001, 4] and second["spread_ticks"] == 1
    assert views.book_series(sequence, tick, blobs, "future") is result  # cached until new rows
    assert views.book_rows(recorder, "missing") == (0, None, [])


def test_book_endpoint_reports_missing_contract_and_retained_rows(tmp_path):
    async def scenario():
        runtime = runtime_for(tmp_path)
        app = create_dashboard_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            assert (await client.get("/api/market/CL/book")).json()["status"] == (
                "CONTRACT_NOT_VERIFIED"
            )
            assert (await client.get("/api/market/BTC/book")).status_code == 422
            runtime.markets["CL"].identity = IDENTITY
            key = "SAXO:SAXO_SIM:ContractFutures:123"
            runtime.recorder.register(key, IDENTITY)
            runtime.recorder.ingest(key, "SNAPSHOT", book(), 2000)
            data = (await client.get("/api/market/CL/book")).json()
            assert data["status"] == "AVAILABLE" and len(data["series"]) == 1
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def private_url(tmp_path, url):
    path = tmp_path / "alert.json"
    path.write_text(json.dumps({"url": url}))
    os.chmod(path, 0o600)
    return path


def test_alerts_send_new_and_resolved_conditions_once_and_never_raise(tmp_path):
    sent = []

    def handler(request):
        sent.append((request.headers["Title"], request.content.decode()))
        return httpx.Response(200 if len(sent) < 3 else 500)

    async def scenario():
        runtime = runtime_for(tmp_path)
        settings = AlertSettings(url_file=private_url(tmp_path, "https://ntfy.example/slrno"))
        alerts = Alerts(settings, transport=httpx.MockTransport(handler))
        assert alerts.enabled and not alerts.problem
        runtime.data.connected = True
        runtime.broker.management_problems["CL-1"] = "FAILED_CLOSURE_REQUIRES_OPERATOR"
        await alerts.check(runtime, at=1000)
        await alerts.check(runtime, at=1010)  # unchanged: no repeat
        runtime.broker.management_problems.clear()
        await alerts.check(runtime, at=1020)
        assert sent == [
            ("SLRNO alert", "Exposure exception CL-1: FAILED_CLOSURE_REQUIRES_OPERATOR"),
            ("SLRNO resolved", "Exposure exception CL-1: FAILED_CLOSURE_REQUIRES_OPERATOR"),
        ]
        runtime.data.connected = False
        await alerts.check(runtime, at=2000)
        await alerts.check(runtime, at=2000 + 119)
        assert len(sent) == 2
        await alerts.check(runtime, at=2000 + 121)
        assert sent[-1][1].startswith("Saxo stream disconnected")
        assert alerts.last_error == "ALERT_HTTP_500"
        await alerts.close()
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_alerts_warn_before_saxo_login_expires_and_reject_unsafe_urls(tmp_path):
    runtime = runtime_for(tmp_path)
    runtime.data.client.oauth.credentials = {"client_id": "fixture"}
    runtime.data.client.oauth.status = "AUTHENTICATED"
    runtime.data.connected = True
    alerts = Alerts(AlertSettings())
    # A healthy rotating Saxo session always has 40-60 minutes of refresh lifetime left.
    for remaining in (41 * 60, 60 * 60):
        runtime.data.client.oauth.tokens = {"refresh_expires_at": 1000 + remaining}
        assert "login" not in alerts.conditions(runtime, at=1000)
    runtime.data.client.oauth.tokens = {"refresh_expires_at": 1000 + 14 * 60}
    first = alerts.conditions(runtime, at=1000)["login"]
    runtime.data.client.oauth.tokens = {"refresh_expires_at": 1000 + 10 * 60}
    assert alerts.conditions(runtime, at=1000)["login"] == first  # stable text, no repeats
    assert first.startswith("Saxo login renewal has stopped") and not alerts.enabled
    insecure = Alerts(AlertSettings(url_file=private_url(tmp_path, "http://ntfy.example/x")))
    assert insecure.problem == "ALERT_URL_FILE_INVALID" and not insecure.enabled
    os.chmod(tmp_path / "alert.json", 0o644)
    exposed = Alerts(AlertSettings(url_file=tmp_path / "alert.json"))
    assert exposed.problem == "ALERT_URL_FILE_INVALID"
    runtime.store.db.close()


def test_event_calendar_flags_releases_around_clocks(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text(
        "events:\n"
        "  - {name: EIA Weekly Petroleum Status, markets: [CL],\n"
        "     weekly: {weekday: WED, time: '10:30'}}\n"
        "  - {name: US CPI, markets: [NQ, GC], at: '2026-10-14T08:30:00-04:00'}\n"
    )
    calendar = load_calendar(path)
    wednesday = datetime(2026, 9, 30, 14, tzinfo=UTC)  # 10:00 New York
    assert calendar.near("CL", wednesday) == [
        {
            "name": "EIA Weekly Petroleum Status",
            "at": "2026-09-30T10:30:00-04:00",
            "relation": "DURING_HOLDING_WINDOW",
        }
    ]
    assert calendar.near("CL", wednesday + timedelta(hours=1))[0]["relation"] == "HOUR_BEFORE_CLOCK"
    assert calendar.near("CL", wednesday + timedelta(hours=2)) == []
    assert calendar.near("NG", wednesday) == []
    assert [n for _, n in calendar.occurrences("GC", datetime(2026, 10, 14).date())] == ["US CPI"]
    with pytest.raises(ValueError):
        EventCalendar.model_validate(
            {
                "events": [
                    {
                        "name": "Both",
                        "markets": ["CL"],
                        "at": "2026-10-14T08:30:00-04:00",
                        "weekly": {"weekday": "WED", "time": "10:30"},
                    }
                ]
            }
        )


def test_runtime_reports_calendar_on_cards_and_tolerates_a_bad_file(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text(
        "events:\n  - {name: Mon, markets: [NG], weekly: {weekday: MON, time: '09:30'}}\n"
        "  - {name: Tue, markets: [NG], weekly: {weekday: TUE, time: '09:30'}}\n"
        "  - {name: Wed, markets: [NG], weekly: {weekday: WED, time: '09:30'}}\n"
        "  - {name: Thu, markets: [NG], weekly: {weekday: THU, time: '09:30'}}\n"
        "  - {name: Fri, markets: [NG], weekly: {weekday: FRI, time: '09:30'}}\n"
    )
    runtime = runtime_for(tmp_path, event_calendar_file=path)
    assert runtime.calendar is not None and not runtime.calendar_problem
    detail = runtime.market_detail("NG")["markets"][0]
    today = datetime.now(UTC).astimezone(views.NY).weekday()
    assert len(detail["events_today"]) == (1 if today < 5 else 0)
    runtime.store.db.close()
    path.write_text("events: [{name: x}]\n")
    broken = runtime_for(tmp_path / "broken", event_calendar_file=path)
    assert broken.calendar is None and broken.calendar_problem == "EVENT_CALENDAR_UNREADABLE"
    assert broken.status()["calendar_problem"] == "EVENT_CALENDAR_UNREADABLE"
    broken.store.db.close()
