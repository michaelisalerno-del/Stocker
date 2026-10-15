"""A whole paper day against an offline Saxo: entries, stops, delayed QQQ reads and exits."""

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from fake_saxo import NAS, SPX, FakeSaxo, client, regular_sessions, trend_bars
from stocker_execution import runtime as runtime_module
from stocker_execution import saxo_client
from stocker_execution.config import PaperConfig
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store

MON, TUE, WED = date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)
OPEN = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)  # 09:30 New York (EDT)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(saxo_client, "REQUEST_SPACING", 0)
    monkeypatch.setattr(runtime_module, "RETRY_EVERY", 0)
    monkeypatch.setattr(runtime_module, "STOP_POLL_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "LOGIN_CHECK_SECONDS", 0)


class Clock:
    def __init__(self, at: datetime):
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def market(fake: FakeSaxo) -> None:
    fake.sessions = regular_sessions(MON, TUE, WED)
    fake.bars[NAS] = trend_bars(OPEN, 29900, 2.0)  # rising into the open -> long
    fake.bars[SPX] = trend_bars(OPEN, 7760, -1.0)  # falling -> short
    fake.cfd_quote(NAS, 29999.0, 30000.0)
    fake.cfd_quote(SPX, 7700.0, 7700.5)
    fake.qqq(699.98, 700.02)
    strikes = [690.0 + i for i in range(21)]
    fake.chain(MON, strikes, 700.0, 100000)
    fake.chain(TUE, strikes, 700.0, 200000)
    # Rising daily closes, then a down day: with a lower 15:50 price, still above the 200-day
    # average and RSI2(today) + RSI2(yesterday) < 35.
    fake.daily = [
        {
            "Time": (datetime(2025, 10, 1, tzinfo=UTC) + timedelta(days=i)).strftime(
                "%Y-%m-%dT00:00:00Z"
            ),
            "Close": 600 + 0.5 * i,
        }
        for i in range(248)
    ]
    fake.daily += [{"Time": "2026-10-16T00:00:00Z", "Close": 718.0}]


def make(tmp_path, fake, clock, store=None):
    return Runtime(
        PaperConfig(), store or Store(tmp_path / "paper.sqlite3"), client(tmp_path, fake), clock
    )


async def at(rt: Runtime, clock: Clock, when: datetime, passes: int = 2) -> None:
    clock.at = when
    for _ in range(passes):
        await rt.step()


def test_a_full_paper_day_and_the_next_day_connors_exit(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN - timedelta(minutes=10))
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        db = rt.store.db
        await at(rt, clock, OPEN + timedelta(minutes=4, seconds=59))
        assert db.execute("SELECT COUNT(*) FROM cfd_trades").fetchone()[0] == 0

        # 09:35: both CFD signals; trades sized for $200 at the live bid/ask.
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        trades = {r["market"]: dict(r) for r in db.execute("SELECT * FROM cfd_trades")}
        assert (trades["NAS100"]["side"], trades["NAS100"]["entry_price"]) == (1, 30000.0)
        assert (trades["NAS100"]["amount"], trades["NAS100"]["stop"]) == (0.66, 29700.0)
        assert (trades["US500"]["side"], trades["US500"]["entry_price"]) == (-1, 7700.0)
        assert (trades["US500"]["amount"], trades["US500"]["stop"]) == (5.1, 7738.5)
        # The QQQ option waits for the 15-minute delay: the 09:35 price is read at 09:50.
        entry = rt.store.event("2026-10-19", "FIRST_CANDLE_QQQ_ENTRY")
        assert entry["status"] == "WAITING" and entry["effective"].startswith("2026-10-19T13:50")
        assert rt.store.option("2026-10-19", "FIRST_CANDLE_QQQ") is None

        # 09:40: the S&P short is stopped out at the ask that crossed its stop.
        fake.cfd_quote(SPX, 7739.0, 7740.0)
        await at(rt, clock, OPEN + timedelta(minutes=10))
        spx = db.execute("SELECT * FROM cfd_trades WHERE market='US500'").fetchone()
        assert (spx["exit_reason"], spx["exit_price"]) == ("STOP", 7740.0)
        assert spx["pnl_usd"] == pytest.approx(-5.1 * 40)

        # 09:50: the call with delta nearest 0.40 at its ask, recorded as of 09:35.
        await at(rt, clock, OPEN + timedelta(minutes=20, seconds=1))
        opt = rt.store.option("2026-10-19", "FIRST_CANDLE_QQQ")
        assert (opt["put_call"], opt["strike"], opt["entry_delta"]) == ("Call", 701.0, 0.42)
        assert opt["entry_for"].startswith("2026-10-19T13:35:01")
        assert opt["entry_ask"] == pytest.approx(0.86)

        # 13:00 sale, read at 13:15.
        fake.option_quotes[opt["uic"]].update(Bid=1.50, Ask=1.54)
        await at(rt, clock, datetime(2026, 10, 19, 17, 15, 1, tzinfo=UTC))
        opt = rt.store.option("2026-10-19", "FIRST_CANDLE_QQQ")
        assert opt["exit_bid"] == 1.50 and opt["exit_reason"] == "AT_13_00"
        assert opt["pnl_usd"] == pytest.approx((1.50 - 0.86) * 100 - 4.08)

        # 15:50: the Connors check measures the QQQ delay and waits until 16:05.
        await at(rt, clock, datetime(2026, 10, 19, 19, 50, 1, tzinfo=UTC))
        assert rt.store.event("2026-10-19", "CONNORS_QQQ")["status"] == "WAITING"

        # 16:00 the Nasdaq CFD closes at the bid.
        await at(rt, clock, datetime(2026, 10, 19, 20, 0, 1, tzinfo=UTC))
        nas = db.execute("SELECT * FROM cfd_trades WHERE market='NAS100'").fetchone()
        assert (nas["exit_reason"], nas["exit_price"]) == ("CLOSE", 29999.0)
        assert nas["pnl_usd"] == pytest.approx(-0.66)

        # 15:50 Connors, read at 16:05: the next expiry's call nearest 0.50 delta.
        fake.qqq(714.98, 715.02)
        await at(rt, clock, datetime(2026, 10, 19, 20, 5, 1, tzinfo=UTC))
        signal = rt.store.signal("2026-10-19", "CONNORS", "QQQ")
        assert signal["side"] == 1
        connors = rt.store.option("2026-10-19", "CONNORS_QQQ")
        assert (connors["strike"], connors["expiry"], connors["exit_day"]) == (
            700.0,
            "2026-10-20",
            "2026-10-20",
        )

        # Tuesday 15:55 (read 16:10): the Connors call is sold at the bid. Tuesday's entries,
        # never tried at their time in this jump, are recorded missed rather than replayed.
        await at(rt, clock, datetime(2026, 10, 20, 20, 10, 1, tzinfo=UTC))
        assert rt.store.event("2026-10-20", "FIRST_CANDLE")["note"] == "NOT_TRIED_IN_TIME"
        connors = rt.store.option("2026-10-19", "CONNORS_QQQ")
        assert connors["exit_reason"] == "BEFORE_CLOSE" and connors["pnl_usd"] is not None

        # Only reads ever left the app.
        assert {m for m, _, _ in fake.requests} == {"GET"}
        await rt.client.close()

    asyncio.run(scenario())


def test_a_restart_never_duplicates_and_a_late_start_misses_entries(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        again = make(tmp_path, fake, clock, rt.store)
        await at(again, clock, OPEN + timedelta(minutes=5, seconds=30))
        assert rt.store.db.execute("SELECT COUNT(*) FROM cfd_trades").fetchone()[0] == 2
        await rt.client.close()
        await again.client.close()

        late_clock = Clock(OPEN + timedelta(minutes=8))
        late = make(tmp_path / "late", fake, late_clock)
        await at(late, late_clock, OPEN + timedelta(minutes=8))
        first = late.store.event("2026-10-19", "FIRST_CANDLE")
        assert (first["status"], first["note"]) == ("MISSED", "APP_NOT_RUNNING_AT_ENTRY")
        assert late.store.db.execute("SELECT COUNT(*) FROM cfd_trades").fetchone()[0] == 0
        await late.client.close()

    asyncio.run(scenario())


def test_missing_data_is_recorded_not_guessed(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)
    fake.bars[NAS] = trend_bars(OPEN - timedelta(minutes=5), 29900, 2.0, forming=False)
    fake.chart_delay = 0

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        assert rt.store.signal("2026-10-19", "FIRST_CANDLE", "NAS100") is None
        assert rt.store.event("2026-10-19", "FIRST_CANDLE")["status"] == "RETRYING"
        await at(rt, clock, OPEN + timedelta(minutes=6, seconds=5))
        assert rt.store.signal("2026-10-19", "FIRST_CANDLE", "NAS100") is None
        first = rt.store.event("2026-10-19", "FIRST_CANDLE")
        assert (first["status"], first["note"]) == ("FAILED", "NO_FIRST_CANDLE")
        # The S&P still traded; the QQQ option has no Nasdaq signal to follow.
        assert [r[0] for r in rt.store.db.execute("SELECT market FROM cfd_trades")] == ["US500"]
        await at(rt, clock, OPEN + timedelta(minutes=20, seconds=1))
        entry = rt.store.event("2026-10-19", "FIRST_CANDLE_QQQ_ENTRY")
        assert (entry["status"], entry["note"]) == ("DONE", "NO_NASDAQ_SIGNAL")
        await rt.client.close()

    asyncio.run(scenario())


def test_early_closes_and_holidays_record_nothing(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)
    fake.sessions = regular_sessions(MON, TUE, early=MON)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        first = rt.store.event("2026-10-19", "FIRST_CANDLE")
        assert (first["status"], first["note"]) == ("DONE", "EARLY_CLOSE_DAY_SKIPPED")
        await at(rt, clock, datetime(2026, 10, 22, 13, 40, tzinfo=UTC))  # no session listed
        assert (
            rt.store.db.execute("SELECT COUNT(*) FROM events WHERE day='2026-10-22'").fetchone()[0]
            == 0
        )
        assert rt.store.db.execute("SELECT COUNT(*) FROM cfd_trades").fetchone()[0] == 0
        await rt.client.close()

    asyncio.run(scenario())


def test_dashboard_views_render_the_day(tmp_path):
    import httpx

    from stocker_dashboard.app import create_dashboard_app

    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        transport = httpx.ASGITransport(create_dashboard_app(rt), client=("127.0.0.1", 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
            today = (await http.get("/api/today")).json()
            events = [p["event"] for p in today["plan"]]
            assert events[:2] == ["FIRST_CANDLE", "FIRST_CANDLE_QQQ_ENTRY"]
            assert today["plan"][0]["new_york"] == "09:35" and today["plan"][0]["uk"] == "14:35"
            assert len(today["cfd"]) == 2 and today["system"]["qqq_delay_minutes"] == 15
            assert all("open_pnl_usd" in t for t in today["cfd"])
            record = (await http.get("/api/record")).json()
            assert {m["method"] for m in record["summary"]} == {"CFD NAS100", "CFD US500"}
            for page in ("/", "/record", "/system"):
                assert (await http.get(page)).status_code == 200
            assert (await http.get("/markets")).status_code == 404
        await rt.client.close()

    asyncio.run(scenario())


def test_a_delay_change_loses_an_entry_but_an_exit_reads_again(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))  # delay measured: 15
        await at(rt, clock, OPEN + timedelta(minutes=20, seconds=1))
        opt = rt.store.option("2026-10-19", "FIRST_CANDLE_QQQ")
        assert opt["entry_delay"] == 15
        # 13:00: delayed as measured; by 13:15 the user's SaxoTraderGO has logged out and the
        # app's session receives live prices, so the read is refused and measured again.
        await at(rt, clock, datetime(2026, 10, 19, 17, 0, 1, tzinfo=UTC))
        fake.qqq_delay = 0
        fake.qqq(699.98, 700.02)
        await at(rt, clock, datetime(2026, 10, 19, 17, 15, 1, tzinfo=UTC), passes=3)
        opt = rt.store.option("2026-10-19", "FIRST_CANDLE_QQQ")
        assert opt["exit_reason"] == "AT_13_00_LATE" and opt["exit_delay"] == 0
        assert opt["exit_for"].startswith("2026-10-19T17:15:01")  # the time it really applies to
        await rt.client.close()

    asyncio.run(scenario())


def test_a_restart_after_the_entry_minute_never_buys_at_a_later_price(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))  # QQQ entry waits for 09:50
        await rt.client.close()
        clock.at = OPEN + timedelta(minutes=22)  # restarted at 09:52: two minutes too late
        again = make(tmp_path, fake, clock, rt.store)
        await at(again, clock, OPEN + timedelta(minutes=22))
        entry = again.store.event("2026-10-19", "FIRST_CANDLE_QQQ_ENTRY")
        assert (entry["status"], entry["note"]) == ("MISSED", "APP_NOT_RUNNING_AT_ENTRY")
        assert again.store.option("2026-10-19", "FIRST_CANDLE_QQQ") is None
        await again.client.close()

    asyncio.run(scenario())


def test_the_signal_survives_a_cfd_quote_problem_and_the_option_still_follows(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)
    fake.cfd_quote(NAS, 29999.0, 30000.0, state="Closed")  # not tradable through the entry minute

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        await at(rt, clock, OPEN + timedelta(minutes=6, seconds=5))
        assert rt.store.signal("2026-10-19", "FIRST_CANDLE", "NAS100")["side"] == 1
        first = rt.store.event("2026-10-19", "FIRST_CANDLE")
        assert (first["status"], first["note"]) == ("FAILED", "CFD_QUOTE_NOT_LIVE")
        assert rt.store.cfd_trade("2026-10-19", "NAS100") is None
        await at(rt, clock, OPEN + timedelta(minutes=20, seconds=1))
        assert rt.store.option("2026-10-19", "FIRST_CANDLE_QQQ")["put_call"] == "Call"
        await rt.client.close()

    asyncio.run(scenario())


def test_missed_exits_are_recorded_without_a_price(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        await at(rt, clock, OPEN + timedelta(minutes=20, seconds=1))
        await rt.client.close()
        # Restarted at 15:00 New York: far past 13:00, so the option's exit has no price.
        clock.at = datetime(2026, 10, 19, 19, 0, tzinfo=UTC)
        again = make(tmp_path, fake, clock, rt.store)
        await at(again, clock, clock.at)
        opt = again.store.option("2026-10-19", "FIRST_CANDLE_QQQ")
        assert opt["exit_reason"] == "MISSED_EXIT" and opt["pnl_usd"] is None
        await again.client.close()
        # Down from before 16:00 Monday to Tuesday morning: the Nasdaq CFD is never stopped
        # overnight at a later price, even after a gap through its stop.
        fake.cfd_quote(NAS, 29000.0, 29001.0)
        clock.at = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
        later = make(tmp_path, fake, clock, rt.store)
        await at(later, clock, clock.at)
        nas = later.store.cfd_trade("2026-10-19", "NAS100")
        assert nas["exit_reason"] == "MISSED_EXIT" and nas["pnl_usd"] is None
        record = {m["method"]: m for m in later.store.record(0)["summary"]}
        assert record["CFD NAS100"]["total_usd"] is None
        await later.client.close()

    asyncio.run(scenario())


def test_a_connors_call_is_never_sold_after_the_close(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        fake.qqq(714.98, 715.02)
        await at(rt, clock, datetime(2026, 10, 19, 19, 50, 1, tzinfo=UTC))
        await at(rt, clock, datetime(2026, 10, 19, 20, 5, 1, tzinfo=UTC))
        assert rt.store.option("2026-10-19", "CONNORS_QQQ") is not None
        # Tuesday 15:55: delay measured 15; at 16:10 SaxoTraderGO has logged off and prices are
        # live, after the option expired. The sale is recorded missed, not at a zero bid.
        await at(rt, clock, datetime(2026, 10, 20, 19, 55, 1, tzinfo=UTC))
        fake.qqq_delay = 0
        fake.qqq(714.98, 715.02)
        for second in (1, 7, 13):
            await at(rt, clock, datetime(2026, 10, 20, 20, 10, second, tzinfo=UTC))
        connors = rt.store.option("2026-10-19", "CONNORS_QQQ")
        assert connors["exit_reason"] == "MISSED_EXIT" and connors["pnl_usd"] is None
        await rt.client.close()

    asyncio.run(scenario())
