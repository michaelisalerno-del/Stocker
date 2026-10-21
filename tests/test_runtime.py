"""A whole paper day against an offline Saxo: entries, stops, delayed QQQ reads and exits."""

import asyncio
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from fake_saxo import NAS, SPX, FakeSaxo, client, qqq_candles, regular_sessions, trend_bars
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
    monkeypatch.setattr(runtime_module, "CROSS_POLL_SECONDS", 0)
    monkeypatch.setattr(runtime_module, "CROSS_DELAYED_POLL_SECONDS", 0)


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
        assert spx["worst"] == 7740.0  # the ask that stopped it was its worst adverse price

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
        assert nas["worst"] == 29999.0  # the bid never moved
        what_if = next(b for b in rt.store.record(0)["books"] if b["book"] == "STOP 75 BP")
        assert what_if["cells"]["NAS100"]["total_usd"] == pytest.approx(-0.66 * 100 / 75)
        # The equity curve books the three closed results (stop, option, close) on their day.
        by_day = rt.store.record(0)["by_day"]
        assert [(r["day"], r["closed"]) for r in by_day] == [("2026-10-19", 3)]
        assert by_day[0]["pnl_usd"] == pytest.approx(-204 + 59.92 - 0.66)

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
            assert record["by_day"] == []  # nothing has closed yet
            # The chart cache: QQQ and the S&P CFD, CFD candles at the bid/ask midpoint.
            charts = (await http.get("/api/charts")).json()
            assert set(charts["markets"]) == {"US500", "QQQ"}
            spx = charts["markets"]["US500"]
            assert spx["delay"] == 0 and len(spx["bars"]) == 61 and len(spx["bars"][-2]) == 6
            assert spx["bars"][-2][4] == pytest.approx(7701.0)  # the 09:30 candle's close
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


FRI = date(2026, 10, 16)
# QQQ: flat 700 (EMA = VWAP), up through VWAP at the 10:30 close, back below at 11:00, up again
# at 13:00 and held to the close.
CROSSING_DAY = [700.0] * 11 + [701.0] * 6 + [690.0] * 24 + [720.0] * 37


def ny(h, m, s=3):
    return datetime(2026, 10, 19, h + 4, m, s, tzinfo=UTC)  # New York (EDT) to UTC


async def until(rt, fake, clock, when):
    fake.qqq_upto = when
    await at(rt, clock, when)


def crosses(rt):
    return [dict(r) for r in rt.store.db.execute("SELECT * FROM cross_trades ORDER BY id")]


def test_vwap_crossings_on_a_long_first_candle_day(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)
    fake.qqq_bars = qqq_candles([FRI, MON], CROSSING_DAY)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await until(rt, fake, clock, ny(9, 35, 2))  # the Nasdaq first candle is long
        await until(rt, fake, clock, ny(10, 0))
        assert crosses(rt) == []
        # 10:30: EMA10 closes above VWAP -> long the Nasdaq CFD at the live ask, sized as the
        # first-candle trade, with the QQQ readings that made the decision.
        await until(rt, fake, clock, ny(10, 30))
        (t,) = crosses(rt)
        assert (t["entry_price"], t["amount"], t["qqq_close"]) == (30000.0, 0.66, 701.0)
        assert t["signal_candle"].startswith("2026-10-19T14:25") and t["exit_at"] is None
        # 11:00: back below VWAP -> sold at the bid.
        fake.cfd_quote(NAS, 30099.0, 30100.0)
        await until(rt, fake, clock, ny(11, 0))
        (t,) = crosses(rt)
        assert (t["exit_reason"], t["exit_price"], t["qqq_exit_close"]) == (
            "BACK_BELOW_VWAP",
            30099.0,
            690.0,
        )
        assert t["pnl_usd"] == pytest.approx(99 * 0.66)
        # A restart repeats nothing; 13:00 opens the second crossing, held to 16:00.
        again = make(tmp_path, fake, clock, rt.store)
        await until(again, fake, clock, ny(11, 0, 30))
        await until(again, fake, clock, ny(13, 0))
        first, second = crosses(again)
        assert second["entry_price"] == 30100.0 and second["exit_at"] is None
        fake.cfd_quote(NAS, 30199.0, 30200.0)
        await until(again, fake, clock, ny(16, 0, 1))
        first, second = crosses(again)
        assert (second["exit_reason"], second["exit_price"]) == ("CLOSE", 30199.0)
        assert again.store.event("2026-10-19", "VWAP_CROSS_EXIT")["status"] == "DONE"
        record = again.store.record(0)
        assert {"method": "CROSS NAS100", "trades": 2, "closed": 2}.items() <= next(
            m for m in record["summary"] if m["method"] == "CROSS NAS100"
        ).items()
        # Nothing but reads left the app.
        assert {m for m, _, _ in fake.requests} == {"GET"}
        await rt.client.close()
        await again.client.close()

    asyncio.run(scenario())


def test_a_crossing_not_acted_on_in_time_is_recorded_missed(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)
    fake.qqq_bars = qqq_candles([FRI, MON], CROSSING_DAY)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await until(rt, fake, clock, ny(9, 35, 2))
        await until(rt, fake, clock, ny(10, 40))  # no pass between 10:30 and 10:40
        (t,) = crosses(rt)
        assert (t["exit_reason"], t["note"], t["entry_price"]) == (
            "MISSED_ENTRY",
            "NOT_TRIED_IN_TIME",
            None,
        )
        # The missed trade is not open: the 13:00 crossing still trades.
        await until(rt, fake, clock, ny(11, 0))
        await until(rt, fake, clock, ny(13, 0))
        assert [r["exit_reason"] for r in crosses(rt)] == ["MISSED_ENTRY", None]
        await rt.client.close()

        # QQQ's chart 15 minutes late: the 10:30 crossing is seen at 10:45, too late to enter.
        delayed = FakeSaxo()
        market(delayed)
        delayed.qqq_bars, delayed.qqq_chart_delay = qqq_candles([FRI, MON], CROSSING_DAY), 15
        dclock = Clock(OPEN)
        late = make(tmp_path / "delayed", delayed, dclock)
        await until(late, delayed, dclock, ny(9, 35, 2))
        await until(late, delayed, dclock, ny(10, 30))
        assert crosses(late) == []
        await until(late, delayed, dclock, ny(10, 45))
        (t,) = crosses(late)
        assert (t["exit_reason"], t["note"]) == ("MISSED_ENTRY", "QQQ_CHART_DELAYED")
        assert late.status()["problems"]["VWAP_CROSS"] == "QQQ_CHART_DELAYED"
        await late.client.close()

    asyncio.run(scenario())


def test_no_crossing_trades_on_a_short_first_candle_day(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN)
    market(fake)
    fake.bars[NAS] = trend_bars(OPEN, 30100, -2.0)  # falling into the open -> short
    fake.qqq_bars = qqq_candles([FRI, MON], CROSSING_DAY)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        for when in (ny(9, 35, 2), ny(10, 30), ny(13, 0), ny(16, 0, 1)):
            await until(rt, fake, clock, when)
        assert rt.store.signal("2026-10-19", "FIRST_CANDLE", "NAS100")["side"] == -1
        assert crosses(rt) == []
        assert (
            not [  # the crossings' 400-candle QQQ reads; the chart's 200-candle reads are display
                p
                for _, _, p in fake.requests
                if (p.get("AssetType"), p.get("Horizon"), p.get("Count")) == ("Etf", "5", "400")
            ]
        )
        assert rt.store.event("2026-10-19", "VWAP_CROSS_EXIT")["note"] == "NO_OPEN_CROSS_TRADE"
        await rt.client.close()

    asyncio.run(scenario())


def dip_market(fake: FakeSaxo) -> None:
    """Monday 15:50 with live QQQ at 710: 1.1% under Friday's 718 (Turnaround Tuesday) and a
    Connors dip in the rising daily closes."""
    market(fake)
    fake.qqq_delay = 0
    fake.qqq(709.98, 710.02)


MON_1550 = datetime(2026, 10, 19, 19, 50, 1, tzinfo=UTC)


def tuesday(fake: FakeSaxo, qqq_mid: float) -> None:
    fake.daily.append({"Time": "2026-10-19T00:00:00Z", "Close": 710.0})
    fake.qqq(qqq_mid - 0.02, qqq_mid + 0.02)
    fake.cfd_quote(NAS, 30299.0, 30300.0)


def test_dip_rules_buy_the_nasdaq_cfd_at_15_50_and_close_on_their_own_exits(tmp_path):
    fake, clock = FakeSaxo(), Clock(MON_1550 - timedelta(minutes=1))
    dip_market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, MON_1550)
        signal = rt.store.signal("2026-10-19", "DIP", "QQQ")
        assert signal["side"] == 1 and signal["reason"] == ""
        dips = {r["rule"]: dict(r) for r in rt.store.db.execute("SELECT * FROM dip_trades")}
        assert set(dips) == {"CONNORS", "TURNAROUND"}
        # Both at the live ask, sized as the first-candle Nasdaq trade; the one-at-a-time book
        # holds the first (Connors) only.
        assert (dips["CONNORS"]["entry_price"], dips["CONNORS"]["amount"]) == (30000.0, 0.66)
        assert (dips["CONNORS"]["book"], dips["TURNAROUND"]["book"]) == (1, 0)
        assert dips["TURNAROUND"]["qqq_price"] == pytest.approx(710.0)

        # Tuesday 15:50: a strong day. Turnaround Tuesday closes at its next session, Connors on
        # RSI2 above 65; both at the live bid. No new dip signal.
        tuesday(fake, 730.0)
        await at(rt, clock, datetime(2026, 10, 20, 19, 50, 1, tzinfo=UTC))
        dips = {r["rule"]: dict(r) for r in rt.store.db.execute("SELECT * FROM dip_trades")}
        assert dips["TURNAROUND"]["exit_reason"] == "NEXT_SESSION"
        assert dips["CONNORS"]["exit_reason"] == "RSI2_ABOVE_65"
        assert dips["CONNORS"]["pnl_usd"] == pytest.approx(0.66 * 299)
        entry = rt.store.event("2026-10-20", "DIP_ENTRY")
        assert (entry["status"], entry["note"]) == ("DONE", "NO_DIP_SIGNAL")
        summary = {m["method"]: m for m in rt.store.record(0)["summary"]}
        assert summary["DIP BOOK"]["trades"] == 1 and summary["DIP TURNAROUND"]["closed"] == 1
        assert {m for m, _, _ in fake.requests} == {"GET"}
        await rt.client.close()

    asyncio.run(scenario())


def test_a_delayed_qqq_price_misses_dip_entries_and_the_connors_check_only(tmp_path):
    fake, clock = FakeSaxo(), Clock(MON_1550 - timedelta(minutes=1))
    dip_market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, MON_1550)  # both bought with live QQQ
        # Tuesday: SaxoTraderGO holds the live slot. Turnaround Tuesday still closes (it needs
        # only the CFD); the Connors check cannot run, and its entry window passes.
        fake.qqq_delay = 15
        tuesday(fake, 730.0)
        await at(rt, clock, datetime(2026, 10, 20, 19, 50, 1, tzinfo=UTC))
        tt = rt.store.dip("2026-10-19", "TURNAROUND")
        assert tt["exit_reason"] == "NEXT_SESSION"
        assert rt.store.dip("2026-10-19", "CONNORS")["exit_at"] is None
        await at(rt, clock, datetime(2026, 10, 20, 20, 21, tzinfo=UTC))
        entry = rt.store.event("2026-10-20", "DIP_ENTRY")
        assert (entry["status"], entry["note"]) == ("FAILED", "QQQ_PRICE_DELAYED")
        assert rt.store.event("2026-10-20", "DIP_EXIT")["status"] == "FAILED"
        # Wednesday: Tuesday's Connors check never completed, so the trade has no exit price.
        fake.qqq_delay = 0
        fake.daily.append({"Time": "2026-10-20T00:00:00Z", "Close": 730.0})
        fake.qqq(729.98, 730.02)
        await at(rt, clock, datetime(2026, 10, 21, 19, 50, 1, tzinfo=UTC))
        con = rt.store.dip("2026-10-19", "CONNORS")
        assert con["exit_reason"] == "MISSED_EXIT" and con["pnl_usd"] is None
        await rt.client.close()

    asyncio.run(scenario())


def test_dip_exits_the_app_was_down_for_are_recorded_missed(tmp_path):
    fake, clock = FakeSaxo(), Clock(MON_1550 - timedelta(minutes=1))
    dip_market(fake)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, MON_1550)
        await rt.client.close()
        tuesday(fake, 730.0)
        fake.daily.append({"Time": "2026-10-20T00:00:00Z", "Close": 730.0})
        clock.at = datetime(2026, 10, 21, 19, 50, 1, tzinfo=UTC)  # back on Wednesday
        again = make(tmp_path, fake, clock, rt.store)
        await at(again, clock, clock.at)
        reasons = {
            r["rule"]: r["exit_reason"] for r in again.store.db.execute("SELECT * FROM dip_trades")
        }
        assert reasons == {"CONNORS": "MISSED_EXIT", "TURNAROUND": "MISSED_EXIT"}
        await again.client.close()

    asyncio.run(scenario())


def test_first_candle_context_books_and_playbook(tmp_path):
    fake, clock = FakeSaxo(), Clock(OPEN - timedelta(minutes=10))
    market(fake)
    # The reference ETFs' closes fell for 300 sessions (the fake serves one daily chart for
    # both): the prior close sits below its 50-day average at the end of a run of down closes.
    fake.daily = [
        {
            "Time": (datetime(2025, 8, 1, tzinfo=UTC) + timedelta(days=i)).strftime(
                "%Y-%m-%dT00:00:00Z"
            ),
            "Close": 760 - 0.2 * i,
        }
        for i in range(300)
    ]
    fake.daily[-1]["Time"] = "2026-10-16T00:00:00Z"

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, OPEN + timedelta(minutes=5, seconds=2))
        nas = json.loads(rt.store.signal("2026-10-19", "FIRST_CANDLE", "NAS100")["detail"])
        assert nas["below_50d"] is True and nas["down_closes"] == 299
        assert nas["prior_close"] == pytest.approx(700.2)
        assert nas["sma50"] == pytest.approx(sum(760 - 0.2 * i for i in range(250, 300)) / 50)
        assert {p["Uic"] for _, _, p in fake.requests if p.get("Horizon") == "1440"} == {
            "4328771",
            "36590",
        }
        play = {p["market"]: p for p in rt.today()["playbook"]}
        assert play["NAS100"]["action"] == "QQQ call"
        assert play["NAS100"]["tags"][0] == "pullback cell: long below the 50-day"
        assert play["US500"]["action"] == "skip: no S&P edge outside the pullback cell"
        assert {"Monday", "short after an oversold close"} <= set(play["US500"]["tags"])
        await at(rt, clock, OPEN + timedelta(minutes=20, seconds=1))  # the call follows
        books = {b["book"]: b["cells"] for b in rt.store.record(0)["books"]}
        assert books["PULLBACK"]["NAS100"]["trades"] == 1
        assert books["PULLBACK"]["QQQ option"]["trades"] == 1 and "US500" not in books["PULLBACK"]
        assert books["WEEKDAYS"] == {} and books["AGREE"] == {}  # a Monday; long against short
        assert set(books["NO OVERSOLD SHORTS"]) == {"NAS100", "QQQ option"}  # the S&P short is out
        assert set(books["PLAYBOOK"]) == {"QQQ option"}  # the call stands in for the Nasdaq CFD
        assert books["SIZED V2"]["NAS100"]["trades"] == 1 and "US500" not in books["SIZED V2"]
        assert books["SIZED V3"]["NAS100"]["trades"] == 1  # Monday cell long: 1 + 1 - 1 = 1 unit
        await rt.client.close()

    asyncio.run(scenario())


def test_the_fear_rule_buys_after_three_down_closes_when_30_day_vol_is_high(tmp_path):
    fake, clock = FakeSaxo(), Clock(MON_1550 - timedelta(minutes=1))
    dip_market(fake)
    # 723.5 -> 722 -> 720 -> 718 are three down closes; QQQ at 710 at 15:50 makes four.
    fake.daily[-1:] = [
        {"Time": f"2026-10-{d}T00:00:00Z", "Close": c}
        for d, c in ((14, 722.0), (15, 720.0), (16, 718.0))
    ]
    fake.sessions = regular_sessions(
        MON, TUE, WED, date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 26)
    )
    fake.mid_vol = 0.35
    fake.chain(date(2026, 11, 20), [690.0 + i for i in range(21)], 700.0, 300000)

    async def scenario():
        rt = make(tmp_path, fake, clock)
        await at(rt, clock, MON_1550)
        d = json.loads(rt.store.signal("2026-10-19", "DIP", "QQQ")["detail"])
        assert d["fear"] and d["down_closes"] == 4 and d["atm_vol_30d"] == pytest.approx(0.35)
        assert (d["atm_vol_expiry"], d["atm_vol_reason"]) == ("2026-11-20", "")
        dips = {r["rule"]: dict(r) for r in rt.store.db.execute("SELECT * FROM dip_trades")}
        assert set(dips) == {"CONNORS", "TURNAROUND", "FEAR"} and dips["FEAR"]["book"] == 0
        # Tuesday: Connors and Turnaround Tuesday close; the fear trade is held.
        tuesday(fake, 730.0)
        await at(rt, clock, datetime(2026, 10, 20, 19, 50, 1, tzinfo=UTC))
        assert rt.store.dip("2026-10-19", "FEAR")["exit_at"] is None
        # The fifth session after its entry (Monday 26 October): sold at 15:50 at the live bid.
        for day in (20, 21, 22, 23):
            fake.daily.append({"Time": f"2026-10-{day}T00:00:00Z", "Close": 730.0})
        await at(rt, clock, datetime(2026, 10, 26, 19, 50, 1, tzinfo=UTC))
        fear = rt.store.dip("2026-10-19", "FEAR")
        assert (fear["exit_reason"], fear["exit_day"]) == ("FIFTH_SESSION", "2026-10-26")
        assert fear["pnl_usd"] == pytest.approx(0.66 * 299)
        summary = {m["method"]: m for m in rt.store.record(0)["summary"]}
        assert summary["DIP FEAR"]["closed"] == 1
        await rt.client.close()

    asyncio.run(scenario())


# stock dips (amendment 5) ------------------------------------------------------------------
from stock_series import dip_series, flat_series, sessions, spy_series  # noqa: E402

ABCD, FLAT = 101, 102


def stock_world(fake: FakeSaxo) -> list:
    """Three listed stocks with daily bars through Thursday: ABCD sets up on MON, fills on TUE
    and rebounds on WED; FLAT never sets up; SPY (an ETF) moves with ABCD."""
    days = sessions(
        WED + timedelta(days=1), 300
    )  # the pattern's last five days: Fri, MON, TUE, WED, Thu
    abcd = dip_series(days)
    spy = spy_series(abcd)
    fake.stock("ABCD", ABCD, "NASDAQ", abcd)
    fake.stock("FLAT", FLAT, "NYSE", flat_series(spy))
    fake.stock("SPY", 36590, "", spy)
    fake.stock_upto = (MON - timedelta(days=3)).isoformat()  # Friday's bars are the newest
    return abcd


def test_stock_dips_scan_fill_and_exit_on_saxo_daily_bars(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "STOCK_RETRY_SECONDS", 0)
    saxo, clock = FakeSaxo(), Clock(OPEN - timedelta(minutes=10))
    market(saxo)
    abcd = stock_world(saxo)
    signal_bar, fill_bar, rebound_bar = abcd[-4], abcd[-3], abcd[-2]
    assert (signal_bar.day, fill_bar.day, rebound_bar.day) == (
        MON.isoformat(),
        TUE.isoformat(),
        WED.isoformat(),
    )

    async def scenario():
        rt = make(tmp_path, saxo, clock)
        db = rt.store.db
        # Monday 19:00 New York: the day's bars are not out yet; the scan waits.
        await at(rt, clock, OPEN + timedelta(hours=9, minutes=30))
        e = rt.store.event(MON.isoformat(), "STOCK_DIPS")
        assert e["status"] == "RETRYING" and e["note"] == "DAILY_BARS_NOT_PUBLISHED"
        saxo.stock_upto = MON.isoformat()
        await at(rt, clock, OPEN + timedelta(hours=9, minutes=45))
        assert rt.store.event(MON.isoformat(), "STOCK_DIPS")["status"] == "DONE"
        orders = list(db.execute("SELECT * FROM stock_orders"))
        assert [(o["code"], o["status"], o["tier"]) for o in orders] == [("ABCD", "PENDING", "A")]
        assert orders[0]["limit_price"] == round(signal_bar.close * 0.92, 4)
        night = json.loads(rt.store.signal(MON.isoformat(), "STOCK", "US")["detail"])
        assert (night["listed"], night["orders"], night["histories_read"], night["universe"]) == (
            2,
            1,
            2,
            2,
        )
        assert rt.store.stock_bar_count("ABCD") == 297  # the 300-bar read, through Monday
        view = rt.today()
        assert [o["code"] for o in view["stock_orders"]] == ["ABCD"] and view["stock_trades"] == []
        # Tuesday night: the low traded through the limit -> bought at the limit, still held.
        saxo.stock_upto = TUE.isoformat()
        await at(rt, clock, OPEN + timedelta(days=1, hours=9, minutes=30))
        order = db.execute("SELECT * FROM stock_orders").fetchone()
        assert (order["status"], order["fill_day"], order["fill_price"]) == (
            "FILLED",
            TUE.isoformat(),
            order["limit_price"],
        )
        t = db.execute("SELECT * FROM stock_trades").fetchone()
        assert t["entry_day"] == TUE.isoformat() and t["exit_day"] is None
        assert db.execute("SELECT COUNT(*) FROM stock_orders").fetchone()[0] == 1  # ABCD held
        assert rt.store.stock_bar_count("ABCD") == 298  # five newest bars merged, no re-read
        # Wednesday night: the rebound lifts ConnorsRSI above 70 -> sold at the close.
        saxo.stock_upto = WED.isoformat()
        await at(rt, clock, OPEN + timedelta(days=2, hours=9, minutes=30))
        t = db.execute("SELECT * FROM stock_trades").fetchone()
        assert (t["exit_day"], t["exit_reason"]) == (WED.isoformat(), "CRSI_ABOVE_70")
        assert t["crsi_exit"] > 70
        expected = 2000 * ((rebound_bar.close / order["limit_price"] - 1) - 0.002)
        assert t["pnl_usd"] == round(expected, 2) and t["pnl_usd"] > 0
        record = rt.store.record(0)
        summary = {m["method"]: m for m in record["summary"]}
        assert summary["STOCK DIP"]["closed"] == 1 and summary["STOCK DIP"]["wins"] == 1
        assert record["trades"][0]["method"] == "STOCK DIP ABCD"
        paths = [p for _, p, _ in saxo.requests]
        # Two exchanges per attempt: two tries before Monday's bars were out, one after, Tuesday,
        # Wednesday.
        assert paths.count("/ref/v1/instruments") == 10
        await rt.stop()

    asyncio.run(scenario())


def test_a_stock_scan_that_waits_past_midnight_still_runs_before_the_open(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "STOCK_RETRY_SECONDS", 0)
    saxo, clock = FakeSaxo(), Clock(OPEN - timedelta(minutes=10))
    market(saxo)
    stock_world(saxo)

    async def scenario():
        rt = make(tmp_path, saxo, clock)
        await at(rt, clock, OPEN + timedelta(hours=9, minutes=30))
        assert rt.store.event(MON.isoformat(), "STOCK_DIPS")["status"] == "RETRYING"
        # 06:00 New York on Tuesday: Monday's bars are out at last; the orders stand for Tuesday.
        saxo.stock_upto = MON.isoformat()
        await at(rt, clock, OPEN + timedelta(days=1) - timedelta(hours=3, minutes=30))
        assert rt.store.event(MON.isoformat(), "STOCK_DIPS")["status"] == "DONE"
        assert [o["code"] for o in rt.store.pending_stock_orders()] == ["ABCD"]
        # Past the grace, a night with no bars is recorded failed, never scanned later.
        await at(rt, clock, OPEN + timedelta(days=1, hours=9, minutes=30))
        await at(rt, clock, OPEN + timedelta(days=2, hours=2))
        e = rt.store.event(TUE.isoformat(), "STOCK_DIPS")
        assert (e["status"], e["note"]) == ("FAILED", "DAILY_BARS_NOT_PUBLISHED")
        await rt.stop()

    asyncio.run(scenario())
