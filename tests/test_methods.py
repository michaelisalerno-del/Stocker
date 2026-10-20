"""The frozen calculations, checked by hand-worked examples."""

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_execution import methods
from stocker_execution.methods import Bar, OptionQuote

OPEN = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)


def bars(closes, last=OPEN, forming=True):
    """Candles ending with the one starting at `last`, plus the next one still forming."""
    out = [
        Bar(last - timedelta(minutes=5 * (len(closes) - 1 - i)), c) for i, c in enumerate(closes)
    ]
    return out + [Bar(last + timedelta(minutes=5), -1.0)] if forming else out


def test_ema_is_seeded_with_the_first_close_like_pandas_adjust_false():
    assert methods.ema([10.0]) == 10.0
    a = 2 / 13
    assert methods.ema([10.0, 13.0]) == pytest.approx(10 + a * 3)
    assert methods.ema([10.0, 13.0, 13.0]) == pytest.approx((10 + a * 3) + a * (13 - (10 + a * 3)))


def test_first_candle_long_short_and_refusals():
    rising = methods.first_candle(bars([100 + i for i in range(40)]), OPEN)
    assert (rising.side, rising.close, rising.prior_bars) == (1, 139, 36)
    assert rising.ema < 139
    falling = methods.first_candle(bars([100 - i for i in range(40)]), OPEN)
    assert falling.side == -1
    flat = methods.first_candle(bars([100.0] * 40), OPEN)
    assert (flat.side, flat.reason) == (0, "CLOSE_EQUALS_EMA")
    # The 09:35 candle (start 09:35) is still forming at 09:35 and never counts, and the 09:30
    # candle is final only once Saxo has started the next one.
    later = bars([100 + i for i in range(41)], OPEN + timedelta(minutes=5))
    assert methods.first_candle(later, OPEN).close == 139
    unfinished = bars([100 + i for i in range(40)], forming=False)
    assert methods.first_candle(unfinished, OPEN).reason == "FIRST_CANDLE_NOT_FINAL"
    early = bars([1.0] * 40, OPEN - timedelta(minutes=5), forming=False)
    assert methods.first_candle(early, OPEN).reason == "NO_FIRST_CANDLE"
    sparse = [b for i, b in enumerate(bars([100 + i for i in range(40)])) if i % 4 or i >= 39]
    assert methods.first_candle(sparse, OPEN).reason == "TOO_FEW_PRIOR_CANDLES"


def test_cfd_size_stop_and_result_at_200_dollars():
    # Nasdaq 30,000, 100 bp stop = 300 points -> 0.66 contracts (two decimals, rounded down).
    assert methods.cfd_amount(30000, 100, 2, 1.0) == 0.66
    # S&P 7,700, 50 bp stop = 38.5 points -> 5.1 contracts (one decimal).
    assert methods.cfd_amount(7700, 50, 1, 1.0) == 5.1
    assert methods.cfd_amount(7700, 50, 1, 2.0) == 2.5  # $2 a point: half the contracts
    assert methods.stop_price(30000, 1, 100) == 29700
    assert methods.stop_price(7700, -1, 50) == pytest.approx(7738.5)
    assert methods.stop_fill(1, 29700, 29700.5, 29701) is None
    assert methods.stop_fill(1, 29700, 29699, 29700.5) == 29699
    assert methods.stop_fill(-1, 7738.5, 7738, 7738.4) is None
    assert methods.stop_fill(-1, 7738.5, 7738.2, 7739) == 7739
    assert methods.cfd_pnl(1, 30000, 29699, 0.66, 1.0) == pytest.approx(-198.66)
    assert methods.cfd_pnl(-1, 7700, 7650, 5.1, 1.0) == pytest.approx(255)


def option(strike, delta, bid=1.0, ask=1.1, put_call="Call"):
    return OptionQuote(int(strike), strike, put_call, bid, ask, delta, 0.2, 2.04, 2.04, 15, "Open")


def test_option_pick_nearest_delta_inside_band_with_two_sided_quotes():
    board = [
        option(700, 0.52),
        option(701, 0.43),
        option(702, 0.37),
        option(703, 0.26),
        option(699, 0.41, bid=0.0),
        option(698, -0.41, put_call="Put"),
    ]
    assert methods.pick_option(board, "Call", 0.40, (0.25, 0.55)).strike == 701
    assert methods.pick_option(board, "Put", 0.40, (0.25, 0.55)).strike == 698
    assert methods.pick_option(board, "Call", 0.50, (0.42, 0.58)).strike == 700
    assert methods.pick_option([option(710, 0.1)], "Call", 0.40, (0.25, 0.55)) is None
    tie = [option(701, 0.42, 1.0, 1.2), option(702, 0.38, 1.0, 1.05)]
    assert methods.pick_option(tie, "Call", 0.40, (0.25, 0.55)).strike == 702
    assert methods.option_pnl(1.10, 1.60, 2.04, 2.04) == pytest.approx(45.92)
    assert methods.option_pnl(1.10, 1.60, None, 2.04) is None


def test_rsi2_and_connors_rule():
    r = methods.rsi2([1.0, 2.0, 3.0, 2.0])
    assert math.isnan(r[0]) and r[1:] == [100.0, 100.0, 50.0]
    assert methods.connors([1.0] * 199).reason == "TOO_FEW_DAILY_CLOSES"
    uptrend_dip = [100 + 0.1 * i for i in range(198)] + [118.0, 117.0]
    c = methods.connors(uptrend_dip)
    assert c.signal and c.close == 117.0 and c.close > c.sma200 and c.rsi + c.rsi_prev < 35
    assert not methods.connors([100 + 0.1 * i for i in range(200)]).signal  # no dip
    assert not methods.connors([200 - 0.1 * i for i in range(198)] + [178.0, 177.0]).signal


def test_dip_rules_turnaround_monday_and_connors_exit():
    mon, tue, wed = date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)
    rising = [100 + 0.1 * i for i in range(199)]
    down = methods.dip_rules([*rising, rising[-1] * 0.989], mon)
    assert down.turnaround and down.change == pytest.approx(-0.011) and down.reason == ""
    assert not methods.dip_rules([*rising, rising[-1] * 0.989], tue).turnaround  # not a Monday
    assert not methods.dip_rules([*rising, rising[-1] * 0.991], mon).turnaround  # only -0.9%
    # Method 3's dip opens Connors; a strong up day after it takes RSI2 above 65 and closes it.
    dip = [100 + 0.1 * i for i in range(198)] + [118.0, 117.0]
    opened = methods.dip_rules(dip, tue)
    assert opened.connors.signal and not opened.connors_exit
    assert methods.dip_rules([*dip, 120.0], wed).connors_exit
    assert methods.dip_rules([1.0] * 199, mon).reason == "TOO_FEW_DAILY_CLOSES"


def test_sessions_regular_days_and_early_closes():
    exchange = {
        "ExchangeSessions": [
            {
                "StartTime": "2026-11-27T14:30:00.000000Z",
                "EndTime": "2026-11-27T18:00:00.000000Z",
                "State": "AutomatedTrading",
            },
            {
                "StartTime": "2026-11-27T18:00:00.000000Z",
                "EndTime": "2026-11-27T22:00:00.000000Z",
                "State": "PostMarket",
            },
            {
                "StartTime": "2026-11-30T14:30:00.000000Z",
                "EndTime": "2026-11-30T21:00:00.000000Z",
                "State": "AutomatedTrading",
            },
        ]
    }
    sessions = methods.trading_sessions(exchange)
    assert [s.day for s in sessions] == [date(2026, 11, 27), date(2026, 11, 30)]
    assert not sessions[0].regular and sessions[1].regular  # EST: 09:30-16:00 is 14:30-21:00
    assert methods.session_on(sessions, date(2026, 11, 28)) is None


def candles(day_closes, prior=78, newest_forming=True):
    """QQQ candles: one flat-700 session the day before, then the day's closes (typical price =
    close, volume 1,000), plus a forming candle after the last one."""
    out = []
    prev = OPEN - timedelta(days=3)  # the Friday before, 09:30 New York
    for i in range(prior):
        out.append(methods.Candle(prev + timedelta(minutes=5 * i), 700.1, 699.9, 700.0, 1000.0))
    for i, c in enumerate(day_closes):
        out.append(methods.Candle(OPEN + timedelta(minutes=5 * i), c + 0.1, c - 0.1, c, 1000.0))
    if newest_forming:
        nxt = OPEN + timedelta(minutes=5 * len(day_closes))
        out.append(methods.Candle(nxt, 0.0, 0.0, 0.0, 0.0))
    return out


DAY = date(2026, 10, 19)


def test_vwap_crossing_up_enters_and_the_close_back_below_exits():
    # Flat 700 (EMA = VWAP: gap 0), 701 from the candle closing 10:30, 690 from 11:00.
    closes = [700.0] * 11 + [701.0] * 6 + [690.0] * 20
    (c,) = methods.vwap_crosses(candles(closes), DAY)
    assert c.entry == OPEN + timedelta(minutes=55)  # the 10:25 candle, closing 10:30
    assert c.entry_close == 701.0
    assert c.entry_ema == pytest.approx(700 + 2 / 11)
    assert c.entry_vwap == pytest.approx((11 * 700 + 701) / 12)
    assert c.exit == OPEN + timedelta(minutes=85) and c.exit_close == 690.0  # closes 11:00


def test_vwap_crossings_are_one_at_a_time_inside_10_00_to_15_00():
    # A jump on the candle closing 10:00 moves the EMA (2/11) more than the six-candle VWAP
    # (1/6): a crossing at exactly 10:00 is taken. One at 15:10 is not.
    (at_ten,) = methods.vwap_crosses(candles([700.0] * 5 + [701.0] * 20), DAY)
    assert at_ten.entry == OPEN + timedelta(minutes=25)
    assert methods.vwap_crosses(candles([700.0] * 67 + [701.0] * 5), DAY) == []
    # A second crossing after the exit is a new trade; it is open while no close is below VWAP.
    closes = [700.0] * 11 + [701.0] * 6 + [690.0] * 24 + [720.0] * 10
    first, second = methods.vwap_crosses(candles(closes), DAY)
    assert first.exit is not None
    assert second.entry == OPEN + timedelta(minutes=5 * 41) and second.exit is None
    # The candle still forming never counts: without a newer one the 13:00 close is unseen.
    seen = methods.vwap_crosses(candles(closes[:42], newest_forming=False), DAY)
    assert len(seen) == 1
    # No 09:30 candle today, or another day's candles only: nothing.
    assert methods.vwap_crosses(candles(closes)[78:][1:], DAY) == []
    assert methods.vwap_crosses(candles(closes), date(2026, 10, 20)) == []


def test_a_crossing_held_past_15_55_ends_with_the_16_00_exit():
    closes = [700.0] * 11 + [701.0] * 66 + [690.0]  # back below only at the 16:00 close
    (c,) = methods.vwap_crosses(candles(closes), DAY)
    assert c.exit is None


def test_trend_context_books_and_playbook():
    rising = [100.0 + i for i in range(110)]
    up = methods.trend_context(rising)
    assert (up.prior_close, up.down_closes) == (209.0, 0)
    assert up.sma50 == pytest.approx(sum(rising[-50:]) / 50)
    assert up.detail()["below_50d"] is False and up.detail()["prior_change"] == pytest.approx(
        1 / 208, abs=1e-6
    )
    falling = [300.0 - i for i in range(110)]
    down = methods.trend_context(falling).detail()
    assert down["below_50d"] and down["below_100d"] and down["down_closes"] == 109
    assert down["rsi2_prior"] == 0.0
    assert methods.trend_context(rising[:100]) is None  # 101 closes are needed
    assert methods.down_run([1.0, 3.0, 2.0, 1.0]) == 2

    cell = methods.Day(1, 0, True, down)  # Monday, long, below the 50-day, S&P agrees
    assert methods.in_book("PULLBACK", cell, "NAS100") and not methods.in_book(
        "WEEKDAYS", cell, "NAS100"
    )
    assert methods.in_book("AGREE", cell, "NAS100")
    # The playbook takes the call in the cell (and the S&P there), not the Nasdaq CFD.
    assert [methods.in_book("PLAYBOOK", cell, i) for i in ("NAS100", "US500", "QQQ option")] == [
        False,
        True,
        True,
    ]
    assert methods.playbook(cell, "NAS100") == (
        "QQQ call",
        ["pullback cell: long below the 50-day", "Monday", "agrees with the other index"],
    )
    assert methods.playbook(cell, "US500")[0] == "S&P CFD"
    oversold_short = methods.Day(-1, 2, False, down)
    assert not methods.in_book("NO OVERSOLD SHORTS", oversold_short, "NAS100")
    assert methods.playbook(oversold_short, "NAS100") == (
        "skip the short",
        ["short after an oversold close", "disagrees with the other index"],
    )
    short = methods.Day(-1, 2, None, up.detail())
    assert methods.in_book("NO OVERSOLD SHORTS", short, "NAS100")
    assert methods.in_book("PLAYBOOK", short, "NAS100")
    assert methods.playbook(short, "NAS100") == ("Nasdaq CFD short, not a put", [])
    assert methods.playbook(short, "US500")[0] == "skip: no S&P edge outside the pullback cell"
    big = methods.Day(1, 1, True, methods.trend_context([*rising, 230.0]).detail())
    assert methods.in_book("AFTER A +1% DAY", big, "NAS100")
    assert "after a +1% day" in methods.playbook(big, "NAS100")[1]
    unknown = methods.Day(1, 2, None, {"close": 1.0})  # recorded before amendment 4
    assert not methods.in_book("PULLBACK", unknown, "NAS100") and methods.in_book(
        "WEEKDAYS", unknown, "NAS100"
    )
    assert methods.playbook(unknown, "NAS100") == ("context not read", [])
    assert methods.playbook(methods.Day(0, 2, None, down), "NAS100")[0] == "no trade"


def test_dip_rules_fear_and_the_30_day_vol_reading():
    closes = [100.0 + 0.5 * i for i in range(231)] + [214.5, 214.0, 213.5]
    high = methods.dip_rules([*closes, 213.0], date(2026, 10, 20), 0.35)
    assert high.connors.signal and high.fear and high.down_closes == 4
    assert not methods.dip_rules([*closes, 213.0], date(2026, 10, 20), 0.25).fear
    assert not methods.dip_rules([*closes, 213.0], date(2026, 10, 20), None).fear
    assert not methods.dip_rules([*closes[:-2], 213.0], date(2026, 10, 20), 0.35).fear  # 2 down

    def q(k, pc, vol):
        return OptionQuote(1, k, pc, 1.0, 1.1, 0.5, vol, 2.0, 2.0, 0, "Open")

    board = [
        q(699.0, "Call", 0.40),
        q(700.0, "Call", 0.30),
        q(700.0, "Put", 0.32),
        q(701.0, "Put", 0.20),
    ]
    assert methods.atm_vol(board, 700.3) == pytest.approx(0.31)
    assert methods.atm_vol([q(700.0, "Call", None)], 700.0) is None
