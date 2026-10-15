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
