"""Synthetic daily bars for the stock dips (amendment 5): a steady uptrend, then the worked
example's sell-off, a limit fill and a rebound that ends the trade."""

from datetime import date, timedelta

from stocker_execution.methods import StockBar

# Closes relative to the trend's last close (100%): the two sell-off days, the fill day and
# the rebound; highs and lows as fractions of the close.
AFTER = [
    (0.97, 1.00, 0.965),  # day -1: down 3%
    (0.91, 0.965, 0.91),  # signal day: down 6%, closing on its low
    (0.92, 0.96, 0.835),  # next session: opens lower, trades through the 8% limit, closes up 1%
    (1.04, 1.05, 0.95),  # rebound: ConnorsRSI rises above 70, the trade closes here
    (1.06, 1.07, 1.03),
]


def sessions(last: date, count: int) -> list[date]:
    """`count` weekdays ending at `last`, oldest first."""
    out, d = [], last
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def dip_series(days: list[date], start: float = 50.0) -> list[StockBar]:
    """Bars for `days`: a trend rising 1.3% a day with a 0.3% dip every fifth day (so streaks
    stay short, as real stocks' do), then AFTER's pattern over the last days."""
    bars, n = [], len(days) - len(AFTER)
    close = start
    for i, d in enumerate(days):
        if i < n:
            prior = close
            close = round(prior * (0.997 if i % 5 == 4 else 1.013), 6)
            h, lo = max(prior, close) * 1.005, min(prior, close) * 0.995
            o = prior
        else:
            frac, hi_f, lo_f = AFTER[i - n]
            base = bars[n - 1].close
            close, h, lo = base * frac, base * hi_f, base * lo_f
            o = min(h, max(lo, bars[-1].close * (1.0 if i - n != 2 else 0.98)))
        bars.append(StockBar(d.isoformat(), o, h, lo, close, close, 1_000_000.0))
    return bars


def flat_series(spy: list[StockBar], level: float = 30.0) -> list[StockBar]:
    """A stock that moves with SPY (half its returns) and never sells off enough to set up."""
    out = []
    for prev, b in zip([spy[0], *spy], spy, strict=False):
        level *= 1 + 0.5 * (b.adj / prev.adj - 1)
        out.append(StockBar(b.day, level, level * 1.005, level * 0.995, level, level, 2e6))
    return out


def spy_series(stock: list[StockBar]) -> list[StockBar]:
    """SPY moving with the stock (half its returns), so the correlation rule passes."""
    out, level = [], 500.0
    for prev, b in zip([stock[0], *stock], stock, strict=False):
        level *= (b.adj / prev.adj) ** 0.5
        out.append(StockBar(b.day, level, level * 1.002, level * 0.998, level, level, 5e7))
    return out
