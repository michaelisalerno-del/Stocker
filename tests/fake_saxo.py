"""An offline Saxo for the paper record: the GET endpoints it reads, with settable prices."""

import time
from datetime import UTC, date, datetime, timedelta

import httpx

from stocker_execution.config import SaxoSettings
from stocker_execution.saxo_auth import OAuth, atomic_json
from stocker_execution.saxo_client import SaxoClient

NAS, SPX, QQQ = 4912, 4913, 4328771


def iso(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def regular_sessions(*days: date, early: date | None = None) -> list[dict]:
    """NASDAQ AutomatedTrading sessions (EDT: 13:30-20:00 UTC; an early close at 17:00)."""
    out = []
    for d in days:
        start = datetime(d.year, d.month, d.day, 13, 30, tzinfo=UTC)
        end = start.replace(hour=17 if d == early else 20, minute=0)
        out.append({"StartTime": iso(start), "EndTime": iso(end), "State": "AutomatedTrading"})
        out.append(
            {"StartTime": iso(end), "EndTime": iso(end + timedelta(hours=1)), "State": "PostMarket"}
        )
    return out


def trend_bars(
    open_at: datetime, start: float, step: float, count: int = 60, forming: bool = True
) -> list[dict]:
    """5-minute CFD candles ending with the one that starts at `open_at`, drifting by `step`,
    plus (as Saxo sends it) the next candle, still forming."""
    rows = []
    for i in range(count + int(forming)):
        at = open_at - timedelta(minutes=5 * (count - 1 - i))
        mid = start + step * min(i, count - 1)
        row = {"Time": iso(at), "CloseBid": mid - 0.5, "CloseAsk": mid + 0.5}
        for k, off in (("Open", -step), ("High", abs(step) + 1), ("Low", -abs(step) - 1)):
            row[k + "Bid"], row[k + "Ask"] = mid + off - 0.5, mid + off + 0.5
        rows.append(row)
    return rows


def qqq_candles(days: list[date], closes: list[float]) -> list[dict]:
    """QQQ 5-minute regular-session candles: a flat 700 session per earlier day, then `closes`
    for the last day (high/low 0.10 either side, volume 1,000, so typical price = close)."""
    rows = []
    for n, d in enumerate(days):
        start = datetime(d.year, d.month, d.day, 13, 30, tzinfo=UTC)
        series = closes if n == len(days) - 1 else [700.0] * 78
        for i, c in enumerate(series):
            at = start + timedelta(minutes=5 * i)
            rows.append(
                {
                    "Time": iso(at),
                    "Open": c,
                    "High": c + 0.1,
                    "Low": c - 0.1,
                    "Close": c,
                    "Volume": 1000.0,
                }
            )
    return rows


class FakeSaxo:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []
        self.sessions: list[dict] = []
        self.bars: dict[int, list[dict]] = {}
        self.chart_delay = 0
        self.quotes: dict[int, dict] = {}
        self.daily: list[dict] = []
        self.expiries: list[date] = []
        self.options: dict[date, list[dict]] = {}  # SpecificOptions per expiry
        self.option_quotes: dict[int, dict] = {}
        self.qqq_delay = 15
        self.qqq_bars: list[dict] = []
        self.qqq_upto: datetime | None = None  # QQQ candles exist up to this time (the clock)
        self.qqq_chart_delay = 0
        self.mid_vol = 0.2  # Saxo's MidVol on every option quote
        # Stocks (amendment 5): listing rows, daily bars per Uic and the day they run to.
        self.listed: list[dict] = []
        self.daily_stock: dict[int, list[dict]] = {}
        self.stock_upto: str = "9999-12-31"

    def stock(self, code: str, uic: int, exchange: str, bars: list) -> None:
        """List a stock and give it daily bars (StockBar rows, oldest first)."""
        if exchange:
            self.listed.append(
                {
                    "Symbol": f"{code}:x",
                    "Identifier": uic,
                    "AssetType": "Stock",
                    "ExchangeId": exchange,
                }
            )
        self.daily_stock[uic] = [
            {
                "Time": b.day + "T00:00:00.000000Z",
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume,
            }
            for b in bars
        ]

    def stock_bars_upto(self, uic: int, count: int) -> list[dict]:
        rows = [b for b in self.daily_stock.get(uic, []) if b["Time"][:10] <= self.stock_upto]
        return rows[-count:]

    def cfd_quote(self, uic: int, bid: float, ask: float, state: str = "Open") -> None:
        self.quotes[uic] = {
            "Bid": bid,
            "Ask": ask,
            "DelayedByMinutes": 0,
            "MarketState": state,
            "PriceTypeBid": "Tradable",
        }

    def qqq(self, bid: float, ask: float) -> None:
        self.quotes[QQQ] = {
            "Bid": bid,
            "Ask": ask,
            "DelayedByMinutes": self.qqq_delay,
            "MarketState": "Open",
            "PriceTypeBid": "Tradable",
        }

    def chain(self, expiry: date, strikes: list[float], centre: float, base_uic: int) -> None:
        """Calls and puts whose delta falls 0.08 per dollar away from `centre`."""
        listed = []
        for i, k in enumerate(strikes):
            for j, pc in enumerate(("Call", "Put")):
                uic = base_uic + 2 * i + j
                listed.append({"Uic": uic, "StrikePrice": k, "PutCall": pc})
                call_delta = max(0.02, min(0.98, 0.5 - 0.08 * (k - centre)))
                delta = call_delta if pc == "Call" else call_delta - 1
                value = max(0.05, 2.0 * abs(delta))
                self.option_quotes[uic] = {
                    "Bid": round(value - 0.02, 2),
                    "Ask": round(value + 0.02, 2),
                    "Delta": delta,
                }
        self.options[expiry] = listed
        if expiry not in self.expiries:
            self.expiries.append(expiry)

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        path = request.url.path.removeprefix("/sim/openapi")
        self.requests.append((request.method, path, params))
        if path == "/ref/v1/exchanges/NASDAQ":
            return httpx.Response(200, json={"ExchangeSessions": self.sessions})
        if path == "/ref/v1/instruments":
            rows = [r for r in self.listed if r["ExchangeId"] == params.get("ExchangeId")]
            return httpx.Response(200, json={"Data": rows})
        if path == "/root/v1/sessions/capabilities":
            return httpx.Response(200, json={"TradeLevel": "OrdersOnly", "DataLevel": "Standard"})
        if path.startswith("/ref/v1/instruments/details/"):
            uic = int(path.split("/")[5])
            return httpx.Response(
                200, json={"AmountDecimals": 2 if uic == NAS else 1, "PriceToContractFactor": 1.0}
            )
        if path == "/chart/v3/charts":
            uic = int(params["Uic"])
            if params["Horizon"] == "1440" and uic in self.daily_stock:
                data = self.stock_bars_upto(uic, int(params.get("Count", 300)))
                return httpx.Response(
                    200, json={"Data": data, "ChartInfo": {"DelayedByMinutes": 0}}
                )
            if params["AssetType"] == "Etf" and params["Horizon"] == "5":
                upto = (self.qqq_upto or datetime.max.replace(tzinfo=UTC)) - timedelta(
                    minutes=self.qqq_chart_delay
                )
                data = [b for b in self.qqq_bars if datetime.fromisoformat(b["Time"]) <= upto]
                return httpx.Response(
                    200,
                    json={"Data": data, "ChartInfo": {"DelayedByMinutes": self.qqq_chart_delay}},
                )
            if params["AssetType"] == "Etf":
                return httpx.Response(
                    200, json={"Data": self.daily, "ChartInfo": {"DelayedByMinutes": 15}}
                )
            return httpx.Response(
                200,
                json={
                    "Data": self.bars.get(uic, []),
                    "ChartInfo": {"DelayedByMinutes": self.chart_delay},
                },
            )
        if path == "/trade/v1/infoprices/list":
            data = []
            for uic in (int(u) for u in params["Uics"].split(",")):
                if params.get("AssetType") == "Stock" and uic in self.daily_stock:
                    bars = self.stock_bars_upto(uic, 2)
                    if bars:
                        last = bars[-1]
                        data.append(
                            {
                                "Uic": uic,
                                "Quote": {
                                    "Bid": last["Close"],
                                    "Ask": last["Close"],
                                    "DelayedByMinutes": 0,
                                },
                                "PriceInfoDetails": {
                                    "LastTraded": last["Close"],
                                    "Volume": last["Volume"],
                                    "Open": last["Open"],
                                    "LastClose": bars[0]["Close"],
                                },
                            }
                        )
                    continue
                if uic in self.quotes:
                    data.append({"Uic": uic, "Quote": self.quotes[uic], "LastUpdated": "fixture"})
                elif uic in self.option_quotes:
                    q = self.option_quotes[uic]
                    data.append(
                        {
                            "Uic": uic,
                            "Quote": {
                                "Bid": q["Bid"],
                                "Ask": q["Ask"],
                                "DelayedByMinutes": self.qqq_delay,
                                "MarketState": "Open",
                            },
                            "Greeks": {"Delta": q["Delta"], "MidVol": self.mid_vol},
                            "Commissions": {"CostBuy": 2.04, "CostSell": 2.04},
                        }
                    )
            return httpx.Response(200, json={"Data": data})
        if path == "/ref/v1/instruments/contractoptionspaces/306":
            if params.get("OptionSpaceSegment") == "SpecificDates":
                expiry = date.fromisoformat(params["ExpiryDates"])
                return httpx.Response(
                    200,
                    json={
                        "OptionSpace": [
                            {
                                "Expiry": expiry.isoformat() + "T00:00:00",
                                "SpecificOptions": self.options.get(expiry, []),
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "OptionSpace": [
                        {"Expiry": e.isoformat() + "T00:00:00"} for e in sorted(self.expiries)
                    ]
                },
            )
        return httpx.Response(404, json={"ErrorCode": "FIXTURE_UNKNOWN_PATH"})


def client(tmp_path, fake: FakeSaxo) -> SaxoClient:
    path = tmp_path / "credentials.json"
    atomic_json(
        path,
        {
            "environment": "SAXO_SIM",
            "client_id": "fixture-key",
            "client_secret": "fixture-secret",
            "account_key": "fixture-account",
        },
    )
    oauth = OAuth("SAXO_SIM", SaxoSettings(credentials_file=path), tmp_path)
    oauth.tokens = {
        "environment": "SAXO_SIM",
        "access_token": "fixture-access",
        "refresh_token": "fixture-refresh",
        "expires_at": time.time() + 3600,
        "refresh_expires_at": time.time() + 7200,
    }
    return SaxoClient(oauth, httpx.MockTransport(fake.handle))
