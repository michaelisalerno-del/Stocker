"""The Saxo reads the paper record needs, parsed into plain values. GET only."""

from dataclasses import dataclass
from datetime import date
from typing import Any

from stocker_execution.methods import Bar, Candle, OptionQuote, Session, trading_sessions, utc
from stocker_execution.saxo_client import SaxoClient

LIST_CHUNK = 50  # instruments per infoprices/list request


@dataclass(frozen=True)
class Quote:
    bid: float | None
    ask: float | None
    delayed: int | None
    market_state: str | None
    price_type: str | None
    last_updated: str | None

    @property
    def mid(self) -> float | None:
        return (self.bid + self.ask) / 2 if self.bid and self.ask else None

    @property
    def live(self) -> bool:
        """A two-sided, undelayed, tradable quote in an open market."""
        return bool(
            self.bid
            and self.ask
            and 0 < self.bid <= self.ask
            and self.delayed == 0
            and self.market_state == "Open"
            and self.price_type == "Tradable"
        )


def quote_from(row: dict[str, Any]) -> Quote:
    q = row.get("Quote") or {}
    return Quote(
        q.get("Bid"),
        q.get("Ask"),
        q.get("DelayedByMinutes"),
        q.get("MarketState"),
        q.get("PriceTypeBid"),
        row.get("LastUpdated"),
    )


async def info_rows(
    client: SaxoClient, uics: list[int], asset_type: str, field_groups: str = "Quote"
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for i in range(0, len(uics), LIST_CHUNK):
        result = await client.get(
            "/trade/v1/infoprices/list",
            {
                "AccountKey": client.oauth.account_key,
                "Uics": ",".join(str(u) for u in uics[i : i + LIST_CHUNK]),
                "AssetType": asset_type,
                "Amount": 1,
                "FieldGroups": field_groups,
            },
        )
        for row in result.get("Data") or []:
            if isinstance(row.get("Uic"), int):
                rows[row["Uic"]] = row
    return rows


async def quotes(client: SaxoClient, uics: list[int], asset_type: str) -> dict[int, Quote]:
    return {u: quote_from(r) for u, r in (await info_rows(client, uics, asset_type)).items()}


async def cfd_bars(client: SaxoClient, uic: int, count: int = 200) -> tuple[list[Bar], int | None]:
    """5-minute CFD candles at the bid/ask midpoint, and the chart's delay in minutes."""
    result = await client.get(
        "/chart/v3/charts",
        {
            "AssetType": "CfdOnIndex",
            "Uic": uic,
            "Horizon": 5,
            "Count": count,
            "FieldGroups": "Data,ChartInfo",
        },
    )
    bars = [
        Bar(utc(x["Time"]), (x["CloseBid"] + x["CloseAsk"]) / 2)
        for x in result.get("Data") or []
        if x.get("Time") and x.get("CloseBid") and x.get("CloseAsk")
    ]
    return bars, (result.get("ChartInfo") or {}).get("DelayedByMinutes")


async def etf_candles(
    client: SaxoClient, uic: int, count: int = 400
) -> tuple[list[Candle], int | None]:
    """5-minute ETF candles (high, low, close, volume; regular session) and the chart's delay."""
    result = await client.get(
        "/chart/v3/charts",
        {
            "AssetType": "Etf",
            "Uic": uic,
            "Horizon": 5,
            "Count": count,
            "FieldGroups": "Data,ChartInfo",
        },
    )
    candles = [
        Candle(
            utc(x["Time"]), float(x["High"]), float(x["Low"]), float(x["Close"]), float(x["Volume"])
        )
        for x in result.get("Data") or []
        if x.get("Time")
        and x.get("High")
        and x.get("Low")
        and x.get("Close")
        and x.get("Volume") is not None
    ]
    return candles, (result.get("ChartInfo") or {}).get("DelayedByMinutes")


async def chart_rows(
    client: SaxoClient, asset_type: str, uic: int, count: int = 200
) -> tuple[list[list[Any]], int | None]:
    """5-minute candles for the dashboard's chart as [start, open, high, low, close, volume]:
    a CFD's prices at the bid/ask midpoint with no volume, an ETF's as Saxo gives them."""
    result = await client.get(
        "/chart/v3/charts",
        {
            "AssetType": asset_type,
            "Uic": uic,
            "Horizon": 5,
            "Count": count,
            "FieldGroups": "Data,ChartInfo",
        },
    )
    rows: list[list[Any]] = []
    for x in result.get("Data") or []:
        if not x.get("Time"):
            continue
        if asset_type == "Etf":
            keys = ("Open", "High", "Low", "Close")
            if any(x.get(k) is None for k in keys):
                continue
            rows.append([x["Time"], *(float(x[k]) for k in keys), float(x.get("Volume") or 0)])
        else:
            sides = [(f"{k}Bid", f"{k}Ask") for k in ("Open", "High", "Low", "Close")]
            if any(x.get(b) is None or x.get(a) is None for b, a in sides):
                continue
            rows.append([x["Time"], *((x[b] + x[a]) / 2 for b, a in sides), 0.0])
    return rows, (result.get("ChartInfo") or {}).get("DelayedByMinutes")


async def daily_closes(
    client: SaxoClient, uic: int, before: date, count: int = 260
) -> list[tuple[date, float]]:
    """QQQ's official daily closes (day, close) before `before`, oldest first."""
    result = await client.get(
        "/chart/v3/charts",
        {
            "AssetType": "Etf",
            "Uic": uic,
            "Horizon": 1440,
            "Count": count,
            "FieldGroups": "Data,ChartInfo",
        },
    )
    rows = sorted(
        (utc(x["Time"]).date(), float(x["Close"]))
        for x in result.get("Data") or []
        if x.get("Time") and x.get("Close")
    )
    return [(day, close) for day, close in rows if day < before]


async def cfd_details(client: SaxoClient, uic: int) -> tuple[int, float]:
    """(amount decimals, price-to-contract factor) of an index CFD."""
    d = await client.get(f"/ref/v1/instruments/details/{uic}/CfdOnIndex")
    return int(d["AmountDecimals"]), float(d["PriceToContractFactor"])


async def sessions(client: SaxoClient, exchange: str = "NASDAQ") -> list[Session]:
    return trading_sessions(await client.get(f"/ref/v1/exchanges/{exchange}"))


async def expiries(client: SaxoClient, root: int) -> list[date]:
    space = await client.get(f"/ref/v1/instruments/contractoptionspaces/{root}")
    return sorted(
        {
            date.fromisoformat(str(e["Expiry"])[:10])
            for e in space.get("OptionSpace") or []
            if e.get("Expiry")
        }
    )


async def series(client: SaxoClient, root: int, expiry: date) -> list[dict[str, Any]]:
    """Every listed option (Uic, StrikePrice, PutCall) of one expiry."""
    space = await client.get(
        f"/ref/v1/instruments/contractoptionspaces/{root}",
        {"OptionSpaceSegment": "SpecificDates", "ExpiryDates": expiry.isoformat()},
    )
    for e in space.get("OptionSpace") or []:
        if str(e.get("Expiry", ""))[:10] == expiry.isoformat():
            return [
                o
                for o in e.get("SpecificOptions") or []
                if o.get("Uic") and o.get("StrikePrice") and o.get("PutCall")
            ]
    return []


def option_from(o: dict[str, Any], row: dict[str, Any]) -> OptionQuote:
    q, g, c = row.get("Quote") or {}, row.get("Greeks") or {}, row.get("Commissions") or {}
    return OptionQuote(
        int(o["Uic"]),
        float(o["StrikePrice"]),
        str(o["PutCall"]),
        q.get("Bid"),
        q.get("Ask"),
        g.get("Delta"),
        g.get("MidVol"),
        c.get("CostBuy"),
        c.get("CostSell"),
        q.get("DelayedByMinutes"),
        q.get("MarketState"),
    )


async def board(
    client: SaxoClient, listed: list[dict[str, Any]], centre: float, half_width: int
) -> tuple[list[OptionQuote], dict[int, dict[str, Any]]]:
    """Calls and puts on the strikes nearest `centre`, with quotes, Greeks and commissions."""
    strikes = sorted({float(o["StrikePrice"]) for o in listed})
    if not strikes:
        return [], {}
    i = min(range(len(strikes)), key=lambda k: abs(strikes[k] - centre))
    keep = set(strikes[max(0, i - half_width) : i + half_width + 1])
    chosen = [o for o in listed if float(o["StrikePrice"]) in keep]
    rows = await info_rows(
        client, [int(o["Uic"]) for o in chosen], "StockOption", "Quote,Greeks,Commissions"
    )
    return [option_from(o, rows[int(o["Uic"])]) for o in chosen if int(o["Uic"]) in rows], rows


async def option_quote(client: SaxoClient, option: dict[str, Any]) -> OptionQuote | None:
    rows = await info_rows(client, [int(option["Uic"])], "StockOption", "Quote,Greeks,Commissions")
    row = rows.get(int(option["Uic"]))
    return option_from(option, row) if row else None
