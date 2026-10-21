"""The one data source outside Saxo: EODHD's official daily bars for US stocks, read after the
close for the stock dips (amendment 5). GET only; the token comes from a mode-0600 file and is
never logged. Each failure is a coded reason; nothing is retried here."""

import asyncio
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from stocker_execution.methods import StockBar

BASE = "https://eodhd.com/api"
REQUEST_SPACING = 0.12  # seconds between requests (EODHD allows far more; the plan counts calls)
RESPONSE_LIMIT = 48 * 1024**2  # the bulk file is about 7 MB
HISTORY_DAYS = 500  # calendar days of history fetched for a stock new to the record


class EodhdError(Exception):
    """A coded reason a read failed (never the server's text, which could hold the token)."""


class EodhdClient:
    def __init__(self, token_file: Path | None, transport: httpx.AsyncBaseTransport | None = None):
        self.token_file = token_file
        self.http = httpx.AsyncClient(transport=transport, timeout=60, follow_redirects=False)
        self.next_request = 0.0
        self.pace = asyncio.Lock()
        self.calls = 0

    def token(self) -> str:
        if self.token_file is None:
            raise EodhdError("EODHD_NOT_CONFIGURED")
        try:
            value = self.token_file.read_text().strip()
        except OSError:
            raise EodhdError("EODHD_TOKEN_UNREADABLE") from None
        if not value:
            raise EodhdError("EODHD_TOKEN_EMPTY")
        return value

    async def get(self, path: str, params: dict[str, Any]) -> Any:
        token = self.token()
        async with self.pace:
            await asyncio.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + REQUEST_SPACING
        try:
            async with self.http.stream(
                "GET", BASE + path, params={**params, "api_token": token, "fmt": "json"}
            ) as streamed:
                content = bytearray()
                async for chunk in streamed.aiter_bytes():
                    if len(content) + len(chunk) > RESPONSE_LIMIT:
                        raise EodhdError("EODHD_RESPONSE_LIMIT")
                    content.extend(chunk)
                status = streamed.status_code
        except httpx.HTTPError:
            raise EodhdError("EODHD_TRANSPORT_UNAVAILABLE") from None
        self.calls += 1
        if status == 404:
            return []
        if status >= 300:
            raise EodhdError(f"EODHD_HTTP_{status}")
        try:
            return json.loads(bytes(content))
        except ValueError:
            raise EodhdError("EODHD_INVALID_RESPONSE") from None

    async def close(self) -> None:
        await self.http.aclose()


def bar_from(row: dict[str, Any]) -> StockBar | None:
    try:
        bar = StockBar(
            str(row["date"])[:10],
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["adjusted_close"]),
            float(row["volume"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    ok = (
        0 < bar.low <= bar.high
        and bar.low <= bar.open <= bar.high
        and bar.low <= bar.close <= bar.high
        and bar.adj > 0
        and bar.volume >= 0
    )
    return bar if ok else None


async def day_bars(client: EodhdClient, day: date) -> dict[str, StockBar]:
    """Every US stock's bar for `day` from the bulk end-of-day file; empty until published."""
    rows = await client.get("/eod-bulk-last-day/US", {"date": day.isoformat()})
    out: dict[str, StockBar] = {}
    for row in rows if isinstance(rows, list) else []:
        bar = bar_from(row)
        code = str(row.get("code", ""))
        if bar is not None and bar.day == day.isoformat() and code.isalnum():
            out[code] = bar
    return out


async def history(client: EodhdClient, code: str, through: date) -> list[StockBar]:
    """A stock's daily bars, oldest first, for the HISTORY_DAYS before `through` inclusive."""
    rows = await client.get(
        f"/eod/{code}.US",
        {
            "from": (through - timedelta(days=HISTORY_DAYS)).isoformat(),
            "to": through.isoformat(),
        },
    )
    bars = [bar_from(r) for r in rows] if isinstance(rows, list) else []
    return sorted((b for b in bars if b is not None), key=lambda b: b.day)
