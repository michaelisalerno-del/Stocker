"""Read-only Saxo access: an explicit list of GET endpoints and nothing else.

The paper record never places, changes or cancels an order and never touches the session, so it
can never take the user's real-time slot from SaxoTraderGO.
"""

import asyncio
import json
import re
import time
from typing import Any

import httpx

from stocker_execution.saxo_auth import OAuth, SaxoError, SaxoRefused

READ_PATHS = (
    r"/root/v1/sessions/capabilities",
    r"/port/v1/accounts/me",
    r"/ref/v1/instruments/details/[0-9]+/(CfdOnIndex|Etf|StockOption)",
    r"/ref/v1/instruments/contractoptionspaces/[0-9]+",
    r"/ref/v1/exchanges/[A-Za-z0-9_-]+",
    r"/trade/v1/infoprices(/list)?",
    r"/chart/v3/charts",
)
REST_QUEUE_LIMIT = 32  # requests waiting for the pacer before new ones are refused
REQUEST_SPACING = 0.55  # seconds between requests, inside Saxo's per-session limit


def allowed(method: str, path: str) -> bool:
    # Accept canonical local paths only; no URL, traversal, escaping or query ambiguity.
    if not re.fullmatch(r"/[A-Za-z0-9/%=_-]+", path) or "//" in path:
        return False
    return method == "GET" and any(re.fullmatch(p, path) for p in READ_PATHS)


def retry_after(headers: Any) -> float:
    """Seconds until the exhausted Saxo limit resets, 1-60 s.

    Saxo names each limit (for example RefDataInstrumentsMinute, 60 a minute, shared by reference
    details and option spaces); the one whose remaining count is zero decides the wait. The session
    limit's reset applies when none is named.
    """
    names = [
        k.lower().removesuffix("-remaining") + "-reset"
        for k, v in headers.items()
        if k.lower().startswith("x-ratelimit-")
        and k.lower().endswith("-remaining")
        and str(v).strip() == "0"
    ] or ["x-ratelimit-session-reset"]
    waits = []
    for name in names:
        try:
            waits.append(float(headers.get(name, "5")))
        except ValueError:
            waits.append(5.0)
    return min(60.0, max(1.0, max(waits)))


class SaxoClient:
    def __init__(self, oauth: OAuth, transport: httpx.AsyncBaseTransport | None = None):
        self.oauth = oauth
        self.environment = oauth.environment
        self.http = httpx.AsyncClient(transport=transport, timeout=15, follow_redirects=False)
        self.rate_headers: dict[str, str] = {}
        self.next_request = 0.0
        self.pace = asyncio.Lock()
        self.waiters = 0
        self.calls = 0

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not allowed("GET", path):
            raise SaxoRefused("ENDPOINT_NOT_ALLOWED")
        if self.waiters >= REST_QUEUE_LIMIT:
            raise SaxoRefused("REST_QUEUE_LIMIT")
        self.waiters += 1
        try:
            attempt = 0
            while True:
                try:
                    token = await self.oauth.access_token()
                except SaxoError as exc:
                    raise SaxoRefused(str(exc)) from None  # nothing has been sent
                async with self.pace:
                    await asyncio.sleep(max(0, self.next_request - time.monotonic()))
                    self.next_request = time.monotonic() + REQUEST_SPACING
                try:
                    async with self.http.stream(
                        "GET",
                        self.oauth.urls["rest"] + path,
                        params=params,
                        headers={"Authorization": "Bearer " + token},
                    ) as streamed:
                        # aiter_bytes decodes the wire compression once; the body is
                        # bounded here so a runaway response never fills memory.
                        content = bytearray()
                        async for chunk in streamed.aiter_bytes():
                            if len(content) + len(chunk) > 4 * 1024**2:
                                raise SaxoError("REST_RESPONSE_LIMIT")
                            content.extend(chunk)
                        status, headers = streamed.status_code, streamed.headers
                except httpx.HTTPError:
                    raise SaxoError("SAXO_TRANSPORT_UNAVAILABLE") from None
                self.calls += 1
                self.rate_headers = {
                    k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit-")
                }
                if status == 429:
                    seconds = retry_after(headers)
                    self.next_request = max(self.next_request, time.monotonic() + seconds)
                    if attempt < 2:
                        attempt += 1
                        continue
                if status == 401:
                    # The server rejects a token our clock still trusts: refresh it next.
                    self.oauth.expire()
                if status >= 300:
                    # Restrict error evidence to code; server text can contain private identifiers.
                    code = "HTTP_" + str(status)
                    try:
                        candidate = str(json.loads(bytes(content)).get("ErrorCode", code))
                        if re.fullmatch(r"[A-Za-z0-9_]{1,100}", candidate):
                            code = candidate
                    except (ValueError, AttributeError):
                        pass
                    raise SaxoError(code)
                if not content:
                    return {}
                result = json.loads(bytes(content))
                if not isinstance(result, dict):
                    raise SaxoError("INVALID_SAXO_RESPONSE")
                return result
        finally:
            self.waiters -= 1

    async def close(self) -> None:
        await self.http.aclose()
        await self.oauth.close()
