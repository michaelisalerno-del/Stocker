"""Explicit endpoint permissions. The LIVE client cannot transmit broker mutations."""

import asyncio
import re
import time
from typing import Any

import httpx

from stocker_execution.saxo_auth import OAuth

READ_PATHS = (
    r"/root/v2/user",
    r"/root/v1/sessions/capabilities",
    r"/port/v1/accounts/me",
    r"/port/v1/clients/me",
    r"/port/v1/balances",
    r"/port/v1/(orders|positions|netpositions)/me",
    r"/ref/v1/instruments",
    r"/ref/v1/instruments/details/[0-9]+/(ContractFutures|FuturesOption|FxSpot)",
    r"/ref/v1/instruments/contractoptionspaces/[0-9]+",
    r"/cs/v1/tradingconditions/ContractOptionSpaces/(?:[A-Za-z0-9_=-]|%(?:2F|2B|3D|7C))+/[0-9]+",
    r"/ref/v1/exchanges/[A-Za-z0-9_-]+",
    r"/ref/v1/currencypairs",
    r"/trade/v1/(prices|infoprices)",
    r"/chart/v1/charts",
    r"/cs/v1/audit/orderactivities",
    r"/ens/v1/activities",
)
SUBSCRIPTIONS = (
    "/port/v1/balances/subscriptions",
    "/trade/v1/prices/subscriptions",
    "/trade/v1/optionschain/subscriptions",
    "/root/v1/sessions/events/subscriptions",
    "/chart/v1/charts/subscriptions",
)


class SaxoError(ValueError):
    pass


def allowed(method: str, path: str, *, sim_orders: bool = False) -> bool:
    # Accept canonical local paths only; no URL, traversal, escaping or query ambiguity.
    if not re.fullmatch(r"/[A-Za-z0-9/%=_-]+", path) or "//" in path:
        return False
    if method == "GET":
        return any(re.fullmatch(p, path) for p in READ_PATHS)
    if method == "POST" and path in SUBSCRIPTIONS:
        return True
    if method == "PATCH" and re.fullmatch(
        r"/trade/v1/optionschain/subscriptions/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", path
    ):
        return True
    if method == "DELETE" and any(
        re.fullmatch(re.escape(p) + r"/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", path) for p in SUBSCRIPTIONS
    ):
        return True
    # Session upgrade is intentionally absent. Reconnect explicitly in Saxo's UI if needed.
    return sim_orders and (
        (method == "POST" and path in {"/trade/v2/orders", "/trade/v2/orders/precheck"})
        or (
            method == "DELETE"
            and re.fullmatch(r"/trade/v2/orders/[A-Za-z0-9_-]+", path) is not None
        )
    )


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
        self.sim_account_verified = False

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        execution: bool = False,
    ) -> dict[str, Any]:
        sim_orders = execution and self.environment == "SAXO_SIM" and self.sim_account_verified
        if not allowed(method, path, sim_orders=sim_orders):
            raise SaxoError("ENDPOINT_BLOCKED_LIVE_ORDERS_DISABLED")
        if execution and (
            not sim_orders or (body or params or {}).get("AccountKey") != self.oauth.account_key
        ):
            raise SaxoError("SIM_ACCOUNT_NOT_VERIFIED")
        if (
            execution
            and method == "POST"
            and (
                not body
                or body.get("AssetType") != "FuturesOption"
                or body.get("Amount") != 1
                or (body.get("BuySell"), body.get("ToOpenClose"))
                not in {("Buy", "ToOpen"), ("Sell", "ToClose")}
            )
        ):
            raise SaxoError("ONLY_LONG_FUTURES_OPTION_ORDERS")
        if self.waiters >= 32:
            raise SaxoError("REST_QUEUE_LIMIT")
        self.waiters += 1
        try:
            for attempt in range(3 if method == "GET" else 1):
                token = await self.oauth.access_token()
                async with self.pace:
                    await asyncio.sleep(max(0, self.next_request - time.monotonic()))
                    self.next_request = time.monotonic() + 0.55
                try:
                    async with self.http.stream(
                        method,
                        self.oauth.urls["rest"] + path,
                        params=params,
                        json=body,
                        headers={"Authorization": "Bearer " + token},
                    ) as streamed:
                        content = bytearray()
                        async for chunk in streamed.aiter_bytes():
                            if len(content) + len(chunk) > 4 * 1024**2:
                                raise SaxoError("REST_RESPONSE_LIMIT")
                            content.extend(chunk)
                        response = httpx.Response(
                            streamed.status_code,
                            # aiter_bytes already decoded the wire compression. Retaining
                            # its encoding header would decompress the JSON a second time.
                            headers={
                                k: v
                                for k, v in streamed.headers.items()
                                if k.lower() not in {"content-encoding", "content-length"}
                            },
                            content=bytes(content),
                        )
                except httpx.HTTPError:
                    raise SaxoError(
                        "AMBIGUOUS_REQUEST" if execution else "SAXO_TRANSPORT_UNAVAILABLE"
                    ) from None
                self.calls += 1
                self.rate_headers = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower().startswith("x-ratelimit-")
                }
                if response.status_code == 429:
                    reset = response.headers.get("x-ratelimit-session-reset", "5")
                    try:
                        seconds = min(60.0, max(1.0, float(reset)))
                    except ValueError:
                        seconds = 5.0
                    self.next_request = max(self.next_request, time.monotonic() + seconds)
                    if method == "GET" and attempt < 2:
                        continue
                if response.status_code == 401:
                    self.oauth.status = "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED"
                if response.status_code >= 300:
                    # Restrict error evidence to code; server text can contain private identifiers.
                    code = "HTTP_" + str(response.status_code)
                    try:
                        candidate = str(response.json().get("ErrorCode", code))
                        if re.fullmatch(r"[A-Za-z0-9_]{1,100}", candidate):
                            code = candidate
                    except (ValueError, AttributeError):
                        pass
                    raise SaxoError(code)
                if not response.content:
                    return {}
                if len(response.content) > 4 * 1024**2:
                    raise SaxoError("REST_RESPONSE_LIMIT")
                result = response.json()
                if not isinstance(result, dict):
                    raise SaxoError("INVALID_SAXO_RESPONSE")
                return result
            raise SaxoError("RATE_LIMIT_REACHED")
        finally:
            self.waiters -= 1

    async def close(self) -> None:
        await self.http.aclose()
        await self.oauth.close()
