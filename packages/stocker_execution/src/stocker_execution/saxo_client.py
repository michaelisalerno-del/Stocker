"""Explicit endpoint permissions. The LIVE client cannot transmit broker mutations."""

import asyncio
import json
import re
import time
from typing import Any

import httpx

from stocker_execution.saxo_auth import OAuth, SaxoError

READ_PATHS = (
    r"/root/v2/user",
    r"/root/v1/sessions/capabilities",
    r"/port/v1/accounts/me",
    r"/port/v1/clients/me",
    r"/port/v1/balances",
    r"/port/v1/(orders|positions|netpositions)/me",
    r"/port/v1/closedpositions",
    r"/ref/v1/instruments",
    r"/ref/v1/instruments/details/[0-9]+/(ContractFutures|FuturesOption|FxSpot)",
    r"/ref/v1/instruments/contractoptionspaces/[0-9]+",
    r"/cs/v1/tradingconditions/ContractOptionSpaces/(?:[A-Za-z0-9_=-]|%(?:2F|2B|3D|7C))+/[0-9]+",
    r"/ref/v1/exchanges/[A-Za-z0-9_-]+",
    r"/ref/v1/currencypairs",
    r"/trade/v1/(prices|infoprices)",
    r"/chart/v3/charts",
    r"/cs/v1/audit/orderactivities",
    r"/ens/v1/activities",
)
REST_QUEUE_LIMIT = 32  # requests waiting for the pacer before new ones are refused
SUBSCRIPTIONS = (
    "/port/v1/balances/subscriptions",
    "/trade/v1/prices/subscriptions",
    "/trade/v1/optionschain/subscriptions",
    "/root/v1/sessions/events/subscriptions",
    "/chart/v3/charts/subscriptions",
    "/ens/v1/activities/subscriptions",
)


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
        self.sim_account_verified = False

    async def authorize_stream(self, context: str) -> None:
        """Bind a renewed access token to the open streaming context (Saxo: 202 Accepted).

        This is the streaming host, not the REST gateway, so it sits outside the allow-list.
        """
        token = await self.oauth.access_token()
        try:
            response = await self.http.put(
                self.oauth.urls["stream"].replace("wss://", "https://", 1) + "/authorize",
                params={"contextid": context},
                headers={"Authorization": "Bearer " + token},
            )
        except httpx.HTTPError:
            raise SaxoError("STREAM_REAUTHORISATION_UNAVAILABLE") from None
        if response.status_code != 202:
            raise SaxoError("STREAM_REAUTHORISATION_HTTP_" + str(response.status_code))

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
        if self.waiters >= REST_QUEUE_LIMIT:
            raise SaxoError("REST_QUEUE_LIMIT")
        self.waiters += 1
        try:
            attempt = 0
            while True:
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
                        # aiter_bytes decodes the wire compression once; the body is
                        # bounded here so a runaway response never fills memory.
                        content = bytearray()
                        async for chunk in streamed.aiter_bytes():
                            if len(content) + len(chunk) > 4 * 1024**2:
                                raise SaxoError("REST_RESPONSE_LIMIT")
                            content.extend(chunk)
                        status, headers = streamed.status_code, streamed.headers
                except httpx.HTTPError:
                    raise SaxoError(
                        "AMBIGUOUS_REQUEST" if execution else "SAXO_TRANSPORT_UNAVAILABLE"
                    ) from None
                self.calls += 1
                self.rate_headers = {
                    k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit-")
                }
                if status == 429:
                    seconds = retry_after(headers)
                    self.next_request = max(self.next_request, time.monotonic() + seconds)
                    if method == "GET" and attempt < 2:  # reads retry twice; writes never
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
