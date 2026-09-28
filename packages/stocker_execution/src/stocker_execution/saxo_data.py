"""One server-owned Saxo stream; browsers only read its state."""

import asyncio
import json
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from stocker_execution.bar_cache import BarCache
from stocker_execution.config import MARKETS, FuturesConfig, Market
from stocker_execution.contracts import future_identity, key, option_identity, quote_check, utc
from stocker_execution.recorder import Recorder
from stocker_execution.reference_sessions import load_selections
from stocker_execution.rules import NY, Bar, reference_summary
from stocker_execution.saxo_client import SaxoClient, SaxoError
from stocker_execution.saxo_stream import Frames, PriceState, merge, merge_board


@dataclass
class MarketState:
    market: Market
    identity: dict[str, Any] | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    price: PriceState = field(default_factory=PriceState)
    bars: list[Bar] = field(default_factory=list)
    references: list[dict[int, dict[str, float]]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(
        default_factory=lambda: {
            "discovery": "UNVERIFIED",
            "permission": "UNVERIFIED",
            "quote": "UNVERIFIED",
            "l2": "UNVERIFIED",
            "options": "UNVERIFIED",
            "history": "UNVERIFIED",
        }
    )
    option_space: list[dict[str, Any]] = field(default_factory=list)
    option_board: dict[str, Any] = field(default_factory=dict)
    option_root: int | None = None
    problem: str = "AUTHENTICATION_REQUIRED"
    history_problem: str = "SAXO_HISTORY_NOT_VERIFIED"
    last_clock: datetime | None = None
    live_since: datetime | None = None
    history_checked: float = 0
    reference_day: str = ""


def completed_bars(raw: dict[str, Any], at: datetime) -> list[Bar]:
    rows = sorted(raw.get("Data", []), key=lambda b: b.get("Time", ""))
    result = []
    # A later sample proves the mutable tail has rolled, even when the wall clock has advanced.
    for row in rows[:-1]:
        try:
            stamp = utc(row["Time"])
            if stamp + timedelta(minutes=1) > at:
                continue
            # Volume absence is a strategy block, not a synthetic zero.
            b = Bar(
                stamp,
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Volume"]),
            )
            if b.valid():
                result.append(b)
        except (KeyError, TypeError, ValueError):
            continue
    return result


class DataService:
    def __init__(self, config: FuturesConfig, client: SaxoClient, recorder: Recorder):
        self.config, self.client, self.recorder = config, client, recorder
        self.markets: dict[str, MarketState] = {m: MarketState(m) for m in MARKETS}
        self.context = ""
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.options: dict[int, tuple[dict[str, Any], PriceState]] = {}
        self.option_required_at: dict[int, float] = {}
        self.session: dict[str, Any] = {}
        self.fx = PriceState()
        self.fx_uic: int | None = None
        self.connected = False
        self.problem = "AUTHENTICATION_REQUIRED"
        self.stopping = False
        self.reconnects = 0
        self.last_message_id: str | None = None
        self.account_verified = False
        self.account_currency: str | None = None
        self.account_id: str | None = None
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.pending_bytes = 0
        self.subscription_lock = asyncio.Lock()
        self.subscribe_lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.bar_cache = BarCache(recorder.directory.parent / "bars", config.recorder)

    async def verify_account(self) -> None:
        self.account_verified = self.client.sim_account_verified = False
        await self.client.request("GET", "/root/v2/user")
        result = await self.client.request("GET", "/port/v1/accounts/me")
        if not self.client.oauth.account_key:
            raise SaxoError("ACCOUNT_SELECTION_REQUIRED")
        accounts = [
            a
            for a in result.get("Data", [])
            if a.get("AccountKey") == self.client.oauth.account_key
        ]
        if len(accounts) != 1:
            raise SaxoError("CONFIGURED_ACCOUNT_NOT_RETURNED_BY_ENVIRONMENT")
        self.account_verified = True
        self.account_currency = accounts[0].get("Currency")
        self.account_id = accounts[0].get("AccountId")
        if not self.account_id:
            raise SaxoError("ACCOUNT_IDENTIFIER_NOT_VERIFIED")
        self.client.sim_account_verified = self.config.data_environment == "SAXO_SIM"
        self.session = await self.client.request("GET", "/root/v1/sessions/capabilities")

    async def discover(self, state: MarketState) -> None:
        result = await self.client.request(
            "GET",
            "/ref/v1/instruments",
            params={
                "Keywords": state.market,
                "AssetTypes": "ContractFutures",
                "$top": 100,
            },
        )
        state.candidates = [
            {
                k: row.get(k)
                for k in ("Identifier", "Symbol", "AssetType", "Description", "ExchangeId")
            }
            for row in result.get("Data", [])[:100]
            if row.get("AssetType") == "ContractFutures"
        ]
        state.capabilities["discovery"] = "AVAILABLE" if state.candidates else "NO_LISTED_FUTURES"
        selection = self.config.contracts.get(state.market)
        if not selection:
            raise SaxoError("REFERENCE_CONTRACT_SELECTION_REQUIRED")
        raw = await self.client.request(
            "GET",
            f"/ref/v1/instruments/details/{selection.uic}/ContractFutures",
            params={"AccountKey": self.client.oauth.account_key, "FieldGroups": "TradingSessions"},
        )
        identity = future_identity(state.market, self.config.data_environment, raw)
        if (
            identity["uic"] != selection.uic
            or identity["symbol"] != selection.symbol
            or identity["exchange"] != selection.exchange
            or identity["contract_month"] != selection.contract_month
        ):
            raise SaxoError("CONFIGURED_REFERENCE_IDENTITY_MISMATCH")
        state.identity, state.reference = identity, raw
        state.capabilities["permission"] = {
            "is_tradable": raw.get("IsTradable"),
            "reason": raw.get("NonTradableReason"),
            "trading_status": raw.get("TradingStatus"),
        }
        self.recorder.register(key(identity), identity)
        roots = [
            r["OptionRootId"]
            for r in raw.get("RelatedOptionRootsEnhanced", [])
            if r.get("AssetType") == "FuturesOption"
        ]
        mapping = self.config.mappings.get(state.market)
        root = (
            mapping.option_root_id
            if mapping and mapping.option_root_id in roots
            else (roots[0] if len(roots) == 1 else None)
        )
        state.option_root = root
        state.capabilities["options"] = "OPTION_ROOT_AMBIGUOUS_OR_UNAVAILABLE"
        if root:
            space = await self.client.request(
                "GET",
                f"/ref/v1/instruments/contractoptionspaces/{root}",
                params={"OptionSpaceSegment": "UnderlyingUic", "UnderlyingUic": identity["uic"]},
            )
            if space.get("AssetType") != "FuturesOption":
                raise SaxoError("WRONG_OPTION_ROOT_ASSET_TYPE")
            state.option_space = [
                {
                    **option,
                    "Expiry": expiry.get("Expiry"),
                    "LastTradeDate": expiry.get("LastTradeDate"),
                }
                for expiry in space.get("OptionSpace", [])
                for option in expiry.get("SpecificOptions", [])
                if option.get("UnderlyingUic") == identity["uic"]
            ][:1000]
            state.capabilities["options"] = {
                "discovered": len(state.option_space),
                "quote": "UNVERIFIED",
                "root": root,
            }
        state.problem = ""

    async def subscribe(
        self,
        kind: str,
        arguments: dict[str, Any],
        target: str,
        old: str | None = None,
    ) -> str:
        async with self.subscribe_lock:
            existing = next(
                (
                    (ref, s)
                    for ref, s in self.subscriptions.items()
                    if s["kind"] == kind and s["target"] == target
                ),
                None,
            )
            if existing and old is None:
                if existing[1]["arguments"] == arguments:
                    return existing[0]
                raise SaxoError("SUBSCRIPTION_ARGUMENT_CONFLICT")
            return await self._subscribe(kind, arguments, target, old)

    async def _subscribe(
        self, kind: str, arguments: dict[str, Any], target: str, old: str | None
    ) -> str:
        if len(self.subscriptions) >= 32 and old is None:
            raise SaxoError("SUBSCRIPTION_LIMIT")
        paths = {
            "PRICE": "/trade/v1/prices/subscriptions",
            "BOARD": "/trade/v1/optionschain/subscriptions",
            "SESSION": "/root/v1/sessions/events/subscriptions",
        }
        ref = "S" + secrets.token_hex(10)
        body: dict[str, Any] = {
            "ContextId": self.context,
            "ReferenceId": ref,
            "Arguments": arguments,
            "RefreshRate": 2000 if kind == "BOARD" else 1000,
            "Format": "application/json",
        }
        if old:
            body["ReplaceReferenceId"] = old
        # A websocket is already reading. Buffer bounded updates until the REST
        # snapshot arrives, then replay in receipt order onto that snapshot.
        self.pending[ref] = []
        try:
            result = await self.client.request("POST", paths[kind], body=body)
        except Exception:
            self.pending_bytes -= sum(len(json.dumps(m)) for m in self.pending.pop(ref, []))
            raise
        self.subscriptions[ref] = {
            "kind": kind,
            "arguments": arguments,
            "target": target,
            "path": paths[kind],
            "refresh_ms": result.get("RefreshRate"),
            "timeout": min(300, max(5, int(result.get("InactivityTimeout", 30)))),
            "contact": time.monotonic(),
        }
        if old:
            self.subscriptions.pop(old, None)
        snapshot = result.get("Snapshot")
        if kind == "PRICE":
            if not isinstance(snapshot, dict):
                await self.client.request("DELETE", paths[kind] + f"/{self.context}/{ref}")
                self.subscriptions.pop(ref, None)
                self.pending_bytes -= sum(len(json.dumps(m)) for m in self.pending.pop(ref, []))
                raise SaxoError("PRICE_SNAPSHOT_MISSING")
            price, identity = self.price_target(target)
            price.refresh_ms = result.get("RefreshRate")
            price.inactivity_timeout = self.subscriptions[ref]["timeout"]
            price.snapshot(snapshot, ref, time.time())
            if identity:
                self.recorder.ingest(
                    key(identity),
                    "SNAPSHOT",
                    snapshot,
                    time.time(),
                    generation=ref,
                    provider_message=snapshot,
                    observation_context=price.observation_context(),
                )
        elif kind == "SESSION":
            self.session = snapshot or {}
        else:
            self.markets[target].option_board = snapshot or {}
        pending = self.pending.pop(ref, [])
        self.pending_bytes -= sum(len(json.dumps(m)) for m in pending)
        for message in pending:
            await self.receive(message["message"], message["receipt"])
        return ref

    def price_target(self, target: str) -> tuple[PriceState, dict[str, Any] | None]:
        if target == "FX":
            return self.fx, None
        if target in self.markets:
            state = self.markets[target]
            return state.price, state.identity
        identity, price = self.options[int(target)]
        return price, identity

    async def option_subscribe(
        self, state: MarketState, selected: dict[str, Any]
    ) -> dict[str, Any]:
        async with self.subscription_lock:
            return await self._option_subscribe(state, selected)

    async def _option_subscribe(
        self, state: MarketState, selected: dict[str, Any]
    ) -> dict[str, Any]:
        assert state.identity is not None and state.option_root is not None
        uic = selected["Uic"]
        self.option_required_at[uic] = time.time()
        if uic in self.options and any(
            s["target"] == str(uic) for s in self.subscriptions.values()
        ):
            return self.options[uic][0]
        if uic not in self.options and len(self.options) >= 16:
            raise SaxoError("OPTION_SUBSCRIPTION_CAPACITY")
        raw = await self.client.request(
            "GET",
            f"/ref/v1/instruments/details/{uic}/FuturesOption",
            params={"AccountKey": self.client.oauth.account_key, "FieldGroups": "TradingSessions"},
        )
        identity = option_identity(state.identity, raw, state.option_root, selected)
        self.options[uic] = (identity, PriceState())
        self.recorder.register(key(identity), identity)
        try:
            await self.subscribe(
                "PRICE",
                {
                    "Uic": uic,
                    "AssetType": "FuturesOption",
                    "Amount": 1,
                    "AccountKey": self.client.oauth.account_key,
                    "ToOpenClose": "ToOpen",
                    "FieldGroups": [
                        "Quote",
                        "PriceInfoDetails",
                        "Greeks",
                        "Commissions",
                        "InstrumentPriceDetails",
                    ],
                },
                str(uic),
            )
        except Exception:
            self.options.pop(uic, None)
            raise
        return identity

    async def history(self, state: MarketState) -> None:
        if not state.identity:
            return
        result = await self.client.request(
            "GET",
            "/chart/v1/charts",
            params={
                "Uic": state.identity["uic"],
                "AssetType": "ContractFutures",
                "Horizon": 1,
                "Count": 1200,
                "FieldGroups": "Data,ChartInfo",
            },
        )
        state.capabilities["history"] = {
            "first_sample": (result.get("ChartInfo") or {}).get("FirstSampleTime"),
            "data_version": result.get("DataVersion"),
            "returned_samples": len(result.get("Data", [])),
            "semantics": "Saxo chart samples; mutable tail excluded; no quote reconstruction",
        }
        bars = completed_bars(result, datetime.now(UTC))
        state.bars = bars[-1440:]
        state.history_checked = time.monotonic()
        try:
            await asyncio.to_thread(
                self.bar_cache.save, state.identity, bars, result.get("DataVersion")
            )
        except (OSError, ValueError):
            self.bar_cache.problem = "BAR_STORAGE_UNAVAILABLE"
        state.history_problem = ""
        if not bars:
            state.history_problem = "SAXO_COMPLETED_OHLCV_UNAVAILABLE"
            return
        day = datetime.now(NY).date()
        if state.reference_day == day.isoformat():
            return
        state.references = []
        if not self.config.reference_selections_file:
            state.history_problem = "REFERENCE_SESSION_CONTRACT_SELECTION_UNVERIFIED"
            return
        try:
            audit, digest = load_selections(
                self.config.reference_selections_file,
                self.config.data_environment,
                state.market,
                day,
                state.identity["uic"],
            )
            references = []
            for selected in audit.sessions:
                raw = await self.client.request(
                    "GET",
                    f"/ref/v1/instruments/details/{selected.contract.uic}/ContractFutures",
                    params={"AccountKey": self.client.oauth.account_key},
                )
                identity = future_identity(state.market, self.config.data_environment, raw)
                if (
                    identity["symbol"] != selected.contract.symbol
                    or identity["exchange"] != selected.contract.exchange
                    or identity["contract_month"] != selected.contract.contract_month
                ):
                    raise ValueError("REFERENCE_SESSION_IDENTITY_MISMATCH")
                start = datetime.combine(selected.day, datetime.min.time(), NY) + timedelta(hours=8)
                prior = await self.history_range(
                    selected.contract.uic, start, start + timedelta(hours=9)
                )
                if len(prior) != 540:
                    raise ValueError("REFERENCE_SESSION_OHLCV_COVERAGE_INCOMPLETE")
                references.append(reference_summary(prior))
            state.references, state.reference_day = references, day.isoformat()
            state.capabilities["reference_selection_audit_sha256"] = digest
        except (OSError, ValueError):
            state.history_problem = "REFERENCE_SESSION_AUDIT_OR_SAXO_COVERAGE_UNVERIFIED"

    async def restore_option(self, plan: dict[str, Any]) -> None:
        """Owned options keep their original future even after a display roll/restart."""
        option = plan["option"]
        state = MarketState(
            market=option["market"],
            identity=plan["underlying"],
            option_root=option["option_root_id"],
        )
        await self.option_subscribe(
            state,
            {
                "Uic": option["uic"],
                "UnderlyingUic": option["underlying_uic"],
                "PutCall": option["right"],
                "StrikePrice": option["strike"],
            },
        )

    async def release_unused_options(self, owned: set[int]) -> None:
        for uic, (identity, _) in list(self.options.items()):
            recording = any(
                key(identity) in c["instruments"] for c in self.recorder.active.values()
            )
            if (
                uic in owned
                or recording
                or time.time() - self.option_required_at.get(uic, 0) <= 900
            ):
                continue
            async with self.subscription_lock:
                for ref, subscription in list(self.subscriptions.items()):
                    if subscription["target"] == str(uic):
                        await self.client.request(
                            "DELETE", subscription["path"] + f"/{self.context}/{ref}"
                        )
                        self.subscriptions.pop(ref, None)
                self.options.pop(uic, None)
                self.option_required_at.pop(uic, None)
                self.recorder.windows.pop(key(identity), None)

    async def history_range(
        self,
        uic: int,
        start: datetime,
        end: datetime,
        *,
        max_pages: int = 8,
    ) -> list[Bar]:
        samples: dict[str, dict[str, Any]] = {}
        cursor, version = start, None
        for _ in range(min(max_pages, 8)):
            result = await self.client.request(
                "GET",
                "/chart/v1/charts",
                params={
                    "Uic": uic,
                    "AssetType": "ContractFutures",
                    "Horizon": 1,
                    "Count": 1200,
                    "Mode": "From",
                    "Time": cursor.isoformat(),
                    "FieldGroups": "Data,ChartInfo",
                },
            )
            if version is not None and result.get("DataVersion") != version:
                raise SaxoError("CHART_VERSION_CHANGED_REFETCH_REQUIRED")
            version = result.get("DataVersion")
            rows = result.get("Data", [])
            if not rows:
                break
            samples.update({r["Time"]: r for r in rows})
            newest = max(utc(r["Time"]) for r in rows)
            if newest >= end:
                break
            if newest <= cursor:
                raise SaxoError("CHART_PAGINATION_DID_NOT_ADVANCE")
            cursor = newest
        return [
            b
            for b in completed_bars({"Data": list(samples.values())}, end + timedelta(minutes=1))
            if start <= b.at < end
        ]

    async def startup(self) -> None:
        await self.subscribe("SESSION", {}, "SESSION")
        for state in self.markets.values():
            try:
                await self.discover(state)
            except SaxoError as exc:
                state.problem = str(exc)
            except (ValueError, KeyError, TypeError):
                state.problem = "REFERENCE_SCHEMA_NOT_VERIFIED"
            if not state.identity:
                continue
            try:
                arguments = {
                    "Uic": state.identity["uic"],
                    "AssetType": "ContractFutures",
                    "Amount": 1,
                    "AccountKey": self.client.oauth.account_key,
                    "FieldGroups": [
                        "Quote",
                        "PriceInfoDetails",
                        "MarketDepth",
                        "InstrumentPriceDetails",
                    ],
                }
                try:
                    await self.subscribe("PRICE", arguments, state.market)
                except SaxoError as exc:
                    if str(exc) not in {
                        "HTTP_400",
                        "HTTP_403",
                        "InstrumentNotAllowed",
                        "InvalidRequest",
                        "InvalidModelState",
                        "NoAccess",
                    }:
                        raise
                    # Some entitlements reject the entire depth field group.
                    # A second L1-only Saxo request preserves monitoring.
                    state.capabilities["depth_request_problem"] = str(exc)
                    arguments["FieldGroups"] = [
                        "Quote",
                        "PriceInfoDetails",
                        "InstrumentPriceDetails",
                    ]
                    await self.subscribe("PRICE", arguments, state.market)
                state.live_since = datetime.now(UTC)
                state.last_clock = state.last_clock or state.live_since
            except (SaxoError, ValueError) as exc:
                state.problem = str(exc)
            if state.option_root:
                try:
                    await self.subscribe(
                        "BOARD",
                        {
                            "Identifier": state.option_root,
                            "AssetType": "FuturesOption",
                            "AccountKey": self.client.oauth.account_key,
                            "MaxStrikesPerExpiry": 12,
                            "Expiries": [{"Index": 0}],
                        },
                        state.market,
                    )
                    if state.option_space:
                        await self.option_subscribe(state, state.option_space[0])
                except (SaxoError, ValueError) as exc:
                    state.capabilities["options"] = {"problem": str(exc)}
        # Discover this environment's GBPUSD UIC; never reuse a SIM identifier in LIVE.
        await self.subscribe_fx()

    async def subscribe_fx(self) -> None:
        fx = await self.client.request(
            "GET",
            "/ref/v1/instruments",
            params={"Keywords": "GBPUSD", "AssetTypes": "FxSpot", "$top": 20},
        )
        matches = [
            r
            for r in fx.get("Data", [])
            if r.get("Symbol") == "GBPUSD" and r.get("AssetType") == "FxSpot"
        ]
        if len(matches) == 1:
            self.fx_uic = int(matches[0]["Identifier"])
            await self.subscribe(
                "PRICE",
                {
                    "Uic": self.fx_uic,
                    "AssetType": "FxSpot",
                    "AccountKey": self.client.oauth.account_key,
                    "FieldGroups": ["Quote"],
                },
                "FX",
            )

    def mark_gap(self, target: str, reason: str) -> None:
        price, identity = self.price_target(target)
        price.gap(reason)
        if identity:
            self.recorder.ingest(
                key(identity),
                "GAP",
                {"reason": reason},
                time.time(),
                generation=price.generation,
                observation_context=price.observation_context(),
            )

    async def receive(self, message: dict[str, Any], received_at: float | None = None) -> None:
        receipt = time.time() if received_at is None else received_at
        ref, payload = message["reference"], message["payload"]
        self.last_message_id = message["message_id"]
        if ref in self.pending:
            queued = {"message": message, "receipt": receipt}
            size = len(json.dumps(queued))
            if self.pending_bytes + size > 1024**2:
                raise SaxoError("SNAPSHOT_PENDING_QUEUE_LIMIT")
            self.pending[ref].append(queued)
            self.pending_bytes += size
            return
        if ref == "_heartbeat":
            for heartbeat in payload.get("Heartbeats", []):
                subscription = self.subscriptions.get(heartbeat.get("OriginatingReferenceId"))
                if subscription:
                    subscription["contact"] = time.monotonic()
                    if heartbeat.get("Reason") != "NoNewData" and subscription["kind"] == "PRICE":
                        self.mark_gap(subscription["target"], "SUBSCRIPTION_DISABLED")
                        raise SaxoError("SUBSCRIPTION_DISABLED_FRESH_SNAPSHOT_REQUIRED")
                    if subscription["kind"] == "PRICE":
                        price, identity = self.price_target(subscription["target"])
                        price.last_contact = receipt
                        if identity:
                            self.recorder.ingest(
                                key(identity),
                                "HEARTBEAT",
                                heartbeat,
                                receipt,
                                message_id=message["message_id"],
                                generation=price.generation,
                                provider_message=message,
                                observation_context=price.observation_context(),
                            )
            return
        if ref in {"_resetsubscriptions", "_disconnect"}:
            raise SaxoError("STREAM_RESET_FRESH_SNAPSHOTS_REQUIRED")
        subscription = self.subscriptions.get(ref)
        if not subscription:
            return  # obsolete generation; never apply it to a replacement snapshot
        subscription["contact"] = time.monotonic() - max(0, time.time() - receipt)
        updates = payload if isinstance(payload, list) else [payload]
        for index, envelope in enumerate(updates):
            if envelope.get("TotalPartitions", 1) > 1:
                raise SaxoError("PARTITIONED_UPDATE_REQUIRES_FRESH_SNAPSHOT")
            data = envelope.get("Data", envelope)
            if subscription["kind"] == "SESSION":
                prior = self.session.get("TradeLevel")
                self.session = merge(self.session, data)
                if prior == "FullTradingAndChat" and self.session.get("TradeLevel") != prior:
                    for market in self.markets:
                        self.mark_gap(market, "SESSION_DOWNGRADED")
                    self.problem = "SESSION_DOWNGRADED_EXPLICIT_UPGRADE_REQUIRED"
                    raise SaxoError("SESSION_DOWNGRADED_FRESH_SNAPSHOT_REQUIRED")
            elif subscription["kind"] == "BOARD":
                state = self.markets[subscription["target"]]
                state.option_board = merge_board(state.option_board, data)
            else:
                price, identity = self.price_target(subscription["target"])
                accepted = price.update(data, message["message_id"] + f":{index}", receipt)
                if identity:
                    self.recorder.ingest(
                        key(identity),
                        "UPDATE",
                        data,
                        receipt,
                        message_id=message["message_id"],
                        duplicate=not accepted,
                        generation=ref,
                        provider_message=message,
                        observation_context=price.observation_context(),
                    )
        self.changed.set()

    async def run(self) -> None:
        backoff = 2
        while not self.stopping:
            setup: asyncio.Task[None] | None = None
            try:
                await self.verify_account()
                self.context = "SLRNO-" + secrets.token_hex(12)
                token = await self.client.oauth.access_token()
                generation = self.client.oauth.generation
                async with connect(
                    self.client.oauth.urls["stream"]
                    + "/connect?"
                    + urlencode({"contextId": self.context}),
                    additional_headers={"Authorization": "Bearer " + token},
                    max_size=self.config.recorder.max_message_bytes,
                    max_queue=16,
                    open_timeout=15,
                    close_timeout=5,
                ) as socket:
                    self.connected = True
                    self.problem = ""
                    self.reconnects += 1
                    connected_at = time.monotonic()
                    frames = Frames(self.config.recorder.max_message_bytes)
                    setup = asyncio.create_task(self.startup())
                    while not self.stopping:
                        if time.monotonic() - connected_at > 60:
                            backoff = 2
                        if setup.done():
                            setup.result()
                        try:
                            payload = await asyncio.wait_for(socket.recv(), timeout=1)
                        except TimeoutError:
                            payload = None
                        if payload is not None:
                            if not isinstance(payload, bytes):
                                raise SaxoError("EXPECTED_BINARY_STREAM")
                            for message in frames.feed(payload):
                                await self.receive(message)
                        await self.client.oauth.access_token()
                        if generation != self.client.oauth.generation:
                            # Authorize a fresh socket using the new token and take new snapshots.
                            # No obsolete streamingws/authorize URL or silent continuity claim.
                            raise SaxoError("TOKEN_RENEWED_STREAM_REAUTHORISATION")
                        if any(
                            time.monotonic() - s["contact"] > s["timeout"]
                            for s in self.subscriptions.values()
                        ):
                            raise SaxoError("SUBSCRIPTION_HEARTBEAT_TIMEOUT")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if setup:
                    setup.cancel()
                    await asyncio.gather(setup, return_exceptions=True)
                    setup = None
                self.connected = False
                self.problem = str(exc) if isinstance(exc, SaxoError) else "SAXO_RECONNECT_REQUIRED"
                for market in self.markets:
                    self.mark_gap(market, self.problem)
                for uic in self.options:
                    self.mark_gap(str(uic), self.problem)
                self.fx.gap(self.problem)
                # Explicitly delete server-owned subscriptions before a new context.
                for ref, subscription in list(self.subscriptions.items()):
                    with suppress(SaxoError):
                        await self.client.request(
                            "DELETE", subscription["path"] + f"/{self.context}/{ref}"
                        )
                self.subscriptions.clear()
                self.pending.clear()
                self.pending_bytes = 0
                await asyncio.sleep(backoff)
                backoff = min(60, backoff * 2)
            finally:
                if setup:
                    setup.cancel()
                    await asyncio.gather(setup, return_exceptions=True)

    def capability_view(self, state: MarketState) -> dict[str, Any]:
        value = state.price.value or {}
        quote = value.get("Quote") or {}
        result = dict(state.capabilities)
        usable = 0
        for identity, price in self.options.values():
            if identity["market"] == state.market:
                with suppress(ValueError):
                    quote_check(price.value or {}, price.receipt, datetime.now(UTC))
                    usable += 1
        result["options"] = {
            "discovered": len(state.option_space),
            "usable_quotes": usable,
            "status": "AVAILABLE" if usable else "UNVERIFIED_OR_UNAVAILABLE",
        }
        result.update(
            quote={
                "bid": quote.get("Bid"),
                "ask": quote.get("Ask"),
                "delay_minutes": quote.get("DelayedByMinutes"),
                "access": quote.get("PriceTypeAsk", "UNVERIFIED"),
            },
            l2=state.price.depth(time.time()),
            session=self.session,
            refresh_ms=state.price.refresh_ms,
            book_flow=self.recorder.book_flow_view(key(state.identity), time.time())
            if state.identity
            else {"status": "UNAVAILABLE"},
            user_action=(
                "Review Saxo TradeLevel; upgrading can downgrade another Saxo application"
                if self.session.get("TradeLevel") != "FullTradingAndChat"
                else ""
            ),
        )
        return result
