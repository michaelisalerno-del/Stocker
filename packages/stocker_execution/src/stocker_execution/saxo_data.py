"""One server-owned Saxo stream; browsers only read its state."""

import asyncio
import json
import secrets
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING
from typing import Any
from urllib.parse import quote as path_quote
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from stocker_execution import option_context, saxo_balance
from stocker_execution.bar_cache import BarCache
from stocker_execution.config import (
    MARKETS,
    OPTION_METADATA_MAX_AGE_SECONDS,
    QUOTE_MAX_AGE_SECONDS,
    ROLLING_WINDOW_SECONDS,
    SUBSCRIPTION_LIMIT,
    FuturesConfig,
    Market,
)
from stocker_execution.contracts import (
    cost_estimate,
    deadline_instant,
    executable_quote,
    future_identity,
    key,
    option_identity,
    option_price,
    quote_check,
    utc,
)
from stocker_execution.recorder import CHAIN, Recorder
from stocker_execution.rules import (
    NY,
    Bar,
    model_delta,
    next_clock,
    opportunity,
    prior_rv,
    session_day,
)
from stocker_execution.saxo_auth import SaxoError
from stocker_execution.saxo_client import PRIMARY_SESSION, SaxoClient
from stocker_execution.saxo_stream import ChartState, Frames, PriceState, merge, merge_board

# Reference details and option spaces share Saxo's 60-a-minute RefDataInstrumentsMinute limit;
# loading an option family one root per 1.05 s keeps a whole family inside it.
OPTION_SPACE_GAP_SECONDS = 1.05


def same_instant(option: dict[str, Any], instant: datetime) -> bool:
    try:
        return utc(str(option.get("LastTradeDate"))) == instant
    except ValueError:
        return False


def chain_identity(future: dict[str, Any], slot: str = "") -> dict[str, Any]:
    """The recorder instrument for a future's chain window (observation only, never traded).

    Slot "" is the nearest expiry's window; "next" the following expiry day's (2026-10-04),
    recorded as its own instrument so readers of the nearest chain never see two expiries.
    """
    return {
        **future,
        "asset_type": CHAIN + ("Next" if slot else ""),
        "chain_slot": slot or "nearest",
    }


def board_target(target: str) -> tuple[str, str]:
    """A chain subscription's target is the market, or market:next for the following expiry."""
    return (target[:-5], "next") if target.endswith(":next") else (target, "")


def roots_verified(state: "MarketState", approved: tuple[int, ...]) -> bool:
    """The pinned future's option space was loaded only from approved roots."""
    return (
        state.identity is not None
        and bool(state.option_roots)
        and set(state.option_roots) <= set(approved)
    )


@dataclass
class MarketState:
    market: Market
    identity: dict[str, Any] | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    price: PriceState = field(default_factory=PriceState)
    bars: list[Bar] = field(default_factory=list)
    # Streamed one-minute samples: the boundary bar at a clock before the REST read has it.
    chart: ChartState = field(default_factory=ChartState)
    # Boundary bars a decision took from the stream, until the REST read of that minute
    # confirms them: minute -> (bar, clock event id).
    stream_bars: dict[datetime, tuple[Bar, str]] = field(default_factory=dict)
    chart_used: int = 0
    chart_confirmed: int = 0
    references: list[dict[int, dict[str, float]]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(
        default_factory=lambda: {
            "permission": "UNVERIFIED",
            "quote": "UNVERIFIED",
            "l2": "UNVERIFIED",
            "options": "UNVERIFIED",
            "history": "UNVERIFIED",
        }
    )
    option_space: list[dict[str, Any]] = field(default_factory=list)
    option_board: dict[str, Any] = field(default_factory=dict)
    option_root: int | None = None  # the chain board's root: the nearest expiry from today
    next_option_root: int | None = None  # the following expiry day's root: a second window
    next_option_day: str = ""  # that following expiry day (a root can list several expiry days)
    next_option_board: dict[str, Any] = field(default_factory=dict)
    option_roots: tuple[int, ...] = ()  # approved roots found for the pinned future
    expiry_instants: dict[str, datetime] = field(default_factory=dict)
    option_space_day: str = ""
    options_retry_at: float = 0.0
    startup_retry_at: float = 0.0  # a failed connect-time step is retried from the history pass
    problem: str = "AUTHENTICATION_REQUIRED"
    history_problem: str = "SAXO_HISTORY_NOT_VERIFIED"
    last_clock: datetime | None = None
    live_since: datetime | None = None
    history_checked: float = 0
    boundary_clock: datetime | None = None
    boundary_checked: float = float("-inf")
    reference_day: str = ""
    reference_retry_at: float = 0.0
    reference_failure: str = ""
    candidate_uic: int | None = None
    warm_uics: set[int] = field(default_factory=set)
    candidate_problem: str = "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED"
    candidate_changes: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=32))
    # Display only: frozen-model delta per ranked strike at the latest ranking.
    candidate_deltas: dict[int, float] = field(default_factory=dict)
    # Display only: average completed daily high-low range, refreshed once per NY day.
    daily_range: float | None = None
    daily_day: str = ""
    daily_problem: str = ""
    # Display and evidence only: the futures' official open interest of the last completed sessions
    # (daily bars; CME publishes it once a day, so minute bars carry 0).
    daily_open_interest: list[dict[str, Any]] = field(default_factory=list)


class DataService:
    def __init__(self, config: FuturesConfig, client: SaxoClient, recorder: Recorder):
        self.config, self.client, self.recorder = config, client, recorder
        self.markets: dict[str, MarketState] = {m: MarketState(m) for m in MARKETS}
        self.context = ""
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.options: dict[int, tuple[dict[str, Any], PriceState]] = {}
        self.option_required_at: dict[int, float] = {}
        self.option_references: dict[int, dict[str, Any]] = {}
        self.owned_options: set[int] = set()
        self.disabled_targets: set[tuple[str, str]] = set()
        self.reset_refs: set[str] = set()  # subscriptions the run loop replaces one by one
        self.session: dict[str, Any] = {}
        self.fx = PriceState()
        self.connected = False
        self.problem = "AUTHENTICATION_REQUIRED"
        self.stopping = False
        self.reconnects = 0
        self.last_message_id: str | None = None
        self.stream_at: float | None = None  # receipt of the latest message of any kind
        self.account_verified = False
        self.account_currency: str | None = None
        self.account_id: str | None = None
        self.client_key: str | None = None  # LIVE balances need it with the AccountKey
        self.balance: dict[str, Any] = {}
        self.balance_stream: dict[str, Any] = {}
        self.balance_account_key: str | None = None
        self.balance_received_at: float | None = None
        self.balance_problem = "ACCOUNT_BALANCE_UNAVAILABLE"
        self.balance_attempt = float("-inf")
        # SIM order/position events only prompt an earlier reconcile; never trusted as state.
        self.last_activity = float("-inf")
        self.activity_attempt = float("-inf")
        self.activity_problem = ""
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.pending_bytes = 0
        self.subscription_lock = asyncio.Lock()
        self.subscribe_lock = asyncio.Lock()
        self.bar_cache = BarCache(recorder.directory.parent / "bars", config.recorder)
        # The chart stream's tripwire: set by the first streamed bar a REST read contradicts,
        # cleared only by a restart. Decisions then wait for REST as before.
        self.chart_stream_problem = ""
        self.chart_mismatch: dict[str, Any] | None = None
        self.chart_audits: list[tuple[str, dict[str, Any]]] = []
        # The configured following contract month per market: (identity, price), target market:next.
        self.next_contracts: dict[str, tuple[dict[str, Any], PriceState]] = {}

    def drain_chart_audits(self) -> list[tuple[str, dict[str, Any]]]:
        audits, self.chart_audits = self.chart_audits, []
        return audits

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
        self.account_currency = accounts[0].get("Currency")
        self.account_id = accounts[0].get("AccountId")
        self.client_key = accounts[0].get("ClientKey")
        if not self.account_id:
            raise SaxoError("ACCOUNT_IDENTIFIER_NOT_VERIFIED")
        self.account_verified = True
        if self.balance_account_key != self.client.oauth.account_key:
            self.balance.clear()
            self.balance_stream.clear()
            self.balance_received_at = None
            self.balance_account_key = self.client.oauth.account_key
        self.client.sim_account_verified = self.config.data_environment == "SAXO_SIM"
        self.session = await self.client.request("GET", "/root/v1/sessions/capabilities")

    def quote_receipt(self, price: PriceState) -> float | None:
        """When this quote was last known current: its last change, or later while it stands."""
        return price.standing(price.receipt, self.stream_at)

    def size_receipt(self, price: PriceState, side: str) -> float | None:
        return price.standing(price.size_times.get(side), self.stream_at)

    async def take_primary_session(self) -> None:
        """The user's System-page click: make this Saxo's primary session for real-time prices.

        Saxo sends real-time prices only to a user's one FullTradingAndChat session, so this can
        delay or log off SaxoTraderGO, and a later SaxoTraderGO login takes the slot back.
        """
        if not self.connected:
            raise SaxoError("SAXO_NOT_CONNECTED")
        already = self.session.get("TradeLevel") == "FullTradingAndChat"
        await self.client.request(
            "PATCH",
            "/root/v1/sessions/capabilities",
            body=PRIMARY_SESSION,
            primary_session=True,
        )
        self.update_session(await self.client.request("GET", "/root/v1/sessions/capabilities"))
        if already:
            # A repeat click (for example once a data subscription starts) renews streams too.
            self.renew_streams()

    def renew_streams(self) -> None:
        # Subscriptions made before real-time applied carried delayed prices; take fresh ones
        # rather than assume Saxo switches them.
        for ref, subscription in self.subscriptions.items():
            if subscription["kind"] == "PRICE":
                self.mark_gap(subscription["target"], "SESSION_UPGRADED")
            if subscription["kind"] == "CHART":
                self.markets[subscription["target"]].chart.gap("SESSION_UPGRADED")
            if subscription["kind"] in {"PRICE", "BOARD", "CHART"}:
                self.reset_refs.add(ref)

    def update_session(self, data: dict[str, Any]) -> None:
        prior = self.session.get("TradeLevel")
        self.session = merge(self.session, data)
        level = self.session.get("TradeLevel")
        if prior == "FullTradingAndChat" and level != prior:
            for market in self.markets:
                self.mark_gap(market, "SESSION_DOWNGRADED")
                self.markets[market].chart.gap("SESSION_DOWNGRADED")
            self.problem = "SESSION_DOWNGRADED_EXPLICIT_UPGRADE_REQUIRED"
            raise SaxoError("SESSION_DOWNGRADED_FRESH_SNAPSHOT_REQUIRED")
        if level == "FullTradingAndChat" and prior != level:
            self.renew_streams()

    async def discover(self, state: MarketState) -> None:
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
        self.recorder.metadata_version(
            key(identity),
            {
                "identity": identity,
                "reference": self.safe_reference(raw),
            },
            time.time(),
        )
        try:
            # A reconnect keeps today's option spaces (refresh_options reloads them each CME
            # session, at the 18:00 open since 2026-10-04): reloading ~70 roots a second apart held
            # each market's price feed back 1-4 min.
            if state.option_space_day != session_day(datetime.now(NY)).isoformat():
                await self.load_options(state, raw)
        except SaxoError as exc:
            # The future is verified; an options problem (a rate limit at connect, say) stays with
            # the options, and the next history pass retries through refresh_options.
            state.options_retry_at = time.monotonic() + 60
            state.capabilities["options"] = {"problem": str(exc)}
        state.problem = ""

    async def load_options(self, state: MarketState, raw: dict[str, Any]) -> None:
        """The approved option roots of the pinned future, their options, and each day's expiry.

        Without an approval a single related root is still shown for observation, as before.
        An expiry instant comes from the approval's listed instants, or from Saxo's timestamped
        LastTradeDate only when it falls on that expiry day at the approved New York clock time.
        The chain board follows the root of the nearest expiry from today.
        """
        assert state.identity is not None
        roots = [
            r["OptionRootId"]
            for r in raw.get("RelatedOptionRootsEnhanced", [])
            if r.get("AssetType") == "FuturesOption"
        ]
        mapping = self.config.mappings.get(state.market)
        chosen = (
            [r for r in roots if r in mapping.option_root_ids]
            if mapping
            else (roots if len(roots) == 1 else [])
        )
        if mapping and set(chosen) != set(mapping.option_root_ids):
            # An approved weekday root Saxo does not relate to this future would silently move
            # SAME_DAY_OR_NEXT_LISTED to the following day's option; the family loads whole or
            # not at all, and the market says why.
            raise SaxoError("APPROVED_OPTION_ROOT_NOT_RELATED")
        space_rows: list[dict[str, Any]] = []
        for i, root in enumerate(chosen):
            if i:
                await asyncio.sleep(OPTION_SPACE_GAP_SECONDS)
            space = await self.client.request(
                "GET",
                f"/ref/v1/instruments/contractoptionspaces/{root}",
                params={
                    "OptionSpaceSegment": "UnderlyingUic",
                    "UnderlyingUic": state.identity["uic"],
                },
            )
            if space.get("AssetType") != "FuturesOption":
                raise SaxoError("WRONG_OPTION_ROOT_ASSET_TYPE")
            rows = [
                {
                    **option,
                    "Expiry": expiry.get("Expiry"),
                    "LastTradeDate": expiry.get("LastTradeDate"),
                    "ExerciseStyle": space.get("ExerciseStyle"),
                    "TickSizeScheme": expiry.get("TickSizeScheme"),
                    "OptionRootId": root,
                }
                for expiry in space.get("OptionSpace", [])
                for option in expiry.get("SpecificOptions", [])
                if option.get("UnderlyingUic") == state.identity["uic"]
            ]
            # Live monthly roots carry several expiries (gold's held 3,060 options on 2026-09-30).
            if len(rows) > 5000 or len(space_rows) + len(rows) > 20000:
                raise SaxoError("OPTION_SPACE_TOO_LARGE")
            space_rows += rows
        instants: dict[str, datetime] = {}
        if mapping and mapping.expiry_instants:
            instants = {d: utc(i.isoformat()) for d, i in mapping.expiry_instants.items()}
        elif mapping and mapping.expiry_clock_new_york:
            conflicts: set[str] = set()
            for row in space_rows:
                day = str(row.get("Expiry", ""))[:10]
                try:
                    stamp = utc(str(row.get("LastTradeDate")))
                except ValueError:
                    continue
                local = stamp.astimezone(NY)
                if (
                    local.date().isoformat() != day
                    or local.strftime("%H:%M") != mapping.expiry_clock_new_york
                ):
                    continue
                if instants.get(day, stamp) != stamp:
                    conflicts.add(day)
                instants[day] = stamp
            for day in conflicts:
                instants.pop(day)
        today = session_day(datetime.now(NY)).isoformat()
        upcoming = sorted(
            (str(r.get("Expiry", ""))[:10], r["OptionRootId"])
            for r in space_rows
            if str(r.get("Expiry", ""))[:10] >= today
        )
        state.option_roots = tuple(chosen)
        state.option_space, state.expiry_instants = space_rows, instants
        state.option_root = upcoming[0][1] if upcoming else None
        # The following expiry day's root feeds a second, separately recorded chain window.
        following = (
            next(((d, r) for d, r in upcoming if d > upcoming[0][0]), None) if upcoming else None
        )
        state.next_option_root = following[1] if following else None
        state.next_option_day = following[0] if following else ""
        state.option_space_day = today
        state.capabilities["options"] = (
            {
                "discovered": len(space_rows),
                "roots": len(chosen),
                "quote": "UNVERIFIED",
                "root": state.option_root,
                "expiry_days": sorted(d for d in instants if d >= today)[:10],
            }
            if chosen
            else "OPTION_ROOT_AMBIGUOUS_OR_UNAVAILABLE"
        )

    async def refresh_options(self, state: MarketState) -> None:
        """Once per CME session (18:00 New York): new weekly listings appear and the board moves to
        the root of the session's expiry.

        A failure is kept to the options (candidates then report their own reason) and is retried
        after 15 minutes, never on every history pass.
        """
        if (
            not state.identity
            or state.option_space_day == session_day(datetime.now(NY)).isoformat()
            or time.monotonic() < state.options_retry_at
        ):
            return
        try:
            raw = await self.client.request(
                "GET",
                f"/ref/v1/instruments/details/{state.identity['uic']}/ContractFutures",
                params={
                    "AccountKey": self.client.oauth.account_key,
                    "FieldGroups": "TradingSessions",
                },
            )
            if raw.get("Uic") != state.identity["uic"]:
                raise SaxoError("CONFIGURED_REFERENCE_IDENTITY_MISMATCH")
            await self.load_options(state, raw)
            await self.subscribe_board(state)
            await self.subscribe_board(state, "next")
        except (SaxoError, ValueError, KeyError, TypeError) as exc:
            state.options_retry_at = time.monotonic() + 900
            state.capabilities["options"] = {"problem": str(exc)}

    async def subscribe_board(self, state: MarketState, slot: str = "") -> None:
        """Observation-only chain window on the root of the nearest expiry (slot "") or of the
        following expiry day (slot "next", 2026-10-04: term structure for the monthly search)."""
        root = state.next_option_root if slot else state.option_root
        if not root:
            return
        arguments = {
            "Identifier": root,
            "AssetType": "FuturesOption",
            "AccountKey": self.client.oauth.account_key,
            "MaxStrikesPerExpiry": self.config.option_chain_strikes,
            "Expiries": [{"Index": 0}],
        }
        target = state.market + (":next" if slot else "")
        current = next(
            (
                (ref, s)
                for ref, s in self.subscriptions.items()
                if s.get("kind") == "BOARD" and s.get("target") == target
            ),
            None,
        )
        if current and current[1]["arguments"].get("Identifier") == root:
            return
        try:
            await self.subscribe("BOARD", arguments, target, current[0] if current else None)
        except (SaxoError, ValueError) as exc:
            if slot:
                state.capabilities["next_chain_problem"] = str(exc)
            else:
                state.capabilities["options"] = {"problem": str(exc)}

    async def subscribe(
        self,
        kind: str,
        arguments: dict[str, Any],
        target: str,
        old: str | None = None,
    ) -> str:
        async with self.subscribe_lock:
            if (kind, target) in self.disabled_targets:
                raise SaxoError("SUBSCRIPTION_PERMANENTLY_DISABLED")
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
        if len(self.subscriptions) >= SUBSCRIPTION_LIMIT and old is None:
            raise SaxoError("SUBSCRIPTION_LIMIT")
        paths = {
            "PRICE": "/trade/v1/prices/subscriptions",
            "BALANCE": "/port/v1/balances/subscriptions",
            "BOARD": "/trade/v1/optionschain/subscriptions",
            "SESSION": "/root/v1/sessions/events/subscriptions",
            "ACTIVITIES": "/ens/v1/activities/subscriptions",
            "CHART": "/chart/v3/charts/subscriptions",
        }
        ref = "S" + secrets.token_hex(10)
        context = self.context
        body: dict[str, Any] = {
            "ContextId": context,
            "ReferenceId": ref,
            "Arguments": arguments,
            # Measured floors on this account (2026-10-04 probe): prices and chains 1,000 ms,
            # charts 300 ms; Saxo grants the floor when less is asked.
            "RefreshRate": 10000 if kind == "BALANCE" else 300 if kind == "CHART" else 1000,
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
            self.drop_pending(ref)
            raise
        if context != self.context:
            # The socket reconnected during the POST: never register against a dead context.
            self.drop_pending(ref)
            with suppress(SaxoError):
                await self.client.request("DELETE", paths[kind] + f"/{context}/{ref}")
            raise SaxoError("SUBSCRIPTION_CONTEXT_REPLACED")
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
                self.drop_pending(ref)
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
        elif kind == "BALANCE":
            if arguments["AccountKey"] != self.client.oauth.account_key:
                self.balance_problem = "BALANCE_ACCOUNT_CHANGED"
                self.drop_pending(ref)
                raise SaxoError("BALANCE_ACCOUNT_CHANGED")
            saxo_balance.receive_balance(self, snapshot, time.time(), snapshot=True)
        elif kind == "SESSION":
            self.session = snapshot or {}
        elif kind == "ACTIVITIES":
            pass  # event stream only; there is no snapshot
        elif kind == "CHART":
            if not isinstance(snapshot, dict):
                await self.client.request("DELETE", paths[kind] + f"/{self.context}/{ref}")
                self.subscriptions.pop(ref, None)
                self.drop_pending(ref)
                raise SaxoError("CHART_SNAPSHOT_MISSING")
            chart = self.markets[target].chart
            chart.inactivity_timeout = self.subscriptions[ref]["timeout"]
            chart.snapshot(snapshot, ref, time.time())
        else:
            market, slot = board_target(target)
            state = self.markets[market]
            if slot:
                state.next_option_board = snapshot or {}
                self.record_chain(state, "SNAPSHOT", snapshot or {}, time.time(), slot)
            else:
                state.option_board = snapshot or {}
                self.record_board(state, snapshot or {}, time.time())
                self.record_chain(state, "SNAPSHOT", snapshot or {}, time.time())
        for message in self.drop_pending(ref):
            await self.receive(message["message"], message["receipt"])
        return ref

    def drop_pending(self, ref: str) -> list[dict[str, Any]]:
        pending = self.pending.pop(ref, [])
        self.pending_bytes -= sum(m["size"] for m in pending)
        return pending

    def price_target(self, target: str) -> tuple[PriceState, dict[str, Any] | None]:
        if target == "FX":
            return self.fx, None
        if target.endswith(":next"):
            identity, price = self.next_contracts[target[:-5]]
            return price, identity
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
        assert state.identity is not None
        uic = selected["Uic"]
        if uic in self.options and any(
            s["target"] == str(uic) for s in self.subscriptions.values()
        ):
            self.option_required_at[uic] = max(self.option_required_at.get(uic, 0), time.time())
            return self.options[uic][0]
        if uic not in self.options and len(self.options) >= self.config.option_subscription_budget:
            retired = next(
                (
                    k
                    for k, (i, _) in self.options.items()
                    if k not in self.owned_options
                    and self.option_required_at.get(k, 0) <= time.time()
                    and not self.recorder.requires_subscription(key(i), time.time())
                ),
                None,
            )
            if retired is None:
                raise SaxoError("OPTION_SUBSCRIPTION_CAPACITY")
            await self.unsubscribe_option(retired)
        raw = await self.client.request(
            "GET",
            f"/ref/v1/instruments/details/{uic}/FuturesOption",
            params={"AccountKey": self.client.oauth.account_key, "FieldGroups": "TradingSessions"},
        )
        try:
            identity = option_identity(state.identity, raw, selected["OptionRootId"], selected)
        except (TypeError, AttributeError) as exc:
            raise SaxoError("OPTION_REFERENCE_SCHEMA_UNVERIFIED") from exc
        expiry_instant = state.expiry_instants.get(str(identity["expiry"])[:10])
        identity["expiry_instant"] = expiry_instant.isoformat() if expiry_instant else None
        self.release_window_slot()
        self.recorder.register(key(identity), identity)
        self.options[uic] = (identity, PriceState())
        self.option_required_at[uic] = max(self.option_required_at.get(uic, 0), time.time())
        try:
            await self.option_conditions(identity, raw)
            arguments = {
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
            }
            try:
                await self.subscribe("PRICE", arguments, str(uic))
            except SaxoError as exc:
                if str(exc) not in {"InvalidRequest", "InvalidModelState", "HTTP_400", "NoAccess"}:
                    raise
                arguments["FieldGroups"] = ["Quote", "PriceInfoDetails", "InstrumentPriceDetails"]
                await self.subscribe("PRICE", arguments, str(uic))
        except Exception:
            self.options.pop(uic, None)
            self.option_required_at.pop(uic, None)
            self.option_references.pop(uic, None)
            raise
        return identity

    @staticmethod
    def safe_reference(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            k: raw[k]
            for k in (
                "Uic",
                "AssetType",
                "Symbol",
                "Exchange",
                "ContractSize",
                "LotSize",
                "LotSizeType",
                "MinimumTradeSize",
                "AmountDecimals",
                "PriceToContractFactor",
                "CurrencyCode",
                "PriceCurrency",
                "TickSize",
                "TickSizeLimitOrder",
                "TickSizeScheme",
                "ExpiryDate",
                "NoticeDate",
                "TradingSessions",
                "ExerciseCutOffTime",
                "SettlementStyle",
                "PutCall",
                "StrikePrice",
                "UnderlyingAssetType",
                "IsTradable",
                "TradingStatus",
            )
            if k in raw
        }

    async def option_conditions(self, identity: dict[str, Any], raw: dict[str, Any]) -> None:
        uic, at = identity["uic"], time.time()
        prior = self.option_references.get(uic)
        if prior and 0 <= at - prior["received_at"] < OPTION_METADATA_MAX_AGE_SECONDS:
            # A re-created identity (reconnect, restore) inherits the cross-checked deadline.
            identity["last_trade_at"] = prior["identity"].get(
                "last_trade_at", identity.get("last_trade_at")
            )
            return
        entry: dict[str, Any] = {
            "received_at": at,
            "reference": self.safe_reference(raw),
            "identity": dict(identity),
            "conditions": {},
            "problem": "",
        }
        try:
            conditions = await self.client.request(
                "GET",
                "/cs/v1/tradingconditions/ContractOptionSpaces/"
                + path_quote(self.client.oauth.account_key, safe="")
                + f"/{identity['option_root_id']}",
                params={"Uic": uic, "FieldGroups": "ScheduledTradingConditions"},
            )
            if conditions.get("Uic") != uic or conditions.get("AssetType") != "FuturesOption":
                raise ValueError("CONTRACT_OPTION_COST_IDENTITY_UNVERIFIED")
            entry["conditions"] = conditions
            # ExpirationTime is explicitly last trading time, not option expiry instant.
            last_trade = deadline_instant(
                conditions.get("ExpirationTime"), str(identity["expiry"])[:10]
            )
            reference_trade = deadline_instant(
                identity.get("last_trade_at"), str(identity["expiry"])[:10]
            )
            if last_trade and reference_trade and last_trade != reference_trade:
                raise ValueError("OPTION_LAST_TRADING_DEADLINE_CONFLICT")
            identity["last_trade_at"] = last_trade or reference_trade
            entry["identity"] = dict(identity)
        except (ValueError, TypeError, AttributeError) as exc:
            entry["problem"] = (
                "CONTRACT_OPTION_COST_SCHEMA_UNVERIFIED"
                if isinstance(exc, (TypeError, AttributeError))
                else str(exc)
            )
        self.option_references[uic] = entry
        entry["version"] = self.recorder.metadata_version(
            key(identity), {k: v for k, v in entry.items() if k != "received_at"}, at
        )

    def rank_candidates(
        self, state: MarketState, event: dict[str, Any], inputs: dict[str, float]
    ) -> list[tuple[float, float, dict[str, Any]]]:
        mapping = self.config.mappings.get(state.market)
        if mapping is None:
            raise ValueError("LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED")
        if state.identity is None or not roots_verified(state, mapping.option_root_ids):
            raise ValueError("OPTION_ROOT_NOT_VERIFIED")
        at = utc(event["signal_at"])
        wanted = at.astimezone(NY).date().isoformat()
        if mapping.expiry_rule == "SAME_DAY_OR_NEXT_LISTED":
            # verified_cutoff needs the exit two minutes before the option stops trading; the DAY
            # trade also refuses any expiry at or before its clock day's close (`expiry_after`).
            last_exit = utc(event["exit_at"]) + timedelta(seconds=120)
            if event.get("expiry_after"):
                last_exit = max(last_exit, utc(event["expiry_after"]))
            wanted = min((d for d, i in state.expiry_instants.items() if i > last_exit), default="")
        candidates = []
        deltas: dict[int, float] = {}
        for selected in state.option_space:
            if selected.get("UnderlyingUic") != state.identity["uic"]:
                continue
            if selected.get("PutCall") != ("Call" if event["right"] == "C" else "Put"):
                continue
            day = str(selected.get("Expiry", ""))[:10]
            if day != wanted:
                continue
            expiry = state.expiry_instants.get(day)
            if expiry is None or expiry <= at:
                continue
            if mapping.expiry_clock_new_york and not same_instant(selected, expiry):
                continue  # another series expiring that day at a different time (AM-settled)
            delta = model_delta(
                inputs["futures_price"],
                float(selected["StrikePrice"]),
                inputs["rv15"],
                at,
                expiry,
                str(event["right"]),
            )
            deltas[selected["Uic"]] = delta
            candidates.append(
                (
                    abs(delta - float(event["target_delta"])),
                    float(selected["StrikePrice"]),
                    selected,
                )
            )
        state.candidate_deltas = deltas
        if not candidates:
            raise ValueError(
                "NO_VERIFIED_REAL_0DTE_EXPIRY_TIME"
                if mapping.expiry_rule == "SAME_DAY"
                else "NO_VERIFIED_LISTED_EXPIRY_TIME"
            )
        candidates.sort(key=lambda x: (x[0], x[1], x[2]["Uic"]))
        if candidates[0][0] > mapping.delta_tolerance:
            raise ValueError("FROZEN_DELTA_OUTSIDE_APPROVED_TOLERANCE")
        return candidates

    async def warm_candidates(self, state: MarketState) -> None:
        # Observation only: reuses the frozen ranking; never creates an event or order.
        for uic, (identity, _) in list(self.options.items()):
            if identity["market"] == state.market and self.option_protected(uic):
                await self.focus_board(state, identity)
                break
        await self.focus_next_board(state)
        if not state.identity or not state.bars:
            return
        try:
            at = state.bars[-1].at + timedelta(minutes=1)
            if not 0 <= time.time() - at.timestamp() <= 120:
                raise ValueError("CANDIDATE_UNDERLYING_HISTORY_STALE")
            # Reuse frozen right/target terms without requiring an entry clock
            # just to warm observations. This descriptor never creates an event.
            event = opportunity(state.market, state.identity["uic"], next_clock(at))
            event["signal_at"] = at.isoformat()
            inputs = {"futures_price": state.bars[-1].close, "rv15": prior_rv(state.bars, at)}
            ranked = self.rank_candidates(state, event, inputs)
            wanted = {c[2]["Uic"] for c in ranked[: self.config.option_candidate_window]}
            state.warm_uics = wanted
            async with self.subscription_lock:
                for uic, (identity, _) in list(self.options.items()):
                    if (
                        identity["market"] == state.market
                        and uic not in wanted
                        and not self.option_protected(uic)
                    ):
                        await self.unsubscribe_option(uic)
            # Leave four lines available for owned/pending/event contracts.
            for _, _, candidate in ranked[: self.config.option_candidate_window]:
                if (
                    candidate["Uic"] not in self.options
                    and len(self.options) >= self.config.option_subscription_budget - 4
                ):
                    break
                await self.option_subscribe(state, candidate)
            selected = ranked[0][2]["Uic"]
            if selected not in self.options:
                raise ValueError("OPTION_SUBSCRIPTION_CAPACITY")
            await self.focus_board(state, self.options[selected][0])
            if selected != state.candidate_uic:
                change = {
                    "old_uic": state.candidate_uic,
                    "new_uic": selected,
                    "received_at": time.time(),
                    "reason": "NEAREST_FROZEN_MODEL_DELTA",
                    "inputs": inputs,
                    "selection_at": at.isoformat(),
                }
                state.candidate_changes.append(change)
                self.recorder.ingest(key(state.identity), "CANDIDATE_CHANGE", change, time.time())
                state.candidate_uic = selected
            state.candidate_problem = ""
        except (ValueError, KeyError) as exc:
            state.candidate_problem = str(exc)

    def option_protected(self, uic: int) -> bool:
        identity = self.options[uic][0]
        return (
            uic in self.owned_options
            or self.option_required_at.get(uic, 0) > time.time()
            or self.recorder.requires_subscription(key(identity), time.time())
        )

    async def focus_board(self, state: MarketState, requested: dict[str, Any]) -> None:
        """Keep the chain window on the strikes nearest the money, on the held or candidate
        option's expiry (2026-10-02: recorded so other strikes can be studied). Regular pinned
        quotes never change UIC; the window moves once the money leaves its middle half."""
        subscription = next(
            (
                (r, s)
                for r, s in self.subscriptions.items()
                if s.get("kind") == "BOARD" and s["target"] == state.market
            ),
            None,
        )
        if not subscription:
            return
        ref, current = subscription
        protected = [
            i
            for uic, (i, _) in self.options.items()
            if i["market"] == state.market and self.option_protected(uic)
        ]
        protected.sort(key=lambda i: (i["uic"] not in self.owned_options, i["uic"]))
        identity = protected[0] if protected else requested
        expiry = next(
            (
                e
                for e in state.option_board.get("Expiries", []) or []
                if str(e.get("Expiry", ""))[:10] == str(identity["expiry"])[:10]
            ),
            None,
        )
        if not expiry:
            state.capabilities["option_chain_problem"] = "SELECTED_EXPIRY_NOT_IN_CHAIN"
            return
        await self.centre_window(ref, current, expiry, state, "option_chain_problem")

    async def focus_next_board(self, state: MarketState) -> None:
        """Keep the following expiry's window centred on its own money."""
        subscription = next(
            (
                (r, s)
                for r, s in self.subscriptions.items()
                if s.get("kind") == "BOARD" and s["target"] == state.market + ":next"
            ),
            None,
        )
        expiries = state.next_option_board.get("Expiries", []) or []
        # A root can list several expiry days and the window opens on its first (often today's):
        # move it to the following expiry day itself (2026-10-09: the US-hours capture was today's).
        expiry = next(
            (e for e in expiries if str(e.get("Expiry", ""))[:10] == state.next_option_day),
            None if state.next_option_day else next(iter(expiries), None),
        )
        if subscription and state.next_option_day and not expiry and expiries:
            state.capabilities["next_chain_problem"] = "NEXT_EXPIRY_NOT_IN_CHAIN"
            return
        if subscription and expiry:
            await self.centre_window(*subscription, expiry, state, "next_chain_problem")

    async def centre_window(
        self,
        ref: str,
        current: dict[str, Any],
        expiry: dict[str, Any],
        state: MarketState,
        problem: str,
    ) -> None:
        mid = expiry.get("MidStrikePrice")
        strikes = [
            s for s in expiry.get("Strikes", []) or [] if isinstance(s.get("Strike"), (int, float))
        ]
        size = self.config.option_chain_strikes
        selection = {"Index": expiry["Index"]}
        if isinstance(mid, (int, float)) and strikes:
            money = min(strikes, key=lambda s: (abs(s["Strike"] - mid), s["Index"]))
            selection["StrikeStartIndex"] = max(0, money["Index"] - size // 2)
            held = (current.get("board_window") or {}).get("Expiries", [{}])[0]
            if (
                held.get("Index") == expiry["Index"]
                and "StrikeStartIndex" in held
                and abs(held["StrikeStartIndex"] - selection["StrikeStartIndex"]) <= size // 4
            ):
                return
        patch = {"Expiries": [selection], "MaxStrikesPerExpiry": size}
        if current.get("board_window") == patch:
            return
        try:
            await self.client.request(
                "PATCH", current["path"] + f"/{self.context}/{ref}", body=patch
            )
            current["board_window"] = patch
            state.capabilities[problem] = (
                "" if "StrikeStartIndex" in selection else "AWAITING_CHAIN_STRIKES"
            )
        except SaxoError as exc:
            state.capabilities[problem] = str(exc)

    def record_chain(
        self, state: MarketState, kind: str, data: dict[str, Any], at: float, slot: str = ""
    ) -> None:
        """Observation only: every chain-window message joins the future's captures."""
        if not state.identity:
            return
        identity = chain_identity(state.identity, slot)
        problem = "chain_recording_problem" + ("_next" if slot else "")
        try:
            self.recorder.register(key(identity), identity)
            self.recorder.ingest(key(identity), kind, data, at, context_source="OPTIONS_CHAIN")
            state.capabilities[problem] = ""
        except ValueError as exc:
            state.capabilities[problem] = str(exc)

    def record_board(self, state: MarketState, update: dict[str, Any], at: float) -> None:
        for expiry in update.get("Expiries", []) or []:
            full_expiry: dict[str, Any] = next(
                (
                    e
                    for e in state.option_board.get("Expiries", []) or []
                    if e["Index"] == expiry["Index"]
                ),
                {},
            )
            for strike in expiry.get("Strikes", []) or []:
                full_strike: dict[str, Any] = next(
                    (
                        s
                        for s in full_expiry.get("Strikes", []) or []
                        if s["Index"] == strike["Index"]
                    ),
                    {},
                )
                for right in ("Call", "Put"):
                    if right not in strike:
                        continue
                    side = strike[right]
                    uic = (side or {}).get("Uic") or (full_strike.get(right) or {}).get("Uic")
                    if uic not in self.options:
                        continue
                    identity, _ = self.options[uic]
                    if not state.identity or identity["underlying_uic"] != state.identity["uic"]:
                        continue
                    window = self.recorder.windows[key(identity)]
                    analytics = dict(window.context.get("OPTIONS_CHAIN", {}).get("analytics", {}))
                    if side is None:
                        analytics = {}
                    else:
                        analytics.update(
                            option_context.fields(
                                option_context.chain_update(side, update.get("LastUpdated")),
                                at,
                                "OPTIONS_CHAIN",
                            )
                        )
                    self.recorder.ingest(
                        key(identity),
                        "CHAIN_CONTEXT",
                        side,
                        at,
                        context_source="OPTIONS_CHAIN",
                        observation_context={
                            "analytics": analytics,
                            "executable": False,
                            "receipt": at,
                            "price_source": "OPTIONS_CHAIN",
                        },
                    )

    def option_view(self, uic: int, at: float) -> dict[str, Any]:
        identity, price = self.options[uic]
        reference = self.option_references.get(uic, {})
        window = self.recorder.windows.get(key(identity))
        costs: dict[str, Any] = {"budget_result": "UNVERIFIED"}
        try:
            if reference.get("problem"):
                raise ValueError(reference["problem"])
            if not 0 <= at - reference.get("received_at", 0) <= OPTION_METADATA_MAX_AGE_SECONDS:
                raise ValueError("CONTRACT_OPTION_COSTS_STALE_OR_UNAVAILABLE")
            q = executable_quote(
                identity,
                price.value or {},
                self.quote_receipt(price),
                datetime.fromtimestamp(at, UTC),
            )
            rate = 1.0
            if identity["currency"] != "GBP":
                if identity["currency"] != "USD":
                    raise ValueError("CURRENCY_CONVERSION_PAIR_UNVERIFIED")
                fx = quote_check(
                    self.fx.value or {},
                    self.quote_receipt(self.fx),
                    datetime.fromtimestamp(at, UTC),
                )
                rate = 1 / float(fx["Bid"])
            limit = option_price(identity, float(q["Ask"]), ROUND_CEILING, 1)
            costs = cost_estimate(identity, limit, reference["conditions"], rate)
            costs["minimum_purchase_cost_gbp"] = cost_estimate(
                identity, float(q["Ask"]), reference["conditions"], rate
            )["minimum_purchase_cost_gbp"]
            costs["minimum_purchase_basis"] = "REGULAR_ASK_PLUS_ENTRY_COSTS"
            costs.pop("option")
            costs.update(
                quote_at=price.receipt,
                fx_at=self.fx.receipt,
                metadata_version=reference.get("version"),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            costs["reason"] = (
                "CONTRACT_OPTION_COST_SCHEMA_UNVERIFIED"
                if isinstance(exc, (TypeError, AttributeError))
                else str(exc)
            )
        return {
            "identity": json.loads(json.dumps(identity)),
            "quote": (price.value or {}).get("Quote"),
            "quote_received_at": price.receipt,
            "sizes": price.sizes(),
            "size_status": {
                side: "OBSERVED"
                if (standing := self.size_receipt(price, side)) is not None
                and 0 <= at - standing <= QUOTE_MAX_AGE_SECONDS
                else "STALE_OR_MISSING"
                for side in ("Bid", "Ask")
            },
            "size_received_at": dict(price.size_times),
            "quote_status": "STALE_OR_MISSING"
            if (standing := self.quote_receipt(price)) is None
            or not 0 <= at - standing <= QUOTE_MAX_AGE_SECONDS
            else "OBSERVED",
            "analytics": option_context.view(price.analytics, at),
            "chain_analytics": option_context.view(
                window.context.get("OPTIONS_CHAIN", {}).get("analytics", {}) if window else {}, at
            ),
            "subscription_started_at": price.subscription_started_at,
            "coverage_seconds": window.coverage(at) if window else 0,
            "metadata_version": reference.get("version"),
            "costs": costs,
            "analytics_basis": "PROVIDER_SUPPLIED_UNVERIFIED_NOT_STRATEGY_PROBABILITY",
        }

    async def restore_option(self, plan: dict[str, Any]) -> None:
        """Owned options keep their original future even after a display roll/restart."""
        option = plan["option"]
        state = MarketState(market=option["market"], identity=plan["underlying"])
        if option.get("expiry_instant"):
            state.expiry_instants = {str(option["expiry"])[:10]: utc(option["expiry_instant"])}
        await self.option_subscribe(
            state,
            {
                "Uic": option["uic"],
                "UnderlyingUic": option["underlying_uic"],
                "PutCall": option["right"],
                "StrikePrice": option["strike"],
                "OptionRootId": option["option_root_id"],
            },
        )

    async def refresh_option_metadata(self) -> None:
        for uic, (identity, _) in list(self.options.items()):
            prior = self.option_references.get(uic, {})
            # A minute ahead of option_view's age limit, so a clock never lands in the gap
            # between the costs going stale and this pass reaching them.
            if time.time() - prior.get("received_at", 0) < OPTION_METADATA_MAX_AGE_SECONDS - 60:
                continue
            try:
                raw = await self.client.request(
                    "GET",
                    f"/ref/v1/instruments/details/{uic}/FuturesOption",
                    params={
                        "AccountKey": self.client.oauth.account_key,
                        "FieldGroups": "TradingSessions",
                    },
                )
                if raw.get("Uic") != uic or raw.get("AssetType") != "FuturesOption":
                    raise ValueError("OPTION_REFERENCE_REFRESH_MISMATCH")
                verified = option_identity(
                    {
                        "uic": identity["underlying_uic"],
                        "environment": identity["environment"],
                        "market": identity["market"],
                        "contract_month": identity["contract_month"],
                    },
                    raw,
                    identity["option_root_id"],
                    {
                        "Uic": uic,
                        "UnderlyingUic": identity["underlying_uic"],
                        "PutCall": identity["right"],
                        "StrikePrice": identity["strike"],
                        "Expiry": identity["expiry"],
                    },
                )
                for name in (
                    "currency",
                    "multiplier",
                    "price_factor",
                    "minimum_quantity",
                    "lot_size",
                    "amount_decimals",
                    "tick_size",
                    "tick_size_scheme",
                    "settlement_style",
                ):
                    if verified.get(name) != identity.get(name):
                        raise ValueError("OPTION_REFERENCE_CONVENTION_CHANGED_" + name.upper())
                identity["trading_sessions"] = raw.get("TradingSessions")
                identity["is_tradable"] = raw.get("IsTradable")
                await self.option_conditions(identity, raw)
            except SaxoError:
                # Transport/HTTP failure: keep the verified entry and retry on the next pass.
                # Advancing received_at here would block exits for the whole metadata age.
                continue
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                prior["problem"] = (
                    "OPTION_METADATA_SCHEMA_UNVERIFIED"
                    if isinstance(exc, (TypeError, AttributeError))
                    else str(exc)
                )
                prior["received_at"] = time.time()
                self.option_references[uic] = prior
                identity["is_tradable"] = False

    async def release_unused_options(self, owned: set[int]) -> None:
        self.owned_options = owned
        for uic in list(self.options):
            wanted = any(uic in s.warm_uics for s in self.markets.values())
            if self.option_protected(uic) or (
                wanted and time.time() - self.option_required_at.get(uic, 0) <= 120
            ):
                continue
            async with self.subscription_lock:
                # Decided again under the lock: a clock decision may have protected (or a
                # candidate change retired) this option while the lock was held.
                if uic in self.options and not self.option_protected(uic):
                    await self.unsubscribe_option(uic)
        # Retired strikes keep separate rolling history until its ordinary expiry.
        for instrument, window in list(self.recorder.windows.items()):
            if window.identity.get("asset_type") != "FuturesOption":
                continue
            if window.identity["uic"] in self.options:
                continue
            if self.recorder.requires_subscription(instrument, time.time()):
                continue
            if not window.rows or window.rows[-1][0] < time.time() - ROLLING_WINDOW_SECONDS:
                self.recorder.windows.pop(instrument)

    def release_window_slot(self) -> None:
        """Retired strikes keep rolling history only while live contracts have window slots."""
        if len(self.recorder.windows) < SUBSCRIPTION_LIMIT:
            return
        at = time.time()
        retired = [
            k
            for k, w in self.recorder.windows.items()
            if w.identity.get("asset_type") == "FuturesOption"
            and w.identity["uic"] not in self.options
            and not self.recorder.requires_subscription(k, at)
        ]
        if not retired:
            raise SaxoError("RECORDER_INSTRUMENT_LIMIT")
        windows = self.recorder.windows
        windows.pop(min(retired, key=lambda k: windows[k].rows[-1][0] if windows[k].rows else 0))

    async def unsubscribe_option(self, uic: int) -> None:
        identity, _ = self.options[uic]
        for ref, subscription in list(self.subscriptions.items()):
            if subscription["target"] == str(uic):
                await self.client.request("DELETE", subscription["path"] + f"/{self.context}/{ref}")
                self.subscriptions.pop(ref, None)
        self.options.pop(uic, None)
        self.option_required_at.pop(uic, None)
        self.option_references.pop(uic, None)
        self.recorder.ingest(key(identity), "GAP", {"reason": "OPTION_UNSUBSCRIBED"}, time.time())

    async def startup(self) -> None:
        await self.subscribe("SESSION", {}, "SESSION")
        await saxo_balance.ensure_balance_subscription(self)
        for state in self.markets.values():
            await self.start_market(state)
        # Discover this environment's GBPUSD UIC; never reuse a SIM identifier in LIVE.
        # An FX problem blocks entries under its own name; it never resets price streams.
        await self.start_fx()

    async def retry_failed_startups(self) -> None:
        """A market (or the FX rate) whose connect-time steps failed once, say on a timeout or a
        rate limit, is retried a minute later from the history pass instead of staying blocked
        until an unrelated reconnect (review 2026-10-04). A market's problem never touches the
        others' feeds."""
        if not self.connected:
            return
        for state in self.markets.values():
            if state.problem and time.monotonic() >= state.startup_retry_at:
                state.startup_retry_at = time.monotonic() + 60
                await self.start_market(state)
        if self.fx.problem and not any(s["target"] == "FX" for s in self.subscriptions.values()):
            await self.start_fx()

    async def start_fx(self) -> None:
        try:
            await self.subscribe_fx()
        except (SaxoError, ValueError) as exc:
            self.fx.gap(
                str(exc) if isinstance(exc, SaxoError) else "FX_REFERENCE_SCHEMA_UNVERIFIED"
            )

    async def start_market(self, state: MarketState) -> None:
        """Verify the pinned future, then its price feed and the observation-only streams."""
        try:
            await self.discover(state)
        except SaxoError as exc:
            state.problem = str(exc)
        except (ValueError, KeyError, TypeError):
            state.problem = "REFERENCE_SCHEMA_NOT_VERIFIED"
        if not state.identity:
            return
        if not any(
            s["kind"] == "PRICE" and s["target"] == state.market
            for s in self.subscriptions.values()
        ):
            try:
                arguments = {
                    "Uic": state.identity["uic"],
                    "AssetType": "ContractFutures",
                    "Amount": 1,
                    "AccountKey": self.client.oauth.account_key,
                    "FieldGroups": [
                        "Quote",
                        "PriceInfo",
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
                        "PriceInfo",
                        "PriceInfoDetails",
                        "InstrumentPriceDetails",
                    ]
                    await self.subscribe("PRICE", arguments, state.market)
                state.live_since = datetime.now(UTC)
                state.last_clock = state.last_clock or state.live_since
            except (SaxoError, ValueError) as exc:
                state.problem = str(exc)
        await self.subscribe_board(state)
        await self.subscribe_board(state, "next")
        await self.subscribe_chart(state)
        await self.subscribe_next_contract(state)

    async def subscribe_chart(self, state: MarketState) -> None:
        """Streamed one-minute samples of the pinned future (2026-10-04): the final minute's bar
        reaches a clock decision as soon as the next sample starts instead of after a REST poll.
        REST stays the record and checks every bar taken from here; a failure here changes
        nothing but the stream's own status."""
        if not state.identity:
            return
        try:
            await self.subscribe(
                "CHART",
                {
                    "Uic": state.identity["uic"],
                    "AssetType": "ContractFutures",
                    "Horizon": 1,
                    "Count": 10,
                    "FieldGroups": ["Data", "ChartInfo"],
                },
                state.market,
            )
        except (SaxoError, ValueError) as exc:
            state.chart.gap(str(exc))

    async def subscribe_next_contract(self, state: MarketState) -> None:
        """The configured following contract month, streamed (quotes and depth) and recorded with
        every clock before its re-pin (2026-10-04) so the roll starts with history. Never a
        decision input; a failure here changes nothing but its own status."""
        selection = self.config.next_contracts.get(state.market)
        target = state.market + ":next"
        if not selection or any(
            s.get("kind") == "PRICE" and s.get("target") == target
            for s in self.subscriptions.values()
        ):
            return
        try:
            if state.market in self.next_contracts:
                identity = self.next_contracts[state.market][0]
            else:
                raw = await self.client.request(
                    "GET",
                    f"/ref/v1/instruments/details/{selection.uic}/ContractFutures",
                    params={
                        "AccountKey": self.client.oauth.account_key,
                        "FieldGroups": "TradingSessions",
                    },
                )
                identity = future_identity(state.market, self.config.data_environment, raw)
                if (
                    identity["uic"] != selection.uic
                    or identity["symbol"] != selection.symbol
                    or identity["exchange"] != selection.exchange
                    or identity["contract_month"] != selection.contract_month
                ):
                    raise SaxoError("CONFIGURED_NEXT_CONTRACT_IDENTITY_MISMATCH")
                identity["role"] = "NEXT_CONTRACT"
                self.recorder.register(key(identity), identity)
                self.recorder.metadata_version(
                    key(identity),
                    {"identity": identity, "reference": self.safe_reference(raw)},
                    time.time(),
                )
                self.next_contracts[state.market] = (identity, PriceState())
            await self.subscribe(
                "PRICE",
                {
                    "Uic": identity["uic"],
                    "AssetType": "ContractFutures",
                    "Amount": 1,
                    "AccountKey": self.client.oauth.account_key,
                    "FieldGroups": [
                        "Quote",
                        "PriceInfo",
                        "PriceInfoDetails",
                        "MarketDepth",
                        "InstrumentPriceDetails",
                    ],
                },
                target,
            )
            state.capabilities["next_contract"] = {
                "symbol": identity["symbol"],
                "uic": identity["uic"],
                "status": "SUBSCRIBED",
            }
        except (SaxoError, ValueError, KeyError, TypeError) as exc:
            state.capabilities["next_contract"] = {
                "problem": str(exc)
                if isinstance(exc, (SaxoError, ValueError))
                else "NEXT_CONTRACT_SCHEMA_UNVERIFIED"
            }

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
        if len(matches) != 1:
            self.fx.gap("GBPUSD_FX_INSTRUMENT_NOT_UNIQUE")
            return
        await self.subscribe(
            "PRICE",
            {
                "Uic": int(matches[0]["Identifier"]),
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
        self.stream_at = max(self.stream_at or receipt, receipt)
        if ref in self.pending:
            queued = {"message": message, "receipt": receipt}
            size = len(json.dumps(queued))
            if self.pending_bytes + size > 1024**2:
                raise SaxoError("SNAPSHOT_PENDING_QUEUE_LIMIT")
            self.pending[ref].append({**queued, "size": size})
            self.pending_bytes += size
            return
        if ref == "_heartbeat":
            envelopes = payload if isinstance(payload, list) else [payload]
            for envelope in envelopes:
                for heartbeat in envelope.get("Heartbeats", []):
                    subscription = self.subscriptions.get(heartbeat.get("OriginatingReferenceId"))
                    if subscription:
                        subscription["contact"] = time.monotonic()
                        if heartbeat.get("Reason") == "SubscriptionPermanentlyDisabled":
                            self.disabled_targets.add(
                                (subscription["kind"], subscription["target"])
                            )
                            if subscription["kind"] == "BALANCE":
                                self.balance_problem = "BALANCE_SUBSCRIPTION_DISABLED"
                            if subscription["kind"] == "PRICE":
                                self.mark_gap(
                                    subscription["target"], "SUBSCRIPTION_PERMANENTLY_DISABLED"
                                )
                            if subscription["kind"] == "CHART":
                                self.markets[subscription["target"]].chart.gap(
                                    "SUBSCRIPTION_PERMANENTLY_DISABLED"
                                )
                            self.subscriptions.pop(heartbeat["OriginatingReferenceId"], None)
                            continue
                        if (
                            subscription["kind"] == "BALANCE"
                            and heartbeat.get("Reason") != "NoNewData"
                        ):
                            self.balance_problem = "BALANCE_SUBSCRIPTION_DISABLED"
                        if subscription["kind"] == "PRICE":
                            price, identity = self.price_target(subscription["target"])
                            price.last_contact = receipt
                            if heartbeat.get("Reason") != "NoNewData":
                                # Saxo: accept the pause on this subscription alone. The next
                                # update clears it; quote age blocks entries meanwhile.
                                price.problem = "SUBSCRIPTION_TEMPORARILY_DISABLED"
                            if identity:
                                self.recorder.ingest(
                                    key(identity),
                                    "HEARTBEAT",
                                    heartbeat,
                                    receipt,
                                    message_id=message["message_id"],
                                    generation=price.generation,
                                    # This subscription's heartbeat only: the frame lists every
                                    # subscription and was stored once per PRICE feed it named.
                                    provider_message=heartbeat,
                                    observation_context=price.observation_context(),
                                )
                        if subscription["kind"] == "CHART":
                            chart = self.markets[subscription["target"]].chart
                            chart.last_contact = receipt
                            if heartbeat.get("Reason") != "NoNewData":
                                # A pause may have skipped samples: a fresh snapshot is needed.
                                chart.gap("SUBSCRIPTION_TEMPORARILY_DISABLED")
                                self.reset_refs.add(heartbeat["OriginatingReferenceId"])
            return
        if (
            ref == "_resetsubscriptions"
            and isinstance(payload, dict)
            and isinstance(payload.get("TargetReferenceIds"), list)
            and payload["TargetReferenceIds"]
        ):
            # Saxo names the affected subscriptions; only those take fresh snapshots.
            for target in payload["TargetReferenceIds"]:
                subscription = self.subscriptions.get(target)
                if subscription:
                    if subscription["kind"] == "PRICE":
                        self.mark_gap(subscription["target"], "SUBSCRIPTION_RESET_BY_PROVIDER")
                    elif subscription["kind"] == "CHART":
                        self.markets[subscription["target"]].chart.gap(
                            "SUBSCRIPTION_RESET_BY_PROVIDER"
                        )
                    self.reset_refs.add(target)
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
                self.update_session(data)
            elif subscription["kind"] == "BALANCE":
                if subscription["arguments"]["AccountKey"] == self.client.oauth.account_key:
                    saxo_balance.receive_balance(self, data, receipt)
                else:
                    self.balance_problem = "BALANCE_ACCOUNT_CHANGED"
            elif subscription["kind"] == "ACTIVITIES":
                self.last_activity = time.monotonic()
            elif subscription["kind"] == "BOARD":
                market, slot = board_target(subscription["target"])
                state = self.markets[market]
                try:
                    if slot:
                        self.record_chain(state, "UPDATE", data, receipt, slot)
                        state.next_option_board = merge_board(state.next_option_board, data)
                    else:
                        self.record_board(state, data, receipt)
                        self.record_chain(state, "UPDATE", data, receipt)
                        state.option_board = merge_board(state.option_board, data)
                except (KeyError, TypeError, ValueError) as exc:
                    # Observation only: a malformed chain delta takes a fresh snapshot of that
                    # window and never resets the trading feeds.
                    state.capabilities["chain_board_problem" + ("_next" if slot else "")] = (
                        str(exc) if isinstance(exc, ValueError) else "CHAIN_SCHEMA_UNVERIFIED"
                    )
                    self.reset_refs.add(ref)
            elif subscription["kind"] == "CHART":
                self.markets[subscription["target"]].chart.update(data, receipt)
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
                        dropped=not accepted,  # gapped state: awaiting a fresh snapshot
                        generation=ref,
                        provider_message=message,
                        observation_context=price.observation_context(),
                    )

    async def replace_reset_subscriptions(self) -> None:
        """Fresh snapshots for the subscriptions Saxo or a timeout singled out, one at a time.

        Runs as its own task beside the socket reader: a session upgrade renews every feed,
        ~37 paced POSTs, and the reader must keep draining meanwhile. A ref stays in
        `reset_refs` until replaced so the reader's silence check does not count it twice."""
        while self.reset_refs:
            ref = next(iter(self.reset_refs))
            subscription = self.subscriptions.get(ref)
            if subscription:
                await self.subscribe(
                    subscription["kind"], subscription["arguments"], subscription["target"], old=ref
                )
            self.reset_refs.discard(ref)

    async def run(self) -> None:
        backoff = 2
        while not self.stopping:
            setup: asyncio.Task[None] | None = None
            replacer: asyncio.Task[None] | None = None
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
                            # Saxo binds the renewed token to the open context (202 Accepted);
                            # only a refused re-authorisation forces fresh snapshots.
                            await self.client.authorize_stream(self.context)
                            generation = self.client.oauth.generation
                        silent = [
                            ref
                            for ref, s in self.subscriptions.items()
                            if time.monotonic() - s["contact"] > s["timeout"]
                            # Optional account/event streams must not reset price streams.
                            and s["kind"] not in {"BALANCE", "ACTIVITIES"}
                            and ref not in self.reset_refs  # already being replaced
                        ]
                        if len(silent) > 1 or any(
                            self.subscriptions[r]["kind"] == "SESSION" for r in silent
                        ):
                            # Several feeds or the session went quiet: the socket, not one feed.
                            raise SaxoError("SUBSCRIPTION_HEARTBEAT_TIMEOUT")
                        for ref in silent:
                            if self.subscriptions[ref]["kind"] == "PRICE":
                                self.mark_gap(
                                    self.subscriptions[ref]["target"],
                                    "SUBSCRIPTION_HEARTBEAT_TIMEOUT",
                                )
                            elif self.subscriptions[ref]["kind"] == "CHART":
                                self.markets[self.subscriptions[ref]["target"]].chart.gap(
                                    "SUBSCRIPTION_HEARTBEAT_TIMEOUT"
                                )
                            self.reset_refs.add(ref)
                        if replacer is not None and replacer.done():
                            replacer.result()  # a refused replacement reconnects, as before
                            replacer = None
                        if self.reset_refs and replacer is None:
                            replacer = asyncio.create_task(self.replace_reset_subscriptions())
            except Exception as exc:
                for task in (setup, replacer):
                    if task:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                setup = replacer = None
                self.connected = False
                self.problem = str(exc) if isinstance(exc, SaxoError) else "SAXO_RECONNECT_REQUIRED"
                for market in self.markets:
                    self.mark_gap(market, self.problem)
                    self.markets[market].chart.gap(self.problem)
                for market in self.next_contracts:
                    self.mark_gap(market + ":next", self.problem)
                for uic in self.options:
                    self.mark_gap(str(uic), self.problem)
                self.fx.gap(self.problem)
                # Explicitly delete server-owned subscriptions before a new context: one call per
                # service for the whole context rather than one per subscription (~48 paced calls).
                for path in sorted({s["path"] for s in self.subscriptions.values()}):
                    with suppress(SaxoError):
                        await self.client.request("DELETE", f"{path}/{self.context}")
                self.subscriptions.clear()
                self.reset_refs.clear()
                # A new context carries no server-side disables; a repeat just disables again.
                self.disabled_targets.clear()
                self.pending.clear()
                self.pending_bytes = 0
                await asyncio.sleep(backoff)
                backoff = min(60, backoff * 2)
            finally:
                for task in (setup, replacer):
                    if task:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    def capability_view(self, state: MarketState) -> dict[str, Any]:
        value = state.price.value or {}
        quote = value.get("Quote") or {}
        result = dict(state.capabilities)
        usable = 0
        for identity, price in self.options.values():
            if identity["market"] == state.market:
                with suppress(ValueError):
                    quote_check(price.value or {}, self.quote_receipt(price), datetime.now(UTC))
                    usable += 1
        recorded = result.get("options")  # the root, or the problem discovery/board hit
        result["options"] = {
            **(
                recorded
                if isinstance(recorded, dict)
                else {"problem": recorded}
                if recorded
                else {}
            ),
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
            chart_stream={
                **state.chart.view(time.time()),
                "boundary_bars_used": state.chart_used,
                "confirmed": state.chart_confirmed,
                "tripped": self.chart_stream_problem,
            },
            session=self.session,
            refresh_ms=state.price.refresh_ms,
            user_action=(
                "Delayed until you click 'Use real-time in SLRNO' on System"
                if self.session.get("TradeLevel") != "FullTradingAndChat"
                else ""
            ),
        )
        return result
