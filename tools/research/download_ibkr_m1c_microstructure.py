#!/usr/bin/env python3
"""Download raw IBKR historical ticks around frozen historical M1C events.

This is a standalone, research-only utility.  The Stocker application does not
import it, and it deliberately exposes no trading, account, position,
execution, portfolio, or PnL operation.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import functools
import hashlib
import importlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = "ibkr_historical_ticks"
DATASET_VERSION = "ibkr-m1c-microstructure-v0"
M1C_THRESHOLD = 0.4883337107940334
PERIOD_BOUNDS = {
    "development": ("2024-01-01", "2024-12-31"),
    "assessment": ("2025-01-01", "2025-08-22"),
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_SOURCE = REPOSITORY_ROOT / (
    "research/directional-readiness/20260728-m1c-tail-phase-v1/"
    "artifacts/primary/checkpoint_results_v1.parquet"
)
DEFAULT_FAMILY_DEFINITION = Path(__file__).with_name("ibkr_m1c_family_definition_v0.json")


def require_local_ibkr_socket(host: str, port: int) -> None:
    """Fail closed unless the configured endpoint is a listening loopback socket.

    Stocker's production guard additionally inspects Linux ``/proc/net``.  The
    standalone downloader can run on macOS through a loopback-only SSH tunnel,
    where ``/proc/net`` is unavailable, so it proves the local listener with a
    connection that sends no IBKR protocol data.
    """

    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeError(
            "blocked_unsafe_runtime_configuration: "
            f"ibkr host must be a literal loopback address: {host}"
        ) from exc
    if not address.is_loopback:
        raise RuntimeError(
            "blocked_unsafe_runtime_configuration: "
            f"ibkr host must be a literal loopback address: {host}"
        )
    if sys.platform.startswith("linux"):
        from stocker_prospective.ibkr import require_ibkr_socket_loopback_only

        require_ibkr_socket_loopback_only(host, port)
        return
    try:
        with socket.create_connection((host, port), timeout=2.0):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"blocked_ibkr_connection: configured_socket_not_listening:{port}"
        ) from exc


@dataclass(frozen=True)
class HistoricalEvent:
    """One frozen historical M1C checkpoint selected for raw-data capture."""

    period: str
    event_id: str
    symbol: str
    session_date: str
    family: str
    probability: float
    checkpoint: int | None
    t0_utc: datetime


@dataclass(frozen=True)
class ContractIdentity:
    """The exact qualified IBKR stock identity persisted with every event."""

    con_id: int
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    local_symbol: str
    primary_exchange: str | None


@dataclass(frozen=True)
class HistoricalPage:
    """One completed official historical-tick callback batch."""

    feed: str
    ticks: tuple[Any, ...]
    done: bool


class HistoricalRequestError(RuntimeError):
    """A classified IBKR request failure suitable for bounded retry policy."""

    def __init__(
        self,
        *,
        kind: str,
        code: int | None,
        message: str,
        request_count: int = 0,
        raw_rows: int = 0,
        duplicate_overlap_removed: int = 0,
    ) -> None:
        super().__init__(f"{kind}:{code}:{message}")
        self.kind = kind
        self.code = code
        self.message = message
        self.request_count = request_count
        self.raw_rows = raw_rows
        self.duplicate_overlap_removed = duplicate_overlap_removed


@dataclass(frozen=True)
class FeedDownload:
    """Raw rows and request accounting for one event/feed interval."""

    rows: tuple[dict[str, object], ...]
    completion_status: str
    request_count: int
    raw_rows: int
    final_rows: int
    duplicate_overlap_removed: int
    earliest_timestamp: datetime | None
    latest_timestamp: datetime | None
    maximum_gap_seconds: int | None
    provider_error: str | None = None


class HistoricalPageClient(Protocol):
    def request_historical_ticks(
        self,
        *,
        contract: object,
        feed: str,
        start_utc: datetime,
        number_of_ticks: int,
        use_rth: bool,
        timeout_seconds: float,
    ) -> HistoricalPage: ...


class DownloadClient(HistoricalPageClient, Protocol):
    api_version: str
    server_version: int | None
    tws_gateway_version: str | None

    def qualify_symbol(
        self, symbol: str, *, timeout_seconds: float
    ) -> tuple[ContractIdentity, object]: ...

    def close(self) -> None: ...


@dataclass
class _PendingHistoricalCallback:
    expected_feed: str
    event: threading.Event
    ticks: list[Any]
    done: bool = False
    error: HistoricalRequestError | None = None


class HistoricalCallbackRegistry:
    """Bounded correlation for contract and historical-tick callbacks."""

    def __init__(self, *, max_pending: int, max_ticks_per_request: int) -> None:
        if max_pending <= 0 or max_ticks_per_request <= 0:
            raise ValueError("callback bounds must be positive")
        self._max_pending = max_pending
        self._max_ticks = max_ticks_per_request
        self._pending: dict[int, _PendingHistoricalCallback] = {}
        self._lock = threading.RLock()

    def begin(self, request_id: int, *, expected_feed: str) -> None:
        with self._lock:
            if request_id in self._pending:
                raise HistoricalRequestError(
                    kind="callback",
                    code=None,
                    message="duplicate historical request ID",
                )
            if len(self._pending) >= self._max_pending:
                raise HistoricalRequestError(
                    kind="callback",
                    code=None,
                    message="bounded historical callback registry exhausted",
                )
            self._pending[request_id] = _PendingHistoricalCallback(
                expected_feed=expected_feed,
                event=threading.Event(),
                ticks=[],
            )

    def deliver(
        self,
        request_id: int,
        *,
        callback_feed: str,
        ticks: tuple[Any, ...],
        done: bool,
    ) -> None:
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                return
            if callback_feed != pending.expected_feed:
                pending.error = HistoricalRequestError(
                    kind="callback",
                    code=None,
                    message=(
                        f"unexpected {callback_feed} callback for {pending.expected_feed} request"
                    ),
                )
                pending.event.set()
                return
            if len(pending.ticks) + len(ticks) > self._max_ticks:
                pending.error = HistoricalRequestError(
                    kind="callback",
                    code=None,
                    message="historical callback tick bound exceeded",
                )
                pending.event.set()
                return
            pending.ticks.extend(ticks)
            pending.done = bool(done)
            if done:
                pending.event.set()

    def fail(self, request_id: int, error: HistoricalRequestError) -> None:
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                return
            pending.error = error
            pending.event.set()

    def fail_all(self, error: HistoricalRequestError) -> None:
        with self._lock:
            for pending in self._pending.values():
                pending.error = error
                pending.event.set()

    def wait(self, request_id: int, *, timeout_seconds: float) -> HistoricalPage:
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                raise HistoricalRequestError(
                    kind="callback",
                    code=None,
                    message="unknown historical request ID",
                )
            event = pending.event
        signalled = event.wait(timeout_seconds)
        with self._lock:
            pending = self._pending.pop(request_id)
        if not signalled:
            raise HistoricalRequestError(
                kind="timeout",
                code=None,
                message="historical request callback timeout",
            )
        if pending.error is not None:
            raise pending.error
        return HistoricalPage(
            feed=pending.expected_feed,
            ticks=tuple(pending.ticks),
            done=pending.done,
        )


class HistoricalDataOnlyFacade:
    """Narrow facade over the inseparable official ``EClient`` object."""

    __slots__ = ("__client",)

    def __init__(self, client: Any) -> None:
        self.__client = client

    def connect(self, host: str, port: int, client_id: int) -> Any:
        return self.__client.connect(host, port, client_id)

    def disconnect(self) -> None:
        self.__client.disconnect()

    def run(self) -> None:
        self.__client.run()

    def reqContractDetails(self, request_id: int, contract: object) -> None:  # noqa: N802
        self.__client.reqContractDetails(request_id, contract)

    def reqHistoricalTicks(self, *arguments: object) -> None:  # noqa: N802
        self.__client.reqHistoricalTicks(*arguments)

    def serverVersion(self) -> int | None:  # noqa: N802
        value = self.__client.serverVersion()
        return None if value is None else int(value)


def create_historical_stock_contract(contract_type: Callable[[], Any], symbol: str) -> Any:
    """Mirror Stocker's exact stock contract after provenance has been verified."""

    contract = contract_type()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def sha256_file(path: str | Path) -> str:
    """Hash one source or output without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def load_family_definition() -> dict[str, object]:
    payload = json.loads(DEFAULT_FAMILY_DEFINITION.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen M1C family definition must be a JSON object")
    source = payload.get("source")
    lineage = payload.get("m1c_lineage")
    expected_families = {
        "HARD_M1C": "score >= T",
        "P01_NEAR_M1": "0.8T < score <= T",
        "R01": "0.5T < score <= 0.8T",
    }
    if (
        payload.get("schema_version") != "ibkr-m1c-family-definition-v0"
        or not isinstance(source, dict)
        or source.get("git_commit") != "848a376d0157a77e00a56c1b216418ccc6acd5fc"
        or source.get("git_blob") != "b78bc00191dd878460b09b5348ffff418c656376"
        or source.get("sha256")
        != "9285afe58d7f32abe706c63632358914fccf0769222ceebeea3afe35e3b2b534"
        or not isinstance(lineage, dict)
        or lineage.get("threshold") != M1C_THRESHOLD
        or lineage.get("families") != expected_families
        or lineage.get("selection_precedence") != ["HARD_M1C", "P01_NEAR_M1", "R01"]
        or lineage.get("optimised_in_downloader") is not False
    ):
        raise ValueError("frozen M1C family definition failed lineage validation")
    return payload


def family_definition_metadata(sha256: str) -> dict[str, object]:
    payload = load_family_definition()
    lineage = cast(dict[str, object], payload["m1c_lineage"])
    return {
        "source": str(DEFAULT_FAMILY_DEFINITION.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256,
        "authoritative_source": payload["source"],
        "m1c_threshold": lineage["threshold"],
        "rules": lineage["families"],
        "selection_precedence": lineage["selection_precedence"],
        "optimised_in_downloader": False,
    }


def _m1c_family(probability: float) -> str | None:
    if probability >= M1C_THRESHOLD:
        return "HARD_M1C"
    if probability > 0.8 * M1C_THRESHOLD:
        return "P01_NEAR_M1"
    if probability > 0.5 * M1C_THRESHOLD:
        return "R01"
    return None


def _utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("frozen M1C signal timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def load_frozen_events(
    source: str | Path,
    *,
    period: str,
    include_r01: bool,
) -> list[HistoricalEvent]:
    """Project immutable M1C checkpoint rows into downloader event records."""

    load_family_definition()
    if period not in PERIOD_BOUNDS:
        raise ValueError(f"unsupported period: {period}")
    source_path = Path(source)
    required = {
        "row_id",
        "stock",
        "session",
        "partition",
        "checkpoint",
        "signal_timestamp",
        "M1C_probability",
        "m1c_high_tail_threshold_v1",
        "m1c_high_tail_v1",
    }
    schema_names = set(pq.read_schema(source_path).names)  # type: ignore[no-untyped-call]
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"canonical M1C source is missing columns: {','.join(missing)}")
    rows = pq.read_table(  # type: ignore[no-untyped-call]
        source_path, columns=sorted(required)
    ).to_pylist()
    start, end = PERIOD_BOUNDS[period]
    allowed = {"HARD_M1C", "P01_NEAR_M1"}
    if include_r01:
        allowed.add("R01")
    events: list[HistoricalEvent] = []
    for row in rows:
        session_date = str(row["session"])
        if row["partition"] != period or not start <= session_date <= end:
            continue
        t0_utc = _utc_datetime(row["signal_timestamp"])
        if t0_utc.year >= 2026:
            raise ValueError("protected 2026 M1C row entered an allowed historical period")
        probability = float(row["M1C_probability"])
        threshold = float(row["m1c_high_tail_threshold_v1"])
        if abs(threshold - M1C_THRESHOLD) > 1e-12:
            raise ValueError("canonical M1C row carries an unexpected frozen threshold")
        if bool(row["m1c_high_tail_v1"]) != (probability >= M1C_THRESHOLD):
            raise ValueError("canonical M1C row contradicts its frozen high-tail flag")
        family = _m1c_family(probability)
        if family not in allowed:
            continue
        event_id = str(row["row_id"])
        symbol = str(row["stock"])
        checkpoint_value = row["checkpoint"]
        if not event_id or not symbol:
            raise ValueError("canonical M1C event identity is incomplete")
        events.append(
            HistoricalEvent(
                period=period,
                event_id=event_id,
                symbol=symbol,
                session_date=session_date,
                family=family,
                probability=probability,
                checkpoint=None if checkpoint_value is None else int(checkpoint_value),
                t0_utc=t0_utc,
            )
        )
    events.sort(key=lambda event: (event.t0_utc, event.symbol, event.event_id))
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("canonical M1C event IDs are not unique within the period")
    return events


def select_validation_events(
    events: list[HistoricalEvent],
    *,
    count: int,
    family: str,
) -> list[HistoricalEvent]:
    """Choose a deterministic sample favouring distinct stocks and sessions."""

    if count <= 0:
        raise ValueError("validation count must be positive")
    candidates = [event for event in events if event.family == family]
    selected: list[HistoricalEvent] = []
    symbols: set[str] = set()
    sessions: set[str] = set()
    while candidates and len(selected) < count:
        index = max(
            range(len(candidates)),
            key=lambda item: (
                candidates[item].symbol not in symbols,
                candidates[item].session_date not in sessions,
                -item,
            ),
        )
        event = candidates.pop(index)
        selected.append(event)
        symbols.add(event.symbol)
        sessions.add(event.session_date)
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} {family} events are available")
    return selected


def event_window(event: HistoricalEvent) -> tuple[datetime, datetime]:
    return event.t0_utc - timedelta(minutes=5), event.t0_utc + timedelta(minutes=20)


def qualify_exact_stock(
    symbol: str,
    candidates: list[object],
) -> tuple[ContractIdentity, object]:
    """Accept exactly one exact USD stock contract and reject ambiguity."""

    matches: list[object] = []
    for candidate in candidates:
        try:
            con_id = int(getattr(candidate, "conId", 0))
        except (TypeError, ValueError):
            continue
        if (
            str(getattr(candidate, "symbol", "")) == symbol
            and str(getattr(candidate, "secType", "")) == "STK"
            and str(getattr(candidate, "currency", "")) == "USD"
            and con_id > 0
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"contract qualification returned {len(matches)} exact matches")
    contract = matches[0]
    candidate_symbol = str(getattr(contract, "symbol", ""))
    sec_type = str(getattr(contract, "secType", ""))
    currency = str(getattr(contract, "currency", ""))
    con_id = int(getattr(contract, "conId", 0))
    exchange = str(getattr(contract, "exchange", ""))
    local_symbol = str(getattr(contract, "localSymbol", ""))
    if not exchange or not local_symbol:
        raise ValueError("qualified stock contract identity is incomplete")
    primary_exchange_value = str(getattr(contract, "primaryExchange", ""))
    return (
        ContractIdentity(
            con_id=con_id,
            symbol=candidate_symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            local_symbol=local_symbol,
            primary_exchange=primary_exchange_value or None,
        ),
        contract,
    )


def classify_historical_error(code: int, message: str) -> HistoricalRequestError:
    """Classify documented historical pacing, permission, and connection errors."""

    normalized = message.casefold()
    if code in {100, 420} or (code == 162 and "pacing" in normalized):
        kind = "pacing"
    elif code in {354, 10089, 10090, 10186, 10197} or (
        code == 162 and ("permission" in normalized or "not subscribed" in normalized)
    ):
        kind = "permission"
    elif code in {200, 321}:
        kind = "contract"
    elif code in {326, 502, 504, 1100, 1300}:
        kind = "connection"
    elif code in {165, 366} or (code == 162 and "no data" in normalized):
        kind = "no_data"
    else:
        kind = "provider"
    return HistoricalRequestError(kind=kind, code=code, message=message)


def _maximum_gap(rows: list[dict[str, object]]) -> int | None:
    timestamps = sorted({cast(int, row["provider_epoch_seconds"]) for row in rows})
    if len(timestamps) < 2:
        return None
    return max(right - left for left, right in zip(timestamps, timestamps[1:], strict=False))


def download_feed_pages(
    *,
    client: HistoricalPageClient,
    provider_contract: object,
    event: HistoricalEvent,
    contract: ContractIdentity,
    feed: str,
    requested_start_utc: datetime,
    requested_end_utc: datetime,
    page_size: int = 1000,
    request_timeout_seconds: float = 30.0,
    max_pages: int = 5000,
    max_retries: int = 3,
    retry_backoff_seconds: float = 15.0,
    use_rth: bool = False,
    sleeper: Callable[[float], None] = time.sleep,
    before_request: Callable[[str], None] | None = None,
) -> FeedDownload:
    """Download one complete interval with deterministic whole-second paging."""

    if feed not in {"TRADES", "BID_ASK"}:
        raise ValueError("only TRADES and BID_ASK historical feeds are permitted")
    if requested_start_utc.tzinfo is None or requested_end_utc.tzinfo is None:
        raise ValueError("requested interval must be timezone-aware")
    if requested_start_utc >= requested_end_utc:
        raise ValueError("requested interval must have positive duration")
    if not 1 <= page_size <= 1000 or max_pages <= 0 or max_retries < 0:
        raise ValueError("historical request bounds are invalid")

    cursor = requested_start_utc.astimezone(UTC).replace(microsecond=0)
    end_utc = requested_end_utc.astimezone(UTC)
    merged: list[dict[str, object]] = []
    raw_rows = 0
    overlap_removed = 0
    request_count = 0
    completed = False

    for _page_number in range(max_pages):
        retries = 0
        while True:
            if before_request is not None:
                before_request(feed)
            request_count += 1
            try:
                page = client.request_historical_ticks(
                    contract=provider_contract,
                    feed=feed,
                    start_utc=cursor,
                    number_of_ticks=page_size,
                    use_rth=use_rth,
                    timeout_seconds=request_timeout_seconds,
                )
                break
            except HistoricalRequestError as error:
                if error.kind != "pacing" or retries >= max_retries:
                    raise HistoricalRequestError(
                        kind=error.kind,
                        code=error.code,
                        message=error.message,
                        request_count=request_count,
                        raw_rows=raw_rows,
                        duplicate_overlap_removed=overlap_removed,
                    ) from error
                sleeper(retry_backoff_seconds * (2**retries))
                retries += 1

        if page.feed != feed:
            raise HistoricalRequestError(
                kind="callback",
                code=None,
                message=f"unexpected {page.feed} callback for {feed} request",
                request_count=request_count,
                raw_rows=raw_rows,
                duplicate_overlap_removed=overlap_removed,
            )
        if not page.done:
            raise HistoricalRequestError(
                kind="callback",
                code=None,
                message="historical callback ended without done=true",
                request_count=request_count,
                raw_rows=raw_rows,
                duplicate_overlap_removed=overlap_removed,
            )
        raw_rows += len(page.ticks)
        if feed == "TRADES":
            normalized = [normalize_trade_tick(tick, event, contract) for tick in page.ticks]
        else:
            normalized = [normalize_bid_ask_tick(tick, event, contract) for tick in page.ticks]
        merged, removed = merge_provider_pages(merged, normalized)
        overlap_removed += removed

        if not normalized:
            completed = True
            break
        last_epoch = max(cast(int, row["provider_epoch_seconds"]) for row in normalized)
        if last_epoch >= int(end_utc.timestamp()) or len(page.ticks) < page_size:
            completed = True
            break
        next_cursor = datetime.fromtimestamp(last_epoch + 1, tz=UTC)
        if next_cursor <= cursor:
            raise HistoricalRequestError(
                kind="pagination",
                code=None,
                message="historical pagination made no forward progress",
                request_count=request_count,
                raw_rows=raw_rows,
                duplicate_overlap_removed=overlap_removed,
            )
        cursor = next_cursor

    if not completed:
        raise HistoricalRequestError(
            kind="pagination",
            code=None,
            message=f"historical pagination exceeded {max_pages} pages",
            request_count=request_count,
            raw_rows=raw_rows,
            duplicate_overlap_removed=overlap_removed,
        )

    bounded = [
        row
        for row in merged
        if requested_start_utc.timestamp()
        <= cast(int, row["provider_epoch_seconds"])
        <= requested_end_utc.timestamp()
    ]
    identified = assign_provider_identities(bounded, feed=feed, event_id=event.event_id)
    earliest = identified[0]["provider_timestamp_utc"] if identified else None
    latest = identified[-1]["provider_timestamp_utc"] if identified else None
    return FeedDownload(
        rows=tuple(identified),
        completion_status="COMPLETE",
        request_count=request_count,
        raw_rows=raw_rows,
        final_rows=len(identified),
        duplicate_overlap_removed=overlap_removed,
        earliest_timestamp=earliest if isinstance(earliest, datetime) else None,
        latest_timestamp=latest if isinstance(latest, datetime) else None,
        maximum_gap_seconds=_maximum_gap(identified),
    )


TRADE_SCHEMA = pa.schema(
    [
        ("m1c_event_id", pa.string()),
        ("symbol", pa.string()),
        ("con_id", pa.int64()),
        ("m1c_t0_utc", pa.timestamp("us", tz="UTC")),
        ("provider_timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("provider_epoch_seconds", pa.int64()),
        ("provider_sequence", pa.int64()),
        ("provider_content_occurrence", pa.int64()),
        ("provider_event_identity", pa.string()),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("provider_size_raw", pa.string()),
        ("exchange", pa.string()),
        ("special_conditions", pa.string()),
        ("tick_attrib_past_limit", pa.bool_()),
        ("tick_attrib_unreported", pa.bool_()),
        ("provider_attributes_json", pa.string()),
        ("source", pa.string()),
        ("feed", pa.string()),
    ]
)

BID_ASK_SCHEMA = pa.schema(
    [
        ("m1c_event_id", pa.string()),
        ("symbol", pa.string()),
        ("con_id", pa.int64()),
        ("m1c_t0_utc", pa.timestamp("us", tz="UTC")),
        ("provider_timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("provider_epoch_seconds", pa.int64()),
        ("provider_sequence", pa.int64()),
        ("provider_content_occurrence", pa.int64()),
        ("provider_event_identity", pa.string()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("bid_size", pa.float64()),
        ("ask_size", pa.float64()),
        ("provider_bid_size_raw", pa.string()),
        ("provider_ask_size_raw", pa.string()),
        ("tick_attrib_bid_past_low", pa.bool_()),
        ("tick_attrib_ask_past_high", pa.bool_()),
        ("provider_attributes_json", pa.string()),
        ("source", pa.string()),
        ("feed", pa.string()),
    ]
)


def _schema_for_feed(feed: str) -> pa.Schema:
    if feed == "TRADES":
        return TRADE_SCHEMA
    if feed == "BID_ASK":
        return BID_ASK_SCHEMA
    raise ValueError(f"unsupported feed: {feed}")


def write_feed_parquet(
    path: str | Path,
    *,
    feed: str,
    rows: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> None:
    """Atomically write a stable, explicit raw-feed Parquet schema."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = _schema_for_feed(feed)
    table = pa.Table.from_pylist(list(rows), schema=schema)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        pq.write_table(  # type: ignore[no-untyped-call]
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            row_group_size=100_000,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def event_output_directory(output_root: str | Path, event: HistoricalEvent) -> Path:
    if event.period not in PERIOD_BOUNDS:
        raise ValueError("event period is outside the frozen downloader periods")
    event_id_path = Path(event.event_id)
    if (
        not event.event_id
        or event.event_id in {".", ".."}
        or event_id_path.is_absolute()
        or len(event_id_path.parts) != 1
        or "/" in event.event_id
        or "\\" in event.event_id
    ):
        raise ValueError("unsafe event ID")
    if "/" in event.symbol or "\\" in event.symbol or event.symbol in {"", ".", ".."}:
        raise ValueError("unsafe event symbol")
    return Path(output_root) / event.period / event.symbol / event.event_id


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _feed_metadata(feed: str, result: FeedDownload) -> dict[str, object]:
    if result.completion_status == "COMPLETE":
        permission_status = "PASS"
    elif result.completion_status == "BLOCKED_PERMISSION":
        permission_status = "BLOCKED"
    else:
        permission_status = "UNKNOWN"
    return {
        "request_count": result.request_count,
        "raw_rows": result.raw_rows,
        "final_rows": result.final_rows,
        "earliest_timestamp": _iso(result.earliest_timestamp),
        "latest_timestamp": _iso(result.latest_timestamp),
        "completion_status": result.completion_status,
        "duplicate_overlap_removed": result.duplicate_overlap_removed,
        "maximum_gap_seconds": result.maximum_gap_seconds,
        "permission_status": permission_status,
        "provider_error": result.provider_error,
        "output_file": "trades.parquet" if feed == "TRADES" else "bid_ask.parquet",
    }


def build_event_metadata(
    *,
    event: HistoricalEvent,
    contract: ContractIdentity | None,
    requested_start_utc: datetime,
    requested_end_utc: datetime,
    feeds: dict[str, FeedDownload],
    ibkr_api_version: str,
    server_version: int | None,
    tws_gateway_version: str | None,
    git_commit: str,
    downloaded_at_utc: datetime,
    canonical_source_sha256: str,
    family_definition_sha256: str,
) -> dict[str, object]:
    """Build the complete event-side provenance record without analysis fields."""

    payload: dict[str, object] = {
        "event_id": event.event_id,
        "dataset_version": DATASET_VERSION,
        "canonical_M1C_source_sha256": canonical_source_sha256,
        "family_definition": family_definition_metadata(family_definition_sha256),
        "period": event.period,
        "symbol": event.symbol,
        "session_date": event.session_date,
        "family": event.family,
        "m1c_probability": event.probability,
        "checkpoint": event.checkpoint,
        "t0_utc": _iso(event.t0_utc),
        "requested_start_utc": _iso(requested_start_utc),
        "requested_end_utc": _iso(requested_end_utc),
        "contract": (
            None
            if contract is None
            else {
                "conId": contract.con_id,
                "symbol": contract.symbol,
                "secType": contract.sec_type,
                "exchange": contract.exchange,
                "currency": contract.currency,
                "localSymbol": contract.local_symbol,
                "primaryExchange": contract.primary_exchange,
            }
        ),
        "contract_status": "PASS" if contract is not None else "BLOCKED",
        "ibkr_api_version": ibkr_api_version,
        "ibkr_server_version": server_version,
        "tws_gateway_version": tws_gateway_version,
        "download_timestamp_utc": _iso(downloaded_at_utc),
        "git_commit": git_commit,
        "research_only": True,
        "execution_enabled": False,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "direction_analysis_performed": False,
    }
    for feed, result in feeds.items():
        payload[feed] = _feed_metadata(feed, result)
    return payload


def _valid_feed_parquet(
    path: Path,
    *,
    feed: str,
    event: HistoricalEvent,
    requested_start_utc: datetime,
    requested_end_utc: datetime,
    expected_rows: int,
) -> bool:
    try:
        if pq.read_schema(path) != _schema_for_feed(feed):  # type: ignore[no-untyped-call]
            return False
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except Exception:
        return False
    if table.num_rows != expected_rows:
        return False
    values = table.to_pylist()
    try:
        rebuilt = assign_provider_identities(values, feed=feed, event_id=event.event_id)
    except (TypeError, ValueError):
        return False
    timestamps = [row["provider_timestamp_utc"] for row in values]
    if not all(
        isinstance(timestamp, datetime)
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        for timestamp in timestamps
    ):
        return False
    return (
        [row["provider_sequence"] for row in values] == list(range(expected_rows))
        and timestamps == sorted(timestamps)
        and [row["provider_event_identity"] for row in values]
        == [row["provider_event_identity"] for row in rebuilt]
        and all(
            row["m1c_event_id"] == event.event_id
            and row["symbol"] == event.symbol
            and isinstance(row["con_id"], int)
            and row["con_id"] > 0
            and row["m1c_t0_utc"] == event.t0_utc
            and row["source"] == SOURCE
            and row["feed"] == feed
            and isinstance(row["provider_content_occurrence"], int)
            and row["provider_content_occurrence"] >= 0
            and isinstance(row["provider_event_identity"], str)
            and len(row["provider_event_identity"]) == 64
            and requested_start_utc <= row["provider_timestamp_utc"] <= requested_end_utc
            for row in values
        )
    )


def resume_completed_feeds(
    event_directory: str | Path,
    *,
    event: HistoricalEvent,
    requested_start_utc: datetime,
    requested_end_utc: datetime,
    canonical_source_sha256: str,
    family_definition_sha256: str,
) -> set[str]:
    """Return only feeds whose metadata, interval, and Parquet all validate."""

    directory = Path(event_directory)
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return set()
    if (
        metadata.get("event_id") != event.event_id
        or metadata.get("dataset_version") != DATASET_VERSION
        or metadata.get("canonical_M1C_source_sha256") != canonical_source_sha256
        or not isinstance((family_definition := metadata.get("family_definition")), dict)
        or family_definition.get("sha256") != family_definition_sha256
        or metadata.get("period") != event.period
        or metadata.get("symbol") != event.symbol
        or metadata.get("session_date") != event.session_date
        or metadata.get("family") != event.family
        or metadata.get("m1c_probability") != event.probability
        or metadata.get("checkpoint") != event.checkpoint
        or metadata.get("t0_utc") != _iso(event.t0_utc)
        or metadata.get("requested_start_utc") != _iso(requested_start_utc)
        or metadata.get("requested_end_utc") != _iso(requested_end_utc)
    ):
        return set()
    completed: set[str] = set()
    for feed, filename in (("TRADES", "trades.parquet"), ("BID_ASK", "bid_ask.parquet")):
        details = metadata.get(feed)
        if not isinstance(details, dict) or details.get("completion_status") != "COMPLETE":
            continue
        expected_rows = details.get("final_rows")
        if not isinstance(expected_rows, int):
            continue
        if _valid_feed_parquet(
            directory / filename,
            feed=feed,
            event=event,
            requested_start_utc=requested_start_utc,
            requested_end_utc=requested_end_utc,
            expected_rows=expected_rows,
        ):
            completed.add(feed)
    return completed


def _read_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_event_metadata(
    *,
    previous: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    merged = dict(current)
    for feed in ("TRADES", "BID_ASK"):
        if feed not in merged and isinstance(previous.get(feed), dict):
            merged[feed] = previous[feed]
    return merged


def _failed_feed(error: Exception) -> FeedDownload:
    kind = error.kind if isinstance(error, HistoricalRequestError) else "provider"
    return FeedDownload(
        rows=(),
        completion_status=f"BLOCKED_{kind.upper()}",
        request_count=(error.request_count if isinstance(error, HistoricalRequestError) else 0),
        raw_rows=(error.raw_rows if isinstance(error, HistoricalRequestError) else 0),
        final_rows=0,
        duplicate_overlap_removed=(
            error.duplicate_overlap_removed if isinstance(error, HistoricalRequestError) else 0
        ),
        earliest_timestamp=None,
        latest_timestamp=None,
        maximum_gap_seconds=None,
        provider_error=str(error),
    )


def _metadata_paths(output_root: Path) -> list[Path]:
    paths: list[Path] = []
    for period in PERIOD_BOUNDS:
        paths.extend(sorted((output_root / period).glob("*/*/metadata.json")))
    return paths


SUMMARY_COLUMNS = (
    "period",
    "event_id",
    "symbol",
    "feed",
    "family",
    "T0",
    "requested_start",
    "requested_end",
    "status",
    "request_count",
    "raw_rows",
    "final_rows",
    "first_tick",
    "last_tick",
    "duplicate_overlap_removed",
    "maximum_gap_seconds",
    "contract_status",
    "permission_status",
    "provider_error",
    "output_file",
)


def write_download_summary(output_root: str | Path) -> list[dict[str, object]]:
    root = Path(output_root)
    rows: list[dict[str, object]] = []
    for path in _metadata_paths(root):
        metadata = _read_metadata(path)
        for feed in ("TRADES", "BID_ASK"):
            details = metadata.get(feed)
            if not isinstance(details, dict):
                continue
            output_value = details.get("output_file")
            output_file = str(path.parent / str(output_value)) if output_value else ""
            rows.append(
                {
                    "period": path.parts[-4],
                    "event_id": metadata.get("event_id"),
                    "symbol": metadata.get("symbol"),
                    "feed": feed,
                    "family": metadata.get("family"),
                    "T0": metadata.get("t0_utc"),
                    "requested_start": metadata.get("requested_start_utc"),
                    "requested_end": metadata.get("requested_end_utc"),
                    "status": details.get("completion_status"),
                    "request_count": details.get("request_count"),
                    "raw_rows": details.get("raw_rows"),
                    "final_rows": details.get("final_rows"),
                    "first_tick": details.get("earliest_timestamp"),
                    "last_tick": details.get("latest_timestamp"),
                    "duplicate_overlap_removed": details.get("duplicate_overlap_removed"),
                    "maximum_gap_seconds": details.get("maximum_gap_seconds"),
                    "contract_status": metadata.get("contract_status"),
                    "permission_status": details.get("permission_status"),
                    "provider_error": details.get("provider_error"),
                    "output_file": output_file,
                }
            )
    rows.sort(key=lambda row: (str(row["period"]), str(row["event_id"]), str(row["feed"])))
    output = root / "download_summary.csv"
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def _event_status(metadata: dict[str, object]) -> str:
    statuses = [
        details.get("completion_status")
        for feed in ("TRADES", "BID_ASK")
        if isinstance((details := metadata.get(feed)), dict)
    ]
    if len(statuses) == 2 and all(status == "COMPLETE" for status in statuses):
        return "complete"
    if any(status == "COMPLETE" for status in statuses):
        return "partial"
    if metadata.get("contract_status") != "PASS":
        return "blocked"
    if any(str(status).startswith("BLOCKED") for status in statuses):
        return "blocked"
    return "partial"


def write_download_manifest(
    output_root: str | Path,
    *,
    canonical_source: Path,
    canonical_source_sha256: str,
    family_definition_sha256: str,
    families: list[str],
    ibkr_api_version: str | None,
    ibkr_server_version: int | None,
    tws_gateway_version: str | None,
    downloaded_at_utc: datetime,
) -> dict[str, object]:
    root = Path(output_root)
    metadata_rows = [_read_metadata(path) for path in _metadata_paths(root)]
    statuses = [_event_status(metadata) for metadata in metadata_rows]
    period_counts = {
        period: sum(1 for metadata in metadata_rows if metadata.get("period") == period)
        for period in PERIOD_BOUNDS
    }
    contracts = sorted(
        {
            (
                int(contract["conId"]),
                str(contract["symbol"]),
                str(contract["exchange"]),
                str(contract["currency"]),
                str(contract["localSymbol"]),
            )
            for metadata in metadata_rows
            if isinstance((contract := metadata.get("contract")), dict)
        }
    )
    total_trades = sum(
        int(details.get("final_rows", 0))
        for metadata in metadata_rows
        if isinstance((details := metadata.get("TRADES")), dict)
    )
    total_bid_ask = sum(
        int(details.get("final_rows", 0))
        for metadata in metadata_rows
        if isinstance((details := metadata.get("BID_ASK")), dict)
    )
    try:
        source_name = str(canonical_source.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        source_name = str(canonical_source.resolve())
    manifest: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "canonical_M1C_source": source_name,
        "canonical_M1C_source_sha256": canonical_source_sha256,
        "family_definition": family_definition_metadata(family_definition_sha256),
        "development_dates": {"start": "2024-01-01", "end": "2024-12-31"},
        "assessment_dates": {"start": "2025-01-01", "end": "2025-08-22"},
        "families": families,
        "window": {"pre_minutes": 5, "post_minutes": 20},
        "feed_types": ["TRADES", "BID_ASK"],
        "event_counts": {**period_counts, "total": len(metadata_rows)},
        "complete_events": statuses.count("complete"),
        "partial_events": statuses.count("partial"),
        "blocked_events": statuses.count("blocked"),
        "total_TRADES_rows": total_trades,
        "total_BID_ASK_rows": total_bid_ask,
        "symbols": sorted(
            {str(metadata["symbol"]) for metadata in metadata_rows if metadata.get("symbol")}
        ),
        "contracts": [
            {
                "conId": contract[0],
                "symbol": contract[1],
                "exchange": contract[2],
                "currency": contract[3],
                "localSymbol": contract[4],
            }
            for contract in contracts
        ],
        "ibkr_api_version": ibkr_api_version,
        "ibkr_server_version": ibkr_server_version,
        "tws_gateway_version": tws_gateway_version,
        "download_timestamp_utc": _iso(downloaded_at_utc),
        "no_order_invariant": "PASS",
        "direction_analysis_performed": False,
        "research_only": True,
        "execution_enabled": False,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json_atomic(root / "download_manifest.json", manifest)
    return manifest


def validate_existing_dataset_root(
    output_root: str | Path,
    *,
    canonical_source_sha256: str,
    family_definition_sha256: str,
    families: list[str],
) -> dict[str, object]:
    """Refuse to mix incompatible frozen inputs in one dataset root."""

    root = Path(output_root)
    for path in _metadata_paths(root):
        metadata = _read_metadata(path)
        if (
            metadata.get("dataset_version") != DATASET_VERSION
            or metadata.get("canonical_M1C_source_sha256") != canonical_source_sha256
            or not isinstance((definition := metadata.get("family_definition")), dict)
            or definition.get("sha256") != family_definition_sha256
            or metadata.get("family") not in families
        ):
            raise ValueError(f"existing event metadata has incompatible provenance: {path}")
    manifest = _read_metadata(root / "download_manifest.json")
    if not manifest:
        return {}
    expected = {
        "dataset_version": DATASET_VERSION,
        "canonical_M1C_source_sha256": canonical_source_sha256,
        "family_definition": family_definition_metadata(family_definition_sha256),
        "families": families,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "existing dataset root has incompatible provenance: " + ", ".join(mismatches)
        )
    return manifest


def record_connection_blocked_events(
    *,
    root: Path,
    events: list[HistoricalEvent],
    starting_index: int,
    total_events: int,
    resume: bool,
    source_hash: str,
    family_definition_hash: str,
    error: Exception,
    git_commit: str,
    now: Callable[[], datetime],
) -> tuple[int, int, int, int]:
    """Persist provenance for every remaining event after connection failure."""

    failure = _failed_feed(HistoricalRequestError(kind="connection", code=None, message=str(error)))
    complete = partial = blocked = skipped = 0
    for offset, event in enumerate(events):
        index = starting_index + offset
        requested_start, requested_end = event_window(event)
        directory = event_output_directory(root, event)
        completed = (
            resume_completed_feeds(
                directory,
                event=event,
                requested_start_utc=requested_start,
                requested_end_utc=requested_end,
                canonical_source_sha256=source_hash,
                family_definition_sha256=family_definition_hash,
            )
            if resume
            else set()
        )
        if completed == {"TRADES", "BID_ASK"}:
            complete += 1
            skipped += 1
            continue
        previous = _read_metadata(directory / "metadata.json")
        attempted_at = now()
        metadata = build_event_metadata(
            event=event,
            contract=None,
            requested_start_utc=requested_start,
            requested_end_utc=requested_end,
            feeds={feed: failure for feed in ("TRADES", "BID_ASK") if feed not in completed},
            ibkr_api_version="unknown",
            server_version=None,
            tws_gateway_version=None,
            git_commit=git_commit,
            downloaded_at_utc=attempted_at,
            canonical_source_sha256=source_hash,
            family_definition_sha256=family_definition_hash,
        )
        metadata["connection_error"] = str(error)
        metadata = _merge_event_metadata(previous=previous, current=metadata)
        metadata["connection_attempt"] = {
            "status": "BLOCKED_CONNECTION",
            "download_timestamp_utc": _iso(attempted_at),
            "provider_error": str(error),
        }
        if completed:
            for key in (
                "contract",
                "contract_status",
                "ibkr_api_version",
                "ibkr_server_version",
                "tws_gateway_version",
                "git_commit",
                "download_timestamp_utc",
            ):
                if key in previous:
                    metadata[key] = previous[key]
        write_json_atomic(directory / "metadata.json", metadata)
        status = _event_status(metadata)
        partial += status == "partial"
        blocked += status == "blocked"
        print(f"[{index}/{total_events}] {event.symbol} {event.family} BLOCKED connection")
    return complete, partial, blocked, skipped


def run_download(
    *,
    period: str,
    source: str | Path,
    output_root: str | Path,
    include_r01: bool,
    resume: bool,
    validation_hard_count: int | None,
    event_limit: int | None,
    client_factory: Callable[[], DownloadClient],
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    before_request: Callable[[str], None] | None,
    git_commit: str,
    now: Callable[[], datetime],
) -> dict[str, int]:
    """Run one physically isolated period and refresh root-level provenance."""

    source_path = Path(source)
    source_hash = sha256_file(source_path)
    family_definition_hash = sha256_file(DEFAULT_FAMILY_DEFINITION)
    families = ["P01_NEAR_M1", "HARD_M1C"]
    if include_r01:
        families.append("R01")
    events = load_frozen_events(source_path, period=period, include_r01=include_r01)
    if validation_hard_count is not None:
        events = select_validation_events(events, count=validation_hard_count, family="HARD_M1C")
    if event_limit is not None:
        if event_limit <= 0:
            raise ValueError("event limit must be positive")
        events = events[:event_limit]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    prior_manifest = validate_existing_dataset_root(
        root,
        canonical_source_sha256=source_hash,
        family_definition_sha256=family_definition_hash,
        families=families,
    )
    client: DownloadClient | None = None
    contracts: dict[str, tuple[ContractIdentity, object]] = {}
    complete_events = 0
    partial_events = 0
    blocked_events = 0
    skipped_events = 0
    api_version: str | None = None
    server_version: int | None = None
    gateway_version: str | None = None
    try:
        for index, event in enumerate(events, start=1):
            requested_start, requested_end = event_window(event)
            directory = event_output_directory(root, event)
            completed = (
                resume_completed_feeds(
                    directory,
                    event=event,
                    requested_start_utc=requested_start,
                    requested_end_utc=requested_end,
                    canonical_source_sha256=source_hash,
                    family_definition_sha256=family_definition_hash,
                )
                if resume
                else set()
            )
            if completed == {"TRADES", "BID_ASK"}:
                skipped_events += 1
                complete_events += 1
                print(f"[{index}/{len(events)}] {event.symbol} {event.family} SKIP (resume)")
                continue
            if client is None:
                try:
                    client = client_factory()
                except Exception as error:
                    counts = record_connection_blocked_events(
                        root=root,
                        events=events[index - 1 :],
                        starting_index=index,
                        total_events=len(events),
                        resume=resume,
                        source_hash=source_hash,
                        family_definition_hash=family_definition_hash,
                        error=error,
                        git_commit=git_commit,
                        now=now,
                    )
                    complete_events += counts[0]
                    partial_events += counts[1]
                    blocked_events += counts[2]
                    skipped_events += counts[3]
                    break
                api_version = client.api_version
                server_version = client.server_version
                gateway_version = client.tws_gateway_version
            previous = _read_metadata(directory / "metadata.json")
            feed_results: dict[str, FeedDownload] = {}
            try:
                if event.symbol not in contracts:
                    contracts[event.symbol] = client.qualify_symbol(
                        event.symbol,
                        timeout_seconds=request_timeout_seconds,
                    )
                contract, provider_contract = contracts[event.symbol]
            except Exception as error:
                contract_failure = _failed_feed(
                    HistoricalRequestError(
                        kind="contract",
                        code=(error.code if isinstance(error, HistoricalRequestError) else None),
                        message=str(error),
                    )
                )
                failed_feeds = {
                    feed: contract_failure
                    for feed in ("TRADES", "BID_ASK")
                    if feed not in completed
                }
                metadata = build_event_metadata(
                    event=event,
                    contract=None,
                    requested_start_utc=requested_start,
                    requested_end_utc=requested_end,
                    feeds=failed_feeds,
                    ibkr_api_version=api_version or "unknown",
                    server_version=server_version,
                    tws_gateway_version=gateway_version,
                    git_commit=git_commit,
                    downloaded_at_utc=now(),
                    canonical_source_sha256=source_hash,
                    family_definition_sha256=family_definition_hash,
                )
                metadata["contract_error"] = str(error)
                metadata = _merge_event_metadata(previous=previous, current=metadata)
                write_json_atomic(directory / "metadata.json", metadata)
                status = _event_status(metadata)
                partial_events += status == "partial"
                blocked_events += status == "blocked"
                print(f"[{index}/{len(events)}] {event.symbol} {event.family} BLOCKED contract")
                continue

            print(f"[{index}/{len(events)}] {event.symbol} {event.family}")
            for feed, filename in (("TRADES", "trades.parquet"), ("BID_ASK", "bid_ask.parquet")):
                if feed in completed:
                    print(f"{feed:<8}: SKIP (resume)")
                    continue
                try:
                    result = download_feed_pages(
                        client=client,
                        provider_contract=provider_contract,
                        event=event,
                        contract=contract,
                        feed=feed,
                        requested_start_utc=requested_start,
                        requested_end_utc=requested_end,
                        request_timeout_seconds=request_timeout_seconds,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                        before_request=before_request,
                    )
                    write_feed_parquet(directory / filename, feed=feed, rows=result.rows)
                except Exception as error:
                    result = _failed_feed(error)
                feed_results[feed] = result
                current = build_event_metadata(
                    event=event,
                    contract=contract,
                    requested_start_utc=requested_start,
                    requested_end_utc=requested_end,
                    feeds=feed_results,
                    ibkr_api_version=api_version or "unknown",
                    server_version=server_version,
                    tws_gateway_version=gateway_version,
                    git_commit=git_commit,
                    downloaded_at_utc=now(),
                    canonical_source_sha256=source_hash,
                    family_definition_sha256=family_definition_hash,
                )
                current = _merge_event_metadata(previous=previous, current=current)
                write_json_atomic(directory / "metadata.json", current)
                previous = current
                print(f"{feed:<8}: {result.completion_status} {result.final_rows:,} rows")
            status = _event_status(previous)
            complete_events += status == "complete"
            partial_events += status == "partial"
            blocked_events += status == "blocked"
    finally:
        if client is not None:
            client.close()

    if api_version is None and isinstance(prior_manifest.get("ibkr_api_version"), str):
        api_version = str(prior_manifest["ibkr_api_version"])
    prior_server_version = prior_manifest.get("ibkr_server_version")
    if server_version is None and isinstance(prior_server_version, int):
        server_version = prior_server_version
    if gateway_version is None and isinstance(prior_manifest.get("tws_gateway_version"), str):
        gateway_version = str(prior_manifest["tws_gateway_version"])
    write_download_summary(root)
    write_download_manifest(
        root,
        canonical_source=source_path,
        canonical_source_sha256=source_hash,
        family_definition_sha256=family_definition_hash,
        families=families,
        ibkr_api_version=api_version,
        ibkr_server_version=server_version,
        tws_gateway_version=gateway_version,
        downloaded_at_utc=now(),
    )
    return {
        "selected_events": len(events),
        "complete_events": complete_events,
        "partial_events": partial_events,
        "blocked_events": blocked_events,
        "skipped_events": skipped_events,
    }


class ConservativeHistoricalPacer:
    """Sequential sliding-window limiter for IBKR historical requests."""

    def __init__(
        self,
        *,
        requests_per_window: int,
        window_seconds: float,
        request_rate_per_second: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_window <= 0 or window_seconds <= 0 or request_rate_per_second <= 0:
            raise ValueError("pacing bounds must be positive")
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._minimum_interval = max(1.0 / request_rate_per_second, 0.4)
        self._monotonic = monotonic
        self._sleep = sleeper
        self._requests: collections.deque[float] = collections.deque()
        self._burst_requests: collections.deque[float] = collections.deque()
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def acquire(self, *, weight: int = 1) -> None:
        if weight <= 0 or weight > self._requests_per_window:
            raise ValueError("pacing weight is outside the configured request window")
        with self._lock:
            while True:
                current = self._monotonic()
                while self._requests and current - self._requests[0] >= self._window_seconds:
                    self._requests.popleft()
                while self._burst_requests and current - self._burst_requests[0] >= 2.0:
                    self._burst_requests.popleft()
                waits = [0.0]
                if len(self._requests) + weight > self._requests_per_window:
                    expiry_index = len(self._requests) + weight - self._requests_per_window - 1
                    waits.append(self._window_seconds - (current - self._requests[expiry_index]))
                if len(self._burst_requests) + weight > 5:
                    expiry_index = len(self._burst_requests) + weight - 6
                    waits.append(2.0 - (current - self._burst_requests[expiry_index]))
                if self._last_request is not None:
                    waits.append(self._minimum_interval * weight - (current - self._last_request))
                wait_seconds = max(waits)
                if wait_seconds <= 0:
                    self._requests.extend(current for _ in range(weight))
                    self._burst_requests.extend(current for _ in range(weight))
                    self._last_request = current
                    return
                self._sleep(wait_seconds)


class OfficialHistoricalClient:
    """Synchronous historical-only wrapper around the verified official API."""

    _INFORMATIONAL_CODES = frozenset({2104, 2106, 2107, 2108, 2119, 2158})

    def __init__(
        self,
        *,
        config: Any,
        client_id: int,
        provenance_path: str | Path | None = None,
    ) -> None:
        from stocker_prospective.ibkr import (
            require_official_ibkr_api,
        )
        from stocker_prospective.market_data import RequestIdAllocator

        if not bool(config.read_only):
            raise RuntimeError("blocked_unsafe_runtime_configuration: IBKR read_only must be true")
        if str(config.expected_environment) not in {"read_only", "paper", "live_read_only"}:
            raise RuntimeError("blocked_unsafe_runtime_configuration: invalid IBKR environment")
        if client_id <= 0:
            raise RuntimeError("blocked_unsafe_runtime_configuration: client ID must be non-zero")
        require_local_ibkr_socket(str(config.host), int(config.port))
        api = require_official_ibkr_api(provenance_path)
        client_module = importlib.import_module("ibapi.client")
        contract_module = importlib.import_module("ibapi.contract")
        wrapper_module = importlib.import_module("ibapi.wrapper")
        EClient = client_module.EClient
        EWrapper = wrapper_module.EWrapper

        self._registry = HistoricalCallbackRegistry(
            max_pending=4,
            max_ticks_per_request=50_000,
        )
        self._request_ids = RequestIdAllocator(start=1)
        self._ready = threading.Event()
        self._closed = threading.Event()
        registry = self._registry
        ready = self._ready
        closed = self._closed
        request_ids = self._request_ids
        informational_codes = self._INFORMATIONAL_CODES

        class _OfficialHistoricalCallbackClient(EWrapper, EClient):  # type: ignore[misc,valid-type]
            def __init__(self) -> None:
                EWrapper.__init__(self)
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                request_ids.synchronise(int(orderId))
                ready.set()

            def connectionClosed(self) -> None:  # noqa: N802
                closed.set()
                registry.fail_all(
                    HistoricalRequestError(
                        kind="connection",
                        code=1100,
                        message="official IBKR socket closed",
                    )
                )

            def error(
                self,
                reqId: int,
                errorTime: int,
                errorCode: int,
                errorString: str,
                advancedOrderRejectJson: str = "",
            ) -> None:
                del errorTime, advancedOrderRejectJson
                code = int(errorCode)
                if code in informational_codes:
                    return
                error = classify_historical_error(code, str(errorString))
                if int(reqId) >= 0:
                    registry.fail(int(reqId), error)
                elif error.kind == "connection":
                    registry.fail_all(error)

            def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
                registry.deliver(
                    int(reqId),
                    callback_feed="CONTRACT",
                    ticks=(contractDetails.contract,),
                    done=False,
                )

            def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
                registry.deliver(
                    int(reqId),
                    callback_feed="CONTRACT",
                    ticks=(),
                    done=True,
                )

            def historicalTicks(self, reqId: int, ticks: list[Any], done: bool) -> None:  # noqa: N802
                registry.deliver(
                    int(reqId),
                    callback_feed="MIDPOINT",
                    ticks=tuple(ticks),
                    done=bool(done),
                )

            def historicalTicksBidAsk(  # noqa: N802
                self, reqId: int, ticks: list[Any], done: bool
            ) -> None:
                registry.deliver(
                    int(reqId),
                    callback_feed="BID_ASK",
                    ticks=tuple(ticks),
                    done=bool(done),
                )

            def historicalTicksLast(  # noqa: N802
                self, reqId: int, ticks: list[Any], done: bool
            ) -> None:
                registry.deliver(
                    int(reqId),
                    callback_feed="TRADES",
                    ticks=tuple(ticks),
                    done=bool(done),
                )

        raw_client = _OfficialHistoricalCallbackClient()
        self._client = HistoricalDataOnlyFacade(raw_client)
        self._stock_contract_factory: Callable[[str], Any] = lambda symbol: (
            create_historical_stock_contract(
                contract_module.Contract,
                symbol,
            )
        )
        self.api_version = str(getattr(api, "__version__", "unknown"))
        self.tws_gateway_version = (
            None
            if getattr(config, "tws_or_gateway_version", None) is None
            else str(config.tws_or_gateway_version)
        )
        self._thread = threading.Thread(
            target=self._client.run,
            name="stocker-ibkr-historical-ticks",
            daemon=True,
        )
        try:
            self._client.connect(str(config.host), int(config.port), client_id)
            self._thread.start()
            if not self._ready.wait(float(config.connect_timeout_seconds)):
                raise HistoricalRequestError(
                    kind="connection",
                    code=None,
                    message="IBKR connection handshake timeout",
                )
            self.server_version = self._client.serverVersion()
        except Exception:
            self.close()
            raise

    def qualify_symbol(
        self, symbol: str, *, timeout_seconds: float
    ) -> tuple[ContractIdentity, object]:
        contract = self._stock_contract_factory(symbol)
        request_id = self._request_ids.next()
        self._registry.begin(request_id, expected_feed="CONTRACT")
        try:
            self._client.reqContractDetails(request_id, contract)
        except Exception as error:
            self._registry.fail(
                request_id,
                HistoricalRequestError(kind="provider", code=None, message=str(error)),
            )
        page = self._registry.wait(request_id, timeout_seconds=timeout_seconds)
        return qualify_exact_stock(symbol, list(page.ticks))

    def request_historical_ticks(
        self,
        *,
        contract: object,
        feed: str,
        start_utc: datetime,
        number_of_ticks: int,
        use_rth: bool,
        timeout_seconds: float,
    ) -> HistoricalPage:
        if feed not in {"TRADES", "BID_ASK"}:
            raise ValueError("only TRADES and BID_ASK may be requested")
        request_id = self._request_ids.next()
        self._registry.begin(request_id, expected_feed=feed)
        start = start_utc.astimezone(UTC).strftime("%Y%m%d %H:%M:%S UTC")
        try:
            self._client.reqHistoricalTicks(
                request_id,
                contract,
                start,
                "",
                number_of_ticks,
                feed,
                int(use_rth),
                False,
                [],
            )
        except Exception as error:
            self._registry.fail(
                request_id,
                HistoricalRequestError(kind="provider", code=None, message=str(error)),
            )
        return self._registry.wait(request_id, timeout_seconds=timeout_seconds)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.disconnect()
        thread = getattr(self, "_thread", None)
        if isinstance(thread, threading.Thread) and thread.is_alive():
            thread.join(timeout=5.0)


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", required=True, choices=tuple(PERIOD_BOUNDS))
    parser.add_argument("--config", type=Path, default=Path("/etc/stocker/prospective.yaml"))
    parser.add_argument("--source", type=Path, default=DEFAULT_CANONICAL_SOURCE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "research-data" / DATASET_VERSION,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-r01", action="store_true")
    parser.add_argument("--validation-hard-count", type=int)
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--request-timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--provenance", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    from stocker_prospective.config import load_prospective_config

    config = load_prospective_config(args.config)
    if bool(config.risk.trading_enabled):
        raise RuntimeError("blocked_unsafe_runtime_configuration: trading must remain disabled")
    if config.ibkr.port is None:
        raise RuntimeError("blocked_unsafe_runtime_configuration: IBKR port must be explicit")
    if not bool(config.ibkr.read_only):
        raise RuntimeError("blocked_unsafe_runtime_configuration: IBKR read_only must be true")
    client_id = config.ibkr.client_id if args.client_id is None else args.client_id
    timeout = (
        config.ibkr.request_timeout_seconds
        if args.request_timeout_seconds is None
        else args.request_timeout_seconds
    )
    pacer = ConservativeHistoricalPacer(
        requests_per_window=config.ibkr.historical_requests_per_window,
        window_seconds=config.ibkr.historical_request_window_seconds,
        request_rate_per_second=config.ibkr.request_rate_per_second,
    )

    def client_factory() -> DownloadClient:
        return OfficialHistoricalClient(
            config=config.ibkr,
            client_id=client_id,
            provenance_path=args.provenance,
        )

    result = run_download(
        period=args.period,
        source=args.source,
        output_root=args.output_root,
        include_r01=args.include_r01,
        resume=args.resume,
        validation_hard_count=args.validation_hard_count,
        event_limit=args.event_limit,
        client_factory=client_factory,
        request_timeout_seconds=timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        before_request=lambda feed: pacer.acquire(weight=2 if feed == "BID_ASK" else 1),
        git_commit=_git_commit(),
        now=lambda: datetime.now(UTC),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["blocked_events"] else 0


def _provider_attributes(value: object) -> dict[str, object]:
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): item
            for key, item in sorted(attributes.items())
            if not str(key).startswith("_")
            and isinstance(item, (str, int, float, bool, type(None)))
        }
    return {}


def _size(value: object) -> float:
    return float(str(value))


def normalize_trade_tick(
    tick: Any,
    event: HistoricalEvent,
    contract: ContractIdentity,
) -> dict[str, object]:
    """Normalise one official ``HistoricalTickLast`` without deriving trade side."""

    timestamp = datetime.fromtimestamp(int(tick.time), tz=UTC)
    attributes = _provider_attributes(tick.tickAttribLast)
    return {
        "m1c_event_id": event.event_id,
        "symbol": event.symbol,
        "con_id": contract.con_id,
        "m1c_t0_utc": event.t0_utc.astimezone(UTC),
        "provider_timestamp_utc": timestamp,
        "provider_epoch_seconds": int(tick.time),
        "price": float(tick.price),
        "size": _size(tick.size),
        "provider_size_raw": str(tick.size),
        "exchange": str(tick.exchange),
        "special_conditions": str(tick.specialConditions),
        "tick_attrib_past_limit": bool(attributes.get("pastLimit", False)),
        "tick_attrib_unreported": bool(attributes.get("unreported", False)),
        "provider_attributes_json": json.dumps(attributes, sort_keys=True, separators=(",", ":")),
        "source": SOURCE,
        "feed": "TRADES",
    }


def normalize_bid_ask_tick(
    tick: Any,
    event: HistoricalEvent,
    contract: ContractIdentity,
) -> dict[str, object]:
    """Normalise one official ``HistoricalTickBidAsk`` without derived fields."""

    timestamp = datetime.fromtimestamp(int(tick.time), tz=UTC)
    attributes = _provider_attributes(tick.tickAttribBidAsk)
    return {
        "m1c_event_id": event.event_id,
        "symbol": event.symbol,
        "con_id": contract.con_id,
        "m1c_t0_utc": event.t0_utc.astimezone(UTC),
        "provider_timestamp_utc": timestamp,
        "provider_epoch_seconds": int(tick.time),
        "bid": float(tick.priceBid),
        "ask": float(tick.priceAsk),
        "bid_size": _size(tick.sizeBid),
        "ask_size": _size(tick.sizeAsk),
        "provider_bid_size_raw": str(tick.sizeBid),
        "provider_ask_size_raw": str(tick.sizeAsk),
        "tick_attrib_bid_past_low": bool(attributes.get("bidPastLow", False)),
        "tick_attrib_ask_past_high": bool(attributes.get("askPastHigh", False)),
        "provider_attributes_json": json.dumps(attributes, sort_keys=True, separators=(",", ":")),
        "source": SOURCE,
        "feed": "BID_ASK",
    }


def _provider_content(row: dict[str, object]) -> str:
    excluded = {
        "m1c_event_id",
        "symbol",
        "con_id",
        "m1c_t0_utc",
        "provider_sequence",
        "provider_content_occurrence",
        "provider_event_identity",
    }
    serialisable = {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in row.items()
        if key not in excluded
    }
    return json.dumps(serialisable, sort_keys=True, separators=(",", ":"), allow_nan=False)


def assign_provider_identities(
    rows: list[dict[str, object]],
    *,
    feed: str,
    event_id: str,
) -> list[dict[str, object]]:
    """Assign stable full-content identities while retaining repeated events."""

    occurrences: dict[str, int] = {}
    identified: list[dict[str, object]] = []
    for sequence, source_row in enumerate(rows):
        row = dict(source_row)
        content = _provider_content(row)
        occurrence = occurrences.get(content, 0)
        occurrences[content] = occurrence + 1
        identity_material = f"{event_id}\0{feed}\0{sequence}\0{content}\0{occurrence}".encode()
        row.update(
            {
                "provider_sequence": sequence,
                "provider_content_occurrence": occurrence,
                "provider_event_identity": hashlib.sha256(identity_material).hexdigest(),
            }
        )
        identified.append(row)
    return identified


def merge_provider_pages(
    existing: list[dict[str, object]],
    incoming: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    """Remove the maximal exact suffix/prefix overlap between ordered pages.

    A content sequence is compared rather than timestamps, so distinct events
    sharing an IBKR whole-second timestamp remain present.
    """

    maximum = min(len(existing), len(incoming))
    existing_keys = [_provider_content(row) for row in existing[-maximum:]]
    incoming_keys = [_provider_content(row) for row in incoming[:maximum]]
    overlap = 0
    for candidate in range(maximum, 0, -1):
        if existing_keys[-candidate:] == incoming_keys[:candidate]:
            overlap = candidate
            break
    return [*existing, *incoming[overlap:]], overlap


if __name__ == "__main__":
    sys.exit(main())
