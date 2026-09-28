"""Bounded, optional displayed-depth observations. No rule or order authority."""

import asyncio
import gzip
import hashlib
import json
import math
import os
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stocker_execution.config import MARKETS, L2Config
from stocker_execution.pacing import OPTIONAL, priority
from stocker_execution.subscriptions import Subscription, Subscriptions

POLICY = "L2_CLOCK_CONTEXT_V1"
EVENT_BYTES = 8192  # includes Python objects, five-row checkpoints and writer scratch headroom


def stamp() -> datetime:
    return datetime.now(UTC)


class Book:
    def __init__(self, market: str, contract: Any, generation: int, config: L2Config):
        self.market, self.contract, self.generation, self.config = (
            market,
            contract,
            generation,
            config,
        )
        self.sides: list[list[dict[str, Any]]] = [[], []]  # ask=0, bid=1
        self.sequence = 0
        self.request_id: int | None = None
        self.events: deque[dict[str, Any]] = deque()
        self.last_receipt: datetime | None = None
        self.snapshot_at: datetime | None = None
        self.valid = False
        self.gaps = self.resets = 0

    def receive(self, raw: dict[str, Any], at: datetime) -> list[dict[str, Any]]:
        if raw["generation"] != self.generation:
            return []
        self.request_id = raw.get("request_id", self.request_id)
        result = []
        if raw["kind"] == "ROW" and (
            self.snapshot_at is None or (at - self.snapshot_at).total_seconds() >= 1
        ):
            result.append(
                self.record(
                    "SNAPSHOT",
                    at,
                    {
                        "asks": list(self.sides[0]),
                        "bids": list(self.sides[1]),
                        "valid": self.valid,
                        "contract": self.contract.dict(),
                    },
                )
            )
            self.snapshot_at = at
        kind = raw["kind"]
        if kind in {"RESET", "GAP"}:
            self.sides = [[], []]
            self.valid = False
            self.resets += kind == "RESET"
            self.gaps += kind == "GAP"
        elif kind == "ROW":
            side, row, operation = raw["side"], raw["position"], raw["operation"]
            invalid = (
                side not in {0, 1}
                or operation not in {0, 1, 2}
                or not isinstance(row, int)
                or not 0 <= row < self.config.levels
                or raw.get("smart", False)
            )
            levels = self.sides[side] if side in {0, 1} else []
            invalid |= row > len(levels) if operation == 0 else row >= len(levels)
            if operation != 2:
                invalid |= any(not math.isfinite(raw[k]) or raw[k] <= 0 for k in ("price", "size"))
            if invalid:
                kind = "GAP"
                raw = {"reason": "INVALID_DEPTH_ROW", "rejected_row": repr(raw)[:500]}
                self.sides = [[], []]
                self.valid = False
                self.gaps += 1
            else:
                if operation == 2:
                    levels.pop(row)
                else:
                    level = {
                        "price": raw["price"],
                        "size": raw["size"],
                        "market_maker": raw.get("market_maker", "")[:64],
                    }
                    if operation == 0:
                        levels.insert(row, level)
                        del levels[self.config.levels :]
                    else:
                        levels[row] = level
                ordered = all(
                    a["price"] <= b["price"]
                    for a, b in zip(self.sides[0], self.sides[0][1:], strict=False)
                ) and all(
                    a["price"] >= b["price"]
                    for a, b in zip(self.sides[1], self.sides[1][1:], strict=False)
                )
                crossed = bool(
                    self.sides[0]
                    and self.sides[1]
                    and self.sides[1][0]["price"] > self.sides[0][0]["price"]
                )
                self.valid = (
                    ordered
                    and not crossed
                    and all(len(s) == self.config.levels for s in self.sides)
                )
                if not ordered or crossed:
                    kind, raw = "GAP", {"reason": "INVALID_DEPTH_ORDERING"}
                    self.sides, self.valid = [[], []], False
                    self.gaps += 1
        self.last_receipt = at
        result.append(self.record(kind, at, raw))
        self.events.extend(result)
        cutoff = at - timedelta(seconds=self.config.pre_seconds)
        # Retain a replayable checkpoint, not a tail whose starting book is missing.
        while len(self.events) > self.config.events_per_book or (
            len(self.events) > 1 and self.events[1]["at"] < cutoff.isoformat()
        ):
            self.events.popleft()
        while self.events and self.events[0]["kind"] != "SNAPSHOT":
            self.events.popleft()
        return result

    def record(self, kind: str, at: datetime, payload: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        return {
            "kind": kind,
            "at": at.isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "sequence": self.sequence,
            "generation": self.generation,
            "con_id": self.contract.conId,
            "request_id": self.request_id,
            "valid": self.valid,
            "payload": payload,
        }

    def view(self, at: datetime) -> dict[str, Any]:
        fresh = bool(
            self.last_receipt
            and (at - self.last_receipt).total_seconds() <= self.config.stale_seconds
        )
        sizes = [sum(r["size"] for r in side) for side in self.sides]
        usable = fresh and self.valid
        return {
            "status": "COLLECTING" if usable else "INCOMPLETE",
            "asks": self.sides[0] if fresh else [],
            "bids": self.sides[1] if fresh else [],
            "received_levels": [len(s) for s in self.sides],
            "requested_levels": self.config.levels,
            "last_receipt": self.last_receipt.isoformat() if self.last_receipt else None,
            "spread": self.sides[0][0]["price"] - self.sides[1][0]["price"] if usable else None,
            "displayed_ask_size": sizes[0] if usable else None,
            "displayed_bid_size": sizes[1] if usable else None,
            "imbalance": (sizes[1] - sizes[0]) / sum(sizes) if usable and sum(sizes) else None,
            "generation": self.generation,
            "sequence": self.sequence,
            "resets": self.resets,
            "gaps": self.gaps,
            "fresh": fresh,
            "timestamp_basis": "LOCAL_CALLBACK_RECEIPT",
        }


def coverage(
    events: list[dict[str, Any]],
    signal: datetime,
    end: datetime,
    pre_seconds: int = 120,
    stale_seconds: int = 30,
) -> dict[str, Any]:
    """Observed spans only; never extend a silent or missing tail to the desired endpoint."""
    result: dict[str, Any] = {
        "coverage_basis": "RECEIVED_CALLBACK_SPANS; COMPLETE_BOOK_SPANS_SEPARATE",
        "resets": sum(e["kind"] == "RESET" for e in events),
        "gaps": sum(e["kind"] in {"GAP", "RELEASE"} for e in events),
    }
    for complete in (False, True):
        segments: list[tuple[datetime, datetime]] = []
        start = last = None
        generation = None
        for event in events:
            at = datetime.fromisoformat(event["at"])
            gap = event["kind"] in {"GAP", "RESET", "RELEASE"} or (
                complete and not event.get("valid", False)
            )
            if gap or event["generation"] != generation:
                if start is not None and last is not None:
                    segments.append((start, last))
                start = last = None
                generation = event["generation"]
            if not gap and event["kind"] in {"ROW", "SNAPSHOT"}:
                if last is not None and (at - last).total_seconds() > stale_seconds:
                    segments.append((start or last, last))
                    start = None
                start = start or at
                last = at
        if start is not None and last is not None:
            segments.append((start, last))
        prefix = "complete_book_" if complete else ""
        result[prefix + "pre_seconds"] = sum(
            max(
                0,
                (min(b, signal) - max(a, signal - timedelta(seconds=pre_seconds))).total_seconds(),
            )
            for a, b in segments
            if a < signal
        )
        result[prefix + "post_seconds"] = sum(
            max(0, (min(b, end) - max(a, signal)).total_seconds())
            for a, b in segments
            if b > signal
        )
        if not complete:
            result["capture_start"] = min((a for a, _ in segments), default=None)
            result["capture_end"] = max((b for _, b in segments), default=None)
    return result


class DepthObserver:
    def __init__(self, data: Subscriptions, store: Any, config: L2Config, directory: Path):
        self.data, self.store, self.config, self.directory = data, store, config, directory
        self.books: dict[str, Book] = {}
        self.feeds: dict[str, Subscription] = {}
        self.allocated_at: dict[str, datetime] = {}
        self.last_allocation = dict.fromkeys(MARKETS, datetime.min.replace(tzinfo=UTC))
        self.last_capture = dict(self.last_allocation)
        self.captures: dict[str, dict[str, Any]] = {}
        self.latest_capture: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=config.writer_queue)
        self.queued_bytes = self.disk_used = self.gaps = self.late_callbacks = 0
        self.high_water = 0
        self.paused = ""
        self.routes: list[Any] | None = None
        self.route_retry = 0.0
        self.retry_at: dict[str, float] = {}
        self.closed = False
        self.worker: asyncio.Task[Any] | None = None

    def memory_used(self) -> int:
        return (
            EVENT_BYTES
            * (
                sum(len(b.events) for b in self.books.values())
                + sum(len(c["events"]) for c in self.captures.values())
            )
            + self.queued_bytes
        )

    async def start(self) -> None:
        self.disk_used = await asyncio.to_thread(
            lambda: sum(p.stat().st_size for p in self.directory.glob("*.gz"))
        )
        self.store.recover_depth()
        self.worker = asyncio.create_task(self.write_loop())

    def enqueue(self, item: dict[str, Any]) -> bool:
        size = max(4096, len(item.get("events", [])) * EVENT_BYTES)
        if self.queue.full() or self.memory_used() + size > self.config.memory_bytes:
            self.paused = "RECORDING_QUEUE_OR_MEMORY_LIMIT"
            self.gaps += 1
            return False
        item["memory_bytes"] = size
        self.queued_bytes += size
        self.queue.put_nowait(item)
        self.high_water = max(self.high_water, self.memory_used())
        return True

    def log(self, action: str, market: str, reason: str = "") -> None:
        self.enqueue(
            {
                "kind": "POLICY",
                "at": stamp().isoformat(),
                "action": action,
                "market": market,
                "reason": reason,
                "policy": POLICY,
            }
        )

    def receive(self, market: str, generation: int, raw: dict[str, Any]) -> None:
        book = self.books.get(market)
        if book and book.generation == 0:
            book.generation = generation
        if not book or generation != book.generation:
            self.late_callbacks += 1
            return
        if self.paused:
            return
        # At most two emitted records (checkpoint + row), copied into at most one clock window.
        if self.memory_used() + 8 * EVENT_BYTES > self.config.memory_bytes:
            self.paused = "RECORDING_MEMORY_LIMIT"
            self.gaps += 1
            return
        emitted = book.receive(raw, stamp())
        for capture in self.captures.values():
            if capture["market"] == market:
                capture["events"].extend(e for e in emitted if e["at"] <= capture["end"])
        self.high_water = max(self.high_water, self.memory_used())

    def signal(self, event: dict[str, Any]) -> None:
        market, identity = event["market"], event["id"]
        signal = datetime.fromisoformat(event["signal_at"])
        book = self.books.get(market)
        captured = list(book.events) if book else []
        if self.memory_used() + len(captured) * EVENT_BYTES + 4096 > self.config.memory_bytes:
            self.paused = "CAPTURE_MEMORY_LIMIT"
            self.gaps += 1
            captured = []
        reason = (
            "DISABLED"
            if not self.config.enabled
            else self.paused or ("" if market in self.feeds else "L2_NOT_CAPTURED_CAPACITY")
        )
        summary = {
            "policy": POLICY,
            "requested": self.config.enabled,
            "allocated": bool(book),
            "target_pre_seconds": self.config.pre_seconds,
            "target_post_seconds": self.config.post_seconds,
            "pre_seconds": 0,
            "post_seconds": 0,
            "reason": reason,
            "status": "CAPTURING" if self.config.enabled and not self.paused else "NOT_CAPTURED",
            "signal_at": event["signal_at"],
            "trading_outcome": "JOIN_CORE_SIGNAL_BY_ID",
        }
        self.store.depth_capture(identity, summary)
        self.latest_capture[market] = {"id": identity, **summary}
        if not self.config.enabled or self.paused:
            return
        self.captures[identity] = {
            "kind": "CAPTURE",
            "id": identity,
            "market": market,
            "signal_at": event["signal_at"],
            "end": (signal + timedelta(seconds=self.config.post_seconds)).isoformat(),
            "events": captured,
            "summary": summary,
            "contract": book.contract.dict() if book else None,
        }
        if book:
            self.last_capture[market] = signal
        else:
            self.log("CAPACITY_DENIAL", market, reason)

    async def tick(
        self,
        candidates: dict[str, Any],
        next_times: dict[str, datetime],
        ready: bool,
        closed_markets: set[str] | None = None,
    ) -> None:
        at = stamp()
        for identity, capture in list(self.captures.items()):
            if at >= datetime.fromisoformat(capture["end"]) or self.paused or not ready:
                self.captures.pop(identity)
                result = coverage(
                    capture["events"],
                    datetime.fromisoformat(capture["signal_at"]),
                    min(at, datetime.fromisoformat(capture["end"])),
                    self.config.pre_seconds,
                    self.config.stale_seconds,
                )
                capture["summary"].update(
                    {k: v.isoformat() if isinstance(v, datetime) else v for k, v in result.items()}
                )
                capture["summary"]["status"] = "RETAINED"
                if not capture["events"]:
                    capture["summary"]["status"] = "NOT_CAPTURED"
                capture["summary"]["reason"] = self.paused or (
                    "DISCONNECTED" if not ready else capture["summary"]["reason"]
                )
                self.latest_capture[capture["market"]] = {"id": identity, **capture["summary"]}
                if not self.enqueue(capture):
                    capture["summary"].update(
                        status="NOT_RETAINED", reason=self.paused, pre_seconds=0, post_seconds=0
                    )
                    self.store.depth_capture(identity, capture["summary"])
        if not self.config.enabled or not ready or self.paused:
            if not ready:
                if self.routes is not None:
                    self.route_retry = 0
                self.routes = None
            for market in list(self.feeds):
                await self.release(market, self.paused or "DISCONNECTED_OR_DISABLED")
            if not ready:
                for market in MARKETS:
                    self.states[market] = {"status": "UNAVAILABLE", "reason": "BROKER_RECONCILING"}
            return
        if self.routes is None:
            if time.monotonic() < self.route_retry:
                return
            self.route_retry = time.monotonic() + self.data.config.rejection_backoff_seconds
            try:
                with priority(OPTIONAL, asyncio.get_running_loop().time() + 5):
                    async with asyncio.timeout(5):
                        self.routes = await self.data.ib.reqMktDepthExchangesAsync()
            except Exception as exc:
                for m in MARKETS:
                    self.states[m] = {"status": "UNAVAILABLE", "reason": str(exc)}
                return
        eligible = {}
        for market in MARKETS:
            detail = candidates.get(market)
            if detail is None:
                self.states[market] = {
                    "status": "MARKET_CLOSED"
                    if market in (closed_markets or set())
                    else "UNAVAILABLE",
                    "reason": "MARKET_CALENDAR_CLOSED"
                    if market in (closed_markets or set())
                    else "OPEN_QUALIFIED_FUTURE_UNAVAILABLE",
                }
                continue
            if self.retry_at.get(market, 0) > time.monotonic():
                continue
            c = detail.contract
            routing = (
                c.secType == "FUT"
                and c.conId > 0
                and c.exchange != "SMART"
                and c.exchange in detail.validExchanges.split(",")
                and any(
                    r.exchange == c.exchange and r.secType == "FUT" and r.serviceDataType == "Deep"
                    for r in self.routes
                )
            )
            if routing:
                eligible[market] = detail
            else:
                self.states[market] = {
                    "status": "UNAVAILABLE",
                    "reason": "DEPTH_EXCHANGE_ROUTING_UNVERIFIED",
                }

        def rank(market: str) -> tuple[Any, ...]:
            active = any(c["market"] == market for c in self.captures.values())
            upcoming = 0 <= (next_times[market] - at).total_seconds() <= self.config.pre_seconds
            return (
                0 if active else 1 if upcoming else 2,
                0 if active and market in self.feeds else 1,
                self.last_capture[market] if upcoming else self.last_allocation[market],
                MARKETS.index(market),
            )

        wanted = sorted(eligible, key=rank)[: self.data.config.depth_slots]
        # Ordinary rotation honours dwell. A higher priority clock/capture may preempt it.
        for market in list(self.feeds):
            sub = self.feeds[market]
            invalid = (
                market not in eligible
                or sub.state in {"FAILED", "DISCONNECTED", "CANCELLING"}
                or self.data.by_request.get(sub.request_id) is not sub
                or sub.contract.conId != eligible[market].contract.conId
            )
            if (
                market not in wanted
                and not invalid
                and (at - self.allocated_at[market]).total_seconds() < self.config.dwell_seconds
            ):
                lower = next(
                    (
                        m
                        for m in reversed(wanted)
                        if m not in self.feeds and rank(m)[0] >= rank(market)[0]
                    ),
                    None,
                )
                if lower:
                    wanted.remove(lower)
                    wanted.append(market)
            if invalid or market not in wanted:
                await self.release(market, "INVALID_OR_PREEMPTED")
        for market in wanted:
            if market in self.feeds or self.paused:
                continue
            try:
                # Generation is reserved synchronously by acquire before any callback can arrive.
                book = Book(market, eligible[market].contract, 0, self.config)
                self.books[market] = book
                sub = await self.data.acquire(
                    eligible[market].contract,
                    "DEPTH",
                    f"depth:{market}",
                    "depth",
                    {"levels": self.config.levels},
                    OPTIONAL,
                    sink=self.sink(market),
                )
                book.generation = sub.generation
                self.feeds[market] = sub
                self.allocated_at[market] = self.last_allocation[market] = at
                self.states[market] = {"status": "SUBSCRIBING", "reason": ""}
                self.log("ALLOCATE", market)
                for capture in self.captures.values():
                    if capture["market"] == market:
                        capture["summary"]["allocated"] = True
                        capture["contract"] = eligible[market].contract.dict()
                        capture["summary"]["reason"] = "MISSING_PRE_TRIGGER_CONTEXT"
                        self.last_capture[market] = datetime.fromisoformat(capture["signal_at"])
            except Exception as exc:
                self.books.pop(market, None)
                self.states[market] = {"status": "UNAVAILABLE", "reason": str(exc)}
                self.retry_at[market] = (
                    time.monotonic() + self.data.config.rejection_backoff_seconds
                )
                self.log("CAPACITY_DENIAL", market, str(exc))
        for market in eligible:
            if (
                market not in self.feeds
                and self.states.get(market, {}).get("status") != "UNAVAILABLE"
            ):
                self.states[market] = {
                    "status": "WAITING_FOR_SLOT",
                    "reason": "L2_NOT_CAPTURED_CAPACITY",
                }

    def sink(self, market: str) -> Any:
        def receive(raw: dict[str, Any]) -> None:
            self.receive(market, raw["generation"], raw)

        return receive

    async def release(self, market: str, reason: str) -> None:
        sub = self.feeds.get(market)
        if sub is None:
            return
        book = self.books.get(market)
        if book:
            self.receive(
                market,
                book.generation,
                {"kind": "GAP", "reason": reason, "generation": book.generation},
            )
        self.log("RELEASE" if reason == "MARKET_CLOSED" else "PREEMPT_OR_RELEASE", market, reason)
        await self.data.release(sub, f"depth:{market}")
        self.feeds.pop(market, None)
        self.books.pop(market, None)

    def view(self, market: str) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "DISABLED", "research_only": True}
        result = dict(self.states.get(market, {"status": "WAITING_FOR_SLOT"}))
        book, feed = self.books.get(market), self.feeds.get(market)
        if self.paused:
            result = {"status": "UNAVAILABLE", "reason": self.paused}
        elif book and feed and feed.state in {"REQUESTED", "ACTIVE"}:
            result.update(book.view(stamp()))
            if book.last_receipt is None:
                result["status"] = "SUBSCRIBING"
        result["research_only"] = True
        result["target_pre_seconds"] = self.config.pre_seconds
        active = next((c for c in self.captures.values() if c["market"] == market), None)
        if active:
            result.update(
                coverage(
                    active["events"],
                    datetime.fromisoformat(active["signal_at"]),
                    stamp(),
                    self.config.pre_seconds,
                    self.config.stale_seconds,
                )
            )
            result["opportunity_id"] = active["id"]
            result["coverage_for"] = "OPPORTUNITY"
        elif market in self.latest_capture:
            c = self.latest_capture[market]
            result.update(
                {
                    k: c.get(k, 0)
                    for k in ("pre_seconds", "post_seconds", "complete_book_pre_seconds")
                }
            )
            result["opportunity_id"] = c["id"]
            result["coverage_for"] = "LAST_OPPORTUNITY"
        else:
            spans = (
                coverage(
                    list(book.events),
                    stamp(),
                    stamp(),
                    self.config.pre_seconds,
                    self.config.stale_seconds,
                )
                if book
                else {}
            )
            result["pre_seconds"] = spans.get("pre_seconds", 0)
            result["complete_book_pre_seconds"] = spans.get("complete_book_pre_seconds", 0)
            result["coverage_for"] = "ROLLING_CONTEXT"
        return result

    def status(self) -> dict[str, Any]:
        return {
            "policy": POLICY,
            "enabled": self.config.enabled,
            "paused_reason": self.paused,
            "assigned_markets": list(self.feeds),
            "memory_bytes": self.memory_used(),
            "memory_limit": self.config.memory_bytes,
            "memory_high_water": self.high_water,
            "disk_bytes": self.disk_used,
            "disk_limit": self.config.disk_bytes,
            "writer_queue": self.queue.qsize(),
            "recording_gaps": self.gaps,
            "late_callbacks_ignored": self.late_callbacks,
        }

    def write(self, item: dict[str, Any]) -> tuple[int, str]:
        content = {k: v for k, v in item.items() if k != "memory_bytes"}
        if item["kind"] == "CAPTURE":
            content["pre_trigger_events"] = [
                e for e in item["events"] if e["at"] < item["signal_at"]
            ]
            content["post_trigger_events"] = [
                e for e in item["events"] if e["at"] >= item["signal_at"]
            ]
            del content["events"]
        text = json.dumps(content, separators=(",", ":"), allow_nan=False)
        blob = gzip.compress((text + ("\n" if item["kind"] == "POLICY" else "")).encode())
        if self.disk_used + len(blob) > self.config.disk_bytes:
            raise ValueError("RECORDING_DISK_LIMIT")
        self.directory.mkdir(parents=True, exist_ok=True)
        if item["kind"] == "CAPTURE":
            name = hashlib.sha256(item["id"].encode()).hexdigest() + ".json.gz"
            path = self.directory / name
            with path.open("xb") as out:
                out.write(blob)
                out.flush()
                os.fsync(out.fileno())
            return len(blob), name
        with (self.directory / "allocation.jsonl.gz").open("ab") as out:
            out.write(blob)
            out.flush()
            os.fsync(out.fileno())
        return len(blob), "allocation.jsonl.gz"

    async def write_loop(self) -> None:
        while not self.closed or not self.queue.empty():
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            try:
                size, filename = await asyncio.to_thread(self.write, item)
                self.disk_used += size
                if item["kind"] == "CAPTURE":
                    item["summary"]["artifact"] = filename
                    self.latest_capture[item["market"]] = {"id": item["id"], **item["summary"]}
                    try:
                        self.store.depth_capture(item["id"], item["summary"])
                    except Exception:
                        self.paused = "OPTIONAL_CAPTURE_METADATA_UNWRITABLE"
            except Exception as exc:
                self.paused = f"OPTIONAL_RECORDING_FAILED:{exc}"
                self.gaps += 1
                if item["kind"] == "CAPTURE":
                    item["summary"].update(
                        status="NOT_RETAINED", reason=self.paused, pre_seconds=0, post_seconds=0
                    )
                    self.latest_capture[item["market"]] = {"id": item["id"], **item["summary"]}
                    try:
                        self.store.depth_capture(item["id"], item["summary"])
                    except Exception:
                        self.paused = "OPTIONAL_CAPTURE_METADATA_UNWRITABLE"
            finally:
                self.queued_bytes -= item["memory_bytes"]
                self.queue.task_done()

    async def close(self) -> None:
        await self.tick({}, {}, False)
        self.closed = True
        if self.worker:
            await self.worker
