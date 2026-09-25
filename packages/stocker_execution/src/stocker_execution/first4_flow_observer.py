"""Optional FIRST4 observer lifecycle. Never arms, reconnects, submits or gates orders."""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ib_async import Contract

from stocker_execution.first4_flow import VERSION, FlowEvent, empty_totals, metrics
from stocker_execution.first4_flow_store import CaptureStart, FlowWriter, read_flow

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


@dataclass
class Capture:
    observer: "FlowObserver"
    event: dict[str, Any]
    contract: Any
    requested_at: str
    state: str = "STARTING"
    reason: str = "WAITING_FOR_EXECUTION_FIRST_AND_PACING"
    capture_id: str = ""
    generation: int = -1
    sequence: int = 0
    segments: int = 0
    trade_request: int | None = None
    quote_request: int | None = None
    segment_requested_at: str | None = None
    first_received_at: str | None = None
    last_trade_at: str | None = None
    last_quote_at: str | None = None
    ended_at: str | None = None
    terminal: bool = False
    pending_stop: bool = False
    stale: bool = False
    gap_count: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, request: int, kind: str, **values: Any) -> None:
        """Callback: immutable copy + bounded enqueue only; exceptions stay optional."""
        try:
            observer = self.observer
            if self.ended_at:
                return
            if kind not in {"gap", "end"} and (
                self.pending_stop
                or self.ended_at
                or not self.capture_id
                or self.generation != observer.broker.data_generation
                or not observer.available()
            ):
                return
            received = utc_now()
            if kind not in {"gap", "end"} and received >= datetime.fromisoformat(
                self.event["close_at"]
            ):
                return
            if kind == "gap":
                if self.gap_count >= 63:
                    kind, values = "end", {"reason": "CAPTURE_GAP_LIMIT"}
                    self.state, self.reason = "PARTIAL_COVERAGE", "CAPTURE_GAP_LIMIT"
                    self.pending_stop = self.terminal = True
                else:
                    self.gap_count += 1
            self.sequence += 1
            event = FlowEvent(
                self.capture_id,
                self.event["session"],
                self.event["con_id"],
                request,
                self.generation,
                self.sequence,
                received.isoformat(),
                time.monotonic_ns(),
                kind,
                observer.config.feed_mode,
                **values,
            )
            if not observer.writer or not observer.writer.put(event):
                self.state, self.reason = "STORAGE_ERROR", "CAPTURE_QUEUE_OR_WRITER_UNAVAILABLE"
                self.pending_stop = self.terminal = True
                return
            if kind == "end":
                self.ended_at = event.received_at
            if kind in {"quote", "trade"}:
                self.first_received_at = self.first_received_at or event.received_at
                if kind == "trade":
                    self.last_trade_at = event.received_at
                else:
                    self.last_quote_at = event.received_at
        except Exception as exc:
            self.state, self.reason = "PARTIAL_COVERAGE", "CALLBACK_ERROR:" + str(exc)
            self.pending_stop = self.terminal = True


class FlowObserver:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.config = runtime.config.order_flow
        self.broker = runtime.broker
        self.writer: FlowWriter | None = None
        self.captures: dict[int, Capture] = {}
        self.session: str | None = None
        self.source_commit = "UNAVAILABLE"
        self.problem = ""
        self.running = False

    def available(self) -> bool:
        return bool(
            self.broker.ib.isConnected()
            and self.broker.upstream_available
            and not self.broker.market_data_block
            and not self.broker.data_problem
        )

    def allocate(self, event: dict[str, Any], contract: Any) -> None:
        if not self.config.enabled or event["con_id"] in self.captures:
            return
        capture = Capture(self, dict(event), contract, utc_now().isoformat())
        if event["slot"] > self.config.stock_capacity():
            capture.state, capture.reason = "CAPACITY_LIMITED", "EXECUTION_RESERVED_SLOT_BUDGET"
            capture.terminal = True
        self.captures[event["con_id"]] = capture

    def error(self, request: int, code: int, message: str, contract: Any = None) -> None:
        # This handler never consumes/clears PaperBroker's shared safety errors.
        if code in {2104, 2106, 2107, 2108, 2158}:
            return
        for capture in self.captures.values():
            sink = self.broker.ib.flow_wire.sinks.get(request)
            if sink != capture.emit:
                continue
            capture.errors.append({"request_id": request, "code": code, "message": message[:500]})
            capture.errors[:] = capture.errors[-8:]
            capture.state = (
                "ENTITLEMENT_MISSING"
                if code in {354, 10089, 10090, 10167, 10186, 10189}
                else "CAPACITY_LIMITED"
                if code in {100, 101, 10190}
                else "PARTIAL_COVERAGE"
            )
            capture.reason = f"REQUEST_REJECTED:{code}:{message[:300]}"
            capture.emit(request, "gap", reason=capture.reason)
            capture.pending_stop = capture.terminal = True

    async def start_capture(self, capture: Capture, *, subscribe: bool = True) -> None:
        if capture.segments >= self.config.max_segments_per_stock:
            capture.state, capture.reason = "CAPACITY_LIMITED", "SEGMENT_LIMIT"
            capture.terminal = True
            return
        capture.capture_id = uuid.uuid4().hex
        capture.generation = self.broker.data_generation
        capture.sequence = 0
        capture.segments += 1
        capture.ended_at = capture.first_received_at = None
        capture.last_trade_at = capture.last_quote_at = None
        capture.pending_stop = False
        capture.stale = False
        capture.gap_count = 0
        capture.segment_requested_at = utc_now().isoformat()
        capture.state, capture.reason = "STARTING", "WAITING_FOR_TRADE_AND_QUOTE"
        metadata = {
            "capture_id": capture.capture_id,
            "session": capture.event["session"],
            "con_id": capture.event["con_id"],
            "symbol": capture.event["symbol"],
            "slot": capture.event["slot"],
            "allocation_at": capture.event["information_at"],
            "entry_anchor_at": capture.event["entry_at"],
            "close_at": capture.event["close_at"],
            "allocation_id": f"F4:{capture.event['session']}:{capture.event['slot']}",
            "requested_at": capture.requested_at,
            "segment_requested_at": capture.segment_requested_at,
            "generation": capture.generation,
            "feed_mode": self.config.feed_mode,
            "classification_version": VERSION,
            "source_commit": self.source_commit,
            "config": self.config.model_dump(mode="json"),
            "order_authoritative": False,
            "may_submit_orders": False,
            "continuity": "NEW_SEGMENT_NO_BACKFILL_RECONNECT_DUPLICATES_UNRESOLVED",
            "volume_unit": "shares (requires Gateway shares mode)",
        }
        assert self.writer is not None
        if not self.writer.put(CaptureStart(capture.capture_id, json.dumps(metadata))):
            raise OSError(self.writer.error)
        # Reuse an existing execution Last stream, or respect the startup/request quarantine.
        if subscribe:
            capture.trade_request = self.broker.ib.flow_wire.retain_last(
                capture.contract, capture.emit
            )

    def finish(self, capture: Capture, reason: str) -> None:
        if capture.capture_id and not capture.ended_at:
            capture.emit(-1, "end", reason=reason)
            capture.ended_at = utc_now().isoformat()
        wire = self.broker.ib.flow_wire
        try:
            if capture.quote_request is not None:
                wire.release_quote(capture.quote_request)
        finally:
            capture.quote_request = None
            wire.release_last(capture.event["con_id"])
            capture.trade_request = None

    async def step(self) -> None:
        current = utc_now()
        day = self.runtime.session
        if day != self.session:
            for capture in self.captures.values():
                self.finish(capture, "SESSION_RESET")
            self.captures.clear()
            self.session = day
            if day:
                # Four permanent slots, also for failed entries/restarts. Never scan the universe.
                for row in self.runtime.store.db.execute(
                    "SELECT * FROM first4_events WHERE session=? AND slot IS NOT NULL "
                    "ORDER BY slot LIMIT 4",
                    (day,),
                ):
                    event = dict(row)
                    self.allocate(
                        event,
                        Contract(
                            conId=event["con_id"],
                            symbol=event["symbol"],
                            secType="STK",
                            exchange="SMART",
                            currency="USD",
                        ),
                    )
                    prior = await asyncio.to_thread(
                        read_flow, self.config.raw_path.resolve(), day, event["con_id"]
                    )
                    self.captures[event["con_id"]].segments = len(prior["captures"])
        for capture in sorted(self.captures.values(), key=lambda c: c.event["slot"]):
            try:
                await self.step_capture(capture, current)
            except Exception as exc:
                capture.state, capture.reason = (
                    "PARTIAL_COVERAGE",
                    "OBSERVER_REQUEST_FAILED:" + str(exc),
                )
                capture.terminal = True
                try:
                    self.finish(capture, capture.reason)
                except Exception:
                    log.exception("Optional capture cleanup failed for %s", capture.event["symbol"])
                log.warning("Optional capture failed for %s: %s", capture.event["symbol"], exc)

    async def step_capture(self, capture: Capture, current: datetime) -> None:
        if capture.terminal or capture.pending_stop:
            if not capture.capture_id and self.writer and not self.writer.error:
                state, reason = capture.state, capture.reason
                await self.start_capture(capture, subscribe=False)
                capture.state, capture.reason = state, reason
            self.finish(capture, capture.reason)
            return
        if self.writer and self.writer.error:
            capture.state, capture.reason = "STORAGE_ERROR", self.writer.error
            capture.terminal = True
            self.finish(capture, capture.reason)
            return
        if current >= datetime.fromisoformat(capture.event["close_at"]):
            capture.state, capture.reason = "PARTIAL_COVERAGE", "REGULAR_SESSION_ENDED"
            capture.terminal = True
            self.finish(capture, capture.reason)
            return
        if not self.available() or (
            capture.trade_request is not None and capture.generation != self.broker.data_generation
        ):
            capture.state, capture.reason = "STALE_OR_DISCONNECTED", "SHARED_DATA_INTERRUPTION"
            self.finish(capture, capture.reason)
            return
        wire = self.broker.ib.flow_wire
        if capture.trade_request is None:
            if wire.valid_last(capture.event["con_id"]) is None and wire.remaining_pacing(
                capture.event["con_id"]
            ):
                return
            await self.start_capture(capture)
            if capture.trade_request is None:
                return
        if capture.quote_request is None and capture.trade_request is not None:
            if self.config.feed_mode == "TBT_TRADES_TBT_QUOTES" and wire.remaining_pacing(
                capture.event["con_id"]
            ):
                capture.state, capture.reason = "PARTIAL_COVERAGE", "QUOTE_REQUEST_PACING_WAIT"
                return
            capture.quote_request = wire.quote_request(
                capture.contract, self.config.feed_mode, capture.emit
            )
        if (
            not capture.first_received_at
            and capture.segment_requested_at
            and (current - datetime.fromisoformat(capture.segment_requested_at)).total_seconds()
            > self.config.stale_seconds
        ):
            capture.state, capture.reason = "STALE_OR_DISCONNECTED", "NO_OBSERVED_EVENTS"
        if capture.first_received_at:
            stamps = [capture.last_trade_at, capture.last_quote_at]
            fresh = all(
                s
                and 0
                <= (current - datetime.fromisoformat(s)).total_seconds()
                <= self.config.stale_seconds
                for s in stamps
            )
            if not fresh and not capture.stale:
                capture.emit(-1, "gap", reason="FRESHNESS_UNVERIFIED")
            if fresh and capture.stale:
                capture.emit(-1, "resume", reason="FRESH_TRADE_AND_QUOTE_OBSERVED")
            if capture.terminal or capture.pending_stop:
                return
            capture.stale = not fresh
            capture.state = "COLLECTING_ESTIMATED_FLOW" if fresh else "STALE_OR_DISCONNECTED"
            capture.reason = "" if fresh else "TRADE_OR_QUOTE_FRESHNESS_UNVERIFIED"

    async def run(self) -> None:
        if not self.config.enabled:
            return
        self.running = True
        self.writer = FlowWriter(self.config)
        self.writer.start()
        self.broker.ib.errorEvent += self.error
        try:
            self.source_commit = await asyncio.to_thread(source_revision)
            await asyncio.to_thread(self.writer.ready.wait, 3)
            if self.writer.error:
                raise OSError(self.writer.error)
            while self.runtime.running:
                await self.step()
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.problem = "OBSERVER_STOPPED: " + str(exc)
            log.exception("Optional FIRST4 observer stopped; position management is independent")
        finally:
            self.running = False
            self.broker.ib.errorEvent -= self.error
            for capture in self.captures.values():
                try:
                    self.finish(capture, self.problem or "OBSERVER_STOPPED")
                except Exception:
                    log.exception("Observer subscription cleanup failed")
            await asyncio.to_thread(self.writer.close)

    async def view(self, session: str, con_id: int, *, detail: bool = False) -> dict[str, Any]:
        base: dict[str, Any] = {
            "state": "DISABLED" if not self.config.enabled else "STARTING",
            "feed_mode": "UNAVAILABLE",
            "order_authoritative": False,
            "may_submit_orders": False,
        }
        try:
            evidence = await asyncio.to_thread(
                read_flow, self.config.raw_path.resolve(), session, con_id
            )
        except Exception as exc:
            return {**base, "state": "STORAGE_ERROR", "reason": str(exc)}
        captures = evidence["captures"]
        capture = self.captures.get(con_id) if session == self.session else None
        latest = captures[-1] if captures else {}
        if capture and capture.capture_id:
            latest = next((row for row in captures if row["capture_id"] == capture.capture_id), {})
        stamp = utc_now()
        if latest:
            base.update(latest)
            base["state"] = "PARTIAL_COVERAGE"
            base["reason"] = latest.get("end_reason") or "INTERRUPTED_OR_NOT_OWNED_BY_THIS_PROCESS"
        if capture:
            base.update(
                state=capture.state,
                reason=capture.reason,
                errors=capture.errors,
                requested_at=capture.requested_at,
                capture_id=capture.capture_id,
                first_received_at=capture.first_received_at,
                last_trade_at=capture.last_trade_at,
                last_quote_at=capture.last_quote_at,
                ended_at=capture.ended_at,
                feed_mode=self.config.feed_mode
                if capture.trade_request is not None or latest
                else "UNAVAILABLE",
            )
        if self.problem or (self.writer and self.writer.error):
            base.update(
                state="STORAGE_ERROR" if self.writer and self.writer.error else "PARTIAL_COVERAGE",
                reason=self.problem or (self.writer.error if self.writer else ""),
            )
        if not self.config.enabled:
            base["state"] = "DISABLED"
        last_quote, last_trade = base.get("last_quote_at"), base.get("last_trade_at")
        base["quote_age_ms"] = (
            (stamp - datetime.fromisoformat(last_quote)).total_seconds() * 1000
            if last_quote
            else None
        )
        base["trade_age_ms"] = (
            (stamp - datetime.fromisoformat(last_trade)).total_seconds() * 1000
            if last_trade
            else None
        )
        if base["state"] == "COLLECTING_ESTIMATED_FLOW" and any(
            age is None or age < 0 or age > self.config.stale_seconds * 1000
            for age in (base["quote_age_ms"], base["trade_age_ms"])
        ):
            base.update(state="STALE_OR_DISCONNECTED", reason="FRESHNESS_EXPIRED")
        if self.writer:
            base["dropped_events"] = self.writer.dropped
            base["uncommitted_events"] = self.writer.uncommitted_events
            base["queue_high_water"] = self.writer.high_water
        first = base.get("first_received_at")
        coverage_end = base.get("ended_at") or (
            stamp.isoformat() if capture and not capture.ended_at else base.get("last_received_at")
        )
        duration = (
            max(
                0,
                (
                    datetime.fromisoformat(coverage_end) - datetime.fromisoformat(first)
                ).total_seconds(),
            )
            if first and coverage_end
            else 0
        )
        for gap in latest.get("gaps", []):
            if first and coverage_end:
                gap_start = max(datetime.fromisoformat(first), datetime.fromisoformat(gap["at"]))
                gap_end = min(
                    datetime.fromisoformat(coverage_end),
                    datetime.fromisoformat(gap.get("end_at") or coverage_end),
                )
                duration -= max(0, (gap_end - gap_start).total_seconds())
        base["coverage_duration_seconds"] = max(0, duration)
        base["pre_capture_gap_seconds"] = (
            max(
                0,
                (
                    datetime.fromisoformat(first) - datetime.fromisoformat(base["requested_at"])
                ).total_seconds(),
            )
            if first
            else None
        )
        base["coverage_warning"] = "Capture starts after allocation; gaps are not zero activity."
        gap_intervals = []
        for row in captures:
            segment_end = row.get("ended_at") or (
                stamp.isoformat()
                if capture and row["capture_id"] == capture.capture_id
                else row.get("last_received_at")
            )
            if segment_end:
                for gap in row.get("gaps", []):
                    end = min(
                        datetime.fromisoformat(segment_end),
                        datetime.fromisoformat(gap.get("end_at") or segment_end),
                    )
                    gap_intervals.append((datetime.fromisoformat(gap["at"]), end))
        cutoff = stamp.replace(second=0, microsecond=0)
        # Rolling windows advance on completed receipt-minute boundaries.
        for name, seconds in (("rolling_1m", 60), ("rolling_5m", 300)):
            # Completed minute bins keep dashboard reads bounded and replayable.
            selected = [
                b
                for b in evidence["bars"]
                if cutoff - timedelta(seconds=seconds)
                <= datetime.fromisoformat(b["minute"])
                < cutoff
            ]
            totals = empty_totals()
            for bar in selected:
                for key in totals:
                    totals[key] += bar[key]
            base[name] = metrics(totals) if selected else None
            if base[name] is not None:
                base[name]["partial_coverage"] = (
                    len({b["minute"] for b in selected}) < seconds // 60
                    or any(b.get("has_gap") for b in selected)
                    or len({b["capture_id"] for b in selected}) > 1
                    or not first
                    or datetime.fromisoformat(first) > cutoff - timedelta(seconds=seconds)
                    or not coverage_end
                    or datetime.fromisoformat(coverage_end) < cutoff
                    or any(
                        start < cutoff and end > cutoff - timedelta(seconds=seconds)
                        for start, end in gap_intervals
                        if end > start
                    )
                )
        base["last_completed_minute"] = base["rolling_1m"]
        base["observation_state"] = (
            "OBSERVED_PRINTS"
            if latest.get("totals", {}).get("trade_count", 0)
            else "NO_OBSERVED_PRINTS"
        )
        if detail:
            base.update(captures=captures, bars=evidence["bars"])
        return base
