"""15-minute RAM windows and bounded, incremental event evidence off the trading loop.

Raw means delivered Saxo messages, not exchange ticks. Each retained window begins with
the checkpoint immediately preceding its first delta. Disk work runs in one worker thread.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import time
import zlib
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from stocker_execution import book_flow
from stocker_execution.config import ROLLING_WINDOW_SECONDS, SUBSCRIPTION_LIMIT, RecorderConfig
from stocker_execution.contracts import utc
from stocker_execution.saxo_auth import atomic_json
from stocker_execution.saxo_stream import merge, merge_board

log = logging.getLogger(__name__)
CHAIN = "OptionsChain"  # a market's observation-only chain window, recorded like an instrument
# Rows gather this long before each gzip member: at 0.1 s a member held 1-3 rows and cost about
# three times the disk of a one-second batch (review 2026-10-04). A crash loses at most this much,
# and recover() already marks such a capture INTERRUPTED_RESTART. Bursts still drain at once.
BATCH_SECONDS = 1.0
# One queued write: segment, the capture manifest when it changed (None for rows only),
# zlib-packed rows and the bytes charged against the queue budget.
Item = tuple[str, dict[str, Any] | None, list[bytes], int]


def packed(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def read_row(blob: bytes) -> dict[str, Any]:
    """Lossless rolling-memory representation; disk archives remain ordinary JSONL."""
    return cast(dict[str, Any], json.loads(zlib.decompress(blob)))


def apply(state: Any, record: dict[str, Any]) -> Any:
    # Chain boards key Expiries/Strikes by Index; prices replace whole arrays.
    combine = merge_board if (record.get("identity") or {}).get("asset_type") == CHAIN else merge
    if record["kind"] == "SNAPSHOT":
        return combine({}, record["payload"])
    if record["kind"] == "GAP":
        return None
    if record["kind"] == "UPDATE" and not record.get("dropped") and state is not None:
        return combine(state, record["payload"])
    return state


@dataclass
class Window:
    identity: dict[str, Any]
    rows: deque[tuple[float, bytes, int]] = field(default_factory=deque)  # receipt, row, sequence
    checkpoint: Any = None
    checkpoint_at: float | None = None
    current: Any = None
    continuous_since: float | None = None
    row_bytes: int = 0
    sequence: int = 0
    flow: dict[str, Any] | None = None
    checkpoint_flow: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    checkpoint_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_version: str | None = None
    flow_history: deque[tuple[int, dict[str, Any], int]] = field(default_factory=deque)
    flow_history_bytes: int = 0

    @property
    def size(self) -> int:
        # Retained rows and the bounded flow history; every other field is derived from them
        # or capped by count (16 reference versions of at most 64 KiB, 512 flow points).
        return self.row_bytes + self.flow_history_bytes

    def remember_flow(self) -> None:
        assert self.flow is not None
        # Only inputs used by temporal calculations; full features stay in compressed evidence.
        point = {
            k: self.flow[k]
            for k in (
                "at",
                "status",
                "delay_minutes",
                "generation",
                "valid_until",
                "depth",
                "basis",
                "volume",
            )
        }
        size = len(packed(point)) * 8 + 128
        self.flow_history.append((self.sequence, point, size))
        self.flow_history_bytes += size
        while len(self.flow_history) > 512 or (
            len(self.flow_history) > 1 and self.flow_history[1][1]["at"] <= point["at"] - 60
        ):
            self.flow_history_bytes -= self.flow_history.popleft()[2]

    def evict(self) -> None:
        at, blob, sequence = self.rows.popleft()
        self.row_bytes -= len(blob) + 128  # include bounded Python object overhead
        record = read_row(blob)
        self.checkpoint = apply(self.checkpoint, record)
        self.checkpoint_flow = record.get("book_flow")
        self.checkpoint_at = at
        if record.get("observation_context"):
            self.checkpoint_context[record.get("context_source", "REGULAR_PRICE")] = record[
                "observation_context"
            ]
        if record["kind"] == "GAP":
            self.checkpoint_context.clear()
        while self.flow_history and self.flow_history[0][0] <= sequence:
            self.flow_history_bytes -= self.flow_history.popleft()[2]

    def coverage(self, at: float) -> float:
        if self.continuous_since is None or self.current is None or not self.rows:
            return 0
        beginning = max(at - ROLLING_WINDOW_SECONDS, self.continuous_since, self.rows[0][0])
        if self.checkpoint is not None and self.checkpoint_at is not None:
            beginning = max(at - ROLLING_WINDOW_SECONDS, self.continuous_since, self.checkpoint_at)
        return max(0, at - beginning)

    def prefix(self, after_sequence: int | None = None) -> list[bytes]:
        """Rows to archive, decoded only once selected; this runs on the decision path."""
        # Reused candidates can have an unrecorded interval. Supply a checkpoint
        # when the intervening raw rows have already left the rolling window.
        if after_sequence is not None and (not self.rows or self.rows[0][2] <= after_sequence + 1):
            return [zlib.decompress(b) for _, b, s in self.rows if s > after_sequence]
        return [
            packed(
                {
                    "kind": "CHECKPOINT",
                    "identity": self.identity,
                    "receipt": self.checkpoint_at,
                    "payload": self.checkpoint,
                    "book_flow": self.checkpoint_flow,
                    "observation_context": self.checkpoint_context,
                    "reconstruction": "state immediately before retained messages",
                }
            ),
            *(zlib.decompress(b) for _, b, _ in self.rows),
        ]


class Recorder:
    def __init__(self, config: RecorderConfig, directory: Path):
        self.config, self.directory = config, directory
        self.windows: dict[str, Window] = {}
        self.active: dict[str, dict[str, Any]] = {}
        self.event_ids: set[str] = set()
        self.queue: asyncio.Queue[Item] = asyncio.Queue(config.queue_max_items)
        self.manifests: dict[str, dict[str, Any]] = {}  # writer thread only: last per segment
        self.markers_pending: set[str] = set()  # captures ended by queue pressure, unmarked
        self.write_ready = asyncio.Event()
        self.queued_bytes = 0
        self.high_water = 0
        self.disk_bytes = 0
        self.disk_free: int | None = None
        self.problem = "" if config.persistent_capture else "RECORDING_PERMISSION_NOT_VERIFIED"
        self.gaps = 0
        self.closed = False
        self.worker: asyncio.Task[None] | None = None
        self.catalog: list[dict[str, Any]] = []
        # (time, disk_bytes) about once a minute over the last day: the System page shows the
        # archive's growth and the days left before archive_max_bytes stops captures.
        self.growth: deque[tuple[float, int]] = deque(maxlen=1441)

    async def start(self) -> None:
        await asyncio.to_thread(self.recover)
        self.worker = asyncio.create_task(self.write_loop())

    def recover(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Only own atomic-write leftovers; never touch the execution ledger.
        for path in self.directory.glob("*.tmp-*"):
            path.unlink()
        self.disk_free = shutil.disk_usage(self.directory).free
        for path in sorted(self.directory.glob("*.manifest.json")):
            manifest = json.loads(path.read_text())
            interrupted = manifest.get("state") == "CAPTURING"
            # A write that failed mid-append (ENOSPC, say) can also leave a partial member; its
            # marker says RECORDER_IO_FAILED. Either is repaired once, then recorded as such.
            if interrupted or (
                manifest.get("reason") == "RECORDER_IO_FAILED" and "recovered_bytes" not in manifest
            ):
                if interrupted:
                    manifest.update(state="INCOMPLETE", reason="INTERRUPTED_RESTART")
                data = self.directory / (manifest["segment"] + ".jsonl.gz")
                # Keep complete gzip members. Only truncate an interrupted, uncommitted tail.
                manifest["recovered_bytes"] = 0
                if data.exists():
                    self.recover_members(data)
                    manifest["recovered_bytes"] = data.stat().st_size
                atomic_json(path, manifest)
            self.catalog.append(
                {
                    "segment": manifest["segment"],
                    "state": manifest["state"],
                    "start": manifest["start"],
                    "end": manifest["end"],
                    "key": manifest["key"],
                    "protected": manifest.get("protected", False),
                }
            )
        # Counted after the repairs and manifest rewrites above, so the figure is what is on disk.
        self.disk_bytes = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        if len(self.catalog) >= 10000:
            self.problem = "STORAGE_LIMIT_REACHED"

    @staticmethod
    def recover_members(path: Path) -> None:
        # Streaming scan with bounded buffers; no full-archive read on restart.
        good, read_at = 0, 0
        decoder = zlib.decompressobj(31)
        with path.open("rb") as src:
            while blob := src.read(65536):
                read_at += len(blob)
                try:
                    while blob:
                        decoder.decompress(blob, 1024**2)
                        # End of member first: when a member ends in an output-limited tail, CPython
                        # leaves the leftover in unconsumed_tail as well as unused_data, and feeding
                        # the stale tail back looped forever (2026-10-01 restart).
                        if decoder.eof:
                            blob = decoder.unused_data
                            good = read_at - len(blob)
                            decoder = zlib.decompressobj(31)
                        elif decoder.unconsumed_tail:
                            blob = decoder.unconsumed_tail
                        else:
                            break
                except zlib.error:
                    break
        if good != path.stat().st_size:
            with path.open("r+b") as out:
                out.truncate(good)
                out.flush()
                os.fsync(out.fileno())

    def register(self, key: str, identity: dict[str, Any]) -> None:
        # Session schedules and permissions are mutable observations, not contract identity.
        identity = {
            k: v
            for k, v in identity.items()
            if k
            in {
                "provider",
                "environment",
                "asset_type",
                "uic",
                "market",
                "symbol",
                "exchange",
                "contract_month",
                "underlying_uic",
                "option_root_id",
                "right",
                "strike",
                "expiry",
                "tick_size",
            }
        }
        if key not in self.windows:
            if len(self.windows) >= SUBSCRIPTION_LIMIT:
                raise ValueError("RECORDER_INSTRUMENT_LIMIT")
            # A retired candidate may return while its shared archive is still open.
            sequence = max(
                (c["last_sequences"].get(key, 0) for c in self.active.values()), default=0
            )
            self.windows[key] = Window(identity, sequence=sequence)
        elif self.windows[key].identity != identity:
            raise ValueError("RECORDING_IDENTITY_IMMUTABLE")

    def memory(self) -> int:
        return sum(w.size for w in self.windows.values())

    def metadata_version(self, key: str, raw: dict[str, Any], received_at: float) -> str:
        window = self.windows[key]
        blob = packed(raw)
        version = hashlib.sha256(blob).hexdigest()
        if version not in window.metadata:
            if len(blob) > 65536:
                raise ValueError("REFERENCE_CACHE_LIMIT")
            # Session schedules and trading status change daily, so a long-lived process sees a
            # new version per reconnect: the oldest gives way rather than blocking the market.
            while len(window.metadata) >= 16:
                window.metadata.pop(next(iter(window.metadata)))
            window.metadata[version] = {"received_at": received_at, "value": json.loads(blob)}
            for segment, capture in self.active.items():
                if key in capture["instruments"]:
                    capture.setdefault("metadata_versions", {})[version] = window.metadata[version]
                    self.enqueue(
                        segment,
                        capture,
                        [
                            packed(
                                {
                                    "kind": "METADATA",
                                    "version": version,
                                    "payload": window.metadata[version],
                                }
                            )
                        ],
                    )
        window.metadata_version = version
        return version

    def expire(self, at: float) -> None:
        for window in self.windows.values():
            while window.rows and window.rows[0][0] < at - ROLLING_WINDOW_SECONDS:
                window.evict()
        while self.memory() > self.config.rolling_max_bytes:
            populated = [w for w in self.windows.values() if w.rows]
            if not populated:
                break
            min(populated, key=lambda w: w.rows[0][0]).evict()
        self.high_water = max(self.high_water, self.memory())

    def ingest(
        self,
        key: str,
        kind: str,
        payload: Any,
        at: float,
        *,
        message_id: str | None = None,
        dropped: bool = False,
        generation: str = "",
        provider_message: Any = None,
        observation_context: dict[str, Any] | None = None,
        context_source: str = "REGULAR_PRICE",
    ) -> None:
        window = self.windows[key]
        window.sequence += 1
        record = {
            "kind": kind,
            "identity": window.identity,
            "receipt": at,
            "local_sequence": window.sequence,
            "message_id": message_id,
            "generation": generation,
            "dropped": dropped,
            "payload": payload,
            "provider_message": provider_message,
            "metadata_version": window.metadata_version,
            "observation_context": observation_context,
            "context_source": context_source,
        }
        blob = packed(record)
        if len(blob) > self.config.max_message_bytes:
            kind, payload = "GAP", {"reason": "MESSAGE_BYTE_LIMIT"}
            record.update(
                kind=kind, payload=payload, provider_message=None, observation_context=None
            )
            blob = packed(record)
        if kind == "SNAPSHOT":
            if window.current is None:
                window.continuous_since = at
        elif kind == "GAP":
            window.continuous_since = None
            window.context.clear()
            self.gaps += 1
        if observation_context and kind != "GAP":
            window.context[context_source] = observation_context
        window.current = apply(window.current, record)
        if window.identity.get("asset_type") == "ContractFutures" and window.identity.get(
            "tick_size"
        ):
            context = observation_context or {"valid_until": at + 5, "subscription_id": generation}
            # Queued deltas can predate the initial REST snapshot's arrival. Keep raw
            # receipt intact, but never backdate a calculation using that later snapshot.
            calculated_at = max(at, window.flow["at"] if window.flow else at)
            history = [point for _, point, _ in window.flow_history]
            try:
                window.flow = book_flow.observe(
                    window.identity, window.current, calculated_at, context, history
                )
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                window.flow = book_flow.observe(
                    window.identity,
                    None,
                    calculated_at,
                    {**context, "problem": "FEATURE_SCHEMA_UNAVAILABLE"},
                    [],
                )
            window.remember_flow()
            record["book_flow"] = window.flow
            blob = packed(record)
        retained = zlib.compress(blob, level=1)
        window.rows.append((at, retained, window.sequence))
        window.row_bytes += len(retained) + 128
        self.expire(at)
        for segment, capture in list(self.active.items()):
            if self.capture_requires(capture, key, at):
                capture["last_receipt"] = at
                capture["last_sequences"][key] = window.sequence
                if kind == "GAP":
                    capture["gaps"] += 1
                if window.flow:
                    capture["book_flow"] = self.flow_metadata(window, at)
                # Rows only: the manifest reaches disk with its next structural change.
                self.enqueue(segment, None, [blob])

    def enqueue(self, segment: str, manifest: dict[str, Any] | None, rows: list[bytes]) -> bool:
        if not self.config.persistent_capture or self.problem:
            return False
        # A detached JSON copy prevents later event-loop changes racing the writer.
        detached = json.loads(packed(manifest)) if manifest is not None else None
        # Prehistory may contain many repeated identities/field names. Queue one
        # lossless batch within the SAME byte/item caps, then expand in the writer.
        if rows:
            compressor = zlib.compressobj(level=1)
            rows = [b"".join(compressor.compress(row) for row in rows) + compressor.flush()]
        size = (len(packed(detached)) if detached else 0) + sum(len(b) + 64 for b in rows)
        if self.queue.full() or self.queued_bytes + size > self.config.queue_max_bytes:
            # Transient pressure ends this capture only; its marker follows from tick()
            # once the writer has drained. Nothing latches the recorder off.
            capture = self.active.get(segment)
            if capture is not None:
                capture.update(state="INCOMPLETE", reason="WRITE_QUEUE_LIMIT_REACHED")
                self.markers_pending.add(segment)
            return False
        self.queue.put_nowait((segment, detached, rows, size))
        self.queued_bytes += size
        if self.queue.qsize() >= 64:
            self.write_ready.set()
        return True

    def trigger(
        self,
        key: str,
        event: dict[str, Any],
        at: float,
        option_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        self.expire(at)
        identity = str(event["id"])
        if identity in self.event_ids:
            return {"state": "DUPLICATE_SUPPRESSED", "event_id": identity}
        # Store.observe is the durable de-duplication gate across process restarts.
        self.event_ids.add(identity)
        if len(self.event_ids) > 8192:
            self.event_ids = {e["id"] for c in self.active.values() for e in c["events"]}
            self.event_ids.add(identity)
        if not self.config.persistent_capture or self.problem:
            return {"state": "UNAVAILABLE", "reason": self.problem, "event_id": identity}
        window = self.windows[key]
        keys = [key, *(k for k in option_keys or [] if k in self.windows)]
        event_at = utc(event["signal_at"]).timestamp() if event.get("signal_at") else at
        end = at + self.config.minimum_post_event_minutes * 60
        # Same identity and overlapping pre/post ranges share one physical archive.
        capture = next(
            (
                c
                for c in self.active.values()
                if c["key"] == key
                and (c["open_trades"] or c["end"] >= at - ROLLING_WINDOW_SECONDS)
                and c["state"] != "INCOMPLETE"
            ),
            None,
        )
        if capture is None:
            if len(self.active) >= self.config.max_open_captures or len(self.catalog) >= 10000:
                return {"state": "UNAVAILABLE", "reason": "CAPTURE_COUNT_LIMIT"}
            segment = hashlib.sha256((key + identity).encode()).hexdigest()
            capture = {
                "segment": segment,
                "key": key,
                "identity": window.identity,
                "start": at - window.coverage(at),
                "end": end,
                "state": "CAPTURING",
                "instruments": [],
                "events": [],
                "gaps": 0,
                "open_trades": [],
                "protected": False,
                "reason": "",
                "last_receipt": None,
                "last_sequences": {},
                "metadata_versions": {},
                "permission_evidence": self.config.recording_permission_evidence,
                "raw_semantics": "Every delivered Saxo message; not a complete exchange tick feed",
            }
            self.active[segment] = capture
        segment = capture["segment"]
        if len(capture["events"]) >= 256 or len(packed(event)) > 16384:
            # This capture alone ends; other segments and later captures keep recording.
            capture.update(state="INCOMPLETE", reason="CAPTURE_EVENT_LIMIT")
            self.markers_pending.add(segment)
            return {"state": "INCOMPLETE", "reason": "CAPTURE_EVENT_LIMIT"}
        capture["state"] = "CAPTURING"
        capture["end"] = max(capture["end"], end)
        capture["events"].append(
            {
                **event,
                "detection_receipt": at,
                "observation_until": end,
                "instruments": list(keys),
                "prehistory_seconds": {
                    k: max(0, self.windows[k].coverage(at) - max(0, at - event_at)) for k in keys
                },
            }
        )
        if window.flow:
            capture["book_flow"] = self.flow_metadata(window, at)
        rows = []
        for instrument in keys:
            capture.setdefault("metadata_versions", {}).update(self.windows[instrument].metadata)
            if instrument not in capture["instruments"]:
                rows.extend(self.windows[instrument].prefix())
                capture["instruments"].append(instrument)
            else:
                # A completed capture can overlap a new event's prehistory. Bridge
                # the intervening window without copying already-written messages.
                rows.extend(
                    self.windows[instrument].prefix(capture["last_sequences"].get(instrument, 0))
                )
            capture["last_sequences"][instrument] = self.windows[instrument].sequence
        rows.append(packed({"kind": "TRIGGER", "payload": capture["events"][-1]}))
        self.enqueue(segment, capture, rows)
        return {
            "segment": segment,
            "state": capture["state"],
            "pre_seconds": window.coverage(at),
            "reason": capture["reason"],
        }

    @staticmethod
    def flow_metadata(window: Window, at: float) -> dict[str, Any]:
        flow = window.flow or {}
        return {
            "version": book_flow.VERSION,
            "observation_only": True,
            "coverage_seconds": window.coverage(at),
            "calculated_at": flow.get("at"),
            "quality_flags": flow.get("quality_flags", []),
            "available_fields": flow.get("available_fields", {}),
            "granted_refresh_ms": flow.get("feed", {}).get("granted_refresh_ms"),
            "observed_receipt_ms": flow.get("feed", {}).get("observed_receipt_ms"),
        }

    def book_flow_view(self, key: str, at: float) -> dict[str, Any]:
        window = self.windows.get(key)
        if not window or not window.flow:
            return {"version": book_flow.VERSION, "status": "UNAVAILABLE"}
        if window.flow["valid_until"] < at or window.current is None:
            return {
                "version": book_flow.VERSION,
                "status": "UNAVAILABLE",
                "quality_flags": ["STREAM_OR_RECONSTRUCTION_UNAVAILABLE"],
                "feed": window.flow["feed"],
            }
        return window.flow

    @staticmethod
    def capture_requires(capture: dict[str, Any], instrument: str, at: float) -> bool:
        if capture["state"] != "CAPTURING":
            return False
        if instrument == capture["key"]:
            return True
        return any(
            instrument in e.get("instruments", [])
            and (e["id"] in capture["open_trades"] or at <= e["observation_until"])
            for e in capture["events"]
        )

    def requires_subscription(self, instrument: str, at: float) -> bool:
        return any(self.capture_requires(c, instrument, at) for c in self.active.values())

    def attach(self, event_id: str, instrument: str, at: float) -> None:
        """Retain evidence membership separately from each event's subscription obligation."""
        for segment, capture in self.active.items():
            event = next((e for e in capture["events"] if e["id"] == event_id), None)
            if capture["state"] != "CAPTURING" or event is None:
                continue
            if instrument in event["instruments"]:
                continue
            event["instruments"].append(instrument)
            window = self.windows[instrument]
            event_at = (
                utc(event["signal_at"]).timestamp()
                if event.get("signal_at")
                else event["detection_receipt"]
            )
            pre = max(0, window.coverage(at) - max(0, at - event_at))
            event.setdefault("prehistory_seconds", {})[instrument] = pre
            capture["metadata_versions"].update(window.metadata)
            previous = (
                capture["last_sequences"].get(instrument, 0)
                if instrument in capture["instruments"]
                else None
            )
            if instrument not in capture["instruments"]:
                capture["instruments"].append(instrument)
            capture["last_sequences"][instrument] = window.sequence
            self.enqueue(
                segment,
                capture,
                [
                    *window.prefix(previous),
                    packed(
                        {
                            "kind": "OPTION_ATTACHED",
                            "event_id": event_id,
                            "identity": window.identity,
                            "receipt": at,
                            "actual_prehistory_seconds": pre,
                        }
                    ),
                ],
            )

    def link_trade(self, event_id: str, is_open: bool, at: float) -> None:
        for segment, capture in self.active.items():
            if any(e["id"] == event_id for e in capture["events"]):
                if is_open and event_id not in capture["open_trades"]:
                    capture["open_trades"].append(event_id)
                elif not is_open and event_id in capture["open_trades"]:
                    capture["open_trades"].remove(event_id)
                    until = at + self.config.post_close_minutes * 60
                    capture["end"] = max(capture["end"], until)
                    event = next(e for e in capture["events"] if e["id"] == event_id)
                    event["observation_until"] = max(event["observation_until"], until)
                self.enqueue(
                    segment,
                    capture,
                    [
                        packed(
                            {
                                "kind": "TRADE_LINK",
                                "event_id": event_id,
                                "open": is_open,
                                "receipt": at,
                            }
                        )
                    ],
                )

    def annotate(self, event_id: str, evidence: dict[str, Any]) -> None:
        for segment, capture in self.active.items():
            for event in capture["events"]:
                if event["id"] == event_id:
                    event.update(json.loads(packed(evidence)))
                    self.enqueue(
                        segment,
                        capture,
                        [
                            packed(
                                {
                                    "kind": "DECISION",
                                    "event_id": event_id,
                                    "payload": evidence,
                                }
                            )
                        ],
                    )

    def tick(self, at: float) -> None:
        self.expire(at)
        if not self.growth or at - self.growth[-1][0] >= 60:
            self.growth.append((at, self.disk_bytes))
        for segment in list(self.markers_pending):
            if segment not in self.active or self.enqueue(segment, self.active[segment], []):
                self.markers_pending.discard(segment)
        for segment, capture in list(self.active.items()):
            # Keep a completed interval available for a later overlapping pre-window.
            if (
                at >= capture["end"]
                and not capture["open_trades"]
                and capture["state"] == "CAPTURING"
            ):
                capture["state"] = "COMPLETE" if not capture["gaps"] else "COMPLETE_WITH_GAPS"
                self.enqueue(segment, capture, [])
            if at > capture["end"] + ROLLING_WINDOW_SECONDS and not capture["open_trades"]:
                self.catalog.append(
                    {
                        k: capture[k]
                        for k in ("segment", "state", "start", "end", "key", "protected")
                    }
                )
                self.active.pop(segment)

    def write_batch(self, items: list[Item]) -> None:
        grouped: dict[str, tuple[dict[str, Any] | None, list[bytes]]] = {}
        for segment, manifest, rows, _ in items:
            previous = grouped.get(segment, (None, []))
            grouped[segment] = (manifest or previous[0], previous[1] + rows)
        for segment, (changed, rows) in grouped.items():
            if changed is not None:
                self.manifests[segment] = changed
            manifest = self.manifests.get(segment)
            if manifest is None:
                raise ValueError("CAPTURE_MANIFEST_UNKNOWN")
            blob = (
                gzip.compress(
                    b"".join(zlib.decompress(row) for row in rows), compresslevel=3, mtime=0
                )
                if rows
                else b""
            )
            self.disk_free = shutil.disk_usage(self.directory).free
            # Leave reserved space for incomplete-state manifests; never prune to make room.
            needed = len(blob) + len(packed(manifest)) * 2 + 4096
            if (
                self.disk_bytes + needed > self.config.archive_max_bytes - 65536
                or self.disk_free - needed < self.config.disk_reserve_bytes
            ):
                raise ValueError("STORAGE_LIMIT_REACHED")
            data = self.directory / (segment + ".jsonl.gz")
            manifest_path = self.directory / (segment + ".manifest.json")
            before = manifest_path.stat().st_size if manifest_path.exists() else 0
            if not before:
                # Durable intent precedes raw bytes: a crash cannot leave an
                # unidentifiable raw segment without event/contract metadata.
                atomic_json(manifest_path, {**manifest, "state": "CAPTURING", "committed_bytes": 0})
                self.disk_bytes += manifest_path.stat().st_size
                before = manifest_path.stat().st_size
            if blob:
                with data.open("ab") as out:
                    out.write(blob)
                    out.flush()
                    os.fsync(out.fileno())
                self.disk_bytes += len(blob)
            manifest["committed_bytes"] = data.stat().st_size if data.exists() else 0
            if changed is not None or not manifest_path.exists():
                # Rows alone do not rewrite the manifest: restart recovery scans the archive.
                atomic_json(manifest_path, manifest)
                self.disk_bytes += manifest_path.stat().st_size - before

    async def write_loop(self) -> None:
        while not self.closed or not self.queue.empty():
            try:
                first = await asyncio.wait_for(self.queue.get(), 0.5)
            except TimeoutError:
                continue
            items = [first]
            if self.queue.qsize() < 64:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.write_ready.wait(), BATCH_SECONDS)
            self.write_ready.clear()
            while not self.queue.empty() and len(items) < 256:
                items.append(self.queue.get_nowait())
            try:
                if self.problem:
                    raise ValueError(self.problem)
                await asyncio.to_thread(self.write_batch, items)
            except Exception as exc:
                self.problem = str(exc) if isinstance(exc, ValueError) else "RECORDER_IO_FAILED"
                # Coded reason plus the exception class only; paths and errno stay out of logs.
                log.warning("%s %s", self.problem, type(exc).__name__)
                for capture in self.active.values():
                    capture.update(state="INCOMPLETE", reason=self.problem)
                # A bounded metadata reserve is available; never exceed the hard
                # archive quota even when marking failure. Restart detects stale intents.
                try:
                    await asyncio.to_thread(self.failure_markers, {i[0] for i in items})
                except Exception as marker_exc:
                    log.warning("RECORDER_MARKER_FAILED %s", type(marker_exc).__name__)
            finally:
                for item in items:
                    self.queued_bytes -= item[3]
                    self.queue.task_done()

    def failure_markers(self, segments: set[str]) -> None:
        for segment in sorted(segments):
            path = self.directory / (segment + ".manifest.json")
            before = path.stat().st_size if path.exists() else 0
            manifest = self.manifests.get(segment)
            if before:
                marker = json.loads(path.read_text())
            elif manifest is not None:
                marker = {k: manifest[k] for k in ("segment", "key", "start", "end")}
            else:
                continue  # nothing durable identifies this segment yet
            marker.update(state="INCOMPLETE", reason=self.problem)
            if manifest is not None and "book_flow" in manifest:
                marker["book_flow"] = manifest["book_flow"]
            size = len(packed(marker))
            if (
                self.disk_bytes + size * 2 < self.config.archive_max_bytes
                and shutil.disk_usage(self.directory).free - size * 2
                >= self.config.disk_reserve_bytes
            ):
                atomic_json(path, marker)
                self.disk_bytes += path.stat().st_size - before

    def view(self, key: str, at: float) -> dict[str, Any]:
        window = self.windows.get(key)
        captures = [c for c in self.active.values() if key in c["instruments"]]
        return {
            "state": "STORAGE_LIMIT"
            if self.problem == "STORAGE_LIMIT_REACHED"
            else "GAP"
            if window and window.current is None
            else "CAPTURING"
            if any(c["state"] == "CAPTURING" for c in captures)
            else "BUFFERING"
            if window
            else "UNAVAILABLE",
            "prehistory_seconds": window.coverage(at) if window else 0,
            "target_seconds": ROLLING_WINDOW_SECONDS,
            "reason": self.problem,
            "persistent_capture_enabled": self.config.persistent_capture,
            "events": [
                {
                    "segment": c["segment"],
                    "state": c["state"],
                    "event_ids": [e["id"] for e in c["events"]],
                }
                for c in captures
            ],
        }

    def status(self) -> dict[str, Any]:
        return {
            "memory_bytes": self.memory(),
            "memory_limit": self.config.rolling_max_bytes,
            "memory_high_water": self.high_water,
            "queue_bytes": self.queued_bytes,
            "writer_queue": self.queue.qsize(),
            "disk_bytes": self.disk_bytes,
            "disk_limit": self.config.archive_max_bytes,
            "disk_free": self.disk_free,
            "disk_reserve": self.config.disk_reserve_bytes,
            "recording_gaps": self.gaps,
            "paused_reason": self.problem,
            "persistent_capture": self.config.persistent_capture,
            "archive_growth": self.archive_growth(time.time()),
        }

    def archive_growth(self, at: float) -> dict[str, Any] | None:
        """Bytes archived over the sampled span within the last day; None until two samples."""
        recent = [s for s in self.growth if s[0] >= at - 86400]
        if len(recent) < 2:
            return None
        return {"bytes": self.disk_bytes - recent[0][1], "seconds": at - recent[0][0]}

    async def close(self) -> None:
        for segment, capture in self.active.items():
            if capture["state"] == "CAPTURING":
                capture.update(state="INCOMPLETE", reason="INTERRUPTED_SHUTDOWN")
                self.enqueue(segment, capture, [])
        self.closed = True
        if self.worker:
            await self.worker

    def prune(self, segment: str, referenced_segments: set[str]) -> None:
        if len(segment) != 64 or any(c not in "0123456789abcdef" for c in segment):
            raise ValueError("INVALID_SEGMENT")
        if segment in self.active or segment in referenced_segments:
            raise ValueError("ACTIVE_OR_REFERENCED_SEGMENT")
        path = self.directory / (segment + ".manifest.json")
        manifest = json.loads(path.read_text())
        if (
            manifest["state"] not in {"COMPLETE", "COMPLETE_WITH_GAPS"}
            or manifest.get("protected")
            or manifest.get("open_trades")
        ):
            raise ValueError("ACTIVE_INCOMPLETE_OR_PROTECTED_SEGMENT")
        for p in (self.directory / (segment + ".jsonl.gz"), path):
            if p.exists():
                self.disk_bytes -= p.stat().st_size
                p.unlink()
        self.catalog = [c for c in self.catalog if c["segment"] != segment]
        if (
            self.problem == "STORAGE_LIMIT_REACHED"
            and len(self.catalog) < 10000
            and self.disk_bytes < self.config.archive_max_bytes
        ):
            self.problem = ""  # room was made; recording resumes with the next capture
