"""15-minute RAM windows and bounded, incremental event evidence off the trading loop.

Raw means delivered Saxo messages, not exchange ticks. Each retained window begins with
the checkpoint immediately preceding its first delta. Disk work runs in one worker thread.
"""

import asyncio
import gzip
import hashlib
import json
import os
import shutil
import zlib
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from stocker_execution import book_flow
from stocker_execution.config import RecorderConfig
from stocker_execution.contracts import utc
from stocker_execution.saxo_auth import atomic_json
from stocker_execution.saxo_stream import merge


def packed(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def read_row(blob: bytes) -> dict[str, Any]:
    """Lossless rolling-memory representation; disk archives remain ordinary JSONL."""
    return cast(dict[str, Any], json.loads(zlib.decompress(blob)))


def apply(state: Any, record: dict[str, Any]) -> Any:
    if record["kind"] == "SNAPSHOT":
        return merge({}, record["payload"])
    if record["kind"] == "GAP":
        return None
    if record["kind"] == "UPDATE" and not record.get("duplicate") and state is not None:
        return merge(state, record["payload"])
    return state


@dataclass
class Window:
    identity: dict[str, Any]
    rows: deque[tuple[float, bytes]] = field(default_factory=deque)
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

    state_bytes: dict[str, int] = field(default_factory=dict)
    flow_history: deque[tuple[int, dict[str, Any], int]] = field(default_factory=deque)
    flow_history_bytes: int = 0

    def __post_init__(self) -> None:
        self.account(
            "checkpoint",
            "current",
            "flow",
            "checkpoint_flow",
            "context",
            "checkpoint_context",
            "metadata",
        )

    def account(self, *names: str) -> None:
        for name in names:
            self.state_bytes[name] = len(packed(getattr(self, name))) * 8

    @property
    def size(self) -> int:
        return self.row_bytes + sum(self.state_bytes.values()) + self.flow_history_bytes + 2048

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
        at, blob = self.rows.popleft()
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
        self.account("checkpoint", "checkpoint_flow", "checkpoint_context")
        while self.flow_history and self.flow_history[0][0] <= record["local_sequence"]:
            self.flow_history_bytes -= self.flow_history.popleft()[2]

    def coverage(self, at: float) -> float:
        if self.continuous_since is None or self.current is None or not self.rows:
            return 0
        beginning = max(at - 900, self.continuous_since, self.rows[0][0])
        if self.checkpoint is not None and self.checkpoint_at is not None:
            beginning = max(at - 900, self.continuous_since, self.checkpoint_at)
        return max(0, at - beginning)

    def prefix(self) -> list[bytes]:
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
            *(zlib.decompress(b) for _, b in self.rows),
        ]


class Recorder:
    def __init__(self, config: RecorderConfig, directory: Path):
        self.config, self.directory = config, directory
        self.windows: dict[str, Window] = {}
        self.active: dict[str, dict[str, Any]] = {}
        self.event_ids: set[str] = set()
        self.queue: asyncio.Queue[tuple[str, dict[str, Any], list[bytes], int]] = asyncio.Queue(
            config.queue_max_items
        )
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

    async def start(self) -> None:
        await asyncio.to_thread(self.recover)
        self.worker = asyncio.create_task(self.write_loop())

    def recover(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Only own atomic-write leftovers; never touch the execution ledger.
        for path in self.directory.glob("*.tmp-*"):
            path.unlink()
        self.disk_bytes = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        self.disk_free = shutil.disk_usage(self.directory).free
        for path in sorted(self.directory.glob("*.manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest.get("state") == "CAPTURING":
                manifest.update(state="INCOMPLETE", reason="INTERRUPTED_RESTART")
                data = self.directory / (manifest["segment"] + ".jsonl.gz")
                # Keep complete gzip members. Only truncate an interrupted, uncommitted tail.
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
                        if decoder.unconsumed_tail:
                            blob = decoder.unconsumed_tail
                        elif decoder.eof:
                            blob = decoder.unused_data
                            good = read_at - len(blob)
                            decoder = zlib.decompressobj(31)
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
            if len(self.windows) >= 32:
                raise ValueError("RECORDER_INSTRUMENT_LIMIT")
            self.windows[key] = Window(identity)
        elif self.windows[key].identity != identity:
            raise ValueError("RECORDING_IDENTITY_IMMUTABLE")

    def memory(self) -> int:
        return sum(w.size for w in self.windows.values())

    def metadata_version(self, key: str, raw: dict[str, Any], received_at: float) -> str:
        window = self.windows[key]
        blob = packed(raw)
        version = hashlib.sha256(blob).hexdigest()
        if version not in window.metadata:
            if len(blob) > 65536 or len(window.metadata) >= 16:
                raise ValueError("REFERENCE_CACHE_LIMIT")
            if self.memory() + len(blob) * 8 > self.config.rolling_max_bytes:
                raise ValueError("REFERENCE_CACHE_MEMORY_LIMIT")
            window.metadata[version] = {"received_at": received_at, "value": json.loads(blob)}
            window.account("metadata")
            if self.memory() > self.config.rolling_max_bytes:
                del window.metadata[version]
                window.account("metadata")
                raise ValueError("REFERENCE_CACHE_MEMORY_LIMIT")
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
            while window.rows and window.rows[0][0] < at - 900:
                window.evict()
        while self.memory() > self.config.rolling_max_bytes:
            populated = [w for w in self.windows.values() if w.rows]
            if not populated:
                # A single provider snapshot can itself exceed the retained-state budget.
                for w in self.windows.values():
                    w.current = w.checkpoint = None
                    w.flow = w.checkpoint_flow = None
                    w.context.clear()
                    w.checkpoint_context.clear()
                    w.continuous_since = None
                    w.flow_history.clear()
                    w.flow_history_bytes = 0
                    w.account(
                        "current",
                        "checkpoint",
                        "flow",
                        "checkpoint_flow",
                        "context",
                        "checkpoint_context",
                    )
                self.problem = "ROLLING_STATE_LIMIT"
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
        duplicate: bool = False,
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
            "duplicate": duplicate,
            "payload": payload,
            "provider_message": provider_message,
            "metadata_version": window.metadata_version,
            "observation_context": observation_context,
            "context_source": context_source,
        }
        provider_times = {}
        envelope = (
            provider_message.get("payload", provider_message)
            if isinstance(provider_message, dict)
            else None
        )
        sources = [(payload, ("LastUpdated",))]
        if isinstance(envelope, list):
            for index, event in enumerate(envelope):
                if isinstance(event, dict) and "Timestamp" in event:
                    with suppress(TypeError, ValueError, AttributeError):
                        provider_times[f"Timestamp[{index}]"] = utc(event["Timestamp"]).isoformat()
        else:
            sources.append((envelope, ("Timestamp",)))
        for source, names in sources:
            if isinstance(source, dict):
                for name in names:
                    if name in source:
                        with suppress(TypeError, ValueError, AttributeError):
                            provider_times[name] = utc(source[name]).isoformat()
        record["provider_timestamps"] = provider_times
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
        previous_state = window.current
        window.current = apply(window.current, record)
        if previous_state is not window.current:
            window.account("current")
        if observation_context or kind == "GAP":
            window.account("context")
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
            window.account("flow")
            window.remember_flow()
            record["book_flow"] = window.flow
            blob = packed(record)
        retained = zlib.compress(blob, level=1)
        window.rows.append((at, retained))
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
                self.enqueue(segment, capture, [blob])

    def enqueue(self, segment: str, manifest: dict[str, Any], rows: list[bytes]) -> bool:
        if not self.config.persistent_capture or self.problem:
            return False
        # A detached JSON copy prevents later event-loop changes racing the writer.
        detached = json.loads(packed(manifest))
        # Prehistory may contain many repeated identities/field names. Queue one
        # lossless batch within the SAME byte/item caps, then expand in the writer.
        if rows:
            compressor = zlib.compressobj(level=1)
            rows = [b"".join(compressor.compress(row) for row in rows) + compressor.flush()]
        size = len(packed(detached)) + sum(len(b) + 64 for b in rows)
        if self.queue.full() or self.queued_bytes + size > self.config.queue_max_bytes:
            self.problem = "WRITE_QUEUE_LIMIT_REACHED"
            for capture in self.active.values():
                capture.update(state="INCOMPLETE", reason=self.problem)
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
                and (c["open_trades"] or c["end"] >= at - 900)
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
            self.problem = "CAPTURE_EVENT_LIMIT"
            capture.update(state="INCOMPLETE", reason=self.problem)
            return {"state": "INCOMPLETE", "reason": self.problem}
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
                    zlib.decompress(blob)
                    for _, blob in self.windows[instrument].rows
                    if read_row(blob)["local_sequence"]
                    > capture["last_sequences"].get(instrument, 0)
                )
            capture["last_sequences"][instrument] = self.windows[instrument].sequence
        rows.append(packed({"kind": "TRIGGER", "payload": capture["events"][-1]}))
        if not self.enqueue(segment, capture, rows):
            capture.update(state="INCOMPLETE", reason=self.problem)
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
            if instrument not in event["instruments"]:
                event["instruments"].append(instrument)
            if instrument in capture["instruments"]:
                self.enqueue(segment, capture, [])
                continue
            window = self.windows[instrument]
            event_at = (
                utc(event["signal_at"]).timestamp()
                if event.get("signal_at")
                else event["detection_receipt"]
            )
            pre = max(0, window.coverage(at) - max(0, at - event_at))
            event.setdefault("prehistory_seconds", {})[instrument] = pre
            capture["metadata_versions"].update(window.metadata)
            capture["instruments"].append(instrument)
            capture["last_sequences"][instrument] = window.sequence
            self.enqueue(
                segment,
                capture,
                [
                    *window.prefix(),
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
        for segment, capture in list(self.active.items()):
            # Keep a completed interval available for a later overlapping pre-window.
            if (
                at >= capture["end"]
                and not capture["open_trades"]
                and capture["state"] == "CAPTURING"
            ):
                capture["state"] = "COMPLETE" if not capture["gaps"] else "COMPLETE_WITH_GAPS"
                self.enqueue(segment, capture, [])
            if at > capture["end"] + 900 and not capture["open_trades"]:
                self.catalog.append(
                    {
                        k: capture[k]
                        for k in ("segment", "state", "start", "end", "key", "protected")
                    }
                )
                self.active.pop(segment)

    def write_batch(self, items: list[tuple[str, dict[str, Any], list[bytes], int]]) -> None:
        grouped: dict[str, tuple[dict[str, Any], list[bytes]]] = {}
        for segment, manifest, rows, _ in items:
            if segment not in grouped:
                grouped[segment] = (manifest, list(rows))
            else:
                grouped[segment] = (manifest, grouped[segment][1] + rows)
        for segment, (manifest, rows) in grouped.items():
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
            atomic_json(manifest_path, manifest)
            self.disk_bytes += manifest_path.stat().st_size - before

    async def write_loop(self) -> None:
        while not self.closed or not self.queue.empty():
            try:
                first = await asyncio.wait_for(self.queue.get(), 0.5)
            except TimeoutError:
                continue
            items = [first]
            # Keep modest batching at normal cadence; drain promptly during bursts.
            if self.queue.qsize() < 64:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.write_ready.wait(), 0.1)
            self.write_ready.clear()
            while not self.queue.empty() and len(items) < 256:
                items.append(self.queue.get_nowait())
            try:
                if self.problem:
                    raise ValueError(self.problem)
                await asyncio.to_thread(self.write_batch, items)
            except Exception as exc:
                self.problem = str(exc) if isinstance(exc, ValueError) else "RECORDER_IO_FAILED"
                for capture in self.active.values():
                    capture.update(state="INCOMPLETE", reason=self.problem)
                # A bounded metadata reserve is available; never exceed the hard
                # archive quota even when marking failure. Restart detects stale intents.
                with suppress(OSError):
                    await asyncio.to_thread(self.failure_markers, items)
            finally:
                for item in items:
                    self.queued_bytes -= item[3]
                    self.queue.task_done()

    def failure_markers(self, items: list[tuple[str, dict[str, Any], list[bytes], int]]) -> None:
        for segment, manifest in {i[0]: i[1] for i in items}.items():
            path = self.directory / (segment + ".manifest.json")
            before = path.stat().st_size if path.exists() else 0
            marker = (
                json.loads(path.read_text())
                if before
                else {k: manifest[k] for k in ("segment", "key", "start", "end")}
            )
            marker.update(state="INCOMPLETE", reason=self.problem)
            if "book_flow" in manifest:
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
            "target_seconds": 900,
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
        }

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
