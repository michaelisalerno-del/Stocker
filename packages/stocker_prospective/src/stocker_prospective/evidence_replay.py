"""Read-only deterministic replay of persisted prospective evidence.

The replay process never imports or receives an IBKR adapter.  It verifies raw
partition hashes, restores raw event contracts, replays the frozen M1C
probabilities and fresh-episode state machine, and emits every persisted
downstream scientific stage in one canonical order.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from stocker_prospective.events import (
    FiveMinuteBarEvent,
    OptionQuoteEvent,
    RawCallbackEnvelopeEvent,
    RawEvent,
    UnderlyingDepthEvent,
    UnderlyingDepthSnapshotEvent,
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickBidAskEvent,
    UnderlyingTickTradeEvent,
)
from stocker_prospective.frozen_m1c import FreshEpisodeTracker, FrozenM1CRuntime


class EvidenceReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str | None
    mode: str
    records_replayed: int
    raw_events_replayed: int
    stage_counts: dict[str, int]
    digest: str
    raw_partition_hash_mismatches: int
    m1c_probability_mismatches: int
    episode_identity_mismatches: int
    maximum_floating_difference: float
    ibkr_connections_attempted: int = 0
    broker_state_mutated: bool = False


_EVENT_MODELS: dict[str, type[RawEvent]] = {
    "raw_callback_envelope_event": RawCallbackEnvelopeEvent,
    "underlying_level1_quote_event": UnderlyingLevel1QuoteEvent,
    "underlying_tick_bidask_event": UnderlyingTickBidAskEvent,
    "underlying_tick_trade_event": UnderlyingTickTradeEvent,
    "underlying_depth_event": UnderlyingDepthEvent,
    "underlying_depth_snapshot": UnderlyingDepthSnapshotEvent,
    "option_quote_event": OptionQuoteEvent,
    "five_minute_bar_event": FiveMinuteBarEvent,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_json_columns(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for name, value in tuple(decoded.items()):
        if isinstance(value, str) and (
            name.endswith("_json")
            or name
            in {
                "conditions",
                "original_payload",
                "quote_attributes",
                "snapshot",
                "stream_owner",
            }
        ):
            try:
                decoded[name.removesuffix("_json")] = json.loads(value)
            except json.JSONDecodeError:
                continue
            if name.endswith("_json"):
                decoded.pop(name)
    return decoded


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("replay evidence timestamp is timezone-naive")
    return result.astimezone(UTC)


def _raw_sort_key(event: RawEvent) -> tuple[datetime, int, int, str]:
    return (
        event.ordering_timestamp,
        event.received_monotonic_ns,
        event.source_sequence,
        event.event_id,
    )


def _load_raw_events(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    stop_event: threading.Event,
) -> tuple[tuple[RawEvent, ...], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("blocked_replay_dependency_unavailable: pyarrow") from exc
    manifests = connection.execute(
        """
        SELECT event_type, file_path, content_hash
        FROM raw_partition_manifest_v0
        WHERE run_id = ?
        ORDER BY minimum_timestamp_utc, content_hash
        """,
        (run_id,),
    ).fetchall()
    events: list[RawEvent] = []
    mismatches = 0
    for manifest in manifests:
        if stop_event.is_set():
            break
        path = Path(str(manifest["file_path"]))
        if not path.is_file() or _sha256(path) != str(manifest["content_hash"]):
            mismatches += 1
            continue
        event_type = str(manifest["event_type"])
        model = _EVENT_MODELS.get(event_type)
        if model is None:
            raise ValueError(f"unsupported replay raw event type: {event_type}")
        table = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
        for raw in table.to_pylist():
            events.append(model.model_validate(_decode_json_columns(dict(raw))))
    return tuple(sorted(events, key=_raw_sort_key)), mismatches


def _stage_rows(
    connection: sqlite3.Connection,
    *,
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    specifications = (
        ("m1c_prediction", "m1c_checkpoint_v0", "bar_end_utc", "id"),
        ("m1c_episode", "m1c_episode_v0", "trigger_bar_end_utc", "episode_id"),
        (
            "directional_archetype",
            "direction_classification_v0",
            "maximum_feature_timestamp_utc",
            "id",
        ),
        (
            "microstructure_summary",
            "microstructure_summary_v0",
            "window_end_utc",
            "id",
        ),
        (
            "promotion_decision",
            "promotion_decision_v0",
            "promotion_time_utc",
            "id",
        ),
        (
            "subscription_lifecycle",
            "subscription_lifecycle_v0",
            "started_at_utc",
            "id",
        ),
        (
            "option_contract",
            "episode_option_contract_v0",
            "recording_started_at_utc",
            "id",
        ),
        (
            "shadow_quote_outcome",
            "shadow_quote_outcome_v0",
            "target_timestamp_utc",
            "id",
        ),
        (
            "shadow_structure_outcome",
            "shadow_structure_outcome_v0",
            "id",
            "id",
        ),
    )
    rows: list[dict[str, Any]] = []
    for stage, table, timestamp_column, identity_column in specifications:
        selected = connection.execute(
            f"SELECT * FROM {table} WHERE run_id = ?",  # noqa: S608 - fixed table contract
            (run_id,),
        ).fetchall()
        for selected_row in selected:
            payload = _decode_json_columns(dict(selected_row))
            raw_timestamp = payload.get(timestamp_column)
            if raw_timestamp is None:
                raw_timestamp = payload.get("id", 0)
            timestamp = (
                datetime.min.replace(tzinfo=UTC).isoformat()
                if isinstance(raw_timestamp, int)
                else _timestamp(raw_timestamp).isoformat()
            )
            rows.append(
                {
                    "stage": stage,
                    "timestamp": timestamp,
                    "identity": str(payload.get(identity_column)),
                    "payload": payload,
                }
            )
    return tuple(rows)


def _verify_m1c(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    runtime: FrozenM1CRuntime | None,
) -> tuple[int, int, float]:
    rows = connection.execute(
        """
        SELECT * FROM m1c_checkpoint_v0
        WHERE run_id = ?
        ORDER BY symbol, session_date, checkpoint
        """,
        (run_id,),
    ).fetchall()
    probability_mismatches = 0
    episode_mismatches = 0
    maximum_difference = 0.0
    tracker = FreshEpisodeTracker()
    replayed_episode_ids: set[str] = set()
    for row in rows:
        payload = dict(row)
        features = json.loads(str(payload["feature_values_json"]))
        if runtime is not None:
            score = runtime.score(
                symbol=str(payload["symbol"]),
                checkpoint=int(payload["checkpoint"]),
                group_o_context={
                    name: features.get(name) for name in runtime.required_group_o_features
                },
                causal_group_i={
                    name: features.get(name) for name in runtime.causal_group_i_features
                },
            )
            difference = abs(score.probability - float(payload["probability"]))
            maximum_difference = max(maximum_difference, difference)
            if difference > 1e-12 or score.threshold_passed != bool(payload["threshold_passed"]):
                probability_mismatches += 1
        decision = tracker.evaluate(
            symbol=str(payload["symbol"]),
            session=datetime.fromisoformat(str(payload["session_date"])).date(),
            checkpoint=int(payload["checkpoint"]),
            trigger_bar_end=_timestamp(payload["bar_end_utc"]),
            probability=float(payload["probability"]),
            eligible=bool(payload["eligible"]),
        )
        if decision.fresh_episode and decision.episode_id is not None:
            replayed_episode_ids.add(decision.episode_id)
    persisted_episode_ids = {
        str(row["episode_id"])
        for row in connection.execute(
            "SELECT episode_id FROM m1c_episode_v0 WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    }
    episode_mismatches = len(replayed_episode_ids.symmetric_difference(persisted_episode_ids))
    return probability_mismatches, episode_mismatches, maximum_difference


def replay_persisted_evidence(
    *,
    database_path: str | Path,
    run_id: str | None,
    mode: str,
    speed: float,
    episode_id: str | None,
    m1c_feature_manifest_path: str | Path | None,
    m1c_threshold_path: str | Path | None,
    stop_event: threading.Event,
) -> EvidenceReplayResult:
    """Replay one immutable evidence run without constructing broker connectivity."""

    if speed <= 0.0:
        raise ValueError("replay speed must be positive")
    uri = f"file:{Path(database_path).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        selected_run = run_id
        if selected_run is None:
            latest = connection.execute(
                "SELECT run_id FROM prospective_run ORDER BY created_at_utc DESC LIMIT 1"
            ).fetchone()
            selected_run = None if latest is None else str(latest["run_id"])
        if selected_run is None:
            return EvidenceReplayResult(
                run_id=None,
                mode=mode,
                records_replayed=0,
                raw_events_replayed=0,
                stage_counts={},
                digest=hashlib.sha256(b"[]").hexdigest(),
                raw_partition_hash_mismatches=0,
                m1c_probability_mismatches=0,
                episode_identity_mismatches=0,
                maximum_floating_difference=0.0,
            )
        raw_events, partition_mismatches = _load_raw_events(
            connection,
            run_id=selected_run,
            stop_event=stop_event,
        )
        runtime = None
        if m1c_feature_manifest_path is not None and m1c_threshold_path is not None:
            manifest = Path(m1c_feature_manifest_path)
            threshold = Path(m1c_threshold_path)
            if manifest.is_file() and threshold.is_file():
                runtime = FrozenM1CRuntime.from_artifacts(
                    feature_manifest_path=manifest,
                    threshold_path=threshold,
                )
        probability_mismatches, episode_mismatches, maximum_difference = _verify_m1c(
            connection,
            run_id=selected_run,
            runtime=runtime,
        )
        if partition_mismatches or probability_mismatches or episode_mismatches:
            raise ValueError(
                "deterministic replay mismatch: "
                f"partitions={partition_mismatches},"
                f"m1c={probability_mismatches},episodes={episode_mismatches}"
            )
        stage_rows = list(_stage_rows(connection, run_id=selected_run))

    raw_records = [
        {
            "stage": (
                "raw_callback_envelope"
                if isinstance(event, RawCallbackEnvelopeEvent)
                else "five_minute_bar"
                if isinstance(event, FiveMinuteBarEvent)
                else "option_quote"
                if isinstance(event, OptionQuoteEvent)
                else "raw_market_event"
            ),
            "timestamp": event.ordering_timestamp.isoformat(),
            "identity": event.event_id,
            "payload": event.model_dump(mode="json"),
        }
        for event in raw_events
    ]
    records = [*raw_records, *stage_rows]
    if mode == "episode_only":
        if not episode_id:
            raise ValueError("episode-only replay requires episode_id")
        records = [
            item for item in records if item["payload"].get("episode_id") in {None, episode_id}
        ]
    records.sort(
        key=lambda item: (
            item["timestamp"],
            item["stage"],
            item["identity"],
        )
    )
    if mode == "step":
        records = records[:1]
    if stop_event.is_set():
        records = []
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    counts = Counter(str(item["stage"]) for item in records)
    return EvidenceReplayResult(
        run_id=selected_run,
        mode=mode,
        records_replayed=len(records),
        raw_events_replayed=sum(
            1
            for item in records
            if item["stage"]
            in {
                "raw_callback_envelope",
                "raw_market_event",
                "five_minute_bar",
                "option_quote",
            }
        ),
        stage_counts=dict(sorted(counts.items())),
        digest=hashlib.sha256(canonical.encode()).hexdigest(),
        raw_partition_hash_mismatches=partition_mismatches,
        m1c_probability_mismatches=probability_mismatches,
        episode_identity_mismatches=episode_mismatches,
        maximum_floating_difference=maximum_difference,
    )


__all__ = ["EvidenceReplayResult", "replay_persisted_evidence"]
