"""Bounded per-contract receipts for completed option market-data snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from typing import cast

from stocker_runtime.domain import JsonValue, canonical_json_bytes

MAX_OPTION_SNAPSHOT_INPUTS = 64


class SnapshotProjectionError(RuntimeError):
    """One completed snapshot cannot be represented by bounded exact evidence."""


def _number(payload: Mapping[str, object], name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _payload(row: sqlite3.Row) -> Mapping[str, object]:
    parsed = json.loads(str(row["payload_json"]))
    if not isinstance(parsed, Mapping):
        raise SnapshotProjectionError("option snapshot source payload is not an object")
    return cast(Mapping[str, object], parsed)


def _latest_number(rows: tuple[sqlite3.Row, ...], name: str) -> float | None:
    for row in reversed(rows):
        value = _number(_payload(row), name)
        if value is not None:
            return value
    return None


def _latest_nonnegative(rows: tuple[sqlite3.Row, ...], name: str) -> float | None:
    value = _latest_number(rows, name)
    return value if value is not None and value >= 0.0 else None


def _computation(rows: tuple[sqlite3.Row, ...], tick_type: int) -> Mapping[str, object] | None:
    for row in reversed(rows):
        if str(row["event_kind"]) != "option_computation":
            continue
        payload = _payload(row)
        value = payload.get("tick_type")
        if isinstance(value, int) and not isinstance(value, bool) and value == tick_type:
            return payload
    return None


def _capture_payload(
    marker: sqlite3.Row,
    sources: tuple[sqlite3.Row, ...],
    *,
    subscription_id: str,
    option_right: str,
) -> dict[str, JsonValue]:
    bid = _latest_number(sources, "bid")
    ask = _latest_number(sources, "ask")
    bid_size = _latest_number(sources, "bid_size")
    ask_size = _latest_number(sources, "ask_size")
    close = _latest_number(sources, "close")
    open_interest_name = "call_open_interest" if option_right == "call" else "put_open_interest"
    volume_name = "call_option_volume" if option_right == "call" else "put_option_volume"
    open_interest = _latest_nonnegative(sources, open_interest_name)
    option_volume = _latest_nonnegative(sources, volume_name)
    computations: dict[str, JsonValue] = {}
    for tick_type, name in ((10, "bid"), (11, "ask"), (12, "last"), (13, "model")):
        item = _computation(sources, tick_type)
        if item is None:
            continue
        computations[name] = cast(
            JsonValue,
            {
                field: _number(item, field)
                for field in (
                    "implied_volatility",
                    "delta",
                    "option_price",
                    "present_value_dividend",
                    "gamma",
                    "vega",
                    "theta",
                    "underlying_price",
                )
            },
        )
    model = cast(Mapping[str, object], computations.get("model", {}))
    model_iv = _number(model, "implied_volatility")
    model_delta = _number(model, "delta")
    errors: list[str] = []
    if bid is None or bid < 0.0:
        errors.append("bid_missing_or_invalid")
    if ask is None or ask < 0.0:
        errors.append("ask_missing_or_invalid")
    if bid is not None and ask is not None and ask < bid:
        errors.append("crossed_quote")
    if bid is not None and ask is not None and (bid + ask) / 2.0 <= 0.0:
        errors.append("midpoint_not_positive")
    if model_iv is None or not 0.005 <= model_iv <= 5.0:
        errors.append("model_implied_volatility_missing_or_invalid")
    marker_payload = _payload(marker)
    if marker_payload.get("complete") is not True:
        errors.append("snapshot_completion_invalid")
    input_ids = tuple(str(row["event_id"]) for row in sources)
    return {
        "schema_version": 1,
        "subscription_id": subscription_id,
        "source_completeness": "complete" if not errors else "incomplete",
        "errors": tuple(errors),
        "input_count": len(input_ids),
        "input_ids_hash": hashlib.sha256(
            canonical_json_bytes(cast(JsonValue, input_ids))
        ).hexdigest(),
        "snapshot_completed_at_us": int(marker["received_at_us"]),
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "close": close,
        "open_interest": open_interest,
        "open_interest_missing": open_interest is None,
        "option_volume": option_volume,
        "option_volume_missing": option_volume is None,
        "model_implied_volatility": model_iv,
        "model_delta": model_delta,
        "option_computations": computations,
    }


def _project_one(
    connection: sqlite3.Connection,
    marker: sqlite3.Row,
) -> None:
    rows = tuple(
        connection.execute(
            "SELECT event.* FROM callback_inbox callback "
            "JOIN market_events event ON event.source_sequence=callback.source_sequence "
            "AND event.run_id=callback.run_id "
            "WHERE callback.run_id=? AND callback.recorder_generation=? "
            "AND callback.connection_generation=? AND callback.request_id=? "
            "AND callback.received_at_us>=? AND callback.source_sequence<=? "
            "AND event.event_kind IN "
            "('quote','option_computation','option_snapshot_end') "
            "ORDER BY callback.source_sequence LIMIT ?",
            (
                marker["run_id"],
                marker["recorder_generation"],
                marker["connection_generation"],
                marker["request_id"],
                marker["opened_at_us"],
                marker["source_sequence"],
                MAX_OPTION_SNAPSHOT_INPUTS + 1,
            ),
        )
    )
    if len(rows) > MAX_OPTION_SNAPSHOT_INPUTS:
        raise SnapshotProjectionError("option snapshot input bound exceeded")
    if not rows or str(rows[-1]["event_id"]) != str(marker["event_id"]):
        raise SnapshotProjectionError("option snapshot completion is not the final source event")
    payload = _capture_payload(
        marker,
        rows,
        subscription_id=str(marker["subscription_id"]),
        option_right=str(marker["option_right"]),
    )
    payload_json = canonical_json_bytes(cast(JsonValue, payload)).decode()
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    event_id = hashlib.sha256(
        canonical_json_bytes(
            cast(
                JsonValue,
                {
                    "run_id": str(marker["run_id"]),
                    "instrument_id": str(marker["instrument_id"]),
                    "event_kind": "option_snapshot_capture",
                    "completion_event_id": str(marker["event_id"]),
                    "payload_sha256": payload_hash,
                },
            )
        )
    ).hexdigest()
    connection.execute(
        "INSERT INTO market_events(event_id, run_id, source_sequence, "
        "derived_after_source_sequence, instrument_id, feed_kind, event_kind, event_at_us, "
        "received_at_us, connection_generation, bid_value, ask_value, bid_size_value, "
        "ask_size_value, close_value, payload_json, payload_sha256) "
        "VALUES (?, ?, NULL, ?, ?, 'quotes', 'option_snapshot_capture', "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            marker["run_id"],
            marker["source_sequence"],
            marker["instrument_id"],
            marker["event_at_us"],
            marker["received_at_us"],
            marker["connection_generation"],
            payload["bid"],
            payload["ask"],
            payload["bid_size"],
            payload["ask_size"],
            payload["close"],
            payload_json,
            payload_hash,
        ),
    )
    connection.executemany(
        "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
        "input_ordinal, input_role, created_at_us) VALUES (?, ?, ?, ?, ?)",
        (
            (
                event_id,
                str(row["event_id"]),
                ordinal,
                "completion" if str(row["event_id"]) == str(marker["event_id"]) else "constituent",
                int(marker["received_at_us"]),
            )
            for ordinal, row in enumerate(rows)
        ),
    )


def project_option_snapshot_captures(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    limit: int = 64,
) -> int:
    """Project completed option snapshots once, in source order."""

    if not 1 <= limit <= 256:
        raise ValueError("option snapshot capture limit must be between 1 and 256")
    markers = tuple(
        connection.execute(
            "SELECT marker.*, callback.recorder_generation, callback.request_id, "
            "subscription.subscription_id, subscription.opened_at_us, "
            "instrument.option_right FROM market_events marker "
            "JOIN callback_inbox callback ON callback.source_sequence=marker.source_sequence "
            "AND callback.run_id=marker.run_id "
            "JOIN subscriptions subscription ON subscription.run_id=callback.run_id "
            "AND subscription.recorder_generation=callback.recorder_generation "
            "AND subscription.connection_generation=callback.connection_generation "
            "AND subscription.request_id=callback.request_id "
            "JOIN instruments instrument ON instrument.instrument_id=marker.instrument_id "
            "WHERE marker.run_id=? AND marker.event_kind='option_snapshot_end' "
            "AND instrument.kind='option' AND subscription.snapshot=1 "
            "AND NOT EXISTS (SELECT 1 FROM market_event_derivations derivation "
            "JOIN market_events capture ON capture.event_id=derivation.derived_event_id "
            "WHERE derivation.input_event_id=marker.event_id "
            "AND derivation.input_role='completion' "
            "AND capture.event_kind='option_snapshot_capture') "
            "ORDER BY marker.source_sequence LIMIT ?",
            (run_id, limit),
        )
    )
    for marker in markers:
        _project_one(connection, marker)
    return len(markers)


__all__ = [
    "MAX_OPTION_SNAPSHOT_INPUTS",
    "SnapshotProjectionError",
    "project_option_snapshot_captures",
]
