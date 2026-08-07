"""Causal, receipt-backed projections from IBKR five-second bars."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, time
from typing import cast
from zoneinfo import ZoneInfo

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ideas.contract import MarketDataRequirement

_NEW_YORK = ZoneInfo("America/New_York")
_FIVE_SECONDS_US = 5_000_000
_FIVE_MINUTES_US = 300_000_000
_EXPECTED_CONSTITUENTS = 60


class BarProjectionError(RuntimeError):
    """The durable raw evidence cannot be projected without violating causality."""


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _session_open_us(event_at_us: int) -> int:
    local = datetime.fromtimestamp(event_at_us / 1_000_000, tz=UTC).astimezone(_NEW_YORK)
    opening = datetime.combine(local.date(), time(9, 30), tzinfo=_NEW_YORK)
    return int(opening.timestamp() * 1_000_000)


def _numeric(payload: Mapping[str, object], name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _project_instrument(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    instrument_id: str,
    after_source_sequence: int,
) -> int:
    projected = connection.execute(
        "SELECT event_at_us, derived_after_source_sequence FROM market_events "
        "WHERE run_id=? AND instrument_id=? AND feed_kind='bars' AND event_kind='bar_5m' "
        "ORDER BY event_at_us DESC LIMIT 1",
        (run_id, instrument_id),
    ).fetchone()
    lower_bound_us = 0 if projected is None else int(projected["event_at_us"])
    # The event at ``lower_bound_us`` may have served as the prior interval's
    # progress proof and must also remain available as this interval's first
    # constituent. Event time excludes immutable prior windows; the activation
    # fence alone excludes pre-activation source sequences.
    lower_sequence = after_source_sequence
    raw_rows = connection.execute(
        "SELECT event_id, source_sequence, event_at_us, received_at_us, "
        "connection_generation, payload_json FROM market_events "
        "WHERE run_id=? AND instrument_id=? AND feed_kind='bars' AND event_kind='bar' "
        "AND event_at_us>=? AND source_sequence>? "
        "ORDER BY source_sequence",
        (run_id, instrument_id, lower_bound_us, lower_sequence),
    ).fetchall()
    if not raw_rows:
        return 0

    sessions = sorted({_session_open_us(int(row["event_at_us"])) for row in raw_rows})
    inserted = 0
    for session_open_us in sessions:
        session_close_us = session_open_us + 78 * _FIVE_MINUTES_US
        session_rows = tuple(
            sorted(
                (
                    row
                    for row in raw_rows
                    if session_open_us <= int(row["event_at_us"]) < session_close_us
                ),
                key=lambda row: (
                    int(row["event_at_us"]),
                    int(row["source_sequence"]),
                    str(row["event_id"]),
                ),
            )
        )
        if not session_rows:
            continue
        session_date = (
            datetime.fromtimestamp(session_open_us / 1_000_000, tz=UTC)
            .astimezone(_NEW_YORK)
            .date()
            .isoformat()
        )
        for bar_number in range(1, 79):
            start_us = session_open_us + (bar_number - 1) * _FIVE_MINUTES_US
            end_us = start_us + _FIVE_MINUTES_US
            already = connection.execute(
                "SELECT 1 FROM market_events WHERE run_id=? AND instrument_id=? "
                "AND event_kind='bar_5m' AND event_at_us=?",
                (run_id, instrument_id, end_us),
            ).fetchone()
            if already is not None:
                continue
            constituents = tuple(
                row for row in session_rows if start_us <= int(row["event_at_us"]) < end_us
            )
            progress = next(
                (
                    row
                    for row in session_rows
                    if int(row["event_at_us"]) >= end_us and int(row["received_at_us"]) >= end_us
                ),
                None,
            )
            expected_final = next(
                (
                    row
                    for row in constituents
                    if int(row["event_at_us"]) == end_us - _FIVE_SECONDS_US
                    and int(row["received_at_us"]) >= end_us
                ),
                None,
            )
            causal_progress = progress or expected_final
            if causal_progress is None:
                # No durable evidence has yet crossed the interval boundary.
                break

            expected_times = tuple(
                start_us + index * _FIVE_SECONDS_US for index in range(_EXPECTED_CONSTITUENTS)
            )
            actual_times = tuple(int(row["event_at_us"]) for row in constituents)
            counts = {
                event_at_us: actual_times.count(event_at_us) for event_at_us in set(actual_times)
            }
            missing_count = sum(event_at_us not in counts for event_at_us in expected_times)
            duplicate_count = sum(max(0, count - 1) for count in counts.values())
            unexpected_count = sum(
                event_at_us not in set(expected_times) for event_at_us in actual_times
            )
            gaps = connection.execute(
                "SELECT gap.gap_id FROM gaps gap JOIN subscriptions subscription "
                "ON subscription.subscription_id=gap.subscription_id "
                "WHERE gap.run_id=? AND subscription.instrument_id=? "
                "AND subscription.feed_kind='bars' AND gap.started_at_us<? "
                "AND coalesce(gap.ended_at_us, 9223372036854775807)>? ORDER BY gap.gap_id",
                (run_id, instrument_id, end_us, start_us),
            ).fetchall()
            gap_ids = tuple(str(row[0]) for row in gaps)
            complete = (
                len(constituents) == _EXPECTED_CONSTITUENTS
                and missing_count == 0
                and duplicate_count == 0
                and unexpected_count == 0
                and not gap_ids
            )
            ordered_input_ids = tuple(str(row["event_id"]) for row in constituents)
            progress_id = str(causal_progress["event_id"])
            first_sequence = min(
                (int(row["source_sequence"]) for row in constituents),
                default=int(causal_progress["source_sequence"]),
            )
            evidence_rows = (
                constituents
                if progress_id in ordered_input_ids
                else (*constituents, causal_progress)
            )
            last_sequence = max(int(row["source_sequence"]) for row in evidence_rows)
            received_watermark = max(int(row["received_at_us"]) for row in evidence_rows)
            connection_generation = max(int(row["connection_generation"]) for row in evidence_rows)
            payloads = tuple(
                cast(Mapping[str, object], json.loads(str(row["payload_json"])))
                for row in constituents
            )
            receipt: dict[str, JsonValue] = {
                "bar_number": bar_number,
                "bar_start_at_us": start_us,
                "bar_end_at_us": end_us,
                "session": session_date,
                "source_completeness": "complete" if complete else "incomplete",
                "expected_input_count": _EXPECTED_CONSTITUENTS,
                "input_count": len(ordered_input_ids),
                "input_ids_hash": _hash(cast(JsonValue, ordered_input_ids)),
                "first_source_sequence": first_sequence,
                "derived_after_source_sequence": last_sequence,
                "missing_count": missing_count,
                "duplicate_count": duplicate_count,
                "unexpected_count": unexpected_count,
                "overlapping_gap_count": len(gap_ids),
                "overlapping_gap_ids_hash": _hash(cast(JsonValue, gap_ids)),
            }
            if complete:
                opens = tuple(_numeric(payload, "open") for payload in payloads)
                highs = tuple(_numeric(payload, "high") for payload in payloads)
                lows = tuple(_numeric(payload, "low") for payload in payloads)
                closes = tuple(_numeric(payload, "close") for payload in payloads)
                volumes = tuple(_numeric(payload, "volume") for payload in payloads)
                if any(
                    value is None for values in (opens, highs, lows, closes) for value in values
                ):
                    receipt["source_completeness"] = "incomplete"
                    receipt["invalid_numeric_payload"] = True
                    complete = False
                else:
                    receipt.update(
                        {
                            "open": cast(float, opens[0]),
                            "high": max(cast(tuple[float, ...], highs)),
                            "low": min(cast(tuple[float, ...], lows)),
                            "close": cast(float, closes[-1]),
                            "volume": sum(value for value in volumes if value is not None),
                        }
                    )
            event_material = cast(
                JsonValue,
                {
                    "run_id": run_id,
                    "instrument_id": instrument_id,
                    "event_kind": "bar_5m",
                    "bar_start_at_us": start_us,
                    "receipt": receipt,
                },
            )
            event_id = _hash(event_material)
            payload_json = canonical_json_bytes(cast(JsonValue, receipt)).decode()
            mapping_rows: list[tuple[str, str, int, str, int]] = []
            for ordinal, input_id in enumerate(ordered_input_ids[:_EXPECTED_CONSTITUENTS]):
                mapping_rows.append((event_id, input_id, ordinal, "constituent", end_us))
            if progress_id not in {row[1] for row in mapping_rows}:
                mapping_rows.append((event_id, progress_id, len(mapping_rows), "progress", end_us))
            first_payload = payloads[0] if payloads else {}
            last_payload = payloads[-1] if payloads else {}
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, open_value, high_value, "
                "low_value, close_value, volume_value, payload_json, payload_sha256) "
                "VALUES (?, ?, NULL, ?, ?, 'bars', 'bar_5m', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    run_id,
                    last_sequence,
                    instrument_id,
                    end_us,
                    received_watermark,
                    connection_generation,
                    _numeric(first_payload, "open") if complete else None,
                    max(cast(tuple[float, ...], tuple(_numeric(p, "high") for p in payloads)))
                    if complete
                    else None,
                    min(cast(tuple[float, ...], tuple(_numeric(p, "low") for p in payloads)))
                    if complete
                    else None,
                    _numeric(last_payload, "close") if complete else None,
                    sum(_numeric(p, "volume") or 0.0 for p in payloads) if complete else None,
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                ),
            )
            connection.executemany(
                "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
                "input_ordinal, input_role, created_at_us) VALUES (?, ?, ?, ?, ?)",
                mapping_rows,
            )
            inserted += 1
    return inserted


def project_required_five_minute_bars(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    requirements: Sequence[MarketDataRequirement],
    after_source_sequence: int = 0,
) -> int:
    """Materialize complete/incomplete causal 5m receipts required by active plugins."""

    instruments = sorted(
        {
            requirement.instrument_id
            for requirement in requirements
            if requirement.feed_kind == "bars" and requirement.event_kind == "bar_5m"
        }
    )
    return sum(
        _project_instrument(
            connection,
            run_id=run_id,
            instrument_id=instrument_id,
            after_source_sequence=after_source_sequence,
        )
        for instrument_id in instruments
    )
