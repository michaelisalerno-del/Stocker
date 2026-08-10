"""Bounded generic session receipts derived from complete five-minute bars."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from typing import cast

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ideas.contract import MarketDataRequirement

_MINIMUM_ACTIVITY_SESSIONS = 10
_REALIZED_VOLATILITY_RETURNS = 20
# Direction classifications exclude the trigger bar and need eleven T-1 bars for
# the frozen boundary-attempt window, so the receipt retains those plus the trigger.
_MAX_TRAILING_BARS = 12
_BARS_PER_SESSION = 78
_MAX_PROJECTION_ROWS = 10_000


class SessionProjectionError(RuntimeError):
    """Durable five-minute evidence cannot form an exact session receipt."""


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise SessionProjectionError("session receipt payload must be an object")
    return cast(Mapping[str, object], payload)


def _number(payload: Mapping[str, object], name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(payload: Mapping[str, object], name: str) -> int | None:
    value = payload.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _event_payload(row: sqlite3.Row) -> Mapping[str, object]:
    parsed = json.loads(str(row["payload_json"]))
    return _mapping(parsed)


def _latest_baseline(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    instrument_id: str,
    session: str,
) -> sqlite3.Row | None:
    row = connection.execute(
        "SELECT * FROM market_events WHERE run_id=? AND instrument_id=? "
        "AND feed_kind='bars' AND event_kind='session_volume_baseline' "
        "AND json_extract(payload_json, '$.session')<? "
        "ORDER BY event_at_us DESC, event_id DESC LIMIT 1",
        (run_id, instrument_id, session),
    ).fetchone()
    return cast(sqlite3.Row | None, row)


def _prior_prefix(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    instrument_id: str,
    session: str,
    bar_number: int,
) -> sqlite3.Row | None:
    if bar_number == 1:
        return None
    row = connection.execute(
        "SELECT * FROM market_events WHERE run_id=? AND instrument_id=? "
        "AND feed_kind='bars' AND event_kind='bar_5m_session_prefix' "
        "AND json_extract(payload_json, '$.session')=? "
        "AND json_extract(payload_json, '$.bar_number')=? LIMIT 1",
        (run_id, instrument_id, session, bar_number - 1),
    ).fetchone()
    return cast(sqlite3.Row | None, row)


def _relative_activity(
    baseline: Mapping[str, object] | None,
    *,
    bar_number: int,
    volume: float | None,
) -> tuple[int, float | None]:
    if baseline is None:
        return 0, None
    count = _integer(baseline, "complete_session_count")
    sums = baseline.get("ordinal_volume_sums")
    if count is None or count < 0 or not isinstance(sums, list) or len(sums) != 78:
        raise SessionProjectionError("session volume baseline payload is invalid")
    if count < _MINIMUM_ACTIVITY_SESSIONS or volume is None or volume < 0.0:
        return count, None
    total = sums[bar_number - 1]
    if isinstance(total, bool) or not isinstance(total, int | float):
        raise SessionProjectionError("session volume baseline contains a non-number")
    mean = float(total) / count
    if not math.isfinite(mean) or mean <= 1e-12:
        return count, None
    return count, volume / mean


def _compact_bar(
    bar: Mapping[str, object],
    *,
    bar_number: int,
    event_at_us: int,
    relative_activity: float | None,
    previous_close: float,
) -> dict[str, JsonValue]:
    values = {name: _number(bar, name) for name in ("open", "high", "low", "close", "volume")}
    if any(value is None for value in values.values()):
        raise SessionProjectionError("complete bar is missing finite OHLCV")
    opening = cast(float, values["open"])
    high = cast(float, values["high"])
    low = cast(float, values["low"])
    close = cast(float, values["close"])
    volume = cast(float, values["volume"])
    if (
        min(opening, high, low, close) <= 0.0
        or volume < 0.0
        or high < max(opening, close, low)
        or low > min(opening, close, high)
    ):
        raise SessionProjectionError("complete bar OHLCV is invalid")
    width = high - low
    true_range_bps = (
        10_000.0
        * max(width, abs(high - previous_close), abs(low - previous_close))
        / previous_close
    )
    return_bps = 10_000.0 * (close / previous_close - 1.0)
    if width > 1e-12:
        upper_wick = (high - max(opening, close)) / width
        lower_wick = (min(opening, close) - low) / width
    else:
        upper_wick = 0.0
        lower_wick = 0.0
    return {
        "bar_number": bar_number,
        "event_at_us": event_at_us,
        "open": opening,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "historical_relative_activity": relative_activity,
        "return_bps": return_bps,
        "true_range_bps": true_range_bps,
        "upper_wick_fraction": min(max(upper_wick, 0.0), 1.0),
        "lower_wick_fraction": min(max(lower_wick, 0.0), 1.0),
    }


def _prefix_payload(
    bar_row: sqlite3.Row,
    prior_row: sqlite3.Row | None,
    baseline_row: sqlite3.Row | None,
) -> dict[str, JsonValue]:
    bar = _event_payload(bar_row)
    session = bar.get("session")
    bar_number = _integer(bar, "bar_number")
    if not isinstance(session, str) or bar_number is None or not 1 <= bar_number <= 78:
        raise SessionProjectionError("five-minute bar session identity is invalid")
    prior = None if prior_row is None else _event_payload(prior_row)
    baseline = None if baseline_row is None else _event_payload(baseline_row)
    prior_complete = bar_number == 1 or (
        prior is not None and prior.get("source_completeness") == "complete"
    )
    current_complete = bar.get("source_completeness") == "complete"
    volume = _number(bar, "volume") if current_complete else None
    history_count, relative_activity = _relative_activity(
        baseline, bar_number=bar_number, volume=volume
    )
    complete = current_complete and prior_complete
    prior_accumulator = _mapping(prior.get("accumulator", {})) if prior is not None else {}
    prior_activity_sum = _number(prior_accumulator, "activity_sum")
    ready = relative_activity is not None and (
        prior is None or (prior.get("activity_ready") is True and prior_activity_sum is not None)
    )
    prior_trailing = prior.get("trailing_bars", []) if prior is not None else []
    prior_volumes = prior.get("session_volumes", []) if prior is not None else []
    if not isinstance(prior_trailing, list) or not isinstance(prior_volumes, list):
        raise SessionProjectionError("prior session prefix is invalid")
    baseline_session = None if baseline is None else baseline.get("session")
    if not isinstance(baseline_session, str):
        baseline_session = None
    baseline_realised = None if baseline is None else _number(baseline, "realised_volatility_20d")
    payload: dict[str, JsonValue] = {
        "schema_version": 1,
        "session": session,
        "bar_number": bar_number,
        "source_completeness": "complete" if complete else "incomplete",
        "historical_session_count": history_count,
        "historical_relative_activity": relative_activity,
        "activity_ready": ready,
        "current_bar_event_id": str(bar_row["event_id"]),
        "prior_prefix_event_id": None if prior_row is None else str(prior_row["event_id"]),
        "baseline_event_id": None if baseline_row is None else str(baseline_row["event_id"]),
        "baseline_session": baseline_session,
        "prior_session_realised_volatility_20d": baseline_realised,
        "derived_after_source_sequence": int(bar_row["derived_after_source_sequence"]),
    }
    if not complete:
        payload["trailing_bars"] = ()
        payload["session_volumes"] = ()
        payload["accumulator"] = {}
        return payload
    previous_close = (
        _number(prior_accumulator, "last_close") if prior is not None else _number(bar, "open")
    )
    if previous_close is None or previous_close <= 0.0:
        raise SessionProjectionError("session prefix previous close is invalid")
    compact = _compact_bar(
        bar,
        bar_number=bar_number,
        event_at_us=int(bar_row["event_at_us"]),
        relative_activity=relative_activity,
        previous_close=previous_close,
    )
    prior_positive_volume = _number(prior_accumulator, "positive_volume_sum") or 0.0
    prior_typical_volume = _number(prior_accumulator, "typical_volume_sum") or 0.0
    current_volume = cast(float, compact["volume"])
    positive_volume = max(current_volume, 0.0)
    typical = (
        cast(float, compact["high"]) + cast(float, compact["low"]) + cast(float, compact["close"])
    ) / 3.0
    positive_volume_sum = prior_positive_volume + positive_volume
    typical_volume_sum = prior_typical_volume + typical * positive_volume
    compact["session_vwap"] = (
        typical_volume_sum / positive_volume_sum if positive_volume_sum > 0.0 else None
    )
    trailing = [*cast(list[JsonValue], prior_trailing), cast(JsonValue, compact)][
        -_MAX_TRAILING_BARS:
    ]
    session_volumes = [
        *cast(list[float], prior_volumes),
        cast(float, compact["volume"]),
    ]
    if len(session_volumes) != bar_number:
        raise SessionProjectionError("session prefix ordinal sequence is incomplete")
    positive = _integer(prior_accumulator, "positive_return_count") or 0
    negative = _integer(prior_accumulator, "negative_return_count") or 0
    zero = _integer(prior_accumulator, "zero_return_count") or 0
    return_bps = cast(float, compact["return_bps"])
    positive += int(return_bps > 0.0)
    negative += int(return_bps < 0.0)
    zero += int(return_bps == 0.0)
    payload["trailing_bars"] = cast(JsonValue, tuple(trailing))
    payload["session_volumes"] = tuple(session_volumes)
    payload["accumulator"] = cast(
        JsonValue,
        {
            "bar_count": bar_number,
            "session_open": (
                cast(float, compact["open"])
                if prior is None
                else cast(float, prior_accumulator["session_open"])
            ),
            "session_high": max(
                cast(float, compact["high"]),
                cast(float, prior_accumulator.get("session_high", compact["high"])),
            ),
            "session_low": min(
                cast(float, compact["low"]),
                cast(float, prior_accumulator.get("session_low", compact["low"])),
            ),
            "last_close": cast(float, compact["close"]),
            "activity_sum": (
                (0.0 if prior is None else cast(float, prior_activity_sum))
                + cast(float, relative_activity)
                if ready
                else None
            ),
            "range_sum": (_number(prior_accumulator, "range_sum") or 0.0)
            + cast(float, compact["true_range_bps"]),
            "travel_sum": (_number(prior_accumulator, "travel_sum") or 0.0) + abs(return_bps),
            "return_sum": (_number(prior_accumulator, "return_sum") or 0.0) + return_bps,
            "width_sum": (_number(prior_accumulator, "width_sum") or 0.0)
            + cast(float, compact["high"])
            - cast(float, compact["low"]),
            "positive_return_count": positive,
            "negative_return_count": negative,
            "zero_return_count": zero,
            "positive_volume_sum": positive_volume_sum,
            "typical_volume_sum": typical_volume_sum,
        },
    )
    return payload


def _insert_derived(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    instrument_id: str,
    event_kind: str,
    event_at_us: int,
    received_at_us: int,
    connection_generation: int,
    derived_after_source_sequence: int,
    payload: Mapping[str, JsonValue],
    inputs: Sequence[tuple[str, str]],
) -> str:
    payload_json = canonical_json_bytes(cast(JsonValue, payload)).decode()
    event_id = _hash(
        cast(
            JsonValue,
            {
                "run_id": run_id,
                "instrument_id": instrument_id,
                "feed_kind": "bars",
                "event_kind": event_kind,
                "event_at_us": event_at_us,
                "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
            },
        )
    )
    connection.execute("SAVEPOINT stocker_session_derived_event")
    try:
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, event_at_us, "
            "received_at_us, connection_generation, payload_json, payload_sha256) "
            "VALUES (?, ?, NULL, ?, ?, 'bars', ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                run_id,
                derived_after_source_sequence,
                instrument_id,
                event_kind,
                event_at_us,
                received_at_us,
                connection_generation,
                payload_json,
                hashlib.sha256(payload_json.encode()).hexdigest(),
            ),
        )
        connection.executemany(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) VALUES (?, ?, ?, ?, ?)",
            (
                (event_id, input_id, ordinal, role, received_at_us)
                for ordinal, (input_id, role) in enumerate(inputs)
            ),
        )
    except BaseException:
        connection.execute("ROLLBACK TO SAVEPOINT stocker_session_derived_event")
        connection.execute("RELEASE SAVEPOINT stocker_session_derived_event")
        raise
    connection.execute("RELEASE SAVEPOINT stocker_session_derived_event")
    return event_id


def _project_prefix(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    bar_row: sqlite3.Row,
) -> int:
    bar = _event_payload(bar_row)
    session = bar.get("session")
    bar_number = _integer(bar, "bar_number")
    if not isinstance(session, str) or bar_number is None:
        raise SessionProjectionError("five-minute bar lacks a session identity")
    prior = _prior_prefix(
        connection,
        run_id=run_id,
        instrument_id=str(bar_row["instrument_id"]),
        session=session,
        bar_number=bar_number,
    )
    if bar_number > 1 and prior is None:
        return 0
    baseline = _latest_baseline(
        connection,
        run_id=run_id,
        instrument_id=str(bar_row["instrument_id"]),
        session=session,
    )
    payload = _prefix_payload(bar_row, prior, baseline)
    inputs = [(str(bar_row["event_id"]), "constituent")]
    if prior is not None:
        inputs.append((str(prior["event_id"]), "prior_receipt"))
    if baseline is not None:
        inputs.append((str(baseline["event_id"]), "context"))
    watermark = max(
        (
            int(bar_row["derived_after_source_sequence"]),
            *(
                int(row["derived_after_source_sequence"])
                for row in (prior, baseline)
                if row is not None
            ),
        )
    )
    _insert_derived(
        connection,
        run_id=run_id,
        instrument_id=str(bar_row["instrument_id"]),
        event_kind="bar_5m_session_prefix",
        event_at_us=int(bar_row["event_at_us"]),
        received_at_us=max(
            (
                int(bar_row["received_at_us"]),
                *(int(row["received_at_us"]) for row in (prior, baseline) if row is not None),
            )
        ),
        connection_generation=max(
            (
                int(bar_row["connection_generation"]),
                *(
                    int(row["connection_generation"])
                    for row in (prior, baseline)
                    if row is not None
                ),
            )
        ),
        derived_after_source_sequence=watermark,
        payload=payload,
        inputs=inputs,
    )
    return 1


def _baseline_payload(
    final_prefix: Mapping[str, object], prior: Mapping[str, object] | None
) -> dict[str, JsonValue]:
    session = final_prefix.get("session")
    volumes = final_prefix.get("session_volumes")
    accumulator = _mapping(final_prefix.get("accumulator", {}))
    if (
        not isinstance(session, str)
        or final_prefix.get("source_completeness") != "complete"
        or _integer(accumulator, "bar_count") != _BARS_PER_SESSION
        or not isinstance(volumes, list)
        or len(volumes) != _BARS_PER_SESSION
    ):
        raise SessionProjectionError("final prefix cannot establish a complete session")
    prior_count = 0
    prior_sums = [0.0] * _BARS_PER_SESSION
    prior_closes: list[float] = []
    prior_id: str | None = None
    if prior is not None:
        prior_count = _integer(prior, "complete_session_count") or 0
        sums = prior.get("ordinal_volume_sums")
        closes = prior.get("session_closes")
        prior_id_value = prior.get("event_id")
        if not isinstance(sums, list) or len(sums) != 78 or not isinstance(closes, list):
            raise SessionProjectionError("prior baseline payload is invalid")
        prior_sums = [float(value) for value in sums]
        prior_closes = [float(value) for value in closes]
        prior_id = prior_id_value if isinstance(prior_id_value, str) else None
    volume_values = [float(value) for value in volumes]
    if any(not math.isfinite(value) or value < 0.0 for value in volume_values):
        raise SessionProjectionError("session volume baseline contains invalid volume")
    close = _number(accumulator, "last_close")
    if close is None or close <= 0.0:
        raise SessionProjectionError("session close is invalid")
    closes = [*prior_closes, close][-_REALIZED_VOLATILITY_RETURNS - 1 :]
    realised: float | None = None
    if len(closes) == _REALIZED_VOLATILITY_RETURNS + 1:
        returns = [
            math.log(current / previous)
            for previous, current in zip(closes, closes[1:], strict=False)
        ]
        realised = statistics.stdev(returns) * math.sqrt(252.0)
    final_prefix_id = final_prefix.get("event_id")
    if not isinstance(final_prefix_id, str):
        final_prefix_id = None
    return {
        "schema_version": 1,
        "session": session,
        "complete_session_count": prior_count + 1,
        "ordinal_volume_sums": tuple(
            prior_value + current
            for prior_value, current in zip(prior_sums, volume_values, strict=True)
        ),
        "session_closes": tuple(closes),
        "realised_volatility_20d": realised,
        "prior_baseline_event_id": prior_id,
        "final_prefix_event_id": final_prefix_id,
    }


def _project_baseline(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    prefix_row: sqlite3.Row,
) -> int:
    prefix = dict(_event_payload(prefix_row))
    if prefix.get("source_completeness") != "complete":
        return 0
    prefix["event_id"] = str(prefix_row["event_id"])
    prior_row = _latest_baseline(
        connection,
        run_id=run_id,
        instrument_id=str(prefix_row["instrument_id"]),
        session=cast(str, prefix["session"]),
    )
    prior = None if prior_row is None else dict(_event_payload(prior_row))
    if prior is not None and prior_row is not None:
        prior["event_id"] = str(prior_row["event_id"])
    payload = _baseline_payload(prefix, prior)
    inputs = [(str(prefix_row["event_id"]), "constituent")]
    if prior_row is not None:
        inputs.append((str(prior_row["event_id"]), "prior_receipt"))
    prior_watermark = (
        int(prior_row["derived_after_source_sequence"])
        if prior_row is not None
        else int(prefix_row["derived_after_source_sequence"])
    )
    watermark = max(int(prefix_row["derived_after_source_sequence"]), prior_watermark)
    _insert_derived(
        connection,
        run_id=run_id,
        instrument_id=str(prefix_row["instrument_id"]),
        event_kind="session_volume_baseline",
        event_at_us=int(prefix_row["event_at_us"]),
        received_at_us=max(
            (
                int(prefix_row["received_at_us"]),
                *((int(prior_row["received_at_us"]),) if prior_row is not None else ()),
            )
        ),
        connection_generation=max(
            (
                int(prefix_row["connection_generation"]),
                *((int(prior_row["connection_generation"]),) if prior_row is not None else ()),
            )
        ),
        derived_after_source_sequence=watermark,
        payload=payload,
        inputs=inputs,
    )
    return 1


def project_required_session_receipts(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    requirements: Sequence[MarketDataRequirement],
    after_source_sequence: int = 0,
    limit: int = 256,
) -> int:
    """Materialize bounded prefix/baseline receipts required by active plugins."""

    if not 1 <= limit <= _MAX_PROJECTION_ROWS:
        raise ValueError("session receipt limit must be between 1 and 10,000")
    instruments = sorted(
        {
            requirement.instrument_id
            for requirement in requirements
            if requirement.feed_kind == "bars" and requirement.event_kind == "bar_5m_session_prefix"
        }
    )
    inserted = 0
    for instrument_id in instruments:
        pending = connection.execute(
            "SELECT bar.* FROM market_events bar "
            "WHERE bar.run_id=? AND bar.instrument_id=? AND bar.feed_kind='bars' "
            "AND bar.event_kind='bar_5m' "
            "AND json_extract(bar.payload_json, '$.first_source_sequence')>? "
            "AND NOT EXISTS (SELECT 1 FROM market_event_derivations derivation "
            "JOIN market_events receipt ON receipt.event_id=derivation.derived_event_id "
            "WHERE derivation.input_event_id=bar.event_id "
            "AND derivation.input_role='constituent' "
            "AND receipt.event_kind='bar_5m_session_prefix') "
            "ORDER BY bar.event_at_us, bar.event_id LIMIT ?",
            (run_id, instrument_id, after_source_sequence, limit - inserted),
        ).fetchall()
        for bar_row in pending:
            if inserted >= limit:
                break
            inserted += _project_prefix(connection, run_id=run_id, bar_row=bar_row)
        if inserted >= limit:
            break
        finals = connection.execute(
            "SELECT prefix.* FROM market_events prefix "
            "WHERE prefix.run_id=? AND prefix.instrument_id=? AND prefix.feed_kind='bars' "
            "AND prefix.event_kind='bar_5m_session_prefix' "
            "AND json_extract(prefix.payload_json, '$.bar_number')=78 "
            "AND json_extract(prefix.payload_json, '$.source_completeness')='complete' "
            "AND NOT EXISTS (SELECT 1 FROM market_event_derivations derivation "
            "JOIN market_events baseline ON baseline.event_id=derivation.derived_event_id "
            "WHERE derivation.input_event_id=prefix.event_id "
            "AND derivation.input_role='constituent' "
            "AND baseline.event_kind='session_volume_baseline') "
            "ORDER BY prefix.event_at_us, prefix.event_id LIMIT ?",
            (run_id, instrument_id, limit - inserted),
        ).fetchall()
        for prefix_row in finals:
            if inserted >= limit:
                break
            inserted += _project_baseline(connection, run_id=run_id, prefix_row=prefix_row)
    return inserted


__all__ = ["SessionProjectionError", "project_required_session_receipts"]
