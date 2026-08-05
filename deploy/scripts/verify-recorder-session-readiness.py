#!/usr/bin/env python3
"""Read-only market-session readiness check for the Stocker recorder."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _timestamp(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label}_invalid")
    return parsed.astimezone(UTC)


def _fresh(value: datetime, *, now: datetime, maximum_age: timedelta) -> bool:
    age = now - value
    return timedelta(0) <= age <= maximum_age


def verify(
    database: Path,
    *,
    now: datetime,
    expected_level1: int,
    expected_bars: int,
    maximum_quote_age: timedelta,
    maximum_bar_age: timedelta,
) -> dict[str, object]:
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        operational = connection.execute(
            """
            SELECT run_id, recorder_generation, state
            FROM recorder_operational_state_v1
            ORDER BY updated_at_utc DESC, recorder_generation DESC
            LIMIT 1
            """
        ).fetchone()
        if operational is None:
            raise ValueError("recorder_operational_state_missing")
        if str(operational["state"]) != "RECORDING_HEALTHY":
            raise ValueError(f"recorder_not_healthy:{operational['state']}")
        run_id = str(operational["run_id"])
        counts = {
            str(row["subscription_kind"]): int(row["used"])
            for row in connection.execute(
                """
                SELECT subscription_kind, COUNT(*) AS used
                FROM web_latest_subscription_state_v0
                WHERE run_id = ? AND status = 'active'
                GROUP BY subscription_kind
                """,
                (run_id,),
            ).fetchall()
        }
        level1_count = counts.get("underlying_level1", 0)
        bar_count = counts.get("underlying_bar", 0)
        if level1_count != expected_level1:
            raise ValueError(
                f"required_level1_subscriptions_missing:{level1_count}/{expected_level1}"
            )
        if bar_count != expected_bars:
            raise ValueError(f"required_bar_subscriptions_missing:{bar_count}/{expected_bars}")
        quote = connection.execute(
            """
            SELECT MAX(received_timestamp_utc) AS latest_quote
            FROM underlying_live_state_v0
            WHERE run_id = ? AND market_data_type = 'live' AND quote_valid = 1
            """,
            (run_id,),
        ).fetchone()
        if quote is None or quote["latest_quote"] is None:
            raise ValueError("current_live_level1_quote_missing")
        quote_at = _timestamp(quote["latest_quote"], label="latest_live_quote")
        if not _fresh(quote_at, now=now, maximum_age=maximum_quote_age):
            raise ValueError("current_live_level1_quote_stale")
        bar = connection.execute(
            """
            SELECT COUNT(*) AS symbol_count, MIN(bar_end_utc) AS slowest_bar
            FROM completed_bar_state_v0
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        assert bar is not None
        if int(bar["symbol_count"]) != expected_bars or bar["slowest_bar"] is None:
            raise ValueError(
                f"required_bar_progress_missing:{int(bar['symbol_count'])}/{expected_bars}"
            )
        slowest_bar_at = _timestamp(bar["slowest_bar"], label="slowest_bar_boundary")
        if not _fresh(slowest_bar_at, now=now, maximum_age=maximum_bar_age):
            raise ValueError("required_bar_progress_stale")
    return {
        "run_id": run_id,
        "recorder_generation": int(operational["recorder_generation"]),
        "level1_subscription_count": level1_count,
        "bar_subscription_count": bar_count,
        "latest_live_quote_at_utc": quote_at.isoformat(),
        "slowest_bar_boundary_utc": slowest_bar_at.isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--now")
    parser.add_argument("--expected-level1", type=int, default=21)
    parser.add_argument("--expected-bars", type=int, default=28)
    parser.add_argument("--maximum-quote-age-seconds", type=int, default=2)
    parser.add_argument("--maximum-bar-age-seconds", type=int, default=600)
    parser.add_argument("--required-distinct-boundaries", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=10)
    arguments = parser.parse_args()
    if (
        arguments.required_distinct_boundaries <= 0
        or arguments.attempts <= 0
        or arguments.interval_seconds <= 0
    ):
        print("recorder_session_readiness:invalid_retry_configuration", file=sys.stderr)
        return 1
    observed_boundaries: list[datetime] = []
    last_error = "required_bar_boundaries_not_observed"
    for attempt in range(arguments.attempts):
        now = datetime.now(UTC) if arguments.now is None else _timestamp(arguments.now, label="now")
        try:
            result = verify(
                arguments.database,
                now=now,
                expected_level1=arguments.expected_level1,
                expected_bars=arguments.expected_bars,
                maximum_quote_age=timedelta(seconds=arguments.maximum_quote_age_seconds),
                maximum_bar_age=timedelta(seconds=arguments.maximum_bar_age_seconds),
            )
        except (OSError, sqlite3.Error, ValueError) as exc:
            last_error = str(exc)
        else:
            boundary = _timestamp(result["slowest_bar_boundary_utc"], label="slowest_bar")
            if not observed_boundaries or boundary > observed_boundaries[-1]:
                observed_boundaries.append(boundary)
            elif boundary < observed_boundaries[-1]:
                last_error = "required_bar_boundary_regressed"
            if len(observed_boundaries) >= arguments.required_distinct_boundaries:
                result["observed_distinct_bar_boundaries"] = len(observed_boundaries)
                print(json.dumps(result, sort_keys=True))
                return 0
            last_error = (
                "required_advancing_bar_boundaries_missing:"
                f"{len(observed_boundaries)}/{arguments.required_distinct_boundaries}"
            )
        if attempt + 1 < arguments.attempts:
            time.sleep(arguments.interval_seconds)
    print(f"recorder_session_readiness:{last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
