"""Immutable prospective quote-observation plans."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from stocker_execution.ibkr_observability.models import ObservationPlanItem


def _observation_id(event_id: str, observation_type: str, timestamp: datetime) -> str:
    value = f"{event_id}|{observation_type}|{timestamp.isoformat()}"
    return f"ibkr_obs_{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def build_observation_plan(events: list[dict[str, Any]]) -> list[ObservationPlanItem]:
    """Create entry/exit quote observations with the frozen ten-second deadline."""

    output: list[ObservationPlanItem] = []
    for event in sorted(events, key=lambda row: (row["assigned_decision_time"], row["symbol"])):
        decision_timestamp = event["assigned_decision_time"]
        entry_timestamp = event["planned_entry_reference_time"]
        exit_timestamp = event["planned_exit_reference_time"]
        if not all(
            isinstance(timestamp, datetime)
            for timestamp in (decision_timestamp, entry_timestamp, exit_timestamp)
        ):
            raise TypeError("decision, entry, and exit timestamps must be datetimes")
        for observation_type, timestamp_key in (
            ("live_top_of_book_entry_reference", "planned_entry_reference_time"),
            ("live_top_of_book_exit_reference", "planned_exit_reference_time"),
        ):
            timestamp = event[timestamp_key]
            if not isinstance(timestamp, datetime):
                raise TypeError(f"{timestamp_key} must be a datetime")
            output.append(
                ObservationPlanItem(
                    observation_id=_observation_id(
                        str(event["event_id"]), observation_type, timestamp
                    ),
                    event_id=str(event["event_id"]),
                    decision_id=str(event["slate_id"]),
                    decision_timestamp=decision_timestamp,
                    planned_entry_reference_timestamp=entry_timestamp,
                    planned_exit_reference_timestamp=exit_timestamp,
                    planned_observation_timestamp=timestamp,
                    symbol=str(event["symbol"]),
                    con_id=int(event["con_id"]),
                    required_observation_type=observation_type,
                    maximum_collection_delay_seconds=10.0,
                    completion_status="planned",
                )
            )
    return output
