"""Create-only, execution-free prospective logging for fixed latency."""

from __future__ import annotations

import json
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REQUIRED_OPPORTUNITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "source_run_id",
        "source_artifact_hash",
        "source_opportunity_hash",
        "opportunity_id",
        "anchor_id",
        "event_lineage_id",
        "symbol",
        "session",
        "loop_id",
        "orientation",
        "frozen_direction",
        "anchor_timestamp",
        "t0_entry_timestamp",
        "t0_entry_price",
        "expected_t1_timestamp",
        "original_terminal_timestamp",
        "provider_data_hash",
        "forecast_freeze_timestamp",
    }
)
REQUIRED_TIMING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "timing_id",
        "opportunity_id",
        "t1_entry_timestamp",
        "t1_entry_price",
        "t1_availability",
        "unavailability_reason",
        "data_availability_timestamp",
        "settlement_command_identity",
    }
)
REQUIRED_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "outcome_id",
        "opportunity_id",
        "settlement_timestamp",
        "t0_gross_return_bps",
        "t0_cost_bps",
        "t0_net_return_bps",
        "t1_gross_return_bps",
        "t1_cost_bps",
        "t1_net_return_bps",
        "paired_difference_bps",
    }
)
_FORBIDDEN_FEATURE_TOKENS = {
    "mfe",
    "mae",
    "hindsight",
    "episode",
    "payoff",
    "realised",
    "realized",
    "future",
    "route_completion",
}


def _json_default(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported immutable value: {type(value).__name__}")


def _timestamp(value: object, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _identifier(value: object, field: str) -> str:
    identifier = str(value)
    if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
        raise ValueError(f"unsafe {field}")
    return identifier


def _contains_forbidden_feature(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
                return True
            if _contains_forbidden_feature(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_feature(item) for item in value)
    return False


class ProspectiveLatencyLedger:
    """Immutable research logger with no order- or position-facing methods."""

    def __init__(self, root: Path, *, opened_periods: Set[int]) -> None:
        self.root = Path(root)
        self.opened_periods = frozenset(int(value) for value in opened_periods)
        self.opportunity_root = self.root / "opportunities"
        self.timing_root = self.root / "timings"
        self.outcome_root = self.root / "outcomes"

    @staticmethod
    def _create(path: Path, record: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")

    def validate_opportunity(
        self, record: Mapping[str, object], *, holdout: bool
    ) -> dict[str, object]:
        missing = sorted(REQUIRED_OPPORTUNITY_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing opportunity fields: {missing}")
        if _contains_forbidden_feature(record.get("feature_values", {})):
            raise ValueError("future outcome or hindsight feature is forbidden")
        session = str(record["session"])
        if holdout and int(session[:4]) in self.opened_periods:
            raise ValueError("opened period is forbidden in prospective holdout mode")
        if holdout and not str(record["data_snapshot_hash"]):
            raise ValueError("a new data snapshot hash is required")
        anchor = _timestamp(record["anchor_timestamp"], "anchor timestamp")
        t0 = _timestamp(record["t0_entry_timestamp"], "T0 entry timestamp")
        expected = _timestamp(record["expected_t1_timestamp"], "expected T1 timestamp")
        terminal = _timestamp(record["original_terminal_timestamp"], "terminal timestamp")
        freeze = _timestamp(record["forecast_freeze_timestamp"], "forecast freeze")
        if expected != t0 + pd.Timedelta(minutes=5):
            raise ValueError("expected T1 must be exact T0 plus five minutes")
        if freeze > t0:
            raise ValueError("opportunity freeze cannot follow the frozen T0 fill")
        if expected >= terminal:
            raise ValueError("expected T1 must precede the original terminal")
        if t0 <= anchor:
            raise ValueError("T0 must follow the source anchor")
        if int(str(record["frozen_direction"])) not in (-1, 1):
            raise ValueError("ambiguous frozen direction")
        payload = dict(record)
        payload.update(
            {
                "research_only": True,
                "execution_enabled": False,
                "broker_connection_enabled": False,
                "order_placement_enabled": False,
                "position_management_enabled": False,
                "existing_exit_management_enabled": False,
                "deployment_enabled": False,
            }
        )
        return payload

    def append_opportunity(self, record: Mapping[str, object], *, holdout: bool) -> Path:
        payload = self.validate_opportunity(record, holdout=holdout)
        identifier = _identifier(record["opportunity_id"], "opportunity_id")
        path = self.opportunity_root / f"{identifier}.json"
        self._create(path, payload)
        return path

    def append_timing(self, record: Mapping[str, object]) -> Path:
        missing = sorted(REQUIRED_TIMING_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing timing fields: {missing}")
        opportunity_id = _identifier(record["opportunity_id"], "opportunity_id")
        opportunity_path = self.opportunity_root / f"{opportunity_id}.json"
        if not opportunity_path.is_file():
            raise ValueError("timing refers to an unknown opportunity")
        opportunity = json.loads(opportunity_path.read_text(encoding="utf-8"))
        expected = _timestamp(opportunity["expected_t1_timestamp"], "expected T1 timestamp")
        availability = str(record["t1_availability"])
        if availability == "available":
            actual = _timestamp(record["t1_entry_timestamp"], "T1 entry timestamp")
            if actual != expected:
                raise ValueError("timing does not use the exact expected T1")
            price = float(str(record["t1_entry_price"]))
            if not np.isfinite(price) or price <= 0.0:
                raise ValueError("available T1 price must be positive")
            if record["unavailability_reason"] not in (None, ""):
                raise ValueError("available timing cannot have an unavailability reason")
        elif record["t1_entry_timestamp"] is not None or record["t1_entry_price"] is not None:
            raise ValueError("unavailable timing cannot contain an entry")
        data_available = _timestamp(
            record["data_availability_timestamp"], "data availability timestamp"
        )
        if data_available < expected:
            raise ValueError("T1 data became available before its exact timestamp")
        payload = dict(record)
        payload.update(
            {
                "run_id": opportunity["run_id"],
                "contract_hash": opportunity["contract_hash"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        path = self.timing_root / f"{opportunity_id}.json"
        self._create(path, payload)
        return path

    def append_outcome(self, record: Mapping[str, object]) -> Path:
        missing = sorted(REQUIRED_OUTCOME_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing outcome fields: {missing}")
        opportunity_id = _identifier(record["opportunity_id"], "opportunity_id")
        opportunity_path = self.opportunity_root / f"{opportunity_id}.json"
        timing_path = self.timing_root / f"{opportunity_id}.json"
        if not opportunity_path.is_file() or not timing_path.is_file():
            raise ValueError("outcome requires exact opportunity and timing records")
        opportunity = json.loads(opportunity_path.read_text(encoding="utf-8"))
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if timing["t1_availability"] != "available":
            raise ValueError("paired outcome requires an available T1 timing")
        settlement = _timestamp(record["settlement_timestamp"], "settlement timestamp")
        terminal = _timestamp(opportunity["original_terminal_timestamp"], "terminal timestamp")
        if settlement < terminal:
            raise ValueError("outcome cannot settle before the original terminal")
        expected_delta = float(str(record["t1_net_return_bps"])) - float(
            str(record["t0_net_return_bps"])
        )
        if not np.isclose(
            float(str(record["paired_difference_bps"])),
            expected_delta,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("paired outcome difference does not reconcile")
        identifier = _identifier(record["outcome_id"], "outcome_id")
        payload = dict(record)
        payload.update(
            {
                "run_id": opportunity["run_id"],
                "contract_hash": opportunity["contract_hash"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        path = self.outcome_root / f"{identifier}.json"
        self._create(path, payload)
        return path
