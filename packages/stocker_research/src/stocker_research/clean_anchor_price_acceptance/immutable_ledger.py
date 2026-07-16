"""Create-only, execution-free prospective acceptance logging."""

from __future__ import annotations

import json
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REQUIRED_FORECAST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "opportunity_id",
        "event_lineage_id",
        "symbol",
        "session",
        "loop_id",
        "orientation",
        "frozen_direction",
        "anchor_timestamp",
        "anchor_reference_price",
        "static_anchor_veto_score",
        "static_anchor_veto_pass",
        "static_anchor_veto_reason_codes",
        "checkpoint_timestamp",
        "checkpoint_open",
        "checkpoint_high",
        "checkpoint_low",
        "checkpoint_close",
        "signed_close_return_bps",
        "favourable_excursion_bps",
        "adverse_excursion_bps",
        "acceptance_balance_bps",
        "price_acceptance_pass",
        "predicted_remaining_range_bps",
        "range_permission_pass",
        "next_entry_timestamp",
        "original_terminal_timestamp",
        "variant_decisions",
        "feature_availability_timestamps",
        "training_cutoff",
        "forecast_freeze_timestamp",
    }
)
REQUIRED_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {"outcome_id", "opportunity_id", "settlement_timestamp", "net_payoff_bps"}
)
_FORBIDDEN = {"mfe", "mae", "realised_loop", "realized_loop", "route_completion", "payoff"}


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


def _has_forbidden_feature(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN):
                return True
            if _has_forbidden_feature(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_forbidden_feature(item) for item in value)
    return False


class ProspectiveAcceptanceLedger:
    """Immutable research logger; it has no execution-facing methods."""

    def __init__(self, root: Path, *, opened_periods: Set[int]) -> None:
        self.root = Path(root)
        self.opened_periods = frozenset(int(value) for value in opened_periods)
        self.forecast_root = self.root / "forecasts"
        self.outcome_root = self.root / "outcomes"
        self.forecast_root.mkdir(parents=True, exist_ok=True)
        self.outcome_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create(path: Path, record: Mapping[str, object]) -> None:
        payload = json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")

    def append_forecast(self, record: Mapping[str, object], *, holdout: bool) -> Path:
        missing = sorted(REQUIRED_FORECAST_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing forecast fields: {missing}")
        if _has_forbidden_feature(record.get("feature_values", {})):
            raise ValueError("future outcome diagnostic is forbidden in forecast features")
        session = str(record["session"])
        if holdout and int(session[:4]) in self.opened_periods:
            raise ValueError("opened period is forbidden in prospective holdout mode")
        if holdout and not str(record["data_snapshot_hash"]):
            raise ValueError("a new data snapshot hash is required")
        anchor = _timestamp(record["anchor_timestamp"], "anchor timestamp")
        checkpoint = _timestamp(record["checkpoint_timestamp"], "checkpoint timestamp")
        freeze = _timestamp(record["forecast_freeze_timestamp"], "forecast freeze")
        entry = _timestamp(record["next_entry_timestamp"], "next entry")
        cutoff = _timestamp(record["training_cutoff"], "training cutoff")
        if checkpoint + pd.Timedelta(minutes=5) != freeze:
            raise ValueError("forecast must freeze only after checkpoint close")
        if entry != freeze:
            raise ValueError("entry must be the next provider open at freeze")
        if cutoff >= anchor:
            raise ValueError("training cutoff must precede anchor")
        availability = record["feature_availability_timestamps"]
        if not isinstance(availability, Mapping):
            raise ValueError("feature availability timestamps must be a mapping")
        if any(
            _timestamp(value, "feature availability") > freeze for value in availability.values()
        ):
            raise ValueError("feature availability occurs after forecast freeze")
        identifier = _identifier(record["opportunity_id"], "opportunity_id")
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
        path = self.forecast_root / f"{identifier}.json"
        self._create(path, payload)
        return path

    def append_outcome(self, record: Mapping[str, object]) -> Path:
        missing = sorted(REQUIRED_OUTCOME_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing outcome fields: {missing}")
        opportunity_id = _identifier(record["opportunity_id"], "opportunity_id")
        forecast_path = self.forecast_root / f"{opportunity_id}.json"
        if not forecast_path.is_file():
            raise ValueError("outcome refers to an unknown forecast")
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        settlement = _timestamp(record["settlement_timestamp"], "settlement timestamp")
        freeze = _timestamp(forecast["forecast_freeze_timestamp"], "forecast freeze")
        if settlement <= freeze:
            raise ValueError("outcome must settle after forecast freeze")
        identifier = _identifier(record["outcome_id"], "outcome_id")
        payload = dict(record)
        payload.update(
            {
                "run_id": forecast["run_id"],
                "contract_hash": forecast["contract_hash"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        path = self.outcome_root / f"{identifier}.json"
        self._create(path, payload)
        return path
