"""Append-only, research-only prospective forecast and settlement records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REQUIRED_FORECAST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "git_sha",
        "contract_hash",
        "model_version",
        "data_snapshot_hash",
        "feature_schema_version",
        "forecast_id",
        "forecast_creation_timestamp",
        "forecast_effective_session",
        "stock_id",
        "loop_id",
        "orientation",
        "horizon",
        "model_name",
        "p_next_payoff_positive",
        "p_edge_positive",
        "p_edge_active",
        "p_change_now",
        "p_on_next",
        "p_off_next",
        "p_survive_horizon",
        "posterior_mean_net_bps",
        "posterior_lower_bound_net_bps",
        "posterior_run_length_mean",
        "edge_state",
        "reason_codes",
        "independent_session_support",
        "independent_stock_support",
        "effective_sample_size",
        "frozen_feature_values",
        "feature_availability_timestamps",
        "feature_max_availability_timestamp",
        "forecast_freeze_timestamp",
    }
)
REQUIRED_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "outcome_id",
        "forecast_id",
        "target_session",
        "target_lead_sessions",
        "settlement_timestamp",
        "target_robust_net_bps",
    }
)


def _json_default(value: object) -> object:
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported immutable-ledger value: {type(value).__name__}")


def _canonical(record: Mapping[str, object]) -> str:
    return (
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )


def _timestamp(value: object, field: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return result


def _contains_hindsight_or_episode(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            "hindsight" in str(key).lower()
            or "episode" in str(key).lower()
            or _contains_hindsight_or_episode(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_hindsight_or_episode(item) for item in value)
    return False


class ProspectiveResearchLedger:
    """Create-only JSON records suitable for prospective, non-executing logging."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.forecast_root = self.root / "forecasts"
        self.outcome_root = self.root / "outcomes"
        self.forecast_root.mkdir(parents=True, exist_ok=True)
        self.outcome_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: object, field: str) -> str:
        identifier = str(value)
        if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
            raise ValueError(f"unsafe {field}")
        return identifier

    @staticmethod
    def _write_create_only(path: Path, record: Mapping[str, object]) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(record))

    def append_forecast(self, record: Mapping[str, object]) -> Path:
        missing = sorted(REQUIRED_FORECAST_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing prospective forecast fields: {missing}")
        freeze = _timestamp(record["forecast_freeze_timestamp"], "forecast freeze")
        creation = _timestamp(record["forecast_creation_timestamp"], "forecast creation")
        feature_max = _timestamp(
            record["feature_max_availability_timestamp"], "feature availability"
        )
        if creation != freeze:
            raise ValueError("forecast creation must equal immutable freeze timestamp")
        if feature_max >= freeze:
            raise ValueError("feature availability must be strictly before forecast freeze")
        if _contains_hindsight_or_episode(record["frozen_feature_values"]):
            raise ValueError("hindsight or episode labels are forbidden in forecast features")
        identifier = self._safe_id(record["forecast_id"], "forecast_id")
        payload = dict(record)
        payload.update(
            {
                "research_only": True,
                "execution_enabled": False,
                "order_placement_enabled": False,
                "broker_connection_enabled": False,
                "position_management_enabled": False,
            }
        )
        path = self.forecast_root / f"{identifier}.json"
        self._write_create_only(path, payload)
        return path

    def append_outcome(self, record: Mapping[str, object]) -> Path:
        missing = sorted(REQUIRED_OUTCOME_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing prospective outcome fields: {missing}")
        forecast_id = self._safe_id(record["forecast_id"], "forecast_id")
        forecast_path = self.forecast_root / f"{forecast_id}.json"
        if not forecast_path.is_file():
            raise ValueError(f"unknown forecast: {forecast_id}")
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        settlement = _timestamp(record["settlement_timestamp"], "settlement timestamp")
        freeze = _timestamp(forecast["forecast_freeze_timestamp"], "forecast freeze")
        if settlement <= freeze:
            raise ValueError("outcome settlement must occur after forecast freeze")
        identifier = self._safe_id(record["outcome_id"], "outcome_id")
        payload = dict(record)
        payload.update(
            {
                "source_run_id": forecast["run_id"],
                "contract_hash": forecast["contract_hash"],
                "model_name": forecast["model_name"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        path = self.outcome_root / f"{identifier}.json"
        self._write_create_only(path, payload)
        return path


__all__ = ["ProspectiveResearchLedger"]
