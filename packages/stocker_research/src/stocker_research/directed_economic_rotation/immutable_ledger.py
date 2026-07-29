"""Create-only research forecast and later activation outcome records."""

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
        "model_version",
        "forecast_id",
        "forecast_session",
        "forecast_timestamp",
        "target_window_sessions",
        "destination_family",
        "destination_pair",
        "destination_current_economic_state",
        "destination_own_history_features",
        "source_family_state_vector",
        "system_state_features",
        "predicted_activation_probability",
        "activation_base_rate",
        "predicted_lift_over_base",
        "probability_interval_lower",
        "probability_interval_upper",
        "probability_no_activation",
        "probability_multiple_activation",
        "prediction_state",
        "reason_codes",
        "feature_availability_timestamp",
        "training_cutoff",
        "forecast_freeze_timestamp",
    }
)
REQUIRED_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "outcome_id",
        "forecast_id",
        "target_start_session",
        "target_end_session",
        "settlement_timestamp",
        "destination_activation_observed",
        "multiple_activation_flag",
    }
)
_FORBIDDEN_FORECAST_KEYS = {
    "activation_target",
    "destination_activation_observed",
    "first_activation_session",
    "target_episode_ids",
    "episode_net_payoff",
    "payoff_available_after_forecast",
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
    result = pd.Timestamp(str(value))
    if result.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return result.tz_convert("UTC")


def _safe_id(value: object, field: str) -> str:
    identifier = str(value)
    if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
        raise ValueError(f"unsafe {field}")
    return identifier


def _contains_future_target(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if name in _FORBIDDEN_FORECAST_KEYS or "hindsight" in name:
                return True
            if _contains_future_target(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_future_target(item) for item in value)
    return False


class ProspectiveRotationLedger:
    """Execution-free immutable logger for unopened family-rotation sessions."""

    def __init__(self, root: Path, *, opened_periods: Set[int]) -> None:
        self.root = Path(root)
        self.opened_periods = frozenset(int(value) for value in opened_periods)
        self.forecast_root = self.root / "forecasts"
        self.outcome_root = self.root / "outcomes"
        self.forecast_root.mkdir(parents=True, exist_ok=True)
        self.outcome_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create(path: Path, record: Mapping[str, object]) -> None:
        payload = (
            json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)

    def append_forecast(self, record: Mapping[str, object], *, holdout: bool) -> Path:
        missing = sorted(REQUIRED_FORECAST_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing forecast fields: {missing}")
        if _contains_future_target(record):
            raise ValueError("future target or hindsight field is forbidden in forecast")
        session = str(record["forecast_session"])
        if holdout and int(session[:4]) in self.opened_periods:
            raise ValueError("opened period is forbidden in prospective holdout mode")
        freeze = _timestamp(record["forecast_freeze_timestamp"], "forecast freeze")
        forecast = _timestamp(record["forecast_timestamp"], "forecast timestamp")
        feature = _timestamp(record["feature_availability_timestamp"], "feature availability")
        cutoff = _timestamp(record["training_cutoff"], "training cutoff")
        if forecast != freeze:
            raise ValueError("forecast timestamp must equal immutable freeze")
        if feature > freeze:
            raise ValueError("feature availability occurs after forecast freeze")
        if cutoff >= freeze:
            raise ValueError("training cutoff must be strictly before forecast freeze")
        identifier = _safe_id(record["forecast_id"], "forecast_id")
        payload = dict(record)
        payload.update(
            {
                "research_only": True,
                "execution_enabled": False,
                "broker_connection_enabled": False,
                "order_placement_enabled": False,
                "position_management_enabled": False,
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
        forecast_id = _safe_id(record["forecast_id"], "forecast_id")
        forecast_path = self.forecast_root / f"{forecast_id}.json"
        if not forecast_path.is_file():
            raise ValueError("outcome refers to an unknown forecast")
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        settlement = _timestamp(record["settlement_timestamp"], "settlement timestamp")
        freeze = _timestamp(forecast["forecast_freeze_timestamp"], "forecast freeze")
        if settlement <= freeze:
            raise ValueError("outcome must settle after forecast freeze")
        identifier = _safe_id(record["outcome_id"], "outcome_id")
        payload = dict(record)
        payload.update(
            {
                "source_run_id": forecast["run_id"],
                "contract_hash": forecast["contract_hash"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        path = self.outcome_root / f"{identifier}.json"
        self._create(path, payload)
        return path


__all__ = ["ProspectiveRotationLedger"]
