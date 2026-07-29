"""Create-only prospective checkpoint forecasts and later settlements."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
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
        "event_lineage_id",
        "opportunity_id",
        "anchor_id",
        "stock",
        "session",
        "decision_timestamp",
        "checkpoint_timestamp",
        "checkpoint_type",
        "bars_since_anchor",
        "bars_remaining",
        "current_state",
        "state_history",
        "clock_phase",
        "compatible_loop_set",
        "loop_posterior",
        "good_loop_mass",
        "bad_loop_mass",
        "unknown_loop_mass",
        "entropy",
        "competitor_eliminations",
        "decision_state",
        "reason_codes",
        "freeze_timestamp",
        "feature_availability_timestamps",
        "training_cutoff",
    }
)
REQUIRED_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "outcome_id",
        "forecast_id",
        "event_lineage_id",
        "settlement_timestamp",
        "constant_terminal_net_bps",
    }
)


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


def _safe_identifier(value: object, field: str) -> str:
    identifier = str(value)
    if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
        raise ValueError(f"unsafe {field}")
    return identifier


def _contains_forbidden_hindsight(record: Mapping[str, object]) -> bool:
    return any(
        "hindsight" in str(key).lower()
        or "realised_loop" in str(key).lower()
        or "realized_loop" in str(key).lower()
        or "episode_label" in str(key).lower()
        for key in record
    )


class ProspectiveCompetitorLedger:
    """Research-only immutable logger; it has no execution interface."""

    def __init__(self, root: Path, *, opened_periods: Set[int]) -> None:
        self.root = Path(root)
        self.opened_periods = frozenset(int(value) for value in opened_periods)
        self.forecasts = self.root / "forecasts"
        self.outcomes = self.root / "outcomes"
        self.forecasts.mkdir(parents=True, exist_ok=True)
        self.outcomes.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create(path: Path, payload: Mapping[str, object]) -> None:
        canonical = (
            json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical)

    def append_forecast(self, record: Mapping[str, object], *, holdout: bool) -> Path:
        missing = sorted(REQUIRED_FORECAST_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing forecast fields: {missing}")
        if _contains_forbidden_hindsight(record):
            raise ValueError("hindsight, episode, and realised-loop fields are forbidden")
        session = str(record["session"])
        if holdout and int(session[:4]) in self.opened_periods:
            raise ValueError("opened period is forbidden in prospective holdout mode")
        checkpoint = _timestamp(record["checkpoint_timestamp"], "checkpoint timestamp")
        freeze = _timestamp(record["freeze_timestamp"], "freeze timestamp")
        decision = _timestamp(record["decision_timestamp"], "decision timestamp")
        if freeze != checkpoint or decision > freeze:
            raise ValueError("forecast must freeze at its causal checkpoint")
        availability = record["feature_availability_timestamps"]
        if not isinstance(availability, Sequence) or isinstance(availability, (str, bytes)):
            raise ValueError("feature availability timestamps must be a sequence")
        for value in availability:
            if _timestamp(value, "feature availability") > freeze:
                raise ValueError("feature availability occurs after forecast freeze")
        posterior = record["loop_posterior"]
        if not isinstance(posterior, Mapping):
            raise ValueError("loop posterior must be a mapping")
        total = sum(float(value) for value in posterior.values())
        if not np.isclose(total, 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("loop posterior must normalise to one")
        identifier = _safe_identifier(record["forecast_id"], "forecast_id")
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
        path = self.forecasts / f"{identifier}.json"
        self._create(path, payload)
        return path

    def append_outcome(self, record: Mapping[str, object]) -> Path:
        missing = sorted(REQUIRED_OUTCOME_FIELDS - set(record))
        if missing:
            raise ValueError(f"missing outcome fields: {missing}")
        forecast_id = _safe_identifier(record["forecast_id"], "forecast_id")
        forecast_path = self.forecasts / f"{forecast_id}.json"
        if not forecast_path.is_file():
            raise ValueError("outcome refers to an unknown forecast")
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        if str(record["event_lineage_id"]) != str(forecast["event_lineage_id"]):
            raise ValueError("outcome event lineage differs from forecast")
        if _timestamp(record["settlement_timestamp"], "settlement timestamp") <= _timestamp(
            forecast["freeze_timestamp"], "freeze timestamp"
        ):
            raise ValueError("settlement must occur after forecast freeze")
        identifier = _safe_identifier(record["outcome_id"], "outcome_id")
        payload = dict(record)
        payload.update(
            {
                "source_run_id": forecast["run_id"],
                "contract_hash": forecast["contract_hash"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        path = self.outcomes / f"{identifier}.json"
        self._create(path, payload)
        return path
