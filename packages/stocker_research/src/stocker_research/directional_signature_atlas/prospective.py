"""Execution-free append-only forecast and settlement ledgers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _condition_fires(feature_row: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    feature = str(condition["feature"])
    if feature not in feature_row or pd.isna(feature_row[feature]):
        return False
    actual = feature_row[feature]
    expected = condition["value"]
    operator = str(condition["operator"])
    if operator == "==":
        return bool(actual == expected)
    if operator == "!=":
        return bool(actual != expected)
    if operator == ">":
        return bool(float(actual) > float(expected))
    if operator == "<":
        return bool(float(actual) < float(expected))
    raise ValueError(f"unsupported prospective condition operator: {operator}")


def build_forecast_record(
    feature_row: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    long_library: Sequence[Mapping[str, Any]],
    short_library: Sequence[Mapping[str, Any]],
    causal_feature_names: Sequence[str],
    forecast_freeze_timestamp: str | pd.Timestamp,
) -> dict[str, Any]:
    """Freeze one execution-free atlas forecast from an outcome-free feature row."""

    decision = pd.Timestamp(str(feature_row["decision_timestamp"]))
    freeze = pd.Timestamp(forecast_freeze_timestamp)
    if decision.tzinfo is None or freeze.tzinfo is None:
        raise ValueError("prospective timestamps must be timezone-aware")
    if freeze > decision:
        raise ValueError("forecast must freeze no later than the completed decision bar")
    entry = pd.Timestamp(
        str(feature_row.get("entry_timestamp", decision + pd.Timedelta(minutes=5)))
    )
    terminal = pd.Timestamp(
        str(feature_row.get("terminal_timestamp", decision + pd.Timedelta(minutes=120)))
    )
    if entry <= decision or terminal <= entry:
        raise ValueError("prospective economic timestamps are invalid")

    def decisions(library: Sequence[Mapping[str, Any]]) -> tuple[dict[str, bool], float]:
        values: dict[str, bool] = {}
        conservative_sum = 0.0
        for entry_payload in library:
            signature = entry_payload["signature"]
            signature_id = str(signature["signature_id"])
            fired = all(
                _condition_fires(feature_row, condition) for condition in signature["conditions"]
            )
            values[signature_id] = fired
            if fired:
                conservative_sum += float(entry_payload.get("conservative_value_bps", 0.0))
        return values, conservative_sum

    long_decisions, long_value = decisions(long_library)
    short_decisions, short_value = decisions(short_library)
    long_votes = sum(long_decisions.values())
    short_votes = sum(short_decisions.values())
    conflict = long_votes > 0 and short_votes > 0
    movement = bool(feature_row.get("movement_permission", False))
    required_features = {
        str(condition["feature"])
        for entry_payload in [*long_library, *short_library]
        for condition in entry_payload["signature"]["conditions"]
    }
    missing_required = sorted(
        feature
        for feature in required_features
        if feature not in feature_row or pd.isna(feature_row[feature])
    )
    state = "NEUTRAL"
    reasons: list[str] = []
    if missing_required:
        reasons.append("required_causal_feature_unavailable")
    if not movement:
        reasons.append("movement_permission_failed")
    elif conflict:
        reasons.append("conflicting_votes")
    elif long_votes and not short_votes and long_value / long_votes > 0.0:
        state = "LONG"
        reasons.append("supported_long_vote")
    elif short_votes and not long_votes and short_value / short_votes > 0.0:
        state = "SHORT"
        reasons.append("supported_short_vote")
    elif not long_votes and not short_votes:
        reasons.append("no_directional_vote")
    else:
        reasons.append("non_positive_conservative_value")

    causal_features = {
        feature: _json_value(feature_row.get(feature)) for feature in causal_feature_names
    }
    availability = {
        feature: _json_value(feature_row.get(f"{feature}__available_at"))
        for feature in causal_feature_names
    }
    return {
        "run_id": str(metadata["run_id"]),
        "git_sha": str(metadata["git_sha"]),
        "contract_hash": str(metadata["contract_sha256"]),
        "data_snapshot_hash": str(metadata["data_snapshot_sha256"]),
        "feature_schema_hash": str(metadata["feature_schema_sha256"]),
        "opportunity_id": str(feature_row["opportunity_id"]),
        "symbol": str(feature_row["symbol"]),
        "session": str(feature_row["session"]),
        "decision_clock": str(feature_row["decision_clock"]),
        "decision_timestamp": decision.isoformat(),
        "entry_timestamp": entry.isoformat(),
        "terminal_timestamp": terminal.isoformat(),
        "causal_features": causal_features,
        "feature_availability_timestamps": availability,
        "movement_permission": movement,
        "long_signature_decisions": long_decisions,
        "short_signature_decisions": short_decisions,
        "long_vote_count": long_votes,
        "short_vote_count": short_votes,
        "conflict_state": conflict,
        "final_atlas_state": state,
        "reason_codes": sorted(set(reasons)),
        "forecast_freeze_timestamp": freeze.isoformat(),
        "research_only": True,
        "execution_enabled": False,
    }


def build_settlement_record(
    outcome_row: Mapping[str, Any],
    *,
    settlement_timestamp: str | pd.Timestamp,
    settlement_code_version: str,
) -> dict[str, Any]:
    """Create a separate immutable settlement without mutating its forecast."""

    return {
        "opportunity_id": str(outcome_row["opportunity_id"]),
        "terminal_timestamp": pd.Timestamp(str(outcome_row["terminal_timestamp"])).isoformat(),
        "gross_long_payoff_bps": float(outcome_row["gross_long_return_bps"]),
        "gross_short_payoff_bps": float(outcome_row["gross_short_return_bps"]),
        "costs_bps": float(outcome_row["round_trip_cost_bps"]),
        "net_long_payoff_bps": float(outcome_row["net_long_return_bps"]),
        "net_short_payoff_bps": float(outcome_row["net_short_return_bps"]),
        "primary_target": str(outcome_row["target"]),
        "secondary_first_touch_target": str(outcome_row.get("first_touch_target", "UNAVAILABLE")),
        "settlement_timestamp": pd.Timestamp(settlement_timestamp).isoformat(),
        "settlement_code_version": settlement_code_version,
        "research_only": True,
        "execution_enabled": False,
    }


class ProspectiveLedger:
    """Separate hash-chained forecast and settlement streams."""

    def __init__(self, root: Path, *, opened_through: str) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.forecast_path = root / "forecast_ledger.jsonl"
        self.settlement_path = root / "settlement_ledger.jsonl"
        self.opened_through = pd.Timestamp(opened_through).date()

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        existing = ProspectiveLedger._records(path)
        previous_hash = existing[-1]["record_hash"] if existing else None
        payload = {**record, "previous_hash": previous_hash}
        digest_input = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["record_hash"] = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_forecast(self, record: dict[str, Any]) -> None:
        if record.get("research_only") is not True or record.get("execution_enabled") is not False:
            raise ValueError("forecast must be research-only with execution disabled")
        session = pd.Timestamp(str(record["session"])).date()
        if session <= self.opened_through:
            raise ValueError("prospective mode rejects opened historical snapshots")
        opportunity_id = str(record["opportunity_id"])
        if any(
            str(row["opportunity_id"]) == opportunity_id
            for row in self._records(self.forecast_path)
        ):
            raise FileExistsError(f"duplicate prospective opportunity_id: {opportunity_id}")
        self._append(self.forecast_path, record)

    def append_settlement(self, record: dict[str, Any]) -> None:
        opportunity_id = str(record["opportunity_id"])
        forecasts = self._records(self.forecast_path)
        if not any(str(row["opportunity_id"]) == opportunity_id for row in forecasts):
            raise ValueError("settlement requires an existing immutable forecast")
        if any(
            str(row["opportunity_id"]) == opportunity_id
            for row in self._records(self.settlement_path)
        ):
            raise FileExistsError(f"duplicate settlement opportunity_id: {opportunity_id}")
        self._append(self.settlement_path, record)
