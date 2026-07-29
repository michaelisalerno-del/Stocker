"""Execution-free append-only forecast and settlement ledgers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from calendar import monthrange
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from stocker_research.directional_signature_atlas.features import (
    assert_outcome_free_feature_names,
)
from stocker_research.directional_signature_atlas.outcomes import classify_terminal_move

_FORECAST_FIELDS = frozenset(
    {
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "training_data_snapshot_hash",
        "feature_schema_hash",
        "long_library_hash",
        "short_library_hash",
        "neutral_library_hash",
        "opportunity_id",
        "symbol",
        "session",
        "decision_clock",
        "decision_timestamp",
        "entry_timestamp",
        "terminal_timestamp",
        "causal_features",
        "feature_availability_timestamps",
        "movement_permission",
        "long_signature_decisions",
        "short_signature_decisions",
        "long_vote_count",
        "short_vote_count",
        "conflict_state",
        "final_atlas_state",
        "reason_codes",
        "forecast_freeze_timestamp",
        "research_only",
        "execution_enabled",
    }
)
_SETTLEMENT_FIELDS = frozenset(
    {
        "opportunity_id",
        "terminal_timestamp",
        "gross_long_payoff_bps",
        "gross_short_payoff_bps",
        "costs_bps",
        "net_long_payoff_bps",
        "net_short_payoff_bps",
        "primary_target",
        "secondary_first_touch_target",
        "settlement_timestamp",
        "settlement_code_version",
        "settlement_status",
        "unavailable_reason",
        "research_only",
        "execution_enabled",
    }
)
_PRIMARY_TARGETS = frozenset({"LONG", "SHORT", "NEUTRAL", "UNAVAILABLE"})
_FIRST_TOUCH_TARGETS = frozenset(
    {"UPPER_FIRST", "LOWER_FIRST", "NEITHER", "SAME_BAR_DUAL_TOUCH", "UNAVAILABLE"}
)
_MOVEMENT_PERMISSION_STATES = frozenset({"PASS", "FAIL", "UNAVAILABLE"})
_DECISION_CLOCKS = {"clock_12": "10:30", "clock_36": "12:30"}


def canonical_library_hash(library: Sequence[Mapping[str, Any]]) -> str:
    """Hash a frozen library without depending on file whitespace."""

    payload = json.dumps(
        list(library), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_opportunity_id(value: Any) -> tuple[str, str, int]:
    parts = str(value).split("|")
    if len(parts) != 5 or parts[0] != "atlas":
        raise ValueError("opportunity_id must use atlas|year|symbol|session|ordinal")
    try:
        ordinal = int(parts[4])
    except ValueError as exc:
        raise ValueError("opportunity_id ordinal is invalid") from exc
    if parts[1] != parts[3][:4]:
        raise ValueError("opportunity_id year and session are inconsistent")
    return parts[2], parts[3], ordinal


def _validate_frozen_clock(record: Mapping[str, Any]) -> None:
    symbol, id_session, ordinal = _parse_opportunity_id(record["opportunity_id"])
    if symbol != str(record["symbol"]) or id_session != str(record["session"]):
        raise ValueError("opportunity_id does not match symbol and session")
    clock = str(record["decision_clock"])
    expected_clock = f"clock_{ordinal}"
    if clock != expected_clock or clock not in _DECISION_CLOCKS:
        raise ValueError("decision clock is not one of the frozen ordinals 12/36")

    decision = _aware_timestamp(record["decision_timestamp"], field="decision_timestamp")
    entry = _aware_timestamp(record["entry_timestamp"], field="entry_timestamp")
    terminal = _aware_timestamp(record["terminal_timestamp"], field="terminal_timestamp")
    local = decision.tz_convert("America/New_York")
    if local.strftime("%Y-%m-%d") != str(record["session"]):
        raise ValueError("decision timestamp does not match the frozen session")
    if local.strftime("%H:%M") != _DECISION_CLOCKS[clock] or local.second or local.microsecond:
        raise ValueError("decision timestamp is not the exact frozen five-minute clock")
    if entry != decision + pd.Timedelta(minutes=5):
        raise ValueError("entry must be the exact next-provider-open timestamp")
    if terminal != decision + pd.Timedelta(minutes=120):
        raise ValueError("terminal must be the fixed 24-bar timestamp")

    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(str(record["session"]))
    if not calendar.is_session(session):
        raise ValueError("forecast session is not an exact XNYS regular-session date")
    close = calendar.session_close(session).tz_convert("America/New_York")
    if terminal.tz_convert("America/New_York") + pd.Timedelta(minutes=5) > close:
        raise ValueError("fixed terminal bar would exceed the exact regular session")


def _require_exact_fields(
    record: Mapping[str, Any], required: frozenset[str], *, kind: str
) -> None:
    fields = set(record)
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"missing required {kind} fields: {missing}")
    unexpected = sorted(fields.difference(required))
    if unexpected:
        raise ValueError(f"unexpected {kind} fields: {unexpected}")


def _aware_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp


def _validate_forecast_record(
    record: Mapping[str, Any],
    *,
    required_causal_feature_names: frozenset[str] | None = None,
    expected_identity: Mapping[str, str] | None = None,
) -> None:
    _require_exact_fields(record, _FORECAST_FIELDS, kind="forecast")
    if record["research_only"] is not True or record["execution_enabled"] is not False:
        raise ValueError("forecast must be research-only with execution disabled")
    for field in (
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "training_data_snapshot_hash",
        "feature_schema_hash",
        "long_library_hash",
        "short_library_hash",
        "neutral_library_hash",
        "opportunity_id",
        "symbol",
        "session",
        "decision_clock",
    ):
        if not str(record[field]).strip():
            raise ValueError(f"forecast {field} must be non-empty")

    if expected_identity is not None:
        mismatches = sorted(
            field
            for field, expected in expected_identity.items()
            if str(record.get(field)) != str(expected)
        )
        if mismatches:
            raise ValueError(f"forecast frozen identity mismatch: {mismatches}")

    decision = _aware_timestamp(record["decision_timestamp"], field="decision_timestamp")
    entry = _aware_timestamp(record["entry_timestamp"], field="entry_timestamp")
    terminal = _aware_timestamp(record["terminal_timestamp"], field="terminal_timestamp")
    freeze = _aware_timestamp(
        record["forecast_freeze_timestamp"], field="forecast_freeze_timestamp"
    )
    if freeze < decision:
        raise ValueError("forecast cannot freeze before the completed decision bar")
    if freeze >= entry:
        raise ValueError("forecast must freeze before entry")
    if entry <= decision or terminal <= entry:
        raise ValueError("prospective economic timestamps are invalid")
    _validate_frozen_clock(record)

    features = record["causal_features"]
    availability = record["feature_availability_timestamps"]
    if not isinstance(features, Mapping) or not isinstance(availability, Mapping):
        raise ValueError("forecast causal features and availability must be mappings")
    names = [str(name) for name in features]
    assert_outcome_free_feature_names(names)
    if set(features) != set(availability):
        raise ValueError("every causal feature requires exactly one availability timestamp")
    if required_causal_feature_names is not None and set(features) != set(
        required_causal_feature_names
    ):
        missing = sorted(required_causal_feature_names.difference(features))
        extra = sorted(set(features).difference(required_causal_feature_names))
        raise ValueError(
            f"forecast causal feature schema mismatch: missing={missing}; extra={extra}"
        )
    for feature, value in features.items():
        available_at = availability[feature]
        if value is None:
            if available_at is not None:
                parsed = _aware_timestamp(available_at, field=f"{feature} availability")
                if parsed > freeze:
                    raise ValueError(f"feature {feature} is available after forecast freeze")
            continue
        if available_at is None:
            raise ValueError(f"populated feature {feature} lacks an availability timestamp")
        parsed = _aware_timestamp(available_at, field=f"{feature} availability")
        if parsed > freeze:
            raise ValueError(f"feature {feature} is available after forecast freeze")
        if parsed > decision:
            raise ValueError(f"feature {feature} is available after decision")

    for direction in ("long", "short"):
        decisions = record[f"{direction}_signature_decisions"]
        if not isinstance(decisions, Mapping) or not all(
            isinstance(value, bool) for value in decisions.values()
        ):
            raise ValueError(f"{direction} signature decisions must be a boolean mapping")
        expected_votes = sum(bool(value) for value in decisions.values())
        if record[f"{direction}_vote_count"] != expected_votes:
            raise ValueError(f"{direction} vote count does not match signature decisions")
    expected_conflict = bool(record["long_vote_count"] and record["short_vote_count"])
    if record["conflict_state"] is not expected_conflict:
        raise ValueError("forecast conflict state does not match vote counts")
    if record["final_atlas_state"] not in {"LONG", "SHORT", "NEUTRAL"}:
        raise ValueError("forecast atlas state is invalid")
    if str(record["movement_permission"]) not in _MOVEMENT_PERMISSION_STATES:
        raise ValueError("forecast movement permission must be PASS, FAIL, or UNAVAILABLE")
    reason_codes = record["reason_codes"]
    if not isinstance(reason_codes, list) or not all(
        isinstance(reason, str) and reason for reason in reason_codes
    ):
        raise ValueError("forecast reason codes must be non-empty strings")
    reasons = set(reason_codes)
    long_votes = int(record["long_vote_count"])
    short_votes = int(record["short_vote_count"])
    state = str(record["final_atlas_state"])
    movement_status = str(record["movement_permission"])
    if movement_status == "UNAVAILABLE":
        if state != "NEUTRAL" or "required_causal_feature_unavailable" not in reasons:
            raise ValueError("unavailable movement evidence must produce a neutral forecast")
    elif movement_status == "FAIL":
        if state != "NEUTRAL" or "movement_permission_failed" not in reasons:
            raise ValueError("movement-permission failure must produce a neutral forecast")
    elif bool(record["conflict_state"]):
        if state != "NEUTRAL" or "conflicting_votes" not in reasons:
            raise ValueError("conflicting votes must produce a neutral forecast")
    elif long_votes == 0 and short_votes == 0:
        if state != "NEUTRAL" or not {
            "no_directional_vote",
            "required_causal_feature_unavailable",
        }.intersection(reasons):
            raise ValueError("no directional vote must produce a neutral forecast")
    elif long_votes > 0 and short_votes == 0:
        valid = (state == "LONG" and "supported_long_vote" in reasons) or (
            state == "NEUTRAL" and "non_positive_conservative_value" in reasons
        )
        if not valid:
            raise ValueError("long-only votes are inconsistent with final atlas state")
    elif short_votes > 0 and long_votes == 0:
        valid = (state == "SHORT" and "supported_short_vote" in reasons) or (
            state == "NEUTRAL" and "non_positive_conservative_value" in reasons
        )
        if not valid:
            raise ValueError("short-only votes are inconsistent with final atlas state")


def _validate_settlement_record(
    record: Mapping[str, Any], *, forecast: Mapping[str, Any] | None = None
) -> None:
    _require_exact_fields(record, _SETTLEMENT_FIELDS, kind="settlement")
    if record["research_only"] is not True or record["execution_enabled"] is not False:
        raise ValueError("settlement must be research-only with execution disabled")
    if (
        not str(record["opportunity_id"]).strip()
        or not str(record["settlement_code_version"]).strip()
    ):
        raise ValueError("settlement identity fields must be non-empty")
    terminal = _aware_timestamp(record["terminal_timestamp"], field="terminal_timestamp")
    settled_at = _aware_timestamp(record["settlement_timestamp"], field="settlement_timestamp")
    if settled_at < terminal:
        raise ValueError("settlement cannot occur before the terminal matures")
    if forecast is not None:
        frozen_terminal = _aware_timestamp(
            forecast["terminal_timestamp"], field="forecast terminal_timestamp"
        )
        if terminal != frozen_terminal:
            raise ValueError("settlement terminal differs from frozen forecast terminal")

    status = str(record["settlement_status"])
    unavailable_reason = record["unavailable_reason"]
    economic_fields = (
        "gross_long_payoff_bps",
        "gross_short_payoff_bps",
        "costs_bps",
        "net_long_payoff_bps",
        "net_short_payoff_bps",
    )
    if status == "UNAVAILABLE":
        if str(record["primary_target"]) != "UNAVAILABLE":
            raise ValueError("unavailable settlement must retain the UNAVAILABLE target")
        if str(record["secondary_first_touch_target"]) != "UNAVAILABLE":
            raise ValueError("unavailable settlement must retain UNAVAILABLE first touch")
        if any(record[field] is not None for field in economic_fields):
            raise ValueError("unavailable settlement economics must remain null")
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            raise ValueError("unavailable settlement requires a reason")
        return
    if status != "SETTLED" or unavailable_reason is not None:
        raise ValueError("available settlement status/reason is inconsistent")
    values = {field: float(record[field]) for field in economic_fields}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("settlement economics must be finite")
    if values["costs_bps"] < 0.0:
        raise ValueError("settlement costs must be non-negative")
    if not math.isclose(
        values["gross_short_payoff_bps"],
        -values["gross_long_payoff_bps"],
        abs_tol=1e-9,
    ):
        raise ValueError("gross short payoff must be the inverse of gross long payoff")
    if not math.isclose(
        values["net_long_payoff_bps"],
        values["gross_long_payoff_bps"] - values["costs_bps"],
        abs_tol=1e-9,
    ):
        raise ValueError("net long payoff is inconsistent with gross payoff and costs")
    if not math.isclose(
        values["net_short_payoff_bps"],
        values["gross_short_payoff_bps"] - values["costs_bps"],
        abs_tol=1e-9,
    ):
        raise ValueError("net short payoff is inconsistent with gross payoff and costs")
    target = str(record["primary_target"])
    if target not in _PRIMARY_TARGETS:
        raise ValueError("settlement primary target is invalid")
    expected_target = classify_terminal_move(
        values["gross_long_payoff_bps"], values["costs_bps"], 2.0
    )
    if target != expected_target:
        raise ValueError("settlement primary target is inconsistent with frozen economics")
    if str(record["secondary_first_touch_target"]) not in _FIRST_TOUCH_TARGETS:
        raise ValueError("settlement first-touch target is invalid")


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
    neutral_library: Sequence[Mapping[str, Any]],
    causal_feature_names: Sequence[str],
    forecast_input_snapshot_hash: str,
    forecast_freeze_timestamp: str | pd.Timestamp,
) -> dict[str, Any]:
    """Freeze one execution-free atlas forecast from an outcome-free feature row."""

    decision = pd.Timestamp(str(feature_row["decision_timestamp"]))
    freeze = pd.Timestamp(forecast_freeze_timestamp)
    if decision.tzinfo is None or freeze.tzinfo is None:
        raise ValueError("prospective timestamps must be timezone-aware")
    entry = pd.Timestamp(
        str(feature_row.get("entry_timestamp", decision + pd.Timedelta(minutes=5)))
    )
    terminal = pd.Timestamp(
        str(feature_row.get("terminal_timestamp", decision + pd.Timedelta(minutes=120)))
    )

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
    raw_movement = feature_row.get("movement_permission")
    movement_status = (
        "UNAVAILABLE"
        if raw_movement is None or pd.isna(raw_movement)
        else ("PASS" if bool(raw_movement) else "FAIL")
    )
    required_features = {
        str(condition["feature"])
        for entry_payload in [*long_library, *short_library]
        for condition in entry_payload["signature"]["conditions"]
    }
    unlogged_required = sorted(required_features.difference(map(str, causal_feature_names)))
    if unlogged_required:
        raise ValueError(f"signature features missing from forecast ledger: {unlogged_required}")
    missing_required = sorted(
        feature
        for feature in required_features
        if feature not in feature_row or pd.isna(feature_row[feature])
    )
    state = "NEUTRAL"
    reasons: list[str] = []
    if missing_required:
        reasons.append("required_causal_feature_unavailable")
    if movement_status == "UNAVAILABLE":
        reasons.append("required_causal_feature_unavailable")
    elif movement_status == "FAIL":
        reasons.append("movement_permission_failed")
    if missing_required or movement_status != "PASS":
        pass
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
    if not str(forecast_input_snapshot_hash).strip():
        raise ValueError("prospective input snapshot hash must be non-empty")
    record = {
        "run_id": str(metadata["run_id"]),
        "git_sha": str(metadata["git_sha"]),
        "contract_hash": str(metadata["contract_sha256"]),
        "data_snapshot_hash": str(forecast_input_snapshot_hash),
        "training_data_snapshot_hash": str(metadata["data_snapshot_sha256"]),
        "feature_schema_hash": str(metadata["feature_schema_sha256"]),
        "long_library_hash": canonical_library_hash(long_library),
        "short_library_hash": canonical_library_hash(short_library),
        "neutral_library_hash": canonical_library_hash(neutral_library),
        "opportunity_id": str(feature_row["opportunity_id"]),
        "symbol": str(feature_row["symbol"]),
        "session": str(feature_row["session"]),
        "decision_clock": str(feature_row["decision_clock"]),
        "decision_timestamp": decision.isoformat(),
        "entry_timestamp": entry.isoformat(),
        "terminal_timestamp": terminal.isoformat(),
        "causal_features": causal_features,
        "feature_availability_timestamps": availability,
        "movement_permission": movement_status,
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
    expected_identity = {
        "run_id": str(metadata["run_id"]),
        "git_sha": str(metadata["git_sha"]),
        "contract_hash": str(metadata["contract_sha256"]),
        "training_data_snapshot_hash": str(metadata["data_snapshot_sha256"]),
        "feature_schema_hash": str(metadata["feature_schema_sha256"]),
        "long_library_hash": str(metadata["long_library_sha256"]),
        "short_library_hash": str(metadata["short_library_sha256"]),
        "neutral_library_hash": str(metadata["neutral_library_sha256"]),
    }
    _validate_forecast_record(
        record,
        required_causal_feature_names=frozenset(map(str, causal_feature_names)),
        expected_identity=expected_identity,
    )
    return record


def build_settlement_record(
    outcome_row: Mapping[str, Any],
    *,
    settlement_timestamp: str | pd.Timestamp,
    settlement_code_version: str,
) -> dict[str, Any]:
    """Create a separate immutable settlement without mutating its forecast."""

    unavailable = str(outcome_row["target"]) == "UNAVAILABLE"
    record = {
        "opportunity_id": str(outcome_row["opportunity_id"]),
        "terminal_timestamp": pd.Timestamp(str(outcome_row["terminal_timestamp"])).isoformat(),
        "gross_long_payoff_bps": None
        if unavailable
        else float(outcome_row["gross_long_return_bps"]),
        "gross_short_payoff_bps": None
        if unavailable
        else float(outcome_row["gross_short_return_bps"]),
        "costs_bps": None if unavailable else float(outcome_row["round_trip_cost_bps"]),
        "net_long_payoff_bps": None if unavailable else float(outcome_row["net_long_return_bps"]),
        "net_short_payoff_bps": None if unavailable else float(outcome_row["net_short_return_bps"]),
        "primary_target": str(outcome_row["target"]),
        "secondary_first_touch_target": str(outcome_row.get("first_touch_target", "UNAVAILABLE")),
        "settlement_timestamp": pd.Timestamp(settlement_timestamp).isoformat(),
        "settlement_code_version": settlement_code_version,
        "settlement_status": "UNAVAILABLE" if unavailable else "SETTLED",
        "unavailable_reason": str(outcome_row.get("score_status", "exact_outcome_unavailable"))
        if unavailable
        else None,
        "research_only": True,
        "execution_enabled": False,
    }
    _validate_settlement_record(record)
    return record


class ProspectiveLedger:
    """Separate hash-chained forecast and settlement streams."""

    def __init__(
        self,
        root: Path,
        *,
        opened_through: str,
        required_causal_feature_names: Sequence[str],
        expected_identity: Mapping[str, str],
        completion_requirements: Mapping[str, int],
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.forecast_path = root / "forecast_ledger.jsonl"
        self.settlement_path = root / "settlement_ledger.jsonl"
        self.lock_path = root / ".ledger.lock"
        self.opened_through = pd.Timestamp(opened_through).date()
        self.required_causal_feature_names = frozenset(map(str, required_causal_feature_names))
        self.expected_identity = {str(key): str(value) for key, value in expected_identity.items()}
        self.completion_requirements = {
            str(key): int(value) for key, value in completion_requirements.items()
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        previous_hash: str | None = None
        for record in records:
            if record.get("previous_hash") != previous_hash:
                raise ValueError(f"broken prospective hash chain: {path}")
            stored_hash = record.get("record_hash")
            payload = {key: value for key, value in record.items() if key != "record_hash"}
            digest_input = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            expected_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
            if stored_hash != expected_hash:
                raise ValueError(f"invalid prospective record hash: {path}")
            previous_hash = str(stored_hash)
        return records

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
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("prospective ledger append made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_forecast(self, record: dict[str, Any]) -> None:
        _validate_forecast_record(
            record,
            required_causal_feature_names=self.required_causal_feature_names,
            expected_identity=self.expected_identity,
        )
        session = pd.Timestamp(str(record["session"])).date()
        if session <= self.opened_through:
            raise ValueError("prospective mode rejects opened historical snapshots")
        opportunity_id = str(record["opportunity_id"])
        with self._locked():
            if any(
                str(row["opportunity_id"]) == opportunity_id
                for row in self._records(self.forecast_path)
            ):
                raise FileExistsError(f"duplicate prospective opportunity_id: {opportunity_id}")
            self._append(self.forecast_path, record)

    def append_settlement(self, record: dict[str, Any]) -> None:
        opportunity_id = str(record["opportunity_id"])
        with self._locked():
            forecasts = self._records(self.forecast_path)
            forecast = next(
                (row for row in forecasts if str(row["opportunity_id"]) == opportunity_id),
                None,
            )
            if forecast is None:
                raise ValueError("settlement requires an existing immutable forecast")
            _validate_settlement_record(record, forecast=forecast)
            if any(
                str(row["opportunity_id"]) == opportunity_id
                for row in self._records(self.settlement_path)
            ):
                raise FileExistsError(f"duplicate settlement opportunity_id: {opportunity_id}")
            self._append(self.settlement_path, record)

    def completion_status(self) -> dict[str, Any]:
        """Return blinded administrative counts; never expose payoff fields."""

        with self._locked():
            forecasts = self._records(self.forecast_path)
            settlements = self._records(self.settlement_path)
        forecast_by_id = {str(row["opportunity_id"]): row for row in forecasts}
        available = [row for row in settlements if str(row.get("settlement_status")) == "SETTLED"]
        settled_forecasts = [
            forecast_by_id[str(row["opportunity_id"])]
            for row in available
            if str(row["opportunity_id"]) in forecast_by_id
        ]
        sessions = {str(row["session"]) for row in settled_forecasts}
        symbols = {str(row["symbol"]) for row in settled_forecasts}
        long_rows = [row for row in settled_forecasts if row["final_atlas_state"] == "LONG"]
        short_rows = [row for row in settled_forecasts if row["final_atlas_state"] == "SHORT"]
        latest = max(
            (
                _aware_timestamp(row["settlement_timestamp"], field="settlement_timestamp")
                for row in settlements
            ),
            default=None,
        )
        completed_months: set[str] = set()
        if latest is not None:
            latest_date = latest.date()
            for row in settled_forecasts:
                session = pd.Timestamp(str(row["session"]))
                month_end = session.date().replace(day=monthrange(session.year, session.month)[1])
                if month_end <= latest_date:
                    completed_months.add(session.strftime("%Y-%m"))
        counts = {
            "minimum_settled_opportunities": len(available),
            "minimum_independent_sessions": len(sessions),
            "minimum_stocks": len(symbols),
            "minimum_completed_calendar_months": len(completed_months),
            "minimum_long_outputs": len(long_rows),
            "minimum_short_outputs": len(short_rows),
            "minimum_sessions_with_long": len({str(row["session"]) for row in long_rows}),
            "minimum_sessions_with_short": len({str(row["session"]) for row in short_rows}),
        }
        deficits = {
            key: max(0, required - counts.get(key, 0))
            for key, required in self.completion_requirements.items()
        }
        return {
            "requirements_met": all(value == 0 for value in deficits.values()),
            "counts": counts,
            "requirements": dict(self.completion_requirements),
            "deficits": deficits,
            "matured_unavailable_records": len(settlements) - len(available),
            "payoff_fields_blinded": True,
        }

    def read_settlements(self) -> list[dict[str, Any]]:
        """Open economic settlement rows only after the frozen completion gate."""

        status = self.completion_status()
        if not bool(status["requirements_met"]):
            raise PermissionError("prospective sample completion rule has not been reached")
        with self._locked():
            return self._records(self.settlement_path)
