"""File-locked, create-only prospective opportunity and outcome ledgers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Set
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from .execution import FillEvidence, score_fill_envelope
from .families import family_spec


class IntegrityError(RuntimeError):
    """An immutable identity, precursor, or payload changed."""


class DuplicateRecordError(FileExistsError):
    """An identical create-only record was submitted more than once."""


REQUIRED_OPPORTUNITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "contract_hash",
        "git_sha",
        "code_version",
        "data_snapshot_hash",
        "provider_data_hash",
        "source_model_version",
        "source_run_id",
        "opportunity_id",
        "anchor_id",
        "event_lineage_id",
        "symbol",
        "session",
        "loop_id",
        "orientation",
        "family",
        "classification",
        "frozen_direction",
        "anchor_timestamp",
        "anchor_close",
        "long_threshold",
        "short_threshold",
        "threshold_known_timestamp",
        "signal_known_timestamp",
        "trigger_type",
        "trigger_timestamp",
        "reference_fill_convention",
        "reference_entry_timestamp",
        "reference_entry_price",
        "original_terminal_timestamp",
        "feature_availability_timestamp",
        "source_availability_timestamp",
        "opportunity_created_timestamp",
        "research_only",
        "execution_enabled",
    }
)
REQUIRED_TRIGGER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "trigger_id",
        "opportunity_id",
        "trigger_observed_timestamp",
        "trigger_bar_timestamp",
        "trigger_bar_open",
        "trigger_bar_high",
        "trigger_bar_low",
        "trigger_bar_close",
        "trigger_type",
        "reference_entry_timestamp",
        "reference_entry_price",
        "fill_evidence_classification",
        "signal_known_timestamp",
        "signal_fill_time_status",
        "evidence_detail",
        "market_data_availability_timestamp",
        "provider_data_hash",
        "append_timestamp",
    }
)
REQUIRED_SETTLEMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "settlement_id",
        "opportunity_id",
        "terminal_timestamp",
        "terminal_price",
        "terminal_data_hash",
        "terminal_data_availability_timestamp",
        "settlement_timestamp",
        "settlement_code_version",
    }
)
_FORBIDDEN_OPPORTUNITY_TOKENS: Final[tuple[str, ...]] = (
    "payoff",
    "outcome",
    "realised",
    "realized",
    "hindsight",
    "episode",
    "future_route",
    "terminal_price",
)
_COMPLETION_THRESHOLDS: Final[dict[str, int]] = {
    "named_total": 100,
    "cycle_04|state_4": 25,
    "cycle_07|state_5": 50,
    "control_total": 20,
    "sessions": 60,
    "stocks": 10,
    "months": 3,
}


def _timestamp(value: object, field: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{field} is required")
    timestamp = pd.Timestamp(cast(Any, value))
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _optional_timestamp(value: object, field: str) -> pd.Timestamp | None:
    return None if value is None else _timestamp(value, field)


def _identifier(value: object, field: str) -> str:
    identifier = str(value)
    if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
        raise ValueError(f"unsafe {field}")
    return identifier


def _normalise(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _timestamp(value, "timestamp").isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("immutable records cannot contain non-finite values")
    return value


def _canonical_bytes(record: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            _normalise(dict(record)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _record_payload(record: Mapping[str, object]) -> dict[str, object]:
    payload = cast(dict[str, object], _normalise(dict(record)))
    payload_without_hash = dict(payload)
    payload_without_hash.pop("record_sha256", None)
    payload["record_sha256"] = hashlib.sha256(_canonical_bytes(payload_without_hash)).hexdigest()
    return payload


def _verify_record(record: Mapping[str, object], *, path: Path) -> None:
    stored = str(record.get("record_sha256", ""))
    payload = dict(record)
    payload.pop("record_sha256", None)
    expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if stored != expected:
        raise IntegrityError(f"record hash mismatch: {path}")


class ProspectiveExecutionLedger:
    """Immutable research logger with no execution-facing method or dependency."""

    def __init__(
        self,
        root: Path,
        *,
        contract_hash: str,
        completion_rule_hash: str,
        opened_periods: Set[int],
        opened_snapshot_hashes: Set[str],
    ) -> None:
        self.root = Path(root)
        self.contract_hash = str(contract_hash)
        self.completion_rule_hash = str(completion_rule_hash)
        self.opened_periods = frozenset(int(value) for value in opened_periods)
        self.opened_snapshot_hashes = frozenset(str(value) for value in opened_snapshot_hashes)
        self.opportunity_root = self.root / "opportunities"
        self.trigger_root = self.root / "triggers"
        self.settlement_root = self.root / "settlements"
        self.identity_path = self.root / "collection_identity.json"
        self.lock_path = self.root / ".ledger.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialise_identity()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _initialise_identity(self) -> None:
        identity = _record_payload(
            {
                "contract_hash": self.contract_hash,
                "completion_rule_hash": self.completion_rule_hash,
                "opened_periods": sorted(self.opened_periods),
                "opened_snapshot_hashes": sorted(self.opened_snapshot_hashes),
                "research_only": True,
                "execution_enabled": False,
            }
        )
        with self._locked():
            if self.identity_path.exists():
                stored = json.loads(self.identity_path.read_text(encoding="utf-8"))
                _verify_record(stored, path=self.identity_path)
                if stored != identity:
                    raise IntegrityError("collection identity changed")
                return
            self._exclusive_write(self.identity_path, identity)

    @staticmethod
    def _exclusive_write(path: Path, record: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical_bytes(record)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            written = 0
            while written < len(data):
                written += os.write(descriptor, data[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _create(self, path: Path, record: Mapping[str, object]) -> Path:
        payload = _record_payload(record)
        with self._locked():
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                _verify_record(existing, path=path)
                if existing == payload:
                    raise DuplicateRecordError(f"duplicate immutable record: {path.name}")
                raise IntegrityError(f"existing record has different payload: {path.name}")
            self._exclusive_write(path, payload)
        return path

    @staticmethod
    def _load(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise ValueError(f"missing immutable precursor: {path}")
        record = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        _verify_record(record, path=path)
        return record

    def _validate_opportunity(
        self, record: Mapping[str, object], *, prospective: bool
    ) -> dict[str, object]:
        if missing := sorted(REQUIRED_OPPORTUNITY_FIELDS - set(record)):
            raise ValueError(f"missing opportunity fields: {missing}")
        lowered_keys = [str(key).lower() for key in record]
        if any(token in key for key in lowered_keys for token in _FORBIDDEN_OPPORTUNITY_TOKENS):
            raise ValueError("future outcome or hindsight field is forbidden")
        if str(record["contract_hash"]) != self.contract_hash:
            raise IntegrityError("opportunity contract hash changed")
        session = str(record["session"])
        try:
            period = int(session[:4])
        except ValueError as error:
            raise ValueError("session must begin with a four-digit year") from error
        snapshot = str(record["data_snapshot_hash"])
        if prospective and period in self.opened_periods:
            raise ValueError("opened historical period is forbidden in prospective mode")
        if prospective and snapshot in self.opened_snapshot_hashes:
            raise ValueError("opened data snapshot is forbidden in prospective mode")
        if prospective and not snapshot:
            raise ValueError("a new prospective data snapshot is required")
        spec = family_spec(str(record["family"]))
        identity = (
            str(record["loop_id"]),
            str(record["orientation"]),
            str(record["classification"]),
        )
        if identity != (spec.loop_id, spec.orientation, spec.classification):
            raise ValueError("family identity does not match the frozen mapping")
        direction = int(str(record["frozen_direction"]))
        if direction not in (-1, 1):
            raise ValueError("frozen direction must be -1 or 1")
        anchor_close = float(str(record["anchor_close"]))
        upper = float(str(record["long_threshold"]))
        lower = float(str(record["short_threshold"]))
        if not np.isfinite([anchor_close, upper, lower]).all() or lower <= 0.0 or upper <= lower:
            raise ValueError("invalid anchor price or thresholds")
        anchor = _timestamp(record["anchor_timestamp"], "anchor timestamp")
        threshold_known = _timestamp(
            record["threshold_known_timestamp"], "threshold-known timestamp"
        )
        if threshold_known != anchor + pd.Timedelta(minutes=5):
            raise ValueError("threshold-known timestamp must be anchor plus five minutes")
        signal_known = _optional_timestamp(
            record["signal_known_timestamp"], "signal-known timestamp"
        )
        reference = _optional_timestamp(
            record["reference_entry_timestamp"], "reference entry timestamp"
        )
        if signal_known is not None and signal_known < threshold_known:
            raise ValueError("signal-known timestamp precedes threshold availability")
        if signal_known is not None and reference is not None and signal_known > reference:
            raise ValueError("signal-known timestamp follows reference fill")
        terminal = _timestamp(record["original_terminal_timestamp"], "terminal timestamp")
        if terminal != anchor + pd.Timedelta(minutes=125):
            raise ValueError("original terminal must remain anchor plus 125 minutes")
        if reference is not None and reference >= terminal:
            raise ValueError("reference entry must precede the original terminal")
        created = _timestamp(
            record["opportunity_created_timestamp"], "opportunity-created timestamp"
        )
        for field in ("feature_availability_timestamp", "source_availability_timestamp"):
            if _timestamp(record[field], field.replace("_", " ")) > created:
                raise ValueError(f"{field} follows opportunity creation")
        if record["research_only"] is not True or record["execution_enabled"] is not False:
            raise ValueError("research-only safety flags are required")
        payload = dict(record)
        payload.update(
            {
                "anchor_timestamp": anchor,
                "threshold_known_timestamp": threshold_known,
                "signal_known_timestamp": signal_known,
                "reference_entry_timestamp": reference,
                "original_terminal_timestamp": terminal,
                "opportunity_created_timestamp": created,
                "research_only": True,
                "execution_enabled": False,
                "broker_connection_enabled": False,
                "order_placement_enabled": False,
                "paper_or_demo_ordering_enabled": False,
                "position_management_enabled": False,
                "deployment_enabled": False,
            }
        )
        return payload

    def append_opportunity(self, record: Mapping[str, object], *, prospective: bool = True) -> Path:
        payload = self._validate_opportunity(record, prospective=prospective)
        identifier = _identifier(record["opportunity_id"], "opportunity_id")
        return self._create(self.opportunity_root / f"{identifier}.json", payload)

    def append_trigger(self, record: Mapping[str, object]) -> Path:
        if missing := sorted(REQUIRED_TRIGGER_FIELDS - set(record)):
            raise ValueError(f"missing trigger fields: {missing}")
        opportunity_id = _identifier(record["opportunity_id"], "opportunity_id")
        opportunity_path = self.opportunity_root / f"{opportunity_id}.json"
        if not opportunity_path.is_file():
            raise ValueError("trigger refers to an unknown opportunity")
        opportunity = self._load(opportunity_path)
        if str(record["provider_data_hash"]) != str(opportunity["provider_data_hash"]):
            raise IntegrityError("trigger provider hash differs from opportunity")
        evidence = FillEvidence(str(record["fill_evidence_classification"]))
        reference_timestamp = _timestamp(
            record["reference_entry_timestamp"], "reference entry timestamp"
        )
        signal_known = _optional_timestamp(
            record["signal_known_timestamp"], "signal-known timestamp"
        )
        valid_classes = {FillEvidence.EXACTLY_OBSERVABLE, FillEvidence.GAP_FILL_OBSERVABLE}
        if evidence in valid_classes and signal_known is None:
            raise ValueError("valid fill evidence requires a signal-known timestamp")
        if signal_known is not None and signal_known > reference_timestamp:
            raise ValueError("signal-known timestamp cannot follow the reference fill")
        opportunity_reference = _optional_timestamp(
            opportunity["reference_entry_timestamp"], "opportunity reference entry timestamp"
        )
        if opportunity_reference is not None and opportunity_reference != reference_timestamp:
            raise IntegrityError("trigger reference timestamp differs from opportunity")
        reference_price = float(str(record["reference_entry_price"]))
        if not np.isfinite(reference_price) or reference_price <= 0.0:
            raise ValueError("reference entry price must be finite and positive")
        if opportunity["reference_entry_price"] is not None and not np.isclose(
            reference_price,
            float(str(opportunity["reference_entry_price"])),
            rtol=0.0,
            atol=1e-12,
        ):
            raise IntegrityError("trigger reference price differs from opportunity")
        if opportunity["trigger_type"] is not None and str(record["trigger_type"]) != str(
            opportunity["trigger_type"]
        ):
            raise IntegrityError("trigger type differs from opportunity")
        observed = _timestamp(record["trigger_observed_timestamp"], "trigger observed timestamp")
        trigger_bar = _timestamp(record["trigger_bar_timestamp"], "trigger bar timestamp")
        available = _timestamp(
            record["market_data_availability_timestamp"], "market-data availability timestamp"
        )
        appended = _timestamp(record["append_timestamp"], "trigger append timestamp")
        if available < trigger_bar or appended < available or observed < trigger_bar:
            raise ValueError("trigger observation or append timing is non-causal")
        values = np.asarray(
            [
                record["trigger_bar_open"],
                record["trigger_bar_high"],
                record["trigger_bar_low"],
                record["trigger_bar_close"],
            ],
            dtype=float,
        )
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("trigger bar OHLC must be finite and positive")
        payload = dict(record)
        payload.update(
            {
                "trigger_observed_timestamp": observed,
                "trigger_bar_timestamp": trigger_bar,
                "reference_entry_timestamp": reference_timestamp,
                "signal_known_timestamp": signal_known,
                "market_data_availability_timestamp": available,
                "append_timestamp": appended,
                "run_id": opportunity["run_id"],
                "contract_hash": opportunity["contract_hash"],
                "opportunity_record_sha256": opportunity["record_sha256"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        return self._create(self.trigger_root / f"{opportunity_id}.json", payload)

    def append_settlement(self, record: Mapping[str, object]) -> Path:
        if missing := sorted(REQUIRED_SETTLEMENT_FIELDS - set(record)):
            raise ValueError(f"missing settlement fields: {missing}")
        if any(str(key).startswith(("F0_", "F5_", "F10_", "F15_", "F20_")) for key in record):
            raise ValueError("caller-supplied payoff fields are forbidden")
        opportunity_id = _identifier(record["opportunity_id"], "opportunity_id")
        opportunity_path = self.opportunity_root / f"{opportunity_id}.json"
        trigger_path = self.trigger_root / f"{opportunity_id}.json"
        if not trigger_path.is_file():
            raise ValueError("settlement requires an immutable trigger record")
        opportunity = self._load(opportunity_path)
        trigger = self._load(trigger_path)
        terminal = _timestamp(record["terminal_timestamp"], "terminal timestamp")
        frozen_terminal = _timestamp(
            opportunity["original_terminal_timestamp"], "frozen terminal timestamp"
        )
        if terminal != frozen_terminal:
            raise IntegrityError("settlement terminal differs from the frozen terminal")
        try:
            terminal_price = float(str(record["terminal_price"]))
        except (TypeError, ValueError) as error:
            raise ValueError("terminal price must be finite and positive") from error
        if not np.isfinite(terminal_price) or terminal_price <= 0.0:
            raise ValueError("terminal price must be finite and positive")
        terminal_available = _timestamp(
            record["terminal_data_availability_timestamp"],
            "terminal-data availability timestamp",
        )
        settled = _timestamp(record["settlement_timestamp"], "settlement timestamp")
        if terminal_available < terminal or settled < terminal or settled < terminal_available:
            raise ValueError("terminal has not fully matured for settlement")
        direction = int(str(opportunity["frozen_direction"]))
        reference_price = float(str(trigger["reference_entry_price"]))
        envelope = score_fill_envelope(
            opportunity_id=opportunity_id,
            direction=direction,
            reference_entry_price=reference_price,
            terminal_timestamp=terminal,
            terminal_price=terminal_price,
            cost_bps=10.0,
        )
        payload = dict(record)
        payload.update(
            {
                "terminal_timestamp": terminal,
                "terminal_price": terminal_price,
                "terminal_data_availability_timestamp": terminal_available,
                "settlement_timestamp": settled,
                "run_id": opportunity["run_id"],
                "contract_hash": opportunity["contract_hash"],
                "opportunity_record_sha256": opportunity["record_sha256"],
                "trigger_record_sha256": trigger["record_sha256"],
                "family": opportunity["family"],
                "classification": opportunity["classification"],
                "symbol": opportunity["symbol"],
                "session": opportunity["session"],
                "direction": direction,
                "fill_evidence_classification": trigger["fill_evidence_classification"],
                "research_only": True,
                "execution_enabled": False,
            }
        )
        for outcome in envelope:
            prefix = outcome.fill_model
            payload[f"{prefix}_stressed_entry_price"] = outcome.stressed_entry_price
            payload[f"{prefix}_gross_payoff_bps"] = outcome.gross_payoff_bps
            payload[f"{prefix}_cost_bps"] = outcome.cost_bps
            payload[f"{prefix}_net_payoff_bps"] = outcome.net_payoff_bps
        return self._create(self.settlement_root / f"{opportunity_id}.json", payload)

    @staticmethod
    def _records(root: Path) -> list[dict[str, object]]:
        if not root.exists():
            return []
        records: list[dict[str, object]] = []
        for path in sorted(root.glob("*.json")):
            records.append(ProspectiveExecutionLedger._load(path))
        return records

    def _completion(self) -> dict[str, object]:
        settlements = self._records(self.settlement_root)
        family_counts = Counter(str(row["family"]) for row in settlements)
        named_total = sum(
            count
            for family, count in family_counts.items()
            if family_spec(family).classification == "named"
        )
        control_total = sum(
            count
            for family, count in family_counts.items()
            if family_spec(family).classification == "control"
        )
        sessions = {str(row["session"]) for row in settlements}
        stocks = {str(row["symbol"]) for row in settlements}
        months = {session[:7] for session in sessions}
        observed = {
            "named_total": named_total,
            "cycle_04|state_4": family_counts["cycle_04|state_4"],
            "cycle_07|state_5": family_counts["cycle_07|state_5"],
            "control_total": control_total,
            "sessions": len(sessions),
            "stocks": len(stocks),
            "months": len(months),
        }
        checks = {
            key: int(observed[key]) >= minimum for key, minimum in _COMPLETION_THRESHOLDS.items()
        }
        return {
            "complete": all(checks.values()),
            "observed": observed,
            "minimums": dict(_COMPLETION_THRESHOLDS),
            "checks": checks,
        }

    def administrative_status(self) -> dict[str, object]:
        opportunities = self._records(self.opportunity_root)
        triggers = self._records(self.trigger_root)
        settlements = self._records(self.settlement_root)
        completion = self._completion()
        evidence = Counter(str(row["fill_evidence_classification"]) for row in triggers)
        return {
            "opportunity_count": len(opportunities),
            "trigger_count": len(triggers),
            "settlement_count": len(settlements),
            "missing_settlement_count": len(opportunities) - len(settlements),
            "fill_evidence_counts": dict(sorted(evidence.items())),
            "completion": completion,
            "safety": {
                "research_only": True,
                "execution_enabled": False,
                "broker_connection_enabled": False,
                "order_placement_enabled": False,
            },
            "prospective_decision": (
                "completion_rule_reached_economic_scoring_permitted"
                if completion["complete"]
                else "prospective_sample_incomplete"
            ),
        }

    def read_settlements_for_economic_scoring(self) -> list[dict[str, object]]:
        if not self._completion()["complete"]:
            raise ValueError("prospective sample is incomplete; economic scoring remains blinded")
        return self._records(self.settlement_root)

    def export_records(self, stage: str) -> Iterable[dict[str, object]]:
        """Export immutable records for audit; settlement economics remain gated."""

        roots = {
            "opportunity": self.opportunity_root,
            "trigger": self.trigger_root,
            "settlement": self.settlement_root,
        }
        if stage not in roots:
            raise ValueError(f"unknown stage: {stage}")
        if stage == "settlement" and not self._completion()["complete"]:
            raise ValueError("prospective sample is incomplete; settlement export remains blinded")
        return tuple(self._records(roots[stage]))
