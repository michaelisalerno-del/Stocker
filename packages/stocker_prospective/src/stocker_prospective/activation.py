"""Immutable first-activation identity and hard prospective chronology gate."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.contract import CONTRACT_VERSION, claims_boundary

NEW_YORK = ZoneInfo("America/New_York")


class ActivationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str = CONTRACT_VERSION
    claims_boundary: dict[str, bool | float]
    prospective_collection_start_utc: datetime
    prospective_collection_start_new_york: datetime
    git_sha: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    model_artifact_hashes: dict[str, str]
    configuration_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    ibkr_api_version: str
    tws_or_gateway_version: str

    @field_validator(
        "prospective_collection_start_utc",
        "prospective_collection_start_new_york",
    )
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("activation timestamps must be timezone-aware")
        return value


class ProspectiveActivationLedger:
    """Create one immutable activation record and reject later identity drift."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ActivationRecord | None:
        if not self.path.is_file():
            return None
        return ActivationRecord.model_validate_json(self.path.read_text(encoding="utf-8"))

    def activate(
        self,
        *,
        activation_timestamp_utc: datetime,
        git_sha: str,
        model_artifact_hashes: dict[str, str],
        configuration_hash: str,
        ibkr_api_version: str,
        tws_or_gateway_version: str,
    ) -> ActivationRecord:
        if activation_timestamp_utc.tzinfo is None or activation_timestamp_utc.utcoffset() is None:
            raise ValueError("activation timestamp must be timezone-aware")
        utc = activation_timestamp_utc.astimezone(UTC)
        proposed = ActivationRecord(
            claims_boundary=claims_boundary(),
            prospective_collection_start_utc=utc,
            prospective_collection_start_new_york=utc.astimezone(NEW_YORK),
            git_sha=git_sha,
            model_artifact_hashes=dict(sorted(model_artifact_hashes.items())),
            configuration_hash=configuration_hash,
            ibkr_api_version=ibkr_api_version,
            tws_or_gateway_version=tws_or_gateway_version,
        )
        existing = self.load()
        if existing is not None:
            if existing != proposed:
                raise ValueError("prospective activation identity is immutable")
            return existing
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                proposed.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if self.path.exists():
            temporary.unlink()
            existing = self.load()
            if existing != proposed:
                raise ValueError("prospective activation identity is immutable")
            assert existing is not None
            return existing
        os.replace(temporary, self.path)
        return proposed

    def require_prospective_timestamp(self, recorded_at: datetime) -> datetime:
        record = self.load()
        if record is None:
            raise RuntimeError("blocked_prospective_activation_missing")
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        observed = recorded_at.astimezone(UTC)
        if observed < record.prospective_collection_start_utc.astimezone(UTC):
            raise ValueError("recorded_at precedes prospective_collection_start")
        return observed
