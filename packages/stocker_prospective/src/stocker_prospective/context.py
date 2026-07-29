"""Exact previous-session signed options-context packages."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextValidationError(RuntimeError):
    """Daily context package is not eligible for scoring."""


class DailyContextUnsigned(BaseModel):
    """Content signed on the research/data-preparation machine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal["1"]
    context_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
    session_date: date
    provider: str = Field(min_length=1)
    source_record_ids: list[str] = Field(min_length=1)
    created_at_utc: datetime
    schema_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    completeness: Literal["complete"]
    features_by_symbol: dict[str, dict[str, float | int | bool | None]]
    key_id: str = Field(min_length=1)


class SignedDailyContext(DailyContextUnsigned):
    """Integrity- and authenticity-protected daily context import."""

    integrity_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    signature_algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    signature: str = Field(pattern=r"^[a-f0-9]{64}$")


class ImportedContextPointer(BaseModel):
    """Exact current-session pointer; no newest-file discovery is allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_session: date
    required_previous_session: date
    context_id: str
    integrity_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    imported_at_utc: datetime
    operator: str


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _unsigned_bytes(package: DailyContextUnsigned | SignedDailyContext) -> bytes:
    payload = package.model_dump(mode="json")
    payload.pop("integrity_hash", None)
    payload.pop("signature_algorithm", None)
    payload.pop("signature", None)
    return _canonical(payload)


def create_signed_context(
    context: DailyContextUnsigned,
    secret: bytes,
) -> SignedDailyContext:
    """Sign a complete context package without embedding the secret."""

    if not secret:
        raise ContextValidationError("context signing secret must not be empty")
    content = _unsigned_bytes(context)
    return SignedDailyContext(
        **context.model_dump(),
        integrity_hash=hashlib.sha256(content).hexdigest(),
        signature=hmac.new(secret, content, hashlib.sha256).hexdigest(),
    )


def previous_xnys_session(current_session: date) -> date:
    """Return the exact prior valid XNYS session."""

    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar("XNYS")
    schedule = calendar.schedule(
        start_date=current_session - timedelta(days=14),
        end_date=current_session - timedelta(days=1),
    )
    if schedule.empty:
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: calendar returned no prior session"
        )
    return date.fromisoformat(str(schedule.index[-1])[:10])


def verify_signed_context(
    package: SignedDailyContext,
    *,
    current_session: date,
    secret: bytes,
    expected_schema_hash: str | None = None,
    expected_feature_hash: str | None = None,
    expected_symbols: tuple[str, ...] | None = None,
) -> SignedDailyContext:
    """Validate signature, hashes, completeness, and exact D-1 session identity."""

    content = _unsigned_bytes(package)
    integrity = hashlib.sha256(content).hexdigest()
    signature = hmac.new(secret, content, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(integrity, package.integrity_hash) or not hmac.compare_digest(
        signature, package.signature
    ):
        raise ContextValidationError("invalid_context_signature")
    required = previous_xnys_session(current_session)
    if package.session_date != required:
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: "
            f"required {required.isoformat()}, observed {package.session_date.isoformat()}"
        )
    if expected_schema_hash is not None and package.schema_hash != expected_schema_hash:
        raise ContextValidationError("blocked_feature_schema_mismatch")
    if expected_feature_hash is not None and package.feature_hash != expected_feature_hash:
        raise ContextValidationError("blocked_feature_schema_mismatch")
    if package.completeness != "complete" or not package.features_by_symbol:
        raise ContextValidationError("blocked_missing_previous_session_options_context")
    if expected_symbols is not None and (
        set(package.features_by_symbol) != set(expected_symbols)
        or any(not package.features_by_symbol[symbol] for symbol in expected_symbols)
    ):
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: anchor symbol context is incomplete"
        )
    return package


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def import_signed_context(
    package_path: str | Path,
    *,
    context_root: str | Path,
    current_session: date,
    secret: bytes,
    operator: str,
    expected_schema_hash: str | None = None,
    expected_feature_hash: str | None = None,
    expected_symbols: tuple[str, ...] | None = None,
) -> SignedDailyContext:
    """Install and atomically map one signed package to one exact US session."""

    if not operator.strip():
        raise ContextValidationError("context import operator identity is required")
    try:
        package = SignedDailyContext.model_validate_json(
            Path(package_path).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ContextValidationError("invalid_context_package") from exc
    verified = verify_signed_context(
        package,
        current_session=current_session,
        secret=secret,
        expected_schema_hash=expected_schema_hash,
        expected_feature_hash=expected_feature_hash,
        expected_symbols=expected_symbols,
    )
    root = Path(context_root)
    installed = root / "installed" / f"{verified.context_id}.json"
    canonical_package = (
        json.dumps(
            verified.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    if installed.exists():
        if installed.read_text(encoding="utf-8") != canonical_package:
            raise ContextValidationError("context_id_collision")
    else:
        _atomic_json(installed, verified.model_dump(mode="json"))

    pointer_path = root / "sessions" / f"{current_session.isoformat()}.json"
    pointer = ImportedContextPointer(
        current_session=current_session,
        required_previous_session=previous_xnys_session(current_session),
        context_id=verified.context_id,
        integrity_hash=verified.integrity_hash,
        imported_at_utc=datetime.now().astimezone(),
        operator=operator,
    )
    if pointer_path.exists():
        existing = ImportedContextPointer.model_validate_json(
            pointer_path.read_text(encoding="utf-8")
        )
        if (
            existing.context_id != pointer.context_id
            or existing.integrity_hash != pointer.integrity_hash
        ):
            raise ContextValidationError("context_session_already_mapped_to_different_package")
    else:
        _atomic_json(pointer_path, pointer.model_dump(mode="json"))
        with (root / "operator-actions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "action": "import_daily_context",
                        **pointer.model_dump(mode="json"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    return verified


def load_imported_context(
    *,
    context_root: str | Path,
    current_session: date,
    secret: bytes,
    expected_schema_hash: str | None = None,
    expected_feature_hash: str | None = None,
    expected_symbols: tuple[str, ...] | None = None,
) -> SignedDailyContext:
    """Load only the explicit pointer for ``current_session`` and reverify it."""

    root = Path(context_root)
    pointer_path = root / "sessions" / f"{current_session.isoformat()}.json"
    if not pointer_path.is_file():
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: "
            f"no import mapped for {current_session.isoformat()}"
        )
    try:
        pointer = ImportedContextPointer.model_validate_json(
            pointer_path.read_text(encoding="utf-8")
        )
        package = SignedDailyContext.model_validate_json(
            (root / "installed" / f"{pointer.context_id}.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: invalid installed package"
        ) from exc
    verified = verify_signed_context(
        package,
        current_session=current_session,
        secret=secret,
        expected_schema_hash=expected_schema_hash,
        expected_feature_hash=expected_feature_hash,
        expected_symbols=expected_symbols,
    )
    if not hmac.compare_digest(pointer.integrity_hash, verified.integrity_hash):
        raise ContextValidationError("invalid_context_signature")
    return verified
