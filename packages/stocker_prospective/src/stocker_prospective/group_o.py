"""Exact previous-close Group O context identity for frozen M1C scoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stocker_prospective.append_only import write_immutable_json
from stocker_prospective.context import previous_xnys_session
from stocker_prospective.live_bars import xnys_session_bounds

GROUP_O_FEATURE_MANIFEST_SHA256 = "fb2b734ce84e545d6839dc6d537aa73532d733f0e2206e0e0a402f96786f3499"
GROUP_O_REGIME_MAPPING_SHA256 = "a73c7e2c0b9220ac598c7051e7ced77ea0e0cf0a71b769e4a4b42ae7885d2985"
GROUP_O_REVISION_SCHEMA_V1: Literal["frozen-m1c-group-o-late-revision-v1"] = (
    "frozen-m1c-group-o-late-revision-v1"
)
GROUP_O_REVISION_REASON_V1: Literal["late_exact_chain_source_correction"] = (
    "late_exact_chain_source_correction"
)
GROUP_O_IMPLIED_MOVEMENT_REVISION_REASON_V1: Literal[
    "missing_implied_movement_source_correction"
] = "missing_implied_movement_source_correction"
GroupORevisionReason = Literal[
    "late_exact_chain_source_correction",
    "missing_implied_movement_source_correction",
]


def _implied_movement_15m_from_atm_iv(atm_iv: float) -> float:
    if not math.isfinite(atm_iv) or atm_iv <= 0.0:
        raise ValueError("Group O ATM IV source must be finite and positive")
    return atm_iv * math.sqrt(15 / (252 * 390)) * math.sqrt(2.0 / math.pi)


class GroupORevisionCutoffError(ValueError):
    """A revision reached its exact signal-session open before publication."""

    def __init__(self, *, observed_at_utc: datetime, signal_open_utc: datetime) -> None:
        super().__init__("Group O revision must precede the signal session open")
        self.observed_at_utc = observed_at_utc
        self.signal_open_utc = signal_open_utc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_hash(package: FrozenGroupOSessionPackage) -> str:
    return hashlib.sha256(
        _canonical_json(package.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _revision_identity_payload(
    *,
    revision_number: int,
    signal_session: date,
    supersedes_sha256: str,
    created_at_utc: datetime,
    signal_open_utc: datetime,
    reason: GroupORevisionReason,
    implied_movement_atm_iv_by_symbol: Mapping[str, float],
    revised_package_hash: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": GROUP_O_REVISION_SCHEMA_V1,
        "revision_number": revision_number,
        "signal_session": signal_session.isoformat(),
        "supersedes_sha256": supersedes_sha256,
        "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
        "signal_open_utc": signal_open_utc.astimezone(UTC).isoformat(),
        "reason": reason,
        "revised_package_hash": revised_package_hash,
    }
    if implied_movement_atm_iv_by_symbol:
        payload["implied_movement_atm_iv_by_symbol"] = dict(
            sorted(implied_movement_atm_iv_by_symbol.items())
        )
    return payload


def _revision_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"group-o-revision-{digest[:24]}"


def _require_revision_preopen(
    *,
    clock: Callable[[], datetime],
    signal_open_utc: datetime,
) -> datetime:
    observed = clock()
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("Group O revision timestamp must be timezone-aware")
    observed_utc = observed.astimezone(UTC)
    if observed_utc >= signal_open_utc:
        raise GroupORevisionCutoffError(
            observed_at_utc=observed_utc,
            signal_open_utc=signal_open_utc,
        )
    return observed_utc


class FrozenGroupOContext(BaseModel):
    """One stock/session context; invalid chronology can be retained but not scored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1)
    signal_session: date
    required_option_observation_session: date
    actual_option_observation_session: date | None
    front_expiry: date | None
    dte: int | None
    atm_strike: float | None
    previous_close_implied_movement_15m: float | None = None
    features: dict[str, float | int | bool | None]
    missing_indicators: dict[str, bool]
    quality_status: str
    source_receipt_hashes: tuple[str, ...]
    context_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    eligible: bool
    rejection_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> FrozenGroupOContext:
        if self.required_option_observation_session != previous_xnys_session(self.signal_session):
            raise ValueError("required Group O session is not the exact prior XNYS session")
        if self.eligible and self.rejection_reasons:
            raise ValueError("eligible Group O context cannot carry rejection reasons")
        if not self.eligible and not self.rejection_reasons:
            raise ValueError("ineligible Group O context requires rejection reasons")
        return self


def build_group_o_context(
    *,
    symbol: str,
    signal_session: date,
    actual_option_observation_session: date | None,
    front_expiry: date | None,
    dte: int | None,
    atm_strike: float | None,
    previous_close_implied_movement_15m: float | None = None,
    features: dict[str, float | int | bool | None],
    missing_indicators: dict[str, bool],
    quality_status: str,
    source_receipt_hashes: tuple[str, ...],
) -> FrozenGroupOContext:
    """Apply exact D-1 chronology and hash the frozen context provenance."""

    required = previous_xnys_session(signal_session)
    reasons: list[str] = []
    if actual_option_observation_session is None:
        reasons.append("group_o_observation_missing")
    elif actual_option_observation_session == signal_session:
        reasons.append("same_day_group_o_rejected")
    elif actual_option_observation_session != required:
        reasons.append("stale_or_future_group_o_rejected")
    if not features:
        reasons.append("group_o_features_missing")
    if front_expiry is None or dte is None or atm_strike is None:
        reasons.append("group_o_front_pair_context_incomplete")
    elif dte < 0 or not math.isfinite(atm_strike) or atm_strike <= 0.0:
        reasons.append("group_o_front_pair_context_invalid")
    if not source_receipt_hashes:
        reasons.append("group_o_source_receipt_missing")
    if quality_status != "valid":
        reasons.append(f"group_o_quality:{quality_status}")
    payload: dict[str, Any] = {
        "symbol": symbol,
        "signal_session": signal_session.isoformat(),
        "required_option_observation_session": required.isoformat(),
        "actual_option_observation_session": (
            None
            if actual_option_observation_session is None
            else actual_option_observation_session.isoformat()
        ),
        "front_expiry": None if front_expiry is None else front_expiry.isoformat(),
        "dte": dte,
        "atm_strike": atm_strike,
        "previous_close_implied_movement_15m": previous_close_implied_movement_15m,
        "features": features,
        "missing_indicators": missing_indicators,
        "quality_status": quality_status,
        "source_receipt_hashes": source_receipt_hashes,
    }
    context_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return FrozenGroupOContext(
        **payload,
        context_hash=context_hash,
        eligible=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
    )


class FrozenGroupOSessionPackage(BaseModel):
    """Prebuilt D-1 contexts produced by the authorised existing Group-O pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str
    signal_session: date
    generated_from_authorised_cache: bool
    feature_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    regime_mapping_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    contexts: tuple[FrozenGroupOContext, ...]

    @model_validator(mode="after")
    def _complete_identity(self) -> FrozenGroupOSessionPackage:
        if self.contract_version != "frozen-m1c-microstructure-recorder-v0/group-o-session-v0":
            raise ValueError("Group O session package version differs")
        if not self.generated_from_authorised_cache:
            raise ValueError("Group O package is not from the authorised cache")
        if self.feature_manifest_hash != GROUP_O_FEATURE_MANIFEST_SHA256:
            raise ValueError("Group O package feature manifest hash differs")
        if self.regime_mapping_hash != GROUP_O_REGIME_MAPPING_SHA256:
            raise ValueError("Group O package regime mapping hash differs")
        if any(item.signal_session != self.signal_session for item in self.contexts):
            raise ValueError("Group O package mixes signal sessions")
        symbols = [item.symbol for item in self.contexts]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Group O package contains duplicate stock contexts")
        return self

    def for_symbol(self, symbol: str) -> FrozenGroupOContext:
        matches = [item for item in self.contexts if item.symbol == symbol]
        if len(matches) != 1:
            raise ValueError(f"Group O context unavailable for {symbol}")
        return matches[0]


class FrozenGroupOSessionRevision(BaseModel):
    """One append-only, pre-signal correction of unavailable source input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["frozen-m1c-group-o-late-revision-v1"]
    revision_number: int = Field(ge=1)
    revision_id: str = Field(pattern=r"^group-o-revision-[a-f0-9]{24}$")
    signal_session: date
    supersedes_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at_utc: datetime
    signal_open_utc: datetime
    reason: GroupORevisionReason
    implied_movement_atm_iv_by_symbol: dict[str, float] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    revised_package_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    package: FrozenGroupOSessionPackage
    revision_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("created_at_utc", "signal_open_utc")
    @classmethod
    def _timestamp_is_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Group O revision timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("implied_movement_atm_iv_by_symbol")
    @classmethod
    def _atm_iv_sources_are_valid(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not symbol or not math.isfinite(atm_iv) or atm_iv <= 0.0
            for symbol, atm_iv in value.items()
        ):
            raise ValueError("Group O revision ATM IV source identity is invalid")
        return value

    @model_validator(mode="after")
    def _identity_is_self_binding_and_causal(self) -> FrozenGroupOSessionRevision:
        expected_open, _ = xnys_session_bounds(self.signal_session)
        if self.signal_open_utc != expected_open:
            raise ValueError("Group O revision signal-open identity differs")
        if self.created_at_utc >= self.signal_open_utc:
            raise ValueError("Group O revision must precede the signal session open")
        if self.package.signal_session != self.signal_session:
            raise ValueError("Group O revision package signal session differs")
        if any(
            context.quality_status == "missing_exact_chain" for context in self.package.contexts
        ):
            raise ValueError("Group O revision cannot finalize a missing exact chain")
        package_hash = _package_hash(self.package)
        if self.revised_package_hash != package_hash:
            raise ValueError("Group O revision package hash differs")
        if self.reason == GROUP_O_IMPLIED_MOVEMENT_REVISION_REASON_V1:
            if not self.implied_movement_atm_iv_by_symbol:
                raise ValueError("Group O implied-movement revision requires ATM IV source")
            for symbol, atm_iv in self.implied_movement_atm_iv_by_symbol.items():
                context = self.package.for_symbol(symbol)
                expected_movement = _implied_movement_15m_from_atm_iv(atm_iv)
                observed_movement = context.previous_close_implied_movement_15m
                if observed_movement is None or not math.isclose(
                    observed_movement,
                    expected_movement,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    raise ValueError("Group O implied movement differs from signed ATM IV")
        elif self.implied_movement_atm_iv_by_symbol:
            raise ValueError("Group O exact-chain revision cannot carry ATM IV correction")
        identity = _revision_identity_payload(
            revision_number=self.revision_number,
            signal_session=self.signal_session,
            supersedes_sha256=self.supersedes_sha256,
            created_at_utc=self.created_at_utc,
            signal_open_utc=self.signal_open_utc,
            reason=self.reason,
            implied_movement_atm_iv_by_symbol=(self.implied_movement_atm_iv_by_symbol),
            revised_package_hash=self.revised_package_hash,
        )
        if self.revision_id != _revision_id(identity):
            raise ValueError("Group O revision ID differs")
        signed_payload = {
            **identity,
            "revision_id": self.revision_id,
            "package": self.package.model_dump(mode="json"),
        }
        expected_hash = hashlib.sha256(_canonical_json(signed_payload).encode("utf-8")).hexdigest()
        if self.revision_sha256 != expected_hash:
            raise ValueError("Group O revision hash differs")
        return self


def requires_group_o_implied_movement_correction(context: FrozenGroupOContext) -> bool:
    return context.quality_status == "valid" and context.previous_close_implied_movement_15m is None


def _assert_implied_movement_correction_compatible(
    before: FrozenGroupOContext,
    after: FrozenGroupOContext,
) -> None:
    movement = after.previous_close_implied_movement_15m
    if movement is None or not math.isfinite(movement) or movement <= 0.0:
        raise ValueError("Group O revision cannot retain missing implied movement")
    before_frozen = before.model_dump(mode="python")
    after_frozen = after.model_dump(mode="python")
    for field in (
        "previous_close_implied_movement_15m",
        "source_receipt_hashes",
        "context_hash",
    ):
        before_frozen.pop(field)
        after_frozen.pop(field)
    if after_frozen != before_frozen:
        raise ValueError("Group O implied-movement revision changed frozen context")
    before_receipts = set(before.source_receipt_hashes)
    after_receipts = set(after.source_receipt_hashes)
    if not before_receipts <= after_receipts:
        raise ValueError("Group O implied-movement revision must retain source provenance")
    rebuilt = build_group_o_context(
        symbol=after.symbol,
        signal_session=after.signal_session,
        actual_option_observation_session=after.actual_option_observation_session,
        front_expiry=after.front_expiry,
        dte=after.dte,
        atm_strike=after.atm_strike,
        previous_close_implied_movement_15m=movement,
        features=after.features,
        missing_indicators=after.missing_indicators,
        quality_status=after.quality_status,
        source_receipt_hashes=after.source_receipt_hashes,
    )
    if after != rebuilt:
        raise ValueError("Group O implied-movement revision context hash differs")


def _assert_revision_compatible(
    previous: FrozenGroupOSessionPackage,
    revised: FrozenGroupOSessionPackage,
    *,
    reason: GroupORevisionReason,
) -> None:
    if (
        revised.contract_version != previous.contract_version
        or revised.signal_session != previous.signal_session
        or revised.generated_from_authorised_cache != previous.generated_from_authorised_cache
        or revised.feature_manifest_hash != previous.feature_manifest_hash
        or revised.regime_mapping_hash != previous.regime_mapping_hash
    ):
        raise ValueError("Group O revision frozen package identity differs")
    previous_symbols = tuple(context.symbol for context in previous.contexts)
    revised_symbols = tuple(context.symbol for context in revised.contexts)
    if revised_symbols != previous_symbols:
        raise ValueError("Group O revision symbol identity or ordering differs")
    has_missing_exact_chain = any(
        context.quality_status == "missing_exact_chain" for context in previous.contexts
    )
    has_missing_implied_movement = any(
        requires_group_o_implied_movement_correction(context) for context in previous.contexts
    )
    if reason == GROUP_O_REVISION_REASON_V1 and not has_missing_exact_chain:
        raise ValueError("Group O exact-chain revision reason does not match source state")
    if reason == GROUP_O_IMPLIED_MOVEMENT_REVISION_REASON_V1 and (
        has_missing_exact_chain or not has_missing_implied_movement
    ):
        raise ValueError("Group O implied-movement revision reason does not match source state")
    for before, after in zip(previous.contexts, revised.contexts, strict=True):
        if after.required_option_observation_session != before.required_option_observation_session:
            raise ValueError("Group O revision observation-session identity differs")
        if before.quality_status == "missing_exact_chain":
            if after.quality_status == "missing_exact_chain":
                raise ValueError("Group O revision cannot retain a missing exact chain")
            if (
                after.actual_option_observation_session
                != before.required_option_observation_session
            ):
                raise ValueError("Group O revised context must use the exact D-1 session")
            continue
        if requires_group_o_implied_movement_correction(before):
            if reason == GROUP_O_REVISION_REASON_V1:
                if after != before:
                    raise ValueError("Group O implied movement requires a separate signed revision")
                continue
            _assert_implied_movement_correction_compatible(before, after)
            continue
        if after != before:
            raise ValueError("Group O revision already-resolved context differs")


def _assert_implied_movement_atm_iv_binding(
    previous: FrozenGroupOSessionPackage,
    revised: FrozenGroupOSessionPackage,
    *,
    reason: GroupORevisionReason,
    implied_movement_atm_iv_by_symbol: Mapping[str, float],
) -> None:
    corrected_symbols = {
        before.symbol
        for before, after in zip(previous.contexts, revised.contexts, strict=True)
        if requires_group_o_implied_movement_correction(before) and after != before
    }
    if reason == GROUP_O_REVISION_REASON_V1:
        if implied_movement_atm_iv_by_symbol:
            raise ValueError("Group O exact-chain revision cannot carry ATM IV correction")
        return
    if set(implied_movement_atm_iv_by_symbol) != corrected_symbols:
        raise ValueError("Group O implied-movement revision ATM IV source identity differs")
    for symbol, atm_iv in implied_movement_atm_iv_by_symbol.items():
        context = revised.for_symbol(symbol)
        expected_movement = _implied_movement_15m_from_atm_iv(float(atm_iv))
        observed_movement = context.previous_close_implied_movement_15m
        if observed_movement is None or not math.isclose(
            observed_movement,
            expected_movement,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("Group O implied movement differs from signed ATM IV source")


def _load_group_o_revision_chain(
    *,
    context_root: Path,
    signal_session: date,
    base_path: Path,
    base_package: FrozenGroupOSessionPackage,
) -> tuple[tuple[Path, FrozenGroupOSessionRevision], ...]:
    revision_root = context_root / "group-o" / "revisions" / signal_session.isoformat()
    if not revision_root.exists():
        return ()
    if not revision_root.is_dir() or revision_root.is_symlink():
        raise ValueError("Group O revision root is invalid")
    parsed: list[tuple[Path, FrozenGroupOSessionRevision]] = []
    for path in revision_root.glob("*.json"):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Group O revision file is invalid")
        try:
            revision = FrozenGroupOSessionRevision.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ValueError(f"Group O revision is invalid: {path.name}") from exc
        expected_name = f"{revision.revision_number:04d}.json"
        if path.name != expected_name:
            raise ValueError("Group O revision filename identity differs")
        parsed.append((path, revision))
    parsed.sort(key=lambda item: item[1].revision_number)
    current_hash = _sha256_path(base_path)
    current_package = base_package
    for expected_number, (path, revision) in enumerate(parsed, start=1):
        if revision.revision_number != expected_number:
            raise ValueError("Group O revision chain is not contiguous")
        if revision.signal_session != signal_session:
            raise ValueError("Group O revision chain mixes signal sessions")
        if revision.supersedes_sha256 != current_hash:
            raise ValueError("Group O revision supersedes hash differs")
        _assert_revision_compatible(
            current_package,
            revision.package,
            reason=revision.reason,
        )
        _assert_implied_movement_atm_iv_binding(
            current_package,
            revision.package,
            reason=revision.reason,
            implied_movement_atm_iv_by_symbol=(revision.implied_movement_atm_iv_by_symbol),
        )
        current_hash = _sha256_path(path)
        current_package = revision.package
    return tuple(parsed)


def append_group_o_session_revision(
    *,
    context_root: str | Path,
    revised_package: FrozenGroupOSessionPackage,
    implied_movement_atm_iv_by_symbol: Mapping[str, float] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FrozenGroupOSessionRevision:
    """Append one causally bounded source correction without replacing the base."""

    root = Path(context_root)
    base_path = root / "group-o" / f"{revised_package.signal_session.isoformat()}.json"
    if not base_path.is_file() or base_path.is_symlink():
        raise ValueError("Group O revision requires an immutable base package")
    base_package = FrozenGroupOSessionPackage.model_validate_json(
        base_path.read_text(encoding="utf-8")
    )
    chain = _load_group_o_revision_chain(
        context_root=root,
        signal_session=revised_package.signal_session,
        base_path=base_path,
        base_package=base_package,
    )
    previous_path, previous_package = (
        (base_path, base_package) if not chain else (chain[-1][0], chain[-1][1].package)
    )
    if previous_package == revised_package and chain:
        return chain[-1][1]
    has_missing_exact_chain = any(
        context.quality_status == "missing_exact_chain" for context in previous_package.contexts
    )
    has_missing_implied_movement = any(
        requires_group_o_implied_movement_correction(context)
        for context in previous_package.contexts
    )
    if has_missing_exact_chain:
        reason: GroupORevisionReason = GROUP_O_REVISION_REASON_V1
    elif has_missing_implied_movement:
        reason = GROUP_O_IMPLIED_MOVEMENT_REVISION_REASON_V1
    else:
        raise ValueError("Group O resolved package does not require a source revision")
    _assert_revision_compatible(
        previous_package,
        revised_package,
        reason=reason,
    )
    atm_iv_sources = dict(implied_movement_atm_iv_by_symbol or {})
    _assert_implied_movement_atm_iv_binding(
        previous_package,
        revised_package,
        reason=reason,
        implied_movement_atm_iv_by_symbol=atm_iv_sources,
    )
    if any(context.quality_status == "missing_exact_chain" for context in revised_package.contexts):
        raise ValueError("Group O revision cannot finalize a missing exact chain")
    signal_open, _ = xnys_session_bounds(revised_package.signal_session)
    created = _require_revision_preopen(clock=clock, signal_open_utc=signal_open)
    package_hash = _package_hash(revised_package)
    revision_number = len(chain) + 1
    identity = _revision_identity_payload(
        revision_number=revision_number,
        signal_session=revised_package.signal_session,
        supersedes_sha256=_sha256_path(previous_path),
        created_at_utc=created,
        signal_open_utc=signal_open,
        reason=reason,
        implied_movement_atm_iv_by_symbol=atm_iv_sources,
        revised_package_hash=package_hash,
    )
    revision_id = _revision_id(identity)
    signed_payload = {
        **identity,
        "revision_id": revision_id,
        "package": revised_package.model_dump(mode="json"),
    }
    revision = FrozenGroupOSessionRevision(
        schema_version=GROUP_O_REVISION_SCHEMA_V1,
        revision_number=revision_number,
        revision_id=revision_id,
        signal_session=revised_package.signal_session,
        supersedes_sha256=_sha256_path(previous_path),
        created_at_utc=created,
        signal_open_utc=signal_open,
        reason=reason,
        implied_movement_atm_iv_by_symbol=atm_iv_sources,
        revised_package_hash=package_hash,
        package=revised_package,
        revision_sha256=hashlib.sha256(_canonical_json(signed_payload).encode("utf-8")).hexdigest(),
    )
    destination = (
        root
        / "group-o"
        / "revisions"
        / revised_package.signal_session.isoformat()
        / f"{revision_number:04d}.json"
    )

    def require_link_preopen() -> None:
        _require_revision_preopen(clock=clock, signal_open_utc=signal_open)

    write_immutable_json(
        destination,
        revision.model_dump(mode="json"),
        conflict_message="immutable Group O revision slot differs",
        before_link=require_link_preopen,
    )
    verified_chain = _load_group_o_revision_chain(
        context_root=root,
        signal_session=revised_package.signal_session,
        base_path=base_path,
        base_package=base_package,
    )
    if not verified_chain or verified_chain[-1][1].revision_number != revision_number:
        raise ValueError("Group O revision publication did not extend the verified chain")
    return verified_chain[-1][1]


def load_group_o_session_package(
    *,
    context_root: str | Path,
    signal_session: date,
) -> FrozenGroupOSessionPackage:
    """Load the explicit session file only; never discover a newest package."""

    path = Path(context_root) / "group-o" / f"{signal_session.isoformat()}.json"
    if not path.is_file():
        raise ValueError(
            "blocked_missing_previous_session_options_context: "
            f"no Group O package mapped for {signal_session.isoformat()}"
        )
    if path.is_symlink():
        raise ValueError("Group O package cannot be a symlink")
    package = FrozenGroupOSessionPackage.model_validate_json(path.read_text(encoding="utf-8"))
    if package.signal_session != signal_session:
        raise ValueError("Group O package signal session differs")
    chain = _load_group_o_revision_chain(
        context_root=Path(context_root),
        signal_session=signal_session,
        base_path=path,
        base_package=package,
    )
    return package if not chain else chain[-1][1].package


__all__ = [
    "FrozenGroupOContext",
    "FrozenGroupOSessionPackage",
    "FrozenGroupOSessionRevision",
    "GroupORevisionCutoffError",
    "GROUP_O_FEATURE_MANIFEST_SHA256",
    "GROUP_O_IMPLIED_MOVEMENT_REVISION_REASON_V1",
    "GROUP_O_REGIME_MAPPING_SHA256",
    "GROUP_O_REVISION_REASON_V1",
    "GROUP_O_REVISION_SCHEMA_V1",
    "append_group_o_session_revision",
    "build_group_o_context",
    "load_group_o_session_package",
    "requires_group_o_implied_movement_correction",
]
