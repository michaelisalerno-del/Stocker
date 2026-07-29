"""Exact previous-close Group O context identity for frozen M1C scoring."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_prospective.context import previous_xnys_session

GROUP_O_FEATURE_MANIFEST_SHA256 = "fb2b734ce84e545d6839dc6d537aa73532d733f0e2206e0e0a402f96786f3499"
GROUP_O_REGIME_MAPPING_SHA256 = "a73c7e2c0b9220ac598c7051e7ced77ea0e0cf0a71b769e4a4b42ae7885d2985"


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
    package = FrozenGroupOSessionPackage.model_validate_json(path.read_text(encoding="utf-8"))
    if package.signal_session != signal_session:
        raise ValueError("Group O package signal session differs")
    return package


__all__ = [
    "FrozenGroupOContext",
    "FrozenGroupOSessionPackage",
    "GROUP_O_FEATURE_MANIFEST_SHA256",
    "GROUP_O_REGIME_MAPPING_SHA256",
    "build_group_o_context",
    "load_group_o_session_package",
]
