"""Frozen-package verification and bounded IBKR option evidence for OLC V0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from pydantic import ConfigDict, Field, field_validator

from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.opening_leader_continuation_v0 import (
    CANONICAL_COHORT_HASH_V0,
    CANONICAL_COHORT_V0,
    RECORDER_VERSION_V0,
    OpeningLeaderFreezeIdentityV0,
    OptionQuoteV0,
    OptionSnapshotCaptureV0,
    select_option_chain_requests_v0,
)

NEW_YORK = ZoneInfo("America/New_York")
PACKAGE_SCHEMA_V0: Literal["opening-leader-continuation-deployment-freeze-v0"] = (
    "opening-leader-continuation-deployment-freeze-v0"
)
SIGNATURE_SCHEME_V0: Literal["sha256-canonical-self-binding-v0"] = (
    "sha256-canonical-self-binding-v0"
)
REQUIRED_ARTIFACTS_V0 = (
    "README.md",
    "checkpoint_manifest.json",
    "cohort_manifest.json",
    "contract.json",
    "future_evaluation_contract.json",
    "options_capture_manifest.json",
    "order_disable_audit.json",
    "protected_boundary_audit.json",
    "quote_capture_manifest.json",
    "rank_manifest.json",
)
REQUIRED_VERIFICATION_V0 = frozenset(
    {
        "focused_tests",
        "prospective_recorder_tests",
        "ibkr_shadow_tests",
        "dashboard_tests",
        "lint",
        "type_check",
        "synthetic_dry_run",
        "restart_recovery",
    }
)


def assert_opening_leader_runtime_configuration_v0(
    *,
    mode: str,
    maximum_quote_age_seconds: float,
    trading_enabled: bool,
) -> None:
    """Reject runtime drift from the signed record-only capture contract."""

    if mode != "record_only":
        raise ValueError("opening-leader V0 requires record_only runtime mode")
    if maximum_quote_age_seconds != 2.0:
        raise ValueError("opening-leader V0 quote-age tolerance is frozen at 2 seconds")
    if trading_enabled:
        raise ValueError("opening-leader V0 requires order placement to remain disabled")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _mapping_hash(value: Mapping[str, str]) -> str:
    return _sha256_bytes(_canonical_json(dict(value)).encode("utf-8"))


def _aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


class OpeningLeaderDeploymentReceiptV0(OpeningLeaderFreezeIdentityV0):
    """Self-binding start receipt checked before this module may consume data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["opening-leader-continuation-deployment-freeze-v0"]
    recorder_version: Literal["opening-leader-continuation-recorder-v0"]
    artifact_hashes: dict[str, str]
    source_hashes: dict[str, str]
    verification: dict[str, Literal["passed"]]
    signature_scheme: Literal["sha256-canonical-self-binding-v0"]
    signature_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("artifact_hashes", "source_hashes")
    @classmethod
    def _hashes_are_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in value.values()
        ):
            raise ValueError("deployment receipt hashes must be lowercase SHA-256 values")
        return value


def _validate_package_contract(package_root: Path) -> None:
    contract = json.loads((package_root / "contract.json").read_text(encoding="utf-8"))
    cohort = json.loads((package_root / "cohort_manifest.json").read_text(encoding="utf-8"))
    order_audit = json.loads(
        (package_root / "order_disable_audit.json").read_text(encoding="utf-8")
    )
    protected_audit = json.loads(
        (package_root / "protected_boundary_audit.json").read_text(encoding="utf-8")
    )
    if (
        contract.get("recorder_version") != RECORDER_VERSION_V0
        or contract.get("record_only") is not True
        or contract.get("primary_checkpoint") != "C6"
        or contract.get("secondary_checkpoint") != "C12"
        or contract.get("checkpoint_pooling_allowed") is not False
        or contract.get("orders", {}).get("submission_allowed") is not False
    ):
        raise ValueError("frozen opening-leader contract semantics differ")
    if (
        tuple(cohort.get("symbols", ())) != CANONICAL_COHORT_V0
        or cohort.get("universe_hash") != CANONICAL_COHORT_HASH_V0
    ):
        raise ValueError("frozen opening-leader cohort identity differs")
    if (
        order_audit.get("orders_disabled") is not True
        or order_audit.get("order_methods_available_to_module") is not False
    ):
        raise ValueError("opening-leader order-disable audit failed")
    if protected_audit.get("protected_historical_outcomes_accessed") is not False:
        raise ValueError("opening-leader protected-boundary audit failed")


def _signature_payload(receipt: OpeningLeaderDeploymentReceiptV0) -> dict[str, object]:
    return receipt.model_dump(mode="json", exclude={"signature_sha256"})


def freeze_opening_leader_package_v0(
    package_root: str | Path,
    *,
    freeze_completed_at_utc: datetime,
    source_files: Mapping[str, str | Path],
    verification: Mapping[str, str],
) -> OpeningLeaderDeploymentReceiptV0:
    """Write the one-time receipt only after every named verification passed."""

    root = Path(package_root)
    receipt_path = root / "deployment_freeze_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("deployment freeze receipt is immutable and already exists")
    absent = tuple(name for name in REQUIRED_ARTIFACTS_V0 if not (root / name).is_file())
    if absent:
        raise ValueError("frozen opening-leader artifacts absent: " + ",".join(absent))
    if set(verification) != REQUIRED_VERIFICATION_V0 or set(verification.values()) != {"passed"}:
        raise ValueError("deployment receipt requires every frozen verification to pass")
    _validate_package_contract(root)
    artifact_hashes = {name: _sha256_path(root / name) for name in REQUIRED_ARTIFACTS_V0}
    source_paths = {name: Path(path) for name, path in source_files.items()}
    if not source_paths or any(not path.is_file() for path in source_paths.values()):
        raise ValueError("deployment source file set is absent or incomplete")
    source_hashes = {name: _sha256_path(path) for name, path in sorted(source_paths.items())}
    frozen_at = _aware(freeze_completed_at_utc, label="deployment freeze timestamp")
    unsigned_identity = {
        "schema_version": PACKAGE_SCHEMA_V0,
        "recorder_version": RECORDER_VERSION_V0,
        "freeze_completed_at_utc": frozen_at.isoformat(),
        "artifact_hashes": artifact_hashes,
        "source_hashes": source_hashes,
        "verification": dict(sorted(verification.items())),
    }
    deployment_receipt_id = "olc-deploy-" + _sha256_bytes(
        _canonical_json(unsigned_identity).encode("utf-8")
    )[:24]
    provisional = OpeningLeaderDeploymentReceiptV0(
        schema_version=PACKAGE_SCHEMA_V0,
        recorder_version=RECORDER_VERSION_V0,
        deployment_receipt_id=deployment_receipt_id,
        freeze_completed_at_utc=frozen_at,
        contract_hash=artifact_hashes["contract.json"],
        code_hash=_mapping_hash(source_hashes),
        cohort_hash=CANONICAL_COHORT_HASH_V0,
        source_hashes_signed=True,
        order_routing_disabled=True,
        protected_historical_outcomes_accessed=False,
        artifact_hashes=artifact_hashes,
        source_hashes=source_hashes,
        verification=cast(dict[str, Literal["passed"]], dict(verification)),
        signature_scheme=SIGNATURE_SCHEME_V0,
        signature_sha256="0" * 64,
    )
    signature = _sha256_bytes(
        _canonical_json(_signature_payload(provisional)).encode("utf-8")
    )
    receipt = provisional.model_copy(update={"signature_sha256": signature})
    receipt_path.write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def load_opening_leader_package_v0(
    package_root: str | Path,
    *,
    prospective_start_utc: datetime,
    source_files: Mapping[str, str | Path],
) -> OpeningLeaderDeploymentReceiptV0:
    """Fail closed on any artifact, source, safety, or time-boundary drift."""

    root = Path(package_root)
    receipt_path = root / "deployment_freeze_receipt.json"
    absent = tuple(
        name
        for name in (*REQUIRED_ARTIFACTS_V0, receipt_path.name)
        if not (root / name).is_file()
    )
    if absent:
        raise ValueError("frozen opening-leader package incomplete: " + ",".join(absent))
    _validate_package_contract(root)
    receipt = OpeningLeaderDeploymentReceiptV0.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    start = _aware(prospective_start_utc, label="prospective start")
    if start < receipt.freeze_completed_at_utc:
        raise ValueError("prospective start precedes the deployment freeze receipt")
    observed_artifacts = {
        name: _sha256_path(root / name) for name in REQUIRED_ARTIFACTS_V0
    }
    if observed_artifacts != receipt.artifact_hashes:
        raise ValueError("opening-leader artifact hash mismatch")
    observed_sources = {
        name: _sha256_path(Path(path)) for name, path in sorted(source_files.items())
    }
    if observed_sources != receipt.source_hashes:
        raise ValueError("opening-leader source hash mismatch")
    if (
        receipt.contract_hash != observed_artifacts["contract.json"]
        or receipt.code_hash != _mapping_hash(observed_sources)
        or receipt.cohort_hash != CANONICAL_COHORT_HASH_V0
    ):
        raise ValueError("opening-leader deployment identity mismatch")
    expected_signature = _sha256_bytes(
        _canonical_json(_signature_payload(receipt)).encode("utf-8")
    )
    if expected_signature != receipt.signature_sha256:
        raise ValueError("opening-leader deployment receipt signature mismatch")
    if set(receipt.verification) != REQUIRED_VERIFICATION_V0 or set(
        receipt.verification.values()
    ) != {"passed"}:
        raise ValueError("opening-leader deployment verification is incomplete")
    return receipt


def opening_leader_runtime_source_files_v0() -> dict[str, Path]:
    """Return the complete installed runtime surface bound by the receipt."""

    package = Path(__file__).parent
    repository_root = package.parents[3]
    return {
        "cli": package / "cli.py",
        "config": package / "config.py",
        "contract_safety": package / "contract.py",
        "database": package / "database.py",
        "event_ingest": package / "event_ingest.py",
        "events": package / "events.py",
        "frozen_live_application": package / "frozen_live_application.py",
        "ibkr": package / "ibkr.py",
        "ibkr_official": package / "ibkr_official.py",
        "live_bars": package / "live_bars.py",
        "live_recorder": package / "live_recorder.py",
        "live_subscriptions": package / "live_subscriptions.py",
        "market_data": package / "market_data.py",
        "migration_0016": package / "migrations" / "0016_opening_leader_continuation_v0.sql",
        "opening_leader_contract": package / "opening_leader_continuation_v0.py",
        "opening_leader_live": package / "opening_leader_live_v0.py",
        "option_discovery": package / "option_discovery.py",
        "partition_store": package / "partition_store.py",
        "project_configuration": repository_root / "pyproject.toml",
        "read_store": package / "read_store.py",
        "static_app": package / "web_static" / "app.js",
        "static_index": package / "web_static" / "index.html",
        "static_style": package / "web_static" / "app.css",
        "uv_lock": repository_root / "uv.lock",
        "web": package / "web.py",
    }


def _attribute(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _parse_expiry(value: object) -> date | None:
    raw = str(value)
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _parse_timestamp(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware(value, label="option source timestamp")
    try:
        return _aware(datetime.fromisoformat(str(value)), label="option source timestamp")
    except ValueError:
        return None


def _snapshot_item_values(item: object) -> dict[str, object]:
    if isinstance(item, dict):
        return dict(item)
    raw = getattr(item, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    return {
        name: getattr(item, name)
        for name in (
            "bid",
            "ask",
            "delta",
            "market_data_type",
            "receive_timestamp_utc",
        )
        if hasattr(item, name)
    }


def _merge_opening_leader_option_snapshot_v0(
    items: tuple[object, ...],
    *,
    right: Literal["C", "P"],
) -> tuple[dict[str, object], datetime | None]:
    """Merge callback-shaped snapshots while freezing Greeks to the model source."""

    merged: dict[str, object] = {}
    computations: dict[str, dict[str, float | None]] = {}
    received_timestamps: list[datetime] = []
    computation_fields = (
        "option_price",
        "present_value_dividend",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "underlying_reference_price",
    )
    for item in items:
        values = _snapshot_item_values(item)
        received = _parse_timestamp(values.get("receive_timestamp_utc"))
        if received is not None:
            received_timestamps.append(received)
        callback_field = values.get("field")
        if callback_field == "option_computation":
            source = str(values.get("computation_source", "unknown"))
            if source in {"bid", "ask", "last", "model"}:
                computations[source] = {
                    name: (
                        None
                        if values.get(name) is None
                        else float(cast(Any, values[name]))
                    )
                    for name in computation_fields
                }
        elif isinstance(callback_field, str) and values.get("value") is not None:
            merged[callback_field] = values["value"]
        for name, value in values.items():
            if name in {
                "field",
                "value",
                "computation_source",
                "receive_timestamp_utc",
                *computation_fields,
            } or value is None:
                continue
            merged[str(name)] = value
    model = computations.get("model", {})
    for name in (
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "underlying_reference_price",
    ):
        if model.get(name) is not None:
            merged[name] = model[name]
    merged["option_computation_by_source"] = computations
    open_interest_key = "call_open_interest" if right == "C" else "put_open_interest"
    if merged.get(open_interest_key) is not None:
        merged["open_interest"] = merged[open_interest_key]
    return merged, max(received_timestamps, default=None)


OptionContractFactoryV0 = Callable[[str, date, float, str, int, str, str], object]


class OpeningLeaderIBKROptionSnapshotterV0:
    """Sequential, bounded exact-contract snapshots with no option policy."""

    def __init__(
        self,
        *,
        adapter: Any,
        underlying_contracts: Mapping[str, QualifiedUnderlying],
        contract_factory: OptionContractFactoryV0,
        request_heartbeat: Callable[[], object] | None,
        maximum_quote_age_seconds: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if maximum_quote_age_seconds <= 0.0:
            raise ValueError("maximum option quote age must be positive")
        self.adapter = adapter
        self.underlying_contracts = dict(underlying_contracts)
        self.contract_factory = contract_factory
        self.request_heartbeat = request_heartbeat
        self.maximum_quote_age_seconds = maximum_quote_age_seconds
        self.clock = clock
        self._metadata_cache: dict[
            tuple[str, date],
            tuple[tuple[date, ...], tuple[float, ...], str, str],
        ] = {}

    def _paced(self) -> None:
        if self.request_heartbeat is not None:
            self.request_heartbeat()

    def _metadata(
        self,
        *,
        symbol: str,
        session: date,
        underlying: QualifiedUnderlying,
    ) -> tuple[tuple[date, ...], tuple[float, ...], str, str]:
        key = (symbol, session)
        cached = self._metadata_cache.get(key)
        if cached is not None:
            return cached
        request = self.adapter.request_option_chain_metadata
        result = request(
            underlying_symbol=symbol,
            exchange="",
            underlying_security_type="STK",
            underlying_contract_id=underlying.con_id,
        )
        self._paced()
        candidates = tuple(
            item
            for item in tuple(getattr(result, "items", ()))
            if int(_attribute(item, "underlying_contract_id", "underlyingConId") or 0)
            == underlying.con_id
        )
        if not candidates:
            raise ValueError("option_chain_metadata_unavailable")
        selected = min(
            candidates,
            key=lambda item: (
                0 if str(_attribute(item, "exchange") or "") == "SMART" else 1,
                str(_attribute(item, "exchange") or ""),
                str(_attribute(item, "trading_class", "tradingClass") or ""),
            ),
        )
        expiries = tuple(
            parsed
            for raw in cast(Iterable[object], _attribute(selected, "expirations") or ())
            for parsed in (_parse_expiry(raw),)
            if parsed is not None
        )
        strikes = tuple(
            float(cast(Any, raw))
            for raw in cast(Iterable[object], _attribute(selected, "strikes") or ())
        )
        resolved = (
            expiries,
            strikes,
            str(_attribute(selected, "exchange") or "SMART"),
            str(_attribute(selected, "trading_class", "tradingClass") or symbol),
        )
        self._metadata_cache[key] = resolved
        return resolved

    @staticmethod
    def _qualified_contract(
        result: object,
        *,
        symbol: str,
        expiry: date,
        strike: float,
        right: str,
    ) -> object | None:
        matching: list[object] = []
        for detail in tuple(getattr(result, "items", ())):
            candidate = _attribute(detail, "contract") or detail
            if (
                str(_attribute(candidate, "symbol") or "") == symbol
                and str(_attribute(candidate, "secType", "sec_type") or "") == "OPT"
                and _parse_expiry(_attribute(candidate, "lastTradeDateOrContractMonth", "expiry"))
                == expiry
                and float(_attribute(candidate, "strike") or 0.0) == strike
                and str(_attribute(candidate, "right") or "") == right
                and int(_attribute(candidate, "conId", "con_id") or 0) > 0
            ):
                matching.append(candidate)
        return matching[0] if len(matching) == 1 else None

    def __call__(
        self,
        symbol: str,
        checkpoint: int,
        observation_name: str,
        spot: float,
        observed_at: datetime,
    ) -> OptionSnapshotCaptureV0:
        observed = _aware(observed_at, label="option observation timestamp")
        snapshot_id = "olc-options-" + _sha256_bytes(
            f"{symbol}|C{checkpoint}|{observation_name}|{observed.isoformat()}".encode()
        )[:24]
        underlying = self.underlying_contracts.get(symbol)
        if underlying is None:
            return OptionSnapshotCaptureV0(
                snapshot_id=snapshot_id,
                observation_name=observation_name,
                captured_at_utc=observed,
                status="UNAVAILABLE",
                reason="underlying_contract_unresolved",
                selection=None,
                quotes=(),
            )
        session = observed.astimezone(NEW_YORK).date()
        try:
            expiries, strikes, exchange, trading_class = self._metadata(
                symbol=symbol,
                session=session,
                underlying=underlying,
            )
            selection = select_option_chain_requests_v0(
                underlying=symbol,
                underlying_con_id=underlying.con_id,
                session=session,
                spot=spot,
                available_expiries=expiries,
                available_strikes=strikes,
                exchange=exchange,
                trading_class=trading_class,
            )
        except (AttributeError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            return OptionSnapshotCaptureV0(
                snapshot_id=snapshot_id,
                observation_name=observation_name,
                captured_at_utc=observed,
                status="UNAVAILABLE",
                reason=f"option_metadata_failure:{type(exc).__name__}:{exc}",
                selection=None,
                quotes=(),
            )
        if selection.status == "UNAVAILABLE":
            return OptionSnapshotCaptureV0(
                snapshot_id=snapshot_id,
                observation_name=observation_name,
                captured_at_utc=observed,
                status="UNAVAILABLE",
                reason=selection.reason,
                selection=selection,
                quotes=(),
            )
        quotes: list[OptionQuoteV0] = []
        failures: list[str] = []
        for request in selection.requests:
            try:
                upstream = self.contract_factory(
                    request.underlying,
                    request.expiry,
                    request.strike,
                    request.right,
                    request.multiplier,
                    request.exchange,
                    request.trading_class,
                )
                qualify = self.adapter.qualify_exact_contract
                qualified_result = qualify(upstream)
                self._paced()
                qualified = self._qualified_contract(
                    qualified_result,
                    symbol=symbol,
                    expiry=request.expiry,
                    strike=request.strike,
                    right=request.right,
                )
                if qualified is None:
                    failures.append(
                        f"{request.expiry}:{request.right}:{request.strike}:qualification_unavailable"
                    )
                    continue
                capture = self.adapter.capture_temporary_quote
                result = capture(contract=qualified)
                self._paced()
                captured = _aware(self.clock(), label="option snapshot receive timestamp")
                values, received = _merge_opening_leader_option_snapshot_v0(
                    tuple(getattr(result, "items", ())),
                    right=request.right,
                )
                values.setdefault("underlying_reference_price", spot)
                provider = _parse_timestamp(
                    values.get("provider_timestamp_utc", values.get("timestamp_utc"))
                )
                quotes.append(
                    OptionQuoteV0.from_snapshot(
                        snapshot_id=snapshot_id,
                        underlying=symbol,
                        con_id=int(_attribute(qualified, "conId", "con_id") or 0),
                        right=request.right,
                        strike=request.strike,
                        expiry=request.expiry,
                        multiplier=int(
                            _attribute(qualified, "multiplier") or request.multiplier
                        ),
                        trading_class=str(
                            _attribute(qualified, "tradingClass", "trading_class")
                            or request.trading_class
                        ),
                        exchange=str(
                            _attribute(qualified, "exchange") or request.exchange
                        ),
                        captured_at_utc=captured,
                        provider_timestamp_utc=provider,
                        received_timestamp_utc=received,
                        values=values,
                        maximum_quote_age_seconds=self.maximum_quote_age_seconds,
                    )
                )
            except (AttributeError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                failures.append(
                    f"{request.expiry}:{request.right}:{request.strike}:"
                    f"{type(exc).__name__}:{exc}"
                )
        usable_quotes = tuple(quote for quote in quotes if quote.available)
        quality_failures = [
            f"{quote.expiry}:{quote.right}:{quote.strike}:"
            + ",".join(quote.data_quality_flags)
            for quote in quotes
            if not quote.available
        ]
        all_failures = (*failures, *quality_failures)
        status: Literal["AVAILABLE", "UNAVAILABLE"] = (
            "AVAILABLE" if usable_quotes else "UNAVAILABLE"
        )
        reason = (
            None
            if not all_failures
            else "partial_or_failed_exact_contracts:" + "|".join(all_failures)
        )
        if not usable_quotes and reason is None:
            reason = "option_quotes_unavailable"
        return OptionSnapshotCaptureV0(
            snapshot_id=snapshot_id,
            observation_name=observation_name,
            captured_at_utc=max((quote.captured_at_utc for quote in quotes), default=observed),
            status=status,
            reason=reason,
            selection=selection,
            quotes=tuple(quotes),
        )


__all__ = [
    "OpeningLeaderDeploymentReceiptV0",
    "OpeningLeaderIBKROptionSnapshotterV0",
    "assert_opening_leader_runtime_configuration_v0",
    "freeze_opening_leader_package_v0",
    "load_opening_leader_package_v0",
    "opening_leader_runtime_source_files_v0",
]
