"""Frozen, collection-only all-universe Level-I microstructure shadow V0."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo

from stocker_prospective.capacity import RuntimeCapacityManifest
from stocker_prospective.config import ShadowMicrostructureConfig
from stocker_prospective.events import (
    RawCallbackEnvelopeEvent,
    RawEvent,
    ShadowBBOEvent,
    ShadowGapEvent,
    ShadowSessionSegment,
    ShadowTradeEvent,
    ShadowTradeStreamStateEvent,
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickTradeEvent,
)
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.partition_store import PartitionedEventStore, PartitionWriteResult

NEW_YORK = ZoneInfo("America/New_York")
BBO_FIELDS = frozenset({"bid", "ask", "bid_size", "ask_size"})
PERMISSION_ERROR_CODES = frozenset({354, 10089, 10090, 10186, 10197})
FORBIDDEN_RAW_RESEARCH_FIELDS = frozenset(
    {
        "future_return",
        "next_price_direction",
        "mfe",
        "mae",
        "ofi",
        "imbalance",
        "weighted_mid",
        "microprice",
        "absorption_score",
        "continuation_score",
        "inferred_aggressor_side",
    }
)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable shadow artifact differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    _write_immutable_bytes(path, encoded)


def _planned_sessions(activation: datetime, count: int) -> tuple[date, ...]:
    """Return complete XNYS sessions whose regular open is after activation."""

    import pandas_market_calendars as mcal

    start = activation.astimezone(NEW_YORK).date()
    schedule = mcal.get_calendar("XNYS").schedule(
        start_date=start,
        end_date=start + timedelta(days=max(60, count * 4)),
    )
    sessions = tuple(
        index.date()
        for index, row in schedule.iterrows()
        if row["market_open"].to_pydatetime().astimezone(UTC) >= activation.astimezone(UTC)
    )
    if len(sessions) < count:
        raise RuntimeError("unable to resolve 20 complete XNYS shadow sessions")
    return sessions[:count]


def session_segment(timestamp: datetime) -> ShadowSessionSegment:
    observed = timestamp.astimezone(UTC)
    local_session = observed.astimezone(NEW_YORK).date()
    try:
        market_open, market_close = xnys_session_bounds(local_session)
    except ValueError:
        return ShadowSessionSegment.CLOSED
    if observed < market_open:
        return ShadowSessionSegment.PREMARKET
    if observed < market_close:
        return ShadowSessionSegment.RTH
    return ShadowSessionSegment.AFTER_HOURS


@dataclass(frozen=True)
class ShadowCapacityAssessment:
    total_line_allowance: int
    externally_reserved_lines: int
    preexisting_internal_lines: int
    future_trading_reserve_lines: int
    safety_margin_lines: int
    frozen_universe_level1_lines: int
    market_proxy_level1_lines: int
    always_on_bar_lines: int
    incremental_shadow_lines: int
    required_research_lines: int
    available_research_lines: int
    optional_tick_by_tick_capacity: int
    optional_depth_capacity: int
    optional_option_line_ceiling: int
    maximum_configured_research_lines: int
    allowed: bool
    blocker: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_shadow_capacity(
    capacity: RuntimeCapacityManifest,
    *,
    universe_size: int,
    always_on_bar_lines: int,
) -> ShadowCapacityAssessment:
    """Prove that shadow capture reuses, rather than adds, universe L1 lines."""

    required = universe_size + 1 + always_on_bar_lines
    available = capacity.available_research_level1_lines
    allowed = required <= available
    option_ceiling = int(capacity.max_active_option_episodes.value) * int(
        capacity.max_option_lines_per_episode.value
    )
    return ShadowCapacityAssessment(
        total_line_allowance=int(capacity.total_level1_allowance.value),
        externally_reserved_lines=int(capacity.externally_reserved_lines.value),
        preexisting_internal_lines=capacity.current_internal_level1_lines,
        future_trading_reserve_lines=int(capacity.reserved_future_trading_lines.value),
        safety_margin_lines=int(capacity.safety_margin_lines.value),
        frozen_universe_level1_lines=universe_size,
        market_proxy_level1_lines=1,
        always_on_bar_lines=always_on_bar_lines,
        incremental_shadow_lines=0,
        required_research_lines=required,
        available_research_lines=available,
        optional_tick_by_tick_capacity=capacity.available_tick_by_tick,
        optional_depth_capacity=capacity.available_depth,
        optional_option_line_ceiling=option_ceiling,
        maximum_configured_research_lines=(
            required + capacity.available_tick_by_tick + capacity.available_depth + option_ceiling
        ),
        allowed=allowed,
        blocker=None if allowed else "BLOCK_SHADOW_ALL_UNIVERSE_CAPACITY",
    )


def evaluate_pilot_readiness(
    *,
    dataset_root: str | Path,
    universe_symbols: tuple[str, ...],
    capacity_allowed: bool,
    reconnect_validation_passed: bool,
    safety_tests_passed: bool,
    disk_capacity_acceptable: bool,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Create a collection-quality readiness report without outcome analysis."""

    root = Path(dataset_root)
    quality_path = root / "collection_quality_summary.csv"
    rows: list[dict[str, str]] = []
    if quality_path.is_file():
        with quality_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    by_session: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_session.setdefault(row["session"], []).append(row)
    latest_session = max(by_session, default=None)
    latest_rows = [] if latest_session is None else by_session[latest_session]
    latest_by_symbol = {row["symbol"]: row for row in latest_rows}
    permission_failures = sum(
        int(row.get("market_data_permission_errors", "0") or 0) for row in rows
    )
    valid_symbols = {
        symbol
        for symbol, row in latest_by_symbol.items()
        if row.get("expected_subscription_active") == "true"
        and int(row.get("bbo_update_count", "0") or 0)
        > int(row.get("invalid_quote_count", "0") or 0)
        and int(row.get("size_change_count", "0") or 0) > 0
    }
    expected = set(universe_symbols)
    manifest_path = root / "shadow_collection_manifest.json"
    universe_manifest_path = root / "universe_manifest.json"
    manifest: dict[str, object] = {}
    universe_manifest: dict[str, object] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        universe_manifest = json.loads(universe_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    raw_manifest_symbols = universe_manifest.get("symbols", [])
    manifest_symbols = {
        str(item["symbol"])
        for item in raw_manifest_symbols
        if isinstance(item, dict) and "symbol" in item
    } if isinstance(raw_manifest_symbols, list) else set()
    manifest_identity_valid = (
        manifest.get("pipeline_pilot") is True
        and manifest.get("dataset_version") == "m1c_microstructure_shadow_v0"
        and manifest_symbols == expected
        and universe_manifest.get("universe_hash") == manifest.get("frozen_universe_hash")
    )
    parquet_files = tuple(root.glob("**/*.parquet"))
    storage_integrity_passed = bool(parquet_files) and manifest_identity_valid
    row_count = 0
    total_bytes = 0
    subscription_start_symbols: set[str] = set()
    subscription_evidence_hashes: list[str] = []
    continuity_evidence_hashes: list[str] = []
    observed_gap_kinds: set[str] = set()
    if storage_integrity_passed:
        try:
            import pyarrow.parquet as pq

            for path in parquet_files:
                metadata = pq.read_metadata(path)  # type: ignore[no-untyped-call]
                row_count += metadata.num_rows
                total_bytes += path.stat().st_size
                if "event_type=collection_gap_event" in str(path):
                    continuity_evidence_hashes.append(
                        hashlib.sha256(path.read_bytes()).hexdigest()
                    )
                    table = pq.ParquetFile(path).read(  # type: ignore[no-untyped-call]
                        columns=["gap_kind", "symbol", "pipeline_pilot"]
                    )
                    rows_for_file = table.to_pylist()
                    observed_gap_kinds.update(str(item["gap_kind"]) for item in rows_for_file)
                    starts = [
                        item
                        for item in rows_for_file
                        if item["gap_kind"] == "subscription_started"
                        and item["pipeline_pilot"] is True
                    ]
                    if starts:
                        subscription_start_symbols.update(
                            str(item["symbol"]) for item in starts
                        )
                        subscription_evidence_hashes.append(
                            hashlib.sha256(path.read_bytes()).hexdigest()
                        )
        except (OSError, ValueError):
            storage_integrity_passed = False
    subscription_start_verified = subscription_start_symbols == expected
    reconnect_evidence_verified = {
        "connection_lost",
        "connection_restored",
        "subscription_rebuilt",
    }.issubset(observed_gap_kinds)
    storage_integrity_passed = (
        storage_integrity_passed
        and subscription_start_verified
        and reconnect_evidence_verified
    )
    observed_bbo_events = sum(int(row.get("bbo_update_count", "0") or 0) for row in latest_rows)
    timestamps = [
        datetime.fromisoformat(value).astimezone(UTC)
        for row in latest_rows
        for value in (row.get("observed_first_bbo", ""), row.get("observed_last_bbo", ""))
        if value
    ]
    observed_span_seconds = (
        0.0 if len(timestamps) < 2 else (max(timestamps) - min(timestamps)).total_seconds()
    )
    estimated_full_session_events = (
        0
        if observed_span_seconds <= 0.0
        else round(observed_bbo_events * 23_400.0 / observed_span_seconds)
    )
    bytes_per_row = 0.0 if row_count == 0 else total_bytes / row_count
    estimated_full_session_bytes = round(estimated_full_session_events * bytes_per_row)
    if not capacity_allowed:
        classification = "BLOCKED_MARKET_DATA_CAPACITY"
    elif permission_failures:
        classification = "BLOCKED_MARKET_DATA_PERMISSION"
    elif valid_symbols != expected:
        classification = "BLOCKED_BBO_SIZE_DATA_MISSING"
    elif not storage_integrity_passed or not reconnect_validation_passed:
        classification = "BLOCKED_STORAGE_OR_RECORDER_INTEGRITY"
    elif not safety_tests_passed:
        classification = "BLOCKED_SAFETY_REGRESSION"
    elif not disk_capacity_acceptable:
        classification = "BLOCKED_STORAGE_OR_RECORDER_INTEGRITY"
    else:
        classification = "READY_FOR_SHADOW_COLLECTION"
    payload: dict[str, object] = {
        "classification": classification,
        "dataset_version": "m1c_microstructure_shadow_v0",
        "readiness_report_schema_version": "m1c-shadow-pilot-readiness-v0",
        "pilot_activation_id": manifest.get("activation_id"),
        "pilot_dataset_root": str(root.resolve()),
        "pilot_manifest_path": str(manifest_path.resolve()),
        "pilot_manifest_hash": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        ),
        "pilot_universe_manifest_hash": (
            hashlib.sha256(universe_manifest_path.read_bytes()).hexdigest()
            if universe_manifest_path.is_file()
            else None
        ),
        "frozen_universe_hash": manifest.get("frozen_universe_hash"),
        "research_contract_hash": manifest.get("research_contract_hash"),
        "subscription_start_verified": subscription_start_verified,
        "subscription_start_evidence_hash": (
            canonical_sha256(sorted(subscription_evidence_hashes))
            if subscription_start_verified
            else None
        ),
        "reconnect_evidence_verified": reconnect_evidence_verified,
        "continuity_evidence_hash": (
            canonical_sha256(sorted(continuity_evidence_hashes))
            if reconnect_evidence_verified
            else None
        ),
        "latest_pilot_session": latest_session,
        "universe_size": len(universe_symbols),
        "valid_bbo_and_size_change_symbols": sorted(valid_symbols),
        "missing_or_size_static_symbols": sorted(expected - valid_symbols),
        "capacity_allowed": capacity_allowed,
        "permission_failures": permission_failures,
        "reconnect_validation_passed": reconnect_validation_passed,
        "storage_integrity_passed": storage_integrity_passed,
        "safety_tests_passed": safety_tests_passed,
        "disk_capacity_acceptable": disk_capacity_acceptable,
        "observed_bbo_events": observed_bbo_events,
        "observed_span_seconds": observed_span_seconds,
        "estimated_bbo_events_per_full_rth_session": estimated_full_session_events,
        "estimated_parquet_bytes_per_full_rth_session": estimated_full_session_bytes,
        "directional_analysis_performed": False,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, destination)
    return payload


@dataclass
class _QualityRow:
    session: date
    symbol: str
    expected_subscription_active: bool = False
    observed_first_bbo: str = ""
    observed_last_bbo: str = ""
    bbo_update_count: int = 0
    size_change_count: int = 0
    provider_timestamp_count: int = 0
    longest_observed_event_gap_seconds: float = 0.0
    reconnect_gaps: int = 0
    locked_quote_count: int = 0
    crossed_quote_count: int = 0
    invalid_quote_count: int = 0
    market_data_permission_errors: int = 0
    market_data_type: str = "unknown"
    optional_trade_stream: str = "BBO_ONLY"


class ShadowMicrostructureCollectorV0:
    """Mirror existing raw callbacks into an isolated collection-only dataset."""

    def __init__(
        self,
        *,
        root: str | Path,
        config: ShadowMicrostructureConfig,
        run_id: str,
        git_commit: str,
        universe_hash: str,
        m1c_artifact_hash: str,
        m1c_configuration_hash: str,
        contracts: tuple[QualifiedUnderlying, ...],
        capacity: RuntimeCapacityManifest,
        always_on_bar_lines: int,
        recorder_version: str,
    ) -> None:
        if not config.enabled or config.activation_timestamp_utc is None:
            raise ValueError("shadow collector requires an enabled activation")
        if len(contracts) != 20 or len({item.symbol for item in contracts}) != 20:
            raise ValueError("shadow collector requires the exact frozen 20-stock universe")
        if FORBIDDEN_RAW_RESEARCH_FIELDS.intersection(ShadowBBOEvent.model_fields):
            raise RuntimeError("shadow raw schema contains a forbidden research field")
        activation = config.activation_timestamp_utc.astimezone(UTC)
        planned_sessions = _planned_sessions(activation, config.eligible_sessions)
        if (
            config.planned_end_session is not None
            and config.planned_end_session != planned_sessions[-1]
        ):
            raise ValueError("planned_end_session differs from the 20-session XNYS contract")
        readiness: dict[str, object] | None = None
        if not config.pilot_mode:
            assert config.pilot_readiness_report is not None
            readiness = json.loads(config.pilot_readiness_report.read_text(encoding="utf-8"))
            if readiness.get("classification") != "READY_FOR_SHADOW_COLLECTION":
                raise RuntimeError("BLOCKED_PILOT_READINESS")
        assert config.research_contract is not None
        research_contract_bytes = config.research_contract.read_bytes()
        research_contract_hash = hashlib.sha256(research_contract_bytes).hexdigest()
        if readiness is not None:
            pilot_root = Path(str(readiness.get("pilot_dataset_root", ""))).resolve()
            pilot_manifest_path = Path(str(readiness.get("pilot_manifest_path", "")))
            pilot_universe_manifest_path = pilot_root / "universe_manifest.json"
            pilot_research_contract_path = pilot_root / "shadow_research_contract_v0.json"
            expected_pilot_manifest_hash = str(readiness.get("pilot_manifest_hash", ""))
            observed_pilot_manifest_hash = (
                hashlib.sha256(pilot_manifest_path.read_bytes()).hexdigest()
                if pilot_manifest_path.is_file()
                else ""
            )
            observed_pilot_universe_manifest_hash = (
                hashlib.sha256(pilot_universe_manifest_path.read_bytes()).hexdigest()
                if pilot_universe_manifest_path.is_file()
                else ""
            )
            observed_pilot_research_contract_hash = (
                hashlib.sha256(pilot_research_contract_path.read_bytes()).hexdigest()
                if pilot_research_contract_path.is_file()
                else ""
            )
            pilot_subscription_symbols: set[str] = set()
            pilot_subscription_hashes: list[str] = []
            pilot_continuity_hashes: list[str] = []
            pilot_gap_kinds: set[str] = set()
            try:
                import pyarrow.parquet as pq

                for path in pilot_root.glob(
                    "**/event_type=collection_gap_event/**/*.parquet"
                ):
                    pilot_continuity_hashes.append(
                        hashlib.sha256(path.read_bytes()).hexdigest()
                    )
                    rows = pq.ParquetFile(path).read(  # type: ignore[no-untyped-call]
                        columns=["gap_kind", "symbol", "pipeline_pilot"]
                    ).to_pylist()
                    pilot_gap_kinds.update(str(item["gap_kind"]) for item in rows)
                    starts = [
                        item
                        for item in rows
                        if item["gap_kind"] == "subscription_started"
                        and item["pipeline_pilot"] is True
                    ]
                    if starts:
                        pilot_subscription_symbols.update(
                            str(item["symbol"]) for item in starts
                        )
                        pilot_subscription_hashes.append(
                            hashlib.sha256(path.read_bytes()).hexdigest()
                        )
            except (OSError, ValueError):
                pilot_subscription_symbols.clear()
            observed_subscription_evidence_hash = (
                canonical_sha256(sorted(pilot_subscription_hashes))
                if pilot_subscription_symbols == {item.symbol for item in contracts}
                else ""
            )
            observed_continuity_evidence_hash = (
                canonical_sha256(sorted(pilot_continuity_hashes))
                if {
                    "connection_lost",
                    "connection_restored",
                    "subscription_rebuilt",
                }.issubset(pilot_gap_kinds)
                else ""
            )
            recomputed_readiness = evaluate_pilot_readiness(
                dataset_root=pilot_root,
                universe_symbols=tuple(item.symbol for item in contracts),
                capacity_allowed=readiness.get("capacity_allowed") is True,
                reconnect_validation_passed=(
                    readiness.get("reconnect_validation_passed") is True
                ),
                safety_tests_passed=readiness.get("safety_tests_passed") is True,
                disk_capacity_acceptable=(
                    readiness.get("disk_capacity_acceptable") is True
                ),
            )
            if (
                readiness.get("dataset_version") != config.dataset_version
                or readiness.get("frozen_universe_hash") != universe_hash
                or readiness.get("research_contract_hash") != research_contract_hash
                or pilot_manifest_path.resolve()
                != (pilot_root / "shadow_collection_manifest.json").resolve()
                or not expected_pilot_manifest_hash
                or observed_pilot_manifest_hash != expected_pilot_manifest_hash
                or readiness.get("pilot_universe_manifest_hash")
                != observed_pilot_universe_manifest_hash
                or observed_pilot_research_contract_hash != research_contract_hash
                or readiness.get("subscription_start_evidence_hash")
                != observed_subscription_evidence_hash
                or readiness.get("continuity_evidence_hash")
                != observed_continuity_evidence_hash
                or recomputed_readiness.get("classification")
                != "READY_FOR_SHADOW_COLLECTION"
            ):
                raise RuntimeError("BLOCKED_PILOT_READINESS_IDENTITY_MISMATCH")
        assessment = assess_shadow_capacity(
            capacity,
            universe_size=len(contracts),
            always_on_bar_lines=always_on_bar_lines,
        )
        if not assessment.allowed:
            raise RuntimeError("BLOCK_SHADOW_ALL_UNIVERSE_CAPACITY")

        self.config = config
        self.run_id = run_id
        self.git_commit = git_commit
        self.universe_hash = universe_hash
        self.contracts = {item.symbol: item for item in contracts}
        self.activation = activation
        self.planned_sessions = planned_sessions
        self.collection_sessions = (
            planned_sessions[: config.pilot_sessions] if config.pilot_mode else planned_sessions
        )
        config_hash = canonical_sha256(config.model_dump(mode="json"))
        self.activation_id = canonical_sha256(
            {
                "dataset_version": config.dataset_version,
                "activation_timestamp_utc": activation.isoformat(),
                "universe_hash": universe_hash,
                "collection_configuration_hash": config_hash,
            }
        )
        collection_mode = "pipeline_pilot" if config.pilot_mode else "confirmatory"
        self.dataset_root = Path(root) / config.dataset_version
        self.root = self.dataset_root / collection_mode / f"activation_id={self.activation_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_store = PartitionedEventStore(
            root=self.root,
            prospective_collection_start=activation,
            recorder_version=recorder_version,
            contract_version="m1c-microstructure-shadow-collection-v0",
            schema_version=config.raw_schema_version,
            run_id=run_id,
        )
        self.quality_path = self.root / "collection_quality_summary.csv"
        self._quality: dict[tuple[date, str], _QualityRow] = {}
        self._last_bbo_received: dict[tuple[date, str], datetime] = {}
        self._last_sizes: dict[tuple[date, str], tuple[float | None, float | None]] = {}
        self._quote_state: dict[tuple[int, date, str], dict[str, float | None]] = {}
        self._optional_trade_active_symbols: set[str] = set()
        self._expected_active_symbols: set[str] = set()
        self._subscription_start_pending = False
        self._active_gap_start: dict[str, datetime] = {}
        self._control_sequence = 0
        self._load_quality()

        universe_payload: dict[str, object] = {
            "dataset_version": config.dataset_version,
            "universe_hash": universe_hash,
            "symbols": [
                {
                    "symbol": item.symbol,
                    "con_id": item.con_id,
                    "exchange": item.exchange,
                    "minimum_tick": item.minimum_tick,
                }
                for item in sorted(contracts, key=lambda value: value.symbol)
            ],
        }
        _write_immutable_json(self.root / "universe_manifest.json", universe_payload)
        manifest: dict[str, object] = {
            "dataset_version": config.dataset_version,
            "raw_schema_version": config.raw_schema_version,
            "activation_id": self.activation_id,
            "activation_timestamp_utc": activation.isoformat(),
            "start_rule": "first_complete_XNYS_session_with_open_at_or_after_activation",
            "planned_eligible_sessions": config.eligible_sessions,
            "planned_pipeline_pilot_sessions": config.pilot_sessions,
            "planned_start_session": planned_sessions[0].isoformat(),
            "planned_end_session": planned_sessions[-1].isoformat(),
            "eligible_session_dates": [item.isoformat() for item in planned_sessions],
            "active_collection_session_dates": [
                item.isoformat() for item in self.collection_sessions
            ],
            "frozen_universe_hash": universe_hash,
            "frozen_m1c_artifact_hash": m1c_artifact_hash,
            "frozen_m1c_configuration_hash": m1c_configuration_hash,
            "collector_code_git_commit": git_commit,
            "collection_configuration_hash": config_hash,
            "research_contract_hash": research_contract_hash,
            "provider": "IBKR_TWS_SOCKET_API",
            "subscription_type": "existing_always_on_level1_reqMktData",
            "symbols_and_con_ids": universe_payload["symbols"],
            "analysis_enabled": False,
            "direction_scoring_enabled": False,
            "execution_enabled": False,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "pipeline_pilot": config.pilot_mode,
            "pilot_exclusion_rule": "pipeline_pilot=true implies confirmatory_eligible=false",
            "optional_trade_rule": (
                "reuse already-selected Last streams only; never expand subscriptions"
            ),
            "depth_enabled": False,
            "capacity_assessment": assessment.to_dict(),
        }
        _write_immutable_json(self.root / "shadow_collection_manifest.json", manifest)
        _write_immutable_json(
            self.root / "runs" / f"run_id={run_id}.json",
            {
                "activation_id": self.activation_id,
                "run_id": run_id,
                "collector_code_git_commit": git_commit,
                "activation_timestamp_utc": activation.isoformat(),
            },
        )
        _write_immutable_bytes(
            self.root / "shadow_research_contract_v0.json", research_contract_bytes
        )
        if (
            hashlib.sha256(
                (self.root / "shadow_research_contract_v0.json").read_bytes()
            ).hexdigest()
            != research_contract_hash
        ):
            raise RuntimeError("shadow research contract copy hash mismatch")
        self._write_quality()

    @property
    def confirmatory_eligible(self) -> bool:
        return not self.config.pilot_mode

    def _event_is_confirmatory(self, event_session: date) -> bool:
        return self.confirmatory_eligible and event_session in self.planned_sessions

    def _within_collection_window(self, event_session: date) -> bool:
        return event_session in self.collection_sessions

    def _quality_row(self, event_session: date, symbol: str) -> _QualityRow:
        for universe_symbol in self.contracts:
            self._quality.setdefault(
                (event_session, universe_symbol),
                _QualityRow(
                    session=event_session,
                    symbol=universe_symbol,
                    expected_subscription_active=(
                        universe_symbol in self._expected_active_symbols
                    ),
                ),
            )
        return self._quality[(event_session, symbol)]

    def _load_quality(self) -> None:
        if not self.quality_path.is_file():
            return
        with self.quality_path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                row = _QualityRow(
                    session=date.fromisoformat(item["session"]),
                    symbol=item["symbol"],
                    expected_subscription_active=item["expected_subscription_active"] == "true",
                    observed_first_bbo=item["observed_first_bbo"],
                    observed_last_bbo=item["observed_last_bbo"],
                    bbo_update_count=int(item["bbo_update_count"]),
                    size_change_count=int(item.get("size_change_count", 0)),
                    provider_timestamp_count=int(item["provider_timestamp_count"]),
                    longest_observed_event_gap_seconds=float(
                        item["longest_observed_event_gap_seconds"]
                    ),
                    reconnect_gaps=int(item["reconnect_gaps"]),
                    locked_quote_count=int(item["locked_quote_count"]),
                    crossed_quote_count=int(item["crossed_quote_count"]),
                    invalid_quote_count=int(item["invalid_quote_count"]),
                    market_data_permission_errors=int(item["market_data_permission_errors"]),
                    market_data_type=item["market_data_type"],
                    optional_trade_stream=item["optional_trade_stream"],
                )
                self._quality[(row.session, row.symbol)] = row
                if row.observed_last_bbo:
                    self._last_bbo_received[(row.session, row.symbol)] = datetime.fromisoformat(
                        row.observed_last_bbo
                    ).astimezone(UTC)

    def _write_quality(self) -> None:
        fields = (
            "session",
            "symbol",
            "expected_subscription_active",
            "observed_first_bbo",
            "observed_last_bbo",
            "bbo_update_count",
            "size_change_count",
            "provider_timestamp_availability",
            "provider_timestamp_count",
            "longest_observed_event_gap_seconds",
            "reconnect_gaps",
            "locked_quote_count",
            "crossed_quote_count",
            "invalid_quote_count",
            "market_data_permission_errors",
            "market_data_type",
            "optional_trade_stream",
            "number_universe_stocks",
            "stocks_with_valid_bbo_collection",
            "stocks_missing_bbo_collection",
            "percent_universe_coverage",
        )
        temporary = self.quality_path.with_name(f".{self.quality_path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for key in sorted(self._quality):
                row = self._quality[key]
                session_rows = [
                    item
                    for (candidate, _), item in self._quality.items()
                    if candidate == row.session
                ]
                valid = sum(
                    item.bbo_update_count > item.invalid_quote_count for item in session_rows
                )
                writer.writerow(
                    {
                        **asdict(row),
                        "session": row.session.isoformat(),
                        "expected_subscription_active": str(
                            row.expected_subscription_active
                        ).lower(),
                        "provider_timestamp_availability": (
                            "ALL"
                            if row.bbo_update_count > 0
                            and row.provider_timestamp_count == row.bbo_update_count
                            else "SOME"
                            if row.provider_timestamp_count > 0
                            else "NONE"
                        ),
                        "number_universe_stocks": len(self.contracts),
                        "stocks_with_valid_bbo_collection": valid,
                        "stocks_missing_bbo_collection": len(self.contracts) - valid,
                        "percent_universe_coverage": round(100.0 * valid / len(self.contracts), 6),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.quality_path)

    def _bbo_from_level1(self, event: UnderlyingLevel1QuoteEvent) -> ShadowBBOEvent | None:
        if event.symbol not in self.contracts or event.tick_type not in BBO_FIELDS:
            return None
        if (
            event.received_timestamp_utc < self.activation
            or not self._within_collection_window(event.session)
        ):
            return None
        event_session = event.session
        state_key = (event.connection_generation, event.session, event.symbol)
        if event.halted is True:
            self._quote_state.pop(state_key, None)
        state = self._quote_state.setdefault(state_key, {})
        state[event.tick_type] = getattr(event, event.tick_type)
        bid = state.get("bid")
        ask = state.get("ask")
        bid_size = state.get("bid_size")
        ask_size = state.get("ask_size")
        flags = dict(event.quote_attributes)
        flags["halted"] = event.halted
        valid = (
            bid is not None
            and ask is not None
            and bid_size is not None
            and ask_size is not None
            and bid > 0.0
            and ask > 0.0
            and bid_size >= 0.0
            and ask_size >= 0.0
            and ask >= bid
        )
        return ShadowBBOEvent(
            **event.model_dump(
                include={
                    "received_timestamp_utc",
                    "received_monotonic_ns",
                    "provider_timestamp_utc",
                    "source_sequence",
                    "session",
                    "symbol",
                    "con_id",
                    "request_id",
                }
            ),
            event_id=canonical_sha256(
                {"dataset": self.config.dataset_version, "source_event_id": event.event_id}
            ),
            dataset_version=self.config.dataset_version,
            raw_schema_version=self.config.raw_schema_version,
            activation_id=self.activation_id,
            run_id=self.run_id,
            source_event_id=event.event_id,
            connection_generation=event.connection_generation,
            callback_arrival_sequence=event.source_sequence,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            changed_field=cast(Literal["bid", "ask", "bid_size", "ask_size"], event.tick_type),
            provider_flags=flags,
            market_data_type=event.market_data_type,
            exchange=event.exchange,
            quote_valid=valid,
            session_segment=session_segment(event.ordering_timestamp),
            stream_availability=(
                "BBO_PLUS_TRADES"
                if event.symbol in self._optional_trade_active_symbols
                else "BBO_ONLY"
            ),
            pipeline_pilot=self.config.pilot_mode,
            confirmatory_eligible=self._event_is_confirmatory(event_session),
        )

    def _level1_from_envelope(
        self,
        event: RawCallbackEnvelopeEvent,
    ) -> ShadowBBOEvent | None:
        if event.symbol not in self.contracts or event.callback_kind != "level1_quote_update":
            return None
        field = str(event.original_payload.get("field", ""))
        if field not in BBO_FIELDS:
            return None
        raw_value = event.original_payload.get("value")
        value = None if raw_value is None else float(raw_value)
        market_data_type = event.original_payload.get("market_data_type") or "unknown"
        source = UnderlyingLevel1QuoteEvent(
            event_id=event.event_id,
            received_timestamp_utc=event.received_timestamp_utc,
            received_monotonic_ns=event.received_monotonic_ns,
            provider_timestamp_utc=event.provider_timestamp_utc,
            source_sequence=event.source_sequence,
            session=event.session,
            symbol=event.symbol,
            con_id=event.con_id,
            request_id=max(0, event.request_id),
            bid=value if field == "bid" else None,
            ask=value if field == "ask" else None,
            bid_size=value if field == "bid_size" else None,
            ask_size=value if field == "ask_size" else None,
            last=None,
            last_size=None,
            market_data_type=MarketDataType(str(market_data_type)),
            source="official_ibkr_tws_socket_api_raw_recovery",
            quote_valid=False,
            tick_type=field,
            exchange=None,
            quote_attributes=dict(event.original_payload.get("attributes") or {}),
            connection_generation=event.connection_generation,
        )
        return self._bbo_from_level1(source)

    def _shadow_trade(self, event: UnderlyingTickTradeEvent) -> ShadowTradeEvent | None:
        if (
            not self.config.collect_optional_trades
            or event.symbol not in self.contracts
            or event.received_timestamp_utc < self.activation
            or not self._within_collection_window(event.session)
        ):
            return None
        return ShadowTradeEvent(
            **event.model_dump(
                include={
                    "received_timestamp_utc",
                    "received_monotonic_ns",
                    "provider_timestamp_utc",
                    "source_sequence",
                    "session",
                    "symbol",
                    "con_id",
                    "request_id",
                }
            ),
            event_id=canonical_sha256(
                {"dataset": self.config.dataset_version, "source_event_id": event.event_id}
            ),
            dataset_version=self.config.dataset_version,
            raw_schema_version=self.config.raw_schema_version,
            activation_id=self.activation_id,
            run_id=self.run_id,
            source_event_id=event.event_id,
            connection_generation=event.connection_generation,
            callback_arrival_sequence=event.source_sequence,
            price=event.price,
            size=event.size,
            exchange=event.exchange,
            conditions=event.conditions,
            provider_flags={
                "past_limit": event.past_limit,
                "unreported": event.unreported,
                "halted": event.halted,
            },
            market_data_type=event.market_data_type,
            session_segment=session_segment(event.ordering_timestamp),
            pipeline_pilot=self.config.pilot_mode,
            confirmatory_eligible=self._event_is_confirmatory(event.session),
        )

    def persist_raw_events(
        self,
        events: tuple[RawEvent, ...],
    ) -> tuple[PartitionWriteResult, ...]:
        if self._subscription_start_pending:
            eligible_level1 = tuple(
                event
                for event in events
                if isinstance(event, UnderlyingLevel1QuoteEvent)
                and event.symbol in self.contracts
                and event.received_timestamp_utc >= self.activation
                and self._within_collection_window(event.session)
            )
            if eligible_level1:
                first = min(eligible_level1, key=lambda event: event.received_timestamp_utc)
                self.record_subscriptions_active(
                    symbols=tuple(self.contracts),
                    observed_at=first.received_timestamp_utc,
                    connection_generation=first.connection_generation,
                )
        represented_callbacks = {
            (event.source_sequence, event.request_id)
            for event in events
            if isinstance(event, UnderlyingLevel1QuoteEvent)
        }
        shadow_events: list[ShadowBBOEvent | ShadowTradeEvent] = []
        for event in events:
            shadow: ShadowBBOEvent | ShadowTradeEvent | None = None
            if isinstance(event, UnderlyingLevel1QuoteEvent):
                shadow = self._bbo_from_level1(event)
            elif isinstance(event, UnderlyingTickTradeEvent):
                shadow = self._shadow_trade(event)
            elif (
                isinstance(event, RawCallbackEnvelopeEvent)
                and (
                    event.source_sequence,
                    event.request_id,
                )
                not in represented_callbacks
            ):
                shadow = self._level1_from_envelope(event)
            if shadow is not None:
                shadow_events.append(shadow)
        if not shadow_events:
            return ()
        results = self.raw_store.write_grouped(
            data_source="ibkr",
            events=tuple(shadow_events),
            complete=True,
        )
        for event in shadow_events:
            row = self._quality_row(event.session, event.symbol)
            if isinstance(event, ShadowTradeEvent):
                row.optional_trade_stream = "BBO_PLUS_TRADES"
                continue
            timestamp = event.received_timestamp_utc.isoformat()
            if not row.observed_first_bbo:
                row.observed_first_bbo = timestamp
            row.observed_last_bbo = timestamp
            row.bbo_update_count += 1
            size_key = (event.session, event.symbol)
            sizes = (event.bid_size, event.ask_size)
            prior_sizes = self._last_sizes.get(size_key)
            if (
                prior_sizes is not None
                and None not in prior_sizes
                and None not in sizes
                and sizes != prior_sizes
            ):
                row.size_change_count += 1
            self._last_sizes[size_key] = sizes
            row.provider_timestamp_count += event.provider_timestamp_utc is not None
            prior = self._last_bbo_received.get((event.session, event.symbol))
            if prior is not None:
                row.longest_observed_event_gap_seconds = max(
                    row.longest_observed_event_gap_seconds,
                    (event.received_timestamp_utc - prior).total_seconds(),
                )
            self._last_bbo_received[(event.session, event.symbol)] = event.received_timestamp_utc
            if event.bid is not None and event.ask is not None:
                row.locked_quote_count += event.bid == event.ask
                row.crossed_quote_count += event.bid > event.ask
            row.invalid_quote_count += not event.quote_valid
            row.market_data_type = event.market_data_type.value
            if event.stream_availability == "BBO_PLUS_TRADES":
                row.optional_trade_stream = "BBO_PLUS_TRADES"
        self._write_quality()
        return results

    def set_optional_trade_stream(
        self,
        *,
        symbol: str,
        active: bool,
        request_id: int = 0,
        observed_at: datetime | None = None,
        connection_generation: int = 0,
    ) -> None:
        """Track existing Last subscription lifecycle without requesting a feed."""

        if symbol not in self.contracts:
            return
        if active:
            self._optional_trade_active_symbols.add(symbol)
        else:
            self._optional_trade_active_symbols.discard(symbol)
        if observed_at is None:
            return
        observed = observed_at.astimezone(UTC)
        event_session = observed.astimezone(NEW_YORK).date()
        if observed < self.activation or not self._within_collection_window(event_session):
            return
        self._control_sequence += 1
        contract = self.contracts[symbol]
        event = ShadowTradeStreamStateEvent(
            event_id=canonical_sha256(
                {
                    "activation_id": self.activation_id,
                    "symbol": symbol,
                    "request_id": request_id,
                    "active": active,
                    "source_sequence": self._control_sequence,
                }
            ),
            received_timestamp_utc=observed,
            received_monotonic_ns=time.monotonic_ns(),
            provider_timestamp_utc=None,
            source_sequence=self._control_sequence,
            session=event_session,
            symbol=symbol,
            con_id=contract.con_id,
            request_id=request_id,
            dataset_version=self.config.dataset_version,
            raw_schema_version=self.config.raw_schema_version,
            activation_id=self.activation_id,
            run_id=self.run_id,
            connection_generation=connection_generation,
            session_segment=session_segment(observed),
            pipeline_pilot=self.config.pilot_mode,
            confirmatory_eligible=self._event_is_confirmatory(event_session),
            stream_state="active" if active else "inactive",
            reason=(
                "existing_subscription_registered"
                if active
                else "existing_subscription_removed"
            ),
        )
        self.raw_store.write_grouped(data_source="ibkr", events=(event,), complete=True)
        if active:
            self._quality_row(event_session, symbol).optional_trade_stream = "BBO_PLUS_TRADES"
            self._write_quality()

    def _gap_event(
        self,
        *,
        symbol: str,
        kind: Literal[
            "subscription_started",
            "connection_lost",
            "connection_restored",
            "subscription_rebuilt",
            "market_data_permission_error",
        ],
        observed_at: datetime,
        connection_generation: int,
        reason: str,
        start: datetime | None,
        end: datetime | None,
    ) -> ShadowGapEvent:
        self._control_sequence += 1
        contract = self.contracts[symbol]
        event_session = observed_at.astimezone(NEW_YORK).date()
        gap_id = canonical_sha256(
            {
                "activation_id": self.activation_id,
                "symbol": symbol,
                "kind": kind,
                "start": None if start is None else start.isoformat(),
                "end": None if end is None else end.isoformat(),
                "generation": connection_generation,
            }
        )
        return ShadowGapEvent(
            event_id=canonical_sha256({"gap_id": gap_id, "kind": kind}),
            received_timestamp_utc=observed_at,
            received_monotonic_ns=time.monotonic_ns(),
            provider_timestamp_utc=None,
            source_sequence=self._control_sequence,
            session=event_session,
            symbol=symbol,
            con_id=contract.con_id,
            request_id=0,
            dataset_version=self.config.dataset_version,
            raw_schema_version=self.config.raw_schema_version,
            activation_id=self.activation_id,
            run_id=self.run_id,
            connection_generation=connection_generation,
            gap_id=gap_id,
            gap_kind=kind,
            gap_start_timestamp_utc=start,
            gap_end_timestamp_utc=end,
            reason=reason,
            session_segment=session_segment(observed_at),
            pipeline_pilot=self.config.pilot_mode,
            confirmatory_eligible=self._event_is_confirmatory(event_session),
        )

    def record_connection_state(
        self,
        *,
        state: str,
        observed_at: datetime,
        connection_generation: int,
        reason: str,
    ) -> None:
        observed = observed_at.astimezone(UTC)
        if (
            observed < self.activation
            or not self._within_collection_window(observed.astimezone(NEW_YORK).date())
        ):
            return
        events: list[ShadowGapEvent] = []
        if state in {"disconnected", "socket_port_mismatch_or_reset"}:
            self._quote_state.clear()
            self._last_sizes.clear()
            self._last_bbo_received.clear()
            for symbol in self.contracts:
                self._active_gap_start.setdefault(symbol, observed)
                events.append(
                    self._gap_event(
                        symbol=symbol,
                        kind="connection_lost",
                        observed_at=observed,
                        connection_generation=connection_generation,
                        reason=reason,
                        start=self._active_gap_start[symbol],
                        end=None,
                    )
                )
        elif state == "connected":
            for symbol, started in tuple(self._active_gap_start.items()):
                events.append(
                    self._gap_event(
                        symbol=symbol,
                        kind="connection_restored",
                        observed_at=observed,
                        connection_generation=connection_generation,
                        reason=reason,
                        start=started,
                        end=observed,
                    )
                )
                self._quality_row(observed.astimezone(NEW_YORK).date(), symbol).reconnect_gaps += 1
                del self._active_gap_start[symbol]
        if events:
            self.raw_store.write_grouped(data_source="ibkr", events=tuple(events), complete=True)
            self._write_quality()

    def record_subscriptions_active(
        self,
        *,
        symbols: tuple[str, ...],
        observed_at: datetime,
        connection_generation: int,
    ) -> None:
        """Record that all protected Level-I requests were accepted by the adapter."""

        requested = set(symbols)
        expected = set(self.contracts)
        if requested != expected:
            missing = ",".join(sorted(expected - requested))
            raise RuntimeError(f"shadow_all_universe_subscription_incomplete:{missing}")
        self._expected_active_symbols = requested
        observed = observed_at.astimezone(UTC)
        if (
            observed < self.activation
            or not self._within_collection_window(observed.astimezone(NEW_YORK).date())
        ):
            self._subscription_start_pending = True
            return
        event_session = observed.astimezone(NEW_YORK).date()
        for symbol in requested:
            self._quality_row(event_session, symbol).expected_subscription_active = True
        events = tuple(
            self._gap_event(
                symbol=symbol,
                kind="subscription_started",
                observed_at=observed,
                connection_generation=connection_generation,
                reason="protected_all_universe_level1_request_accepted",
                start=observed,
                end=observed,
            )
            for symbol in sorted(requested)
        )
        self.raw_store.write_grouped(data_source="ibkr", events=events, complete=True)
        self._subscription_start_pending = False
        self._write_quality()

    def record_subscriptions_rebuilt(
        self,
        *,
        observed_at: datetime,
        connection_generation: int,
    ) -> None:
        observed = observed_at.astimezone(UTC)
        self._quote_state.clear()
        self._last_sizes.clear()
        self._last_bbo_received.clear()
        self._expected_active_symbols = set(self.contracts)
        if (
            observed < self.activation
            or not self._within_collection_window(observed.astimezone(NEW_YORK).date())
        ):
            return
        event_session = observed.astimezone(NEW_YORK).date()
        for symbol in self.contracts:
            self._quality_row(event_session, symbol).expected_subscription_active = True
        events = tuple(
            self._gap_event(
                symbol=symbol,
                kind="subscription_rebuilt",
                observed_at=observed,
                connection_generation=connection_generation,
                reason="all_universe_level1_subscription_rebuilt",
                start=self._active_gap_start.get(symbol),
                end=None,
            )
            for symbol in self.contracts
        )
        self.raw_store.write_grouped(data_source="ibkr", events=events, complete=True)
        self._write_quality()

    def record_permission_error(
        self,
        *,
        symbol: str,
        error_code: int,
        observed_at: datetime,
        connection_generation: int,
    ) -> None:
        if symbol not in self.contracts or error_code not in PERMISSION_ERROR_CODES:
            return
        observed = observed_at.astimezone(UTC)
        if (
            observed < self.activation
            or not self._within_collection_window(observed.astimezone(NEW_YORK).date())
        ):
            return
        event = self._gap_event(
            symbol=symbol,
            kind="market_data_permission_error",
            observed_at=observed_at.astimezone(UTC),
            connection_generation=connection_generation,
            reason=f"ibkr_error_{error_code}",
            start=observed_at.astimezone(UTC),
            end=observed_at.astimezone(UTC),
        )
        self.raw_store.write_grouped(data_source="ibkr", events=(event,), complete=True)
        self._quality_row(event.session, symbol).market_data_permission_errors += 1
        self._write_quality()


__all__ = [
    "FORBIDDEN_RAW_RESEARCH_FIELDS",
    "ShadowCapacityAssessment",
    "ShadowMicrostructureCollectorV0",
    "assess_shadow_capacity",
    "canonical_sha256",
    "evaluate_pilot_readiness",
    "session_segment",
]
