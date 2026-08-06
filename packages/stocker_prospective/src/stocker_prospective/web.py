"""Read-only FastAPI process for prospective evidence monitoring."""

from __future__ import annotations

import csv
import ipaddress
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from stocker_prospective.bundle import BundleError, load_active_bundle
from stocker_prospective.config import (
    ProspectiveConfig,
    operational_thresholds,
    public_config,
    validate_runtime_safety,
)
from stocker_prospective.contract import claims_boundary
from stocker_prospective.evidence_replay import replay_persisted_evidence
from stocker_prospective.ibkr import (
    FORBIDDEN_BROKER_SURFACE,
    IBKRMarketDataAdapter,
    official_ibkr_api_projection,
)
from stocker_prospective.operational_logging import (
    OperationLogFields,
    RequestOperationMetrics,
    begin_request_metrics,
    reset_request_metrics,
    structured_log,
)
from stocker_prospective.parity import FeatureParityError, load_feature_parity_report
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.replay_control import ReplayController, ReplayStartRequest
from stocker_prospective.transfer import classify_cross_vendor_validation_status

LOGGER = logging.getLogger("uvicorn.error.stocker_prospective.web")


def _active_bundle_projection(config: ProspectiveConfig) -> dict[str, Any]:
    try:
        verification = load_active_bundle(config.paths.bundle_root)
        return {
            "bundle_id": verification.manifest.bundle_id,
            "manifest_sha256": verification.manifest_sha256,
            "verified": verification.verified,
            "blockers": verification.blockers,
            "feature_runtime": {
                "installed": verification.manifest.feature_runtime is not None,
                "contract_version": (
                    None
                    if verification.manifest.feature_runtime is None
                    else verification.manifest.feature_runtime.contract_version
                ),
                "scoring_authorized_by_registry": (
                    False
                    if verification.manifest.feature_runtime is None
                    else verification.manifest.feature_runtime.scoring_authorized_by_registry
                ),
            },
        }
    except BundleError as exc:
        return {
            "bundle_id": None,
            "manifest_sha256": None,
            "verified": False,
            "blockers": [str(exc).split(":", 1)[0]],
            "feature_runtime": {
                "installed": False,
                "contract_version": None,
                "scoring_authorized_by_registry": False,
            },
        }


def _parity_projection(config: ProspectiveConfig) -> dict[str, Any]:
    try:
        report = load_feature_parity_report(config.paths.feature_parity_report)
        counts: dict[str, int] = defaultdict(int)
        for item in report.features:
            counts[item.parity_status] += 1
        return {
            "scoring_allowed": report.overall_scoring_allowed,
            "blocker": None,
            "diagnostic_warning": report.overall_blocker,
            "diagnostic_only": True,
            "counts": dict(counts),
            "report": report.model_dump(mode="json"),
        }
    except FeatureParityError as exc:
        return {
            "scoring_allowed": False,
            "blocker": None,
            "diagnostic_warning": str(exc).split(":", 1)[0],
            "diagnostic_only": True,
            "counts": {},
            "report": None,
        }


def _json_artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_csv_value(value: str | None) -> str | int | float | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _csv_artifact(path: Path | None) -> list[dict[str, str | int | float | None]]:
    if path is None or not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return [
                {key: _coerce_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except (OSError, csv.Error):
        return []


def _concentration_audit_projection(root: Path | None) -> dict[str, Any]:
    """Load only committed retrospective artifacts; never recompute or relax the gate."""

    if root is None:
        return {
            "available": False,
            "original_decision": "blocked_insufficient_low_tail_support",
            "original_gate_passed": False,
        }
    decision = _json_artifact(root / "decision.json")
    month_explanation = _json_artifact(root / "stress_month_concentration_explanation.json")
    surprise_explanation = _json_artifact(root / "surprise_concentration_explanation.json")
    small_count = _json_artifact(root / "small_count_feasibility.json")
    representations = [
        row
        for row in _csv_artifact(root / "checkpoint_vs_episode_concentration.csv")
        if row.get("period") == "stress" and row.get("dimension") == "month"
    ]
    available = all(
        artifact is not None
        for artifact in (decision, month_explanation, surprise_explanation, small_count)
    )
    return {
        "available": available,
        "original_decision": "blocked_insufficient_low_tail_support",
        "original_gate_passed": False,
        "decision": decision,
        "month_explanation": month_explanation,
        "surprise_explanation": surprise_explanation,
        "small_count_feasibility": small_count,
        "stress_month_exposure": _csv_artifact(root / "stress_month_exposure_audit.csv"),
        "stress_month_tail_incidence": _csv_artifact(root / "stress_month_tail_incidence.csv"),
        "representation_month_concentration": representations,
        "leave_one_month_out": _csv_artifact(root / "leave_one_month_out.csv"),
    }


def _daily_report_packages(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.is_dir():
        return []
    packages: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("????-??-??/package-*.json"), reverse=True):
        safe_metadata = _safe_contained_file(root, metadata_path)
        if safe_metadata is None:
            continue
        payload = _json_artifact(safe_metadata)
        if payload is None:
            continue
        archive_name = str(payload.get("archive", ""))
        archive = _safe_contained_file(root, metadata_path.parent / archive_name)
        if (
            not archive_name
            or Path(archive_name).name != archive_name
            or archive is None
            or archive.suffix != ".zip"
        ):
            continue
        packages.append(
            {
                **payload,
                "download_path": (f"/api/reports/daily/{metadata_path.parent.name}/{archive_name}"),
                "archive_size_bytes": archive.stat().st_size,
            }
        )
    return packages


def _safe_contained_file(root: Path, candidate: Path) -> Path | None:
    """Resolve one regular file without following symlinked report components."""

    try:
        if root.is_symlink():
            return None
        relative = candidate.relative_to(root)
        current = root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                return None
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved_candidate if resolved_candidate.is_file() else None


def _path_exposes_forbidden_broker_resource(path: str) -> bool:
    segments = {
        segment.lower() for segment in path.split("/") if segment and not segment.startswith("{")
    }
    return bool(
        segments.intersection(
            {
                "order",
                "orders",
                "account",
                "accounts",
                "position",
                "positions",
                "trade",
                "trades",
                "buy",
                "buys",
                "sell",
                "sells",
                "broker",
                "brokers",
                "execution",
                "executions",
                "credential",
                "credentials",
                "upload",
                "uploads",
            }
        )
    )


class _OperationalProjectionCache:
    """Small TTL cache for bounded filesystem and package safety checks."""

    def __init__(self, *, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._loaded_at = 0.0
        self._value: dict[str, Any] | None = None

    def get(self, loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if (
                self._value is None
                or self._ttl_seconds == 0.0
                or now - self._loaded_at >= self._ttl_seconds
            ):
                self._value = loader()
                self._loaded_at = now
            return self._value


def create_web_app(config: ProspectiveConfig) -> FastAPI:
    """Create a web app that receives no recorder or broker object."""

    validate_runtime_safety(config, object())
    store = ProspectiveReadStore(config.paths.database, run_id=config.runtime.run_id)
    operational_projection_cache = _OperationalProjectionCache(
        ttl_seconds=config.web.operational_projection_cache_seconds
    )
    static_no_order_projection_cache = _OperationalProjectionCache(
        ttl_seconds=config.web.operational_projection_cache_seconds
    )

    def operational_artifacts() -> dict[str, Any]:
        return operational_projection_cache.get(
            lambda: {
                "bundle": _active_bundle_projection(config),
                "parity": _parity_projection(config),
                "ibkr_api": official_ibkr_api_projection(),
                "m1c_live_parity": _json_artifact(config.paths.m1c_live_parity_report),
                "direction_live_parity": _json_artifact(config.paths.direction_live_parity_report),
            }
        )

    def run_replay(
        request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> Any:
        artifact_root = config.paths.frozen_m1c_artifact_root
        return replay_persisted_evidence(
            database_path=config.paths.database,
            run_id=config.runtime.run_id,
            mode=request.mode,
            speed=request.speed,
            episode_id=request.episode_id,
            m1c_feature_manifest_path=(
                None
                if artifact_root is None
                else artifact_root / "causal_movement_feature_manifest.json"
            ),
            m1c_threshold_path=(
                None if artifact_root is None else artifact_root / "causal_movement_threshold.json"
            ),
            stop_event=stop_event,
            maximum_records=config.web.replay_maximum_records,
            maximum_materialized_bytes=config.web.replay_maximum_materialized_bytes,
        )

    replay_controller = ReplayController(
        runner=run_replay,
        stop_join_timeout_seconds=config.web.replay_stop_timeout_seconds,
    )
    static_root = Path(__file__).with_name("web_static")
    authentication_token: str | None = None
    if config.web.authentication_enabled:
        assert config.web.auth_token_env is not None
        authentication_token = os.environ.get(config.web.auth_token_env)
        if not authentication_token:
            raise RuntimeError(
                "blocked_unsafe_runtime_configuration: web authentication token is absent"
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        store.open_anchor()
        try:
            yield
        finally:
            replay_controller.shutdown()
            store.close_anchor()

    app = FastAPI(
        title="Stocker Prospective Evidence Recorder",
        version=config.runtime.app_version,
        docs_url=None if config.web.production else "/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    allowed_hosts = list(config.web.allowed_hosts)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")
    rate_windows: OrderedDict[str, deque[float]] = OrderedDict()
    maximum_rate_limit_identities = 4096
    replay_control_paths = {"/api/replay/start", "/api/replay/stop"}

    def route_template(request: Request) -> str:
        route = request.scope.get("route")
        return str(getattr(route, "path", request.url.path))

    def operation_log_fields(request: Request) -> OperationLogFields:
        metrics = getattr(request.state, "operation_metrics", None)
        if isinstance(metrics, RequestOperationMetrics):
            return metrics.log_fields()
        return RequestOperationMetrics().log_fields()

    def security_headers(request_id: str) -> dict[str, str]:
        return {
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-Correlation-ID": request_id,
            "X-Request-ID": request_id,
        }

    @app.middleware("http")
    async def security_boundary(request: Request, call_next: Any) -> Any:
        supplied_request_id = request.headers.get(
            "x-request-id",
            request.headers.get("x-correlation-id", ""),
        )
        request_id = (
            supplied_request_id
            if supplied_request_id
            and len(supplied_request_id) <= 128
            and all(character.isalnum() or character in "-_." for character in supplied_request_id)
            else str(uuid.uuid4())
        )
        request.state.correlation_id = request_id
        request.state.request_id = request_id
        request.state.request_started_monotonic = time.monotonic()
        metrics, metrics_token = begin_request_metrics()
        request.state.operation_metrics = metrics
        try:
            response: Any | None = None
            if (
                request.url.path.startswith("/api/")
                and request.method not in {"GET", "HEAD", "OPTIONS"}
                and not (request.method == "POST" and request.url.path in replay_control_paths)
            ):
                response = JSONResponse(
                    status_code=404,
                    content={
                        "detail": "not_found",
                        "correlation_id": request_id,
                    },
                )
            if response is None and authentication_token is not None:
                authorization = request.headers.get("authorization", "")
                bearer = (
                    authorization.removeprefix("Bearer ").strip()
                    if authorization.startswith("Bearer ")
                    else ""
                )
                cookie = request.cookies.get(config.web.auth_cookie_name, "")
                supplied = bearer or cookie
                if not supplied or not secrets.compare_digest(
                    supplied,
                    authentication_token,
                ):
                    response = JSONResponse(
                        status_code=401,
                        content={
                            "detail": "authentication_required",
                            "correlation_id": request_id,
                        },
                        headers={"WWW-Authenticate": "Bearer"},
                    )

            if response is None:
                client_ip = "unknown" if request.client is None else request.client.host
                if (
                    config.web.trust_proxy_headers
                    and client_ip in config.web.trusted_proxy_ips
                    and request.headers.get("x-forwarded-for")
                ):
                    client_ip = request.headers["x-forwarded-for"].split(",", 1)[0].strip()
                now = time.monotonic()
                window = rate_windows.setdefault(client_ip, deque())
                rate_windows.move_to_end(client_ip)
                while len(rate_windows) > maximum_rate_limit_identities:
                    rate_windows.popitem(last=False)
                while window and window[0] <= now - 60:
                    window.popleft()
                if len(window) >= config.web.requests_per_minute:
                    response = JSONResponse(
                        status_code=429,
                        content={
                            "detail": "rate_limit_exceeded",
                            "correlation_id": request_id,
                        },
                    )
                else:
                    window.append(now)

            if response is None:
                response = await call_next(request)
            elapsed_ms = (time.monotonic() - request.state.request_started_monotonic) * 1_000.0
            request.state.elapsed_ms = elapsed_ms
            response.headers.update(security_headers(request_id))
            structured_log(
                LOGGER,
                event="request_completed",
                request_id=request_id,
                method=request.method,
                route=route_template(request),
                response_status=int(response.status_code),
                elapsed_ms=round(elapsed_ms, 3),
                run_id=config.runtime.run_id,
                replay_execution_id=replay_controller.status().execution_id,
                **operation_log_fields(request),
            )
            return response
        except Exception:
            request.state.elapsed_ms = (
                time.monotonic() - request.state.request_started_monotonic
            ) * 1_000.0
            raise
        finally:
            reset_request_metrics(metrics_token)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
        response = JSONResponse(
            status_code=422,
            content={
                "detail": "invalid_request",
                "correlation_id": request_id,
            },
        )
        response.headers.update(security_headers(request_id))
        return response

    @app.exception_handler(Exception)
    async def production_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
        structured_log(
            LOGGER,
            event="unexpected_exception",
            level=logging.ERROR,
            exception=exc,
            request_id=request_id,
            method=request.method,
            route=route_template(request),
            response_status=500,
            elapsed_ms=round(float(getattr(request.state, "elapsed_ms", 0.0)), 3),
            run_id=config.runtime.run_id,
            replay_execution_id=replay_controller.status().execution_id,
            exception_class=type(exc).__name__,
            error_code="WEB_INTERNAL_ERROR",
            **operation_log_fields(request),
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "internal_error"},
        )
        response.headers.update(security_headers(request_id))
        return response

    def static_no_order_safety_projection() -> dict[str, Any]:
        def load() -> dict[str, Any]:
            forbidden_methods = set(FORBIDDEN_BROKER_SURFACE)
            adapter_methods = set(dir(IBKRMarketDataAdapter))
            route_safety = all(
                not _path_exposes_forbidden_broker_resource(str(getattr(route, "path", "")))
                and (
                    set(getattr(route, "methods", set()) or set()) <= {"GET", "HEAD", "OPTIONS"}
                    or (
                        str(getattr(route, "path", "")) in replay_control_paths
                        and set(getattr(route, "methods", set()) or set()) == {"POST"}
                    )
                )
                for route in app.routes
                if str(getattr(route, "path", "")).startswith("/api/")
            )
            read_only = store.read_only_verification()
            checks = {
                "risk_trading_enabled_false": config.risk.trading_enabled is False,
                "web_has_no_broker_reference": not any(
                    hasattr(app.state, name)
                    for name in ("broker", "recorder", "adapter", "execution_client")
                ),
                "web_database_opened_read_only": bool(read_only["verified"]),
                "http_order_routes_absent": route_safety,
                "adapter_order_methods_absent": not bool(
                    forbidden_methods.intersection(adapter_methods)
                ),
                "ibkr_read_only_configured": config.ibkr.read_only
                and config.ibkr.expected_environment in {"read_only", "live_read_only", "paper"},
                "ibkr_socket_loopback_only": ipaddress.ip_address(config.ibkr.host).is_loopback,
                "runtime_order_surface_absent": route_safety
                and not bool(forbidden_methods.intersection(adapter_methods)),
            }
            return {"checks": checks, "database": read_only}

        return static_no_order_projection_cache.get(load)

    def no_order_safety_projection(
        *,
        broker_state_mutation_count_zero: bool | None = None,
    ) -> dict[str, Any]:
        static = static_no_order_safety_projection()
        if broker_state_mutation_count_zero is None:
            operational = store.recorder_operational_state(
                now=datetime.now(UTC),
                prospective_start_utc=config.runtime.prospective_start_utc,
                thresholds=operational_thresholds(config),
            )
            broker_state_mutation_count_zero = (
                operational.conditions["broker_state_mutation_count_zero"] is True
                if operational.run_id is not None
                else config.runtime.source == "replay"
                and replay_controller.status().broker_state_mutated is False
            )
        checks = {
            **static["checks"],
            "broker_state_mutation_count_zero": broker_state_mutation_count_zero,
        }
        return {
            **checks,
            "aggregate_no_order_verdict": all(checks.values()),
            "ibkr_read_only_evidence": {
                "configured": checks["ibkr_read_only_configured"],
                "locally_enforced_by_adapter_surface": checks["adapter_order_methods_absent"],
                "observed_runtime_broker_mutation_count_zero": checks[
                    "broker_state_mutation_count_zero"
                ],
                "external_ibkr_environment_verification": "not_externally_verifiable",
            },
            "database": static["database"],
        }

    def operational_alerts(
        operational: Any,
    ) -> list[dict[str, str]]:
        state = (
            operational.state.value if hasattr(operational, "state") else str(operational["state"])
        )
        reason = (
            operational.reason_code
            if hasattr(operational, "reason_code")
            else str(operational["reason_code"])
        )
        if state in {
            "RECORDING_HEALTHY",
            "MARKET_CLOSED",
            "WAITING_FOR_PROSPECTIVE_START",
        }:
            return []
        severity = "fatal" if state in {"INGESTION_FATAL", "STORAGE_FATAL"} else "warning"
        return [
            {
                "stable_error_code": reason,
                "severity": severity,
                "state": state,
            }
        ]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        runtime = store.runtime_projection()
        operational = store.recorder_operational_state(
            now=datetime.now(UTC),
            prospective_start_utc=config.runtime.prospective_start_utc,
            thresholds=operational_thresholds(config),
        )
        runtime_artifacts = store.runtime_artifact_verification()
        artifacts = operational_artifacts()
        bundle = artifacts["bundle"]
        parity = artifacts["parity"]
        ibkr_api = artifacts["ibkr_api"]
        parallel_credential_configured = bool(
            os.environ.get(config.parallel_validation.credential_status_env) == "1"
        )
        transfer_projection = store.source_transfer_status_v0()
        cross_vendor_validation_status = classify_cross_vendor_validation_status(
            enabled=config.parallel_validation.enabled,
            credential_configured=parallel_credential_configured,
            valid_session_count=transfer_projection["valid_session_count"],
            decision=transfer_projection["decision"],
            latest_diagnostic_status=transfer_projection["latest_diagnostic_status"],
        )
        blocker_candidates = [
            *(item["blocker_code"] for item in runtime["blockers"]),
            *bundle["blockers"],
            *runtime_artifacts["blockers"],
            ibkr_api["blocker"] if config.runtime.source == "ibkr" else None,
            (None if operational.scientific_recording_valid else operational.reason_code),
        ]
        blockers = list(dict.fromkeys(str(blocker) for blocker in blocker_candidates if blocker))
        no_order = no_order_safety_projection()
        if not no_order["aggregate_no_order_verdict"]:
            blockers.append("NO_ORDER_INVARIANT_FAILED")
        return {
            "status": "blocked" if blockers else "healthy",
            "research_only": True,
            "trading_status": "LIVE TRADING DISABLED",
            "instance_identity": config.runtime.instance_id,
            "application": {
                "version": config.runtime.app_version,
                "git_commit": config.runtime.git_commit,
            },
            "recorder": {
                "mode": config.runtime.mode,
                "run_id": config.runtime.run_id,
                "lease": runtime["recorder_lease"],
                "operational_status": operational.state.value,
                "operational": operational.model_dump(mode="json"),
            },
            "operational_alerts": operational_alerts(operational),
            "ibkr": runtime["ibkr_connection"],
            "ibkr_api": ibkr_api,
            "market_data": {
                "latest": runtime["latest_capture"],
                "line_budget": config.ibkr.market_data_line_budget,
                "reserved_headroom": config.ibkr.reserved_line_headroom,
                "current_budget": runtime["market_data_budget"],
            },
            "database": store.database_health(),
            "active_bundle": bundle,
            "runtime_artifact_verification": runtime_artifacts,
            "feature_parity": {
                "scope": "legacy_m1_diagnostic_not_frozen_m1c_runtime_gate",
                "scoring_allowed": parity["scoring_allowed"],
                "blocker": parity["blocker"],
                "diagnostic_warning": parity["diagnostic_warning"],
                "diagnostic_only": True,
                "counts": parity["counts"],
            },
            "parallel_validation": {
                "enabled": config.parallel_validation.enabled,
                "provider": config.parallel_validation.provider,
                "credential_configured": parallel_credential_configured,
                "capture_delay_seconds": (config.parallel_validation.capture_delay_seconds),
                "latest_capture": runtime["parallel_source_capture"],
                "cross_vendor_validation_status": cross_vendor_validation_status,
                "diagnostic_only": True,
                "prospective_ibkr_evidence_allowed": True,
                "blocker": None,
            },
            "previous_session_context": runtime["previous_session_context"],
            "last_completed_bar": runtime["last_completed_bar"],
            "latest_score": runtime["latest_score"],
            "latest_signal_episode": runtime["latest_signal_episode"],
            "blockers": blockers,
            "no_order_path_verified": no_order["aggregate_no_order_verdict"],
            "no_order_checks": no_order,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/runtime")
    def runtime() -> dict[str, Any]:
        projection = store.runtime_projection()
        projection["active_bundle"] = _active_bundle_projection(config)
        projection["feature_parity"] = _parity_projection(config)
        projection["scientific_claim_limit"] = (
            "underlying_movement_selection_not_option_profitability"
        )
        projection["claims_boundary"] = claims_boundary()
        return projection

    @app.get("/api/universe")
    def universe() -> dict[str, Any]:
        cohorts = store.universe()
        return {
            "cohorts": cohorts,
            "anchor_count": len(cohorts["anchor_frozen_20"]),
            "exploratory_count": len(cohorts["prospective_external_universe_exploratory"]),
            "pooled": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/signals")
    def signals() -> dict[str, Any]:
        return {
            "items": store.signals(),
            "score_claim": "synthetic replay or underlying movement selection only",
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/signals/{signal_id}")
    def signal_detail(signal_id: str) -> dict[str, Any]:
        result = store.signal_detail(signal_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        result["feature_parity"] = _parity_projection(config)
        result["claims_boundary"] = claims_boundary()
        return result

    @app.get("/api/shadow")
    def shadow() -> dict[str, Any]:
        return {
            "ledger": "quoted_research_ledger",
            "items": store.shadow(),
            "paper_ledger": {"implemented": False, "items": []},
            "claim_limit": "observed_quotes_not_proof_of_option_profitability",
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/shadow/{structure_id}")
    def shadow_detail(structure_id: str) -> dict[str, Any]:
        result = store.shadow_detail(structure_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        result["claims_boundary"] = claims_boundary()
        return result

    @app.get("/api/audit")
    def audit() -> dict[str, Any]:
        return {
            "ordered": True,
            "items": store.audit(),
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/config/public")
    def safe_config() -> dict[str, object]:
        return {
            **public_config(config),
            "claims_boundary": claims_boundary(),
        }

    def recorder_status_projection(
        *,
        include_gap_details: bool = True,
    ) -> dict[str, Any]:
        status = store.recorder_status_v0(
            now=datetime.now(UTC),
            prospective_start_utc=config.runtime.prospective_start_utc,
            thresholds=operational_thresholds(config),
            include_gap_details=include_gap_details,
        )
        artifacts = operational_artifacts()
        m1c_parity = artifacts["m1c_live_parity"]
        direction_parity = artifacts["direction_live_parity"]
        completed_bar = status["latest_completed_bar"]
        last_completed = None if completed_bar is None else completed_bar["bar_end_utc"]
        next_expected = (
            None
            if last_completed is None
            else (
                datetime.fromisoformat(str(last_completed)).astimezone(UTC) + timedelta(minutes=5)
            ).isoformat()
        )
        status.update(
            {
                "banner": "RECORD ONLY — NO ORDERS",
                "market_data_type_required": config.ibkr.market_data_type_required,
                "capacity": {
                    "level1": {
                        "used": status["subscriptions"].get("level1", 0),
                        "available": config.ibkr.market_data_line_budget,
                    },
                    "tick_by_tick": {
                        "used": status["subscriptions"].get("tick_by_tick", 0),
                        "available": config.ibkr.max_tick_by_tick_subscriptions,
                    },
                    "depth": {
                        "used": status["subscriptions"].get("depth", 0),
                        "available": config.ibkr.max_depth_subscriptions,
                    },
                    "option": {
                        "used": status["subscriptions"].get("option", 0),
                        "available": config.ibkr.max_option_subscriptions,
                    },
                },
                "model_parity": {
                    "m1c": (
                        "not_run"
                        if m1c_parity is None
                        else "passed"
                        if m1c_parity.get("passed") is True
                        else "failed"
                    ),
                    "direction": (
                        "not_run"
                        if direction_parity is None
                        else "passed"
                        if direction_parity.get("passed") is True
                        else "failed"
                    ),
                },
                "bar": {
                    "source": (None if completed_bar is None else completed_bar["source"]),
                    "last_completed": (last_completed),
                    "next_expected_completion": next_expected,
                    "freshness_seconds": (
                        None
                        if last_completed is None
                        else max(
                            0.0,
                            (
                                datetime.now(UTC)
                                - datetime.fromisoformat(str(last_completed)).astimezone(UTC)
                            ).total_seconds(),
                        )
                    ),
                    "source_completeness": (
                        None if completed_bar is None else completed_bar["source_completeness"]
                    ),
                    "compatibility_status": "parity_gated",
                },
                "replay": replay_controller.status().model_dump(mode="json"),
                "runtime_artifact_verification": (store.runtime_artifact_verification()),
                "operational_alerts": operational_alerts(status["operational"]),
                "no_order_checks": no_order_safety_projection(),
                "claims_boundary": claims_boundary(),
            }
        )
        return status

    @app.get("/api/recorder/status")
    def recorder_status() -> dict[str, Any]:
        return recorder_status_projection()

    @app.get("/api/dashboard/summary")
    def dashboard_summary() -> dict[str, Any]:
        """Return only bounded, frequently changing primary-view projections."""

        observed = datetime.now(UTC)
        with store.snapshot_transaction():
            status = store.dashboard_summary_v0(
                now=observed,
                prospective_start_utc=config.runtime.prospective_start_utc,
                thresholds=operational_thresholds(config),
            )

        artifacts = operational_artifacts()
        operational = status["operational"]
        replay_status = replay_controller.status()
        mutation_count_zero = bool(
            operational.conditions["broker_state_mutation_count_zero"] is True
            if operational.run_id is not None
            else config.runtime.source == "replay" and replay_status.broker_state_mutated is False
        )
        no_order = no_order_safety_projection(broker_state_mutation_count_zero=mutation_count_zero)
        static_alert_codes = [
            *artifacts["bundle"]["blockers"],
            artifacts["parity"]["blocker"],
            (artifacts["ibkr_api"]["blocker"] if config.runtime.source == "ibkr" else None),
        ]
        alerts_by_code = {
            str(item["blocker_code"]): {
                "code": str(item["blocker_code"]),
                "severity": str(item["severity"]),
                "message": str(item["message"]),
                "source": "recorder_projection",
            }
            for item in status["alerts"]
        }
        for code in static_alert_codes:
            if code:
                alerts_by_code.setdefault(
                    str(code),
                    {
                        "code": str(code),
                        "severity": "warning",
                        "message": str(code),
                        "source": "startup_verification",
                    },
                )
        for item in operational_alerts(operational):
            code = str(item["stable_error_code"])
            alerts_by_code.setdefault(
                code,
                {
                    "code": code,
                    "severity": str(item["severity"]),
                    "message": code,
                    "source": "operational_state",
                },
            )
        if not no_order["aggregate_no_order_verdict"]:
            alerts_by_code["NO_ORDER_INVARIANT_FAILED"] = {
                "code": "NO_ORDER_INVARIANT_FAILED",
                "severity": "fatal",
                "message": "NO_ORDER_INVARIANT_FAILED",
                "source": "no_order_verification",
            }
        timestamps = operational.timestamps
        subscriptions = status["subscriptions"]
        return {
            "summary_at_utc": observed.isoformat(),
            "run_id": status["run_id"],
            "recorder": {
                "state": operational.state.value,
                "reason_code": operational.reason_code,
                "heartbeat_at_utc": timestamps["process_heartbeat_at_utc"],
                "latest_callback_received_at_utc": timestamps["latest_callback_received_at_utc"],
                "latest_callback_durably_admitted_at_utc": timestamps[
                    "latest_callback_durably_admitted_at_utc"
                ],
                "latest_inbox_acknowledgement_at_utc": timestamps[
                    "latest_inbox_acknowledgement_at_utc"
                ],
                "callback_inbox": {
                    "pending": status["pending_inbox_count"],
                    "leased": status["leased_inbox_count"],
                    "backlog": operational.inbox["backlog"],
                },
                "latest_completed_five_minute_bar": status["completed_bar"],
                "latest_successful_checkpoint": status["checkpoint"],
                "latest_episode": status["episode"],
            },
            "ibkr": {
                "connection_state": (
                    None if status["connection"] is None else status["connection"]["state"]
                ),
                "connection": status["connection"],
                "subscriptions": {
                    "by_kind": subscriptions,
                    "total": sum(subscriptions.values()),
                },
            },
            "alerts": list(alerts_by_code.values()),
            "no_order": {
                "aggregate_no_order_verdict": no_order["aggregate_no_order_verdict"],
                "static_surface_verified": all(
                    bool(value)
                    for name, value in no_order.items()
                    if name
                    in {
                        "risk_trading_enabled_false",
                        "web_has_no_broker_reference",
                        "web_database_opened_read_only",
                        "http_order_routes_absent",
                        "adapter_order_methods_absent",
                        "ibkr_read_only_configured",
                        "ibkr_socket_loopback_only",
                        "runtime_order_surface_absent",
                    }
                ),
                "broker_state_mutation_count_zero": no_order["broker_state_mutation_count_zero"],
                "research_only": True,
                "execution_enabled": False,
                "order_routing": "disabled",
            },
            "replay": replay_status.model_dump(mode="json"),
        }

    @app.get("/api/opening-leader-continuation-v0")
    def opening_leader_continuation_v0() -> dict[str, Any]:
        projection = store.opening_leader_continuation_v0()
        projection["claims_boundary"] = claims_boundary()
        return projection

    @app.get("/api/market-data-budget")
    def market_data_budget() -> dict[str, Any]:
        return {
            **store.market_data_budget_dashboard_v0(),
            "banner": "RECORD ONLY — NO ORDERS",
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/source-transfer")
    def source_transfer() -> dict[str, Any]:
        aggregate = _json_artifact(config.paths.aggregate_transfer_report)
        projection = store.source_transfer_status_v0()
        credential_configured = bool(
            os.environ.get(config.parallel_validation.credential_status_env) == "1"
        )
        validation_status = classify_cross_vendor_validation_status(
            enabled=config.parallel_validation.enabled,
            credential_configured=credential_configured,
            valid_session_count=projection["valid_session_count"],
            decision=projection["decision"],
            latest_diagnostic_status=projection["latest_diagnostic_status"],
        )
        return {
            **projection,
            "aggregate": aggregate,
            "market_data_source": "ibkr",
            "historical_research_source": "eodhd",
            "cross_vendor_validation_status": validation_status,
            "cross_vendor_validation_status_code": (
                "cross_vendor_validation_not_configured"
                if validation_status == "not_configured"
                else f"cross_vendor_validation_{validation_status}"
            ),
            "cross_vendor_validation_diagnostic_only": True,
            "prospective_ibkr_evidence_allowed": True,
            "recorder_blocking": False,
            "exact_vendor_bar_equality_required": False,
            "historical_decision": "blocked_insufficient_low_tail_support",
            "strategy_profitability_decision_allowed": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/reports/daily")
    def daily_report_packages() -> dict[str, Any]:
        return {
            "items": _daily_report_packages(config.paths.prospective_report_root),
            "package_contents": [
                "session_summary.json",
                "ibkr_bar_quality.csv",
                "m1c_ibkr_predictions.csv",
                "eodhd_ibkr_bar_comparison.csv",
                "eodhd_ibkr_feature_comparison.csv",
                "eodhd_ibkr_probability_comparison.csv",
                "tail_membership_comparison.csv",
                "episode_comparison.csv",
                "market_data_budget_report.json",
                "subscription_lifecycle.csv",
                "option_episode_quality.csv",
                "shadow_outcomes.csv",
                "skipped_recordings.csv",
                "report.md",
            ],
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/reports/daily/{session_date}/{archive_name}")
    def download_daily_report(
        session_date: str,
        archive_name: str,
    ) -> FileResponse:
        root = config.paths.prospective_report_root
        try:
            datetime.strptime(session_date, "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="not_found") from exc
        if (
            root is None
            or Path(archive_name).name != archive_name
            or not archive_name.startswith("chatgpt-report-package-")
            or not archive_name.endswith(".zip")
        ):
            raise HTTPException(status_code=404, detail="not_found")
        archive = _safe_contained_file(root, root / session_date / archive_name)
        if archive is None:
            raise HTTPException(status_code=404, detail="not_found")
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=archive.name,
        )

    @app.get("/api/recorder/capabilities")
    def recorder_capabilities() -> dict[str, Any]:
        artifact = _json_artifact(config.paths.ibkr_capability_manifest)
        observation = (
            None
            if artifact is None or not isinstance(artifact.get("observation"), dict)
            else artifact["observation"]
        )
        manifest = (
            None
            if artifact is None
            else {
                **artifact,
                **({} if observation is None else observation),
            }
        )
        return {
            "manifest": manifest,
            "scientific_recording_valid": (
                False if artifact is None else artifact.get("scientific_recording_valid") is True
            ),
            "diagnostic_display_allowed": True,
            "required_market_data_type": "live",
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/recorder/session-reports")
    def recorder_session_reports() -> dict[str, Any]:
        return {
            "items": store.session_reports_v0(),
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/universe/live")
    def universe_live() -> dict[str, Any]:
        return {
            "items": store.universe_live_v0(),
            "classification_label": "directional research classification",
            "recommendations": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/episodes")
    def episodes() -> dict[str, Any]:
        return {
            "items": store.episodes_v0(),
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str) -> dict[str, Any]:
        result = store.episode_v0(episode_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        result["claims_boundary"] = claims_boundary()
        return result

    @app.get("/api/episodes/{episode_id}/microstructure")
    def episode_microstructure(episode_id: str) -> dict[str, Any]:
        return {
            "items": store.episode_microstructure_v0(episode_id),
            "quote_series": store.episode_quote_series_v0(
                episode_id,
                maximum_points=config.web.quote_series_maximum_points,
                maximum_input_rows=config.web.parquet_projection_maximum_input_rows,
            ),
            "latest_depth_snapshot": store.episode_depth_snapshot_v0(
                episode_id,
                maximum_input_rows=config.web.parquet_projection_maximum_input_rows,
            ),
            "label": "microstructure descriptive score",
            "direction_model_fitted": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/episodes/{episode_id}/options")
    def episode_options(episode_id: str) -> dict[str, Any]:
        return {
            "items": store.episode_options_v0(episode_id),
            "top_of_book_updates": True,
            "true_tick_by_tick_options_claimed": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/shadow-outcomes")
    def shadow_outcomes() -> dict[str, Any]:
        return {
            "items": store.shadow_outcomes_v0(),
            "primary_return": "ask_entry_to_bid_exit",
            "option_pnl_is_shadow_quote_pnl": True,
            "claims_boundary": claims_boundary(),
        }

    def _virtual_ledgers_projection(
        *,
        opening_limit: int,
        opening_leader_limit: int,
        quiet_limit: int,
        quiet_capture_limit: int,
    ) -> dict[str, Any]:
        """Build a bounded projection of segregated virtual evidence."""

        return {
            "opening_reversal": {
                "ledger_scope": "opening_reversal_v1_1",
                "items": store.opening_reversal_virtual_positions_v1(
                    limit=opening_limit,
                ),
                "item_limit": opening_limit,
                "entry_convention": "first_valid_live_ask_at_or_after_entry",
                "exit_convention": ("first_valid_live_bid_at_or_after_frozen_15m_horizon"),
                "quantity": 1,
                "opposite_leg_is_control_only": True,
            },
            "quiet_state": {
                "ledger_scope": "quiet_state_short_premium",
                "capture_ledger_scope": "quiet_state_short_premium_capture",
                "capture_items": store.quiet_state_virtual_captures_v1(
                    limit=quiet_capture_limit,
                ),
                "capture_item_limit": quiet_capture_limit,
                "items": store.quiet_state_virtual_positions_v1(
                    limit=quiet_limit,
                ),
                "item_limit": quiet_limit,
                "fill_convention": ("open_short_bid_long_ask_close_short_ask_long_bid"),
                "latest_quotes_are_diagnostic_only": True,
                "controls_included": False,
                "long_option_candidates_included": False,
            },
            "opening_leader": {
                "ledger_scope": "opening_leader_option_strategy_accounting_v0",
                "items": store.opening_leader_option_accounting_v0(
                    limit=opening_leader_limit,
                ),
                "item_limit": opening_leader_limit,
                "entry_observation": "E0",
                "strategies": ["P20", "P30", "BPS20"],
                "fill_convention": ("open_short_bid_long_ask_close_short_ask_long_bid"),
                "executable_pnl_is_primary": True,
                "greek_attribution_is_diagnostic_only": True,
                "margin_is_observed_only": True,
            },
            "ledgers_combined_for_analysis": False,
            "execution_claimed": False,
            "broker_positions_claimed": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/virtual-ledgers")
    def virtual_ledgers(
        opening_limit: int = 200,
        opening_leader_limit: int = 500,
        quiet_limit: int = 500,
        quiet_capture_limit: int = 50,
    ) -> dict[str, Any]:
        """Expose bounded segregated evidence ledgers without broker semantics."""

        requested_limits = (
            opening_limit,
            opening_leader_limit,
            quiet_limit,
            quiet_capture_limit,
        )
        if any(limit < 1 or limit > 1000 for limit in requested_limits):
            raise HTTPException(status_code=422, detail="virtual_ledger_limit_out_of_range")
        return _virtual_ledgers_projection(
            opening_limit=opening_limit,
            opening_leader_limit=opening_leader_limit,
            quiet_limit=quiet_limit,
            quiet_capture_limit=quiet_capture_limit,
        )

    def audit_events_projection(
        *,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        if limit > config.web.audit_page_maximum_items:
            raise HTTPException(status_code=422, detail="invalid_request")
        try:
            page = store.audit_event_page_v0(limit=limit, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_cursor") from exc
        return {
            "ordered": True,
            **page,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/audit/events")
    def audit_events(
        limit: int = Query(default=100, ge=1, le=1_000),
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return audit_events_projection(limit=limit, cursor=cursor)

    @app.get("/api/audit/raw-events/{content_hash}")
    def raw_event_detail(
        content_hash: str,
        limit: int = Query(default=100, ge=1, le=100),
    ) -> dict[str, Any]:
        if len(content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in content_hash
        ):
            raise HTTPException(status_code=404, detail="not_found")
        result = store.raw_event_detail_v0(
            content_hash,
            limit=limit,
            maximum_input_rows=config.web.parquet_projection_maximum_input_rows,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        return {
            **result,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/status")
    def quiet_state_status() -> dict[str, Any]:
        return {
            **store.quiet_state_status_v0(),
            "banner": "RESEARCH ONLY — RECORD ONLY — NO ORDERS",
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/universe")
    def quiet_state_universe() -> dict[str, Any]:
        return {
            "items": store.quiet_state_universe_v0(),
            "directional_model_required": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/episodes")
    def quiet_state_episodes() -> dict[str, Any]:
        return {
            "items": store.quiet_state_episodes_v0(),
            "cohorts_merged": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/episodes/{episode_id}")
    def quiet_state_episode(episode_id: str) -> dict[str, Any]:
        result = store.quiet_state_episode_v0(episode_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        result["claims_boundary"] = claims_boundary()
        return result

    @app.get("/api/quiet-state/episodes/{episode_id}/options")
    def quiet_state_episode_options(episode_id: str) -> dict[str, Any]:
        return {
            "items": store.quiet_state_episode_options_v0(episode_id),
            "bounded_chain": True,
            "full_chain_streamed": False,
            "expiry_substitution_allowed": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/shadow-structures")
    def quiet_state_shadow_structures() -> dict[str, Any]:
        return {
            "items": store.quiet_state_shadow_structures_v0(),
            "primary_fill_convention": "conservative_observed_bid_ask",
            "strategy_selection_allowed": False,
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/concentration-audit")
    def quiet_state_concentration_audit() -> dict[str, Any]:
        return {
            **_concentration_audit_projection(config.paths.quiet_state_concentration_audit_root),
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/quiet-state/session-quality")
    def quiet_state_session_quality() -> dict[str, Any]:
        return {
            "items": store.quiet_state_session_quality_v0(),
            "quality_cohorts": [
                "all_attempted_structures",
                "complete_quote_quality_structures",
                "strict_quote_quality_structures",
            ],
            "claims_boundary": claims_boundary(),
        }

    @app.get("/api/dashboard-snapshot")
    def dashboard_snapshot(request: Request) -> dict[str, Any]:
        """Return the main dashboard from one SQLite read transaction."""

        section_builders = {
            "health": health,
            "status": recorder_status,
            "capabilities": recorder_capabilities,
            "universe": universe_live,
            "episodes": episodes,
            "shadow": shadow_outcomes,
            "virtual_ledgers": lambda: _virtual_ledgers_projection(
                opening_limit=25,
                opening_leader_limit=50,
                quiet_limit=50,
                quiet_capture_limit=25,
            ),
            "audit": lambda: audit_events_projection(
                limit=min(100, config.web.audit_page_maximum_items),
                cursor=None,
            ),
            "session_reports": recorder_session_reports,
            "quiet_status": quiet_state_status,
            "quiet_universe": quiet_state_universe,
            "quiet_episodes": quiet_state_episodes,
            "quiet_shadow": quiet_state_shadow_structures,
            "quiet_session_quality": quiet_state_session_quality,
            "concentration_audit": quiet_state_concentration_audit,
            "budget": market_data_budget,
            "transfer": source_transfer,
            "report_packages": daily_report_packages,
        }
        sections: dict[str, Any] = {}
        errors: dict[str, dict[str, str]] = {}
        snapshot_at = datetime.now(UTC)
        with store.snapshot_transaction():
            for name, build_section in section_builders.items():
                try:
                    sections[name] = build_section()
                except Exception as exc:
                    error_code = f"DASHBOARD_SECTION_{name.upper()}_UNAVAILABLE"
                    errors[name] = {
                        "error_code": error_code,
                    }
                    structured_log(
                        LOGGER,
                        event="dashboard_section_error",
                        level=logging.ERROR,
                        exception=exc,
                        request_id=str(getattr(request.state, "request_id", "unavailable")),
                        method=request.method,
                        route=route_template(request),
                        response_status=200,
                        elapsed_ms=round(
                            (
                                time.monotonic()
                                - float(
                                    getattr(
                                        request.state,
                                        "request_started_monotonic",
                                        time.monotonic(),
                                    )
                                )
                            )
                            * 1_000.0,
                            3,
                        ),
                        run_id=config.runtime.run_id,
                        replay_execution_id=replay_controller.status().execution_id,
                        exception_class=type(exc).__name__,
                        error_code=error_code,
                        section=name,
                        **operation_log_fields(request),
                    )
        return {
            "snapshot_at_utc": snapshot_at.isoformat(),
            "sections": sections,
            "section_errors": errors,
            "partial": bool(errors),
            "claims_boundary": claims_boundary(),
        }

    @app.post("/api/replay/start")
    def replay_start(request: ReplayStartRequest) -> dict[str, Any]:
        try:
            state = replay_controller.start(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            **state.model_dump(mode="json"),
            "record_only": True,
            "ibkr_connections_attempted": 0,
            "claims_boundary": claims_boundary(),
        }

    @app.post("/api/replay/stop")
    def replay_stop() -> dict[str, Any]:
        return {
            **replay_controller.stop().model_dump(mode="json"),
            "record_only": True,
            "ibkr_connections_attempted": 0,
            "claims_boundary": claims_boundary(),
        }

    # Warm nearly-static verification only after every route is registered.
    # Configuration or release changes recreate the application; otherwise the
    # small caches refresh on the configured TTL.
    operational_artifacts()
    static_no_order_safety_projection()

    return app
