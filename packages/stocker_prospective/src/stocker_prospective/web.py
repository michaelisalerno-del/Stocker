"""Read-only FastAPI process for prospective evidence monitoring."""

from __future__ import annotations

import csv
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
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
    public_config,
    validate_runtime_safety,
)
from stocker_prospective.contract import claims_boundary
from stocker_prospective.dashboard_projection import (
    project_recorder_operational_state,
)
from stocker_prospective.evidence_replay import replay_persisted_evidence
from stocker_prospective.ibkr import official_ibkr_api_projection
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

_LOGGER = logging.getLogger("stocker_prospective.web")


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
            "blocker": report.overall_blocker,
            "counts": dict(counts),
            "report": report.model_dump(mode="json"),
        }
    except FeatureParityError as exc:
        return {
            "scoring_allowed": False,
            "blocker": str(exc).split(":", 1)[0],
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
        payload = _json_artifact(metadata_path)
        if payload is None:
            continue
        archive_name = str(payload.get("archive", ""))
        archive = metadata_path.parent / archive_name
        if (
            not archive_name
            or Path(archive_name).name != archive_name
            or archive.suffix != ".zip"
            or not archive.is_file()
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
                "buy",
                "sell",
                "credential",
                "credentials",
                "upload",
            }
        )
    )


def create_web_app(config: ProspectiveConfig) -> FastAPI:
    """Create a web app that receives no recorder or broker object."""

    validate_runtime_safety(config, object())
    store = ProspectiveReadStore(config.paths.database, run_id=config.runtime.run_id)

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
        )

    replay_controller = ReplayController(
        runner=run_replay,
        stop_timeout_seconds=config.web.replay_stop_timeout_seconds,
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

    def apply_response_headers(response: Any, *, request_id: str) -> None:
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

    @app.middleware("http")
    async def security_boundary(request: Request, call_next: Any) -> Any:
        request_id = str(uuid.uuid4())
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
                    content={"detail": "not_found"},
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
                        content={"detail": "authentication_required"},
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
                        content={"detail": "rate_limit_exceeded"},
                    )
                else:
                    window.append(now)

            if response is None:
                response = await call_next(request)
            elapsed_ms = (time.monotonic() - request.state.request_started_monotonic) * 1_000.0
            request.state.elapsed_ms = elapsed_ms
            apply_response_headers(response, request_id=request_id)
            structured_log(
                _LOGGER,
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
        response = JSONResponse(status_code=422, content={"detail": "invalid_request"})
        apply_response_headers(response, request_id=request_id)
        return response

    @app.exception_handler(Exception)
    async def production_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
        structured_log(
            _LOGGER,
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
            **operation_log_fields(request),
        )
        response = JSONResponse(status_code=500, content={"detail": "internal_error"})
        apply_response_headers(response, request_id=request_id)
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    def no_order_path_verified() -> bool:
        return config.risk.trading_enabled is False and all(
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

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        runtime = store.runtime_projection()
        recorder_operational_state = project_recorder_operational_state(
            runtime=runtime,
            prospective_start_utc=config.runtime.prospective_start_utc,
            stale_after_seconds=config.runtime.recorder_lease_stale_seconds,
        )
        bundle = _active_bundle_projection(config)
        parity = _parity_projection(config)
        ibkr_api = official_ibkr_api_projection()
        parallel_credential_configured = bool(
            os.environ.get(config.parallel_validation.credential_status_env) == "1"
        )
        parallel_blocker = (
            "blocked_missing_eodhd_server_token"
            if config.parallel_validation.enabled and not parallel_credential_configured
            else None
        )
        frozen_m1c_configured = config.paths.frozen_m1c_artifact_root is not None
        blocker_candidates = [
            *(item["blocker_code"] for item in runtime["blockers"]),
            *(
                blocker
                for blocker in bundle["blockers"]
                if not (
                    frozen_m1c_configured and blocker == "blocked_feature_source_semantics_mismatch"
                )
            ),
            None if frozen_m1c_configured else parity["blocker"],
            ibkr_api["blocker"] if config.runtime.source == "ibkr" else None,
            parallel_blocker,
        ]
        blockers = list(dict.fromkeys(str(blocker) for blocker in blocker_candidates if blocker))
        recorder_state = str(recorder_operational_state["state"])
        health_status = (
            "blocked"
            if blockers or recorder_state == "blocked"
            else "healthy"
            if recorder_state == "recording"
            else "waiting"
            if recorder_state == "waiting_for_prospective_start"
            else "degraded"
        )
        return {
            "status": health_status,
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
                "operational_status": recorder_state,
                "operational_state": recorder_operational_state,
            },
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
            "feature_parity": {
                "scope": "legacy_m1_diagnostic_not_frozen_m1c_runtime_gate",
                "scoring_allowed": parity["scoring_allowed"],
                "blocker": parity["blocker"],
                "counts": parity["counts"],
            },
            "parallel_validation": {
                "enabled": config.parallel_validation.enabled,
                "provider": config.parallel_validation.provider,
                "credential_configured": parallel_credential_configured,
                "capture_delay_seconds": (config.parallel_validation.capture_delay_seconds),
                "latest_capture": runtime["parallel_source_capture"],
                "scoring_allowed": False,
                "blocker": parallel_blocker,
            },
            "previous_session_context": runtime["previous_session_context"],
            "last_completed_bar": runtime["last_completed_bar"],
            "latest_score": runtime["latest_score"],
            "latest_signal_episode": runtime["latest_signal_episode"],
            "blockers": blockers,
            "no_order_path_verified": no_order_path_verified(),
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

    @app.get("/api/recorder/status")
    def recorder_status() -> dict[str, Any]:
        runtime_projection = store.runtime_projection()
        operational_state = project_recorder_operational_state(
            runtime=runtime_projection,
            prospective_start_utc=config.runtime.prospective_start_utc,
            stale_after_seconds=config.runtime.recorder_lease_stale_seconds,
        )
        status = store.recorder_status_v0()
        status["state"] = operational_state["state"]
        status["operational_state"] = operational_state
        m1c_parity = _json_artifact(config.paths.m1c_live_parity_report)
        direction_parity = _json_artifact(config.paths.direction_live_parity_report)
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
                "claims_boundary": claims_boundary(),
            }
        )
        return status

    @app.get("/api/dashboard/summary")
    def dashboard_summary() -> dict[str, Any]:
        recorder_projection = recorder_status()
        operational = recorder_projection["operational_state"]
        blockers = list(operational["runtime_blockers"])
        recorder_state = str(operational["state"])
        health_projection = {
            "status": (
                "blocked"
                if recorder_state == "blocked"
                else "healthy"
                if recorder_state == "recording"
                else "waiting"
                if recorder_state == "waiting_for_prospective_start"
                else "degraded"
            ),
            "research_only": True,
            "trading_status": "LIVE TRADING DISABLED",
            "recorder": {
                "mode": config.runtime.mode,
                "run_id": config.runtime.run_id,
                "operational_status": recorder_state,
                "operational_state": operational,
            },
            "blockers": blockers,
            "no_order_path_verified": no_order_path_verified(),
        }
        return {
            "health": health_projection,
            "recorder": recorder_projection,
            "latest_checkpoints": {
                "m1c": recorder_projection["latest_checkpoint"],
                "completed_bar": recorder_projection["latest_completed_bar"],
            },
            "current_universe": {
                "items": store.universe_live_v0(),
                "classification_label": "directional research classification",
                "recommendations": False,
            },
            "capacity": recorder_projection["capacity"],
            "replay": recorder_projection["replay"],
            "blockers": blockers,
            "claims_boundary": claims_boundary(),
        }

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
        return {
            **store.source_transfer_status_v0(),
            "aggregate": aggregate,
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
        archive = root / session_date / archive_name
        if not archive.is_file():
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

    @app.get("/api/audit/events")
    def audit_events(
        limit: int = Query(default=100, ge=1, le=1_000),
        cursor: str | None = None,
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

    return app
