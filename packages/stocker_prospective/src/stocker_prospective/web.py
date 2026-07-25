"""Read-only FastAPI process for prospective evidence monitoring."""

from __future__ import annotations

import os
import secrets
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
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
from stocker_prospective.ibkr import official_ibkr_api_projection
from stocker_prospective.parity import FeatureParityError, load_feature_parity_report
from stocker_prospective.read_store import ProspectiveReadStore


def _active_bundle_projection(config: ProspectiveConfig) -> dict[str, Any]:
    try:
        verification = load_active_bundle(config.paths.bundle_root)
        return {
            "bundle_id": verification.manifest.bundle_id,
            "manifest_sha256": verification.manifest_sha256,
            "verified": verification.verified,
            "blockers": verification.blockers,
        }
    except BundleError as exc:
        return {
            "bundle_id": None,
            "manifest_sha256": None,
            "verified": False,
            "blockers": [str(exc).split(":", 1)[0]],
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


def create_web_app(config: ProspectiveConfig) -> FastAPI:
    """Create a web app that receives no recorder or broker object."""

    validate_runtime_safety(config, object())
    store = ProspectiveReadStore(config.paths.database, run_id=config.runtime.run_id)
    static_root = Path(__file__).with_name("web_static")
    authentication_token: str | None = None
    if config.web.authentication_enabled:
        assert config.web.auth_token_env is not None
        authentication_token = os.environ.get(config.web.auth_token_env)
        if not authentication_token:
            raise RuntimeError(
                "blocked_unsafe_runtime_configuration: web authentication token is absent"
            )

    app = FastAPI(
        title="Stocker Prospective Evidence Recorder",
        version=config.runtime.app_version,
        docs_url=None if config.web.production else "/docs",
        redoc_url=None,
    )
    allowed_hosts = list(config.web.allowed_hosts)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")
    rate_windows: OrderedDict[str, deque[float]] = OrderedDict()
    maximum_rate_limit_identities = 4096

    @app.middleware("http")
    async def security_boundary(request: Request, call_next: Any) -> Any:
        if authentication_token is not None:
            authorization = request.headers.get("authorization", "")
            bearer = (
                authorization.removeprefix("Bearer ").strip()
                if authorization.startswith("Bearer ")
                else ""
            )
            cookie = request.cookies.get(config.web.auth_cookie_name, "")
            supplied = bearer or cookie
            if not supplied or not secrets.compare_digest(supplied, authentication_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "authentication_required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

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
            return JSONResponse(
                status_code=429,
                content={"detail": "rate_limit_exceeded"},
            )
        window.append(now)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})

    @app.exception_handler(Exception)
    async def production_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        runtime = store.runtime_projection()
        bundle = _active_bundle_projection(config)
        parity = _parity_projection(config)
        ibkr_api = official_ibkr_api_projection()
        blocker_candidates = [
            *(item["blocker_code"] for item in runtime["blockers"]),
            *bundle["blockers"],
            parity["blocker"],
            ibkr_api["blocker"] if config.runtime.source == "ibkr" else None,
        ]
        blockers = list(dict.fromkeys(str(blocker) for blocker in blocker_candidates if blocker))
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
                "scoring_allowed": parity["scoring_allowed"],
                "blocker": parity["blocker"],
                "counts": parity["counts"],
            },
            "previous_session_context": runtime["previous_session_context"],
            "last_completed_bar": runtime["last_completed_bar"],
            "latest_score": runtime["latest_score"],
            "latest_signal_episode": runtime["latest_signal_episode"],
            "blockers": blockers,
            "no_order_path_verified": config.risk.trading_enabled is False
            and all(
                not any(
                    forbidden in str(getattr(route, "path", "")).lower()
                    for forbidden in ("order", "account", "credential", "threshold", "upload")
                )
                and set(getattr(route, "methods", set()) or set()) <= {"GET", "HEAD", "OPTIONS"}
                for route in app.routes
                if str(getattr(route, "path", "")).startswith("/api/")
            ),
        }

    @app.get("/api/runtime")
    def runtime() -> dict[str, Any]:
        projection = store.runtime_projection()
        projection["active_bundle"] = _active_bundle_projection(config)
        projection["feature_parity"] = _parity_projection(config)
        projection["scientific_claim_limit"] = (
            "underlying_movement_selection_not_option_profitability"
        )
        return projection

    @app.get("/api/universe")
    def universe() -> dict[str, Any]:
        cohorts = store.universe()
        return {
            "cohorts": cohorts,
            "anchor_count": len(cohorts["anchor_frozen_20"]),
            "exploratory_count": len(cohorts["prospective_external_universe_exploratory"]),
            "pooled": False,
        }

    @app.get("/api/signals")
    def signals() -> dict[str, Any]:
        return {
            "items": store.signals(),
            "score_claim": "synthetic replay or underlying movement selection only",
        }

    @app.get("/api/signals/{signal_id}")
    def signal_detail(signal_id: str) -> dict[str, Any]:
        result = store.signal_detail(signal_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        result["feature_parity"] = _parity_projection(config)
        return result

    @app.get("/api/shadow")
    def shadow() -> dict[str, Any]:
        return {
            "ledger": "quoted_research_ledger",
            "items": store.shadow(),
            "paper_ledger": {"implemented": False, "items": []},
            "claim_limit": "observed_quotes_not_proof_of_option_profitability",
        }

    @app.get("/api/shadow/{structure_id}")
    def shadow_detail(structure_id: str) -> dict[str, Any]:
        result = store.shadow_detail(structure_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not_found")
        return result

    @app.get("/api/audit")
    def audit() -> dict[str, Any]:
        return {"ordered": True, "items": store.audit()}

    @app.get("/api/config/public")
    def safe_config() -> dict[str, object]:
        return public_config(config)

    return app
