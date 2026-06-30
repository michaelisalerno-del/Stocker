"""MCP server for safe local Stocker inspection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from hmac import compare_digest
from typing import Any
from urllib.parse import urlparse

from stocker_mcp import __version__
from stocker_mcp.oauth import OAuthMiddleware, OAuthState
from stocker_mcp.schemas import (
    READ_ONLY_ANNOTATIONS,
    TOOL_NAMES,
    get_tool_spec,
)
from stocker_mcp.schemas import (
    tool_metadata as _tool_metadata,
)
from stocker_mcp.security import SecurityError, default_context, redact_secrets
from stocker_mcp.tools import code, diagnostics, discovery, reports, research, workspace
from stocker_mcp.tools import database as database_tools

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_AUTH_TOKEN_ENV = "STOCKER_MCP_TOKEN"
DEFAULT_OAUTH_SETUP_CODE_ENV = "STOCKER_MCP_TOKEN"
DEFAULT_ALLOWED_HOSTS_ENV = "STOCKER_MCP_ALLOWED_HOSTS"
AUTH_MODES = ("oauth", "bearer")
MCP_PATH = "/mcp"
CONNECTOR_NAME = "Stocker Research"
CONNECTOR_DESCRIPTION = (
    "Read-only connector for Stocker historical research, code, reports, bars, database "
    "summaries, and hypothesis analysis. No trading or execution."
)


def tool_names() -> tuple[str, ...]:
    """Return the tools registered by the Stocker MCP server."""

    return TOOL_NAMES


def tool_metadata() -> list[dict[str, Any]]:
    """Return ChatGPT-friendly tool metadata."""

    return _tool_metadata()


def _json(payload: Any) -> str:
    return redact_secrets(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _plain_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _tool_kwargs(name: str) -> dict[str, Any]:
    from mcp.types import ToolAnnotations

    spec = get_tool_spec(name)
    return {
        "title": spec.title,
        "description": spec.description,
        "annotations": ToolAnnotations(**READ_ONLY_ANNOTATIONS),
        "meta": {"securitySchemes": [{"type": "oauth2", "scopes": ["stocker.read"]}]},
        "structured_output": True,
    }


def _normalise_allowed_host(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    if "://" in candidate:
        candidate = urlparse(candidate).netloc
    candidate = candidate.split("/", 1)[0].strip()
    return candidate.rstrip("/")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _allowed_hosts_from_env(env_var: str = DEFAULT_ALLOWED_HOSTS_ENV) -> tuple[str, ...]:
    raw = os.environ.get(env_var, "")
    values = raw.replace(",", " ").split()
    return tuple(host for host in (_normalise_allowed_host(value) for value in values) if host)


def _transport_security(
    *,
    host: str,
    allowed_hosts: list[str] | tuple[str, ...],
) -> Any:
    from mcp.server.transport_security import TransportSecuritySettings

    loopback = host in {"127.0.0.1", "localhost", "::1"}
    normalised = [
        host
        for host in (_normalise_allowed_host(value) for value in allowed_hosts)
        if host
    ]
    if not loopback and not normalised:
        return None

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"] if loopback else []
    origins = (
        ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"] if loopback else []
    )
    for allowed in normalised:
        hosts.append(allowed)
        if ":" not in allowed:
            hosts.append(f"{allowed}:*")
        origins.extend([f"https://{allowed}", f"http://{allowed}"])
        if ":" not in allowed:
            origins.extend([f"https://{allowed}:*", f"http://{allowed}:*"])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_dedupe(hosts),
        allowed_origins=_dedupe(origins),
    )


def build_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    allowed_hosts: list[str] | tuple[str, ...] = (),
) -> Any:
    """Build a FastMCP server with only read-only Stocker tools."""

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "Stocker",
        host=host,
        port=port,
        streamable_http_path=MCP_PATH,
        transport_security=_transport_security(host=host, allowed_hosts=allowed_hosts),
    )

    @mcp.tool(**_tool_kwargs("search"))
    def search(query: str, limit: int = 20) -> dict[str, Any]:
        return discovery.search(query=query, limit=limit)

    @mcp.tool(**_tool_kwargs("fetch"))
    def fetch(id: str) -> dict[str, Any]:
        return discovery.fetch(id=id)

    @mcp.tool(**_tool_kwargs("get_repo_info"))
    def get_repo_info() -> dict[str, Any]:
        return code.get_repo_info()

    @mcp.tool(**_tool_kwargs("git_status"))
    def git_status() -> dict[str, Any]:
        return code.git_status()

    @mcp.tool(**_tool_kwargs("git_log"))
    def git_log(limit: int = 10) -> dict[str, Any]:
        return code.git_log(limit=limit)

    @mcp.tool(**_tool_kwargs("git_current_commit"))
    def git_current_commit() -> dict[str, Any]:
        return code.git_current_commit()

    @mcp.tool(**_tool_kwargs("list_files"))
    def list_files(path: str = "", glob: str | None = None, limit: int = 200) -> dict[str, Any]:
        return code.list_files(path=path, glob=glob, limit=limit)

    @mcp.tool(**_tool_kwargs("search_code"))
    def search_code(query: str, path_glob: str | None = None, limit: int = 100) -> dict[str, Any]:
        return code.search_code(query=query, path_glob=path_glob, limit=limit)

    @mcp.tool(**_tool_kwargs("read_code_file"))
    def read_code_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        return code.read_code_file(path=path, start_line=start_line, end_line=end_line)

    @mcp.tool(**_tool_kwargs("git_diff"))
    def git_diff(
        ref: str | None = None,
        path: str | None = None,
        max_lines: int = 1_000,
    ) -> dict[str, Any]:
        return code.git_diff(ref=ref, path=path, max_lines=max_lines)

    @mcp.tool(**_tool_kwargs("workspace_doctor"))
    def workspace_doctor() -> dict[str, Any]:
        return workspace.workspace_doctor()

    @mcp.tool(**_tool_kwargs("list_recent_research_runs"))
    def list_recent_research_runs(limit: int = 20) -> dict[str, Any]:
        return reports.list_recent_research_runs(limit=limit)

    @mcp.tool(**_tool_kwargs("get_latest_universe_run"))
    def get_latest_universe_run(hypothesis_id: str | None = None) -> dict[str, Any]:
        return reports.get_latest_universe_run(hypothesis_id=hypothesis_id)

    @mcp.tool(**_tool_kwargs("read_universe_run"))
    def read_universe_run(run_id_or_path: str) -> dict[str, Any]:
        return reports.read_universe_run(run_id_or_path=run_id_or_path)

    @mcp.tool(**_tool_kwargs("summarise_universe_run"))
    def summarise_universe_run(run_id_or_path: str) -> dict[str, Any]:
        return reports.summarise_universe_run(run_id_or_path=run_id_or_path)

    @mcp.tool(**_tool_kwargs("list_symbol_reports"))
    def list_symbol_reports(
        run_id_or_path: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return reports.list_symbol_reports(
            run_id_or_path=run_id_or_path, symbol=symbol, limit=limit
        )

    @mcp.tool(**_tool_kwargs("read_symbol_report"))
    def read_symbol_report(
        symbol: str | None = None,
        path: str | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        return reports.read_symbol_report(symbol=symbol, path=path, experiment_id=experiment_id)

    @mcp.tool(**_tool_kwargs("compare_universe_runs"))
    def compare_universe_runs(run_a: str, run_b: str) -> dict[str, Any]:
        return reports.compare_universe_runs(run_a=run_a, run_b=run_b)

    @mcp.tool(**_tool_kwargs("find_candidate_symbols"))
    def find_candidate_symbols(run_id_or_path: str) -> dict[str, Any]:
        return reports.find_candidate_symbols(run_id_or_path=run_id_or_path)

    @mcp.tool(**_tool_kwargs("find_interesting_symbols"))
    def find_interesting_symbols(run_id_or_path: str) -> dict[str, Any]:
        return reports.find_interesting_symbols(run_id_or_path=run_id_or_path)

    @mcp.tool(**_tool_kwargs("filter_symbol_results"))
    def filter_symbol_results(
        run_id_or_path: str,
        classification: str | None = None,
        null_pass: bool | None = None,
        benchmark_pass: bool | None = None,
        min_net_return: float | None = None,
        min_trade_count: int | None = None,
    ) -> dict[str, Any]:
        return reports.filter_symbol_results(
            run_id_or_path=run_id_or_path,
            classification=classification,
            null_pass=null_pass,
            benchmark_pass=benchmark_pass,
            min_net_return=min_net_return,
            min_trade_count=min_trade_count,
        )

    @mcp.tool(**_tool_kwargs("db_list_databases"))
    def db_list_databases() -> dict[str, Any]:
        return database_tools.db_list_databases()

    @mcp.tool(**_tool_kwargs("db_list_tables"))
    def db_list_tables(database: str | None = None) -> dict[str, Any]:
        return database_tools.db_list_tables(database=database)

    @mcp.tool(**_tool_kwargs("db_describe_table"))
    def db_describe_table(table: str, database: str | None = None) -> dict[str, Any]:
        return database_tools.db_describe_table(table=table, database=database)

    @mcp.tool(**_tool_kwargs("db_preview_table"))
    def db_preview_table(
        table: str,
        database: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return database_tools.db_preview_table(table=table, database=database, limit=limit)

    @mcp.tool(**_tool_kwargs("db_get_symbol_bars"))
    def db_get_symbol_bars(
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
        database: str | None = None,
    ) -> dict[str, Any]:
        return database_tools.db_get_symbol_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            database=database,
        )

    @mcp.tool(**_tool_kwargs("db_get_latest_catalysts"))
    def db_get_latest_catalysts(
        symbol: str | None = None,
        limit: int = 100,
        database: str | None = None,
    ) -> dict[str, Any]:
        return database_tools.db_get_latest_catalysts(
            symbol=symbol, limit=limit, database=database
        )

    @mcp.tool(**_tool_kwargs("db_get_trade_attribution"))
    def db_get_trade_attribution(
        run_id: str | None = None,
        symbol: str | None = None,
        limit: int = 500,
        database: str | None = None,
    ) -> dict[str, Any]:
        return database_tools.db_get_trade_attribution(
            run_id=run_id,
            symbol=symbol,
            limit=limit,
            database=database,
        )

    @mcp.tool(**_tool_kwargs("db_select"))
    def db_select(
        sql: str,
        database: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        return database_tools.db_select(sql=sql, database=database, limit=limit)

    @mcp.tool(**_tool_kwargs("summarise_latest_research_state"))
    def summarise_latest_research_state() -> dict[str, Any]:
        return research.summarise_latest_research_state()

    @mcp.tool(**_tool_kwargs("find_positive_rejected_symbols"))
    def find_positive_rejected_symbols(run_id: str | None = None) -> dict[str, Any]:
        return research.find_positive_rejected_symbols(run_id=run_id)

    @mcp.tool(**_tool_kwargs("find_null_pass_benchmark_fail_symbols"))
    def find_null_pass_benchmark_fail_symbols(run_id: str | None = None) -> dict[str, Any]:
        return research.find_null_pass_benchmark_fail_symbols(run_id=run_id)

    @mcp.tool(**_tool_kwargs("find_benchmark_pass_rejected_symbols"))
    def find_benchmark_pass_rejected_symbols(run_id: str | None = None) -> dict[str, Any]:
        return research.find_benchmark_pass_rejected_symbols(run_id=run_id)

    @mcp.tool(**_tool_kwargs("find_common_rejection_reasons"))
    def find_common_rejection_reasons(run_id: str | None = None) -> dict[str, Any]:
        return research.find_common_rejection_reasons(run_id=run_id)

    @mcp.tool(**_tool_kwargs("get_symbol_bar_summary"))
    def get_symbol_bar_summary(
        symbol: str,
        timeframe: str = "5m",
        start: str | None = None,
        end: str | None = None,
        database: str | None = None,
    ) -> dict[str, Any]:
        return research.get_symbol_bar_summary(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            database_name=database,
        )

    @mcp.tool(**_tool_kwargs("get_symbol_recent_sessions"))
    def get_symbol_recent_sessions(
        symbol: str,
        timeframe: str = "5m",
        limit: int = 20,
        database: str | None = None,
    ) -> dict[str, Any]:
        return research.get_symbol_recent_sessions(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            database_name=database,
        )

    @mcp.tool(**_tool_kwargs("get_trade_feature_buckets"))
    def get_trade_feature_buckets(
        run_id: str | None = None,
        feature: str | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        return research.get_trade_feature_buckets(run_id=run_id, feature=feature, symbol=symbol)

    @mcp.tool(**_tool_kwargs("compare_template_runs"))
    def compare_template_runs(run_a: str, run_b: str) -> dict[str, Any]:
        return research.compare_template_runs(run_a=run_a, run_b=run_b)

    @mcp.tool(**_tool_kwargs("suggest_research_questions"))
    def suggest_research_questions(run_id: str | None = None) -> dict[str, Any]:
        return research.suggest_research_questions(run_id=run_id)

    @mcp.tool(**_tool_kwargs("export_diagnostics_zip"))
    def export_diagnostics_zip(
        output_path: str | None = None,
        include_code_summary: bool = True,
        include_reports: bool = True,
        include_db_schema: bool = True,
    ) -> dict[str, Any]:
        return diagnostics.export_diagnostics_zip(
            output_path=output_path,
            include_code_summary=include_code_summary,
            include_reports=include_reports,
            include_db_schema=include_db_schema,
        )

    @mcp.resource("stocker://workspace/doctor")
    def workspace_doctor_resource() -> str:
        return _json(workspace.workspace_doctor())

    @mcp.resource("stocker://reports/latest")
    def reports_latest_resource() -> str:
        try:
            return _json(reports.get_latest_universe_run())
        except SecurityError as exc:
            return _json({"error": str(exc)})

    @mcp.resource("stocker://repo/status")
    def repo_status_resource() -> str:
        return _json(code.git_status())

    @mcp.prompt()
    def summarise_latest_stocker_scan() -> str:
        return "Summarise the latest Stocker universe scan using get_latest_universe_run."

    @mcp.prompt()
    def compare_two_stocker_universe_runs() -> str:
        return "Compare two Stocker universe runs using compare_universe_runs."

    @mcp.prompt()
    def investigate_symbol_gate_failure() -> str:
        return "Investigate why a symbol failed Stocker gates using read_symbol_report."

    return mcp


class _BearerAuthMiddleware:
    """Small ASGI middleware for local HTTP bearer-token enforcement."""

    def __init__(self, app: Any, *, token: str, context: Any) -> None:
        self.app = app
        self.token = token
        self.context = context

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        method = str(scope.get("method", ""))
        if path.startswith(MCP_PATH):
            headers = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in scope.get("headers", [])
            }
            expected = f"Bearer {self.token}"
            supplied = headers.get("authorization", "")
            if not compare_digest(supplied, expected):
                await self._unauthorized(send, method, path)
                return
            if method == "GET" and path == MCP_PATH and "mcp-session-id" not in headers:
                await self._health(send, method, path)
                return
        await self.app(scope, receive, self._logging_send(send, method, path))

    def _logging_send(self, send: Any, method: str, path: str) -> Any:
        async def wrapped(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                self.context.log_tool_call(
                    "http_request",
                    {
                        "method": method,
                        "path": path,
                        "status_code": message.get("status"),
                    },
                )
            await send(message)

        return wrapped

    async def _unauthorized(self, send: Any, method: str, path: str) -> None:
        self.context.log_tool_call(
            "http_request",
            {"method": method, "path": path, "status_code": 401},
        )
        body = b'{"error":"unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _health(self, send: Any, method: str, path: str) -> None:
        self.context.log_tool_call(
            "http_request",
            {"method": method, "path": path, "status_code": 200},
        )
        body = b'{"status":"ok","path":"/mcp"}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _unsafe_bind_warning(host: str) -> str | None:
    if host in {"0.0.0.0", "::"}:
        return "HTTP server is configured for a public interface; prefer 127.0.0.1."
    return None


def build_http_app(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV,
    auth_mode: str = "bearer",
    oauth_setup_code_env: str = DEFAULT_OAUTH_SETUP_CODE_ENV,
    allowed_hosts: list[str] | tuple[str, ...] = (),
    require_auth: bool = True,
) -> Any:
    """Build the authenticated Streamable HTTP MCP app."""

    if auth_mode not in AUTH_MODES:
        raise SecurityError(f"unsupported HTTP auth mode: {auth_mode}")
    mcp = build_server(host=host, port=port, allowed_hosts=allowed_hosts)
    app = mcp.streamable_http_app()
    if not require_auth:
        return app
    context = default_context()
    if auth_mode == "bearer":
        token = os.environ.get(auth_token_env)
        if not token:
            raise SecurityError(f"HTTP mode requires auth token env var {auth_token_env} to be set")
        app = _BearerAuthMiddleware(app, token=token, context=context)
    else:
        setup_code = os.environ.get(oauth_setup_code_env)
        if not setup_code:
            raise SecurityError(
                f"OAuth mode requires setup code env var {oauth_setup_code_env} to be set"
            )
        app = OAuthMiddleware(app, state=OAuthState(setup_code=setup_code), context=context)
    return app


def connector_info(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV,
    auth_mode: str = "bearer",
    oauth_setup_code_env: str = DEFAULT_OAUTH_SETUP_CODE_ENV,
    allowed_hosts: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return local ChatGPT connector setup info without exposing secrets."""

    context = default_context()
    auth_set = bool(os.environ.get(auth_token_env))
    oauth_setup_code_set = bool(os.environ.get(oauth_setup_code_env))
    if auth_mode == "oauth":
        auth = {
            "enabled": True,
            "type": "oauth",
            "setup_code_env_var": oauth_setup_code_env,
            "setup_code_env_var_set": oauth_setup_code_set,
            "authorization_metadata": "https://YOUR-TUNNEL-URL/.well-known/oauth-authorization-server",
            "protected_resource_metadata": (
                "https://YOUR-TUNNEL-URL/.well-known/oauth-protected-resource"
            ),
        }
    else:
        auth = {
            "enabled": True,
            "type": "bearer",
            "env_var": auth_token_env,
            "env_var_set": auth_set,
            "header": f"Authorization: Bearer <{auth_token_env}>",
        }
    return {
        "local_url": f"http://{host}:{port}{MCP_PATH}",
        "expected_https_tunnel_url": "https://YOUR-TUNNEL-URL/mcp",
        "connector_name": CONNECTOR_NAME,
        "connector_description": CONNECTOR_DESCRIPTION,
        "auth": auth,
        "tools": list(tool_names()),
        "tool_count": len(tool_names()),
        "security_mode": "read-only",
        "repo_root": str(context.repo_root),
        "stocker_home": str(context.stocker_home),
        "allowed_hosts": list(allowed_hosts),
        "docs_path": "docs/chatgpt_connector.md",
    }


def doctor(
    transport: str = "stdio",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV,
    auth_mode: str = "bearer",
    oauth_setup_code_env: str = DEFAULT_OAUTH_SETUP_CODE_ENV,
    allowed_hosts: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return safe MCP server diagnostics."""

    context = default_context()
    http_deps: dict[str, bool] = {}
    for module in ("starlette", "uvicorn"):
        try:
            __import__(module)
        except ImportError:
            http_deps[module] = False
        else:
            http_deps[module] = True
    stdio_can_initialise = True
    http_can_initialise = all(http_deps.values())
    try:
        build_server(host=host, port=port, allowed_hosts=allowed_hosts)
    except Exception:
        stdio_can_initialise = False
        http_can_initialise = False
    auth_env_ready = bool(os.environ.get(auth_token_env))
    oauth_setup_ready = bool(os.environ.get(oauth_setup_code_env))
    if transport == "http" and (
        (auth_mode == "bearer" and auth_env_ready)
        or (auth_mode == "oauth" and oauth_setup_ready)
    ):
        try:
            build_http_app(
                host=host,
                port=port,
                auth_token_env=auth_token_env,
                auth_mode=auth_mode,
                oauth_setup_code_env=oauth_setup_code_env,
                allowed_hosts=allowed_hosts,
            )
        except Exception:
            http_can_initialise = False
    return {
        "version": __version__,
        "workspace": workspace.workspace_doctor(context=context),
        "tool_count": len(tool_names()),
        "tools": list(tool_names()),
        "stdio": {"can_initialise": stdio_can_initialise},
        "http": {
            "can_initialise": http_can_initialise,
            "local_url": f"http://{host}:{port}{MCP_PATH}",
            "host": host,
            "port": port,
            "path": MCP_PATH,
            "auth_enabled": True,
            "auth_mode": auth_mode,
            "auth_env_var": auth_token_env,
            "auth_env_var_set": auth_env_ready,
            "oauth_setup_code_env_var": oauth_setup_code_env,
            "oauth_setup_code_env_var_set": oauth_setup_ready,
            "allowed_hosts": list(allowed_hosts),
            "dependencies": http_deps,
            "unsafe_bind_warning": _unsafe_bind_warning(host),
        },
        "security_mode": "read-only",
    }


def _startup_payload(
    host: str,
    port: int,
    auth_token_env: str,
    auth_mode: str,
    oauth_setup_code_env: str,
    allowed_hosts: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    context = default_context()
    return {
        "local_mcp_url": f"http://{host}:{port}{MCP_PATH}",
        "transport": "http",
        "repo_root": str(context.repo_root),
        "stocker_home": str(context.stocker_home),
        "tool_count": len(tool_names()),
        "auth_enabled": True,
        "auth_mode": auth_mode,
        "auth_token_env": auth_token_env,
        "auth_token_env_set": bool(os.environ.get(auth_token_env)),
        "oauth_setup_code_env": oauth_setup_code_env,
        "oauth_setup_code_env_set": bool(os.environ.get(oauth_setup_code_env)),
        "allowed_hosts": list(allowed_hosts),
        "unsafe_bind_warning": _unsafe_bind_warning(host),
    }


def _run_http(
    host: str,
    port: int,
    auth_token_env: str,
    auth_mode: str,
    oauth_setup_code_env: str,
    allowed_hosts: list[str] | tuple[str, ...],
) -> None:
    if auth_mode == "bearer" and not os.environ.get(auth_token_env):
        raise SecurityError(f"set {auth_token_env} before starting HTTP bearer mode")
    if auth_mode == "oauth" and not os.environ.get(oauth_setup_code_env):
        raise SecurityError(f"set {oauth_setup_code_env} before starting HTTP OAuth mode")
    print(
        _json(
            _startup_payload(
                host,
                port,
                auth_token_env,
                auth_mode,
                oauth_setup_code_env,
                allowed_hosts,
            )
        ),
        file=sys.stderr,
    )
    app = build_http_app(
        host=host,
        port=port,
        auth_token_env=auth_token_env,
        auth_mode=auth_mode,
        oauth_setup_code_env=oauth_setup_code_env,
        allowed_hosts=allowed_hosts,
    )
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stocker-mcp",
        description="Run the read-only local Stocker MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport. Stdio is the default; HTTP serves Streamable MCP at /mcp.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="HTTP bind host. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help="HTTP bind port. Defaults to 8765.",
    )
    parser.add_argument(
        "--auth-token-env",
        default=DEFAULT_AUTH_TOKEN_ENV,
        help="Environment variable containing the HTTP bearer token.",
    )
    parser.add_argument(
        "--auth-mode",
        choices=AUTH_MODES,
        default="bearer",
        help="HTTP auth mode. Use oauth for ChatGPT custom connectors.",
    )
    parser.add_argument(
        "--oauth-setup-code-env",
        default=DEFAULT_OAUTH_SETUP_CODE_ENV,
        help="Environment variable containing the local OAuth setup code.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=(
            "Additional exact Host header allowed by MCP transport security. "
            "Use the HTTPS tunnel hostname only, not 0.0.0.0."
        ),
    )
    parser.add_argument(
        "--allowed-hosts-env",
        default=DEFAULT_ALLOWED_HOSTS_ENV,
        help="Optional comma/space-separated env var of additional allowed Host headers.",
    )
    parser.add_argument("--version", action="version", version=f"stocker-mcp {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Print safe workspace and connector diagnostics and exit.")
    subparsers.add_parser("tools", help="Print registered MCP tool names and exit.")
    subparsers.add_parser("connector-info", help="Print ChatGPT connector setup info and exit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console script entry point."""

    args = _parser().parse_args(argv)
    allowed_hosts = tuple(args.allowed_host) + _allowed_hosts_from_env(args.allowed_hosts_env)
    if args.command == "doctor":
        print(
            _json(
                doctor(
                    transport=args.transport,
                    host=args.host,
                    port=args.port,
                    auth_token_env=args.auth_token_env,
                    auth_mode=args.auth_mode,
                    oauth_setup_code_env=args.oauth_setup_code_env,
                    allowed_hosts=allowed_hosts,
                )
            )
        )
        return
    if args.command == "tools":
        print(_json({"tools": list(tool_names()), "metadata": tool_metadata()}))
        return
    if args.command == "connector-info":
        print(
            _plain_json(
                connector_info(
                    host=args.host,
                    port=args.port,
                    auth_token_env=args.auth_token_env,
                    auth_mode=args.auth_mode,
                    oauth_setup_code_env=args.oauth_setup_code_env,
                    allowed_hosts=allowed_hosts,
                )
            )
        )
        return
    if args.transport == "http":
        _run_http(
            host=args.host,
            port=args.port,
            auth_token_env=args.auth_token_env,
            auth_mode=args.auth_mode,
            oauth_setup_code_env=args.oauth_setup_code_env,
            allowed_hosts=allowed_hosts,
        )
        return
    build_server(host=args.host, port=args.port, allowed_hosts=allowed_hosts).run(transport="stdio")


if __name__ == "__main__":
    main()
