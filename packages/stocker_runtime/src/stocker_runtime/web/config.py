"""Fail-closed configuration for the Stocker V2 read-only web process."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WebConfig(BaseModel):
    """Configuration containing presentation authority only."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    database: Path
    backup_directory: Path | None = None
    run_id: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    production: bool = True
    app_version: str = Field(default="0.1.0", min_length=1, max_length=64)
    git_commit: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_budget_ms: int = Field(default=100, ge=1, le=100)
    maximum_response_bytes: int = Field(default=512 * 1024, ge=1_024, le=512 * 1024)
    trust_proxy_headers: bool = False
    trusted_proxy_ips: list[str] = Field(default_factory=list)
    authentication_enabled: bool = False
    auth_token_env: str | None = None
    auth_cookie_name: str = "__Host-stocker_session"
    auth_cookie_secure: bool = True
    requests_per_minute: int = Field(default=120, ge=1, le=10_000)
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        if self.host in {"0.0.0.0", "::"}:
            raise ValueError("web host may not bind all interfaces")
        if not self.allowed_hosts or "*" in self.allowed_hosts:
            raise ValueError("at least one explicit trusted host is required")
        if self.authentication_enabled and not self.auth_token_env:
            raise ValueError("auth_token_env is required when authentication is enabled")
        if self.authentication_enabled and not self.auth_cookie_name.startswith("__Host-"):
            raise ValueError("authentication cookie must use the __Host- prefix")
        if self.authentication_enabled and self.production and not self.auth_cookie_secure:
            raise ValueError("production authentication requires secure cookies")
        if self.trust_proxy_headers and not self.trusted_proxy_ips:
            raise ValueError("trusted_proxy_ips are required before proxy headers are trusted")
        return self
