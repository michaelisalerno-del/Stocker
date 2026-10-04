"""Server-side authorization code flow. Secrets never enter public runtime state."""

import asyncio
import hmac
import json
import os
import secrets
import stat
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from stocker_execution.config import Environment, SaxoSettings

ENDPOINTS = {
    "SAXO_SIM": {
        "rest": "https://gateway.saxobank.com/sim/openapi",
        "auth": "https://sim.logonvalidation.net",
        "stream": "wss://sim-streaming.saxobank.com/sim/oapi/streaming/ws",
    },
    "SAXO_LIVE": {
        "rest": "https://gateway.saxobank.com/openapi",
        "auth": "https://live.logonvalidation.net",
        "stream": "wss://live-streaming.saxobank.com/oapi/streaming/ws",
    },
}


class SaxoError(ValueError):
    """Coded transport or authentication failure; never carries provider text."""


def private_read(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("SECRET_FILE_MUST_BE_OWNED_REGULAR_MODE_0600")
    if info.st_size > 16384:
        raise ValueError("SECRET_FILE_TOO_LARGE")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("INVALID_SECRET_FILE")
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + secrets.token_hex(8))
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, separators=(",", ":"), allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


class OAuth:
    def __init__(
        self,
        environment: Environment,
        settings: SaxoSettings,
        directory: Path,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.environment, self.settings = environment, settings
        self.urls = ENDPOINTS[environment]
        self.token_file = directory / environment / "oauth-tokens.json"
        self.http = httpx.AsyncClient(transport=transport, timeout=15, follow_redirects=False)
        self.lock = asyncio.Lock()
        self.pending: tuple[str, str, float] | None = None
        self.tokens: dict[str, Any] = {}
        self.credentials: dict[str, Any] = {}
        self.status = "NOT_CONFIGURED"
        self.failure_reason = ""
        self.generation = 0
        if settings.credentials_file:
            self.credentials = private_read(settings.credentials_file)
            if self.credentials.get("environment") != environment:
                raise ValueError("CREDENTIAL_ENVIRONMENT_MISMATCH")
            # OAuth identifies the application before the operator can discover/select
            # an account. DataService still requires a verified account before any stream.
            if not all(self.credentials.get(k) for k in ("client_id", "client_secret")):
                raise ValueError("INCOMPLETE_SAXO_CREDENTIALS")
            self.status = "RECONNECT_REQUIRED"
            if self.token_file.exists():
                self.tokens = private_read(self.token_file)
                if self.tokens.get("environment") != environment:
                    raise ValueError("TOKEN_ENVIRONMENT_MISMATCH")

    @property
    def account_key(self) -> str:
        return str(self.credentials.get("account_key", ""))

    def begin(self) -> tuple[str, str]:
        if not self.credentials:
            raise SaxoError("SAXO_CREDENTIALS_NOT_CONFIGURED")
        state, binding = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.pending = (state, binding, time.monotonic() + 600)
        return self.urls["auth"] + "/authorize?" + urlencode(
            {
                "response_type": "code",
                "client_id": self.credentials["client_id"],
                "redirect_uri": self.settings.redirect_uri,
                "state": state,
            }
        ), binding

    async def callback(self, state: str, binding: str, code: str) -> None:
        async with self.lock:
            pending, self.pending = self.pending, None
            if (
                not pending
                or time.monotonic() > pending[2]
                or not hmac.compare_digest(state, pending[0])
                or not hmac.compare_digest(binding, pending[1])
                or not code
                or len(code) > 4096
            ):
                raise SaxoError("OAUTH_STATE_INVALID_OR_EXPIRED")
            await self.exchange({"grant_type": "authorization_code", "code": code})

    async def exchange(self, form: dict[str, str]) -> None:
        reason = "OAUTH_TOKEN_NETWORK_ERROR"
        try:
            response = await self.http.post(
                self.urls["auth"] + "/token",
                data={**form, "redirect_uri": self.settings.redirect_uri},
                auth=(self.credentials["client_id"], self.credentials["client_secret"]),
            )
            # Saxo SIM also returns 201 Created for issued tokens. Both success
            # statuses still require the full payload/lifetime/storage validation below.
            if response.status_code not in {200, 201}:
                reason = f"OAUTH_TOKEN_HTTP_{response.status_code}"
                try:
                    error = response.json()
                except ValueError:
                    error = None
                code = error.get("error") if isinstance(error, dict) else None
                # Never expose arbitrary provider text, which may echo a code/secret.
                if isinstance(code, str) and code in {
                    "invalid_client",
                    "invalid_grant",
                    "invalid_request",
                    "unauthorized_client",
                    "unsupported_grant_type",
                    "server_error",
                    "temporarily_unavailable",
                }:
                    reason = "OAUTH_" + code.upper()
                raise SaxoError(reason)
            reason = "OAUTH_TOKEN_RESPONSE_INVALID"
            raw = response.json()
            if not isinstance(raw, dict):
                raise SaxoError(reason)
            reason = "OAUTH_TOKEN_RESPONSE_INCOMPLETE"
            if not raw.get("access_token") or not raw.get("refresh_token"):
                raise SaxoError(reason)
            reason = "OAUTH_LIFETIME_INVALID"
            expires = float(raw["expires_in"])
            refresh_expires = float(raw["refresh_token_expires_in"])
            if not 0 < expires <= 86400 or not 0 < refresh_expires <= 86400 * 365:
                raise SaxoError("OAUTH_LIFETIME_INVALID")
            tokens = {
                "environment": self.environment,
                "access_token": raw["access_token"],
                "refresh_token": raw["refresh_token"],
                "expires_at": time.time() + expires,
                "refresh_expires_at": time.time() + refresh_expires,
            }
            # Replace the rotating refresh token atomically before making it available.
            reason = "OAUTH_TOKEN_STORAGE_FAILED"
            await asyncio.to_thread(atomic_json, self.token_file, tokens)
            self.tokens = tokens
            self.generation += 1
            self.status = "AUTHENTICATED"
            self.failure_reason = ""
        except httpx.HTTPError:
            # The request never completed, so the refresh token cannot have rotated:
            # keep it for the next attempt inside the pre-expiry window.
            self.failure_reason = reason
            raise SaxoError(reason) from None
        except Exception:
            self.status = "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED"
            self.failure_reason = reason
            self.tokens = {}
            # Never render transport errors, request bodies or token responses.
            raise SaxoError(reason) from None

    def expire(self) -> None:
        """The server rejected the current access token: refresh on the next call."""
        if self.tokens:
            self.tokens["expires_at"] = 0
        self.status = "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED"

    async def access_token(self) -> str:
        async with self.lock:
            if not self.credentials or not self.tokens:
                raise SaxoError(self.status)
            if float(self.tokens.get("expires_at", 0)) <= time.time() + 90:
                if float(self.tokens.get("refresh_expires_at", 0)) <= time.time() + 10:
                    self.status = "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED"
                    raise SaxoError(self.status)
                await self.exchange(
                    {"grant_type": "refresh_token", "refresh_token": self.tokens["refresh_token"]}
                )
            self.status = "AUTHENTICATED"
            return str(self.tokens["access_token"])

    async def close(self) -> None:
        await self.http.aclose()
