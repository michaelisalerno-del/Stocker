"""Optional push alerts for conditions that need a person. Isolated from trading.

Messages carry only market/event identifiers and the application's coded reasons.
A failed delivery is recorded and retried on the next change; it never affects trading.
"""

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from stocker_execution.config import AlertSettings
from stocker_execution.saxo_auth import private_read

if TYPE_CHECKING:
    from stocker_execution.runtime import Runtime


class Alerts:
    def __init__(self, settings: AlertSettings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.url: str | None = None
        self.problem = ""
        if settings.url_file:
            try:
                url = str(private_read(settings.url_file)["url"])
                parts = urlsplit(url)
                if parts.scheme != "https" or not parts.hostname or parts.username:
                    raise ValueError("ALERT_URL_MUST_BE_HTTPS")
                self.url = url
            except (OSError, ValueError, KeyError, TypeError):
                self.problem = "ALERT_URL_FILE_INVALID"
        self.http = httpx.AsyncClient(transport=transport, timeout=10, follow_redirects=False)
        self.active: dict[str, str] = {}
        self.sent = 0
        self.last_error = ""
        self.down_since: float | None = None

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def conditions(self, runtime: "Runtime", at: float) -> dict[str, str]:
        found = {
            f"exposure:{identity}": f"Exposure exception {identity}: {reason}"
            for identity, reason in runtime.broker.management_problems.items()
        }
        if runtime.broker.fatal_error:
            found["fatal"] = f"Entries stopped: {runtime.broker.fatal_error}"
        for name in ("worker_health", "manager_health", "web_health"):
            if getattr(runtime, name) == "FAILED":
                found[name] = f"{name.replace('_', ' ')} FAILED"
        if runtime.data.connected:
            self.down_since = None
        else:
            self.down_since = self.down_since or at
            if at - self.down_since >= self.settings.stream_down_seconds:
                found["stream"] = f"Saxo stream disconnected: {runtime.data.problem}"
        oauth = runtime.data.client.oauth
        if oauth.credentials:
            expires = float(oauth.tokens.get("refresh_expires_at", 0) or 0)
            if oauth.status == "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED" or not expires:
                found["login"] = "Saxo login required: reconnect in System"
            elif expires - at < self.settings.login_warning_hours * 3600:
                when = datetime.fromtimestamp(expires, UTC).strftime("%d %b %H:%M UTC")
                found["login"] = f"Saxo login expires {when}: reconnect in System before then"
        return found

    async def check(self, runtime: "Runtime", at: float | None = None) -> None:
        current = self.conditions(runtime, time.time() if at is None else at)
        messages = [
            ("SLRNO alert", text) for k, text in current.items() if self.active.get(k) != text
        ]
        messages += [
            ("SLRNO resolved", text) for k, text in self.active.items() if k not in current
        ]
        self.active = current
        for title, text in messages:
            await self.send(title, text)

    async def send(self, title: str, text: str) -> None:
        if not self.url:
            return
        try:
            response = await self.http.post(
                self.url, content=text.encode(), headers={"Title": title, "Tags": "slrno"}
            )
            if response.status_code >= 300:
                raise ValueError(f"ALERT_HTTP_{response.status_code}")
            self.sent += 1
            self.last_error = ""
        except httpx.HTTPError:
            self.last_error = "ALERT_TRANSPORT_UNAVAILABLE"
        except ValueError as exc:
            self.last_error = str(exc)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "problem": self.problem,
            "active": sorted(self.active.values()),
            "sent": self.sent,
            "last_error": self.last_error,
        }

    async def close(self) -> None:
        await self.http.aclose()
