"""Locked server-only installation smoke; every network connection is forbidden."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


def smoke() -> None:
    import stocker_launcher

    stocker_launcher.configure_numeric_runtime()
    stocker_launcher._ensure_monorepo_src_paths()
    import httpx

    from stocker_dashboard.app import create_dashboard_app
    from stocker_execution.first4_config import First4Config
    from stocker_execution.first4_runtime import Runtime
    from stocker_execution.first4_store import Store

    def disconnected(*args: object, **kwargs: object) -> None:
        raise AssertionError("Server smoke must never contact a network")

    socket.socket.connect = disconnected
    assert importlib.util.find_spec("pytest") is None
    assert importlib.util.find_spec("jupyterlab") is None
    assert importlib.util.find_spec("sklearn") is None
    assert importlib.util.find_spec("joblib") is None
    with tempfile.TemporaryDirectory(prefix="stocker-smoke-state-") as directory:
        root = Path(directory)
        runtime = Runtime(First4Config(), Store(root / "state.sqlite"))
        app = create_dashboard_app(runtime)

        async def check() -> None:
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
                ) as client,
            ):
                for path in ("/", "/static/dashboard.js", "/api/system", "/api/overview"):
                    response = await client.get(path)
                    assert response.status_code == 200, (path, response.text)

        asyncio.run(check())
    print("PASS: server-only imports, FIRST4 and offline dashboard startup/assets")


def main() -> None:
    if "--installed" in sys.argv:
        smoke()
        return
    with tempfile.TemporaryDirectory(prefix="stocker-server-install-") as directory:
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(Path(directory) / "venv"))
        # Avoid inheriting desktop Python paths or authenticated dashboard settings.
        for key in (
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "STOCKER_DASHBOARD_PASSWORD",
            "STOCKER_DASHBOARD_PROXY_TOKEN",
            "STOCKER_DASHBOARD_ORIGIN",
        ):
            env.pop(key, None)
        subprocess.run(
            ["uv", "sync", "--locked", "--no-default-groups", "--group", "server"],
            env=env,
            check=True,
        )
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python"
        subprocess.run([str(python), __file__, "--installed"], env=env, check=True)


if __name__ == "__main__":
    main()
