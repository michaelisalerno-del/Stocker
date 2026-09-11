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

    from stocker_core.methods import verified_q1_spec
    from stocker_dashboard.factory import build_dashboard_app
    from stocker_execution.session_hard_method import FrozenWhipsawModel

    def disconnected(*args: object, **kwargs: object) -> None:
        raise AssertionError("Server smoke must never contact a network")

    socket.socket.connect = disconnected
    assert importlib.util.find_spec("pytest") is None
    assert importlib.util.find_spec("jupyterlab") is None
    verified_q1_spec()
    model = FrozenWhipsawModel()
    assert model.columns
    with tempfile.TemporaryDirectory(prefix="stocker-smoke-state-") as directory:
        root = Path(directory)
        runs = root / "runs.yaml"
        runs.write_text("universes: []\nruns: []\n")
        broker = root / "ibkr.yaml"
        broker.write_text("{}\n")
        app = build_dashboard_app(
            runs_config_path=runs, ibkr_config_path=broker, database_path=root / "state.sqlite"
        )

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
    print("PASS: server-only imports, frozen model and offline dashboard startup/assets")


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
