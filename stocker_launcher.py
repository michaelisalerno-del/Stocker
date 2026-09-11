"""Console-script launcher for Stocker's multi-package monorepo."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import urlparse

PACKAGE_SRC_DIRS: tuple[str, ...] = (
    "packages/stocker_core/src",
    "packages/stocker_data/src",
    "packages/stocker_research/src",
    "packages/stocker_backtest/src",
    "packages/stocker_execution/src",
    "packages/stocker_mcp/src",
    "packages/stocker_dashboard/src",
)


def configure_numeric_runtime() -> None:
    """Select the frozen evidence's NumPy path before importing numerical modules.

    NumPy 2.4's X86_V4 log kernel differs by one ULP on saved candidate fixtures.
    Keep the verified x86 V2/V3 path on newer CPUs too; ARM is unchanged. Existing
    operator restrictions remain in force. This applies to execution and tests,
    rather than relaxing exact-score assertions or rewriting frozen mathematics.
    """
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return
    disabled = os.environ.get("NPY_DISABLE_CPU_FEATURES", "").replace(",", " ").split()
    disabled.extend(("X86_V4", "AVX512_ICL", "AVX512_SPR"))
    os.environ["NPY_DISABLE_CPU_FEATURES"] = ",".join(dict.fromkeys(disabled))


def _editable_project_root() -> Path | None:
    try:
        dist = distribution("stocker")
    except PackageNotFoundError:
        return None
    for file in dist.files or ():
        if str(file).endswith("direct_url.json"):
            direct_url_path = Path(str(dist.locate_file(file)))
            payload = json.loads(direct_url_path.read_text(encoding="utf-8"))
            url = str(payload.get("url", ""))
            if url.startswith("file://"):
                return Path(urlparse(url).path)
    return None


def _ensure_monorepo_src_paths() -> None:
    root = _editable_project_root() or Path.cwd()
    for relative in PACKAGE_SRC_DIRS:
        path = root / relative
        path_string = str(path)
        if path.exists() and path_string not in sys.path:
            sys.path.insert(0, path_string)


def main() -> object:
    """Run the Stocker Typer app from editable or installed environments."""

    configure_numeric_runtime()
    try:
        from stocker_core.cli import app
    except ModuleNotFoundError:
        _ensure_monorepo_src_paths()
        from stocker_core.cli import app
    return app()


def mcp_main() -> object:
    """Run the Stocker MCP server from editable or installed environments."""

    configure_numeric_runtime()
    try:
        from stocker_mcp.server import main as server_main
    except ModuleNotFoundError:
        _ensure_monorepo_src_paths()
        from stocker_mcp.server import main as server_main
    return server_main()
