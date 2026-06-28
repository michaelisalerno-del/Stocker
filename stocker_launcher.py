"""Console-script launcher for Stocker's multi-package monorepo."""

from __future__ import annotations

import json
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
)


def _editable_project_root() -> Path | None:
    try:
        dist = distribution("stocker")
    except PackageNotFoundError:
        return None
    for file in dist.files or ():
        if str(file).endswith("direct_url.json"):
            direct_url_path = Path(dist.locate_file(file))
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

    try:
        from stocker_core.cli import app
    except ModuleNotFoundError:
        _ensure_monorepo_src_paths()
        from stocker_core.cli import app
    return app()
