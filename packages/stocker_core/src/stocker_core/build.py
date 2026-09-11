"""Build identity captured once when this process imports the application."""

import os
import re
import subprocess
from pathlib import Path


def _revision() -> dict[str, object]:
    supplied = os.environ.get("STOCKER_BUILD_REVISION", "")
    if re.fullmatch(r"[0-9a-f]{40}", supplied):
        return {"commit": supplied, "dirty": None, "source": "build environment"}
    root = Path(__file__).resolve().parents[4]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None, "source": "unavailable"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=2
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                text=True,
                timeout=2,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty, "source": "checkout at process import"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}


CODE_REVISION = _revision()
