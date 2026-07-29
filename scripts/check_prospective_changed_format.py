"""Check Ruff formatting only for maintained prospective Python files in a change."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE_PACKAGE = "packages/stocker_prospective/"
PROSPECTIVE_TEST_PREFIXES = ("tests/test_prospective_", "tests/test_m1c_")


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _usable_base(candidate: str) -> str:
    value = candidate.strip()
    if value and set(value) != {"0"}:
        probe = _git("cat-file", "-e", f"{value}^{{commit}}", check=False)
        if probe.returncode == 0:
            return _git("merge-base", value, "HEAD").stdout.strip()
    parent = _git("rev-parse", "HEAD^", check=False)
    return parent.stdout.strip() if parent.returncode == 0 else "HEAD"


def _is_maintained_prospective_python(path: str) -> bool:
    return path.endswith(".py") and (
        path.startswith(PROSPECTIVE_PACKAGE) or path.startswith(PROSPECTIVE_TEST_PREFIXES)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="")
    arguments = parser.parse_args()
    base = _usable_base(arguments.base)
    changed = _git(
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        base,
        "HEAD",
    ).stdout.splitlines()
    files = sorted(path for path in changed if _is_maintained_prospective_python(path))
    if not files:
        print("No changed maintained prospective Python files require a format check.")
        return 0
    print("Checking Ruff formatting for:")
    print("\n".join(f"  {path}" for path in files))
    return subprocess.run(
        ("uv", "run", "ruff", "format", "--check", *files),
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
