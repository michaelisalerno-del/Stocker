"""Create an offline-verifiable SQLite/config/artifact bundle; never connect IBKR."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup(database: Path, runs: Path, broker: Path, artifacts: Path, output: Path) -> None:
    # Require a new destination. Config changes must be paused during this operation.
    output.mkdir(mode=0o700)
    configuration = output / "configuration"
    configuration.mkdir()
    sources = {runs: configuration / runs.name, broker: configuration / broker.name}
    raw = yaml.safe_load(runs.read_text())
    reference = raw.get("named_universe_snapshot") if isinstance(raw, dict) else None
    if reference:
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Bundle requires a contained relative named_universe_snapshot")
        sources[runs.parent / relative] = configuration / relative
    if len(set(sources.values())) != len(sources):
        raise ValueError("Configuration filenames collide")
    before = {source: digest(source) for source in sources}
    for source, destination in sources.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    with (
        sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as source,
        sqlite3.connect(output / "state.sqlite") as destination,
    ):
        source.backup(destination)
        assert destination.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    shutil.copytree(artifacts, output / "method_artifacts")
    if any(digest(source) != checksum for source, checksum in before.items()):
        raise RuntimeError("Configuration changed during backup; discard this incomplete bundle")
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "runs_config": str(sources[runs].relative_to(output)),
                "ibkr_config": str(sources[broker].relative_to(output)),
                "files": files,
            },
            indent=2,
        )
    )
    print("Backup complete; verify offline before relying on it")


def verify(bundle: Path) -> dict[str, object]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()) or digest(path) != checksum:
            raise ValueError("Bundle checksum mismatch")
    with sqlite3.connect(
        f"file:{(bundle / 'state.sqlite').resolve()}?mode=ro", uri=True
    ) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Database integrity failure")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--ibkr-config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backup(args.database, args.runs_config, args.ibkr_config, args.artifacts, args.output)
    verify(args.output)
