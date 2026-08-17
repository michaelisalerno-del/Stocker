from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from stocker_runtime.cli import app
from stocker_runtime.ingestion import (
    CallbackFence,
    InstrumentSpec,
    Recorder,
    SubscriptionSpec,
    load_recorder_config,
)
from stocker_runtime.storage import connect_v2, initialize_database
from stocker_runtime.web import WebConfig

ROOT = Path(__file__).parents[1]
SYSTEMD = ROOT / "deploy/systemd"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def test_v2_services_have_distinct_least_privilege_filesystem_boundaries() -> None:
    recorder = _unit("stocker-v2-recorder.service")
    web = _unit("stocker-v2-web.service")
    daily = _unit("stocker-v2-backup-daily.service")
    weekly = _unit("stocker-v2-backup-weekly.service")

    assert "User=stocker-recorder" in recorder
    assert "User=stocker-web" in web
    assert "User=stocker-backup" in daily
    assert "User=stocker-backup" in weekly
    assert "ExecStart=/opt/stocker/v2-current/.venv/bin/stocker-runtime recorder run" in recorder
    assert (
        "validate-recorder /etc/stocker/recorder.json --inputs /etc/stocker/market-data.json"
        in recorder
    )
    assert "ExecStart=/opt/stocker/v2-current/.venv/bin/stocker-runtime web run" in web
    for unit in (recorder, web, daily, weekly):
        assert "/opt/stocker/current" not in unit
        assert "/opt/stocker/v2-current" in unit
    assert "--tier daily" in daily
    assert "--tier weekly" in weekly
    assert "--working-directory /var/cache/stocker-v2-backup-work" in daily
    assert "CacheDirectory=stocker-v2-backup-work" in daily
    assert "TimeoutStartSec=31min" in daily
    assert "TimeoutStartSec=31min" in weekly
    assert "ExecCondition=" not in daily
    assert "ExecCondition=" not in weekly

    assert "ReadWritePaths=/var/lib/stocker/v2" in recorder
    assert "ReadOnlyPaths=/var/lib/stocker/backups-v2" in recorder
    assert "ReadWritePaths=/var/lib/stocker/backups-v2" not in recorder
    assert "ReadOnlyPaths=/var/lib/stocker/v2" in web
    assert "ReadOnlyPaths=/var/lib/stocker/backups-v2" in web
    assert "ReadWritePaths=/var/lib/stocker/v2" not in web.splitlines()
    assert "ReadOnlyPaths=/var/lib/stocker/v2" in daily
    assert "ReadWritePaths=/var/lib/stocker/backups-v2" in daily
    assert "ReadWritePaths=/var/lib/stocker/v2" not in daily.splitlines()

    assert "IPAddressDeny=any" in recorder
    assert "IPAddressAllow=localhost" in recorder
    assert "IPAddressDeny=any" in web
    assert "IPAddressAllow=localhost" in web
    assert "IPAddressDeny=any" in daily
    assert "Restart=on-failure" in recorder
    assert "KillSignal=SIGTERM" in recorder
    assert "Restart=on-failure" in web
    assert "KillSignal=SIGTERM" in web


def test_daily_and_weekly_timers_are_bounded_and_not_implicitly_enabled() -> None:
    daily = _unit("stocker-v2-backup-daily.timer")
    weekly = _unit("stocker-v2-backup-weekly.timer")

    assert "OnCalendar=*-*-* 23:45:00 UTC" in daily
    assert "OnCalendar=Sun *-*-* 22:45:00 UTC" in weekly
    assert "Persistent=true" in daily
    assert "Persistent=true" in weekly
    assert "RandomizedDelaySec=" in daily
    assert "RandomizedDelaySec=" in weekly
    assert not (SYSTEMD / "stocker-backup.service").exists()
    assert not (SYSTEMD / "stocker-backup.timer").exists()


def test_cutover_release_contains_only_v2_stocker_application_services() -> None:
    for retired in (
        "stocker-recorder.service",
        "stocker-web.service",
        "stocker-backup.service",
        "stocker-backup.timer",
    ):
        assert not (SYSTEMD / retired).exists()
    assert not (ROOT / "deploy/stocker.env.example").exists()
    assert not (ROOT / "deploy/scripts/prepare-web-sqlite-boundary.py").exists()
    assert (ROOT / "docs/operations/stocker-v2-cutover.md").is_file()
    assert not (ROOT / "docs/operations/prospective-server-runbook.md").exists()

    active_surface = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "deploy").rglob("*") if path.is_file()
    ).lower()
    assert "stocker-prospective" not in active_surface
    assert "configs/prospective" not in active_surface


def test_cutover_builds_official_client_dependencies_offline_before_release_publish() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    regular_wheel = runbook.index('sudo test -f "$STOCKER_PROTOBUF_WHEEL"')
    exact_wheel = runbook.index("protobuf-5.29.5-*.whl")
    manifest_binding = runbook.index("NR == 1 && length($1) == 64")
    protobuf_hash = runbook.index('sha256sum --check "$STOCKER_PROTOBUF_WHEEL_SHA256"')
    regular_ibapi_wheel = runbook.index('sudo test -f "$STOCKER_IBAPI_WHEEL"')
    exact_ibapi_wheel = runbook.index(
        'test "${STOCKER_IBAPI_WHEEL##*/}" = ibapi-10.49.1-py3-none-any.whl'
    )
    ibapi_manifest_binding = runbook.index('sudo awk -v expected="$STOCKER_IBAPI_WHEEL"')
    ibapi_hash = runbook.index('sha256sum --check "$STOCKER_IBAPI_WHEEL_SHA256"')
    preinstall_gate = runbook.index('"$STOCKER_RELEASE_ARTIFACT_VERIFIER" preinstall')
    protobuf_install = runbook.index('--offline --no-deps --reinstall "$STOCKER_PROTOBUF_WHEEL"')
    ibapi_install = runbook.index('--offline --no-deps --reinstall "$STOCKER_IBAPI_WHEEL"')
    postinstall_gate = runbook.index('"$STOCKER_RELEASE_ARTIFACT_VERIFIER" postinstall')
    metadata_gate = runbook.index('requires("ibapi")')
    concrete_import_gate = runbook.index("from ibapi.client import EClient")
    provenance_gate = runbook.index('ibkr-api verify --provenance "$STOCKER_IBAPI_PROVENANCE"')
    final_postinstall_gate = runbook.rindex('"$STOCKER_RELEASE_ARTIFACT_VERIFIER" postinstall')
    publish = runbook.index('sudo ln -s "$STOCKER_V2_RELEASE" /opt/stocker/v2-current')

    assert "--offline" in runbook[protobuf_hash:ibapi_install]
    assert "--no-deps" in runbook[protobuf_hash:ibapi_install]
    assert regular_wheel < exact_wheel < manifest_binding < protobuf_hash
    assert protobuf_hash < regular_ibapi_wheel < exact_ibapi_wheel
    assert exact_ibapi_wheel < ibapi_manifest_binding < ibapi_hash
    assert ibapi_hash < preinstall_gate < protobuf_install < ibapi_install
    assert ibapi_install < postinstall_gate < metadata_gate
    assert metadata_gate < concrete_import_gate < provenance_gate < final_postinstall_gate < publish
    assert postinstall_gate != final_postinstall_gate
    assert runbook.count('"$STOCKER_RELEASE_ARTIFACT_VERIFIER" postinstall') == 2
    assert (
        "sudo env PYTHONDONTWRITEBYTECODE=1 \"$STOCKER_V2_RELEASE/.venv/bin/python\" -B - <<'PY'"
    ) in runbook[postinstall_gate:provenance_gate]
    assert (
        "sudo env PYTHONDONTWRITEBYTECODE=1 "
        '"$STOCKER_V2_RELEASE/.venv/bin/python" -B '
        '"$STOCKER_V2_RELEASE/.venv/bin/stocker-runtime" '
        'ibkr-api verify --provenance "$STOCKER_IBAPI_PROVENANCE"'
    ) in runbook[concrete_import_gate:final_postinstall_gate]
    assert "len(declared_dependencies) != 1" in runbook[metadata_gate:publish]
    assert r'r"protobuf[ \t]*==[ \t]*5\.29\.5"' in runbook[metadata_gate:publish]
    assert "flags=re.ASCII" in runbook[metadata_gate:publish]
    assert 'version("ibapi") != "10.49.1"' in runbook[metadata_gate:publish]
    assert 'version("protobuf") != "5.29.5"' in runbook[metadata_gate:publish]
    assert '--reinstall "$STOCKER_IBAPI_SOURCE"' not in runbook
    assert 'sudo test ! -L "$STOCKER_IBAPI_PROVENANCE"' not in runbook
    assert "export STOCKER_TRUSTED_PYTHON=/usr/bin/python3" in runbook
    assert "export STOCKER_UV_BIN=/usr/local/bin/uv" in runbook
    assert "command -v uv" not in runbook
    assert "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb" in runbook
    assert "uv 0.11.32 (x86_64-unknown-linux-gnu)" in runbook
    assert '"$STOCKER_TRUSTED_PYTHON" -I -' in runbook
    assert runbook.count('"$STOCKER_TRUSTED_PYTHON" -I "$STOCKER_RELEASE_ARTIFACT_VERIFIER"') == 3
    verifier = ROOT / "deploy/scripts/verify_v2_release_artifacts.py"
    assert verifier.is_file()
    verifier_sha256 = hashlib.sha256(verifier.read_bytes()).hexdigest()
    assert f"export STOCKER_RELEASE_ARTIFACT_VERIFIER_SHA256={verifier_sha256}" in runbook
    for protected_argument in (
        '--ibapi-wheel "$STOCKER_IBAPI_WHEEL"',
        '--ibapi-manifest "$STOCKER_IBAPI_WHEEL_SHA256"',
        '--protobuf-wheel "$STOCKER_PROTOBUF_WHEEL"',
        '--protobuf-manifest "$STOCKER_PROTOBUF_WHEEL_SHA256"',
        '--uv-bin "$STOCKER_UV_BIN"',
        '--uv-manifest "$STOCKER_UV_SHA256"',
        '--official-source-root "$STOCKER_IBAPI_SOURCE"',
        '--release-root "$STOCKER_V2_RELEASE"',
        '--verifier-path "$STOCKER_RELEASE_ARTIFACT_VERIFIER"',
        '--verifier-sha256 "$STOCKER_RELEASE_ARTIFACT_VERIFIER_SHA256"',
        '--venv "$STOCKER_V2_RELEASE/.venv"',
    ):
        assert runbook.count(protected_argument) == 3
    assert "literal target must be\n`provenance/10.49.1.json`" in runbook
    assert "Group-write is admitted only\nfor group ID 0" in runbook
    assert "must not come from a package registry" in runbook
    assert "Do not vendor the derived wheel" in runbook


def test_cutover_dependency_failures_cannot_reach_release_pointer(tmp_path: Path) -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    block_start = runbook.index("```bash", runbook.index("Prepare the separate V2 pointer"))
    block_start += len("```bash\n")
    block_end = runbook.index("getent group stocker-readers", block_start)
    block = runbook[block_start:block_end]

    release = tmp_path / "reviewed-release"
    source = tmp_path / "reviewed-ibapi-source"
    protobuf_wheel = tmp_path / "protobuf-5.29.5-reviewed.whl"
    protobuf_manifest = tmp_path / "protobuf-5.29.5-reviewed.sha256"
    ibapi_wheel = tmp_path / "ibapi-10.49.1-py3-none-any.whl"
    ibapi_manifest = tmp_path / "ibapi-10.49.1-reviewed.sha256"
    uv_bin = tmp_path / "bin/uv"
    uv_manifest = tmp_path / "uv-0.11.32.sha256"
    provenance = tmp_path / "active-provenance.json"
    replacements = {
        "export STOCKER_V2_RELEASE=/opt/stocker/releases/REPLACE_WITH_REVIEWED_COMMIT": (
            f'export STOCKER_V2_RELEASE="{release}"'
        ),
        (
            "export STOCKER_IBAPI_SOURCE="
            "/var/lib/stocker/ibkr-api/install/IBJts/source/pythonclient"
        ): f'export STOCKER_IBAPI_SOURCE="{source}"',
        (
            "export STOCKER_PROTOBUF_WHEEL=/var/lib/stocker/ibkr-api/install/"
            "REPLACE_WITH_REVIEWED_PROTOBUF_5_29_5_WHEEL.whl"
        ): f'export STOCKER_PROTOBUF_WHEEL="{protobuf_wheel}"',
        (
            "export STOCKER_PROTOBUF_WHEEL_SHA256=/var/lib/stocker/ibkr-api/install/"
            "protobuf-5.29.5-wheel.sha256"
        ): f'export STOCKER_PROTOBUF_WHEEL_SHA256="{protobuf_manifest}"',
        (
            "export STOCKER_IBAPI_WHEEL=/var/lib/stocker/ibkr-api/install/"
            "ibapi-10.49.1-py3-none-any.whl"
        ): f'export STOCKER_IBAPI_WHEEL="{ibapi_wheel}"',
        (
            "export STOCKER_IBAPI_WHEEL_SHA256=/var/lib/stocker/ibkr-api/install/"
            "ibapi-10.49.1-wheel.sha256"
        ): f'export STOCKER_IBAPI_WHEEL_SHA256="{ibapi_manifest}"',
        (
            "export STOCKER_IBAPI_PROVENANCE=/var/lib/stocker/ibkr-api/active-provenance.json"
        ): f'export STOCKER_IBAPI_PROVENANCE="{provenance}"',
        "export STOCKER_UV_BIN=/usr/local/bin/uv": f'export STOCKER_UV_BIN="{uv_bin}"',
        (
            "export STOCKER_UV_SHA256=/var/lib/stocker/ibkr-api/install/uv-0.11.32.sha256"
        ): f'export STOCKER_UV_SHA256="{uv_manifest}"',
    }
    for original, replacement in replacements.items():
        assert original in block
        block = block.replace(original, replacement, 1)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("getfacl", "readlink", "setfacl", "uv"):
        executable = fake_bin / command
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text(
        """#!/usr/bin/env bash
set -u
printf '%s\\n' "$*" >> "$TRACE"
if [[ -n "${FAIL_MATCH:-}" && "$*" == *"$FAIL_MATCH"* ]]; then
  exit 23
fi
if [[ "${FAIL_FINAL_POSTINSTALL:-}" == 1 ]] &&
  [[ "$*" == *"verify_v2_release_artifacts.py postinstall"* ]]; then
  postinstall_count=0
  if [[ -f "$POSTINSTALL_COUNT" ]]; then
    IFS= read -r postinstall_count < "$POSTINSTALL_COUNT"
  fi
  postinstall_count=$((postinstall_count + 1))
  printf '%s\n' "$postinstall_count" > "$POSTINSTALL_COUNT"
  if [[ "$postinstall_count" -eq 2 ]]; then
    exit 24
  fi
fi
if [[ "$1" == "ln" && "$2" == "-s" ]]; then
  : > "$PUBLISHED"
fi
if [[ "$1" == */uv && "${2:-}" == "--version" ]]; then
  printf '%s\n' 'uv 0.11.32 (x86_64-unknown-linux-gnu)'
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_sudo.chmod(0o755)

    trace = tmp_path / "trace"
    published = tmp_path / "published"
    postinstall_count = tmp_path / "postinstall-count"
    environment = os.environ.copy()
    environment.update(
        PATH=f"{fake_bin}:{environment['PATH']}",
        POSTINSTALL_COUNT=str(postinstall_count),
        PUBLISHED=str(published),
        TRACE=str(trace),
    )
    failures = (
        "sha256sum --check",
        f"sha256sum --check {ibapi_manifest}",
        f"sha256sum --check {uv_manifest}",
        "/usr/bin/python3 -I -",
        "verify_v2_release_artifacts.py preinstall",
        (
            f"pip install --python {release}/.venv/bin/python --offline --no-deps "
            f"--reinstall {protobuf_wheel}"
        ),
        (
            f"pip install --python {release}/.venv/bin/python --offline --no-deps "
            f"--reinstall {ibapi_wheel}"
        ),
        "verify_v2_release_artifacts.py postinstall",
        f"{uv_bin} --version",
        f"{release}/.venv/bin/python -B -",
        f"stocker-runtime ibkr-api verify --provenance {provenance}",
    )
    for failure in failures:
        published.unlink(missing_ok=True)
        trace.write_text("", encoding="utf-8")
        failed_environment = environment | {"FAIL_MATCH": failure}
        completed = subprocess.run(
            ["bash", "-c", block],
            check=False,
            capture_output=True,
            env=failed_environment,
            text=True,
            timeout=5,
        )

        assert completed.returncode == 78, failure
        assert not published.exists(), failure
        assert "ln -s" not in trace.read_text(encoding="utf-8"), failure

    published.unlink(missing_ok=True)
    postinstall_count.unlink(missing_ok=True)
    trace.write_text("", encoding="utf-8")
    completed = subprocess.run(
        ["bash", "-c", block],
        check=False,
        capture_output=True,
        env=environment | {"FAIL_FINAL_POSTINSTALL": "1", "FAIL_MATCH": ""},
        text=True,
        timeout=5,
    )
    final_trace = trace.read_text(encoding="utf-8")
    assert completed.returncode == 78
    assert final_trace.count("verify_v2_release_artifacts.py postinstall") == 2
    assert final_trace.count("verify_v2_release_artifacts.py preinstall") == 1
    assert not published.exists()
    assert "ln -s" not in final_trace

    published.unlink(missing_ok=True)
    continued = tmp_path / "continued-after-accepted-nonzero"
    success_environment = environment | {
        "CONTINUED": str(continued),
        "FAIL_MATCH": "",
    }
    successful_block = f"""set +e
set +u
set +o pipefail
{block}
case "$-" in
  *e*|*u*) exit 91 ;;
esac
if shopt -qo pipefail; then
  exit 92
fi
false
: > "$CONTINUED"
"""
    completed = subprocess.run(
        ["bash", "-c", successful_block],
        check=False,
        capture_output=True,
        env=success_environment,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert published.exists()
    assert continued.exists()


def test_cutover_runtime_dependency_gate_accepts_only_exact_semantic_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    marker = (
        "sudo env PYTHONDONTWRITEBYTECODE=1 \"$STOCKER_V2_RELEASE/.venv/bin/python\" -B - <<'PY'\n"
    )
    gate_start = runbook.index(marker) + len(marker)
    gate_end = runbook.index("\nPY\n", gate_start)
    gate = runbook[gate_start:gate_end]

    dependency_result: dict[str, list[str] | None] = {"value": ["protobuf ==5.29.5"]}
    metadata = ModuleType("importlib.metadata")
    metadata.requires = lambda name: dependency_result["value"]
    metadata.version = lambda name: {
        "ibapi": "10.49.1",
        "protobuf": "5.29.5",
    }[name]
    google = ModuleType("google")
    google.__path__ = []
    protobuf = ModuleType("google.protobuf")
    google.protobuf = protobuf
    ibapi = ModuleType("ibapi")
    ibapi.__path__ = []
    client = ModuleType("ibapi.client")
    client.EClient = type("EClient", (), {})
    ibapi.client = client
    for name, module in (
        ("importlib.metadata", metadata),
        ("google", google),
        ("google.protobuf", protobuf),
        ("ibapi", ibapi),
        ("ibapi.client", client),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    for accepted in (
        ["protobuf ==5.29.5"],
        ["protobuf==5.29.5"],
        ["protobuf\t==\t5.29.5"],
    ):
        dependency_result["value"] = accepted
        exec(compile(gate, "<cutover-runtime-dependency-gate>", "exec"), {})

    for rejected in (
        None,
        [],
        ["protobuf>=5.29.5"],
        ["Protobuf==5.29.5"],
        ["protobuf ==5.29.5", "other==1"],
    ):
        dependency_result["value"] = rejected
        with pytest.raises(SystemExit, match="ibapi declared dependency mismatch"):
            exec(compile(gate, "<cutover-runtime-dependency-gate>", "exec"), {})


def test_cutover_import_and_console_probes_leave_no_generated_bytecode(
    tmp_path: Path,
) -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    marker = (
        "sudo env PYTHONDONTWRITEBYTECODE=1 \"$STOCKER_V2_RELEASE/.venv/bin/python\" -B - <<'PY'\n"
    )
    gate_start = runbook.index(marker) + len(marker)
    gate_end = runbook.index("\nPY\n", gate_start)
    gate = runbook[gate_start:gate_end]

    venv = tmp_path / "probe-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    site_packages = next((venv / "lib").glob("python*/site-packages"))
    for package in ("google", "google/protobuf", "ibapi", "stocker_runtime"):
        package_root = site_packages / package
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")
    (site_packages / "ibapi/client.py").write_text(
        "class EClient:\n    pass\n",
        encoding="utf-8",
    )
    (site_packages / "stocker_runtime/cli_probe.py").write_text(
        """import os
import sys
from pathlib import Path

import google.protobuf
from ibapi.client import EClient

def main():
    if not isinstance(EClient, type):
        raise SystemExit(31)
    if sys.argv[1:] != ["ibkr-api", "verify", "--provenance", os.environ["PROVENANCE"]]:
        raise SystemExit(32)
    Path(os.environ["PROBE_MARKER"]).write_text("verified", encoding="utf-8")
""",
        encoding="utf-8",
    )
    for distribution, metadata in (
        (
            "ibapi-10.49.1.dist-info",
            "Name: ibapi\nVersion: 10.49.1\nRequires-Dist: protobuf ==5.29.5\n",
        ),
        (
            "protobuf-5.29.5.dist-info",
            "Name: protobuf\nVersion: 5.29.5\n",
        ),
    ):
        dist_info = site_packages / distribution
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\n{metadata}\n",
            encoding="utf-8",
        )
    console = venv / "bin/stocker-runtime"
    console.write_text(
        "from stocker_runtime.cli_probe import main\nmain()\n",
        encoding="utf-8",
    )
    console.chmod(0o755)
    provenance = tmp_path / "active-provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    probe_marker = tmp_path / "probe-complete"
    environment = os.environ | {
        "PROBE_MARKER": str(probe_marker),
        "PROVENANCE": str(provenance),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    python = venv / "bin/python"

    subprocess.run(
        [str(python), "-B", "-"],
        input=gate,
        check=True,
        capture_output=True,
        env=environment,
        text=True,
        timeout=5,
    )
    subprocess.run(
        [
            str(python),
            "-B",
            str(console),
            "ibkr-api",
            "verify",
            "--provenance",
            str(provenance),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
        timeout=5,
    )

    assert probe_marker.read_text(encoding="utf-8") == "verified"
    generated = [
        path
        for path in venv.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    ]
    assert generated == []


def test_cutover_verifier_bootstrap_rejects_trust_boundary_mutations(
    tmp_path: Path,
) -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    bootstrap_command = runbook.index('"$STOCKER_TRUSTED_PYTHON" -I -')
    bootstrap_start = runbook.index("import hashlib", bootstrap_command)
    bootstrap_end = runbook.index(
        '\nPY\nsudo "$STOCKER_TRUSTED_PYTHON" -I "$STOCKER_RELEASE_ARTIFACT_VERIFIER"',
        bootstrap_start,
    )
    bootstrap = runbook[bootstrap_start:bootstrap_end]

    release = tmp_path / "releases/reviewed"
    verifier = release / "deploy/scripts/verify_v2_release_artifacts.py"
    verifier.parent.mkdir(parents=True)
    reviewed_payload = b"# reviewed verifier\n"
    verifier.write_bytes(reviewed_payload)
    verifier.chmod(0o644)
    expected_hash = hashlib.sha256(reviewed_payload).hexdigest()
    reached_verifier = tmp_path / "reached-verifier"
    compatibility_prefix = """
import os
import pathlib
import stat

_real_lstat = pathlib.Path.lstat
def _controlled_test_lstat(path):
    metadata = _real_lstat(path)
    values = list(metadata)
    if stat.S_ISDIR(metadata.st_mode):
        values[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
    values[4] = 1 if os.environ.get("WRONG_OWNER") == str(path) else 0
    values[5] = 0
    return os.stat_result(values)
pathlib.Path.lstat = _controlled_test_lstat
os.listxattr = lambda path, *, follow_symlinks=False: []
"""
    program = (
        compatibility_prefix
        + bootstrap
        + '\nPath(os.environ["REACHED_VERIFIER"]).write_text("reached", encoding="utf-8")\n'
    )

    def run_bootstrap(
        *,
        verifier_argument: Path = verifier,
        reviewed_hash: str = expected_hash,
        wrong_owner: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        reached_verifier.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment["REACHED_VERIFIER"] = str(reached_verifier)
        environment["WRONG_OWNER"] = "" if wrong_owner is None else str(wrong_owner)
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-",
                str(verifier_argument),
                str(release),
                reviewed_hash,
            ],
            input=program,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=5,
        )

    assert run_bootstrap().returncode == 0
    assert reached_verifier.exists()

    failures: list[subprocess.CompletedProcess[str]] = [
        run_bootstrap(reviewed_hash="0" * 64),
        run_bootstrap(wrong_owner=verifier),
    ]
    verifier.chmod(0o666)
    failures.append(run_bootstrap())
    verifier.chmod(0o644)
    verifier.write_bytes(b"# verifier changed after review\n")
    failures.append(run_bootstrap())
    verifier.write_bytes(reviewed_payload)
    alternate = verifier.with_name("alternate.py")
    alternate.write_bytes(reviewed_payload)
    failures.append(run_bootstrap(verifier_argument=alternate))

    assert all(completed.returncode != 0 for completed in failures)
    assert not reached_verifier.exists()


def test_v2_deployment_contains_no_legacy_vendor_transfer_or_execution_fields() -> None:
    deployment_files = [
        ROOT / "deploy/stocker-recorder.env.example",
        ROOT / "deploy/stocker-v2-web.env.example",
        ROOT / "deploy/stocker-backup.env.example",
        ROOT / "configs/runtime/recorder.example.json",
        ROOT / "configs/runtime/market-data.example.json",
        ROOT / "configs/runtime/web.example.json",
        SYSTEMD / "stocker-v2-recorder.service",
        SYSTEMD / "stocker-v2-web.service",
        SYSTEMD / "stocker-v2-backup-daily.service",
        SYSTEMD / "stocker-v2-backup-weekly.service",
        SYSTEMD / "stocker-v2-backup-daily.timer",
        SYSTEMD / "stocker-v2-backup-weekly.timer",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in deployment_files).lower()

    for forbidden in (
        "eodhd",
        "source_transfer",
        "source-transfer",
        "parallel",
        "parquet",
        "report.zip",
        "place_order",
        "order service",
        "paper_enabled",
        "live_enabled",
    ):
        assert forbidden not in joined

    v2_web_unit = (SYSTEMD / "stocker-v2-web.service").read_text(encoding="utf-8")
    assert "EnvironmentFile=/etc/stocker/stocker-v2-web.env" in v2_web_unit
    assert "EnvironmentFile=/etc/stocker/stocker-web.env" not in v2_web_unit
    assert not (ROOT / "deploy/stocker-web.env.example").exists()
    assert "ibkr_read_only" not in joined
    assert "credential" not in joined


def test_v2_deployment_examples_validate_with_only_record_shadow_authority() -> None:
    recorder = load_recorder_config(ROOT / "configs/runtime/recorder.example.json")
    web = WebConfig.model_validate_json(
        (ROOT / "configs/runtime/web.example.json").read_text(encoding="utf-8")
    )
    market_data = json.loads(
        (ROOT / "configs/runtime/market-data.example.json").read_text(encoding="utf-8")
    )

    assert recorder.mode == "prospective_record"
    assert recorder.host == "127.0.0.1"
    assert recorder.read_only is True
    assert recorder.external_read_only_verified is True
    assert str(recorder.database) == "/var/lib/stocker/v2/stocker-v2.sqlite3"
    assert web.host == "127.0.0.1"
    assert web.query_budget_ms == 300
    assert str(web.database) == "/var/lib/stocker/v2/stocker-v2.sqlite3"
    assert str(web.backup_directory) == "/var/lib/stocker/backups-v2"
    assert len(market_data["instruments"]) == 1
    assert len(market_data["subscriptions"]) == 2
    assert all("EXAMPLE_ONLY" in item["name"] for item in market_data["subscriptions"])
    assert any(not item["optional"] for item in market_data["subscriptions"])


def test_gateway_units_are_conspicuously_market_data_only_without_capability_change() -> None:
    related = tuple(SYSTEMD.glob("stocker-ibgateway*.service")) + tuple(
        SYSTEMD.glob("stocker-ibgateway*.socket")
    )
    assert related
    for path in related:
        text = path.read_text(encoding="utf-8")
        assert "market-data-only" in text.lower(), path.name
    gateway = _unit("stocker-ibgateway.service")
    assert "ExecStart=/opt/ibgateway/current/ibgateway" in gateway
    assert "Restart=always" in gateway
    assert "EnvironmentFile=" not in gateway


def test_sqlite_boundary_preparation_targets_only_v2_and_backup_paths() -> None:
    script_path = ROOT / "deploy/scripts/prepare-v2-sqlite-boundary.py"
    source = script_path.read_text(encoding="utf-8")

    assert not (ROOT / "deploy/scripts/prepare-web-sqlite-boundary.py").exists()
    assert 'DATABASE_DIRECTORY_NAME = "v2"' in source
    assert 'BACKUP_DIRECTORY_NAME = "backups-v2"' in source
    assert 'DATABASE_NAME = "stocker-v2.sqlite3"' in source
    assert 'RECORDER_USER = "stocker-recorder"' in source
    assert 'WEB_USER = "stocker-web"' in source
    assert 'BACKUP_USER = "stocker-backup"' in source
    assert 'READER_GROUP = "stocker-readers"' in source
    assert "prospective.sqlite3" not in source
    assert "bundles" not in source


def test_cutover_precreates_reader_boundary_before_import_and_verifies_afterward() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")

    assert "groupadd --system stocker-readers" in runbook
    assert runbook.count("--gid stocker-readers") == 3
    assert "install -d -o root -g stocker-readers -m 0750 /var/lib/stocker" in runbook
    assert (
        "install -d -o stocker-recorder -g stocker-readers -m 2750 /var/lib/stocker/v2" in runbook
    )
    assert (
        "install -d -o stocker-backup -g stocker-readers -m 2750 "
        "/var/lib/stocker/backups-v2" in runbook
    )
    install_verifier = "sudo install -o root -g root -m 0755"
    assert install_verifier in runbook
    assert "/opt/stocker/v2-current/deploy/scripts/prepare-v2-sqlite-boundary.py" in runbook
    assert "/usr/local/libexec/stocker-prepare-v2-sqlite-boundary" in runbook
    importer = runbook.index("stocker-runtime legacy import")
    verifier = runbook.index("sudo /usr/local/libexec/stocker-prepare-v2-sqlite-boundary")
    assert runbook.index(install_verifier) < importer
    assert importer < verifier
    assert "command -v setfacl" in runbook
    assert "setfacl -m u:stocker-recorder:r--" in runbook
    assert "snapshot-access-control-before.txt" in runbook
    assert "rollback-access-control-before.txt" in runbook
    rollback_acl_capture = runbook.index('rollback_acl_before="$(sudo getfacl -p')
    rollback_chmod = runbook.index('sudo chmod 0400 "$STOCKER_V1_ROLLBACK_DB"')
    assert rollback_acl_capture < rollback_chmod
    assert (
        'rollback_acl_before="$(sudo getfacl -p "$STOCKER_V1_ROLLBACK_DB")" || exit 78' in runbook
    )
    assert 'test -n "$rollback_acl_before" || exit 78' in runbook
    assert runbook.count("set -o pipefail") >= 3
    assert "chgrp stocker-readers /var/lib/stocker/backups" not in runbook
    assert "chown stocker:stocker-readers" not in runbook
    revoke = runbook.index("revoke_v1_snapshot_access")
    grant = runbook.index("setfacl -m u:stocker-recorder:r--")
    assert revoke < importer
    assert revoke < grant < importer
    assert "fail_after_snapshot_revoke" in runbook[revoke:importer]
    assert "getfacl -cp" in runbook[revoke:importer]
    assert runbook.index("setfacl -x u:stocker-recorder") < verifier
    assert runbook.index("sudo -u stocker-web test ! -r") < importer
    assert runbook.rindex("sudo -u stocker-web test ! -r") < verifier
    assert runbook.index("sudo -u stocker-backup test ! -r") < importer
    assert runbook.rindex("sudo -u stocker-backup test ! -r") < verifier


def test_cutover_runbook_preserves_one_writer_and_two_distinct_rollback_paths() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    lowered = runbook.lower()

    for required in (
        "market-closed",
        "owner approval",
        "read-only recovery set",
        "quick_check",
        "foreign_key_check",
        "source_row_count",
        "imported_row_count",
        "omitted_row_count",
        "query-only",
        "first callback",
        "zero order capability",
        "observation window",
        "checked v2 backup",
        "there is no dual write",
        "before first callback",
        "after first callback",
        "new v1 run and recorder generation",
        "explicit gap",
        "never reverse-import",
        "owner closes the rollback window",
        "owner-only rollback-window closure",
        "installed v1 units",
        "/opt/stocker/v2-current",
        "/opt/stocker/current",
        "prospective_record",
        "michael owns a seven-day rollback",
        "accept-quiescent-unclean-generations",
        "canonical full-row digest",
    ):
        assert required in lowered
    assert lowered.index("start the v2 web") < lowered.index("start the v2 recorder")
    assert lowered.index("declare v2 admission operational") < lowered.index(
        "owner-only rollback-window closure"
    )
    assert "never remove the v1 units" in lowered
    assert "never repoint\n`/opt/stocker/current`, before the owner" in lowered
    assert "id -u stocker >/dev/null 2>&1" in lowered
    assert (
        "stat -c '%u' \\\n  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3"
        in lowered
    )
    assert (
        "sudo -u stocker sqlite3 \\\n"
        "  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3" in lowered
    )
    assert "disable --now stocker-backup.timer" in lowered
    assert "stocker-recorder-session-readiness.timer" in lowered
    assert "stocker-backup-daily.timer" not in lowered
    assert "stocker-backup-weekly.timer" not in lowered
    assert "paper trading" not in lowered
    assert "live trading" not in lowered


def test_attended_cutover_keeps_import_rollback_and_retirement_paths_distinct() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")

    import_source = "/var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3"
    rollback_database = "/var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3"
    retirement_database = "/var/lib/stocker/prospective/prospective.sqlite3"
    exact_delete = (
        "sudo rm -- /var/lib/stocker/prospective/prospective.sqlite3 \\\n"
        "    /var/lib/stocker/prospective/prospective.sqlite3-wal \\\n"
        "    /var/lib/stocker/prospective/prospective.sqlite3-shm"
    )

    assert f"--source {import_source}" in runbook
    assert f"sudo -u stocker sqlite3 \\\n  {rollback_database}" in runbook
    assert exact_delete in runbook
    assert runbook.count(exact_delete) == 1
    fence_start = runbook.index("install_v1_runtime_start_fence()")
    guarded_delete = runbook.index("retire_authorised_v1_candidate()")
    deletion = runbook.index(exact_delete)
    guard_call = runbook.index("retire_authorised_v1_candidate\n")
    assert fence_start < guarded_delete < deletion < guard_call
    deletion_guard = runbook[fence_start:deletion]
    assert "install_v1_runtime_start_fence || return 78" in deletion_guard
    assert "RefuseManualStart=yes" in deletion_guard
    assert "ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL" in deletion_guard
    verify_command = 'LC_ALL=C systemd-analyze verify "$unit" >/dev/null 2>&1 || return 78'
    condition_command = 'LC_ALL=C systemd-analyze condition --unit="$unit"'
    assert verify_command in deletion_guard
    assert condition_command in deletion_guard
    assert "condition_output=" in deletion_guard
    assert '*"ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL failed."*) ;;' in deletion_guard
    assert 'test "$condition_status" -eq 1' in deletion_guard
    assert "assert_v1_fence_condition_false" not in deletion_guard
    assert 'ConditionPathExists=/"' not in deletion_guard
    assert "--property=FragmentPath --value" in deletion_guard
    assert "--property=RefuseManualStart --value" in deletion_guard
    assert "--property=DropInPaths --value" in deletion_guard
    assert "--property=ActiveState --value" in deletion_guard
    assert "--property=Conditions --value" not in deletion_guard
    assert (
        deletion_guard.index(verify_command)
        < deletion_guard.index(condition_command)
        < deletion_guard.index("--property=RefuseManualStart --value")
    )
    assert 'test "$fragment_path" = "/etc/systemd/system/$unit"' in deletion_guard
    assert 'test "$refuse_manual" = yes' in deletion_guard
    assert 'if sudo systemctl start "$unit"' in deletion_guard
    assert deletion_guard.count('test "$active_state" = inactive') >= 2
    assert "systemctl mask" not in deletion_guard
    assert "lsof_status" in runbook[guarded_delete:deletion]
    assert 'test "$lsof_status" -ne 1' in runbook[guarded_delete:deletion]
    assert "without a glob" in runbook
    assert "no-handle/dependency" in runbook
    assert "does not authorise deletion of the import snapshot or rollback database" in runbook
    preservation = runbook.index("pre-deletion\npreservation manifest")
    compressed_copies = runbook.index("checked\ncompressed recovery copies")
    recorder_grant = runbook.index("setfacl -m u:stocker-recorder:r--")
    importer = runbook.index("--source " + import_source)
    assert runbook.index("Before deleting the retirement candidate") < preservation
    assert preservation < deletion < compressed_copies < recorder_grant < importer
    assert "sudo gzip --test" in runbook[compressed_copies:importer]
    assert "sudo cmp --silent" in runbook[compressed_copies:importer]
    assert "restore-integrity.txt" in runbook[compressed_copies:importer]
    assert retirement_database not in runbook[runbook.index("--source ") :]
    control_archive = runbook[runbook.index("sudo tar --create --gzip") :]
    control_archive = control_archive[: control_archive.index("sudo sha256sum")]
    assert '"$STOCKER_V1_RELEASE"' not in control_archive
    assert "v1-release-sha256.txt" in runbook[:deletion]
    assert "v1-release-manifest-sha256.txt" in runbook[:deletion]
    common_rollback = runbook.index("For either rollback path")
    before_callback_rollback = runbook.index("### Before first callback")
    after_callback_rollback = runbook.index("### After first callback")
    assert common_rollback < before_callback_rollback < after_callback_rollback
    common_rollback_commands = runbook[common_rollback:before_callback_rollback]
    assert "systemctl unmask" not in common_rollback_commands
    assert 'sudo rm -- "$dropin"' in common_rollback_commands
    assert "rmdir " not in common_rollback_commands
    assert 'sudo test ! -L "$dropin"' in common_rollback_commands
    assert 'test "$actual_fence" = "$expected_fence"' in common_rollback_commands
    assert "systemctl daemon-reload" in common_rollback_commands
    assert "active_state=" in common_rollback_commands
    assert "load_state=" in common_rollback_commands
    assert "fragment_path=" in common_rollback_commands
    assert "refuse_manual=" in common_rollback_commands
    assert "merged_unit=" in common_rollback_commands
    assert "dropin_paths=" in common_rollback_commands
    assert "--property=LoadState --value" in common_rollback_commands
    assert "--property=DropInPaths --value" in common_rollback_commands
    assert "--property=Conditions --value" not in common_rollback_commands
    rollback_verify = 'LC_ALL=C systemd-analyze verify "$unit" >/dev/null 2>&1 || exit 78'
    assert rollback_verify in common_rollback_commands
    assert 'merged_unit="$(sudo systemctl cat "$unit")"' in common_rollback_commands
    assert '*"$STOCKER_V1_FENCE_SENTINEL"*) exit 78' in common_rollback_commands
    assert "systemd-analyze condition" not in common_rollback_commands
    assert "assert_v1_fence_condition_false" not in common_rollback_commands
    assert (
        common_rollback_commands.index("sudo systemctl daemon-reload")
        < common_rollback_commands.index(rollback_verify)
        < common_rollback_commands.index('merged_unit="$(sudo systemctl cat "$unit")"')
        < common_rollback_commands.index("--property=RefuseManualStart --value")
    )
    assert 'test "$refuse_manual" = no' in common_rollback_commands
    assert "setfacl --restore=" in common_rollback_commands
    assert "rollback-access-control-before.txt" in common_rollback_commands
    assert 'sudo -u stocker test -w "$STOCKER_V1_ROLLBACK_DB"' in common_rollback_commands


def test_v1_runtime_start_fence_is_exact_reversible_and_preserves_unit_files() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    fence = runbook[
        runbook.index("STOCKER_V1_FENCE_SENTINEL=") : runbook.index(
            "retire_authorised_v1_candidate\n"
        )
    ]
    units = (
        "stocker-recorder.service",
        "stocker-web.service",
        "stocker-backup.service",
        "stocker-backup.timer",
        "stocker-recorder-session-readiness.service",
        "stocker-recorder-session-readiness.timer",
    )

    assert "99-stocker-v1-cutover-start-fence.conf" in fence
    assert "/run/systemd/system/${unit}.d" in fence
    assert 'sudo test ! -e "$dropin"' in fence
    assert "sudo install -d -o root -g root -m 0755" in fence
    assert "'[Unit]'" in fence
    assert "'RefuseManualStart=yes'" in fence
    assert '"ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL"' in fence
    assert 'sudo test -f "$dropin"' in fence
    assert 'sudo test ! -L "$dropin"' in fence
    assert 'test "$actual_fence" = "$expected_fence"' in fence
    assert 'LC_ALL=C systemd-analyze verify "$unit"' in fence
    assert 'LC_ALL=C systemd-analyze condition --unit="$unit"' in fence
    assert "condition_output=" in fence
    assert '*"ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL failed."*) ;;' in fence
    assert 'test "$condition_status" -eq 1' in fence
    assert "assert_v1_fence_condition_false" not in fence
    assert 'ConditionPathExists=/"' not in fence
    assert "--property=Conditions --value" not in fence
    assert 'sudo test -f "/etc/systemd/system/$unit"' in fence
    assert 'sudo test ! -L "/etc/systemd/system/$unit"' in fence
    assert 'sudo systemctl start "$unit"' in fence
    assert "mv /etc/systemd/system" not in fence
    assert "rm -- /etc/systemd/system" not in fence
    for unit in units:
        assert unit in fence


def test_legacy_integrity_checks_use_immutable_uris_and_leave_no_sidecars(
    tmp_path: Path,
) -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")

    prescribed_reads = (
        '"file:$STOCKER_V1_SNAPSHOT?mode=ro&immutable=1"',
        '"file:$STOCKER_V1_ROLLBACK_DB?mode=ro&immutable=1"',
        '"file:$STOCKER_V1_RECOVERY_COPIES/restore-check/'
        'import-snapshot.sqlite3?mode=ro&immutable=1"',
        '"file:$STOCKER_V1_RECOVERY_COPIES/restore-check/rollback.sqlite3?mode=ro&immutable=1"',
    )
    assert all(read in runbook for read in prescribed_reads)
    assert runbook.count("sqlite3 -readonly") == len(prescribed_reads)
    assert 'sqlite3 -readonly "$STOCKER_' not in runbook

    original_checks = runbook.index("sqlite-integrity.txt")
    original_sidecar_gate = runbook.index(
        'for checked_db in "$STOCKER_V1_SNAPSHOT" "$STOCKER_V1_ROLLBACK_DB"'
    )
    restore_checks = runbook.index("restore-integrity.txt")
    restore_sidecar_gate = runbook.index("for checked_restore in")
    restore_removal = runbook.index(
        'sudo rm -- "$STOCKER_V1_RECOVERY_COPIES/restore-check/import-snapshot.sqlite3"'
    )
    assert original_checks < original_sidecar_gate
    assert restore_checks < restore_sidecar_gate < restore_removal
    for suffix in ("-journal", "-wal", "-shm"):
        assert f'"${{checked_db}}{suffix}"' in runbook[original_sidecar_gate:restore_checks]
        assert f'"${{checked_restore}}{suffix}"' in runbook[restore_sidecar_gate:restore_removal]

    database = tmp_path / "frozen-wal.sqlite3"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO evidence VALUES (1)")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-journal", "-wal", "-shm"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)

    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert tuple(connection.execute("PRAGMA foreign_key_check")) == ()
    assert all(not Path(f"{database}{suffix}").exists() for suffix in ("-journal", "-wal", "-shm"))


def test_official_api_update_service_uses_the_v2_runtime_cli() -> None:
    unit = _unit("stocker-ibkr-api-update.service")

    assert "stocker-runtime ibkr-api check-update" in unit
    assert "stocker-prospective" not in unit
    assert "/opt/stocker/v2-current" in unit
    assert "/opt/stocker/current" not in unit
    assert "User=stocker-recorder" in unit
    assert "Group=stocker-readers" in unit


def test_installable_release_has_no_v1_runtime_or_entrypoint() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    check_script = (ROOT / "scripts/check.sh").read_text(encoding="utf-8")

    assert not tuple((ROOT / "packages/stocker_prospective").rglob("*.py"))
    assert not (ROOT / "configs/prospective").exists()
    assert "stocker-prospective" not in project
    assert "packages/stocker_prospective" not in project
    assert "check_prospective" not in workflow
    assert "stocker-prospective" not in workflow
    assert "check_prospective" not in check_script
    final_fixture = (
        ROOT
        / "tests/fixtures/legacy_prospective_migrations/0026_opening_leader_continuation_v0.sql"
    )
    assert final_fixture.is_file()


class _SignalMarketData:
    capabilities = frozenset({"market_data"})

    def __init__(self, on_connect: object) -> None:
        self.on_connect = on_connect
        self.callback = None

    def set_callback(self, callback: object) -> None:
        self.callback = callback

    def set_disconnect_callback(self, _callback: object) -> None:
        return None

    def set_status_callback(self, _callback: object) -> None:
        return None

    def configure_subscriptions(self, _subscriptions: object) -> None:
        return None

    def connect(self) -> None:
        assert callable(self.on_connect)
        self.on_connect(signal.SIGTERM, None)

    def disconnect(self) -> None:
        return None

    def subscribe(self, _fence: CallbackFence) -> None:
        return None

    def retry_subscription(self, _fence: CallbackFence) -> None:
        return None

    def cancel(self, _request_id: int) -> None:
        return None


def _market_data_input_json() -> str:
    return json.dumps(
        {
            "instruments": [
                {
                    "instrument_id": "instrument-1",
                    "ibkr_con_id": 123,
                    "kind": "stock",
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                }
            ],
            "subscriptions": [
                {
                    "name": "required-quotes",
                    "instrument_id": "instrument-1",
                    "feed_kind": "quotes",
                    "request_id": 3,
                    "continuity_required": True,
                    "optional": False,
                    "stale_after_us": 15_000_000,
                }
            ],
        }
    )


def test_recorder_health_window_tracks_exact_xnys_sessions() -> None:
    from stocker_runtime.cli import (
        _market_data_expected_since_us,
        _retention_work_expected,
    )

    def at_us(year: int, month: int, day: int, hour: int, minute: int) -> int:
        return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1_000_000)

    regular_open = at_us(2026, 8, 10, 13, 30)
    regular_close = at_us(2026, 8, 10, 20, 0)
    assert _market_data_expected_since_us(regular_open - 1) is None
    assert _market_data_expected_since_us(regular_open) == regular_open
    assert _market_data_expected_since_us(regular_close - 1) == regular_open
    assert _market_data_expected_since_us(regular_close) is None
    assert _retention_work_expected(regular_open - 1) is True
    assert _retention_work_expected(regular_open) is False
    assert _retention_work_expected(regular_close - 1) is False
    assert _retention_work_expected(regular_close) is True

    thanksgiving_midday = at_us(2026, 11, 26, 17, 0)
    assert _market_data_expected_since_us(thanksgiving_midday) is None
    assert _retention_work_expected(thanksgiving_midday) is True

    early_open = at_us(2026, 11, 27, 14, 30)
    early_close = at_us(2026, 11, 27, 18, 0)
    assert _market_data_expected_since_us(early_open) == early_open
    assert _market_data_expected_since_us(early_close - 1) == early_open
    assert _market_data_expected_since_us(early_close) is None
    assert _retention_work_expected(early_open) is False
    assert _retention_work_expected(early_close) is True

    standard_time_open = at_us(2026, 1, 5, 14, 30)
    standard_time_close = at_us(2026, 1, 5, 21, 0)
    assert _retention_work_expected(standard_time_open) is False
    assert _retention_work_expected(standard_time_close) is True


def test_off_session_retention_capacity_exceeds_frozen_regular_session_load() -> None:
    from stocker_runtime.cli import RECORDER_MAINTENANCE_INTERVAL_US
    from stocker_runtime.storage import RetentionPolicy
    from stocker_runtime.storage.retention import MAX_RECEIPT_CHECKPOINT_CALLBACKS_PER_PASS

    off_session_seconds = int(17.5 * 60 * 60)
    opportunities = off_session_seconds * 1_000_000 // RECORDER_MAINTENANCE_INTERVAL_US
    receipt_capacity = opportunities * MAX_RECEIPT_CHECKPOINT_CALLBACKS_PER_PASS
    evidence_row_capacity = opportunities * RetentionPolicy().maintenance_batch_rows
    frozen_regular_session_callbacks = 12_132 * 390

    assert opportunities == 6_300
    assert receipt_capacity == 7_560_000
    assert frozen_regular_session_callbacks == 4_731_480
    assert receipt_capacity > frozen_regular_session_callbacks
    assert evidence_row_capacity == 12_600_000


def test_maintenance_tick_takes_fresh_session_decision_at_call_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.cli import _recorder_maintenance_tick

    calls: list[tuple[int, bool]] = []
    session_checks: list[int] = []

    class MaintenanceRecorder:
        def maintain(self, *, now_us: int, retention_work_expected: bool) -> None:
            calls.append((now_us, retention_work_expected))

    monkeypatch.setattr("stocker_runtime.cli.time.time_ns", lambda: 200_000_000)

    def session_policy(at_us: int) -> bool:
        session_checks.append(at_us)
        return False

    monkeypatch.setattr("stocker_runtime.cli._retention_work_expected", session_policy)

    assert _recorder_maintenance_tick(cast(Any, MaintenanceRecorder())) == 200_000
    assert session_checks == [200_000]
    assert calls == [(200_000, False)]


def test_server_dependency_closure_includes_the_runtime_market_calendar() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    server_dependencies = project["dependency-groups"]["server"]
    bootstrap = (ROOT / "scripts/bootstrap_server.sh").read_text(encoding="utf-8")

    assert "pandas-market-calendars>=4.4" in server_dependencies
    assert "uv sync --locked --no-editable --no-default-groups --group server" in bootstrap


def test_recorder_health_tick_marks_expected_staleness_and_always_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.cli import _recorder_health_tick

    class HealthRecorder:
        def __init__(self) -> None:
            self.stale_calls: list[tuple[int, bool, int | None]] = []
            self.recovery_calls: list[int] = []
            self.subscription_recovery_calls: list[int] = []

        def mark_stale(
            self,
            *,
            now_us: int,
            market_data_expected: bool,
            expected_since_us: int | None,
        ) -> int:
            self.stale_calls.append((now_us, market_data_expected, expected_since_us))
            return 0

        def recover_connection(self, *, now_us: int) -> bool:
            self.recovery_calls.append(now_us)
            return False

        def recover_subscriptions(self, *, now_us: int) -> int:
            self.subscription_recovery_calls.append(now_us)
            return 0

    recorder = HealthRecorder()
    monkeypatch.setattr(
        "stocker_runtime.cli._market_data_expected_since_us",
        lambda _now_us: 995,
    )
    _recorder_health_tick(cast(Any, recorder), now_us=1_000)
    monkeypatch.setattr(
        "stocker_runtime.cli._market_data_expected_since_us",
        lambda _now_us: None,
    )
    _recorder_health_tick(cast(Any, recorder), now_us=2_000)

    assert recorder.stale_calls == [(1_000, True, 995), (2_000, False, None)]
    assert recorder.recovery_calls == [1_000, 2_000]
    assert recorder.subscription_recovery_calls == [1_000, 2_000]


def test_recorder_service_loop_invokes_bounded_health_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    initialize_database(database)
    config = tmp_path / "recorder.json"
    inputs = tmp_path / "market-data.json"
    config.write_text(
        json.dumps(
            {
                "database": str(database),
                "run_id": "run-health-service",
                "owner_id": "owner-health-service",
                "mode": "prospective_record",
                "host": "127.0.0.1",
                "port": 4003,
                "client_id": 71,
                "read_only": True,
                "external_read_only_verified": True,
                "config_hash": "c" * 64,
                "git_commit": "0000000",
            }
        ),
        encoding="utf-8",
    )
    inputs.write_text(_market_data_input_json(), encoding="utf-8")
    handlers: dict[int, object] = {}

    def install_handler(signum: int, handler: object) -> object:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr("stocker_runtime.cli.signal.signal", install_handler)
    clock_ns = {"value": 1_000_000_000}
    monkeypatch.setattr("stocker_runtime.cli.time.time_ns", lambda: clock_ns["value"])
    monkeypatch.setattr(
        "stocker_runtime.cli.IBKRMarketData.official",
        lambda **_kwargs: _SignalMarketData(lambda _signum, _frame: None),
    )
    health_calls: list[int] = []
    monkeypatch.setattr(
        "stocker_runtime.cli._recorder_health_tick",
        lambda _recorder, *, now_us: health_calls.append(now_us),
    )
    real_drain = Recorder.drain
    drain_calls = 0
    idle_work_flags: list[bool] = []

    def drain_then_stop(
        recorder: Recorder,
        *,
        now_us: int,
        limit: int = 256,
        run_downstream_when_idle: bool = True,
    ) -> int:
        nonlocal drain_calls
        drain_calls += 1
        idle_work_flags.append(run_downstream_when_idle)
        result = real_drain(
            recorder,
            now_us=now_us,
            limit=limit,
            run_downstream_when_idle=run_downstream_when_idle,
        )
        if drain_calls == 2:
            clock_ns["value"] = 2_000_000_000
        if drain_calls == 3:
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
        return result

    monkeypatch.setattr(Recorder, "drain", drain_then_stop)
    result = CliRunner().invoke(
        app,
        ["recorder", "run", "--config", str(config), "--inputs", str(inputs)],
    )

    assert result.exit_code == 0, result.output
    assert idle_work_flags == [True, False, True]
    assert health_calls == [2_000_000]
    assert json.loads(result.stdout)["termination"] == "sigterm"


def test_recorder_service_command_handles_sigterm_as_a_clean_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    (backup_directory / "backup-status.json").write_text(
        '{"checked_at_us":1,"code":"BackupCapacityError","format_version":1,'
        '"latest_manifest_filename":null,"state":"degraded"}\n',
        encoding="utf-8",
    )
    initialize_database(database)
    config = tmp_path / "recorder.json"
    inputs = tmp_path / "market-data.json"
    config.write_text(
        json.dumps(
            {
                "database": str(database),
                "run_id": "run-service",
                "owner_id": "owner-service",
                "mode": "prospective_record",
                "host": "127.0.0.1",
                "port": 4003,
                "client_id": 71,
                "read_only": True,
                "external_read_only_verified": True,
                "config_hash": "a" * 64,
                "git_commit": "0000000",
                "backup_directory": str(backup_directory),
            }
        ),
        encoding="utf-8",
    )
    inputs.write_text(_market_data_input_json(), encoding="utf-8")
    handlers: dict[int, object] = {}

    def install_handler(signum: int, handler: object) -> object:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr("stocker_runtime.cli.signal.signal", install_handler)

    def create_adapter(**_kwargs: object) -> _SignalMarketData:
        return _SignalMarketData(lambda signum, frame: handlers[signum](signum, frame))

    monkeypatch.setattr("stocker_runtime.cli.IBKRMarketData.official", create_adapter)
    result = CliRunner().invoke(
        app,
        ["recorder", "run", "--config", str(config), "--inputs", str(inputs)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["termination"] == "sigterm"
    with connect_v2(database) as connection:
        run = connection.execute(
            "SELECT status, ended_at_us FROM runs WHERE run_id='run-service'"
        ).fetchone()
        generation = connection.execute(
            "SELECT clean_stop, termination_code FROM recorder_generations "
            "WHERE run_id='run-service'"
        ).fetchone()
        incident = connection.execute(
            "SELECT scope, severity, code, details_json FROM incidents "
            "WHERE run_id='run-service' AND code='BACKUP_DEGRADED'"
        ).fetchone()
    assert tuple(run) == ("running", None)
    assert tuple(generation) == (1, "CLEAN_STOP")
    assert tuple(incident) == (
        "storage",
        "degraded",
        "BACKUP_DEGRADED",
        '{"backup_status_code":"BackupCapacityError"}',
    )


def test_recorder_service_failure_leaves_an_unclean_generation_for_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    (backup_directory / "backup-status.json").write_text(
        '{"checked_at_us":1,"code":"BackupCapacityError","format_version":1,'
        '"latest_manifest_filename":null,"state":"degraded"}\n',
        encoding="utf-8",
    )
    initialize_database(database)
    config = tmp_path / "recorder.json"
    inputs = tmp_path / "market-data.json"
    config.write_text(
        json.dumps(
            {
                "database": str(database),
                "run_id": "run-restart",
                "owner_id": "owner-restart",
                "mode": "prospective_record",
                "host": "127.0.0.1",
                "port": 4003,
                "client_id": 71,
                "read_only": True,
                "external_read_only_verified": True,
                "config_hash": "b" * 64,
                "git_commit": "0000000",
                "backup_directory": str(backup_directory),
            }
        ),
        encoding="utf-8",
    )
    inputs.write_text(_market_data_input_json(), encoding="utf-8")
    monkeypatch.setattr(
        "stocker_runtime.cli.signal.signal",
        lambda _signum, _handler: signal.SIG_DFL,
    )
    monkeypatch.setattr("stocker_runtime.cli.time.time_ns", lambda: 1_000_000_000)

    monkeypatch.setattr(
        "stocker_runtime.cli.IBKRMarketData.official",
        lambda **_kwargs: _SignalMarketData(lambda _signum, _frame: None),
    )
    real_drain = Recorder.drain

    def fail_drain(_recorder: Recorder, *, now_us: int) -> None:
        raise RuntimeError(f"temporary drain failure at {now_us}")

    monkeypatch.setattr(Recorder, "drain", fail_drain)
    failed = CliRunner().invoke(
        app,
        ["recorder", "run", "--config", str(config), "--inputs", str(inputs)],
    )

    assert failed.exit_code == 1
    with connect_v2(database) as connection:
        run = connection.execute(
            "SELECT status, ended_at_us FROM runs WHERE run_id='run-restart'"
        ).fetchone()
        generation = connection.execute(
            "SELECT ended_at_us, clean_stop, termination_code FROM recorder_generations "
            "WHERE run_id='run-restart' AND generation=1"
        ).fetchone()
    assert tuple(run) == ("running", None)
    assert tuple(generation) == (None, 0, None)

    monkeypatch.setattr(Recorder, "drain", real_drain)
    restarted = Recorder(
        load_recorder_config(config),
        _SignalMarketData(lambda _signum, _frame: None),
    )
    state = restarted.start(
        now_us=62_000_000,
        instruments=(InstrumentSpec("instrument-1", 123, "stock", "AAPL", "SMART", "USD"),),
        subscriptions=(
            SubscriptionSpec(
                "required-quotes",
                "instrument-1",
                "quotes",
                3,
                True,
                False,
                15_000_000,
            ),
        ),
    )
    assert state.recorder_generation == 2
    restarted.stop(now_us=63_000_000)
