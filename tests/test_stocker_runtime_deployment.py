from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from stocker_runtime.cli import app
from stocker_runtime.ingestion import CallbackFence, Recorder, load_recorder_config
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
    protobuf_install = runbook.index('--offline --no-deps --reinstall "$STOCKER_PROTOBUF_WHEEL"')
    ibapi_install = runbook.index('--offline --no-deps --reinstall "$STOCKER_IBAPI_SOURCE"')
    metadata_gate = runbook.index('requires("ibapi")')
    concrete_import_gate = runbook.index("from ibapi.client import EClient")
    publish = runbook.index('sudo ln -s "$STOCKER_V2_RELEASE" /opt/stocker/v2-current')

    assert "--offline" in runbook[protobuf_hash:ibapi_install]
    assert "--no-deps" in runbook[protobuf_hash:ibapi_install]
    assert regular_wheel < exact_wheel < manifest_binding < protobuf_hash
    assert protobuf_hash < protobuf_install < ibapi_install
    assert ibapi_install < metadata_gate < concrete_import_gate < publish
    assert 'version("ibapi") != "10.49.1"' in runbook[metadata_gate:publish]
    assert 'version("protobuf") != "5.29.5"' in runbook[metadata_gate:publish]


def test_cutover_dependency_failures_cannot_reach_release_pointer(tmp_path: Path) -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    block_start = runbook.index("```bash", runbook.index("Prepare the separate V2 pointer"))
    block_start += len("```bash\n")
    block_end = runbook.index("getent group stocker-readers", block_start)
    block = runbook[block_start:block_end]

    release = tmp_path / "reviewed-release"
    source = tmp_path / "reviewed-ibapi-source"
    wheel = tmp_path / "protobuf-5.29.5-reviewed.whl"
    manifest = tmp_path / "protobuf-5.29.5-reviewed.sha256"
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
        ): f'export STOCKER_PROTOBUF_WHEEL="{wheel}"',
        (
            "export STOCKER_PROTOBUF_WHEEL_SHA256=/var/lib/stocker/ibkr-api/install/"
            "protobuf-5.29.5-wheel.sha256"
        ): f'export STOCKER_PROTOBUF_WHEEL_SHA256="{manifest}"',
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
if [[ "$1" == "ln" && "$2" == "-s" ]]; then
  : > "$PUBLISHED"
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_sudo.chmod(0o755)

    trace = tmp_path / "trace"
    published = tmp_path / "published"
    environment = os.environ.copy()
    environment.update(
        PATH=f"{fake_bin}:{environment['PATH']}",
        PUBLISHED=str(published),
        TRACE=str(trace),
    )
    failures = (
        "sha256sum --check",
        (
            f"pip install --python {release}/.venv/bin/python --offline --no-deps "
            f"--reinstall {wheel}"
        ),
        (
            f"pip install --python {release}/.venv/bin/python --offline --no-deps "
            f"--reinstall {source}"
        ),
        f"{release}/.venv/bin/python -",
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

        assert completed.returncode != 0, failure
        assert not published.exists(), failure
        assert "ln -s" not in trace.read_text(encoding="utf-8"), failure


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
    assert str(web.database) == "/var/lib/stocker/v2/stocker-v2.sqlite3"
    assert str(web.backup_directory) == "/var/lib/stocker/backups-v2"
    assert market_data == {"instruments": [], "subscriptions": []}


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

    def cancel(self, _request_id: int) -> None:
        return None


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
    inputs.write_text('{"instruments":[],"subscriptions":[]}', encoding="utf-8")
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
    assert tuple(run) == ("stopped", run["ended_at_us"])
    assert run["ended_at_us"] is not None
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
    inputs.write_text('{"instruments":[],"subscriptions":[]}', encoding="utf-8")
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
    state = restarted.start(now_us=62_000_000, instruments=(), subscriptions=())
    assert state.recorder_generation == 2
    restarted.stop(now_us=63_000_000)
