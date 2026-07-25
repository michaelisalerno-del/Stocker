from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SYSTEMD = ROOT / "deploy/systemd"
VERIFY_SCRIPT = ROOT / "deploy/scripts/verify-ibgateway-installation.sh"
PROXY_SCRIPT = ROOT / "deploy/scripts/run-ibgateway-loopback-proxy.sh"
BOUNDARY_SCRIPT = ROOT / "deploy/scripts/verify-ibgateway-loopback-boundary.sh"
RUNBOOK = ROOT / "docs/operations/prospective-server-runbook.md"
SERVER_CONFIG = ROOT / "configs/prospective/server.example.yaml"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def test_gateway_display_never_exposes_x11_tcp() -> None:
    unit = _unit("stocker-ibgateway-display.service")

    assert "User=ibgateway" in unit
    assert "Group=ibgateway" in unit
    assert "-nolisten tcp" in unit
    assert "-auth /var/lib/ibgateway/.Xauthority" in unit
    assert " -ac" not in unit
    assert "ConditionPathExists=/var/lib/ibgateway/.Xauthority" in unit
    assert "StartLimitBurst=" in unit
    assert "ReadWritePaths=/var/lib/ibgateway /tmp" in unit
    assert "ReadWritePaths=/var/lib/stocker" not in unit


def test_gateway_process_uses_installed_official_boundary_without_credentials() -> None:
    unit = _unit("stocker-ibgateway.service")
    lowered = unit.lower()

    assert "ExecStart=/opt/ibgateway/current/ibgateway" in unit
    assert "DISPLAY=:71" in unit
    assert "XAUTHORITY=/var/lib/ibgateway/.Xauthority" in unit
    assert "User=ibgateway" in unit
    assert "HOME=/var/lib/ibgateway" in unit
    assert "WorkingDirectory=/var/lib/ibgateway" in unit
    assert "ConditionPathExists=/opt/ibgateway/current/ibgateway" in unit
    assert "ExecCondition=/usr/bin/test -x /usr/local/libexec/stocker-verify-ibgateway" in unit
    assert "ExecCondition=+/usr/local/libexec/stocker-verify-ibgateway" in unit
    assert (
        "ExecCondition=/usr/bin/test -x "
        "/usr/local/libexec/stocker-verify-ibgateway-loopback-boundary"
    ) in unit
    assert ("ExecCondition=+/usr/local/libexec/stocker-verify-ibgateway-loopback-boundary") in unit
    assert "ReadWritePaths=/var/lib/ibgateway /tmp" in unit
    assert "ReadWritePaths=/var/lib/stocker" not in unit
    assert "EnvironmentFile=" not in unit
    assert "username" not in lowered
    assert "password" not in lowered
    assert "2fa" not in lowered


def test_gateway_vnc_is_loopback_only_and_password_protected() -> None:
    unit = _unit("stocker-ibgateway-vnc.service")

    assert "-localhost" in unit
    assert "-rfbport 5901" in unit
    assert "-rfbportv6 5901" in unit
    assert "-rfbauth /var/lib/ibgateway/vnc.pass" in unit
    assert "ConditionPathExists=/var/lib/ibgateway/vnc.pass" in unit
    assert "StartLimitBurst=" in unit
    assert "0.0.0.0" not in unit
    assert "WantedBy=multi-user.target" not in unit


def test_gateway_api_proxy_exposes_only_a_verified_loopback_endpoint() -> None:
    socket_unit = _unit("stocker-ibgateway-loopback-proxy.socket")
    service_unit = _unit("stocker-ibgateway-loopback-proxy.service")
    boundary_unit = _unit("stocker-ibgateway-loopback-boundary.service")
    gateway_unit = _unit("stocker-ibgateway.service")

    assert "ListenStream=127.0.0.1:4003" in socket_unit
    assert "0.0.0.0" not in socket_unit
    assert "Requires=stocker-ibgateway-loopback-boundary.service" in socket_unit
    assert "After=stocker-ibgateway-loopback-boundary.service" in socket_unit
    assert "TriggerLimitIntervalSec=1h" in socket_unit
    assert "TriggerLimitBurst=1" in socket_unit
    assert "User=ibgateway" in service_unit
    assert "ExecStart=/usr/local/libexec/stocker-ibgateway-loopback-proxy" in service_unit
    assert "StartLimitIntervalSec=1h" in service_unit
    assert "StartLimitBurst=1" in service_unit
    assert "RestartPreventExitStatus=78" in service_unit
    assert "EnvironmentFile=" not in service_unit
    assert "User=root" in boundary_unit
    assert (
        "ExecStart=/usr/local/libexec/stocker-verify-ibgateway-loopback-boundary"
    ) in boundary_unit
    assert "Requires=ufw.service" in boundary_unit
    assert "stocker-ibgateway-loopback-proxy.socket" in gateway_unit


def test_gateway_proxy_runner_accepts_only_one_numeric_upstream_port(
    tmp_path: Path,
) -> None:
    config = tmp_path / "proxy.env"
    config.write_text("IBGATEWAY_UPSTREAM_PORT=4999\n", encoding="ascii")
    environment = {
        **os.environ,
        "IBGATEWAY_PROXY_CONFIG": str(config),
        "IBGATEWAY_SOCKET_PROXYD": "/bin/echo",
    }

    accepted = subprocess.run(
        [str(PROXY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert accepted.returncode == 0
    assert "--connections-max=4 127.0.0.1:4999" in accepted.stdout

    config.write_text(
        "IBGATEWAY_UPSTREAM_PORT=4999\nUNEXPECTED=value\n",
        encoding="ascii",
    )
    rejected = subprocess.run(
        [str(PROXY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert rejected.returncode != 0
    assert "blocked_unsafe_runtime_configuration" in rejected.stderr


@pytest.mark.parametrize("reserved_port", (22, 80, 443, 4003))
def test_gateway_proxy_runner_rejects_public_and_proxy_ports(
    tmp_path: Path,
    reserved_port: int,
) -> None:
    config = tmp_path / "proxy.env"
    config.write_text(
        f"IBGATEWAY_UPSTREAM_PORT={reserved_port}\n",
        encoding="ascii",
    )
    rejected = subprocess.run(
        [str(PROXY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "IBGATEWAY_PROXY_CONFIG": str(config),
            "IBGATEWAY_SOCKET_PROXYD": "/bin/true",
        },
    )

    assert rejected.returncode == 78
    assert "upstream_port_reserved" in rejected.stderr


def _mock_command(path: Path, output: str) -> Path:
    path.write_text(
        "#!/bin/sh\nprintf '%s' " + shlex.quote(output) + "\n",
        encoding="ascii",
    )
    path.chmod(0o755)
    return path


def _run_boundary_verifier(
    tmp_path: Path,
    *,
    ufw_status: str = "Status: active\n",
    ipv4_rules: str | None = None,
    ipv6_rules: str | None = None,
    include_config: bool = True,
) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "proxy.env"
    if include_config:
        config.write_text("IBGATEWAY_UPSTREAM_PORT=4002\n", encoding="ascii")
    valid_ipv4 = (
        "-N ufw-user-input\n"
        "-A ufw-user-input -i lo -p tcp -m tcp --dport 4002 -j ACCEPT\n"
        "-A ufw-user-input -p tcp -m tcp --dport 4002 -j DROP\n"
        "-A ufw-user-input -p tcp -m tcp --dport 22 -j ACCEPT\n"
    )
    valid_ipv6 = (
        "-N ufw6-user-input\n"
        "-A ufw6-user-input -i lo -p tcp -m tcp --dport 4002 -j ACCEPT\n"
        "-A ufw6-user-input -p tcp -m tcp --dport 4002 -j DROP\n"
        "-A ufw6-user-input -p tcp -m tcp --dport 22 -j ACCEPT\n"
    )
    return subprocess.run(
        [str(BOUNDARY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "IBGATEWAY_PROXY_CONFIG": str(config),
            "IBGATEWAY_UFW": str(_mock_command(tmp_path / "ufw", ufw_status)),
            "IBGATEWAY_IPTABLES": str(
                _mock_command(
                    tmp_path / "iptables",
                    valid_ipv4 if ipv4_rules is None else ipv4_rules,
                )
            ),
            "IBGATEWAY_IP6TABLES": str(
                _mock_command(
                    tmp_path / "ip6tables",
                    valid_ipv6 if ipv6_rules is None else ipv6_rules,
                )
            ),
        },
    )


def test_gateway_boundary_verifier_accepts_effective_ipv4_and_ipv6_rules(
    tmp_path: Path,
) -> None:
    verified = _run_boundary_verifier(tmp_path)

    assert verified.returncode == 0, verified.stderr
    assert "ibgateway_loopback_boundary:verified:4002" in verified.stdout


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"include_config": False}, "config_missing"),
        ({"ufw_status": "Status: inactive\n"}, "ufw_inactive"),
        (
            {
                "ipv4_rules": (
                    "-N ufw-user-input\n"
                    "-A ufw-user-input -j ACCEPT\n"
                    "-A ufw-user-input -i lo -p tcp --dport 4002 -j ACCEPT\n"
                    "-A ufw-user-input -p tcp --dport 4002 -j DROP\n"
                )
            },
            "ipv4_first_rule_not_loopback_allow",
        ),
        (
            {
                "ipv6_rules": (
                    "-N ufw6-user-input\n-A ufw6-user-input -i lo -p tcp --dport 4002 -j ACCEPT\n"
                )
            },
            "ipv6_second_rule_not_non_loopback_deny",
        ),
    ),
)
def test_gateway_boundary_verifier_fails_closed(
    tmp_path: Path,
    kwargs: dict[str, object],
    reason: str,
) -> None:
    rejected = _run_boundary_verifier(tmp_path, **kwargs)  # type: ignore[arg-type]

    assert rejected.returncode == 78
    assert f"ibgateway_loopback_boundary:{reason}" in rejected.stderr


def test_record_only_server_template_uses_only_the_stocker_proxy_port() -> None:
    config = SERVER_CONFIG.read_text(encoding="utf-8")

    assert "port: 4003" in config
    assert "Stocker loopback proxy" in config
    assert "port: 7497" not in config


def test_gateway_login_runbook_requires_ssh_tunnel_and_manual_2fa() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    assert "ssh -N -L 5901:127.0.0.1:5901" in runbook
    assert "manual IBKR username, password, and 2FA" in runbook
    assert "never enter the Stocker website" in runbook
    assert "Read-Only API" in runbook
    assert "ufw default deny incoming" in runbook
    assert 'case "$IBKR_GATEWAY_PORT" in' in runbook
    assert "sudo ufw insert 1 deny in" in runbook
    assert "sudo ufw insert 1 allow in on lo" in runbook
    assert "stocker-verify-ibgateway-loopback-boundary" in runbook
    assert "127.0.0.1:4003" in runbook
    assert "frozen internal deployment contract" in runbook
    assert "IBGATEWAY_UPSTREAM_PORT=" in runbook
    assert "sha256sum --check" in runbook
    assert 'sudo ln "$INSTALLER_TMP" "$INSTALLER"' in runbook
    assert 'sudo mkdir --mode=0750 "$TARGET"' in runbook
    assert 'sudo test ! -e "$PROVENANCE"' in runbook
    assert 'sudo ln "$PROVENANCE_TMP" "$PROVENANCE"' in runbook
    assert "verify-ibgateway-installation.sh" in runbook
    assert "useradd --system --home-dir /var/lib/ibgateway" in runbook
    assert 'x11vnc -storepasswd "$VNC_PASSWORD"' not in runbook
    assert "ReadWritePaths=/var/lib/stocker" not in "".join(
        _unit(name)
        for name in (
            "stocker-ibgateway-display.service",
            "stocker-ibgateway-window-manager.service",
            "stocker-ibgateway-vnc.service",
            "stocker-ibgateway.service",
        )
    )


@pytest.mark.skipif(sys.platform != "linux", reason="verifier targets Linux systemd hosts")
def test_gateway_integrity_verifier_fails_after_installed_file_mutation(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "releases"
    installer_root = tmp_path / "installers"
    target = release_root / "10481e-test"
    target.mkdir(parents=True)
    installer_root.mkdir()
    launcher = target / "ibgateway"
    launcher.write_bytes(b"verified launcher\n")
    launcher.chmod(0o755)
    installer = installer_root / "ibgateway.sh"
    installer.write_bytes(b"verified installer\n")
    identity = target.name
    manifest = installer_root / f"{identity}.manifest.sha256"
    launcher_sha = hashlib.sha256(launcher.read_bytes()).hexdigest()
    manifest.write_text(f"{launcher_sha}  ./ibgateway\n", encoding="ascii")
    launcher_link = target / "launcher-link"
    launcher_link.symlink_to("ibgateway")
    symlink_manifest = installer_root / f"{identity}.symlinks"
    symlink_manifest.write_text("launcher-link\tibgateway\n", encoding="ascii")
    provenance = installer_root / f"{identity}.runtime-provenance"
    provenance.write_text(
        "\n".join(
            (
                "manifest_version=2",
                "source_url=https://download2.interactivebrokers.com/installers/"
                "ibgateway/latest-standalone/"
                "ibgateway-latest-standalone-linux-x64.sh",
                f"installer_path={installer}",
                f"installer_sha256={hashlib.sha256(installer.read_bytes()).hexdigest()}",
                f"installed_path={target}",
                f"file_manifest_path={manifest}",
                f"file_manifest_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}",
                f"symlink_manifest_path={symlink_manifest}",
                "symlink_manifest_sha256="
                f"{hashlib.sha256(symlink_manifest.read_bytes()).hexdigest()}",
                "recorded_at_utc=2026-07-25T00:00:00Z",
            )
        )
        + "\n",
        encoding="ascii",
    )
    active = tmp_path / "current"
    active.symlink_to(target)
    environment = {
        **os.environ,
        "IBGATEWAY_ACTIVE_LINK": str(active),
        "IBGATEWAY_INSTALLER_ROOT": str(installer_root),
        "IBGATEWAY_RELEASE_ROOT": str(release_root),
        "IBGATEWAY_EXPECTED_OWNER": target.owner(),
    }

    verified = subprocess.run(
        ["/bin/sh", str(VERIFY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert verified.returncode == 0, verified.stderr

    launcher_link.unlink()
    launcher_link.symlink_to("/bin/sh")
    escaped_link = subprocess.run(
        ["/bin/sh", str(VERIFY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert escaped_link.returncode != 0
    assert "ibgateway_integrity:symlink_target_outside_release" in escaped_link.stderr

    launcher_link.unlink()
    launcher_link.symlink_to("ibgateway")
    launcher.write_bytes(b"mutated launcher\n")
    rejected = subprocess.run(
        ["/bin/sh", str(VERIFY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert rejected.returncode != 0
    assert "ibgateway_integrity:installed_file_manifest_mismatch" in rejected.stderr
