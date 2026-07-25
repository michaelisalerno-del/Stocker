from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
SYSTEMD = ROOT / "deploy/systemd"
RUNBOOK = ROOT / "docs/operations/prospective-server-runbook.md"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def test_gateway_display_never_exposes_x11_tcp() -> None:
    unit = _unit("stocker-ibgateway-display.service")

    assert "User=stocker" in unit
    assert "Group=stocker" in unit
    assert "-nolisten tcp" in unit
    assert "-auth /var/lib/stocker/ibgateway/.Xauthority" in unit
    assert " -ac" not in unit


def test_gateway_process_uses_installed_official_boundary_without_credentials() -> None:
    unit = _unit("stocker-ibgateway.service")
    lowered = unit.lower()

    assert "ExecStart=/opt/ibgateway/current/ibgateway" in unit
    assert "DISPLAY=:71" in unit
    assert "XAUTHORITY=/var/lib/stocker/ibgateway/.Xauthority" in unit
    assert "User=stocker" in unit
    assert "EnvironmentFile=" not in unit
    assert "username" not in lowered
    assert "password" not in lowered
    assert "2fa" not in lowered


def test_gateway_vnc_is_loopback_only_and_password_protected() -> None:
    unit = _unit("stocker-ibgateway-vnc.service")

    assert "-localhost" in unit
    assert "-rfbport 5901" in unit
    assert "-rfbauth /var/lib/stocker/ibgateway/vnc.pass" in unit
    assert "0.0.0.0" not in unit
    assert "WantedBy=multi-user.target" not in unit


def test_gateway_login_runbook_requires_ssh_tunnel_and_manual_2fa() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    assert "ssh -N -L 5901:127.0.0.1:5901" in runbook
    assert "manual IBKR username, password, and 2FA" in runbook
    assert "never enter the Stocker website" in runbook
    assert "Read-Only API" in runbook
