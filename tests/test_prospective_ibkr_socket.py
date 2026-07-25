from __future__ import annotations

from pathlib import Path

import pytest

from stocker_prospective.config import RuntimeSafetyError
from stocker_prospective.ibkr import require_ibkr_socket_loopback_only


def _write_proc_table(path: Path, *listeners: tuple[str, int]) -> None:
    rows = [
        "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt",
    ]
    for index, (address, port) in enumerate(listeners):
        rows.append(
            f"{index:4d}: {address}:{port:04X} 00000000:0000 0A "
            "00000000:00000000 00:00000000 00000000"
        )
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def test_ibkr_socket_preflight_accepts_only_loopback_listeners(tmp_path: Path) -> None:
    _write_proc_table(tmp_path / "tcp", ("0100007F", 4002))
    _write_proc_table(
        tmp_path / "tcp6",
        ("00000000000000000000000001000000", 4002),
    )

    addresses = require_ibkr_socket_loopback_only(4002, proc_net_root=tmp_path)

    assert addresses == ("127.0.0.1", "::1")


@pytest.mark.parametrize(
    "address",
    [
        "00000000",
        "0101A8C0",
    ],
)
def test_ibkr_socket_preflight_rejects_public_or_wildcard_ipv4(
    tmp_path: Path,
    address: str,
) -> None:
    _write_proc_table(tmp_path / "tcp", (address, 4002))
    _write_proc_table(tmp_path / "tcp6")

    with pytest.raises(RuntimeSafetyError, match="ibkr socket is not loopback-only"):
        require_ibkr_socket_loopback_only(4002, proc_net_root=tmp_path)


def test_ibkr_socket_preflight_rejects_wildcard_ipv6(tmp_path: Path) -> None:
    _write_proc_table(tmp_path / "tcp")
    _write_proc_table(tmp_path / "tcp6", ("0" * 32, 4002))

    with pytest.raises(RuntimeSafetyError, match="ibkr socket is not loopback-only"):
        require_ibkr_socket_loopback_only(4002, proc_net_root=tmp_path)


def test_ibkr_socket_preflight_reports_missing_exact_port_as_transient(
    tmp_path: Path,
) -> None:
    _write_proc_table(tmp_path / "tcp", ("0100007F", 4001))
    _write_proc_table(tmp_path / "tcp6")

    with pytest.raises(RuntimeError, match="configured_socket_not_listening:4002"):
        require_ibkr_socket_loopback_only(4002, proc_net_root=tmp_path)
