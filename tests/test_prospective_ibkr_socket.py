from __future__ import annotations

import threading
from pathlib import Path

import pytest

from stocker_prospective.config import RuntimeSafetyError
from stocker_prospective.ibkr import (
    IBKRConnectionConfig,
    IBKRMarketDataAdapter,
    require_ibkr_socket_loopback_only,
)
from stocker_prospective.market_data import MarketDataBudget, MarketDataType


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

    addresses = require_ibkr_socket_loopback_only(
        "127.0.0.1",
        4002,
        proc_net_root=tmp_path,
    )

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
        require_ibkr_socket_loopback_only(
            "127.0.0.1",
            4002,
            proc_net_root=tmp_path,
        )


def test_ibkr_socket_preflight_rejects_wildcard_ipv6(tmp_path: Path) -> None:
    _write_proc_table(tmp_path / "tcp")
    _write_proc_table(tmp_path / "tcp6", ("0" * 32, 4002))

    with pytest.raises(RuntimeSafetyError, match="ibkr socket is not loopback-only"):
        require_ibkr_socket_loopback_only(
            "::1",
            4002,
            proc_net_root=tmp_path,
        )


def test_ibkr_socket_preflight_reports_missing_exact_port_as_transient(
    tmp_path: Path,
) -> None:
    _write_proc_table(tmp_path / "tcp", ("0100007F", 4001))
    _write_proc_table(tmp_path / "tcp6")

    with pytest.raises(RuntimeError, match="configured_socket_not_listening:4002"):
        require_ibkr_socket_loopback_only(
            "127.0.0.1",
            4002,
            proc_net_root=tmp_path,
        )


@pytest.mark.parametrize("host", ["localhost", "192.168.1.20", "broker.example"])
def test_ibkr_socket_preflight_rejects_nonliteral_or_remote_hosts(
    tmp_path: Path,
    host: str,
) -> None:
    _write_proc_table(tmp_path / "tcp", ("0100007F", 4002))
    _write_proc_table(tmp_path / "tcp6")

    with pytest.raises(RuntimeSafetyError, match="literal loopback address"):
        require_ibkr_socket_loopback_only(host, 4002, proc_net_root=tmp_path)


def test_market_data_adapter_rechecks_socket_before_start_and_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stocker_prospective.ibkr as ibkr_module

    monkeypatch.setattr(ibkr_module, "require_official_ibkr_api", lambda: object())
    preflight_calls: list[tuple[str, int]] = []
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=4,
            reserved_headroom=1,
            request_rate_limit=2,
        ),
        socket_preflight=lambda host, port: preflight_calls.append((host, port)) or ("127.0.0.1",),
    )

    class FakeClient:
        def connect(self, *_: object) -> bool:
            return True

        def disconnect(self) -> None:
            return None

        def run(self) -> None:
            adapter.on_connected(MarketDataType.LIVE)

        def reqMktData(self, *_: object) -> None:  # noqa: N802
            return None

        def cancelMktData(self, *_: object) -> None:  # noqa: N802
            return None

    adapter.attach_official_client(FakeClient())
    adapter.start()
    adapter.reconnect()
    adapter.stop()

    assert preflight_calls == [("127.0.0.1", 4002), ("127.0.0.1", 4002)]


def test_market_data_adapter_contains_client_loop_error_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stocker_prospective.ibkr as ibkr_module

    monkeypatch.setattr(ibkr_module, "require_official_ibkr_api", lambda: object())
    uncaught: list[BaseException] = []
    monkeypatch.setattr(
        threading,
        "excepthook",
        lambda arguments: uncaught.append(arguments.exc_value),
    )
    disconnected = threading.Event()
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=4,
            reserved_headroom=1,
            request_rate_limit=2,
        ),
        socket_preflight=lambda _host, _port: ("127.0.0.1",),
    )

    class FailingOnDisconnectClient:
        def connect(self, *_: object) -> bool:
            return True

        def disconnect(self) -> None:
            disconnected.set()

        def run(self) -> None:
            adapter.on_connected(MarketDataType.LIVE)
            assert disconnected.wait(1)
            raise TypeError("client loop observed cleared server version")

        def reqMktData(self, *_: object) -> None:  # noqa: N802
            return None

        def cancelMktData(self, *_: object) -> None:  # noqa: N802
            return None

    adapter.attach_official_client(FailingOnDisconnectClient())
    adapter.start()
    adapter.stop()

    assert uncaught == []
    assert adapter.fatal_callback_code is None


def test_market_data_adapter_latches_unexpected_client_loop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stocker_prospective.ibkr as ibkr_module

    monkeypatch.setattr(ibkr_module, "require_official_ibkr_api", lambda: object())
    uncaught: list[BaseException] = []
    monkeypatch.setattr(
        threading,
        "excepthook",
        lambda arguments: uncaught.append(arguments.exc_value),
    )
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=4,
            reserved_headroom=1,
            request_rate_limit=2,
        ),
        socket_preflight=lambda _host, _port: ("127.0.0.1",),
    )

    class FailingClient:
        def connect(self, *_: object) -> bool:
            return True

        def disconnect(self) -> None:
            return None

        def run(self) -> None:
            adapter.on_connected(MarketDataType.LIVE)
            raise RuntimeError("unexpected official client loop failure")

        def reqMktData(self, *_: object) -> None:  # noqa: N802
            return None

        def cancelMktData(self, *_: object) -> None:  # noqa: N802
            return None

    adapter.attach_official_client(FailingClient())
    adapter.start()
    assert adapter._loop_thread is not None
    adapter._loop_thread.join(timeout=1)

    assert uncaught == []
    assert adapter.fatal_callback_code == "IBKR_CLIENT_LOOP_FAILURE"

    adapter.stop()


def test_connection_contract_rejects_remote_ibkr_host() -> None:
    with pytest.raises(ValueError, match="literal loopback address"):
        IBKRConnectionConfig(
            host="10.0.0.5",
            port=4002,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        )
