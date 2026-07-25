from __future__ import annotations

import sys
from types import ModuleType

import pytest

import stocker_prospective.ibkr_official as ibkr_official_module


def test_current_official_error_callback_accepts_error_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__path__ = []  # type: ignore[attr-defined]
    fake_client_module = ModuleType("ibapi.client")
    fake_wrapper_module = ModuleType("ibapi.wrapper")

    class FakeWrapper:
        pass

    class FakeClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper = wrapper

        def run(self) -> None:
            self.wrapper.error(  # type: ignore[attr-defined]
                17,
                1_721_943_900,
                2104,
                "Market data farm connection is OK",
                "",
            )

    fake_client_module.EClient = FakeClient  # type: ignore[attr-defined]
    fake_wrapper_module.EWrapper = FakeWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setitem(sys.modules, "ibapi.client", fake_client_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", fake_wrapper_module)
    monkeypatch.setattr(
        ibkr_official_module,
        "require_official_ibkr_api",
        lambda: fake_ibapi,
    )

    errors: list[tuple[int, int, str]] = []

    class Adapter:
        def on_error(self, request_id: int, code: int, message: str) -> None:
            errors.append((request_id, code, message))

    client = ibkr_official_module.create_official_callback_client(Adapter())  # type: ignore[arg-type]

    client.run()

    assert errors == [(17, 2104, "Market data farm connection is OK")]
