from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest
import stocker_launcher


@pytest.mark.parametrize(
    "entrypoint,module_name,function_name",
    [("main", "stocker_core.cli", "app"), ("mcp_main", "stocker_mcp.server", "main")],
)
def test_console_entrypoints_apply_frozen_numeric_profile_before_app(
    monkeypatch, entrypoint, module_name, function_name
):
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "X86_V3")
    application = ModuleType(module_name)

    def observe_startup():
        return set(os.environ["NPY_DISABLE_CPU_FEATURES"].split(","))

    setattr(application, function_name, observe_startup)
    monkeypatch.setitem(sys.modules, module_name, application)
    assert getattr(stocker_launcher, entrypoint)() == {
        "X86_V3",
        "X86_V4",
        "AVX512_ICL",
        "AVX512_SPR",
    }


def test_arm_numeric_environment_is_preserved(monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "ASIMDHP")
    stocker_launcher.configure_numeric_runtime()
    assert os.environ["NPY_DISABLE_CPU_FEATURES"] == "ASIMDHP"
