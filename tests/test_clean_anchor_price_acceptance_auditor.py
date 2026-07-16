from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = (
    ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work/"
    "audit_clean_anchor_price_acceptance_v1.py"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("clean_anchor_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auditor_reconstructs_long_and_short_acceptance() -> None:
    module = _module()

    long = module.reconstruct_acceptance(
        anchor_reference_price=100.0, high=102.0, low=99.5, close=101.0, direction=1
    )
    short = module.reconstruct_acceptance(
        anchor_reference_price=100.0, high=100.5, low=98.0, close=99.0, direction=-1
    )

    assert long[:4] == pytest.approx((100.0, 200.0, 50.0, 150.0))
    assert short[:4] == pytest.approx((100.0, 200.0, 50.0, 150.0))
    assert long[4] is True and short[4] is True


def test_auditor_safety_rejects_runtime_paths() -> None:
    module = _module()

    assert module.prohibited_changed_paths(
        [
            "research/slrno-v2/report.md",
            "packages/stocker_research/src/example.py",
            "packages/stocker_execution/src/orders.py",
        ]
    ) == ["packages/stocker_execution/src/orders.py"]


def test_exact_identity_includes_plot_hashes(tmp_path: Path) -> None:
    module = _module()
    primary = tmp_path / "primary"
    exact = tmp_path / "exact"
    (primary / "plots").mkdir(parents=True)
    (exact / "plots").mkdir(parents=True)
    (primary / "table.csv").write_text("x\n1\n")
    (exact / "table.csv").write_text("x\n1\n")
    (primary / "plots/plot.png").write_bytes(b"same")
    (exact / "plots/plot.png").write_bytes(b"different")

    result = module.verify_exact_identity(primary, exact)

    assert result["byte_identical"] is False
    assert result["hash_mismatches"] == ["plots/plot.png"]
