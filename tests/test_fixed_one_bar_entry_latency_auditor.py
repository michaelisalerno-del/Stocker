from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = (
    ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work/"
    "audit_fixed_one_bar_entry_latency_v1.py"
)


def _auditor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fixed_latency_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auditor_independently_reconstructs_long_and_short_pair_delta() -> None:
    auditor = _auditor()

    long = auditor.reconstruct_returns(
        direction=1,
        t0_entry_price=100.0,
        t1_entry_price=99.0,
        terminal_price=102.0,
        cost_bps_per_side=5.0,
    )
    short = auditor.reconstruct_returns(
        direction=-1,
        t0_entry_price=100.0,
        t1_entry_price=101.0,
        terminal_price=98.0,
        cost_bps_per_side=5.0,
    )

    assert long == pytest.approx(
        (200.0, 303.030303030303, 190.0, 293.030303030303, 103.030303030303)
    )
    assert short == pytest.approx(
        (200.0, 297.029702970297, 190.0, 287.029702970297, 97.029702970297)
    )


def test_auditor_rejects_hash_mismatch(tmp_path: Path) -> None:
    auditor = _auditor()
    candidate = tmp_path / "data.parquet"
    candidate.write_bytes(b"not the frozen tape")

    assert auditor.verify_hash(candidate, "0" * 64) is False


def test_auditor_safety_flags_runtime_paths() -> None:
    auditor = _auditor()

    assert auditor.prohibited_changed_paths(
        [
            "research/slrno-v2/report.md",
            "packages/stocker_research/src/stocker_research/example.py",
            "backend/app/orders.py",
        ]
    ) == ["backend/app/orders.py"]


def test_auditor_exact_identity_includes_plots(tmp_path: Path) -> None:
    auditor = _auditor()
    primary = tmp_path / "primary"
    exact = tmp_path / "exact"
    (primary / "plots").mkdir(parents=True)
    (exact / "plots").mkdir(parents=True)
    (primary / "table.csv").write_text("x\n1\n")
    (exact / "table.csv").write_text("x\n1\n")
    (primary / "plots/plot.png").write_bytes(b"same")
    (exact / "plots/plot.png").write_bytes(b"different")

    result = auditor.verify_exact_identity(primary, exact)

    assert result["byte_identical"] is False
    assert result["hash_mismatches"] == ["plots/plot.png"]
