from __future__ import annotations

import ast
from pathlib import Path

from stocker_runtime import (
    DiscoveryReceipt,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    MarketDataInterest,
    MarketDataRequirement,
    MarketEvent,
    Observation,
    ProposedPosition,
    ProposedTrade,
    Signal,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "packages/stocker_runtime/src/stocker_runtime"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_runtime_contract_has_no_privileged_imports() -> None:
    imported = _imported_modules(RUNTIME / "ideas/contract.py")
    forbidden_roots = {
        "ibapi",
        "os",
        "pathlib",
        "stocker_execution",
        "stocker_prospective",
    }
    forbidden_runtime_modules = {
        "stocker_runtime.config",
        "stocker_runtime.ingestion",
        "stocker_runtime.storage",
        "stocker_runtime.web",
    }

    assert not ({module.split(".", maxsplit=1)[0] for module in imported} & forbidden_roots)
    assert not (imported & forbidden_runtime_modules)


def test_public_contract_schemas_reject_extra_fields_and_expose_no_authority_fields() -> None:
    public_models = (
        Observation,
        Signal,
        ProposedPosition,
        ProposedTrade,
        MarketEvent,
        IdeaManifest,
        IdeaActivation,
        MarketDataRequirement,
        MarketDataInterest,
        DiscoveryReceipt,
        IdeaBatch,
        IdeaEvaluation,
    )
    forbidden_fields = {
        "account",
        "account_id",
        "approval",
        "approval_id",
        "broker_order",
        "broker_order_id",
        "execution",
        "execution_id",
        "live",
        "paper",
        "risk",
        "risk_decision",
        "transmit",
    }

    for model in public_models:
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert not (set(model.model_fields) & forbidden_fields)

    json_value_schema = IdeaActivation.model_json_schema()["$defs"]["JsonValue"]
    json_value_types = {branch.get("type") for branch in json_value_schema["anyOf"]}
    assert {"array", "object"} <= json_value_types


def test_runtime_replaces_execution_placeholder_in_packaging_and_launcher() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    launcher = (ROOT / "stocker_launcher.py").read_text(encoding="utf-8")

    assert "packages/stocker_runtime/src" in pyproject
    assert '"stocker_runtime"' in pyproject
    assert "packages/stocker_runtime/src" in launcher
    assert "packages/stocker_execution" not in pyproject
    assert "packages/stocker_execution" not in launcher
    assert not any((ROOT / "packages/stocker_execution").rglob("*.py"))
    assert not (ROOT / "apps/server/scripts/run_executor.py").exists()
    assert not (ROOT / "configs/server.example.yaml").exists()
