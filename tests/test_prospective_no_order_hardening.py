from __future__ import annotations

import ast
import inspect
from pathlib import Path

from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.ibkr_official import OfficialMarketDataOnlyClient
from stocker_prospective.web import create_web_app

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "packages/stocker_prospective/src/stocker_prospective"
FORBIDDEN_METHODS = {
    "placeOrder",
    "cancelOrder",
    "exerciseOptions",
    "reqGlobalCancel",
    "place_order",
    "cancel_order",
    "exercise_options",
    "request_global_cancel",
    "reqAccountSummary",
    "reqAccountUpdates",
    "reqAccountUpdatesMulti",
    "reqPositions",
    "reqPositionsMulti",
    "reqExecutions",
    "reqCompletedOrders",
    "reqPnL",
    "reqPnLSingle",
}
FORBIDDEN_CALLBACKS = {
    "accountSummary",
    "accountUpdateMulti",
    "commissionReport",
    "completedOrder",
    "execDetails",
    "openOrder",
    "orderStatus",
    "position",
    "positionMulti",
    "updateAccountValue",
    "updatePortfolio",
}


def test_web_factory_accepts_only_configuration_not_broker_or_recorder() -> None:
    parameters = inspect.signature(create_web_app).parameters

    assert tuple(parameters) == ("config",)


def test_adapter_and_official_facade_expose_no_order_methods() -> None:
    adapter_surface = set(dir(IBKRMarketDataAdapter))
    official_surface = set(dir(OfficialMarketDataOnlyClient))

    assert not FORBIDDEN_METHODS.intersection(adapter_surface)
    assert not FORBIDDEN_METHODS.intersection(official_surface)


def test_durable_inbox_has_no_execution_order_or_broker_imports() -> None:
    source = (PACKAGE / "durable_inbox.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not {
        module
        for module in imported_modules
        if {"execution", "order", "broker"}.intersection(module.lower().split("."))
    }


def test_virtual_ledger_models_have_no_execution_order_or_broker_imports() -> None:
    source = (PACKAGE / "virtual_positions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not {
        module
        for module in imported_modules
        if {"execution", "order", "broker", "ibkr"}.intersection(module.lower().split("."))
    }


def test_official_callback_bridge_has_only_contained_market_data_callbacks() -> None:
    source = (PACKAGE / "ibkr_official.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    callback_class = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "_StockerOfficialMarketDataClient"
    )
    direct_methods = {
        node.name: node
        for node in callback_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not FORBIDDEN_CALLBACKS.intersection(direct_methods)
    for name, method in direct_methods.items():
        if name == "connect" or name.startswith("_"):
            continue
        assert any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "callback_boundary"
            for decorator in method.decorator_list
        ), name
