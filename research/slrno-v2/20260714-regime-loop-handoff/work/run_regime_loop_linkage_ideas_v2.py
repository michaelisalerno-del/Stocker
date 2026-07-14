"""V2 identity amendment for the causal regime-to-loop linkage test.

V1 stopped before scoring because two differently constructed structural
probabilities were not identical.  V2 explicitly uses the fold-local factor
qhistory for occurrence and excludes the quality ledger's parent loop score.
The immutable V1 algorithm body supplies every unchanged model and gate.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v2.json"
CONTRACT_SHA256 = "090aaec2860f7c0df0dc8f0cb86d1d7fe1d2d72a1110695fce1ff7af859f2aa1"
PARENT_CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v1.json"
PARENT_CONTRACT_SHA256 = "3a8812b1c8a7980565329ab46c88f60ae1cb80bbfe5f738767cb15969589e950"
PARENT_RUNNER = HERE / "run_regime_loop_linkage_ideas_v1.py"
PARENT_RUNNER_SHA256 = "e134f01f4d6da58581205fe8070f90a2f17d0fc0945dea0b42a2ca1c96bfa51a"
OUT = Path("/private/tmp/stocker_regime_loop_linkage_ideas_v2_20260711")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent_module():
    if sha256(PARENT_RUNNER) != PARENT_RUNNER_SHA256:
        raise AssertionError("immutable V1 linkage implementation changed")
    spec = importlib.util.spec_from_file_location("regime_loop_linkage_v1_core", PARENT_RUNNER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load immutable V1 linkage implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARENT = load_parent_module()


def load_contract() -> dict[str, Any]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise AssertionError("V2 amendment contract changed")
    if sha256(PARENT_CONTRACT) != PARENT_CONTRACT_SHA256:
        raise AssertionError("V1 semantic parent changed")
    amendment = json.loads(CONTRACT.read_text())
    base = json.loads(PARENT_CONTRACT.read_text())
    checks = {
        "id": amendment.get("contract_id") == "regime_loop_linkage_ideas_v2",
        "research": amendment.get("research_only") is True,
        "live": amendment.get("live_ordering_enabled") is False,
        "orders": amendment.get("order_placement") == "disabled",
        "parent_contract": amendment["semantic_parent"].get("sha256")
        == PARENT_CONTRACT_SHA256,
        "parent_runner": amendment["implementation_parent"].get("sha256")
        == PARENT_RUNNER_SHA256,
        "v1_unscored": amendment["v1_fail_closed_record"].get(
            "joint_scores_or_metrics_calculated"
        )
        is False,
        "no_promotion": amendment["unchanged_boundaries"].get(
            "named_loop_good_or_high_promotion_permitted"
        )
        is False,
        "no_trading": amendment["unchanged_boundaries"].get(
            "trading_rule_or_PnL_model_permitted"
        )
        is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"V2 amendment safety/semantic failure: {checks}")
    base["contract_id"] = amendment["contract_id"]
    base["contract_frozen_at_utc"] = amendment["contract_frozen_at_utc"]
    base["scientific_status"] = amendment["scientific_status"]
    base["v2_amendment"] = amendment
    base["population_and_join"].pop(
        "qhistory_must_equal_parent_loop_probability_tolerance", None
    )
    base["population_and_join"]["occurrence_probability"] = (
        "factor_occurrence_oof.qhistory"
    )
    base["population_and_join"]["quality_parent_loop_probability_role"] = (
        "diagnostic_only_excluded_from_link_probabilities"
    )
    base["planned_artifact_root"] = str(OUT)
    return base


def load_common_population(contract: dict[str, Any]):
    factor_columns = [
        *PARENT.JOIN_KEYS,
        "anchor_id",
        "cycle_id",
        "state",
        "current_state",
        "target",
        "inverse_compatible_weight",
        "entry_clock_quartile",
        "qhistory",
        "qlimited4",
        "qfull9",
        "month",
    ]
    quality_columns = [
        *PARENT.JOIN_KEYS,
        "anchor_id",
        "cycle_id",
        "state",
        "current_state",
        "loop_occurs",
        "loop_probability",
        "month_key",
        "quarter",
        *PARENT.quality_class_columns(),
        *PARENT.quality_probability_columns(),
    ]
    factor = pd.read_parquet(PARENT.FACTOR_OOF, columns=factor_columns)
    quality = pd.read_parquet(PARENT.QUALITY_OOF, columns=quality_columns)
    if len(factor) != 361220 or len(quality) != 216438:
        raise AssertionError("parent OOF row count changed")
    if factor.duplicated(list(PARENT.JOIN_KEYS)).any() or quality.duplicated(
        list(PARENT.JOIN_KEYS)
    ).any():
        raise AssertionError("parent OOF join keys are not one-to-one")
    factor = factor.rename(
        columns={
            "anchor_id": "factor_anchor_id",
            "cycle_id": "factor_cycle_id",
            "state": "factor_state",
            "current_state": "factor_current_state",
            "target": "factor_loop_occurs",
            "month": "factor_month",
        }
    )
    common = quality.merge(
        factor,
        on=list(PARENT.JOIN_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    identity_checks = {
        "all_quality_rows_joined": common["_merge"].eq("both").all(),
        "cycle_id": common["cycle_id"].astype(str).eq(
            common["factor_cycle_id"].astype(str)
        ).all(),
        "state": common["state"].astype(int).eq(common["factor_state"].astype(int)).all(),
        "current_state": common["current_state"].astype(int).eq(
            common["factor_current_state"].astype(int)
        ).all(),
        "loop_label": common["loop_occurs"].astype(int).eq(
            common["factor_loop_occurs"].astype(int)
        ).all(),
        "month": common["month_key"].astype(str).eq(
            common["factor_month"].astype(str)
        ).all(),
    }
    if not all(identity_checks.values()):
        raise AssertionError(f"V2 parent identity mismatch: {identity_checks}")
    structural_difference = (
        common["qhistory"].to_numpy(float)
        - common["loop_probability"].to_numpy(float)
    )
    diagnostic = {
        "correlation": float(
            np.corrcoef(common["qhistory"], common["loop_probability"])[0, 1]
        ),
        "mean_absolute_difference": float(np.abs(structural_difference).mean()),
        "maximum_absolute_difference": float(np.abs(structural_difference).max()),
    }
    rules = contract["v2_amendment"]["explicit_amendments"][
        "replace_parent_qhistory_equality_gate_with"
    ]
    diagnostic_checks = {
        "correlation": diagnostic["correlation"] >= float(rules["minimum_correlation"]),
        "mean_absolute_difference": diagnostic["mean_absolute_difference"]
        <= float(rules["maximum_mean_absolute_difference"]),
        "maximum_absolute_difference": diagnostic["maximum_absolute_difference"]
        <= float(rules["maximum_absolute_difference"]),
    }
    if not all(diagnostic_checks.values()):
        raise AssertionError(
            f"V2 structural diagnostic outside amendment: {diagnostic} {diagnostic_checks}"
        )
    common = common.drop(columns=["_merge"])
    common["month"] = common["month_key"].astype(str)
    common["session_date"] = common["session_date"].astype(str)
    common["symbol_norm"] = common["symbol_norm"].astype(str)
    if set(common["month"].unique()) != set(PARENT.SOURCE_MONTHS):
        raise AssertionError("source month surface changed")
    if common["anchor_id"].nunique() != 34169:
        raise AssertionError("quality anchor surface changed")
    weight_sums = common.groupby("anchor_id", sort=False)[
        "inverse_compatible_weight"
    ].sum()
    maximum_weight_error = float(
        np.max(np.abs(weight_sums.to_numpy(float) - 1.0))
    )
    if maximum_weight_error > 1e-12:
        raise AssertionError("inverse-compatible weights changed")
    for column in [
        "qhistory",
        "qlimited4",
        "qfull9",
        *PARENT.quality_probability_columns(),
    ]:
        values = common[column].to_numpy(float)
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise AssertionError(f"invalid source probability: {column}")
    audit = {
        "factor_rows": len(factor),
        "quality_rows": len(quality),
        "common_rows": len(common),
        "quality_anchors": int(common["anchor_id"].nunique()),
        "stocks": int(common["symbol_norm"].nunique()),
        "cycles": int(common["cycle_id"].nunique()),
        "states": sorted(int(value) for value in common["current_state"].unique()),
        "identity_checks": identity_checks,
        "structural_probability_diagnostic": diagnostic,
        "structural_probability_diagnostic_checks": diagnostic_checks,
        "occurrence_probability_used": "factor_qhistory",
        "quality_parent_loop_probability_used_in_link": False,
        "maximum_anchor_weight_error": maximum_weight_error,
        "factor_only_rows_excluded": len(factor) - len(common),
    }
    return common.reset_index(drop=True), audit


def run() -> None:
    PARENT.CONTRACT = CONTRACT
    PARENT.CONTRACT_SHA256 = CONTRACT_SHA256
    PARENT.OUT = OUT
    PARENT.load_contract = load_contract
    PARENT.load_common_population = load_common_population
    PARENT.__file__ = str(Path(__file__).resolve())
    PARENT.run()


def self_test() -> None:
    contract = load_contract()
    assert contract["contract_id"] == "regime_loop_linkage_ideas_v2"
    assert contract["population_and_join"]["occurrence_probability"] == (
        "factor_occurrence_oof.qhistory"
    )
    assert contract["research_only"] is True
    print("self-test passed")


if __name__ == "__main__":
    run()
