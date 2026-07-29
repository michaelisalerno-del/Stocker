"""Code-only serialization amendment for regime-loop linkage V2.

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


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v3.json"
CONTRACT_SHA256 = "88a60956857e6ccb4fb5e74beb9085e46765e55b31763b26927dc496822ce947"
V2_CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v2.json"
V2_CONTRACT_SHA256 = "090aaec2860f7c0df0dc8f0cb86d1d7fe1d2d72a1110695fce1ff7af859f2aa1"
V2_RUNNER = HERE / "run_regime_loop_linkage_ideas_v2.py"
V2_RUNNER_SHA256 = "b38a17b5e5023951e992004fac51e4c264af2c65e7f19c4b35ecea14cbd5e6ba"
OUT = Path("/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v2_module():
    if sha256(V2_RUNNER) != V2_RUNNER_SHA256:
        raise AssertionError("immutable V2 adapter changed")
    spec = importlib.util.spec_from_file_location("regime_loop_linkage_v2_adapter", V2_RUNNER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load immutable V2 adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_v2_module()
CORE = ADAPTER.PARENT


def load_contract() -> dict[str, Any]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise AssertionError("V3 code-amendment contract changed")
    if sha256(V2_CONTRACT) != V2_CONTRACT_SHA256:
        raise AssertionError("V2 semantic parent changed")
    amendment = json.loads(CONTRACT.read_text())
    base = ADAPTER.load_contract()
    checks = {
        "id": amendment.get("contract_id") == "regime_loop_linkage_ideas_v3",
        "research": amendment.get("research_only") is True,
        "live": amendment.get("live_ordering_enabled") is False,
        "orders": amendment.get("order_placement") == "disabled",
        "v2_uninspected": amendment["v2_fail_closed_record"].get(
            "metrics_or_probabilities_printed_or_inspected"
        )
        is False,
        "statistical_change": amendment["only_code_amendment"].get(
            "changes_probabilities_metrics_gates_or_slice_qualification"
        )
        is False,
        "no_promotion": amendment["unchanged_boundaries"].get(
            "named_loop_good_or_high_promotion_permitted"
        )
        is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"V3 code-amendment safety failure: {checks}")
    base["contract_id"] = amendment["contract_id"]
    base["contract_frozen_at_utc"] = amendment["contract_frozen_at_utc"]
    base["scientific_status"] = amendment["scientific_status"]
    base["v3_amendment"] = amendment
    base["planned_artifact_root"] = str(OUT)
    return base


def qualified_slice_ids(attraction) -> list[str]:
    if attraction.empty:
        return []
    qualified = attraction.loc[
        attraction["qualified_development_attraction_slice"].astype(bool)
    ]
    return [
        f"{row.cycle_id}@state_{int(row.current_state)}@clock_{row.entry_clock_quartile}"
        for row in qualified.itertuples(index=False)
    ]


def run() -> None:
    CORE.CONTRACT = CONTRACT
    CORE.CONTRACT_SHA256 = CONTRACT_SHA256
    CORE.OUT = OUT
    CORE.__file__ = str(Path(__file__).resolve())
    contract = load_contract()
    source_hashes = CORE.verify_sources()
    if OUT.exists():
        raise AssertionError(f"artifact root already exists: {OUT}")
    common, join_audit = ADAPTER.load_common_population(contract)
    composed = CORE.add_fixed_compositions(common)
    linked, fold_audit, parameters = CORE.fit_meta_models(composed)
    evaluated = CORE.evaluate_variants(linked, contract)
    gates = evaluated["gates"]
    attraction = CORE.evaluate_attraction_slices(
        evaluated["primary"], bool(gates["dependency_stack"]["pass"])
    )
    passing = [
        variant
        for variant in CORE.CANDIDATE_VARIANTS
        if bool(gates[variant]["pass"])
    ]
    priority = contract["decision"]["priority_if_multiple_global_variants_pass"]
    selected = next((variant for variant in priority if variant in passing), None)
    decision = {
        "label": (
            "development_linkage_candidate_pending_unseen_validation"
            if selected is not None
            else "linkage_idea_rejected_or_unconfirmed"
        ),
        "passing_variants": passing,
        "selected_variant": selected,
        "qualified_attraction_slices": qualified_slice_ids(attraction),
        "named_loop_good_or_high_promoted": False,
        "same_experiment_refinement_performed": False,
        "later_period_scoring_performed": False,
        "prospective_validated": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    CORE.write_json(OUT / "join_audit.json", join_audit)
    fold_audit.to_csv(OUT / "meta_fold_audit.csv", index=False)
    np.savez_compressed(OUT / "meta_model_parameters.npz", **parameters)
    prediction_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "month",
        "quarter",
        "cycle_index",
        "cycle_id",
        "state",
        "current_state",
        "entry_clock_quartile",
        "inverse_compatible_weight",
        "loop_occurs",
        "qhistory",
        "qlimited4",
        "qfull9",
        *CORE.quality_class_columns(),
        *CORE.quality_probability_columns(),
        *[
            CORE.joint_target_column(target, horizon, tier)
            for target in CORE.TARGETS
            for horizon in CORE.HORIZONS
            for tier in CORE.TIERS
        ],
        *[
            CORE.probability_column(variant, target, horizon, tier)
            for variant in CORE.ALL_VARIANTS
            for target in CORE.TARGETS
            for horizon in CORE.HORIZONS
            for tier in CORE.TIERS
        ],
    ]
    evaluated["primary"].loc[:, prediction_columns].to_parquet(
        OUT / "linkage_predictions_2024_sep_dec.parquet", index=False
    )
    evaluated["cells"].to_csv(OUT / "cell_metrics.csv", index=False)
    evaluated["pooled"].to_csv(OUT / "pooled_metrics.csv", index=False)
    evaluated["multiplicity"].to_csv(OUT / "multiplicity.csv", index=False)
    evaluated["temporal"].to_csv(OUT / "temporal_slices.csv", index=False)
    evaluated["stocks"].to_csv(OUT / "stock_deletions.csv", index=False)
    evaluated["orientations"].to_csv(OUT / "orientation_slices.csv", index=False)
    evaluated["ranking"].to_csv(OUT / "ranking.csv", index=False)
    attraction.to_csv(OUT / "attraction_slices.csv", index=False)
    CORE.write_json(OUT / "variant_gates.json", gates)
    CORE.write_json(OUT / "decision.json", decision)
    summary = {
        "contract_id": contract["contract_id"],
        "scientific_status": contract["scientific_status"],
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(Path(__file__)),
        "implementation_parent_sha256": CORE.sha256(ADAPTER.PARENT_RUNNER),
        "v2_adapter_sha256": sha256(V2_RUNNER),
        "source_hashes": source_hashes,
        "join_audit": join_audit,
        "source_months": list(CORE.SOURCE_MONTHS),
        "primary_evaluation_months": list(CORE.EVALUATION_MONTHS),
        "primary_rows": len(evaluated["primary"]),
        "primary_anchors": int(evaluated["primary"]["anchor_id"].nunique()),
        "meta_fits": len(fold_audit),
        "global_variant_pass": {
            variant: bool(gates[variant]["pass"])
            for variant in CORE.CANDIDATE_VARIANTS
        },
        "decision": decision,
        "direct_volume_fields_used": [],
        "volume_label": "historical_volume_not_used",
        "direction_or_signed_return_used": False,
        "later_period_scoring_performed": False,
        "prospective_shadow_read_or_write_performed": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    CORE.write_json(OUT / "summary.json", summary)
    names = [
        "attraction_slices.csv",
        "cell_metrics.csv",
        "decision.json",
        "join_audit.json",
        "linkage_predictions_2024_sep_dec.parquet",
        "meta_fold_audit.csv",
        "meta_model_parameters.npz",
        "multiplicity.csv",
        "orientation_slices.csv",
        "pooled_metrics.csv",
        "ranking.csv",
        "stock_deletions.csv",
        "summary.json",
        "temporal_slices.csv",
        "variant_gates.json",
    ]
    CORE.write_json(OUT / "artifact_manifest.json", CORE.artifact_manifest(OUT, names))
    print(json.dumps(summary, indent=2, sort_keys=True))


def self_test() -> None:
    contract = load_contract()
    assert contract["contract_id"] == "regime_loop_linkage_ideas_v3"
    assert qualified_slice_ids(
        __import__("pandas").DataFrame(
            columns=[
                "cycle_id",
                "current_state",
                "entry_clock_quartile",
                "qualified_development_attraction_slice",
            ]
        )
    ) == []
    print("self-test passed")


if __name__ == "__main__":
    run()
