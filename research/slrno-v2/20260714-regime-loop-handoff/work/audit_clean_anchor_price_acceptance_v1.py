#!/usr/bin/env python3
"""Independent auditor for Clean Anchor Price Acceptance V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260716-clean-anchor-price-acceptance-v1.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260716-clean-anchor-price-acceptance-v1/primary"
DEFAULT_EXACT = WORK / "artifacts/20260716-clean-anchor-price-acceptance-v1/exact_rerun"
EXCLUSIONS = {"artifact_manifest.json", "exact_rerun_identity.json", "independent_audit.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _resolved(value: str) -> Path:
    return (CONTRACT_PATH.parent / value).resolve()


def prohibited_changed_paths(paths: Iterable[str]) -> list[str]:
    allowed = ("research/slrno-v2/", "packages/stocker_research/", "tests/")
    return sorted(path for path in paths if not path.startswith(allowed))


def reconstruct_acceptance(
    *,
    anchor_reference_price: float,
    high: float,
    low: float,
    close: float,
    direction: int,
) -> tuple[float, float, float, float, bool]:
    """Independent literal reconstruction of the registered primary rule."""

    if direction == 1:
        signed = 10_000.0 * (close / anchor_reference_price - 1.0)
        favourable = 10_000.0 * (high / anchor_reference_price - 1.0)
        adverse = 10_000.0 * (1.0 - low / anchor_reference_price)
    elif direction == -1:
        signed = 10_000.0 * (1.0 - close / anchor_reference_price)
        favourable = 10_000.0 * (1.0 - low / anchor_reference_price)
        adverse = 10_000.0 * (high / anchor_reference_price - 1.0)
    else:
        raise ValueError("ambiguous direction")
    balance = favourable - adverse
    return signed, favourable, adverse, balance, signed > 0.0 and favourable > adverse


def verify_exact_identity(primary: Path, exact: Path) -> dict[str, object]:
    """Compare all machine files and plot bytes, independently and recursively."""

    suffixes = {".parquet", ".csv", ".json", ".png"}
    left = {
        str(path.relative_to(primary)): path
        for path in primary.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in EXCLUSIONS
    }
    right = {
        str(path.relative_to(exact)): path
        for path in exact.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in EXCLUSIONS
    }
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    mismatches = sorted(
        name for name in set(left) & set(right) if sha256(left[name]) != sha256(right[name])
    )
    return {
        "byte_identical": not missing and not extra and not mismatches,
        "compared_files_including_plots": len(left),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
    }


def _check_contract_and_inputs(
    contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[bool, str]:
    contract_hash = sha256(CONTRACT_PATH)
    if metadata["contract_hash"] != contract_hash:
        return False, "metadata contract hash mismatch"
    entries = {
        "v2_trade_decisions": contract["inputs"]["v2_trade_decisions"],
        "static_anchor_policy_ledger": contract["inputs"]["static_anchor_policy_ledger"],
        "anchor_mass_ledger": contract["inputs"]["anchor_mass_ledger"],
        "episode_states": contract["inputs"]["episode_states"],
        "episode_diagnostics": contract["inputs"]["episode_diagnostics"],
        "provider_hash_manifest": contract["inputs"]["provider_hash_manifest"],
        "range_prediction_ledger": contract["inputs"]["range_prediction_ledger"],
    }
    for name, entry in entries.items():
        actual = sha256(_resolved(str(entry["path"])))
        if actual != str(entry["sha256"]):
            return False, f"input drift: {name}"
        if metadata["input_hashes"][name] != actual:
            return False, f"metadata input hash mismatch: {name}"
    return True, f"contract {contract_hash} and {len(entries)} inputs independently hashed"


def _check_population(root: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    source = pd.read_parquet(root / "named_source_opportunity_ledger.parquet")
    policy = pd.read_parquet(_resolved(contract["inputs"]["static_anchor_policy_ledger"]["path"]))
    frozen = policy.loc[
        policy["track"].eq("track_a_named_family")
        & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
    ]
    source_ids = set(source["opportunity_id"].astype(str))
    frozen_ids = set(frozen["opportunity_id"].astype(str))
    if source_ids != frozen_ids or len(source) != 1663:
        return False, "named source population does not equal frozen policy population"
    expected = {
        (2023, "cycle_04", "state_4"): 132,
        (2023, "cycle_07", "state_5"): 722,
        (2025, "cycle_04", "state_4"): 96,
        (2025, "cycle_07", "state_5"): 713,
    }
    actual = source.groupby(["period", "loop_id", "orientation"]).size().to_dict()
    if actual != expected:
        return False, f"named family count drift: {actual}"
    if not bool(source["direction"].isin([-1, 1]).all()):
        return False, "ambiguous source direction survived fail-closed construction"
    controls = pd.read_parquet(root / "control_source_opportunity_ledger.parquet")
    expected_controls = {
        (2023, "negative_control", "cycle_07", "state_6"): 331,
        (2023, "neutral_control", "cycle_04", "state_2"): 8,
        (2025, "negative_control", "cycle_07", "state_6"): 296,
        (2025, "neutral_control", "cycle_04", "state_2"): 6,
    }
    actual_controls = (
        controls.groupby(["period", "population_role", "loop_id", "orientation"]).size().to_dict()
    )
    if actual_controls != expected_controls:
        return False, f"frozen control population drift: {actual_controls}"
    return True, "all 1,663 named and 641 frozen control identities and counts match"


def _check_static_veto(root: Path) -> tuple[bool, str]:
    veto = pd.read_parquet(root / "static_anchor_veto_ledger.parquet")
    good = pd.to_numeric(veto["anchor_good_mass"], errors="coerce")
    bad = pd.to_numeric(veto["anchor_bad_mass"], errors="coerce")
    odds = np.where(bad.eq(0.0), np.inf, good / bad)
    expected = pd.Series(odds, index=veto.index).gt(1.0)
    if not expected.eq(veto["static_anchor_veto_pass"]).all():
        return False, "static anchor score/threshold reconstruction failed"
    if not pd.to_numeric(veto["static_anchor_veto_threshold"], errors="coerce").eq(1.0).all():
        return False, "static anchor threshold drift"
    return True, f"reconstructed {len(veto)} frozen odds decisions at strict threshold 1.0"


def _provider(contract: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    path = (
        Path(contract["inputs"]["provider_2025_root"])
        / f"symbol={symbol}"
        / "timeframe=5m/data.parquet"
    )
    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _check_provider_hashes(
    contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[bool, str]:
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_hash_manifest"]["path"]).read_text(encoding="utf-8")
    )["sha256"]
    checked = 0
    for key, expected in manifest.items():
        if not key.startswith("provider_2025_"):
            continue
        symbol = key.removeprefix("provider_2025_")
        path = (
            Path(contract["inputs"]["provider_2025_root"])
            / f"symbol={symbol}"
            / "timeframe=5m/data.parquet"
        )
        actual = sha256(path)
        if actual != expected or metadata["provider_hashes"].get(symbol) != actual:
            return False, f"provider hash mismatch for {symbol}"
        checked += 1
    return checked == 20, f"independently hashed {checked} frozen 2025 provider files"


def _check_checkpoints_acceptance_and_payoff(
    root: Path, contract: Mapping[str, Any]
) -> tuple[bool, str]:
    source = pd.read_parquet(root / "named_source_opportunity_ledger.parquet")
    features = pd.read_parquet(root / "price_acceptance_feature_ledger.parquet")
    outcomes = pd.read_parquet(root / "constant_terminal_remaining_payoff_ledger.parquet")
    source_2025 = source.loc[source["period"].eq(2025)].set_index("opportunity_id")
    features_2025 = features.loc[features["period"].eq(2025)].set_index("opportunity_id")
    outcomes_2025 = outcomes.loc[outcomes["period"].eq(2025)].set_index("opportunity_id")
    providers: dict[str, pd.DataFrame] = {}
    for opportunity_id, row in source_2025.iterrows():
        opportunity = str(opportunity_id)
        symbol = str(row["symbol"])
        providers.setdefault(symbol, _provider(contract, symbol))
        frame = providers[symbol]
        anchor = pd.Timestamp(row["anchor_timestamp"])
        terminal = pd.Timestamp(row["original_terminal_timestamp"])
        exact_anchor = frame.loc[frame["timestamp"].eq(anchor)]
        exact_checkpoint = frame.loc[frame["timestamp"].eq(anchor + pd.Timedelta(minutes=5))]
        exact_entry = frame.loc[frame["timestamp"].eq(anchor + pd.Timedelta(minutes=10))]
        exact_terminal = frame.loc[frame["timestamp"].eq(terminal - pd.Timedelta(minutes=5))]
        if not all(
            len(value) == 1
            for value in [exact_anchor, exact_checkpoint, exact_entry, exact_terminal]
        ):
            return False, f"exact provider clock unavailable for {opportunity}"
        feature = cast(pd.Series, features_2025.loc[opportunity])
        outcome = cast(pd.Series, outcomes_2025.loc[opportunity])
        checkpoint = exact_checkpoint.iloc[0]
        expected = reconstruct_acceptance(
            anchor_reference_price=float(row["anchor_close"]),
            high=float(checkpoint["high"]),
            low=float(checkpoint["low"]),
            close=float(checkpoint["close"]),
            direction=int(row["direction"]),
        )
        observed = (
            float(feature["signed_close_return_bps"]),
            float(feature["favourable_excursion_bps"]),
            float(feature["adverse_excursion_bps"]),
            float(feature["acceptance_balance_bps"]),
        )
        if not np.allclose(expected[:4], observed, rtol=0.0, atol=1e-10):
            return False, f"acceptance arithmetic mismatch for {opportunity}"
        if bool(feature["price_acceptance_pass"]) != expected[4]:
            return False, f"acceptance decision mismatch for {opportunity}"
        if pd.Timestamp(feature["checkpoint_freeze_timestamp"]) != anchor + pd.Timedelta(
            minutes=10
        ):
            return False, f"checkpoint freeze mismatch for {opportunity}"
        entry = float(exact_entry.iloc[0]["open"])
        exit_price = float(exact_terminal.iloc[0]["close"])
        gross = 10_000.0 * int(row["direction"]) * (exit_price / entry - 1.0)
        if not math.isclose(float(outcome["gross_payoff_bps"]), gross, abs_tol=1e-10):
            return False, f"constant-terminal gross payoff mismatch for {opportunity}"
        if not math.isclose(float(outcome["net_payoff_bps"]), gross - 10.0, abs_tol=1e-10):
            return False, f"entry/exit cost mismatch for {opportunity}"
        if pd.Timestamp(outcome["entry_timestamp"]) != anchor + pd.Timedelta(minutes=10):
            return False, f"same-clock entry mismatch for {opportunity}"
        if pd.Timestamp(outcome["exit_timestamp"]) != terminal:
            return False, f"original terminal mismatch for {opportunity}"
    unavailable_2023 = source.loc[source["period"].eq(2023), "source_available"]
    if unavailable_2023.any():
        return False, "2023 missing provider evidence was not kept unavailable"
    return True, "independently reconstructed all 809 available 2025 checkpoints and payoffs"


def _check_variant_pairing(root: Path) -> tuple[bool, str]:
    decisions = pd.read_parquet(root / "variant_decision_ledger.parquet")
    populations = {
        variant: tuple(sorted(group["opportunity_id"].astype(str)))
        for variant, group in decisions.groupby("variant", sort=True)
    }
    if len(populations) != 5 or len(set(populations.values())) != 1:
        return False, "A--E source populations differ"
    if decisions["replacement_opportunity_id"].notna().any():
        return False, "replacement opportunity entered paired comparison"
    if decisions["overlap_or_capacity_refilled"].any():
        return False, "overlap/capacity refill contaminated comparison"
    if not decisions["existing_position_action"].eq("unchanged").all():
        return False, "existing-position action changed"
    d = decisions.loc[decisions["variant"].eq("D_anchor_veto_plus_price_acceptance")]
    expected_d = d["static_anchor_veto_pass"] & d["price_acceptance_pass"] & d["source_available"]
    if not d["admitted"].eq(expected_d).all():
        return False, "Variant D is not exact registered intersection"
    e = decisions.loc[decisions["variant"].eq("E_anchor_veto_plus_price_acceptance_plus_range")]
    if not e["decision"].eq("unavailable").all():
        return False, "Variant E was scored without immutable range forecasts"
    return (
        True,
        f"five variants preserve {len(populations[next(iter(populations))])} source identities",
    )


def _check_paired_metrics(root: Path) -> tuple[bool, str]:
    decisions = pd.read_parquet(root / "variant_decision_ledger.parquet")
    metrics = pd.read_csv(root / "paired_comparison_metrics.csv")
    available = decisions.loc[decisions["decision"].ne("unavailable")]
    pivot = available.pivot(
        index="opportunity_id", columns="variant", values="policy_net_payoff_bps"
    )
    difference = pivot["D_anchor_veto_plus_price_acceptance"] - pivot["A_same_clock_base"]
    row = metrics.loc[
        metrics["comparison"].eq("D_vs_A_primary") & metrics["slice_type"].eq("all")
    ].iloc[0]
    if not math.isclose(
        float(row["paired_total_difference_bps"]), float(difference.sum()), abs_tol=1e-8
    ):
        return False, "primary paired total does not reconstruct"
    if not math.isclose(
        float(row["paired_mean_difference_bps"]), float(difference.mean()), abs_tol=1e-8
    ):
        return False, "primary paired mean does not reconstruct"
    return True, f"independently paired {len(difference)} observable source opportunities"


def _check_timing_and_feature_isolation(root: Path) -> tuple[bool, str]:
    features = pd.read_parquet(root / "price_acceptance_feature_ledger.parquet")
    available = features.loc[features["price_acceptance_status"].eq("available")]
    start = pd.to_datetime(available["checkpoint_bar_start_timestamp"], utc=True)
    freeze = pd.to_datetime(available["forecast_freeze_timestamp"], utc=True)
    feature = pd.to_datetime(available["feature_max_availability_timestamp"], utc=True)
    if not (start + pd.Timedelta(minutes=5)).eq(freeze).all() or not feature.le(freeze).all():
        return False, "feature/freeze timestamp violation"
    forbidden = [
        column
        for column in features.columns
        if any(token in column.lower() for token in ["mfe", "mae", "realised", "realized", "route"])
    ]
    if forbidden:
        return False, f"future outcome fields entered feature ledger: {forbidden}"
    if not available["future_path_feature_count"].eq(0).all():
        return False, "future path feature count is nonzero"
    return True, f"verified causal freeze and feature isolation on {len(available)} rows"


def _check_stress_and_separation(root: Path) -> tuple[bool, str]:
    twice = pd.read_parquet(root / "twice_cost_ledger.parquet")
    if not twice["stressed_total_cost_bps"].eq(20.0).all():
        return False, "twice-cost stress did not charge 20 bps"
    if not np.allclose(twice["stressed_net_payoff_bps"], twice["gross_payoff_bps"] - 20.0):
        return False, "twice-cost stress changed more than costs"
    stress = pd.read_csv(root / "stress_test_results.csv")
    source = pd.read_parquet(root / "named_source_opportunity_ledger.parquet")
    available = source.loc[source["source_available"]]
    admitted = available.loc[
        available["static_anchor_veto_pass"] & available["price_acceptance_pass"]
    ]
    twice_summary = stress.loc[stress["stress_test"].eq("twice_costs")].iloc[0]
    expected_twice_base = float((available["gross_payoff_bps"] - 20.0).sum())
    expected_twice_d = float((admitted["gross_payoff_bps"] - 20.0).sum())
    if not math.isclose(
        float(twice_summary["paired_D_minus_A_bps"]),
        expected_twice_d - expected_twice_base,
        abs_tol=1e-8,
    ):
        return False, "twice-cost paired D-minus-A stress does not reconstruct"
    delay = pd.read_parquet(root / "additional_bar_delay_ledger.parquet")
    if delay["price_acceptance_decision_recomputed"].any():
        return False, "extra-delay stress recomputed first-bar acceptance"
    delay_summary = stress.loc[stress["stress_test"].eq("one_additional_bar_execution_delay")].iloc[
        0
    ]
    expected_delay_base = float(available["additional_delay_net_payoff_bps"].sum())
    expected_delay_d = float(admitted["additional_delay_net_payoff_bps"].sum())
    if not math.isclose(
        float(delay_summary["paired_D_minus_A_bps"]),
        expected_delay_d - expected_delay_base,
        abs_tol=1e-8,
    ):
        return False, "extra-delay paired D-minus-A stress does not reconstruct"
    restarted = pd.read_parquet(root / "restarted_horizon_diagnostic_ledger.parquet")
    if "net_payoff_bps" in restarted.columns:
        return False, "constant-terminal net was mixed into restarted table"
    return True, "twice-cost, frozen extra delay, and restarted separation verified"


def _check_safety(contract: Mapping[str, Any]) -> tuple[bool, str]:
    start = str(contract["frozen_lineage"]["starting_commit"])
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{start}..HEAD"], cwd=REPO, text=True
    )
    changed = [line for line in output.splitlines() if line]
    prohibited = prohibited_changed_paths(changed)
    if prohibited:
        return False, f"prohibited runtime paths changed: {prohibited}"
    return (
        True,
        f"{len(changed)} changed paths are confined to research, tests, and research package",
    )


def run_audit(primary: Path, exact: Path, output: Path) -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    checks: dict[str, tuple[bool, str]] = {
        "contract_and_input_identity": _check_contract_and_inputs(contract, metadata),
        "provider_hash_identity": _check_provider_hashes(contract, metadata),
        "named_candidate_population": _check_population(primary, contract),
        "static_anchor_veto_identity": _check_static_veto(primary),
        "checkpoint_acceptance_and_payoff": _check_checkpoints_acceptance_and_payoff(
            primary, contract
        ),
        "variant_population_pairing": _check_variant_pairing(primary),
        "paired_primary_metric": _check_paired_metrics(primary),
        "feature_availability_and_no_future_path": _check_timing_and_feature_isolation(primary),
        "stress_and_horizon_separation": _check_stress_and_separation(primary),
        "research_only_changed_paths": _check_safety(contract),
    }
    exact_result = verify_exact_identity(primary, exact)
    checks["primary_exact_rerun_identity"] = (
        bool(exact_result["byte_identical"]),
        f"compared {exact_result['compared_files_including_plots']} machine and plot files",
    )
    passed = all(value[0] for value in checks.values())
    result = {
        "audit_id": "clean_anchor_price_acceptance_v1_independent_audit",
        "auditor_version": "1.0.0",
        "passed": passed,
        "contract_hash": sha256(CONTRACT_PATH),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "checks": {
            name: {"passed": value[0], "detail": value[1]} for name, value in checks.items()
        },
        "exact_identity": exact_result,
        "primary_artifact_hashes": {
            str(path.relative_to(primary)): sha256(path)
            for path in sorted(primary.rglob("*"))
            if path.is_file() and path.name != "independent_audit.json"
        },
        "research_only": True,
        "execution_enabled": False,
        "broker_connection_enabled": False,
        "order_placement_enabled": False,
        "position_management_enabled": False,
        "existing_exit_management_enabled": False,
        "deployment_enabled": False,
    }
    write_json(output, result)
    if not passed:
        failed = {name: detail for name, detail in checks.items() if not detail[0]}
        raise AssertionError(f"independent audit failed: {failed}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = Path(args.primary)
    exact = Path(args.exact)
    output = Path(args.output) if args.output else primary / "independent_audit.json"
    result = run_audit(primary, exact, output)
    print(json.dumps({"passed": result["passed"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
