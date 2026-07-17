#!/usr/bin/env python3
"""Independent audit for Fixed One-Bar Entry Latency V1."""

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
CONTRACT_PATH = WORK / "contracts/20260716-fixed-one-bar-entry-latency-v1.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260716-fixed-one-bar-entry-latency-v1/primary"
DEFAULT_EXACT = WORK / "artifacts/20260716-fixed-one-bar-entry-latency-v1/exact_rerun"
EXCLUSIONS = {"artifact_manifest.json", "exact_rerun_identity.json", "independent_audit.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


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


def _resolved(value: object) -> Path:
    return (CONTRACT_PATH.parent / str(value)).resolve()


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def reconstruct_returns(
    *,
    direction: int,
    t0_entry_price: float,
    t1_entry_price: float,
    terminal_price: float,
    cost_bps_per_side: float,
) -> tuple[float, float, float, float, float]:
    """Literal independent reconstruction of T0, T1, and paired delta."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    t0_gross = 10_000.0 * direction * (terminal_price / t0_entry_price - 1.0)
    t1_gross = 10_000.0 * direction * (terminal_price / t1_entry_price - 1.0)
    costs = 2.0 * cost_bps_per_side
    t0_net = t0_gross - costs
    t1_net = t1_gross - costs
    return t0_gross, t1_gross, t0_net, t1_net, t1_net - t0_net


def prohibited_changed_paths(paths: Iterable[str]) -> list[str]:
    allowed = ("research/slrno-v2/", "packages/stocker_research/", "tests/")
    return sorted(path for path in paths if not path.startswith(allowed))


def verify_exact_identity(primary: Path, exact: Path) -> dict[str, object]:
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


def _provider(contract: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    path = (
        Path(str(contract["inputs"]["provider_2025_root"]))
        / f"symbol={symbol}"
        / "timeframe=5m/data.parquet"
    )
    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    return frame


def _check_contract_inputs(
    contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[bool, str]:
    contract_hash = sha256(CONTRACT_PATH)
    if str(metadata["contract_hash"]) != contract_hash:
        return False, "metadata contract hash mismatch"
    checked = 0
    for name, specification in contract["inputs"].items():
        if not isinstance(specification, Mapping) or "path" not in specification:
            continue
        path = _resolved(specification["path"])
        actual = sha256(path)
        if actual != str(specification["sha256"]):
            return False, f"input drift: {name}"
        if str(metadata["input_hashes"][name]) != actual:
            return False, f"metadata input mismatch: {name}"
        checked += 1
    return True, f"independently hashed contract and {checked} frozen inputs"


def _check_provider_hashes(
    contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[bool, str]:
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_2025_hash_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )["sha256"]
    root = Path(str(contract["inputs"]["provider_2025_root"]))
    checked = 0
    for key, expected in manifest.items():
        if not key.startswith("provider_2025_"):
            continue
        symbol = key.removeprefix("provider_2025_")
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        actual = sha256(path)
        if actual != expected or str(metadata["provider_hashes"].get(symbol)) != actual:
            return False, f"provider hash mismatch: {symbol}"
        checked += 1
    return checked == 20, f"independently hashed {checked} provider files"


def _expected_population(
    contract: Mapping[str, Any], *, track: str, controls: bool
) -> pd.DataFrame:
    policy = pd.read_parquet(_resolved(contract["inputs"]["sequential_veto_policy"]["path"]))
    selected = policy.loc[
        policy["track"].eq(track) & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
    ].copy()
    if controls:
        selected = selected.loc[
            selected["population_role"].isin(["neutral_control", "negative_control"])
        ]
    return selected


def _check_population(root: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    named = pd.read_parquet(root / "source_named_opportunity_ledger.parquet")
    controls = pd.read_parquet(root / "source_control_opportunity_ledger.parquet")
    expected_named = _expected_population(contract, track="track_a_named_family", controls=False)
    expected_controls = _expected_population(contract, track="track_b_prior_only", controls=True)
    if set(named["opportunity_id"].astype(str)) != set(
        expected_named["opportunity_id"].astype(str)
    ):
        return False, "named opportunity population differs from frozen policy"
    if set(controls["opportunity_id"].astype(str)) != set(
        expected_controls["opportunity_id"].astype(str)
    ):
        return False, "control population differs from frozen policy"
    named_counts = named.groupby(["period", "loop_id", "orientation"]).size().to_dict()
    expected_named_counts = {
        (2023, "cycle_04", "state_4"): 132,
        (2023, "cycle_07", "state_5"): 722,
        (2025, "cycle_04", "state_4"): 96,
        (2025, "cycle_07", "state_5"): 713,
    }
    control_counts = controls.groupby(["period", "loop_id", "orientation"]).size().to_dict()
    expected_control_counts = {
        (2023, "cycle_04", "state_2"): 8,
        (2023, "cycle_07", "state_6"): 331,
        (2025, "cycle_04", "state_2"): 6,
        (2025, "cycle_07", "state_6"): 296,
    }
    if named_counts != expected_named_counts or control_counts != expected_control_counts:
        return False, "named or control count drift"
    if not named["direction"].isin([-1, 1]).all() or not controls["direction"].isin([-1, 1]).all():
        return False, "ambiguous direction survived source construction"
    return True, "all 1,663 named and 641 control identities independently match"


def _check_t0_against_v2(root: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    t0 = pd.read_parquet(root / "t0_entry_ledger.parquet")
    v2 = pd.read_parquet(_resolved(contract["inputs"]["v2_trade_decisions"]["path"]))
    v2 = v2.loc[v2["model_name"].eq("no_payoff_state_filter")].set_index("opportunity_id")
    t0 = t0.set_index("opportunity_id")
    source = v2.loc[t0.index]
    timestamp_equal = pd.to_datetime(t0["original_entry_timestamp"], utc=True).eq(
        pd.to_datetime(source["entry_timestamp"], utc=True)
    )
    terminal_equal = pd.to_datetime(t0["original_terminal_timestamp"], utc=True).eq(
        pd.to_datetime(source["exit_timestamp"], utc=True)
    )
    if not timestamp_equal.all() or not terminal_equal.all():
        return False, "stored T0 or terminal timestamp differs from V2"
    if not np.allclose(t0["original_entry_price"], source["entry_price"], rtol=0.0, atol=1e-12):
        return False, "stored T0 price differs from V2"
    expected_t0 = pd.to_datetime(source["start_timestamp"], utc=True) + pd.to_timedelta(
        5 * source["entry_step"].astype(int), unit="m"
    )
    if not pd.to_datetime(t0["original_entry_timestamp"], utc=True).eq(expected_t0).all():
        return False, "T0 timestamp does not equal anchor plus frozen entry_step"
    terminal = pd.to_datetime(source["start_timestamp"], utc=True) + pd.Timedelta(minutes=125)
    if not pd.to_datetime(t0["original_terminal_timestamp"], utc=True).eq(terminal).all():
        return False, "terminal is not anchor plus 125 minutes"
    return True, f"reconciled {len(t0)} exact T0 entries and terminals to raw V2"


def _check_t1_and_returns(root: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    paired = pd.concat(
        [
            pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet"),
            pd.read_parquet(root / "control_exact_paired_t0_t1_ledger.parquet"),
        ],
        ignore_index=True,
    )
    providers: dict[str, pd.DataFrame] = {}
    for row in paired.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in providers:
            providers[symbol] = _provider(contract, symbol)
        provider = providers[symbol]
        anchor = _as_timestamp(row.anchor_timestamp)
        t0 = _as_timestamp(row.original_entry_timestamp)
        t1 = _as_timestamp(row.t1_entry_timestamp)
        terminal = _as_timestamp(row.original_terminal_timestamp)
        if t0 != anchor + pd.Timedelta(minutes=5 * _as_int(row.entry_step)):
            return False, f"T0 clock mismatch: {row.opportunity_id}"
        if t1 != t0 + pd.Timedelta(minutes=5):
            return False, f"T1 is not exact T0 plus five minutes: {row.opportunity_id}"
        if terminal != anchor + pd.Timedelta(minutes=125):
            return False, f"terminal clock mismatch: {row.opportunity_id}"
        t0_bar = provider.loc[provider["timestamp"].eq(t0)]
        t1_bar = provider.loc[provider["timestamp"].eq(t1)]
        terminal_bar = provider.loc[provider["timestamp"].eq(terminal - pd.Timedelta(minutes=5))]
        if len(t0_bar) != 1 or len(t1_bar) != 1 or len(terminal_bar) != 1:
            return False, f"exact provider row unavailable: {row.opportunity_id}"
        t0_price = _as_float(row.original_entry_price)
        if not (
            float(t0_bar.iloc[0]["low"]) - 1e-8 <= t0_price <= float(t0_bar.iloc[0]["high"]) + 1e-8
        ):
            return False, f"T0 fill lies outside trigger bar: {row.opportunity_id}"
        t1_price = float(t1_bar.iloc[0]["open"])
        terminal_price = float(terminal_bar.iloc[0]["close"])
        expected = reconstruct_returns(
            direction=_as_int(row.direction),
            t0_entry_price=t0_price,
            t1_entry_price=t1_price,
            terminal_price=terminal_price,
            cost_bps_per_side=5.0,
        )
        observed = (
            _as_float(row.t0_gross_return_bps),
            _as_float(row.t1_gross_return_bps),
            _as_float(row.t0_net_return_bps),
            _as_float(row.t1_net_return_bps),
            _as_float(row.paired_difference_bps),
        )
        if not np.allclose(expected, observed, rtol=0.0, atol=1e-8):
            return False, f"return or cost mismatch: {row.opportunity_id}"
        exact_effect = (
            10_000.0 * _as_int(row.direction) * terminal_price * (1.0 / t1_price - 1.0 / t0_price)
        )
        if not math.isclose(_as_float(row.paired_difference_bps), exact_effect, abs_tol=1e-8):
            return False, f"entry-price effect does not reconcile: {row.opportunity_id}"
    named_count = len(pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet"))
    return True, f"independently reconstructed {len(paired)} pairs ({named_count} named)"


def _check_missing_and_pairing(root: Path) -> tuple[bool, str]:
    unavailable = pd.read_parquet(root / "t1_unavailable_ledger.parquet")
    if (
        len(unavailable.loc[unavailable["period"].eq(2023) & unavailable["population"].eq("named")])
        != 854
    ):
        return False, "2023 named missing rows were lost or imputed"
    if unavailable.loc[unavailable["period"].eq(2023), "t1_net_return_bps"].notna().any():
        return False, "2023 T1 payoff was imputed"
    available = pd.read_parquet(root / "t1_available_entry_ledger.parquet")
    if available["replacement_opportunity_id"].notna().any():
        return False, "replacement opportunity entered T1 population"
    if available["overlap_or_capacity_refilled"].any():
        return False, "overlap/capacity refill changed population"
    if not available["existing_position_action"].eq("unchanged").all():
        return False, "existing position behavior changed"
    return True, f"missing rows explicit ({len(unavailable)}); no replacement or refill"


def _independent_bootstrap(paired: pd.DataFrame) -> tuple[float, float, float]:
    session = (
        paired.groupby(["period", "session_date"], sort=True)["paired_difference_bps"]
        .mean()
        .reset_index()
    )
    values = [
        group["paired_difference_bps"].to_numpy(float)
        for _, group in session.groupby("period", sort=True)
    ]
    rng = np.random.default_rng(20260716)
    draws = np.empty(2000, dtype=float)
    for draw in range(2000):
        sample: list[float] = []
        for period_values in values:
            count = len(period_values)
            starts = rng.integers(0, count, size=int(np.ceil(count / 5)))
            rebuilt = [
                float(period_values[(int(start) + offset) % count])
                for start in starts
                for offset in range(5)
            ]
            sample.extend(rebuilt[:count])
        draws[draw] = float(np.mean(sample))
    return (
        float(session["paired_difference_bps"].mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _check_metrics(root: Path) -> tuple[bool, str]:
    paired = pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet")
    metrics = pd.read_csv(root / "primary_paired_metrics.csv")
    primary = metrics.loc[metrics["slice_type"].eq("all") & metrics["slice_value"].eq("all")].iloc[
        0
    ]
    delta = paired["t1_net_return_bps"] - paired["t0_net_return_bps"]
    checks = [
        math.isclose(
            float(primary["t0_net_payoff_bps"]),
            float(paired["t0_net_return_bps"].sum()),
            abs_tol=1e-8,
        ),
        math.isclose(
            float(primary["t1_net_payoff_bps"]),
            float(paired["t1_net_return_bps"].sum()),
            abs_tol=1e-8,
        ),
        math.isclose(
            float(primary["paired_total_difference_bps"]), float(delta.sum()), abs_tol=1e-8
        ),
        math.isclose(
            float(primary["paired_mean_difference_bps"]), float(delta.mean()), abs_tol=1e-8
        ),
    ]
    if not all(checks):
        return False, "primary paired metrics do not reconstruct"
    observed, lower, upper = _independent_bootstrap(paired)
    bootstrap_checks = [
        math.isclose(float(primary["observed_session_mean_delta_bps"]), observed, abs_tol=1e-10),
        math.isclose(float(primary["bootstrap_lower_95_bps"]), lower, abs_tol=1e-10),
        math.isclose(float(primary["bootstrap_upper_95_bps"]), upper, abs_tol=1e-10),
    ]
    if not all(bootstrap_checks):
        return False, "session-block interval does not independently reconstruct"
    return True, f"paired metrics and 2,000-draw block interval reconstruct for {len(paired)} rows"


def _check_stress_and_concentration(root: Path) -> tuple[bool, str]:
    paired = pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet")
    costs = pd.read_csv(root / "cost_stress_results.csv")
    twice = costs.loc[
        costs["population"].eq("named")
        & costs["slice_type"].eq("all")
        & costs["stress"].eq("twice_costs")
    ].iloc[0]
    t0 = float((paired["t0_gross_return_bps"] - 20.0).sum())
    t1 = float((paired["t1_gross_return_bps"] - 20.0).sum())
    if not math.isclose(float(twice["t0_net_payoff_bps"]), t0, abs_tol=1e-8):
        return False, "twice-cost T0 level mismatch"
    if not math.isclose(float(twice["t1_net_payoff_bps"]), t1, abs_tol=1e-8):
        return False, "twice-cost T1 level mismatch"
    if not math.isclose(float(twice["paired_difference_bps"]), t1 - t0, abs_tol=1e-8):
        return False, "twice-cost paired delta mismatch"
    concentration = pd.read_csv(root / "concentration_results.csv")
    stock = concentration.loc[concentration["dimension"].eq("symbol")]
    contribution = paired.groupby("symbol")["paired_difference_bps"].sum()
    shares = contribution.abs() / contribution.abs().sum()
    if not math.isclose(
        float(stock.iloc[0]["top_one_absolute_share"]), float(shares.max()), abs_tol=1e-10
    ):
        return False, "stock concentration mismatch"
    t2 = pd.read_parquet(root / "t2_sensitivity_ledger.parquet")
    if "t1_net_return_bps" in t2.columns or "paired_difference_bps" in t2.columns:
        return False, "T2 table was not kept separate from T1 naming"
    restarted = pd.read_parquet(root / "restarted_h24_diagnostic_ledger.parquet")
    if "paired_difference_bps" in restarted.columns:
        return False, "restarted h24 contaminated the primary paired table"
    return True, "twice costs, concentration, T2, and restarted separation verified"


def _check_episode_isolation_and_prospective_safety(root: Path) -> tuple[bool, str]:
    causal_files = [
        "source_named_opportunity_ledger.parquet",
        "source_control_opportunity_ledger.parquet",
        "t0_entry_ledger.parquet",
        "t1_expected_entry_ledger.parquet",
        "t1_available_entry_ledger.parquet",
        "t1_unavailable_ledger.parquet",
    ]
    forbidden: list[str] = []
    for filename in causal_files:
        frame = pd.read_parquet(root / filename)
        forbidden.extend(
            f"{filename}:{column}"
            for column in frame.columns
            if any(token in column.lower() for token in ["hindsight", "episode", "mfe", "mae"])
        )
    if forbidden:
        return False, f"evaluation-only fields entered causal ledgers: {forbidden}"
    schema = json.loads((root / "prospective_immutable_ledger_schema.json").read_text())
    if not schema["research_only"] or schema["execution_enabled"]:
        return False, "prospective schema crossed research boundary"
    if not schema["opportunity_ledger"]["immutable"]:
        return False, "prospective opportunities are mutable"
    if not schema["timing_ledger"]["create_only"] or not schema["outcome_ledger"]["create_only"]:
        return False, "timing or outcomes are not create-only"
    return True, "episode labels isolated; prospective ledgers immutable and execution-free"


def _check_missing_2023(root: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    report = json.loads((root / "missing_2023_input_report.json").read_text())
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_2023_hash_manifest"]["path"]).read_text()
    )["sha256"]
    expected = [value for key, value in manifest.items() if key.startswith("provider_2023_")]
    if len(expected) != 20 or report["pre_score_exact_hash_matches"] != 0:
        return False, "2023 archival hash status mismatch"
    if report["t1_outcomes_imputed"]:
        return False, "2023 outcome imputation enabled"
    return True, "twenty frozen 2023 hashes retained; zero archival matches and no imputation"


def _check_safety(contract: Mapping[str, Any]) -> tuple[bool, str]:
    start = str(contract["frozen_lineage"]["starting_commit"])
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{start}..HEAD"], cwd=REPO, text=True
    )
    changed = [line for line in output.splitlines() if line]
    prohibited = prohibited_changed_paths(changed)
    if prohibited:
        return False, f"prohibited runtime paths changed: {prohibited}"
    safety = contract["safety"]
    if not (
        safety["research_only"] is True
        and safety["broker_connection_enabled"] is False
        and safety["live_ordering_enabled"] is False
        and safety["position_management_changed"] is False
        and safety["existing_exit_logic_changed"] is False
        and safety["deployment_enabled"] is False
        and safety["application_runtime_changed"] is False
    ):
        return False, "contract safety flags drift"
    return True, f"{len(changed)} changed paths remain confined to research package/tests/reports"


def run_audit(primary: Path, exact: Path, output: Path) -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    checks: dict[str, tuple[bool, str]] = {
        "contract_and_input_identity": _check_contract_inputs(contract, metadata),
        "provider_hash_identity": _check_provider_hashes(contract, metadata),
        "named_and_control_population": _check_population(primary, contract),
        "t0_source_identity": _check_t0_against_v2(primary, contract),
        "exact_t1_clock_returns_and_costs": _check_t1_and_returns(primary, contract),
        "missing_data_pairing_and_no_replacement": _check_missing_and_pairing(primary),
        "paired_metrics_and_block_interval": _check_metrics(primary),
        "stress_concentration_and_horizon_separation": _check_stress_and_concentration(primary),
        "episode_isolation_and_prospective_safety": _check_episode_isolation_and_prospective_safety(
            primary
        ),
        "provider_2023_hash_status": _check_missing_2023(primary, contract),
        "research_only_changed_paths": _check_safety(contract),
    }
    exact_result = verify_exact_identity(primary, exact)
    checks["primary_exact_rerun_identity"] = (
        bool(exact_result["byte_identical"]),
        f"compared {exact_result['compared_files_including_plots']} machine and plot files",
    )
    passed = all(value[0] for value in checks.values())
    result = {
        "audit_id": "fixed_one_bar_entry_latency_v1_independent_audit",
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
