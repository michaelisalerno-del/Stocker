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


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    for actual, expected, label in [
        (named, expected_named, "named"),
        (controls, expected_controls, "control"),
    ]:
        joined = actual.set_index("opportunity_id").join(
            expected.set_index("opportunity_id")[
                [
                    "event_lineage_id",
                    "period",
                    "session_date",
                    "stock",
                    "target_loop",
                    "orientation",
                ]
            ],
            how="left",
            rsuffix="_policy",
            validate="one_to_one",
        )
        identity_checks = {
            "event_lineage_id": joined["event_lineage_id"]
            .astype(str)
            .eq(joined["event_lineage_id_policy"].astype(str)),
            "period": joined["period"].astype(str).eq(joined["period_policy"].astype(str)),
            "session": joined["session_date"]
            .astype(str)
            .eq(joined["session_date_policy"].astype(str)),
            "symbol": joined["symbol"].astype(str).eq(joined["stock"].astype(str)),
            "loop": joined["loop_id"].astype(str).eq(joined["target_loop"].astype(str)),
            "orientation": joined["orientation"]
            .astype(str)
            .eq(joined["orientation_policy"].astype(str)),
        }
        failed = [name for name, values in identity_checks.items() if not bool(values.all())]
        if failed:
            return False, f"{label} policy identity mismatch: {failed}"
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
    field_checks = {
        "anchor_id": t0["anchor_id"].astype(str).eq(source["anchor_id"].astype(str)),
        "symbol": t0["symbol"].astype(str).eq(source["symbol_norm"].astype(str)),
        "period": t0["period"].astype(str).eq(source["period"].astype(str)),
        "session": t0["session_date"].astype(str).eq(source["session_date"].astype(str)),
        "loop": t0["loop_id"].astype(str).eq(source["loop_id"].astype(str)),
        "orientation": t0["orientation"].astype(str).eq(source["orientation"].astype(str)),
        "direction": t0["direction"].astype(int).eq(source["direction"].astype(int)),
    }
    failed_fields = [name for name, values in field_checks.items() if not bool(values.all())]
    if failed_fields:
        return False, f"T0 identity differs from raw V2: {failed_fields}"
    numeric_checks = [
        np.allclose(t0["original_terminal_price"], source["exit_price"], rtol=0.0, atol=1e-12),
        np.allclose(
            t0["original_gross_payoff_bps"], source["gross_payoff_bps"], rtol=0.0, atol=1e-10
        ),
        np.allclose(
            t0["original_total_cost_bps"], source["primary_total_cost_bps"], rtol=0.0, atol=1e-12
        ),
        np.allclose(
            t0["original_net_payoff_bps"], source["primary_net_payoff_bps"], rtol=0.0, atol=1e-10
        ),
    ]
    if not all(numeric_checks):
        return False, "T0 terminal price, gross, cost, or net differs from raw V2"
    expected_hashes = [
        stable_hash(
            {
                "opportunity_id": opportunity_id,
                "anchor_id": row["anchor_id"],
                "direction": row["direction"],
                "entry_timestamp": row["original_entry_timestamp"],
                "entry_price": row["original_entry_price"],
                "terminal_timestamp": row["original_terminal_timestamp"],
                "terminal_price": row["original_terminal_price"],
            }
        )
        for opportunity_id, row in t0.iterrows()
    ]
    if expected_hashes != t0["source_opportunity_hash"].astype(str).tolist():
        return False, "source opportunity hashes do not independently reconstruct"
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
        if not math.isclose(_as_float(row.terminal_price), terminal_price, abs_tol=1e-12):
            return False, f"stored terminal price mismatch: {row.opportunity_id}"
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


def _check_breakdowns(root: Path) -> tuple[bool, str]:
    paired = pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet")
    metrics = pd.read_csv(root / "primary_paired_metrics.csv")
    checked = 0
    for dimension in ["period", "loop_id", "direction_label"]:
        for value, group in paired.groupby(dimension, dropna=False, sort=True):
            observed = metrics.loc[
                metrics["slice_type"].eq(dimension)
                & metrics["slice_value"].astype(str).eq(str(value))
            ]
            if len(observed) != 1:
                return False, f"missing or duplicate {dimension} breakdown: {value}"
            row = observed.iloc[0]
            expected = {
                "paired_opportunities": len(group),
                "t0_net_payoff_bps": float(group["t0_net_return_bps"].sum()),
                "t1_net_payoff_bps": float(group["t1_net_return_bps"].sum()),
                "paired_total_difference_bps": float(group["paired_difference_bps"].sum()),
            }
            for column, expected_value in expected.items():
                if not math.isclose(float(row[column]), float(expected_value), abs_tol=1e-8):
                    return False, f"{dimension} breakdown mismatch: {value}:{column}"
            checked += 1
    controls = pd.read_parquet(root / "control_exact_paired_t0_t1_ledger.parquet")
    control_metrics = pd.read_csv(root / "control_breakdowns.csv")
    for value, group in controls.groupby("orientation", sort=True):
        observed = control_metrics.loc[
            control_metrics["slice_type"].eq("orientation")
            & control_metrics["slice_value"].astype(str).eq(str(value))
        ]
        if len(observed) != 1:
            return False, f"missing control orientation breakdown: {value}"
        if not math.isclose(
            float(observed.iloc[0]["paired_total_difference_bps"]),
            float(group["paired_difference_bps"].sum()),
            abs_tol=1e-8,
        ):
            return False, f"control orientation breakdown mismatch: {value}"
        checked += 1
    return True, f"independently reconstructed {checked} period/loop/direction/control breakdowns"


def _check_deletions_and_nulls(root: Path) -> tuple[bool, str]:
    paired = pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet")
    deletion = pd.read_csv(root / "deletion_stress_results.csv")
    stock = paired.groupby("symbol")["paired_difference_bps"].sum().sort_values(ascending=False)
    episodes = (
        paired.loc[paired["hindsight_episode_id"].notna()]
        .groupby("hindsight_episode_id")["paired_difference_bps"]
        .sum()
        .sort_values(ascending=False)
    )
    removals = {
        "remove_best_stock": ("symbol", list(stock.head(1).index.astype(str))),
        "remove_top_five_stocks": ("symbol", list(stock.head(5).index.astype(str))),
        "remove_best_episode": (
            "hindsight_episode_id",
            list(episodes.head(1).index.astype(str)),
        ),
        "remove_top_five_episodes": (
            "hindsight_episode_id",
            list(episodes.head(5).index.astype(str)),
        ),
    }
    for name, (dimension, contributors) in removals.items():
        observed = deletion.loc[deletion["stress"].eq(name)]
        if len(observed) != 1:
            return False, f"missing deletion stress: {name}"
        expected_rows = paired.loc[~paired[dimension].astype("string").isin(contributors)]
        row = observed.iloc[0]
        if str(row["removed_contributors"]) != "|".join(contributors):
            return False, f"deletion contributor mismatch: {name}"
        if not math.isclose(
            float(row["paired_total_difference_bps"]),
            float(expected_rows["paired_difference_bps"].sum()),
            abs_tol=1e-8,
        ):
            return False, f"deletion delta mismatch: {name}"

    nulls = pd.read_csv(root / "null_test_results.csv")
    actual = paired["paired_difference_bps"].to_numpy(float)
    rng = np.random.default_rng(20260716)
    draws = np.array(
        [float((actual * rng.integers(0, 2, size=len(actual))).sum()) for _ in range(500)]
    )
    random_row = nulls.loc[nulls["null_test"].eq("random_T0_or_T1_timing_500_repetitions")]
    if len(random_row) != 1 or not math.isclose(
        float(random_row.iloc[0]["null_mean_increment_bps"]), float(draws.mean()), abs_tol=1e-8
    ):
        return False, "random timing null does not reconstruct"

    shifted = paired.sort_values(
        ["period", "symbol", "session_date", "opportunity_id"], kind="stable"
    ).copy()
    shifted["ratio"] = shifted["t1_entry_price"] / shifted["t0_entry_price"]
    groups = shifted.groupby(["period", "symbol"], sort=False)
    shifted["prior_ratio"] = groups["ratio"].shift(1)
    shifted["prior_date"] = groups["session_date"].shift(1)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    first_symbol = str(paired["symbol"].iloc[0])
    provider = _provider(contract, first_symbol)
    calendar = sorted(
        {
            str(value)
            for value in provider["timestamp"].dt.tz_convert("America/New_York").dt.date.unique()
        }
    )
    previous_session = {
        session: calendar[index - 1] if index > 0 else None
        for index, session in enumerate(calendar)
    }
    shifted["expected_prior_date"] = shifted["session_date"].astype(str).map(previous_session)
    valid = shifted.loc[
        shifted["prior_ratio"].notna()
        & shifted["prior_date"].astype(str).eq(shifted["expected_prior_date"].astype(str))
    ].copy()
    null_price = valid["t0_entry_price"] * valid["prior_ratio"]
    expected_shift = (
        10_000.0
        * valid["direction"]
        * valid["terminal_price"]
        * (1.0 / null_price - 1.0 / valid["t0_entry_price"])
    )
    shifted_row = nulls.loc[nulls["null_test"].eq("prior_session_entry_displacement")]
    if len(shifted_row) != 1 or not math.isclose(
        float(shifted_row.iloc[0]["shifted_increment_bps"]),
        float(expected_shift.sum()),
        abs_tol=1e-8,
    ):
        return False, "prior-session displacement null does not reconstruct"
    return True, "deletion contributors and two predeclared nulls independently reconstruct"


def _check_stress_and_concentration(root: Path) -> tuple[bool, str]:
    paired = pd.read_parquet(root / "exact_paired_t0_t1_ledger.parquet")
    controls = pd.read_parquet(root / "control_exact_paired_t0_t1_ledger.parquet")
    costs = pd.read_csv(root / "cost_stress_results.csv")
    cost_rows_checked = 0
    for population, frame in [("named", paired), ("control", controls)]:
        slices: list[tuple[str, str, pd.DataFrame]] = [("all", "all", frame)]
        for dimension in ["period", "loop_id", "orientation"]:
            slices.extend(
                (dimension, str(value), group)
                for value, group in frame.groupby(dimension, dropna=False, sort=True)
            )
        for slice_type, slice_value, group in slices:
            for stress, cost in [("frozen_costs", 10.0), ("twice_costs", 20.0)]:
                observed = costs.loc[
                    costs["population"].eq(population)
                    & costs["slice_type"].eq(slice_type)
                    & costs["slice_value"].astype(str).eq(slice_value)
                    & costs["stress"].eq(stress)
                ]
                if len(observed) != 1:
                    return (
                        False,
                        f"cost stress row missing: {population}:{slice_type}:{slice_value}",
                    )
                t0 = float((group["t0_gross_return_bps"] - cost).sum())
                t1 = float((group["t1_gross_return_bps"] - cost).sum())
                row = observed.iloc[0]
                if not all(
                    [
                        math.isclose(float(row["t0_net_payoff_bps"]), t0, abs_tol=1e-8),
                        math.isclose(float(row["t1_net_payoff_bps"]), t1, abs_tol=1e-8),
                        math.isclose(float(row["paired_difference_bps"]), t1 - t0, abs_tol=1e-8),
                    ]
                ):
                    return False, f"cost stress mismatch: {population}:{slice_type}:{slice_value}"
                cost_rows_checked += 1
    concentration = pd.read_csv(root / "concentration_results.csv")
    concentration_rows_checked = 0
    for dimension in concentration["dimension"].drop_duplicates().astype(str):
        contribution = paired.groupby(dimension, dropna=False)["paired_difference_bps"].sum()
        denominator = float(contribution.abs().sum())
        shares = contribution.abs() / denominator if denominator > 0.0 else contribution * np.nan
        expected_top_one = float(shares.max()) if len(shares) else np.nan
        expected_top_five = float(shares.sort_values(ascending=False).head(5).sum())
        expected_hhi = float(np.square(shares).sum()) if denominator > 0.0 else np.nan
        observed_dimension = concentration.loc[concentration["dimension"].eq(dimension)]
        if len(observed_dimension) != len(contribution):
            return False, f"concentration row count mismatch: {dimension}"
        for contributor, value in contribution.items():
            observed = observed_dimension.loc[
                observed_dimension["contributor"].astype(str).eq(str(contributor))
            ]
            if len(observed) != 1:
                return False, f"concentration contributor missing: {dimension}:{contributor}"
            concentration_row = observed.iloc[0]
            expected_share = float(shares.get(contributor, np.nan))
            checks = [
                math.isclose(
                    float(concentration_row["contribution_bps"]), float(value), abs_tol=1e-8
                ),
                math.isclose(
                    float(concentration_row["absolute_contribution_share"]),
                    expected_share,
                    abs_tol=1e-10,
                ),
                math.isclose(
                    float(concentration_row["top_one_absolute_share"]),
                    expected_top_one,
                    abs_tol=1e-10,
                ),
                math.isclose(
                    float(concentration_row["top_five_absolute_share"]),
                    expected_top_five,
                    abs_tol=1e-10,
                ),
                math.isclose(float(concentration_row["herfindahl"]), expected_hhi, abs_tol=1e-10),
            ]
            if not all(checks):
                return False, f"concentration mismatch: {dimension}:{contributor}"
            concentration_rows_checked += 1
    loo = pd.read_csv(root / "leave_one_stock_out_results.csv")
    for symbol in sorted(paired["symbol"].astype(str).unique()):
        observed = loo.loc[loo["excluded_symbol"].astype(str).eq(symbol)]
        remaining = paired.loc[paired["symbol"].astype(str).ne(symbol)]
        if len(observed) != 1 or not math.isclose(
            float(observed.iloc[0]["paired_total_difference_bps"]),
            float(remaining["paired_difference_bps"].sum()),
            abs_tol=1e-8,
        ):
            return False, f"leave-one-stock-out mismatch: {symbol}"
    deletion = pd.read_csv(root / "deletion_stress_results.csv")
    threshold = paired.groupby("period")["dollar_volume_proxy"].transform("median")
    liquid = paired.loc[paired["dollar_volume_proxy"].ge(threshold)]
    liquid_row = deletion.loc[deletion["stress"].eq("minimum_liquidity_within_period_median")]
    if len(liquid_row) != 1 or not math.isclose(
        float(liquid_row.iloc[0]["paired_total_difference_bps"]),
        float(liquid["paired_difference_bps"].sum()),
        abs_tol=1e-8,
    ):
        return False, "minimum-liquidity stress mismatch"
    t2 = pd.read_parquet(root / "t2_sensitivity_ledger.parquet")
    if "t1_net_return_bps" in t2.columns or "paired_difference_bps" in t2.columns:
        return False, "T2 table was not kept separate from T1 naming"
    available_t2 = t2.loc[t2["t2_available"] & t2["population"].eq("named")].copy()
    providers: dict[str, pd.DataFrame] = {}
    for t2_row in available_t2.itertuples(index=False):
        symbol = str(t2_row.symbol)
        if symbol not in providers:
            contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
            providers[symbol] = _provider(contract, symbol)
        expected = _as_timestamp(t2_row.original_entry_timestamp) + pd.Timedelta(minutes=10)
        if _as_timestamp(t2_row.t2_entry_timestamp) != expected:
            return False, f"T2 clock mismatch: {t2_row.opportunity_id}"
        provider_row = providers[symbol].loc[providers[symbol]["timestamp"].eq(expected)]
        if len(provider_row) != 1:
            return False, f"T2 provider open missing: {t2_row.opportunity_id}"
        t2_price = float(provider_row.iloc[0]["open"])
        if not math.isclose(_as_float(t2_row.t2_entry_price), t2_price, abs_tol=1e-12):
            return False, f"T2 price mismatch: {t2_row.opportunity_id}"
        t2_gross = (
            10_000.0
            * _as_int(t2_row.direction)
            * (_as_float(t2_row.terminal_price) / t2_price - 1.0)
        )
        if not math.isclose(_as_float(t2_row.t2_net_return_bps), t2_gross - 10.0, abs_tol=1e-8):
            return False, f"T2 net mismatch: {t2_row.opportunity_id}"
        if not math.isclose(
            _as_float(t2_row.t2_minus_t0_bps),
            (t2_gross - 10.0) - _as_float(t2_row.t0_net_return_bps),
            abs_tol=1e-8,
        ):
            return False, f"T2 minus T0 mismatch: {t2_row.opportunity_id}"
    t2_metrics = pd.read_csv(root / "t2_sensitivity_metrics.csv").iloc[0]
    if not math.isclose(
        float(t2_metrics["t2_minus_t0_bps"]),
        float(available_t2["t2_minus_t0_bps"].sum()),
        abs_tol=1e-8,
    ):
        return False, "T2 summary mismatch"
    if not math.isclose(
        float(t2_metrics["mean_exposure_bars_remaining"]),
        float(available_t2["exposure_bars_remaining"].mean()),
        abs_tol=1e-10,
    ):
        return False, "T2 exposure summary mismatch"
    restarted = pd.read_parquet(root / "restarted_h24_diagnostic_ledger.parquet")
    if "paired_difference_bps" in restarted.columns:
        return False, "restarted h24 contaminated the primary paired table"
    detail = (
        f"{cost_rows_checked} cost rows, {concentration_rows_checked} concentration rows, "
        f"LOO/liquidity, {len(available_t2)} T2 rows, and restarted separation verified"
    )
    return True, detail


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
    commands = [
        ["git", "diff", "--name-only", f"{start}..HEAD"],
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    changed = sorted(
        {
            line
            for command in commands
            for line in subprocess.check_output(command, cwd=REPO, text=True).splitlines()
            if line
        }
    )
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
        "period_loop_direction_and_control_breakdowns": _check_breakdowns(primary),
        "deletion_and_null_reconstruction": _check_deletions_and_nulls(primary),
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
