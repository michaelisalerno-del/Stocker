# ruff: noqa: E501
"""Independent audit for Sequential Competitive Loop Exclusion V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT = WORK / "contracts/20260715-sequential-loop-competitor-veto-v1.json"
PRIMARY = WORK / "artifacts/20260715-sequential-loop-competitor-veto-v1/primary"
V2 = WORK / "artifacts/20260714-dynamic-loop-edge-state-v2/primary"
TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cycle_states(cycle: str) -> tuple[int, ...]:
    states = tuple(int(value) for value in cycle.split("->"))
    if len(states) < 3 or states[0] != states[-1]:
        raise ValueError(f"invalid closed cycle: {cycle}")
    return states


def _independent_status(cycle: str, anchor_state: int, observed: tuple[int, ...]) -> str:
    states = _cycle_states(cycle)
    open_path = states[:-1]
    rotations: list[tuple[int, ...]] = []
    for index, state in enumerate(open_path):
        if state == anchor_state:
            rotated = open_path[index:] + open_path[:index]
            rotations.append(rotated + (rotated[0],))
    if not rotations:
        return "impossible"
    compatible = False
    for path in rotations:
        expected = path[1:]
        comparable = observed[: len(expected)]
        if comparable != expected[: len(comparable)]:
            continue
        if len(observed) >= len(expected):
            return "completed"
        compatible = True
    return "compatible" if compatible else "impossible"


def _safe(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


class Auditor:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, passed: object, detail: object = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": _safe(detail)})


def _load(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _source_paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "scoring_2023": Path(contract["inputs"]["scoring_predictions"]["2023"]["path"]),
        "scoring_2025": Path(contract["inputs"]["scoring_predictions"]["2025"]["path"]),
        "accepted_signal_ledger": Path(contract["inputs"]["accepted_signal_ledger"]["path"]),
        "execution_anchors_2023": Path(contract["inputs"]["execution_anchors"]["2023"]["path"]),
        "execution_anchors_2025": Path(contract["inputs"]["execution_anchors"]["2025"]["path"]),
    }


def _check_source_identity(audit: Auditor, contract: Mapping[str, Any]) -> None:
    paths = _source_paths(contract)
    expected = {
        "scoring_2023": contract["inputs"]["scoring_predictions"]["2023"]["sha256"],
        "scoring_2025": contract["inputs"]["scoring_predictions"]["2025"]["sha256"],
        "accepted_signal_ledger": contract["inputs"]["accepted_signal_ledger"]["sha256"],
        "execution_anchors_2023": contract["inputs"]["execution_anchors"]["2023"]["sha256"],
        "execution_anchors_2025": contract["inputs"]["execution_anchors"]["2025"]["sha256"],
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    audit.check("data_snapshot_identity", actual == expected, actual)


def _check_classification(audit: Auditor, root: Path) -> None:
    exported = _load(root, "training_only_loop_payoff_classifications.parquet")
    raw = pd.read_parquet(V2 / "causal_edge_state_forecasts.parquet")
    raw = raw.loc[raw["model_name"].eq("hierarchical_payoff_history_change_point")].copy()
    support = (
        raw["effective_sessions"].ge(8)
        & raw["independent_stocks"].ge(5)
        & raw["effective_sample_size"].ge(12.0)
        & raw["posterior_std_net_bps"].le(80.0)
    )
    upper = raw["posterior_mean_net_bps"] + 1.6448536269514722 * raw["posterior_std_net_bps"]
    expected = np.select(
        [
            support
            & raw["posterior_mean_net_bps"].gt(0.0)
            & raw["posterior_lower_bound_net_bps"].gt(0.0),
            support & upper.lt(0.0),
        ],
        ["good", "bad"],
        default="unknown",
    )
    keys = ["period", "score_session", "loop_id", "orientation", "horizon"]
    lookup = raw[keys].copy()
    lookup["expected_class"] = expected
    compared = exported.merge(lookup, on=keys, how="inner", validate="one_to_one")
    audit.check(
        "good_bad_unknown_classification_rebuilt",
        len(compared) == len(exported)
        and compared["payoff_class"].eq(compared["expected_class"]).all(),
        {"rows": len(compared)},
    )
    decision = pd.to_datetime(exported["decision_timestamp"], utc=True)
    training = pd.to_datetime(
        exported["training_latest_availability_timestamp"], utc=True, errors="coerce"
    )
    audit.check(
        "training_cutoffs_strict",
        training[training.notna()].lt(decision[training.notna()]).all(),
    )


def _check_structural_posterior(audit: Auditor, root: Path) -> None:
    posterior = _load(root, "sequential_checkpoint_posterior_ledger.parquet")
    anchor_sets = _load(root, "anchor_compatible_loop_sets.parquet")
    timeline = _load(root, "good_bad_unknown_mass_timeline.parquet")
    group_total = posterior.groupby(["checkpoint_id", "track"])["posterior_probability"].sum()
    audit.check(
        "posterior_normalisation",
        np.allclose(group_total.to_numpy(float), 1.0, atol=TOLERANCE, rtol=0.0),
        {"groups": len(group_total), "max_error": float((group_total - 1.0).abs().max())},
    )
    independently_summarised_unknown = (
        posterior.loc[posterior["payoff_class"].eq("unknown")]
        .groupby(["checkpoint_id", "track"])["posterior_probability"]
        .sum()
        .rename("rebuilt_unknown_mass")
        .reset_index()
    )
    unknown_join = timeline.merge(
        independently_summarised_unknown,
        on=["checkpoint_id", "track"],
        how="left",
        validate="one_to_one",
    )
    audit.check(
        "explicit_unknown_mass",
        np.allclose(
            unknown_join["unknown_loop_mass"],
            unknown_join["rebuilt_unknown_mass"],
            atol=TOLERANCE,
            rtol=0.0,
        ),
    )
    cycles = (
        anchor_sets[["loop_id", "cycle"]].drop_duplicates().set_index("loop_id")["cycle"].to_dict()
    )
    sample_groups = list(posterior.groupby(["checkpoint_id", "track"], sort=False))[:500]
    status_ok = True
    equation_ok = True
    anchor_lookup = {
        opportunity: group.set_index("loop_id")["initial_posterior_probability"].to_dict()
        for opportunity, group in anchor_sets.groupby("opportunity_id", sort=False)
    }
    for (checkpoint_id_value, track_value), group in sample_groups:
        checkpoint_id = str(checkpoint_id_value)
        track = str(track_value)
        known = group.loc[~group["loop_id"].eq("__unknown__")].copy()
        row = timeline.loc[
            timeline["checkpoint_id"].eq(checkpoint_id) & timeline["track"].eq(track)
        ].iloc[0]
        observed = tuple(int(value) for value in json.loads(row["observed_transitions_json"]))
        anchor_state = int(row["current_state"])
        prior = anchor_lookup[str(row["opportunity_id"])]
        weights: dict[str, float] = {}
        for candidate_value in known.itertuples(index=False):
            candidate: Any = candidate_value
            expected_status = _independent_status(
                str(cycles[candidate.loop_id]), anchor_state, observed
            )
            status_ok = status_ok and expected_status == str(candidate.compatibility_status)
            if expected_status == "impossible":
                weights[str(candidate.loop_id)] = 0.0
            else:
                likelihood = max(
                    0.05,
                    float(candidate.timing_likelihood) * float(candidate.score_likelihood_ratio),
                )
                weights[str(candidate.loop_id)] = float(prior[candidate.loop_id]) * likelihood
        unknown_prior = max(0.0, 1.0 - sum(float(value) for value in prior.values()))
        denominator = sum(weights.values()) + unknown_prior
        expected_unknown = unknown_prior / denominator if denominator > 0.0 else 1.0
        actual_unknown = float(
            group.loc[group["loop_id"].eq("__unknown__"), "posterior_probability"].iloc[0]
        )
        equation_ok = equation_ok and math.isclose(
            expected_unknown, actual_unknown, abs_tol=TOLERANCE, rel_tol=0.0
        )
        for candidate_value in known.itertuples(index=False):
            evaluated: Any = candidate_value
            expected_probability = weights[str(evaluated.loop_id)] / denominator
            equation_ok = equation_ok and math.isclose(
                expected_probability,
                float(evaluated.posterior_probability),
                abs_tol=TOLERANCE,
                rel_tol=0.0,
            )
    audit.check("transition_compatibility_rebuilt", status_ok, {"groups": len(sample_groups)})
    audit.check("posterior_equation_rebuilt", equation_ok, {"groups": len(sample_groups)})
    audit.check(
        "smoothed_nonzero_likelihoods",
        posterior.loc[~posterior["loop_id"].eq("__unknown__"), "timing_likelihood"].ge(0.05).all(),
    )
    availability = pd.to_datetime(timeline["feature_max_availability_timestamp"], utc=True)
    freeze = pd.to_datetime(timeline["checkpoint_timestamp"], utc=True)
    audit.check("checkpoint_feature_availability", availability.le(freeze).all())
    audit.check(
        "event_lineage_integrity",
        timeline.groupby("opportunity_id")["event_lineage_id"].nunique().eq(1).all(),
    )
    forbidden = [
        column
        for column in posterior.columns
        if "hindsight" in column.lower()
        or "realised_loop" in column.lower()
        or "realized_loop" in column.lower()
        or "future_state" in column.lower()
    ]
    audit.check("future_and_episode_inputs_absent", not forbidden, forbidden)


def _check_remaining_payoff(audit: Auditor, root: Path, contract: Mapping[str, Any]) -> None:
    outcomes = _load(root, "constant_terminal_remaining_payoff_outcomes.parquet")
    available = outcomes.loc[outcomes["outcome_status"].eq("available")].copy()
    entry = pd.to_datetime(available["remaining_entry_timestamp"], utc=True)
    checkpoint = pd.to_datetime(available["checkpoint_timestamp"], utc=True)
    terminal = pd.to_datetime(available["terminal_timestamp"], utc=True)
    audit.check("remaining_payoff_clock", entry.ge(checkpoint).all() and entry.lt(terminal).all())
    audit.check(
        "constant_terminal_costs",
        np.allclose(
            available["constant_terminal_gross_bps"] - available["constant_terminal_net_bps"],
            10.0,
            atol=TOLERANCE,
            rtol=0.0,
        ),
    )
    restarted = _load(root, "restarted_horizon_sensitivity_outcomes.parquet")
    audit.check(
        "restarted_horizon_separate",
        "constant_terminal_net_bps" not in restarted.columns
        and "restarted_net_bps" in restarted.columns,
    )
    unavailable = outcomes.loc[outcomes["outcome_status"].ne("available")]
    audit.check(
        "missing_and_too_late_not_zero",
        unavailable["constant_terminal_net_bps"].isna().all(),
    )

    source = pd.read_parquet(
        contract["inputs"]["accepted_signal_ledger"]["path"],
        columns=["anchor_id", "period", "strategy", "horizon", "direction"],
    )
    source = source.loc[
        source["strategy"].eq("breakout_loop_scores_range_p75") & source["horizon"].eq(24)
    ].copy()
    source["period"] = source["period"].astype(int)
    source = source.drop_duplicates(["period", "anchor_id"])
    sample = available.loc[available["period"].eq(2025)].head(50).copy()
    sample["period"] = sample["period"].astype(int)
    if sample.empty:
        audit.check("independent_2025_price_replay", False, "no 2025 rows")
        return
    sample["anchor_id"] = sample["opportunity_id"].str.split("-").str[1].astype(int)
    sample = sample.merge(source, on=["period", "anchor_id"], how="left", validate="many_to_one")
    replay_ok = True
    root_2025 = Path(contract["inputs"]["provider_2025_root"])
    for (stock, session), group in sample.groupby(["stock", "session_date"], sort=False):
        bars = pd.read_parquet(
            root_2025 / f"symbol={stock}" / "timeframe=5m" / "data.parquet",
            columns=["timestamp", "open", "close"],
        )
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        local_session = bars["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        bars = bars.loc[local_session.eq(str(session))].sort_values("timestamp", kind="stable")
        for row in group.itertuples(index=False):
            checkpoint_time = pd.Timestamp(str(row.checkpoint_timestamp))
            terminal_time = pd.Timestamp(str(row.terminal_timestamp))
            entry_row = bars.loc[bars["timestamp"].ge(checkpoint_time)].iloc[0]
            terminal_rows = bars.loc[
                (bars["timestamp"] + pd.Timedelta(minutes=5)).eq(terminal_time)
            ]
            if len(terminal_rows) != 1:
                replay_ok = False
                continue
            gross = (
                10_000.0
                * int(str(row.direction))
                * (float(terminal_rows.iloc[0]["close"]) / float(entry_row["open"]) - 1.0)
            )
            replay_ok = replay_ok and math.isclose(
                gross - 10.0,
                float(str(row.constant_terminal_net_bps)),
                abs_tol=1e-8,
                rel_tol=0.0,
            )
    audit.check("independent_2025_price_replay", replay_ok, {"rows": len(sample)})


def _check_accounting(audit: Auditor, root: Path) -> None:
    accounting = _load(root, "veto_accounting.parquet")
    general = accounting.loc[accounting["track"].eq("track_b_prior_only")]
    policy_counts = general.groupby("opportunity_id")["policy"].nunique()
    audit.check("paired_fixed_population", policy_counts.eq(6).all(), {"rows": len(general)})
    audit.check(
        "no_replacement_opportunities",
        accounting["replacement_opportunity_id"].isna().all(),
    )
    audit.check(
        "existing_positions_unchanged",
        accounting["existing_position_action"].eq("unchanged").all(),
    )
    delayed = _load(root, "sequential_delayed_admission_decisions.parquet")
    audit.check(
        "delayed_admission_constant_terminal_label",
        delayed["economic_clock"].eq("constant_terminal_next_open").all(),
    )
    comparator = _load(root, "static_anchor_comparator_predictions.parquet")
    audit.check(
        "static_sequential_paired_identity",
        not comparator["opportunity_id"].duplicated().any(),
    )


def _check_stress_rebuilds(audit: Auditor, root: Path) -> None:
    loo = _load(root, "leave_one_stock_out_results.csv")
    rebuilt = (
        not loo.empty
        and loo["excluded_stock"].nunique() >= 20
        and loo["all_stock_dependent_inputs_rebuilt"].fillna(False).astype(bool).all()
    )
    audit.check("leave_one_stock_out_rebuilt", rebuilt, {"rows": len(loo)})
    stress = _load(root, "stress_test_results.csv")
    required = {
        "twice_costs",
        "one_additional_bar_execution_delay",
        "median_session_aggregation",
        "minimum_two_bar_state_dwell",
        "coarse_clock_bins",
        "prior_smoothing_alpha_0.5",
        "prior_smoothing_alpha_2.0",
    }
    present = set(stress["stress_test"].astype(str))
    audit.check("predeclared_stress_family", required <= present, sorted(required - present))


def _check_evaluation_attribution(audit: Auditor, root: Path) -> None:
    comparator = _load(root, "static_anchor_comparator_predictions.parquet")
    concentration = _load(root, "concentration_results.csv")
    primary = concentration.loc[
        concentration["analysis_scope"].eq("primary_paired_economic_increment")
    ]
    row_contribution = comparator["target_remaining_net_bps"] * (
        comparator["sequential_probability"] - comparator["anchor_probability"]
    )
    expected_total = float(row_contribution.sum(min_count=1))
    stock_total = float(
        primary.loc[primary["dimension"].eq("stock"), "net_contribution_bps"].sum(min_count=1)
    )
    audit.check(
        "primary_concentration_rebuilt",
        math.isclose(expected_total, stock_total, abs_tol=TOLERANCE, rel_tol=0.0),
        {"expected_total": expected_total, "stock_total": stock_total},
    )

    census = _load(root, "pairwise_target_competitor_census.parquet")
    anchor = _load(root, "anchor_compatible_loop_sets.parquet")
    eliminated = _load(root, "loop_elimination_events.parquet")
    named = comparator.loc[comparator["population_role"].eq("named_target")][
        ["opportunity_id", "target_loop", "original_net_payoff_bps"]
    ].copy()
    rows = anchor.merge(named, on="opportunity_id", how="inner", validate="many_to_one")
    rows = rows.loc[~rows["loop_id"].eq(rows["target_loop"])].copy()
    first = (
        eliminated.sort_values(["opportunity_id", "checkpoint_timestamp", "loop_id"], kind="stable")
        .drop_duplicates(["opportunity_id", "loop_id"], keep="first")[
            ["opportunity_id", "loop_id", "bars_consumed"]
        ]
        .rename(columns={"bars_consumed": "elimination_bars"})
    )
    rows = rows.merge(first, on=["opportunity_id", "loop_id"], how="left", validate="one_to_one")
    rows["profitable"] = rows["original_net_payoff_bps"].gt(0.0)
    rebuilt: list[dict[str, object]] = []
    for keys, group in rows.groupby(["target_loop", "loop_id", "payoff_class"], sort=True):
        observed = group["elimination_bars"].notna()
        profitable = group["profitable"]
        rebuilt.append(
            {
                "target_loop": keys[0],
                "competitor_loop": keys[1],
                "competitor_payoff_class": keys[2],
                "rebuilt_profitable_rate": float(observed.loc[profitable].mean()),
                "rebuilt_losing_rate": float(observed.loc[~profitable].mean()),
                "rebuilt_profitable_median": float(
                    group.loc[observed & profitable, "elimination_bars"].median()
                ),
                "rebuilt_losing_median": float(
                    group.loc[observed & ~profitable, "elimination_bars"].median()
                ),
            }
        )
    joined = census.merge(
        pd.DataFrame(rebuilt),
        on=["target_loop", "competitor_loop", "competitor_payoff_class"],
        how="inner",
        validate="one_to_one",
    )
    comparisons = [
        ("profitable_target_elimination_rate", "rebuilt_profitable_rate"),
        ("losing_target_elimination_rate", "rebuilt_losing_rate"),
        ("profitable_target_median_elimination_bars", "rebuilt_profitable_median"),
        ("losing_target_median_elimination_bars", "rebuilt_losing_median"),
    ]
    timing_ok = len(joined) == len(census) and all(
        np.allclose(joined[left], joined[right], atol=TOLERANCE, rtol=0.0, equal_nan=True)
        for left, right in comparisons
    )
    audit.check("profitable_losing_elimination_timing_rebuilt", timing_ok, {"rows": len(joined)})


def _check_artifacts(audit: Auditor, root: Path, exact: Path | None) -> None:
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    actual = {
        name: sha256(root / name)
        for name in manifest
        if (root / name).is_file() and name != "independent_audit.json"
    }
    expected = {name: value for name, value in manifest.items() if name in actual}
    audit.check("machine_readable_and_plot_hashes", actual == expected, {"files": len(actual)})
    if exact is None:
        audit.check("primary_exact_rerun_identity", False, "exact rerun root not supplied")
        return
    extensions = {".parquet", ".csv", ".json"}
    excluded = {"artifact_manifest.json", "independent_audit.json", "exact_rerun_identity.json"}
    names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix in extensions and path.name not in excluded
    )
    exact_names = sorted(
        path.name
        for path in exact.iterdir()
        if path.is_file() and path.suffix in extensions and path.name not in excluded
    )
    mismatches = [
        name
        for name in set(names) & set(exact_names)
        if sha256(root / name) != sha256(exact / name)
    ]
    audit.check(
        "primary_exact_rerun_identity",
        names == exact_names and not mismatches,
        {"files": len(names), "mismatches": sorted(mismatches)},
    )


def _check_metadata_and_safety(audit: Auditor, root: Path, contract: Mapping[str, Any]) -> None:
    metadata = json.loads((root / "run_metadata.json").read_text())
    audit.check("contract_identity", metadata["contract_hash"] == sha256(CONTRACT))
    audit.check(
        "git_and_configuration_metadata",
        bool(metadata.get("git_sha"))
        and metadata.get("model_version") == contract["experiment_version"]
        and metadata.get("fixed_horizon_bars") == 24,
    )
    census = _load(root, "regime_clock_loop_census.parquet")
    training_ok = census["training_latest_session"].isna() | (
        census["training_latest_session"].astype(str) < census["score_session"].astype(str)
    )
    audit.check("clock_census_past_only", training_ok.all())
    timeline = _load(root, "good_bad_unknown_mass_timeline.parquet")
    periods = timeline.groupby("event_lineage_id")["period"].nunique()
    audit.check("period_boundaries", periods.eq(1).all())
    safety = metadata["safety"]
    audit.check(
        "research_only_safety_flags",
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["broker_connection_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["position_or_frozen_exit_logic_changed"] is False,
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", contract["frozen_lineage"]["starting_commit"], "HEAD"],
        cwd=REPO,
        text=True,
    ).splitlines()
    forbidden_roots = (
        "packages/stocker_execution/",
        "apps/",
        "deployment/",
        "broker/",
        "orders/",
        "positions/",
    )
    forbidden = sorted(path for path in changed if path.startswith(forbidden_roots))
    audit.check("no_runtime_execution_path_changed", not forbidden, forbidden)


def run_audit(root: Path, exact: Path | None, output: Path) -> dict[str, object]:
    contract = json.loads(CONTRACT.read_text())
    audit = Auditor()
    _check_source_identity(audit, contract)
    _check_metadata_and_safety(audit, root, contract)
    _check_classification(audit, root)
    _check_structural_posterior(audit, root)
    _check_remaining_payoff(audit, root, contract)
    _check_accounting(audit, root)
    _check_stress_rebuilds(audit, root)
    _check_evaluation_attribution(audit, root)
    _check_artifacts(audit, root, exact)
    passed = sum(bool(check["passed"]) for check in audit.checks)
    result: dict[str, object] = {
        "auditor_version": "sequential_loop_competitor_veto_independent_audit_v1",
        "status": "pass" if passed == len(audit.checks) else "fail",
        "passed_checks": passed,
        "total_checks": len(audit.checks),
        "checks": audit.checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=PRIMARY)
    parser.add_argument("--exact-rerun-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root)
    output = Path(args.output) if args.output else root / "independent_audit.json"
    result = run_audit(root, args.exact_rerun_root, output)
    print(json.dumps({key: result[key] for key in ("status", "passed_checks", "total_checks")}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
