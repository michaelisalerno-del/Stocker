"""Independent checks for the M1C quiet-state concentration audit V0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260727-m1c-low-movement-short-premium-v0"
    / "artifacts"
    / "primary"
)
RUNTIME_MANIFEST = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts"
    / "primary"
    / "causal_movement_feature_manifest.json"
)
BOTTOM_5 = 0.115697407847643
BOTTOM_10 = 0.135896965695626
BOTTOM_20 = 0.167095528962669
ORIGINAL_DECISION = "blocked_insufficient_low_tail_support"


class IndependentAuditFailure(RuntimeError):
    """Raised when an independent reconstruction does not agree."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise IndependentAuditFailure(detail)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maximum_difference(left: Sequence[float], right: Sequence[float]) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    _require(first.shape == second.shape, "numeric comparison shapes differ")
    finite_first = np.isfinite(first)
    finite_second = np.isfinite(second)
    _require(
        bool(np.array_equal(finite_first, finite_second)),
        "numeric missing-value masks differ",
    )
    finite = finite_first & finite_second
    return float(np.max(np.abs(first[finite] - second[finite]))) if finite.any() else 0.0


def _manual_probabilities(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    features = [str(value) for value in specification["numeric_features"]]
    raw = frame[features].to_numpy(float)
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    numeric = (np.where(np.isfinite(raw), raw, medians) - means) / scales
    parts = [numeric]
    generated = list(features)
    for control in specification["category_controls"]:
        levels = [
            str(value)
            for value in cast(Mapping[str, Any], specification["category_levels"])[str(control)]
        ]
        observed = frame[str(control)].astype(str).to_numpy()
        for level in levels[1:]:
            parts.append((observed == level).astype(float)[:, None])
            generated.append(f"control_{control}__{level}")
    _require(
        generated == [str(value) for value in specification["design_columns"]],
        "manual M1C design-column order differs",
    )
    design = np.concatenate(parts, axis=1)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    linear = design @ coefficients + float(specification["intercept"])
    return np.asarray(
        1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))),
        dtype=float,
    )


def _evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    _require(len(frame) >= count, f"fewer than {count} rows are available")
    ordered = frame.sort_values(
        ["session", "stock", "checkpoint", "row_id"],
        kind="mergesort",
    )
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return ordered.iloc[indices].copy()


def _analytic_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(PREDECESSOR / "checkpoint_predictions.parquet")
    movement = pd.read_parquet(
        PREDECESSOR / "movement_outcomes.parquet",
        columns=[
            "row_id",
            "available_15m",
            "terminal_iv_residual_15m",
            "movement_remains_below_iv_15m",
        ],
    )
    excursion = pd.read_parquet(
        PREDECESSOR / "path_excursion_outcomes.parquet",
        columns=["row_id", "excursion_sigma_ratio_15m"],
    )
    analytic = predictions.merge(movement, on="row_id", validate="one_to_one").merge(
        excursion,
        on="row_id",
        validate="one_to_one",
    )
    analytic = analytic.loc[analytic["period"].isin(["assessment", "stress"])].copy()
    return predictions.loc[predictions["period"].isin(["assessment", "stress"])].copy(), analytic


def _independent_fresh_ids(predictions: pd.DataFrame) -> set[str]:
    output: set[str] = set()
    ordered = predictions.sort_values(
        ["stock", "session", "entry_timestamp", "checkpoint", "row_id"],
        kind="mergesort",
    ).copy()
    ordered["entry_timestamp"] = pd.to_datetime(
        ordered["entry_timestamp"],
        utc=True,
        errors="raise",
    )
    for _, group in ordered.groupby(["stock", "session"], sort=True):
        previous_probability: float | None = None
        previous_episode: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            probability = float(row.M1C_probability)
            timestamp = pd.Timestamp(row.entry_timestamp)
            crossing = probability <= BOTTOM_10 and (
                previous_probability is None or previous_probability > BOTTOM_10
            )
            spaced = previous_episode is None or timestamp - previous_episode >= pd.Timedelta(
                minutes=30
            )
            if crossing and spaced:
                output.add(str(row.row_id))
                previous_episode = timestamp
            previous_probability = probability
    return output


def _verify_runs(predictions: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    source = predictions.set_index(predictions["row_id"].astype(str), drop=False)
    observed_members: list[str] = []
    maximum_gap = 0.0
    for run in runs.itertuples(index=False):
        members = list(run.member_row_ids)
        _require(bool(members), "quiet run has no members")
        records = source.loc[[str(value) for value in members]].copy()
        _require(
            records["stock"].astype(str).nunique() == 1
            and records["session"].astype(str).nunique() == 1,
            "quiet run crosses stock or session",
        )
        _require(
            bool(records["M1C_probability"].le(BOTTOM_10).all()),
            "quiet run contains a non-tail row",
        )
        timestamps = pd.to_datetime(records["entry_timestamp"], utc=True, errors="raise")
        gaps = timestamps.sort_values().diff().dropna().dt.total_seconds().div(60.0)
        if not gaps.empty:
            maximum_gap = max(maximum_gap, float(gaps.max()))
            _require(bool(gaps.le(15.0).all()), "quiet run exceeds the 15-minute gap")
        observed_members.extend(str(value) for value in members)
    expected = set(
        predictions.loc[predictions["M1C_probability"].le(BOTTOM_10), "row_id"].astype(str)
    )
    _require(len(observed_members) == len(set(observed_members)), "tail row appears in two runs")
    _require(set(observed_members) == expected, "quiet runs do not partition the frozen tail")
    return {
        "runs_checked": int(len(runs)),
        "run_member_rows_checked": len(observed_members),
        "maximum_within_run_gap_minutes": maximum_gap,
        "passed": True,
    }


def _verify_surprise_clusters(
    analytic: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    sample_count = min(50, len(events))
    sample = _evenly_spaced(events, sample_count) if sample_count else events
    source = analytic.set_index(analytic["row_id"].astype(str), drop=False)
    maximum_member_count = 0
    for event in sample.itertuples(index=False):
        members = [str(value) for value in json.loads(str(event.member_row_ids))]
        records = source.loc[members].copy()
        threshold = float(event.sigma_threshold)
        _require(
            bool(records["excursion_sigma_ratio_15m"].ge(threshold).all()),
            "cluster contains a non-surprise member",
        )
        _require(
            records["stock"].astype(str).nunique() == 1
            and records["session"].astype(str).nunique() == 1,
            "surprise cluster crosses stock or session",
        )
        timestamps = pd.to_datetime(
            records["entry_timestamp"],
            utc=True,
            errors="raise",
        ).sort_values()
        previous: pd.Timestamp | None = None
        window_end: pd.Timestamp | None = None
        for timestamp in timestamps:
            if previous is not None:
                _require(
                    timestamp - previous <= pd.Timedelta(minutes=30),
                    "cluster trigger gap exceeds 30 minutes",
                )
                _require(
                    window_end is not None and timestamp <= window_end,
                    "cluster excursion windows do not overlap",
                )
            row_end = timestamp + pd.Timedelta(minutes=15)
            window_end = row_end if window_end is None else max(window_end, row_end)
            previous = timestamp
        maximum_member_count = max(maximum_member_count, len(members))
    return {
        "clustered_surprise_events_available": int(len(events)),
        "clustered_surprise_events_manually_checked": sample_count,
        "maximum_checked_member_count": maximum_member_count,
        "passed": True,
    }


def _weights(frame: pd.DataFrame, scheme: str) -> pd.Series:
    base = pd.to_numeric(frame["row_weight"], errors="raise")
    if scheme == "original":
        return base / float(base.sum())
    groups = {
        "equal_month": ["month"],
        "equal_stock": ["stock"],
        "equal_stock_month": ["stock", "month"],
    }[scheme]
    masses = frame.assign(_base=base).groupby(groups, sort=True)["_base"].transform("sum")
    return base / masses / frame.groupby(groups, sort=True).ngroups


def _weighted_quantile(
    values: pd.Series,
    weights: pd.Series,
    quantile: float,
) -> float:
    observed = values.to_numpy(float)
    mass = weights.to_numpy(float)
    order = np.argsort(observed, kind="mergesort")
    observed = observed[order]
    mass = mass[order]
    positions = (np.cumsum(mass) - 0.5 * mass) / mass.sum()
    return float(np.interp(quantile, positions, observed, left=observed[0], right=observed[-1]))


def _metric_values(frame: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    available = frame["available_15m"].astype(bool)
    support = frame.loc[available]
    mass = weights.loc[support.index]
    mass = mass / float(mass.sum())
    return {
        "remains_below_iv_rate": float(
            np.average(support["movement_remains_below_iv_15m"].astype(float), weights=mass)
        ),
        "mean_iv_residual": float(
            np.average(support["terminal_iv_residual_15m"].to_numpy(float), weights=mass)
        ),
        "median_iv_residual": _weighted_quantile(
            support["terminal_iv_residual_15m"],
            mass,
            0.5,
        ),
        "breach_1_5_sigma_rate": float(
            np.average(support["excursion_sigma_ratio_15m"].ge(1.5), weights=mass)
        ),
        "breach_2_0_sigma_rate": float(
            np.average(support["excursion_sigma_ratio_15m"].ge(2.0), weights=mass)
        ),
    }


def _verify_equal_exposure(
    analytic: pd.DataFrame,
    artifact: pd.DataFrame,
) -> dict[str, Any]:
    population = analytic.loc[analytic["period"].eq("stress")].copy()
    tail = population.loc[population["M1C_probability"].le(BOTTOM_10)].copy()
    maximum_difference = 0.0
    for scheme in ("original", "equal_month", "equal_stock", "equal_stock_month"):
        population_weights = _weights(population, scheme)
        tail_weights = population_weights.loc[tail.index]
        baseline = _metric_values(population, population_weights)
        metrics = _metric_values(tail, tail_weights)
        expected = {
            **metrics,
            "npv_lift": metrics["remains_below_iv_rate"] - baseline["remains_below_iv_rate"],
        }
        observed = artifact.loc[artifact["sensitivity"].eq(scheme)].iloc[0]
        for column, value in expected.items():
            maximum_difference = max(maximum_difference, abs(float(observed[column]) - value))
    _require(maximum_difference <= 1e-12, "equal-exposure sensitivity differs")
    return {
        "weighting_schemes_checked": 4,
        "maximum_floating_difference": maximum_difference,
        "passed": True,
    }


def _verify_leave_out(
    analytic: pd.DataFrame,
    month_artifact: pd.DataFrame,
    stock_artifact: pd.DataFrame,
) -> dict[str, Any]:
    tail = analytic.loc[analytic["period"].eq("stress") & analytic["M1C_probability"].le(BOTTOM_10)]
    for row in month_artifact.itertuples(index=False):
        expected = tail.loc[tail["month"].ne(str(row.omitted_month))]
        _require(len(expected) == int(row.rows), "leave-one-month row count differs")
    for row in stock_artifact.itertuples(index=False):
        expected = tail.loc[tail["stock"].ne(str(row.omitted_stock))]
        _require(len(expected) == int(row.rows), "leave-one-stock row count differs")
    return {
        "leave_one_month_cases_checked": int(len(month_artifact)),
        "leave_one_stock_cases_checked": int(len(stock_artifact)),
        "passed": True,
    }


def audit_primary_artifacts() -> dict[str, Any]:
    required = [
        PRIMARY / name
        for name in (
            "contract.json",
            "source_manifest.json",
            "original_decision_preservation.json",
            "reconstruction_audit.json",
            "stress_month_exposure_audit.csv",
            "stress_month_tail_incidence.csv",
            "stress_month_concentration_explanation.json",
            "quiet_state_runs.parquet",
            "surprise_mover_row_audit.csv",
            "surprise_event_clusters.csv",
            "surprise_concentration_metrics.csv",
            "surprise_concentration_explanation.json",
            "small_count_feasibility.json",
            "equal_exposure_sensitivities.csv",
            "leave_one_month_out.csv",
            "leave_one_stock_out.csv",
            "leave_one_checkpoint_group_out.csv",
            "decision.json",
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    _require(not missing, "primary audit artifacts are missing: " + ", ".join(missing))

    contract = _read_json(PRIMARY / "contract.json")
    decision = _read_json(PRIMARY / "decision.json")
    original = _read_json(PREDECESSOR / "decision.json")
    for payload in (contract, decision):
        _require(payload["research_only"] is True, "research-only claim changed")
        _require(payload["m1c_frozen"] is True, "M1C frozen claim changed")
        _require(float(payload["m1c_bottom_5_threshold"]) == BOTTOM_5, "bottom-5 changed")
        _require(float(payload["m1c_bottom_10_threshold"]) == BOTTOM_10, "bottom-10 changed")
        _require(float(payload["m1c_bottom_20_threshold"]) == BOTTOM_20, "bottom-20 changed")
        _require(payload["live_orders_allowed"] is False, "live orders were enabled")
        _require(payload["paper_orders_allowed"] is False, "paper orders were enabled")
    _require(
        original["overall_decision"] == decision["original_overall_decision"] == ORIGINAL_DECISION,
        "original decision was not preserved",
    )

    manifest = _read_json(RUNTIME_MANIFEST)
    predictions, analytic = _analytic_sources()
    original_tail = pd.read_parquet(PREDECESSOR / "raw_low_tail_checkpoint_rows.parquet")
    reconstructed_ids = set(
        predictions.loc[predictions["M1C_probability"].le(BOTTOM_10), "row_id"].astype(str)
    )
    original_ids = set(original_tail["row_id"].astype(str))
    _require(reconstructed_ids == original_ids, "frozen bottom-tail identities differ")
    sample = _evenly_spaced(
        predictions.loc[predictions["row_id"].astype(str).isin(original_ids)],
        100,
    )
    manual = _manual_probabilities(
        sample,
        cast(Mapping[str, Any], manifest["model_specification"]),
    )
    probability_difference = _maximum_difference(manual, sample["M1C_probability"])
    _require(probability_difference <= 1e-12, "manual M1C probabilities differ")

    fresh_source = pd.read_parquet(PREDECESSOR / "fresh_quiet_episodes.parquet")
    fresh_source = fresh_source.loc[
        fresh_source["tail"].eq("bottom_10_percent")
        & fresh_source["period"].isin(["assessment", "stress"])
    ]
    fresh_ids = _independent_fresh_ids(predictions)
    _require(
        fresh_ids == set(fresh_source["row_id"].astype(str)),
        "fresh quiet episode identities differ",
    )

    exposure = pd.read_csv(PRIMARY / "stress_month_exposure_audit.csv")
    incidence = pd.read_csv(PRIMARY / "stress_month_tail_incidence.csv")
    stress = analytic.loc[analytic["period"].eq("stress")]
    tail = stress.loc[stress["M1C_probability"].le(BOTTOM_10)]
    for row in exposure.itertuples(index=False):
        month = str(row.month)
        expected = stress.loc[stress["month"].eq(month)]
        _require(len(expected) == int(row.eligible_checkpoint_rows), "source exposure differs")
    for row in incidence.itertuples(index=False):
        month = str(row.month)
        expected = tail.loc[tail["month"].eq(month)]
        _require(len(expected) == int(row.m1c_bottom_10_rows), "month tail count differs")
        _require(
            abs(len(expected) / len(tail) - float(row.bottom_tail_composition_share)) <= 1e-12,
            "month composition differs",
        )
    october_share = float(
        incidence.loc[incidence["month"].eq("2025-10"), "bottom_tail_composition_share"].iloc[0]
    )
    _require(
        abs(october_share - 0.3709677419354839) <= 1e-15,
        "binding October share differs",
    )

    run_audit = _verify_runs(predictions, pd.read_parquet(PRIMARY / "quiet_state_runs.parquet"))
    event_audit = _verify_surprise_clusters(
        analytic,
        pd.read_csv(PRIMARY / "surprise_event_clusters.csv"),
    )
    surprise_rows = pd.read_csv(PRIMARY / "surprise_mover_row_audit.csv")
    _require(
        bool(surprise_rows["excursion_sigma_ratio_15m"].ge(1.5).all()),
        "surprise-row definition differs",
    )
    equal_audit = _verify_equal_exposure(
        analytic,
        pd.read_csv(PRIMARY / "equal_exposure_sensitivities.csv"),
    )
    leave_audit = _verify_leave_out(
        analytic,
        pd.read_csv(PRIMARY / "leave_one_month_out.csv"),
        pd.read_csv(PRIMARY / "leave_one_stock_out.csv"),
    )
    month_explanation = _read_json(PRIMARY / "stress_month_concentration_explanation.json")
    surprise_explanation = _read_json(PRIMARY / "surprise_concentration_explanation.json")
    _require(
        month_explanation["month_concentration_explanation"]
        == "month_concentration_has_multiple_causes",
        "month explanation differs",
    )
    _require(
        surprise_explanation["surprise_concentration_explanation"]
        == "surprise_concentration_is_small_count_fragile",
        "surprise explanation differs",
    )

    source_manifest = _read_json(PRIMARY / "source_manifest.json")
    source_hashes_checked = 0
    for source in source_manifest["sources"]:
        path = REPO_ROOT / str(source["path"])
        _require(path.is_file(), f"frozen source is missing: {path}")
        _require(_sha256(path) == source["sha256"], f"frozen source hash differs: {path}")
        source_hashes_checked += 1

    return {
        "research_only": True,
        "original_low_movement_decision_preserved": True,
        "original_decision": ORIGINAL_DECISION,
        "retrospective_gate_relaxation_allowed": False,
        "m1c_frozen": True,
        "m1c_bottom_5_threshold": BOTTOM_5,
        "m1c_bottom_10_threshold": BOTTOM_10,
        "m1c_bottom_20_threshold": BOTTOM_20,
        "primary_quiet_state": "bottom_10_percent",
        "prospective_record_only": True,
        "option_shadow_outcomes_only": True,
        "defined_risk_short_premium_only": True,
        "naked_short_options_allowed": False,
        "paper_orders_allowed": False,
        "live_orders_allowed": False,
        "broker_order_methods_allowed": False,
        "strategy_promotion": False,
        "protected_historical_start": "2026-01-01",
        "original_decision_verified": True,
        "source_hashes_checked": source_hashes_checked,
        "retrospective_bottom_tail_rows_manually_reconstructed": 100,
        "maximum_manual_m1c_probability_difference": probability_difference,
        "tail_membership_mismatches": 0,
        "fresh_episode_identity_mismatches": 0,
        "binding_october_share": october_share,
        "source_exposure_verified": True,
        "bottom_tail_incidence_verified": True,
        "quiet_run_audit": run_audit,
        "surprise_row_definition_verified": True,
        "surprise_event_audit": event_audit,
        "equal_exposure_audit": equal_audit,
        "leave_one_group_out_audit": leave_audit,
        "concentration_explanations_verified": True,
        "unexplained_discrepancies": 0,
        "passed": True,
    }


__all__ = ["IndependentAuditFailure", "audit_primary_artifacts"]
