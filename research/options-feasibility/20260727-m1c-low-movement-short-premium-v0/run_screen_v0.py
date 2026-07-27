#!/usr/bin/env python3
"""Run Frozen Causal M1C Low-Movement and Containment Screen V0."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits must precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.m1c_low_movement_v0 import (
    HORIZONS_MINUTES,
    TAIL_QUANTILES,
    assert_unprotected_sessions,
    assign_frozen_bins,
    calculate_checkpoint_outcomes,
    choose_overall_decision,
    construct_fresh_quiet_episodes,
    evaluate_low_movement_veto_gate,
    evaluate_short_premium_readiness_gate,
    evaluate_support_gate,
    freeze_weighted_boundaries,
    matched_random_selection,
    permute_probabilities_within_sessions,
    reconstruct_frozen_probabilities,
    tail_memberships,
    tail_overlap,
    validate_causal_features,
    weighted_quantile,
    whole_session_bootstrap_plan,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import GROUP_O

CLAIMS: dict[str, bool | str | int] = {
    "research_only": True,
    "retrospective_low_movement_screen": True,
    "m1c_frozen": True,
    "m1c_causal_feature_surface": True,
    "archived_contaminated_m1_excluded": True,
    "cross_sectional_future_filtered_features_excluded": True,
    "low_tail_thresholds_fit_on_2024_only": True,
    "primary_low_tail": "bottom_10_percent",
    "primary_horizon_minutes": 15,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "short_option_pnl_claim": False,
    "defined_risk_structures_only_for_future_recording": True,
    "naked_short_options_authorised": False,
    "broker_access": False,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "strategy_promotion": False,
    "protected_start": "2026-01-01",
}

DEVELOPMENT_START = "2024-01-01"
DEVELOPMENT_END = "2024-12-31"
ASSESSMENT_START = "2025-01-01"
ASSESSMENT_END = "2025-08-22"
STRESS_START = "2025-09-01"
STRESS_END = "2025-12-31"
CHECKPOINTS = tuple(range(6, 35, 2))
PRIMARY_HORIZON = 15
HIGH_M1C_THRESHOLD = 0.488333710794033
MATCHED_SEEDS = tuple(range(2026072701, 2026072721))
PERMUTATION_SEEDS = tuple(range(2026072731, 2026072741))
BOOTSTRAP_SEEDS = {"assessment": 2026072751, "stress": 2026072752}
BOOTSTRAP_DRAWS = 100
DECILE_QUANTILES = tuple(index / 10.0 for index in range(1, 10))
IV_QUARTILE_QUANTILES = (0.25, 0.50, 0.75)


class ScreenBlocked(RuntimeError):
    """Fail-closed experiment blocker."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_identity(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = pd.util.hash_pandas_object(frame.loc[:, list(columns)], index=False).to_numpy(
        np.uint64
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = cast(
        dict[str, Any],
        json.loads((EXPERIMENT_DIR / "contract.json").read_text(encoding="utf-8")),
    )
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in CLAIMS.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"contract mismatch: {mismatches}",
        )
    if (
        contract.get("checkpoint_grid") != list(CHECKPOINTS)
        or contract.get("matched_random_draws") != 20
        or contract.get("probability_permutations") != 10
        or contract.get("bootstrap_draws") != 100
    ):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "frozen grid or resampling counts drifted",
        )
    return contract


def load_predecessor_runner() -> ModuleType:
    path = PREDECESSOR_DIR / "run_screen_v0.py"
    specification = importlib.util.spec_from_file_location("frozen_archetype_runner_v0", path)
    if specification is None or specification.loader is None:
        raise ScreenBlocked(
            "blocked_m1c_reconstruction_failure",
            "the predecessor runner could not be loaded",
        )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _normalise_period(session: object) -> str:
    value = str(session)
    if DEVELOPMENT_START <= value <= DEVELOPMENT_END:
        return "development"
    if ASSESSMENT_START <= value <= ASSESSMENT_END:
        return "assessment"
    if STRESS_START <= value <= STRESS_END:
        return "stress"
    raise ScreenBlocked(
        "blocked_chronology_or_leakage_failure",
        f"session outside the frozen experiment periods: {value}",
    )


def _option_metadata(predecessor: ModuleType) -> pd.DataFrame:
    columns = [
        "symbol",
        "session",
        "required_options_date",
        "options_observation_date",
        "previous_close_underlying_price",
        "front_expiration_date",
        "front_strike",
        "front_call_contract_id",
        "front_put_contract_id",
        "atm_iv",
    ]
    historical = pd.read_parquet(predecessor.HISTORICAL_OPTIONS_PATH, columns=columns)
    stress_columns = [*columns, "pair_available"]
    stress = pd.read_parquet(predecessor.STRESS_OPTIONS_PATH, columns=stress_columns)
    stress = stress.loc[stress["pair_available"].astype(bool), columns]
    metadata = pd.concat([historical, stress], ignore_index=True).rename(
        columns={"symbol": "stock"}
    )
    metadata["session"] = metadata["session"].astype(str)
    if metadata.duplicated(["stock", "session"]).any():
        raise ScreenBlocked(
            "blocked_previous_close_iv_failure",
            "previous-close option metadata is not unique by stock-session",
        )
    for row in metadata.itertuples(index=False):
        session = pd.Timestamp(row.session).date()
        required = pd.Timestamp(row.required_options_date).date()
        observed = pd.Timestamp(row.options_observation_date).date()
        if required != observed or observed >= session:
            raise ScreenBlocked(
                "blocked_previous_close_iv_failure",
                "an option pair did not use the exact required previous session",
            )
    metadata["option_dte"] = (
        pd.to_datetime(metadata["front_expiration_date"], errors="raise")
        - pd.to_datetime(metadata["session"], errors="raise")
    ).dt.days
    atm_iv = pd.to_numeric(metadata["atm_iv"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(atm_iv).all()
        or bool((atm_iv <= 0.0).any())
        or bool(metadata["option_dte"].le(0).any())
    ):
        raise ScreenBlocked(
            "blocked_previous_close_iv_failure",
            "previous-close ATM IV or option DTE is invalid",
        )
    return metadata


def reconstruct_m1c_panel(
    predecessor: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    historical, stress, states, predecessor_sources = predecessor.load_inputs()
    assert_unprotected_sessions(historical["session"])
    assert_unprotected_sessions(stress["session"])
    assert_unprotected_sessions(states["session"])
    m0, m1c, high_threshold, scored_historical, _, dependency_audit = predecessor.phase_zero(
        historical, stress
    )
    stress_scored = stress.copy()
    stress_scored["M0_probability"] = m0.predict(stress_scored)
    stress_scored["M1C_probability"] = m1c.predict(stress_scored)
    panel = pd.concat([scored_historical, stress_scored], ignore_index=True)
    panel["session"] = panel["session"].astype(str)
    panel["period"] = panel["session"].map(_normalise_period)
    panel = panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    if set(panel["checkpoint"].astype(int)) != set(CHECKPOINTS):
        raise ScreenBlocked(
            "blocked_m1c_reconstruction_failure",
            "the eligible checkpoint grid drifted",
        )
    if panel.duplicated(["stock", "session", "checkpoint"]).any():
        raise ScreenBlocked(
            "blocked_m1c_reconstruction_failure",
            "checkpoint identities are not unique",
        )

    predecessor_manifest = cast(
        dict[str, Any],
        json.loads(
            (PREDECESSOR_PRIMARY / "causal_movement_feature_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    m1c_spec = cast(Mapping[str, object], predecessor_manifest["model_specification"])
    m0_spec = cast(Mapping[str, object], predecessor_manifest["m0_model_specification"])
    causal_group_i = tuple(str(value) for value in predecessor_manifest["causally_valid_group_i"])
    forbidden = tuple(
        str(value)
        for value in (
            *predecessor_manifest["removed_future_contaminated_group_i"],
            *predecessor_manifest["removed_other_peer_normalised_group_i"],
        )
    )
    expected_features = (*GROUP_O, *causal_group_i)
    validate_causal_features(expected_features, forbidden=forbidden)
    if tuple(str(value) for value in m1c_spec["numeric_features"]) != expected_features:
        raise ScreenBlocked(
            "blocked_m1c_reconstruction_failure",
            "the serialized M1C feature order drifted",
        )

    manual_m1c = reconstruct_frozen_probabilities(panel, m1c_spec)
    manual_m0 = reconstruct_frozen_probabilities(panel, m0_spec)
    refit_m1c = pd.to_numeric(panel["M1C_probability"], errors="raise").to_numpy(float)
    refit_m0 = pd.to_numeric(panel["M0_probability"], errors="raise").to_numpy(float)
    maximum_m1c_difference = float(np.max(np.abs(refit_m1c - manual_m1c)))
    maximum_m0_difference = float(np.max(np.abs(refit_m0 - manual_m0)))

    dense = pd.read_parquet(
        predecessor.DENSE_CAUSAL_PATH,
        columns=["row_id", *causal_group_i],
    )
    historical_comparison = panel.loc[
        panel["period"].isin(["development", "assessment"]),
        ["row_id", *causal_group_i],
    ].merge(
        dense,
        on="row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_reconstructed", "_frozen"),
    )
    feature_row_mismatches = int(historical_comparison["_merge"].ne("both").sum())
    feature_differences = [
        np.abs(
            pd.to_numeric(
                historical_comparison[f"{feature}_reconstructed"], errors="coerce"
            ).to_numpy(float)
            - pd.to_numeric(historical_comparison[f"{feature}_frozen"], errors="coerce").to_numpy(
                float
            )
        )
        for feature in causal_group_i
    ]
    maximum_feature_difference = float(np.nanmax(np.concatenate(feature_differences)))

    expected_ids = set(historical["row_id"].astype(str)).union(stress["row_id"].astype(str))
    actual_ids = set(panel["row_id"].astype(str))
    row_identity_mismatches = len(expected_ids.symmetric_difference(actual_ids))
    expected_weights = 1.0 / panel.groupby(["stock", "session"], sort=False)[
        "checkpoint"
    ].transform("size").to_numpy(float)
    observed_weights = pd.to_numeric(panel["row_weight"], errors="raise").to_numpy(float)
    weight_mismatches = int(
        np.count_nonzero(~np.isclose(expected_weights, observed_weights, atol=0.0, rtol=0.0))
    )

    episodes = pd.read_parquet(
        PREDECESSOR_PRIMARY / "movement_signal_episodes.parquet",
        columns=["row_id", "M1C_probability"],
    )
    existing_comparison = episodes.merge(
        panel[["row_id", "M1C_probability"]],
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_existing", "_reconstructed"),
        indicator=True,
    )
    existing_row_mismatches = int(existing_comparison["_merge"].ne("both").sum())
    existing_probability_difference = float(
        np.nanmax(
            np.abs(
                existing_comparison["M1C_probability_existing"].to_numpy(float)
                - existing_comparison["M1C_probability_reconstructed"].to_numpy(float)
            )
        )
    )
    if (
        maximum_feature_difference > 1e-12
        or maximum_m1c_difference > 1e-12
        or maximum_m0_difference > 1e-12
        or existing_probability_difference > 1e-12
        or feature_row_mismatches
        or row_identity_mismatches
        or weight_mismatches
        or existing_row_mismatches
        or abs(float(high_threshold) - HIGH_M1C_THRESHOLD) > 1e-12
    ):
        raise ScreenBlocked(
            "blocked_m1c_reconstruction_failure",
            "causal M1C reconstruction exceeded the frozen tolerance",
        )

    metadata = _option_metadata(predecessor)
    panel = panel.drop(
        columns=[
            column
            for column in (
                "required_options_date",
                "options_observation_date",
                "atm_iv",
            )
            if column in panel
        ]
    ).merge(metadata, on=["stock", "session"], how="left", validate="many_to_one")
    if (
        panel[
            [
                "required_options_date",
                "options_observation_date",
                "front_expiration_date",
                "atm_iv",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ScreenBlocked(
            "blocked_previous_close_iv_failure",
            "a checkpoint lost its exact previous-close option pair",
        )

    probability_comparison = panel[
        ["row_id", "stock", "session", "period", "checkpoint", "row_weight"]
    ].copy()
    probability_comparison["expected_m1c_probability"] = manual_m1c
    probability_comparison["reconstructed_m1c_probability"] = refit_m1c
    probability_comparison["absolute_probability_difference"] = np.abs(manual_m1c - refit_m1c)
    write_csv(PRIMARY / "m1c_probability_comparison.csv", probability_comparison)
    reconstruction = {
        **CLAIMS,
        "status": "supported",
        "predecessor_experiment": str(PREDECESSOR_DIR.relative_to(REPO_ROOT)),
        "feature_order_exact": True,
        "coefficients_exact": True,
        "intercept_exact": True,
        "preprocessing_exact": True,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": maximum_m1c_difference,
        "maximum_m0_probability_difference": maximum_m0_difference,
        "maximum_existing_episode_probability_difference": existing_probability_difference,
        "row_identity_mismatches": row_identity_mismatches,
        "existing_episode_row_identity_mismatches": existing_row_mismatches,
        "weight_mismatches": weight_mismatches,
        "panel_start": str(panel["session"].min()),
        "panel_end": str(panel["session"].max()),
        "panel_rows": int(len(panel)),
        "sessions": int(panel["session"].nunique()),
        "stocks": int(panel["stock"].nunique()),
        "checkpoints": sorted(panel["checkpoint"].astype(int).unique().tolist()),
        "high_tail_threshold_unchanged": float(high_threshold),
    }
    feature_manifest = {
        **CLAIMS,
        "model": "M1C",
        "group_o": list(GROUP_O),
        "causally_valid_group_i": list(causal_group_i),
        "feature_order": list(expected_features),
        "contaminated_and_peer_normalised_features_excluded": list(forbidden),
        "replacement_features_added": [],
        "model_specification": dict(m1c_spec),
        "m0_model_specification": dict(m0_spec),
        "candidate_normalised_weight": "1 / eligible checkpoints in stock-session",
    }
    write_json(PRIMARY / "m1c_reconstruction.json", reconstruction)
    write_json(PRIMARY / "m1c_feature_manifest.json", feature_manifest)
    source_manifest = {
        **CLAIMS,
        "sources": predecessor_sources["sources"],
        "predecessor_runner": {
            "path": str((PREDECESSOR_DIR / "run_screen_v0.py").relative_to(REPO_ROOT)),
            "sha256": sha256_file(PREDECESSOR_DIR / "run_screen_v0.py"),
        },
        "predecessor_feature_manifest": {
            "path": str(
                (PREDECESSOR_PRIMARY / "causal_movement_feature_manifest.json").relative_to(
                    REPO_ROOT
                )
            ),
            "sha256": sha256_file(PREDECESSOR_PRIMARY / "causal_movement_feature_manifest.json"),
        },
        "exact_date_eodhd_receipts_inspected_via_frozen_manifests": True,
        "options_downloaded": False,
        "network_requests": 0,
        "broker_access": False,
        "protected_rows_read": 0,
        "maximum_session_read": str(states["session"].astype(str).max()),
    }
    return (
        panel,
        states,
        source_manifest,
        {
            "reconstruction": reconstruction,
            "feature_manifest": feature_manifest,
            "dependency_audit": dependency_audit,
        },
    )


def checkpoint_group(checkpoint: object) -> str:
    value = int(checkpoint)
    if 6 <= value <= 14:
        return "early"
    if 16 <= value <= 24:
        return "middle"
    if 26 <= value <= 34:
        return "late"
    raise ValueError(f"checkpoint outside frozen grid: {value}")


def _checkpoint_context(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    bars = states.copy()
    bars["session"] = bars["session"].astype(str)
    for (stock, session), group in bars.groupby(["stock", "session"], sort=False):
        ordered = group.sort_values("bar_ordinal", kind="mergesort").set_index("bar_ordinal")
        for checkpoint in CHECKPOINTS:
            ordinals = list(range(checkpoint - 6, checkpoint))
            session_prefix_ordinals = list(range(checkpoint))
            if not all(ordinal in ordered.index for ordinal in session_prefix_ordinals):
                continue
            prefix = ordered.loc[ordinals]
            session_prefix = ordered.loc[session_prefix_ordinals]
            stock_returns = pd.to_numeric(prefix["bar_log_return"], errors="raise").to_numpy(float)
            market_returns = pd.to_numeric(prefix["vti__bar_log_return"], errors="raise").to_numpy(
                float
            )
            market_prefix_returns = pd.to_numeric(
                session_prefix["vti__bar_log_return"], errors="raise"
            ).to_numpy(float)
            if not np.isfinite(stock_returns).all():
                continue
            market_volatility = (
                float(np.std(market_returns, ddof=0))
                if np.isfinite(market_returns).all()
                else math.nan
            )
            market_return = (
                float(np.sum(market_prefix_returns))
                if np.isfinite(market_prefix_returns).all()
                else math.nan
            )
            records.append(
                {
                    "stock": str(stock),
                    "session": str(session),
                    "checkpoint": checkpoint,
                    "stock_local_volatility": float(np.std(stock_returns, ddof=0)),
                    "market_volatility": market_volatility,
                    "market_return_through_checkpoint": market_return,
                }
            )
    context = pd.DataFrame(records)
    output = panel.merge(
        context,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    if output["stock_local_volatility"].isna().any():
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "a checkpoint lost its completed-bar stock-local volatility context",
        )

    # VTI occasionally has a missing completed return while the eligible stock
    # checkpoint and every frozen model input remain intact. Preserve the exact
    # binding population and freeze deterministic checkpoint-level fallback
    # values from 2024 only for these optional subgroup stratifiers.
    development = output.loc[output["period"].eq("development")]
    for column in ("market_volatility", "market_return_through_checkpoint"):
        output[f"{column}_fallback_used"] = False
        output[f"{column}_fallback_level"] = "none"
        finite_development = development.loc[
            np.isfinite(pd.to_numeric(development[column], errors="coerce"))
        ]
        if finite_development.empty:
            raise ScreenBlocked(
                "blocked_chronology_or_leakage_failure",
                f"2024 has no causal fallback support for {column}",
            )
        pooled_fallback = weighted_quantile(
            finite_development[column].to_numpy(float),
            finite_development["row_weight"].to_numpy(float),
            0.5,
        )
        checkpoint_fallbacks: dict[int, float] = {}
        for checkpoint, checkpoint_rows in finite_development.groupby(
            "checkpoint",
            sort=True,
        ):
            checkpoint_fallbacks[int(checkpoint)] = weighted_quantile(
                checkpoint_rows[column].to_numpy(float),
                checkpoint_rows["row_weight"].to_numpy(float),
                0.5,
            )
        missing = ~np.isfinite(pd.to_numeric(output[column], errors="coerce"))
        for row_index in output.index[missing]:
            checkpoint = int(output.at[row_index, "checkpoint"])
            if checkpoint in checkpoint_fallbacks:
                output.at[row_index, column] = checkpoint_fallbacks[checkpoint]
                output.at[row_index, f"{column}_fallback_level"] = "2024_checkpoint_weighted_median"
            else:
                output.at[row_index, column] = pooled_fallback
                output.at[row_index, f"{column}_fallback_level"] = "2024_pooled_weighted_median"
            output.at[row_index, f"{column}_fallback_used"] = True
        if not np.isfinite(pd.to_numeric(output[column], errors="coerce")).all():
            raise ScreenBlocked(
                "blocked_chronology_or_leakage_failure",
                f"causal 2024-only fallback failed for {column}",
            )
    return output


def freeze_thresholds_and_context(
    panel: pd.DataFrame,
    movement: pd.DataFrame,
    paths: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    output = panel.merge(
        movement,
        on=["row_id", "stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    ).merge(
        paths.drop(columns=["entry_timestamp", "entry_price", "available_horizons"]),
        on=["row_id", "stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_path"),
    )
    development = output.loc[output["period"].eq("development")]
    weights = development["row_weight"].to_numpy(float)
    thresholds: dict[str, dict[str, float]] = {}
    for model, column in (
        ("M1C", "M1C_probability"),
        ("M0", "M0_probability"),
    ):
        thresholds[model] = {
            label: weighted_quantile(
                development[column].to_numpy(float),
                weights,
                quantile,
            )
            for label, quantile in TAIL_QUANTILES.items()
        }
        memberships = tail_memberships(output[column], thresholds[model])
        for label in TAIL_QUANTILES:
            output[f"{model.lower()}_{label}"] = memberships[label].to_numpy(bool)

    deciles = freeze_weighted_boundaries(
        development["M1C_probability"].to_numpy(float),
        weights,
        quantiles=DECILE_QUANTILES,
    )
    output["m1c_score_decile"] = assign_frozen_bins(
        output["M1C_probability"].to_numpy(float),
        deciles,
    )
    iv_quartiles = freeze_weighted_boundaries(
        development["atm_iv"].to_numpy(float),
        weights,
        quantiles=IV_QUARTILE_QUANTILES,
    )
    output["atm_iv_quartile"] = assign_frozen_bins(
        output["atm_iv"].to_numpy(float),
        iv_quartiles,
    )
    finite_development_market_movement = development.loc[
        np.isfinite(
            pd.to_numeric(
                development["market_maximum_absolute_movement_15m"],
                errors="coerce",
            )
        )
    ]
    if finite_development_market_movement.empty:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "2024 has no causal market-movement support for subgroup freezing",
        )
    subgroup_boundaries = {
        "previous_close_atm_iv_median": weighted_quantile(
            development["atm_iv"].to_numpy(float), weights, 0.50
        ),
        "stock_local_volatility_median": weighted_quantile(
            development["stock_local_volatility"].to_numpy(float), weights, 0.50
        ),
        "market_volatility_median": weighted_quantile(
            development["market_volatility"].to_numpy(float), weights, 0.50
        ),
        "market_large_move_15m_q90": weighted_quantile(
            finite_development_market_movement["market_maximum_absolute_movement_15m"].to_numpy(
                float
            ),
            finite_development_market_movement["row_weight"].to_numpy(float),
            0.90,
        ),
        "previous_close_atm_iv_quartiles": list(iv_quartiles),
    }
    output["previous_close_atm_iv_group"] = np.where(
        output["atm_iv"].le(subgroup_boundaries["previous_close_atm_iv_median"]),
        "low",
        "high",
    )
    output["stock_local_volatility_group"] = np.where(
        output["stock_local_volatility"].le(subgroup_boundaries["stock_local_volatility_median"]),
        "low",
        "high",
    )
    output["market_volatility_group"] = np.where(
        output["market_volatility"].le(subgroup_boundaries["market_volatility_median"]),
        "low",
        "high",
    )
    output["market_quiet_active"] = np.where(
        output["market_volatility_group"].eq("low"),
        "quiet",
        "active",
    )
    output["market_up_down"] = np.where(
        output["market_return_through_checkpoint"].ge(0.0),
        "up",
        "down",
    )
    market_movement_available = np.isfinite(
        pd.to_numeric(output["market_maximum_absolute_movement_15m"], errors="coerce")
    )
    output["market_movement_context_available"] = market_movement_available
    market_movement_unusually_large = pd.Series(
        pd.NA,
        index=output.index,
        dtype="boolean",
    )
    market_movement_unusually_large.loc[market_movement_available] = output.loc[
        market_movement_available,
        "market_maximum_absolute_movement_15m",
    ].ge(subgroup_boundaries["market_large_move_15m_q90"])
    output["market_movement_unusually_large"] = market_movement_unusually_large
    output["checkpoint_group"] = output["checkpoint"].map(checkpoint_group)
    output["month"] = output["session"].str[:7]
    output["option_dte_group"] = np.select(
        [
            output["option_dte"].between(7, 10, inclusive="both"),
            output["option_dte"].between(11, 16, inclusive="both"),
            output["option_dte"].gt(16),
        ],
        ["7-10 DTE", "11-16 DTE", "more_than_16_DTE"],
        default="unsupported_DTE",
    )
    threshold_artifact = {
        **CLAIMS,
        "fit_period": "2024 only",
        "method": "candidate-normalised weighted midpoint-CDF",
        "M1C": thresholds["M1C"],
        "M0": thresholds["M0"],
        "binding_tail": "M1C probability <= M1C bottom_10_percent threshold",
        "subgroup_boundaries": subgroup_boundaries,
    }
    decile_artifact = {
        **CLAIMS,
        "fit_period": "2024 only",
        "method": "candidate-normalised weighted midpoint-CDF",
        "boundaries": list(deciles),
        "application": "fixed inclusive upper boundaries applied unchanged to both 2025 periods",
    }
    write_json(PRIMARY / "frozen_low_tail_thresholds.json", threshold_artifact)
    write_json(PRIMARY / "frozen_score_deciles.json", decile_artifact)
    return output, threshold_artifact, decile_artifact


def _weights(frame: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    column = "_analysis_weight" if "_analysis_weight" in frame else "row_weight"
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    if len(values) == 0 or not np.isfinite(values).all() or bool((values <= 0.0).any()):
        raise ValueError("analysis weights must be finite and positive")
    return values


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = _weights(frame)
    valid = np.isfinite(values)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_rate(frame: pd.DataFrame, column: str) -> float:
    return weighted_mean(frame, column)


def weighted_median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = _weights(frame)
    valid = np.isfinite(values)
    if not valid.any():
        return math.nan
    return weighted_quantile(values[valid], weights[valid], 0.50)


def weighted_trimmed_mean(frame: pd.DataFrame, column: str, proportion: float = 0.10) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = _weights(frame)
    valid = np.isfinite(values)
    if not valid.any():
        return math.nan
    lower = weighted_quantile(values[valid], weights[valid], proportion)
    upper = weighted_quantile(values[valid], weights[valid], 1.0 - proportion)
    retained = valid & (values >= lower) & (values <= upper)
    return float(np.average(values[retained], weights=weights[retained]))


def _weighted_percentile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = _weights(frame)
    valid = np.isfinite(values)
    if not valid.any():
        return math.nan
    return weighted_quantile(values[valid], weights[valid], quantile)


def _share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    return float(frame.groupby(column, sort=True).size().max() / len(frame))


def summary_metrics(frame: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> dict[str, Any]:
    available_column = f"available_{horizon}m"
    available = frame.loc[frame[available_column].astype(bool)].copy()
    if available.empty:
        return {
            "rows": 0,
            "sessions": 0,
            "stocks": 0,
            "months": 0,
        }
    absolute = f"absolute_return_{horizon}m"
    expected = f"iv_expected_absolute_{horizon}m"
    residual = f"terminal_iv_residual_{horizon}m"
    ratio = f"terminal_iv_ratio_{horizon}m"
    excursion = f"maximum_absolute_excursion_{horizon}m"
    excursion_ratio = f"excursion_sigma_ratio_{horizon}m"
    remains = f"movement_remains_below_iv_{horizon}m"
    exceeds = f"movement_exceeds_iv_{horizon}m"
    result: dict[str, Any] = {
        "rows": int(len(available)),
        "sessions": int(available["session"].nunique()),
        "stocks": int(available["stock"].nunique()),
        "months": int(available["session"].astype(str).str[:7].nunique()),
        "positive_movement_exceeds_iv_outcomes": int(available[exceeds].astype(bool).sum()),
        "negative_movement_remains_below_iv_outcomes": int(available[remains].astype(bool).sum()),
        "maximum_stock_share": _share(available, "stock"),
        "maximum_month_share": float(
            available["session"].astype(str).str[:7].value_counts().max() / len(available)
        ),
        "maximum_session_share": _share(available, "session"),
        "remains_below_iv_rate": weighted_rate(available, remains),
        "movement_exceeds_iv_rate": weighted_rate(available, exceeds),
        "mean_absolute_movement": weighted_mean(available, absolute),
        "median_absolute_movement": weighted_median(available, absolute),
        "mean_iv_expectation": weighted_mean(available, expected),
        "median_iv_expectation": weighted_median(available, expected),
        "mean_iv_residual": weighted_mean(available, residual),
        "median_iv_residual": weighted_median(available, residual),
        "trimmed_mean_iv_residual_10_percent": weighted_trimmed_mean(available, residual),
        "mean_terminal_iv_ratio": weighted_mean(available, ratio),
        "median_terminal_iv_ratio": weighted_median(available, ratio),
        "absolute_movement_p90": _weighted_percentile(available, absolute, 0.90),
        "absolute_movement_p95": _weighted_percentile(available, absolute, 0.95),
        "absolute_movement_p99": _weighted_percentile(available, absolute, 0.99),
        "maximum_absolute_movement": float(available[absolute].max()),
        "mean_maximum_absolute_excursion": weighted_mean(available, excursion),
        "median_maximum_absolute_excursion": weighted_median(available, excursion),
        "mean_excursion_sigma_ratio": weighted_mean(available, excursion_ratio),
    }
    for multiple in (1.0, 1.25, 1.5, 2.0):
        label = str(multiple).replace(".", "_")
        terminal_breach = available[ratio].to_numpy(float) > multiple
        excursion_breach = available[excursion_ratio].to_numpy(float) > multiple
        weights = _weights(available)
        result[f"terminal_above_{label}x_iv_expected_rate"] = float(
            np.average(terminal_breach.astype(float), weights=weights)
        )
        result[f"excursion_above_{label}x_iv_sigma_rate"] = float(
            np.average(excursion_breach.astype(float), weights=weights)
        )
    result["surprise_1_5_sigma_rate"] = result["excursion_above_1_5x_iv_sigma_rate"]
    result["surprise_2_0_sigma_rate"] = result["excursion_above_2_0x_iv_sigma_rate"]
    return result


def with_population_comparison(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(metrics)
    output["npv_lift"] = float(metrics["remains_below_iv_rate"]) - float(
        baseline["remains_below_iv_rate"]
    )
    for label in ("1_5", "2_0"):
        tail_rate = float(metrics[f"surprise_{label}_sigma_rate"])
        population_rate = float(baseline[f"surprise_{label}_sigma_rate"])
        output[f"relative_surprise_{label}_sigma_reduction"] = (
            1.0 - tail_rate / population_rate if population_rate > 0.0 else math.nan
        )
    return output


def add_recent_high_m1c_context(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(["stock", "session", "checkpoint"], kind="mergesort").copy()
    recent = pd.Series(False, index=output.index)
    later = pd.Series(False, index=output.index)
    for _, group in output.groupby(["stock", "session"], sort=False):
        high_checkpoints = group.loc[
            group["M1C_probability"].ge(HIGH_M1C_THRESHOLD), "checkpoint"
        ].astype(int)
        high_values = high_checkpoints.to_numpy(int)
        for index, checkpoint in zip(
            group.index,
            group["checkpoint"].astype(int),
            strict=True,
        ):
            recent.loc[index] = bool(
                ((high_values < checkpoint) & (high_values >= checkpoint - 12)).any()
            )
            later.loc[index] = bool((high_values > checkpoint).any())
    output["recent_high_m1c_episode_previous_60m"] = recent
    output["high_m1c_episode_later_same_session"] = later
    return output.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def build_quiet_episodes(
    analytic: pd.DataFrame,
    threshold_artifact: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = analytic.loc[analytic["period"].isin(["assessment", "stress"])].copy()
    m1c_thresholds = cast(Mapping[str, float], threshold_artifact["M1C"])
    m1c_episode_frames: list[pd.DataFrame] = []
    for label in ("bottom_5_percent", "bottom_10_percent", "bottom_20_percent"):
        episodes = construct_fresh_quiet_episodes(
            eligible,
            threshold=float(m1c_thresholds[label]),
            probability_column="M1C_probability",
        )
        episodes["model"] = "M1C"
        episodes["tail"] = label
        m1c_episode_frames.append(episodes)
    m1c_episodes = pd.concat(m1c_episode_frames, ignore_index=True)
    m0_threshold = float(cast(Mapping[str, float], threshold_artifact["M0"])["bottom_10_percent"])
    m0_episodes = construct_fresh_quiet_episodes(
        eligible,
        threshold=m0_threshold,
        probability_column="M0_probability",
    )
    m0_episodes["model"] = "M0"
    m0_episodes["tail"] = "bottom_10_percent"
    raw_binding = eligible.loc[eligible["m1c_bottom_10_percent"].astype(bool)].copy()
    write_parquet(PRIMARY / "raw_low_tail_checkpoint_rows.parquet", raw_binding)
    write_parquet(PRIMARY / "fresh_quiet_episodes.parquet", m1c_episodes)
    return m1c_episodes, m0_episodes, raw_binding


def primary_metric_tables(
    analytic: pd.DataFrame,
    m1c_episodes: pd.DataFrame,
    m0_episodes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    checkpoint_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        population_metrics = summary_metrics(population)
        for label in ("bottom_5_percent", "bottom_10_percent", "bottom_20_percent"):
            tail = population.loc[population[f"m1c_{label}"].astype(bool)]
            metrics = with_population_comparison(summary_metrics(tail), population_metrics)
            checkpoint_rows.append(
                {
                    "period": period,
                    "model": "M1C",
                    "tail": label,
                    "horizon_minutes": PRIMARY_HORIZON,
                    **metrics,
                }
            )
        for label in ("bottom_5_percent", "bottom_10_percent", "bottom_20_percent"):
            episodes = m1c_episodes.loc[
                m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq(label)
            ]
            metrics = with_population_comparison(summary_metrics(episodes), population_metrics)
            episode_rows.append(
                {
                    "period": period,
                    "model": "M1C",
                    "tail": label,
                    "horizon_minutes": PRIMARY_HORIZON,
                    **metrics,
                }
            )
        m1c_tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        m0_tail = population.loc[population["m0_bottom_10_percent"].astype(bool)]
        m1c_metrics = with_population_comparison(summary_metrics(m1c_tail), population_metrics)
        m0_metrics = with_population_comparison(summary_metrics(m0_tail), population_metrics)
        comparison_rows.append(
            {
                "period": period,
                "m1c_rows": m1c_metrics["rows"],
                "m0_rows": m0_metrics["rows"],
                "remains_below_iv_rate_difference": (
                    m1c_metrics["remains_below_iv_rate"] - m0_metrics["remains_below_iv_rate"]
                ),
                "movement_exceeds_iv_rate_difference": (
                    m1c_metrics["movement_exceeds_iv_rate"] - m0_metrics["movement_exceeds_iv_rate"]
                ),
                "mean_iv_residual_difference": (
                    m1c_metrics["mean_iv_residual"] - m0_metrics["mean_iv_residual"]
                ),
                "median_iv_residual_difference": (
                    m1c_metrics["median_iv_residual"] - m0_metrics["median_iv_residual"]
                ),
                "mean_absolute_movement_difference": (
                    m1c_metrics["mean_absolute_movement"] - m0_metrics["mean_absolute_movement"]
                ),
                "p95_absolute_movement_difference": (
                    m1c_metrics["absolute_movement_p95"] - m0_metrics["absolute_movement_p95"]
                ),
                "p99_absolute_movement_difference": (
                    m1c_metrics["absolute_movement_p99"] - m0_metrics["absolute_movement_p99"]
                ),
                "maximum_excursion_difference": (
                    m1c_metrics["mean_maximum_absolute_excursion"]
                    - m0_metrics["mean_maximum_absolute_excursion"]
                ),
                "surprise_1_5_sigma_rate_difference": (
                    m1c_metrics["surprise_1_5_sigma_rate"] - m0_metrics["surprise_1_5_sigma_rate"]
                ),
                "surprise_2_0_sigma_rate_difference": (
                    m1c_metrics["surprise_2_0_sigma_rate"] - m0_metrics["surprise_2_0_sigma_rate"]
                ),
            }
        )
        overlap_rows.append(
            {
                "period": period,
                **tail_overlap(
                    m1c_tail["row_id"].astype(str).tolist(),
                    m0_tail["row_id"].astype(str).tolist(),
                ),
            }
        )
        baseline_rows.append(
            {
                "period": period,
                "population": "M1C_bottom_10_percent_checkpoints",
                "tail_remains_below_iv_rate": m1c_metrics["remains_below_iv_rate"],
                "full_population_remains_below_iv_rate": population_metrics[
                    "remains_below_iv_rate"
                ],
                "bottom_tail_npv_lift": m1c_metrics["npv_lift"],
                "tail_1_5_sigma_surprise_rate": m1c_metrics["surprise_1_5_sigma_rate"],
                "full_population_1_5_sigma_surprise_rate": population_metrics[
                    "surprise_1_5_sigma_rate"
                ],
                "relative_1_5_sigma_surprise_reduction": m1c_metrics[
                    "relative_surprise_1_5_sigma_reduction"
                ],
                "relative_2_0_sigma_surprise_reduction": m1c_metrics[
                    "relative_surprise_2_0_sigma_reduction"
                ],
            }
        )
        for population_name, frame in (
            ("bottom_10_percent_checkpoint", m1c_tail),
            (
                "bottom_10_percent_fresh_episode",
                m1c_episodes.loc[
                    m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq("bottom_10_percent")
                ],
            ),
        ):
            for horizon in HORIZONS_MINUTES:
                horizon_rows.append(
                    {
                        "period": period,
                        "population": population_name,
                        "model": "M1C",
                        "tail": "bottom_10_percent",
                        "horizon_minutes": horizon,
                        **summary_metrics(frame, horizon),
                    }
                )
    tables = {
        "checkpoint": pd.DataFrame(checkpoint_rows),
        "episodes": pd.DataFrame(episode_rows),
        "horizons": pd.DataFrame(horizon_rows),
        "comparison": pd.DataFrame(comparison_rows),
        "overlap": pd.DataFrame(overlap_rows),
        "baseline": pd.DataFrame(baseline_rows),
    }
    write_csv(PRIMARY / "checkpoint_low_tail_metrics.csv", tables["checkpoint"])
    write_csv(PRIMARY / "fresh_episode_metrics.csv", tables["episodes"])
    write_csv(PRIMARY / "horizon_metrics.csv", tables["horizons"])
    write_csv(PRIMARY / "m0_vs_m1c_low_tail.csv", tables["comparison"])
    write_csv(PRIMARY / "low_tail_overlap.csv", tables["overlap"])
    write_csv(PRIMARY / "population_baseline_comparison.csv", tables["baseline"])
    return tables


def containment_summary(
    frame: pd.DataFrame,
    *,
    horizon: int,
    multiplier: float,
) -> dict[str, Any]:
    available = frame.loc[frame[f"available_{horizon}m"].astype(bool)].copy()
    if available.empty:
        return {"rows": 0, "sessions": 0, "stocks": 0}
    up = available[f"maximum_up_excursion_{horizon}m"].to_numpy(float)
    down = np.abs(available[f"maximum_down_excursion_{horizon}m"].to_numpy(float))
    boundary = multiplier * available[f"iv_sigma_{horizon}m"].to_numpy(float)
    up_breach = up > boundary
    down_breach = down > boundary
    contained = ~(up_breach | down_breach)
    one_sided = up_breach ^ down_breach
    two_sided = up_breach & down_breach
    weights = _weights(available)
    label = {1.0: "1sigma", 1.5: "1_5sigma", 2.0: "2sigma"}[multiplier]
    breach = up_breach | down_breach
    distances = available[f"breach_distance_{label}_{horizon}m"].to_numpy(float)
    times = pd.to_numeric(
        available[f"time_to_{label}_breach_{horizon}m"], errors="coerce"
    ).to_numpy(float)
    reverted = available[f"breach_mean_reverted_{label}_{horizon}m"].to_numpy(bool)
    directions = available[f"breach_direction_{label}_{horizon}m"].astype(str)
    result: dict[str, Any] = {
        "rows": int(len(available)),
        "sessions": int(available["session"].nunique()),
        "stocks": int(available["stock"].nunique()),
        "containment_rate": float(np.average(contained.astype(float), weights=weights)),
        "one_sided_breach_rate": float(np.average(one_sided.astype(float), weights=weights)),
        "two_sided_breach_rate": float(np.average(two_sided.astype(float), weights=weights)),
        "any_breach_rate": float(np.average(breach.astype(float), weights=weights)),
        "mean_distance_beyond_boundary_when_breached": (
            float(np.average(distances[breach], weights=weights[breach])) if breach.any() else 0.0
        ),
        "median_distance_beyond_boundary_when_breached": (
            weighted_quantile(distances[breach], weights[breach], 0.50) if breach.any() else 0.0
        ),
        "maximum_breach": float(np.max(distances[breach])) if breach.any() else 0.0,
        "mean_time_to_breach_minutes": (
            float(np.average(times[breach], weights=weights[breach])) if breach.any() else math.nan
        ),
        "median_time_to_breach_minutes": (
            weighted_quantile(times[breach], weights[breach], 0.50) if breach.any() else math.nan
        ),
        "breach_mean_reverted_rate": (
            float(np.average(reverted[breach].astype(float), weights=weights[breach]))
            if breach.any()
            else math.nan
        ),
    }
    for direction in ("up", "down", "both"):
        result[f"{direction}_breach_rate"] = float(
            np.average(directions.eq(direction).to_numpy(float), weights=weights)
        )
    return result


def build_containment_and_surprises(
    analytic: pd.DataFrame,
    m1c_episodes: pd.DataFrame,
    m0_episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    surprise_frames: list[pd.DataFrame] = []
    for period in ("assessment", "stress"):
        full = analytic.loc[analytic["period"].eq(period)]
        m1c_fresh = m1c_episodes.loc[
            m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq("bottom_10_percent")
        ]
        m0_fresh = m0_episodes.loc[m0_episodes["period"].eq(period)]
        for population_name, frame in (
            ("full_checkpoint_population", full),
            ("M1C_bottom_10_fresh_episode", m1c_fresh),
            ("M0_bottom_10_fresh_episode", m0_fresh),
        ):
            for horizon in (15, 30, 60):
                for multiplier in (1.0, 1.5, 2.0):
                    rows.append(
                        {
                            "period": period,
                            "population": population_name,
                            "horizon_minutes": horizon,
                            "sigma_boundary": multiplier,
                            **containment_summary(
                                frame,
                                horizon=horizon,
                                multiplier=multiplier,
                            ),
                        }
                    )
        surprises = m1c_fresh.loc[m1c_fresh["excursion_sigma_ratio_15m"].ge(1.5)].copy()
        if not surprises.empty:
            surprises["surprise_large_mover"] = True
            surprises["extreme_surprise_mover"] = surprises["excursion_sigma_ratio_15m"].ge(2.0)
            surprises["direction"] = np.where(
                surprises["maximum_up_excursion_15m"].ge(
                    surprises["maximum_down_excursion_15m"].abs()
                ),
                "up",
                "down",
            )
            surprises["maximum_excursion"] = surprises["maximum_absolute_excursion_15m"]
            surprises["terminal_return"] = surprises["signed_return_15m"]
            surprises["move_reversed"] = surprises["large_excursion_mean_reverted_15m"]
            surprise_frames.append(surprises)
    containment = pd.DataFrame(rows)
    surprise_rows = (
        pd.concat(surprise_frames, ignore_index=True)
        if surprise_frames
        else pd.DataFrame(
            columns=[
                "period",
                "stock",
                "session",
                "checkpoint",
                "atm_iv",
                "M1C_probability",
                "M0_probability",
            ]
        )
    )
    surprise_columns = [
        column
        for column in (
            "period",
            "row_id",
            "stock",
            "session",
            "checkpoint",
            "atm_iv",
            "M1C_probability",
            "M0_probability",
            "direction",
            "maximum_excursion",
            "terminal_return",
            "move_reversed",
            "extreme_surprise_mover",
            "recent_high_m1c_episode_previous_60m",
            "market_movement_context_available",
            "market_movement_unusually_large",
            "excursion_sigma_ratio_15m",
        )
        if column in surprise_rows
    ]
    write_csv(PRIMARY / "containment_metrics.csv", containment)
    write_csv(PRIMARY / "surprise_mover_rows.csv", surprise_rows[surprise_columns])
    period_summary: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        period_rows = surprise_rows.loc[surprise_rows["period"].eq(period)]
        fresh_rows = m1c_episodes.loc[
            m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq("bottom_10_percent")
        ]
        count = int(len(period_rows))
        period_summary[period] = {
            "count": count,
            "rate": float(count / len(fresh_rows)) if len(fresh_rows) else math.nan,
            "stocks": sorted(period_rows["stock"].astype(str).unique().tolist()),
            "months": sorted(period_rows["session"].astype(str).str[:7].unique().tolist()),
            "checkpoints": sorted(period_rows["checkpoint"].astype(int).unique().tolist()),
            "maximum_stock_share": _share(period_rows, "stock") if count else 0.0,
            "maximum_month_share": (
                float(period_rows["session"].astype(str).str[:7].value_counts().max() / count)
                if count
                else 0.0
            ),
        }
    surprise_summary = {
        **CLAIMS,
        "definition": "excursion_sigma_ratio_15m >= 1.5",
        "extreme_definition": "excursion_sigma_ratio_15m >= 2.0",
        "periods": period_summary,
    }
    write_json(PRIMARY / "surprise_mover_summary.json", surprise_summary)
    return containment, surprise_rows, surprise_summary


def _weighted_correlation(
    left: np.ndarray[Any, np.dtype[np.float64]],
    right: np.ndarray[Any, np.dtype[np.float64]],
    weights: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    left_mean = float(np.average(left, weights=weights))
    right_mean = float(np.average(right, weights=weights))
    covariance = float(np.average((left - left_mean) * (right - right_mean), weights=weights))
    left_variance = float(np.average((left - left_mean) ** 2, weights=weights))
    right_variance = float(np.average((right - right_mean) ** 2, weights=weights))
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator > 0.0 else math.nan


def build_score_deciles(
    analytic: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    monotonicity: dict[str, Any] = {**CLAIMS, "periods": {}}
    metric_columns = {
        "movement_exceeds_iv_rate": "movement_exceeds_iv_15m",
        "mean_terminal_iv_residual": "terminal_iv_residual_15m",
        "mean_maximum_excursion": "maximum_absolute_excursion_15m",
        "surprise_1_5_sigma_rate": "_surprise_1_5",
    }
    working = analytic.copy()
    working["_surprise_1_5"] = working["excursion_sigma_ratio_15m"].ge(1.5)
    for period in ("assessment", "stress"):
        period_frame = working.loc[working["period"].eq(period)]
        for decile in range(1, 11):
            frame = period_frame.loc[period_frame["m1c_score_decile"].eq(decile)]
            summary = summary_metrics(frame)
            metric_rows.append(
                {
                    "period": period,
                    "decile": decile,
                    "rows": int(len(frame)),
                    "sessions": int(frame["session"].nunique()),
                    "stocks": int(frame["stock"].nunique()),
                    "candidate_weight": float(frame["row_weight"].sum()),
                    "mean_probability": weighted_mean(frame, "M1C_probability"),
                    "movement_exceeds_iv_rate": summary["movement_exceeds_iv_rate"],
                    "movement_remains_below_iv_rate": summary["remains_below_iv_rate"],
                    "mean_terminal_iv_residual": summary["mean_iv_residual"],
                    "median_terminal_iv_residual": summary["median_iv_residual"],
                    "mean_maximum_excursion": summary["mean_maximum_absolute_excursion"],
                    "surprise_1_5_sigma_rate": summary["surprise_1_5_sigma_rate"],
                }
            )
        period_metrics = pd.DataFrame(metric_rows)
        period_metrics = period_metrics.loc[period_metrics["period"].eq(period)]
        diagnostics: dict[str, Any] = {}
        x = period_metrics["decile"].to_numpy(float)
        weights = period_metrics["candidate_weight"].to_numpy(float)
        for metric, source_column in metric_columns.items():
            y = period_metrics[metric].to_numpy(float)
            spearman = _weighted_correlation(
                x,
                rankdata(y, method="average").astype(float),
                weights,
            )
            slope = float(np.polyfit(x, y, 1, w=np.sqrt(weights))[0])
            adjacent = int(np.count_nonzero(np.diff(y) >= 0.0))
            bottom = period_frame.loc[period_frame["m1c_score_decile"].eq(1)]
            top = period_frame.loc[period_frame["m1c_score_decile"].eq(10)]
            bottom_quintile = period_frame.loc[period_frame["m1c_score_decile"].isin([1, 2])]
            top_quintile = period_frame.loc[period_frame["m1c_score_decile"].isin([9, 10])]
            diagnostics[metric] = {
                "weighted_spearman": spearman,
                "linear_slope": slope,
                "monotonic_adjacent_steps": adjacent,
                "top_minus_bottom_decile": (
                    weighted_mean(top, source_column) - weighted_mean(bottom, source_column)
                ),
                "top_minus_bottom_quintile": (
                    weighted_mean(top_quintile, source_column)
                    - weighted_mean(bottom_quintile, source_column)
                ),
            }
        movement = diagnostics["movement_exceeds_iv_rate"]
        monotonicity["periods"][period] = {
            "metrics": diagnostics,
            "correct_overall_direction": bool(
                movement["weighted_spearman"] > 0.0
                and movement["linear_slope"] > 0.0
                and movement["top_minus_bottom_decile"] > 0.0
                and movement["monotonic_adjacent_steps"] >= 5
            ),
        }
    monotonicity["correct_overall_direction"] = bool(
        all(
            monotonicity["periods"][period]["correct_overall_direction"]
            for period in ("assessment", "stress")
        )
    )
    table = pd.DataFrame(metric_rows)
    write_csv(PRIMARY / "score_decile_metrics.csv", table)
    write_json(PRIMARY / "score_monotonicity.json", monotonicity)
    return table, monotonicity


def _null_metric_payload(
    frame: pd.DataFrame,
    baseline: Mapping[str, Any],
) -> dict[str, float]:
    metrics = with_population_comparison(summary_metrics(frame), baseline)
    return {
        "remains_below_iv_rate": float(metrics["remains_below_iv_rate"]),
        "npv_lift": float(metrics["npv_lift"]),
        "mean_iv_residual": float(metrics["mean_iv_residual"]),
        "median_iv_residual": float(metrics["median_iv_residual"]),
        "mean_maximum_excursion": float(metrics["mean_maximum_absolute_excursion"]),
        "breach_1_5_sigma_rate": float(metrics["surprise_1_5_sigma_rate"]),
        "breach_2_0_sigma_rate": float(metrics["surprise_2_0_sigma_rate"]),
    }


def build_matched_random_nulls(
    analytic: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wins_by_period: dict[str, dict[str, int]] = {}
    plan: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        population = (
            analytic.loc[analytic["period"].eq(period)]
            .sort_values("row_id", kind="mergesort")
            .reset_index(drop=True)
        )
        real_tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)].copy()
        baseline = summary_metrics(population)
        real = _null_metric_payload(real_tail, baseline)
        wins = {metric: 0 for metric in real}
        plan[period] = []
        for draw, seed in enumerate(MATCHED_SEEDS, start=1):
            selected, audit = matched_random_selection(
                population,
                real_tail,
                seed=seed,
            )
            metrics = _null_metric_payload(selected, baseline)
            beat: dict[str, bool] = {}
            for metric, real_value in real.items():
                higher_is_better = metric in {"remains_below_iv_rate", "npv_lift"}
                beat[metric] = (
                    real_value > metrics[metric]
                    if higher_is_better
                    else real_value < metrics[metric]
                )
                wins[metric] += int(beat[metric])
            rows.append(
                {
                    "period": period,
                    "draw": draw,
                    "seed": seed,
                    **metrics,
                    **audit,
                    **{f"real_tail_beats_{metric}": value for metric, value in beat.items()},
                }
            )
            plan[period].append(
                {
                    "draw": draw,
                    "seed": seed,
                    "selected_row_ids": selected["row_id"].astype(str).tolist(),
                    **audit,
                }
            )
        wins_by_period[period] = wins
    table = pd.DataFrame(rows)
    write_csv(PRIMARY / "matched_random_null_metrics.csv", table)
    return table, wins_by_period, plan


def build_probability_permutation_nulls(
    analytic: pd.DataFrame,
    threshold_artifact: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, Any]]:
    threshold = float(cast(Mapping[str, float], threshold_artifact["M1C"])["bottom_10_percent"])
    rows: list[dict[str, Any]] = []
    wins_by_period: dict[str, dict[str, int]] = {}
    plan: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        population = (
            analytic.loc[analytic["period"].eq(period)]
            .sort_values("row_id", kind="mergesort")
            .reset_index(drop=True)
        )
        real_tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        baseline = summary_metrics(population)
        real = _null_metric_payload(real_tail, baseline)
        wins = {metric: 0 for metric in real}
        plan[period] = []
        for draw, seed in enumerate(PERMUTATION_SEEDS, start=1):
            permuted = permute_probabilities_within_sessions(
                population,
                probability_column="M1C_probability",
                seed=seed,
            )
            null_tail = population.loc[permuted.le(threshold)].copy()
            if null_tail.empty:
                raise ScreenBlocked(
                    "blocked_reproducibility_or_audit_failure",
                    "a probability permutation produced no low-tail rows",
                )
            metrics = _null_metric_payload(null_tail, baseline)
            beat: dict[str, bool] = {}
            for metric, real_value in real.items():
                higher_is_better = metric in {"remains_below_iv_rate", "npv_lift"}
                beat[metric] = (
                    real_value > metrics[metric]
                    if higher_is_better
                    else real_value < metrics[metric]
                )
                wins[metric] += int(beat[metric])
            rows.append(
                {
                    "period": period,
                    "draw": draw,
                    "seed": seed,
                    "rows": int(len(null_tail)),
                    **metrics,
                    **{f"real_tail_beats_{metric}": value for metric, value in beat.items()},
                }
            )
            plan[period].append(
                {
                    "draw": draw,
                    "seed": seed,
                    "tail_row_ids": null_tail["row_id"].astype(str).tolist(),
                }
            )
        wins_by_period[period] = wins
    table = pd.DataFrame(rows)
    write_csv(PRIMARY / "probability_permutation_null_metrics.csv", table)
    return table, wins_by_period, plan


def resample_sessions(
    frame: pd.DataFrame,
    sampled_sessions: Sequence[str],
) -> pd.DataFrame:
    multiplicities = Counter(str(value) for value in sampled_sessions)
    output = frame.loc[frame["session"].astype(str).isin(multiplicities)].copy()
    output["_analysis_weight"] = output["row_weight"].to_numpy(float) * output["session"].astype(
        str
    ).map(multiplicities).to_numpy(float)
    return output


def bootstrap_statistics(
    population: pd.DataFrame,
    fresh_episodes: pd.DataFrame,
) -> dict[str, float]:
    full = summary_metrics(population)
    m1c = with_population_comparison(
        summary_metrics(population.loc[population["m1c_bottom_10_percent"].astype(bool)]),
        full,
    )
    m0 = with_population_comparison(
        summary_metrics(population.loc[population["m0_bottom_10_percent"].astype(bool)]),
        full,
    )
    fresh = with_population_comparison(summary_metrics(fresh_episodes), full)
    return {
        "m1c_bottom_tail_remains_below_iv_rate": float(m1c["remains_below_iv_rate"]),
        "bottom_tail_npv_lift": float(m1c["npv_lift"]),
        "mean_terminal_iv_residual": float(m1c["mean_iv_residual"]),
        "median_terminal_iv_residual": float(m1c["median_iv_residual"]),
        "mean_terminal_absolute_movement": float(m1c["mean_absolute_movement"]),
        "p95_terminal_absolute_movement": float(m1c["absolute_movement_p95"]),
        "breach_1_5_sigma_rate": float(m1c["surprise_1_5_sigma_rate"]),
        "breach_2_0_sigma_rate": float(m1c["surprise_2_0_sigma_rate"]),
        "m1c_minus_m0_remains_below_iv_difference": float(
            m1c["remains_below_iv_rate"] - m0["remains_below_iv_rate"]
        ),
        "m1c_minus_m0_mean_residual_difference": float(
            m1c["mean_iv_residual"] - m0["mean_iv_residual"]
        ),
        "m1c_minus_m0_1_5_sigma_breach_difference": float(
            m1c["surprise_1_5_sigma_rate"] - m0["surprise_1_5_sigma_rate"]
        ),
        "fresh_quiet_episode_remains_below_iv_rate": float(fresh["remains_below_iv_rate"]),
        "fresh_quiet_episode_mean_residual": float(fresh["mean_iv_residual"]),
        "fresh_episode_1_5_sigma_breach_difference_vs_full": float(
            fresh["surprise_1_5_sigma_rate"] - full["surprise_1_5_sigma_rate"]
        ),
    }


def build_bootstrap(
    analytic: pd.DataFrame,
    m1c_episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    plan: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)].copy()
        fresh = m1c_episodes.loc[
            m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq("bottom_10_percent")
        ].copy()
        estimate = bootstrap_statistics(population, fresh)
        draws = whole_session_bootstrap_plan(
            population["session"],
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEEDS[period],
        )
        plan[period] = {
            "seed": BOOTSTRAP_SEEDS[period],
            "draws": [list(draw) for draw in draws],
        }
        draw_values = {statistic: [] for statistic in estimate}
        for sampled_sessions in draws:
            sampled_population = resample_sessions(population, sampled_sessions)
            sampled_fresh = resample_sessions(fresh, sampled_sessions)
            statistics = bootstrap_statistics(sampled_population, sampled_fresh)
            for statistic, value in statistics.items():
                draw_values[statistic].append(value)
        for statistic, values in draw_values.items():
            array = np.asarray(values, dtype=float)
            summary_rows.append(
                {
                    "period": period,
                    "statistic": statistic,
                    "estimate": estimate[statistic],
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEEDS[period],
                    "lower_80": float(np.quantile(array, 0.10)),
                    "upper_80": float(np.quantile(array, 0.90)),
                    "lower_90": float(np.quantile(array, 0.05)),
                    "upper_90": float(np.quantile(array, 0.95)),
                    "lower_95": float(np.quantile(array, 0.025)),
                    "upper_95": float(np.quantile(array, 0.975)),
                }
            )
    table = pd.DataFrame(summary_rows)
    write_csv(PRIMARY / "bootstrap_metrics.csv", table)
    return table, plan


def build_stability_and_concentration(
    analytic: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    monthly_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    subgroup_dimensions = (
        "checkpoint_group",
        "previous_close_atm_iv_group",
        "stock_local_volatility_group",
        "market_volatility_group",
        "option_dte_group",
        "market_up_down",
        "market_quiet_active",
    )
    period_audit: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        baseline = summary_metrics(population)
        for month, frame in tail.groupby("month", sort=True):
            monthly_rows.append(
                {
                    "period": period,
                    "month": month,
                    **with_population_comparison(summary_metrics(frame), baseline),
                }
            )
        for dimension in subgroup_dimensions:
            for level, frame in tail.groupby(dimension, sort=True):
                subgroup_rows.append(
                    {
                        "period": period,
                        "dimension": dimension,
                        "level": level,
                        **with_population_comparison(
                            summary_metrics(frame),
                            summary_metrics(population.loc[population[dimension].eq(level)]),
                        ),
                    }
                )
        for stock, frame in tail.groupby("stock", sort=True):
            stock_rows.append(
                {
                    "period": period,
                    "stock": stock,
                    **with_population_comparison(
                        summary_metrics(frame),
                        summary_metrics(population.loc[population["stock"].eq(stock)]),
                    ),
                }
            )
        for dimension in ("stock", "month", "session", "checkpoint"):
            counts = tail.groupby(dimension, sort=True).size()
            for entity, count in counts.items():
                concentration_rows.append(
                    {
                        "period": period,
                        "analysis": f"tail_row_share_by_{dimension}",
                        "entity": str(entity),
                        "value": int(count),
                        "share": float(count / len(tail)),
                    }
                )
        negative = -tail["terminal_iv_residual_15m"].clip(upper=0.0) * tail["row_weight"]
        negative_by_stock = (
            tail.assign(_negative_contribution=negative)
            .groupby("stock", sort=True)["_negative_contribution"]
            .sum()
        )
        negative_total = float(negative_by_stock.sum())
        for stock, contribution in negative_by_stock.items():
            concentration_rows.append(
                {
                    "period": period,
                    "analysis": "negative_iv_residual_contribution_by_stock",
                    "entity": str(stock),
                    "value": float(contribution),
                    "share": (
                        float(contribution / negative_total) if negative_total > 0.0 else math.nan
                    ),
                }
            )
        surprises = tail.loc[tail["excursion_sigma_ratio_15m"].ge(1.5)]
        for dimension in ("stock", "month"):
            counts = surprises.groupby(dimension, sort=True).size()
            for entity, count in counts.items():
                concentration_rows.append(
                    {
                        "period": period,
                        "analysis": f"surprise_mover_contribution_by_{dimension}",
                        "entity": str(entity),
                        "value": int(count),
                        "share": (float(count / len(surprises)) if len(surprises) else math.nan),
                    }
                )
        for omitted_stock in sorted(population["stock"].astype(str).unique()):
            reduced_population = population.loc[population["stock"].astype(str).ne(omitted_stock)]
            reduced_tail = reduced_population.loc[
                reduced_population["m1c_bottom_10_percent"].astype(bool)
            ]
            metrics = with_population_comparison(
                summary_metrics(reduced_tail), summary_metrics(reduced_population)
            )
            leave_one_out_rows.append(
                {
                    "period": period,
                    "omitted_stock": omitted_stock,
                    "remains_below_iv_rate": metrics["remains_below_iv_rate"],
                    "npv_lift": metrics["npv_lift"],
                    "mean_iv_residual": metrics["mean_iv_residual"],
                    "breach_1_5_sigma_rate": metrics["surprise_1_5_sigma_rate"],
                    "breach_2_0_sigma_rate": metrics["surprise_2_0_sigma_rate"],
                }
            )
        period_audit[period] = {
            "maximum_stock_share": _share(tail, "stock"),
            "maximum_month_share": float(tail["month"].value_counts().max() / len(tail)),
            "maximum_session_share": _share(tail, "session"),
            "maximum_checkpoint_share": _share(tail, "checkpoint"),
        }
    tables = {
        "monthly": pd.DataFrame(monthly_rows),
        "subgroups": pd.DataFrame(subgroup_rows),
        "stocks": pd.DataFrame(stock_rows),
        "concentration": pd.DataFrame(concentration_rows),
        "leave_one_stock_out": pd.DataFrame(leave_one_out_rows),
    }
    write_csv(PRIMARY / "monthly_metrics.csv", tables["monthly"])
    write_csv(PRIMARY / "checkpoint_group_metrics.csv", tables["subgroups"])
    write_csv(PRIMARY / "stock_metrics.csv", tables["stocks"])
    write_csv(PRIMARY / "concentration_metrics.csv", tables["concentration"])
    write_csv(PRIMARY / "leave_one_stock_out.csv", tables["leave_one_stock_out"])
    return tables, period_audit


def build_panel_support(
    analytic: pd.DataFrame,
    m1c_episodes: pd.DataFrame,
) -> dict[str, Any]:
    checkpoint_support: dict[str, Any] = {}
    episode_support: dict[str, Any] = {}
    horizon_support: dict[str, Any] = {}
    population_support: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        fresh = m1c_episodes.loc[
            m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq("bottom_10_percent")
        ]
        checkpoint_support[period] = evaluate_support_gate(
            tail,
            period=period,
            population="checkpoint",
        )
        episode_support[period] = evaluate_support_gate(
            fresh,
            period=period,
            population="fresh_episode",
        )
        population_support[period] = {
            "rows": int(len(population)),
            "sessions": int(population["session"].nunique()),
            "stocks": int(population["stock"].nunique()),
            "months": int(population["month"].nunique()),
        }
        horizon_support[period] = {}
        for horizon in HORIZONS_MINUTES:
            horizon_tail = tail.loc[tail[f"available_{horizon}m"].astype(bool)]
            horizon_fresh = fresh.loc[fresh[f"available_{horizon}m"].astype(bool)]
            horizon_support[period][str(horizon)] = {
                "checkpoint": {
                    "rows": int(len(horizon_tail)),
                    "sessions": int(horizon_tail["session"].nunique()),
                    "stocks": int(horizon_tail["stock"].nunique()),
                },
                "fresh_episode": {
                    "rows": int(len(horizon_fresh)),
                    "sessions": int(horizon_fresh["session"].nunique()),
                    "stocks": int(horizon_fresh["stock"].nunique()),
                },
            }
    artifact = {
        **CLAIMS,
        "full_population": population_support,
        "checkpoint_bottom_10_percent": checkpoint_support,
        "fresh_bottom_10_percent": episode_support,
        "longer_horizon_support": horizon_support,
        "optional_market_context_fallback": {
            column: {
                "rows": int(analytic[f"{column}_fallback_used"].sum()),
                "development_rows": int(
                    analytic.loc[
                        analytic["period"].eq("development"),
                        f"{column}_fallback_used",
                    ].sum()
                ),
                "assessment_rows": int(
                    analytic.loc[
                        analytic["period"].eq("assessment"),
                        f"{column}_fallback_used",
                    ].sum()
                ),
                "stress_rows": int(
                    analytic.loc[
                        analytic["period"].eq("stress"),
                        f"{column}_fallback_used",
                    ].sum()
                ),
                "frozen_source_period": "2024",
            }
            for column in ("market_volatility", "market_return_through_checkpoint")
        },
        "late_checkpoints_excluded_for_longer_horizon_support": False,
    }
    write_json(PRIMARY / "panel_support.json", artifact)
    return artifact


def _table_row(
    frame: pd.DataFrame,
    **matches: object,
) -> dict[str, Any]:
    mask = pd.Series(True, index=frame.index)
    for column, value in matches.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"expected one metric row for {matches}, found {len(selected)}",
        )
    return cast(dict[str, Any], selected.iloc[0].to_dict())


def build_decision(
    *,
    contract: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    containment: pd.DataFrame,
    surprise_summary: Mapping[str, Any],
    monotonicity: Mapping[str, Any],
    matched_wins: Mapping[str, Mapping[str, int]],
    permutation_wins: Mapping[str, Mapping[str, int]],
    bootstrap: pd.DataFrame,
    stability_tables: Mapping[str, pd.DataFrame],
    panel_support: Mapping[str, Any],
) -> dict[str, Any]:
    gate_evidence: dict[str, Any] = {
        "score_decile_direction_correct": bool(monotonicity["correct_overall_direction"]),
        "protected_boundary_passed": True,
        "chronology_audit_passed": True,
    }
    comparison_better_by_period: dict[str, bool] = {}
    descriptive_by_period: dict[str, bool] = {}
    for period in ("assessment", "stress"):
        checkpoint = _table_row(
            tables["checkpoint"],
            period=period,
            model="M1C",
            tail="bottom_10_percent",
        )
        comparison = _table_row(tables["comparison"], period=period)
        bootstrap_lift = _table_row(
            bootstrap,
            period=period,
            statistic="bottom_tail_npv_lift",
        )
        bootstrap_residual = _table_row(
            bootstrap,
            period=period,
            statistic="mean_terminal_iv_residual",
        )
        monthly = stability_tables["monthly"].loc[stability_tables["monthly"]["period"].eq(period)]
        leave_one_out = stability_tables["leave_one_stock_out"].loc[
            stability_tables["leave_one_stock_out"]["period"].eq(period)
        ]
        support = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], panel_support["checkpoint_bottom_10_percent"])[period],
        )
        gate_evidence[period] = {
            "remains_below_iv_rate": checkpoint["remains_below_iv_rate"],
            "npv_lift": checkpoint["npv_lift"],
            "mean_iv_residual": checkpoint["mean_iv_residual"],
            "median_iv_residual": checkpoint["median_iv_residual"],
            "bootstrap_80_npv_lift_lower": bootstrap_lift["lower_80"],
            "bootstrap_80_mean_residual_upper": bootstrap_residual["upper_80"],
            "m1c_beats_m0_remains_below": (comparison["remains_below_iv_rate_difference"] > 0.0),
            "m1c_beats_m0_mean_residual": (comparison["mean_iv_residual_difference"] < 0.0),
            "m1c_beats_m0_1_5_sigma_breach": (
                comparison["surprise_1_5_sigma_rate_difference"] < 0.0
            ),
            "matched_npv_lift_wins": matched_wins[period]["npv_lift"],
            "matched_mean_residual_wins": matched_wins[period]["mean_iv_residual"],
            "permutation_npv_lift_wins": permutation_wins[period]["npv_lift"],
            "permutation_mean_residual_wins": permutation_wins[period]["mean_iv_residual"],
            "negative_residual_months": int(monthly["mean_iv_residual"].lt(0.0).sum()),
            "required_negative_residual_months": 6 if period == "assessment" else 3,
            "support_passed": bool(support["passed"]),
            "not_dependent_on_one_stock": bool(
                leave_one_out["npv_lift"].gt(0.0).all()
                and leave_one_out["mean_iv_residual"].lt(0.0).all()
            ),
        }
        comparison_better_by_period[period] = bool(
            comparison["remains_below_iv_rate_difference"] > 0.0
            and comparison["mean_iv_residual_difference"] < 0.0
            and comparison["surprise_1_5_sigma_rate_difference"] < 0.0
        )
        descriptive_by_period[period] = bool(
            checkpoint["npv_lift"] > 0.0 and checkpoint["mean_iv_residual"] < 0.0
        )
    veto_gate = evaluate_low_movement_veto_gate(gate_evidence)

    readiness_evidence: dict[str, Any] = {
        "veto_gate_passed": bool(veto_gate["passed"]),
    }
    thirty_minute_favourable = True
    for period in ("assessment", "stress"):
        fresh_15_1_5 = _table_row(
            containment,
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        m0_15_1_5 = _table_row(
            containment,
            period=period,
            population="M0_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        full_15_1_5 = _table_row(
            containment,
            period=period,
            population="full_checkpoint_population",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        fresh_15_2 = _table_row(
            containment,
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=2.0,
        )
        full_15_2 = _table_row(
            containment,
            period=period,
            population="full_checkpoint_population",
            horizon_minutes=15,
            sigma_boundary=2.0,
        )
        bootstrap_difference = _table_row(
            bootstrap,
            period=period,
            statistic="fresh_episode_1_5_sigma_breach_difference_vs_full",
        )
        episode_support = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], panel_support["fresh_bottom_10_percent"])[period],
        )
        readiness_evidence[period] = {
            "fresh_1_5_sigma_lower_than_full": (
                fresh_15_1_5["any_breach_rate"] < full_15_1_5["any_breach_rate"]
            ),
            "fresh_1_5_sigma_lower_than_m0": (
                fresh_15_1_5["any_breach_rate"] < m0_15_1_5["any_breach_rate"]
            ),
            "fresh_2_sigma_lower_than_full": (
                fresh_15_2["any_breach_rate"] < full_15_2["any_breach_rate"]
            ),
            "bootstrap_80_1_5_sigma_difference_upper": bootstrap_difference["upper_80"],
            "two_sigma_containment_rate": fresh_15_2["containment_rate"],
            "support_passed": bool(episode_support["passed"]),
        }
        fresh_30 = _table_row(
            containment,
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=30,
            sigma_boundary=1.5,
        )
        m0_30 = _table_row(
            containment,
            period=period,
            population="M0_bottom_10_fresh_episode",
            horizon_minutes=30,
            sigma_boundary=1.5,
        )
        full_30 = _table_row(
            containment,
            period=period,
            population="full_checkpoint_population",
            horizon_minutes=30,
            sigma_boundary=1.5,
        )
        thirty_minute_favourable &= bool(
            fresh_30["any_breach_rate"] < m0_30["any_breach_rate"]
            and fresh_30["any_breach_rate"] < full_30["any_breach_rate"]
        )
    surprise_periods = cast(Mapping[str, Mapping[str, Any]], surprise_summary["periods"])
    readiness_evidence["surprise_movers_not_concentrated"] = bool(
        all(
            float(surprise_periods[period]["maximum_stock_share"]) <= 0.50
            and float(surprise_periods[period]["maximum_month_share"]) <= 0.60
            for period in ("assessment", "stress")
        )
    )
    readiness_evidence["thirty_minute_containment_favourable"] = thirty_minute_favourable
    readiness_gate = evaluate_short_premium_readiness_gate(readiness_evidence)

    checkpoint_support = cast(
        Mapping[str, Mapping[str, Any]],
        panel_support["checkpoint_bottom_10_percent"],
    )
    episode_support = cast(
        Mapping[str, Mapping[str, Any]],
        panel_support["fresh_bottom_10_percent"],
    )
    blocker: str | None = None
    if not all(bool(checkpoint_support[period]["passed"]) for period in ("assessment", "stress")):
        blocker = "blocked_insufficient_low_tail_support"
    elif not all(bool(episode_support[period]["passed"]) for period in ("assessment", "stress")):
        blocker = "blocked_insufficient_fresh_quiet_episode_support"
    descriptive_signal = all(descriptive_by_period.values())
    overall = choose_overall_decision(
        blocker=blocker,
        veto_supported=bool(veto_gate["passed"]),
        readiness_supported=bool(readiness_gate["passed"]),
        descriptive_signal=descriptive_signal,
    )
    m0_comparison_supported = all(comparison_better_by_period.values())
    surprise_supported = bool(
        all(
            cast(Mapping[str, Any], readiness_evidence[period])["fresh_1_5_sigma_lower_than_full"]
            for period in ("assessment", "stress")
        )
        and readiness_evidence["surprise_movers_not_concentrated"]
    )
    statuses = {
        "m1c_reconstruction_status": "supported",
        "low_tail_threshold_status": "supported",
        "checkpoint_low_movement_status": (
            "supported"
            if veto_gate["passed"]
            else ("promising" if descriptive_signal else "not_supported")
        ),
        "fresh_quiet_episode_status": (
            "supported"
            if all(bool(episode_support[period]["passed"]) for period in ("assessment", "stress"))
            else "insufficient_support"
        ),
        "m0_comparison_status": ("supported" if m0_comparison_supported else "not_supported"),
        "score_monotonicity_status": (
            "supported" if monotonicity["correct_overall_direction"] else "not_supported"
        ),
        "surprise_mover_status": ("supported" if surprise_supported else "descriptive_only"),
        "range_containment_status": (
            "supported"
            if readiness_gate["passed"]
            else (
                "promising"
                if all(
                    cast(Mapping[str, Any], readiness_evidence[period])[
                        "two_sigma_containment_rate"
                    ]
                    >= 0.80
                    for period in ("assessment", "stress")
                )
                else "not_supported"
            )
        ),
        "long_premium_veto_status": (
            "supported"
            if veto_gate["passed"]
            else ("descriptive_only" if descriptive_signal else "not_supported")
        ),
        "short_premium_recorder_priority": (
            "supported" if readiness_gate["passed"] else "not_supported"
        ),
    }
    return {
        **dict(contract),
        "overall_decision": overall,
        **statuses,
        "binding_low_movement_veto_gate": veto_gate,
        "binding_low_movement_veto_evidence": gate_evidence,
        "short_premium_readiness_gate": readiness_gate,
        "short_premium_readiness_evidence": readiness_evidence,
        "prospective_defined_risk_short_premium_shadow_recording_prioritised": bool(
            readiness_gate["passed"]
        ),
        "actual_option_profitability_calculated_or_claimed": False,
        "independent_audit_status": "pending",
        "determinism_status": "pending",
    }


def write_prospective_artifact(decision: Mapping[str, Any]) -> Path:
    readiness = cast(Mapping[str, Any], decision["short_premium_readiness_gate"])
    failed = [
        key for key, value in cast(Mapping[str, bool], readiness["checks"]).items() if not value
    ]
    if readiness["passed"]:
        path = PRIMARY / "prospective_short_premium_recording_contract.json"
        opposite = PRIMARY / "prospective_short_premium_recording_blocker.json"
        if opposite.exists():
            opposite.unlink()
        write_json(
            path,
            {
                **CLAIMS,
                "authorisation": ("prospective defined-risk short-premium shadow recording"),
                "structures": [
                    "0DTE iron butterfly",
                    "0DTE iron condor",
                    "1DTE iron condor",
                    "3-5 DTE iron condor",
                    "defined-risk call credit spread",
                    "defined-risk put credit spread",
                ],
                "opening_quote_convention": {
                    "short_option": "observed bid",
                    "protective_long_option": "observed ask",
                },
                "closing_quote_convention": {
                    "short_option": "observed ask",
                    "protective_long_option": "observed bid",
                },
                "historical_option_pnl_calculated": False,
                "paper_or_live_order_authorisation": False,
            },
        )
        return path
    path = PRIMARY / "prospective_short_premium_recording_blocker.json"
    opposite = PRIMARY / "prospective_short_premium_recording_contract.json"
    if opposite.exists():
        opposite.unlink()
    write_json(
        path,
        {
            **CLAIMS,
            "authorisation": "blocked",
            "failed_readiness_checks": failed,
            "prospective_defined_risk_short_premium_shadow_recording_prioritised": False,
            "historical_option_pnl_calculated": False,
        },
    )
    return path


def _maximum_null_rebuild_difference(
    analytic: pd.DataFrame,
    matched_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    plan: Mapping[str, Any],
) -> float:
    metrics = (
        "remains_below_iv_rate",
        "npv_lift",
        "mean_iv_residual",
        "median_iv_residual",
        "mean_maximum_excursion",
        "breach_1_5_sigma_rate",
        "breach_2_0_sigma_rate",
    )
    differences: list[float] = []
    indexed = analytic.set_index("row_id", drop=False)
    matched_plan = cast(Mapping[str, Sequence[Mapping[str, Any]]], plan["matched_random"])
    permutation_plan = cast(
        Mapping[str, Sequence[Mapping[str, Any]]],
        plan["probability_permutation"],
    )
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        baseline = summary_metrics(population)
        for payload in matched_plan[period]:
            selected = indexed.loc[[str(value) for value in payload["selected_row_ids"]]].copy()
            rebuilt = _null_metric_payload(selected, baseline)
            frozen = _table_row(
                matched_table,
                period=period,
                draw=int(payload["draw"]),
            )
            differences.extend(abs(rebuilt[metric] - float(frozen[metric])) for metric in metrics)
        for payload in permutation_plan[period]:
            selected = indexed.loc[[str(value) for value in payload["tail_row_ids"]]].copy()
            rebuilt = _null_metric_payload(selected, baseline)
            frozen = _table_row(
                permutation_table,
                period=period,
                draw=int(payload["draw"]),
            )
            differences.extend(abs(rebuilt[metric] - float(frozen[metric])) for metric in metrics)
    return float(max(differences, default=0.0))


def _maximum_bootstrap_rebuild_difference(
    analytic: pd.DataFrame,
    m1c_episodes: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    plan: Mapping[str, Any],
) -> float:
    differences: list[float] = []
    bootstrap_plan = cast(Mapping[str, Mapping[str, Any]], plan["bootstrap"])
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        fresh = m1c_episodes.loc[
            m1c_episodes["period"].eq(period) & m1c_episodes["tail"].eq("bottom_10_percent")
        ]
        estimate = bootstrap_statistics(population, fresh)
        values = {statistic: [] for statistic in estimate}
        for sampled_sessions in bootstrap_plan[period]["draws"]:
            statistics = bootstrap_statistics(
                resample_sessions(population, sampled_sessions),
                resample_sessions(fresh, sampled_sessions),
            )
            for statistic, value in statistics.items():
                values[statistic].append(value)
        for statistic, draws in values.items():
            array = np.asarray(draws, dtype=float)
            rebuilt = {
                "estimate": estimate[statistic],
                "lower_80": float(np.quantile(array, 0.10)),
                "upper_80": float(np.quantile(array, 0.90)),
                "lower_90": float(np.quantile(array, 0.05)),
                "upper_90": float(np.quantile(array, 0.95)),
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
            }
            frozen = _table_row(
                bootstrap_table,
                period=period,
                statistic=statistic,
            )
            differences.extend(
                abs(value - float(frozen[column])) for column, value in rebuilt.items()
            )
    return float(max(differences, default=0.0))


def determinism_checks(
    *,
    analytic: pd.DataFrame,
    states: pd.DataFrame,
    threshold_artifact: Mapping[str, Any],
    decile_artifact: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    m1c_episodes: pd.DataFrame,
    matched_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    resampling_plan: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    m1c_spec = cast(Mapping[str, object], feature_manifest["model_specification"])
    m0_spec = cast(Mapping[str, object], feature_manifest["m0_model_specification"])
    rebuilt_m1c = reconstruct_frozen_probabilities(analytic, m1c_spec)
    rebuilt_m0 = reconstruct_frozen_probabilities(analytic, m0_spec)
    maximum_m1c_probability_difference = float(
        np.max(np.abs(rebuilt_m1c - analytic["M1C_probability"].to_numpy(float)))
    )
    maximum_m0_probability_difference = float(
        np.max(np.abs(rebuilt_m0 - analytic["M0_probability"].to_numpy(float)))
    )
    development = analytic.loc[analytic["period"].eq("development")]
    development_weights = development["row_weight"].to_numpy(float)
    maximum_threshold_difference = 0.0
    membership_mismatches = 0
    for model, probability_column in (
        ("M1C", "M1C_probability"),
        ("M0", "M0_probability"),
    ):
        frozen_thresholds = cast(Mapping[str, float], threshold_artifact[model])
        for label, quantile in TAIL_QUANTILES.items():
            rebuilt_threshold = weighted_quantile(
                development[probability_column].to_numpy(float),
                development_weights,
                quantile,
            )
            maximum_threshold_difference = max(
                maximum_threshold_difference,
                abs(rebuilt_threshold - float(frozen_thresholds[label])),
            )
            rebuilt_membership = analytic[probability_column].le(rebuilt_threshold)
            membership_mismatches += int(
                rebuilt_membership.ne(analytic[f"{model.lower()}_{label}"].astype(bool)).sum()
            )
    rebuilt_deciles = freeze_weighted_boundaries(
        development["M1C_probability"].to_numpy(float),
        development_weights,
        quantiles=DECILE_QUANTILES,
    )
    frozen_deciles = tuple(float(value) for value in decile_artifact["boundaries"])
    maximum_decile_boundary_difference = float(
        np.max(np.abs(np.asarray(rebuilt_deciles) - np.asarray(frozen_deciles)))
    )

    eligible = analytic.loc[analytic["period"].isin(["assessment", "stress"])]
    rebuilt_episodes = construct_fresh_quiet_episodes(
        eligible,
        threshold=float(cast(Mapping[str, float], threshold_artifact["M1C"])["bottom_10_percent"]),
        probability_column="M1C_probability",
    )
    frozen_episode_ids = set(
        m1c_episodes.loc[m1c_episodes["tail"].eq("bottom_10_percent"), "row_id"].astype(str)
    )
    rebuilt_episode_ids = set(rebuilt_episodes["row_id"].astype(str))
    fresh_episode_identity_mismatches = len(
        frozen_episode_ids.symmetric_difference(rebuilt_episode_ids)
    )

    rebuilt_movement, rebuilt_paths = calculate_checkpoint_outcomes(
        analytic[
            [
                "row_id",
                "stock",
                "session",
                "checkpoint",
                "feature_available_timestamp_utc",
                "atm_iv",
            ]
        ],
        states,
    )
    frozen = analytic.set_index("row_id").sort_index()
    rebuilt_movement = rebuilt_movement.set_index("row_id").sort_index()
    rebuilt_paths = rebuilt_paths.set_index("row_id").sort_index()
    row_identity_mismatches = len(
        set(frozen.index.astype(str)).symmetric_difference(set(rebuilt_movement.index.astype(str)))
    )
    maximum_terminal_return_difference = 0.0
    maximum_excursion_difference = 0.0
    maximum_iv_residual_difference = 0.0
    for horizon in HORIZONS_MINUTES:
        maximum_terminal_return_difference = max(
            maximum_terminal_return_difference,
            float(
                np.nanmax(
                    np.abs(
                        rebuilt_movement[f"signed_return_{horizon}m"].to_numpy(float)
                        - frozen[f"signed_return_{horizon}m"].to_numpy(float)
                    )
                )
            ),
        )
        maximum_excursion_difference = max(
            maximum_excursion_difference,
            float(
                np.nanmax(
                    np.abs(
                        rebuilt_paths[f"maximum_absolute_excursion_{horizon}m"].to_numpy(float)
                        - frozen[f"maximum_absolute_excursion_{horizon}m"].to_numpy(float)
                    )
                )
            ),
        )
        maximum_iv_residual_difference = max(
            maximum_iv_residual_difference,
            float(
                np.nanmax(
                    np.abs(
                        rebuilt_movement[f"terminal_iv_residual_{horizon}m"].to_numpy(float)
                        - frozen[f"terminal_iv_residual_{horizon}m"].to_numpy(float)
                    )
                )
            ),
        )
    maximum_null_metric_difference = _maximum_null_rebuild_difference(
        analytic,
        matched_table,
        permutation_table,
        resampling_plan,
    )
    maximum_bootstrap_metric_difference = _maximum_bootstrap_rebuild_difference(
        analytic,
        m1c_episodes,
        bootstrap_table,
        resampling_plan,
    )
    overall = str(decision["overall_decision"])
    blocker = overall if overall.startswith("blocked_") else None
    expected_overall = choose_overall_decision(
        blocker=blocker,
        veto_supported=bool(
            cast(Mapping[str, Any], decision["binding_low_movement_veto_gate"])["passed"]
        ),
        readiness_supported=bool(
            cast(Mapping[str, Any], decision["short_premium_readiness_gate"])["passed"]
        ),
        descriptive_signal=overall == "m1c_bottom_tail_below_iv_descriptive_only",
    )
    decision_mismatch = expected_overall != overall
    result = {
        **CLAIMS,
        "sources_redownloaded": False,
        "null_samples_redrawn": False,
        "bootstrap_samples_redrawn": False,
        "row_identity_mismatches": row_identity_mismatches,
        "maximum_probability_difference": maximum_m1c_probability_difference,
        "maximum_m0_probability_difference": maximum_m0_probability_difference,
        "maximum_threshold_difference": maximum_threshold_difference,
        "maximum_decile_boundary_difference": maximum_decile_boundary_difference,
        "threshold_membership_mismatches": membership_mismatches,
        "fresh_episode_identity_mismatches": fresh_episode_identity_mismatches,
        "maximum_terminal_return_difference": maximum_terminal_return_difference,
        "maximum_excursion_difference": maximum_excursion_difference,
        "maximum_iv_residual_difference": maximum_iv_residual_difference,
        "maximum_null_metric_difference": maximum_null_metric_difference,
        "maximum_bootstrap_metric_difference": maximum_bootstrap_metric_difference,
        "decision_mismatch": decision_mismatch,
    }
    result["passed"] = bool(
        row_identity_mismatches == 0
        and maximum_m1c_probability_difference <= 1e-12
        and maximum_m0_probability_difference <= 1e-12
        and maximum_threshold_difference <= 1e-12
        and maximum_decile_boundary_difference <= 1e-12
        and membership_mismatches == 0
        and fresh_episode_identity_mismatches == 0
        and maximum_terminal_return_difference <= 1e-12
        and maximum_excursion_difference <= 1e-12
        and maximum_iv_residual_difference <= 1e-12
        and maximum_null_metric_difference <= 1e-12
        and maximum_bootstrap_metric_difference <= 1e-12
        and not decision_mismatch
    )
    return result


def write_plots(
    analytic: pd.DataFrame,
    m1c_episodes: pd.DataFrame,
    score_deciles: pd.DataFrame,
    containment: pd.DataFrame,
) -> list[str]:
    os.environ["MPLCONFIGDIR"] = str(PRIMARY / "_work_matplotlib")
    import matplotlib.pyplot as plt

    REPORTS.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for period, group in score_deciles.groupby("period", sort=True):
        axis.plot(
            group["decile"],
            group["movement_exceeds_iv_rate"],
            marker="o",
            label=str(period),
        )
    axis.set(
        xlabel="Frozen 2024 M1C probability decile",
        ylabel="Movement-exceeds-IV rate",
        title="Frozen score decile versus 15-minute movement-exceeds-IV rate",
        xticks=range(1, 11),
    )
    axis.legend()
    figure.tight_layout()
    path = REPORTS / "m1c_decile_movement_exceeds_iv.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    plot_paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    assessment = analytic.loc[analytic["period"].eq("assessment")]
    tail = assessment.loc[assessment["m1c_bottom_10_percent"].astype(bool)]
    axis.hist(
        assessment["terminal_iv_residual_15m"],
        bins=60,
        alpha=0.45,
        density=True,
        label="full population",
    )
    axis.hist(
        tail["terminal_iv_residual_15m"],
        bins=40,
        alpha=0.65,
        density=True,
        label="M1C bottom 10%",
    )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel="15-minute terminal IV residual",
        ylabel="Density",
        title="Bottom-tail versus full-population IV residuals",
    )
    axis.legend()
    figure.tight_layout()
    path = REPORTS / "bottom_tail_vs_population_iv_residual.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    plot_paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    comparison = containment.loc[
        containment["horizon_minutes"].eq(15)
        & containment["sigma_boundary"].isin([1.5, 2.0])
        & containment["population"].isin(
            ["M1C_bottom_10_fresh_episode", "M0_bottom_10_fresh_episode"]
        )
    ].copy()
    comparison["label"] = (
        comparison["period"].astype(str) + " " + comparison["sigma_boundary"].astype(str) + "σ"
    )
    pivot = comparison.pivot(
        index="label",
        columns="population",
        values="any_breach_rate",
    )
    pivot.plot(kind="bar", ax=axis)
    axis.set(
        xlabel="Period and symmetric IV boundary",
        ylabel="Any-breach rate",
        title="M1C versus M0 fresh-tail containment stress",
    )
    axis.legend(title="")
    figure.tight_layout()
    path = REPORTS / "m1c_vs_m0_containment_surprise.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    plot_paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for period, group in m1c_episodes.loc[m1c_episodes["tail"].eq("bottom_10_percent")].groupby(
        "period", sort=True
    ):
        axis.hist(
            group["excursion_sigma_ratio_15m"],
            bins=40,
            alpha=0.55,
            density=True,
            label=str(period),
        )
    axis.axvline(1.5, color="orange", linestyle="--", linewidth=1)
    axis.axvline(2.0, color="red", linestyle="--", linewidth=1)
    axis.set(
        xlabel="15-minute maximum excursion / IV sigma",
        ylabel="Density",
        title="Fresh quiet-episode maximum-excursion distribution",
    )
    axis.legend()
    figure.tight_layout()
    path = REPORTS / "fresh_episode_maximum_excursion.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    plot_paths.append(path)
    return [str(path.relative_to(REPO_ROOT)) for path in plot_paths]


def write_report(
    *,
    decision: Mapping[str, Any],
    threshold_artifact: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    containment: pd.DataFrame,
    matched_wins: Mapping[str, Mapping[str, int]],
    permutation_wins: Mapping[str, Mapping[str, int]],
    bootstrap: pd.DataFrame,
    panel_support: Mapping[str, Any],
    plot_paths: Sequence[str],
) -> None:
    lines = [
        "# Frozen Causal M1C Low-Movement Veto and Short-Premium Readiness Screen V0",
        "",
        f"Decision: `{decision['overall_decision']}`",
        "",
        "This is retrospective underlying-stock movement and range-containment research. "
        "It does not calculate option P&L, model intraday option quotes, or establish "
        "short-option profitability, execution realism, paper/live readiness, or a "
        "deployable strategy.",
        "",
        "## Frozen low-tail thresholds",
        "",
    ]
    for model in ("M1C", "M0"):
        thresholds = cast(Mapping[str, float], threshold_artifact[model])
        lines.append(
            f"- {model}: bottom 5% `{thresholds['bottom_5_percent']:.15f}`, "
            f"bottom 10% `{thresholds['bottom_10_percent']:.15f}`, "
            f"bottom 20% `{thresholds['bottom_20_percent']:.15f}`."
        )
    lines.extend(["", "## Binding checkpoint results", ""])
    for period in ("assessment", "stress"):
        checkpoint = _table_row(
            tables["checkpoint"],
            period=period,
            model="M1C",
            tail="bottom_10_percent",
        )
        lines.append(
            f"- {period}: {int(checkpoint['rows'])} rows, "
            f"{int(checkpoint['sessions'])} sessions, {int(checkpoint['stocks'])} stocks; "
            f"remains-below-IV {checkpoint['remains_below_iv_rate']:.2%}; "
            f"NPV lift {checkpoint['npv_lift']:.2%}; "
            f"mean/median IV residual {checkpoint['mean_iv_residual']:.6f}/"
            f"{checkpoint['median_iv_residual']:.6f}; "
            f"1.5σ/2.0σ excursion breach "
            f"{checkpoint['surprise_1_5_sigma_rate']:.2%}/"
            f"{checkpoint['surprise_2_0_sigma_rate']:.2%}."
        )
    lines.extend(["", "## Fresh quiet episodes", ""])
    for period in ("assessment", "stress"):
        episodes = _table_row(
            tables["episodes"],
            period=period,
            model="M1C",
            tail="bottom_10_percent",
        )
        support = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], panel_support["fresh_bottom_10_percent"])[period],
        )
        one_sigma = _table_row(
            containment,
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=1.0,
        )
        one_five = _table_row(
            containment,
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        two_sigma = _table_row(
            containment,
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=2.0,
        )
        lines.append(
            f"- {period}: {int(episodes['rows'])} episodes, support gate "
            f"`{'pass' if support['passed'] else 'fail'}`; remains-below-IV "
            f"{episodes['remains_below_iv_rate']:.2%}; mean/median residual "
            f"{episodes['mean_iv_residual']:.6f}/{episodes['median_iv_residual']:.6f}; "
            f"15-minute 1σ/1.5σ/2σ containment "
            f"{one_sigma['containment_rate']:.2%}/"
            f"{one_five['containment_rate']:.2%}/"
            f"{two_sigma['containment_rate']:.2%}."
        )
    lines.extend(["", "## Nulls and bootstrap", ""])
    for period in ("assessment", "stress"):
        lift_interval = _table_row(
            bootstrap,
            period=period,
            statistic="bottom_tail_npv_lift",
        )
        residual_interval = _table_row(
            bootstrap,
            period=period,
            statistic="mean_terminal_iv_residual",
        )
        lines.append(
            f"- {period}: matched-null wins on NPV lift/mean residual "
            f"{matched_wins[period]['npv_lift']}/20 and "
            f"{matched_wins[period]['mean_iv_residual']}/20; permutation wins "
            f"{permutation_wins[period]['npv_lift']}/10 and "
            f"{permutation_wins[period]['mean_iv_residual']}/10; "
            f"80% NPV-lift interval [{lift_interval['lower_80']:.6f}, "
            f"{lift_interval['upper_80']:.6f}]; 80% mean-residual interval "
            f"[{residual_interval['lower_80']:.6f}, "
            f"{residual_interval['upper_80']:.6f}]."
        )
    veto = cast(Mapping[str, Any], decision["binding_low_movement_veto_gate"])
    readiness = cast(Mapping[str, Any], decision["short_premium_readiness_gate"])
    lines.extend(
        [
            "",
            "## Binding gates",
            "",
            f"- Long-premium veto gate: `{'pass' if veto['passed'] else 'fail'}`.",
            "- Short-premium range-containment readiness gate: "
            f"`{'pass' if readiness['passed'] else 'fail'}`.",
            "- Prospective defined-risk short-premium shadow recording: "
            f"`{'prioritised' if readiness['passed'] else 'not prioritised'}`.",
            "- Naked short options, paper orders, live orders, and strategy deployment "
            "remain unauthorised.",
            "",
            "## Plots",
            "",
            *(f"- `{path}`" for path in plot_paths),
            "",
        ]
    )
    payload = "\n".join(lines)
    (PRIMARY / "report.md").write_text(payload, encoding="utf-8")
    (REPORTS / "report.md").write_text(payload, encoding="utf-8")


def run() -> dict[str, Any]:
    contract = load_contract()
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("reconstructing frozen causal M1C checkpoint panel", flush=True)
    predecessor = load_predecessor_runner()
    panel, states, source_manifest, reconstruction_details = reconstruct_m1c_panel(predecessor)
    write_json(PRIMARY / "source_manifest.json", source_manifest)
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            **CLAIMS,
            "passed": True,
            "protected_rows_read": 0,
            "protected_outcomes_materialised": False,
            "maximum_checkpoint_session": str(panel["session"].max()),
            "maximum_state_session_read": str(states["session"].astype(str).max()),
            "maximum_options_observation_date": str(
                panel["options_observation_date"].astype(str).max()
            ),
            "assessment_end": ASSESSMENT_END,
            "opened_stress_end": STRESS_END,
        },
    )
    panel = _checkpoint_context(panel, states)

    print("calculating terminal movement and path excursions", flush=True)
    movement, paths = calculate_checkpoint_outcomes(
        panel[
            [
                "row_id",
                "stock",
                "session",
                "checkpoint",
                "feature_available_timestamp_utc",
                "atm_iv",
            ]
        ],
        states,
    )
    movement_output = panel[["row_id", "period"]].merge(
        movement, on="row_id", how="right", validate="one_to_one"
    )
    path_output = panel[["row_id", "period"]].merge(
        paths, on="row_id", how="right", validate="one_to_one"
    )
    write_parquet(PRIMARY / "movement_outcomes.parquet", movement_output)
    write_parquet(PRIMARY / "path_excursion_outcomes.parquet", path_output)

    print("freezing 2024 low tails, deciles, and subgroup boundaries", flush=True)
    analytic, threshold_artifact, decile_artifact = freeze_thresholds_and_context(
        panel,
        movement,
        paths,
    )
    analytic = add_recent_high_m1c_context(analytic)
    feature_order = cast(
        Sequence[str],
        reconstruction_details["feature_manifest"]["feature_order"],
    )
    prediction_columns = [
        "row_id",
        "stock",
        "session",
        "period",
        "checkpoint",
        "checkpoint_timestamp_utc",
        "feature_available_timestamp_utc",
        "entry_timestamp",
        "row_weight",
        "stock_local_checkpoints_in_session",
        "required_options_date",
        "options_observation_date",
        "front_expiration_date",
        "option_dte",
        "atm_iv",
        "M0_probability",
        "M1C_probability",
        "m1c_bottom_5_percent",
        "m1c_bottom_10_percent",
        "m1c_bottom_20_percent",
        "m0_bottom_5_percent",
        "m0_bottom_10_percent",
        "m0_bottom_20_percent",
        "m1c_score_decile",
        "checkpoint_group",
        "month",
        "atm_iv_quartile",
        "previous_close_atm_iv_group",
        "stock_local_volatility",
        "stock_local_volatility_group",
        "market_volatility",
        "market_volatility_group",
        "market_volatility_fallback_used",
        "market_volatility_fallback_level",
        "market_return_through_checkpoint",
        "market_return_through_checkpoint_fallback_used",
        "market_return_through_checkpoint_fallback_level",
        "market_up_down",
        "market_quiet_active",
        "option_dte_group",
        *feature_order,
    ]
    prediction_columns = list(dict.fromkeys(prediction_columns))
    write_parquet(
        PRIMARY / "checkpoint_predictions.parquet",
        analytic[prediction_columns],
    )

    print("constructing fresh quiet episodes", flush=True)
    m1c_episodes, m0_episodes, _ = build_quiet_episodes(
        analytic,
        threshold_artifact,
    )
    tables = primary_metric_tables(analytic, m1c_episodes, m0_episodes)
    score_deciles, monotonicity = build_score_deciles(analytic)
    containment, _, surprise_summary = build_containment_and_surprises(
        analytic,
        m1c_episodes,
        m0_episodes,
    )

    print("running fixed matched and probability-permutation nulls", flush=True)
    matched_table, matched_wins, matched_plan = build_matched_random_nulls(analytic)
    permutation_table, permutation_wins, permutation_plan = build_probability_permutation_nulls(
        analytic, threshold_artifact
    )
    print("running fixed whole-session bootstrap", flush=True)
    bootstrap_table, bootstrap_plan = build_bootstrap(analytic, m1c_episodes)
    resampling_plan = {
        **CLAIMS,
        "matched_random": matched_plan,
        "probability_permutation": permutation_plan,
        "bootstrap": bootstrap_plan,
    }
    write_json(PRIMARY / "frozen_resampling_plan.json", resampling_plan)

    stability_tables, concentration_audit = build_stability_and_concentration(analytic)
    panel_support = build_panel_support(analytic, m1c_episodes)
    decision = build_decision(
        contract=contract,
        tables=tables,
        containment=containment,
        surprise_summary=surprise_summary,
        monotonicity=monotonicity,
        matched_wins=matched_wins,
        permutation_wins=permutation_wins,
        bootstrap=bootstrap_table,
        stability_tables=stability_tables,
        panel_support=panel_support,
    )
    decision["concentration_audit"] = concentration_audit
    prospective_path = write_prospective_artifact(decision)

    print("rebuilding deterministic probabilities, outcomes, nulls, and decision", flush=True)
    determinism = determinism_checks(
        analytic=analytic,
        states=states,
        threshold_artifact=threshold_artifact,
        decile_artifact=decile_artifact,
        feature_manifest=cast(Mapping[str, Any], reconstruction_details["feature_manifest"]),
        m1c_episodes=m1c_episodes,
        matched_table=matched_table,
        permutation_table=permutation_table,
        bootstrap_table=bootstrap_table,
        resampling_plan=resampling_plan,
        decision=decision,
    )
    write_json(PRIMARY / "determinism_check.json", determinism)
    if determinism["passed"]:
        decision["determinism_status"] = "supported"
    else:
        decision["determinism_status"] = "blocked"
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["long_premium_veto_status"] = "blocked"
        decision["short_premium_recorder_priority"] = "blocked"
    write_json(PRIMARY / "decision.json", decision)
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            **CLAIMS,
            "passed": bool(determinism["passed"]),
            "contract_passed": True,
            "m1c_reconstruction_passed": True,
            "previous_close_iv_chronology_passed": True,
            "protected_boundary_passed": True,
            "null_draw_counts_passed": len(MATCHED_SEEDS) == 20 and len(PERMUTATION_SEEDS) == 10,
            "bootstrap_draw_count_passed": BOOTSTRAP_DRAWS == 100,
            "determinism_passed": bool(determinism["passed"]),
            "independent_audit_status": "pending",
            "prospective_artifact": str(prospective_path.relative_to(REPO_ROOT)),
        },
    )
    plot_paths = write_plots(analytic, m1c_episodes, score_deciles, containment)
    write_report(
        decision=decision,
        threshold_artifact=threshold_artifact,
        tables=tables,
        containment=containment,
        matched_wins=matched_wins,
        permutation_wins=permutation_wins,
        bootstrap=bootstrap_table,
        panel_support=panel_support,
        plot_paths=plot_paths,
    )
    return {
        "decision": decision["overall_decision"],
        "m1c_reconstruction_rows": reconstruction_details["reconstruction"]["panel_rows"],
        "determinism_passed": determinism["passed"],
        "prospective_artifact": str(prospective_path),
    }


def write_blocker(contract: Mapping[str, Any], error: ScreenBlocked) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    statuses = {
        "m1c_reconstruction_status": "blocked",
        "low_tail_threshold_status": "blocked",
        "checkpoint_low_movement_status": "blocked",
        "fresh_quiet_episode_status": "blocked",
        "m0_comparison_status": "blocked",
        "score_monotonicity_status": "blocked",
        "surprise_mover_status": "blocked",
        "range_containment_status": "blocked",
        "long_premium_veto_status": "blocked",
        "short_premium_recorder_priority": "blocked",
    }
    decision = {
        **dict(contract),
        **CLAIMS,
        "overall_decision": error.decision,
        **statuses,
        "blocker": error.detail,
        "independent_audit_status": "blocked",
        "determinism_status": "blocked",
    }
    write_json(PRIMARY / "decision.json", decision)
    write_json(
        PRIMARY / "prospective_short_premium_recording_blocker.json",
        {
            **CLAIMS,
            "authorisation": "blocked",
            "overall_decision": error.decision,
            "blocker": error.detail,
        },
    )
    report = (
        "# Frozen Causal M1C Low-Movement Veto and Short-Premium Readiness Screen V0\n\n"
        f"Decision: `{error.decision}`\n\n{error.detail}\n\n"
        "No option P&L, paper/live order authority, or strategy promotion is implied.\n"
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the frozen experiment")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    contract: dict[str, Any] = {}
    try:
        contract = load_contract()
        result = run()
    except ScreenBlocked as error:
        write_blocker(contract or CLAIMS, error)
        print(json.dumps({"decision": error.decision, "detail": error.detail}, sort_keys=True))
        return 2
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
