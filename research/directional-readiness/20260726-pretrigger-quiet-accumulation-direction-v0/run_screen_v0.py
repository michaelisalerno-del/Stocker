#!/usr/bin/env python3
"""Run Pre-Trigger Quiet Accumulation / Distribution Direction Screen V0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stocker_research.movement_qualified_direction_v0 import (  # noqa: E402
    FrozenDirectionModel,
    assign_contiguous_session_folds,
    binary_direction_metrics,
    construct_fresh_episodes,
    fit_direction_model,
)
from stocker_research.pretrigger_quiet_accumulation_v0 import (  # noqa: E402
    ASSESSMENT_END,
    ASSESSMENT_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    EPSILON,
    GROUP_A,
    GROUP_C,
    GROUP_P,
    M1_THRESHOLD,
    MODEL_CATEGORICAL_FEATURES,
    PRIMARY_RAW_FEATURES,
    PRIMARY_WINDOW_BARS,
    Q0_NUMERIC_FEATURES,
    Q1_NUMERIC_FEATURES,
    QS_NUMERIC_FEATURES,
    QUIET_SIGNED_COMPONENTS,
    QuietScoreParameters,
    apply_quiet_score_parameters,
    attach_pretrigger_direction_targets,
    build_pretrigger_feature_rows,
    decide_pretrigger_candidate,
    fit_quiet_score_parameters,
    freeze_confidence_boundary,
    grouped_feature_permutation,
    label_null_within_slates,
    selective_actions,
    temporal_placebo_bundle,
    validate_authorized_sessions,
)

FloatArray = NDArray[np.float64]

EXPERIMENT = Path(__file__).resolve().parent
PRIMARY = EXPERIMENT / "artifacts" / "primary"
REPORTS = EXPERIMENT / "reports"
CONTRACT_PATH = EXPERIMENT / "contract.json"
PREDECESSOR = (
    ROOT / "research" / "directional-readiness" / "20260726-movement-qualified-direction-screen-v0"
)
PREDECESSOR_RUNNER = PREDECESSOR / "run_screen_v0.py"
SOURCE_PREDECESSOR_PRIMARY = (
    Path("/Users/michaelsalerno/Documents/Codex/2026-07-26-you-are-working-in-the-github")
    / "research"
    / "directional-readiness"
    / "20260726-movement-qualified-direction-screen-v0"
    / "artifacts"
    / "primary"
)
EXPECTED_BRANCH_C_SHA256: Final[str] = (
    "f62ef0144c12c813cbc665ba6d5ba1a235a6f77101a04b9f491c77b24c295529"
)
EXPECTED_STATE_SHA256: Final[str] = (
    "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0"
)
PERMUTATION_SEEDS: Final[tuple[int, ...]] = tuple(range(2026072601, 2026072621))
BOOTSTRAP_SEED: Final[int] = 2026072621
NULL_SEEDS: Final[tuple[int, ...]] = tuple(range(2026072631, 2026072636))
SCORE_BIN_LABELS: Final[tuple[str, ...]] = (
    "strong_distribution",
    "moderate_distribution",
    "neutral",
    "moderate_accumulation",
    "strong_accumulation",
)
MODEL_IDS: Final[tuple[str, ...]] = ("Q0", "QS", "Q1")
PRIMARY_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "stock",
    "session",
    "checkpoint",
    "signal_timestamp",
)
SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "retrospective_candidate_screen": True,
    "movement_model_frozen": True,
    "movement_model_refit_allowed": False,
    "m1_threshold": M1_THRESHOLD,
    "fresh_episode_definition_frozen": True,
    "direction_marker_precedes_trigger_bar": True,
    "trigger_bar_excluded_from_direction_features": True,
    "primary_pretrigger_window_bars": 5,
    "primary_direction_horizon_minutes": 10,
    "quiet_accumulation_and_distribution_mirrored": True,
    "direct_order_flow_claim": False,
    "activity_proxy_not_exchange_volume": True,
    "route_orientation_features_excluded": True,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "broker_access": False,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

RAW_QUIET_BUNDLE_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        (
            *PRIMARY_RAW_FEATURES,
            "net_return_25",
            "path_length_25",
            "range_sum_25",
            *(f"_pressure_sum_3bar_position_{position}" for position in range(PRIMARY_WINDOW_BARS)),
            *(f"_net_return_3bar_position_{position}" for position in range(PRIMARY_WINDOW_BARS)),
        )
    )
)


class ScreenBlocked(RuntimeError):
    """Fail-closed experiment blocker."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.15g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_identity(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].copy()
    for column in ordered.columns:
        if "timestamp" in column:
            ordered[column] = pd.to_datetime(ordered[column], utc=True, errors="raise").astype(str)
    payload = ordered.sort_values(list(columns), kind="mergesort").to_csv(
        index=False,
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def model_features() -> dict[str, tuple[str, ...]]:
    return {
        "Q0": Q0_NUMERIC_FEATURES,
        "QS": QS_NUMERIC_FEATURES,
        "Q1": Q1_NUMERIC_FEATURES,
    }


def load_contract() -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise ScreenBlocked(
                "blocked_chronology_or_leakage_failure",
                f"contract safety flag drifted: {key}",
            )
    return contract


@dataclass
class FrozenPreparation:
    """State frozen before constructing any 2025 assessment outcome."""

    contract: dict[str, Any]
    predecessor: ModuleType
    sources: dict[str, Any]
    bars: pd.DataFrame
    panel: pd.DataFrame
    episodes: pd.DataFrame
    identity_comparison: pd.DataFrame
    raw_features: pd.DataFrame
    development_raw_targets: pd.DataFrame
    development_oof: pd.DataFrame
    score_parameters: QuietScoreParameters
    scored_features: pd.DataFrame
    full_models: dict[str, FrozenDirectionModel]
    thresholds: dict[str, float]
    score_boundaries: dict[str, float]
    subgroup_boundaries: dict[str, float]
    movement_audit: dict[str, Any]
    episode_audit: dict[str, Any]
    pretrigger_audit: dict[str, Any]
    source_manifest: dict[str, Any]
    preprocessing_manifest: dict[str, Any]


def load_extended_bars(state_path: Path) -> pd.DataFrame:
    columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bar_log_return",
        "vti__bar_log_return",
        "historical_relative_activity",
        "feature_available_timestamp_max",
    ]
    bars = pd.read_parquet(
        state_path,
        columns=columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", ASSESSMENT_END),
        ],
    ).rename(columns={"symbol": "stock"})
    bars["stock"] = bars["stock"].astype(str)
    bars["session"] = bars["session"].astype(str)
    validate_authorized_sessions(bars["session"])
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    available = pd.to_datetime(bars["feature_available_timestamp_max"], utc=True, errors="coerce")
    if bool(available.gt(bars["bar_complete_timestamp"]).fillna(False).any()):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "a frozen state feature is available after its completed-bar close",
        )
    return bars.sort_values(["stock", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )


def align_exact_signed_pressure(
    bars: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "symbol",
        "session",
        "checkpoint",
        "signed_pressure",
        "feature_available_timestamp_utc",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            f"audited pressure source columns missing: {missing}",
        )
    snapshots = panel.loc[
        :,
        [
            "symbol",
            "session",
            "checkpoint",
            "signed_pressure",
            "feature_available_timestamp_utc",
        ],
    ].rename(
        columns={
            "symbol": "stock",
            "feature_available_timestamp_utc": "pressure_available_timestamp",
        }
    )
    snapshots["stock"] = snapshots["stock"].astype(str)
    snapshots["session"] = snapshots["session"].astype(str)
    snapshots["bar_ordinal"] = snapshots["checkpoint"].astype(int) - 1
    snapshots["pressure_available_timestamp"] = pd.to_datetime(
        snapshots["pressure_available_timestamp"], utc=True, errors="raise"
    )
    if snapshots.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "audited signed-pressure snapshot identities are not unique",
        )
    aligned = bars.merge(
        snapshots[
            [
                "stock",
                "session",
                "bar_ordinal",
                "checkpoint",
                "signed_pressure",
                "pressure_available_timestamp",
            ]
        ].rename(columns={"checkpoint": "pressure_source_checkpoint"}),
        on=["stock", "session", "bar_ordinal"],
        how="left",
        validate="one_to_one",
    ).sort_values(["stock", "session", "bar_ordinal"], kind="mergesort")
    pressure_available = pd.to_datetime(
        aligned["pressure_available_timestamp"], utc=True, errors="coerce"
    )
    if bool(pressure_available.gt(aligned["bar_complete_timestamp"]).fillna(False).any()):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "an exact signed-pressure snapshot came from the future",
        )
    audit = {
        "source_field": "signed_pressure",
        "source_snapshots": int(snapshots["signed_pressure"].notna().sum()),
        "source_checkpoint_values": sorted(snapshots["checkpoint"].astype(int).unique().tolist()),
        "alignment": "exact_checkpoint_bar_close_only",
        "interpolation": False,
        "forward_fill": False,
        "redefinition": False,
        "bars_without_exact_snapshot_missing": int(aligned["signed_pressure"].isna().sum()),
        "bars_with_exact_pressure_snapshot": int(aligned["signed_pressure"].notna().sum()),
        "future_pressure_rows": 0,
    }
    return aligned.reset_index(drop=True), audit


def load_and_reconstruct() -> tuple[
    dict[str, Any],
    ModuleType,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    contract = load_contract()
    predecessor = load_module(PREDECESSOR_RUNNER, "pretrigger_predecessor_runner")
    sources = cast(dict[str, Any], predecessor.load_frozen_inputs())
    branch_path = cast(Path, sources["branch_c_path"])
    state_path = cast(Path, sources["state_path"])
    if sha256_file(branch_path) != EXPECTED_BRANCH_C_SHA256:
        raise ScreenBlocked(
            "blocked_movement_episode_reconstruction_failure",
            "frozen Branch C source hash drifted",
        )
    if sha256_file(state_path) != EXPECTED_STATE_SHA256:
        raise ScreenBlocked(
            "blocked_movement_episode_reconstruction_failure",
            "frozen state-surface source hash drifted",
        )
    historical = cast(pd.DataFrame, sources["historical"])
    _, _, panel, movement_audit = predecessor.reconstruct_frozen_m1(historical)
    predecessor_episodes, episode_audit = predecessor.build_episode_panel(
        panel,
        cast(pd.DataFrame, sources["states"]),
    )
    panel = cast(pd.DataFrame, panel)
    predecessor_episodes = cast(pd.DataFrame, predecessor_episodes)
    movement_audit = cast(dict[str, Any], movement_audit)
    episode_audit = cast(dict[str, Any], episode_audit)
    if float(movement_audit["maximum_direct_probability_difference"]) > 1e-12:
        raise ScreenBlocked(
            "blocked_movement_episode_reconstruction_failure",
            "frozen M1 probabilities exceed the reconstruction tolerance",
        )
    expected_counts = {
        "raw_above_threshold_checkpoint_rows": 1266,
        "fresh_episodes": 538,
        "development_episodes": 285,
        "assessment_episodes": 253,
    }
    count_differences = {
        key: int(episode_audit[key]) - expected for key, expected in expected_counts.items()
    }
    if any(value != 0 for value in count_differences.values()):
        raise ScreenBlocked(
            "blocked_movement_episode_reconstruction_failure",
            f"frozen episode counts drifted: {count_differences}",
        )

    causal = panel.rename(columns={"symbol": "stock"}).copy()
    causal["stock"] = causal["stock"].astype(str)
    causal["session"] = causal["session"].astype(str)
    causal["partition"] = np.where(
        causal["session"].le(DEVELOPMENT_END), "development", "assessment"
    )
    state_times = cast(pd.DataFrame, sources["states"]).copy()
    state_times["stock"] = state_times["stock"].astype(str)
    state_times["session"] = state_times["session"].astype(str)
    signals = state_times[["stock", "session", "bar_ordinal", "bar_complete_timestamp"]].copy()
    signals["checkpoint"] = signals["bar_ordinal"].astype(int) + 1
    signals = signals.rename(columns={"bar_complete_timestamp": "signal_timestamp"})
    entries = state_times[["stock", "session", "bar_ordinal", "bar_start_timestamp"]].copy()
    entries["checkpoint"] = entries["bar_ordinal"].astype(int)
    entries = entries.rename(columns={"bar_start_timestamp": "prospective_entry_timestamp"})
    causal = causal.merge(
        signals[["stock", "session", "checkpoint", "signal_timestamp"]],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    ).merge(
        entries[["stock", "session", "checkpoint", "prospective_entry_timestamp"]],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    direct_input = causal[
        [
            "stock",
            "session",
            "checkpoint",
            "signal_timestamp",
            "prospective_entry_timestamp",
            "M1_probability",
            "partition",
        ]
    ].rename(columns={"M1_probability": "m1_probability"})
    direct_episodes = construct_fresh_episodes(direct_input)
    comparison = predecessor_episodes[[*PRIMARY_IDENTITY_COLUMNS, "partition"]].merge(
        direct_episodes[[*PRIMARY_IDENTITY_COLUMNS, "partition"]],
        on=list(PRIMARY_IDENTITY_COLUMNS),
        how="outer",
        indicator=True,
        suffixes=("_predecessor", "_direct"),
    )
    comparison["episode_identity_match"] = comparison["_merge"].eq("both")
    comparison["episode_identity"] = comparison.apply(
        lambda row: "|".join(
            (
                str(row["stock"]),
                str(row["session"]),
                str(int(row["checkpoint"])),
                pd.Timestamp(row["signal_timestamp"]).isoformat(),
            )
        ),
        axis=1,
    )
    mismatches = int((~comparison["episode_identity_match"]).sum())
    if mismatches:
        raise ScreenBlocked(
            "blocked_movement_episode_reconstruction_failure",
            f"fresh episode identity mismatches: {mismatches}",
        )

    bars = load_extended_bars(state_path)
    bars, pressure_audit = align_exact_signed_pressure(bars, panel)
    source_manifest = {
        "branch_c": {
            "path": str(branch_path),
            "sha256": sha256_file(branch_path),
            "role": "frozen_M1_Group_O_plus_Group_I_and_audited_pressure_snapshots",
        },
        "state_surface": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
            "role": "five_minute_OHLC_activity_proxy_market_proxy_and_causal_timestamps",
        },
        "predecessor_runner": {
            "path": str(PREDECESSOR_RUNNER),
            "sha256": sha256_file(PREDECESSOR_RUNNER),
        },
        "signed_pressure": pressure_audit,
        "activity_proxy": {
            "field": "historical_relative_activity",
            "label": "activity proxy",
            "confirmed_exchange_volume": False,
            "causal_definition": (
                "current activity divided by same-stock same-clock expanding "
                "prior-session mean with repository minimum history"
            ),
        },
        "market_proxy": {
            "symbol": "VTI",
            "field": "vti__bar_log_return",
            "causal": True,
        },
        "authorized_materialized_session_min": str(bars["session"].min()),
        "authorized_materialized_session_max": str(bars["session"].max()),
        "protected_outcomes_materialized": False,
        "intraday_option_quotes_used": False,
    }
    return (
        contract,
        predecessor,
        sources,
        bars,
        panel,
        predecessor_episodes,
        comparison,
        movement_audit,
        {**episode_audit, "episode_identity_mismatches": mismatches},
        source_manifest,
    )


def feature_formula_manifest() -> dict[str, Any]:
    return {
        "epsilon": EPSILON,
        "direction_marker": "completed_bar_T_minus_1_close",
        "trigger_bar": "completed_bar_T",
        "trigger_bar_excluded": True,
        "primary_window_ordinals": ["T-5", "T-4", "T-3", "T-2", "T-1"],
        "signed_return": "log(close_i / close_(i-1))",
        "market_return": "log(market_close_i / market_close_(i-1))",
        "relative_return": "r_i - market_r_i",
        "normalised_range": "(high_i - low_i) / close_(i-1)",
        "clv": "(2*close_i-high_i-low_i)/(high_i-low_i+epsilon)",
        "wick_asymmetry": (
            "(min(open_i,close_i)-low_i - (high_i-max(open_i,close_i)))/(high_i-low_i+epsilon)"
        ),
        "signed_pressure": (
            "exact audited signed_pressure snapshot on that bar close only; "
            "no interpolation or forward-fill; otherwise missing"
        ),
        "activity_proxy": "historical_relative_activity; not asserted exchange volume",
        "vwap": (
            "causal session cumulative typical-price VWAP weighted by the "
            "existing positive activity field"
        ),
        "vwap_distance": "(close_i-vwap_i)/(prior_completed_bar_atr_i+epsilon)",
        "prior_completed_bar_atr": ("shift(1) rolling mean over 14 completed true ranges by stock"),
        "break_failure_asymmetry": (
            "1[low_i<min(previous_6_lows)]*(close_i-low_i)/(range_i+epsilon) "
            "- 1[high_i>max(previous_6_highs)]*(high_i-close_i)/(range_i+epsilon)"
        ),
        "pressure_persistence": "mean(sign(signed_pressure_i)); zero contributes zero",
        "pressure_slope": "OLS slope of cumulative signed pressure over five bars",
        "signed_absorption_divergence": (
            "sign(pressure_sum_25)*(abs(robust_z(pressure_sum_25))-abs(robust_z(net_return_25)))"
        ),
        "activity_without_displacement": (
            "sign(pressure_sum_25)*mean(max(activity_proxy_i,0))"
            "*(1-min(1,abs(net_return_25)/(path_length_25+epsilon)))"
        ),
        "quietness": ("sigmoid(-robust_z(range_sum_25))*sigmoid(-robust_z(path_length_25))"),
        "quiet_absorption_score": (
            "quietness_25 * equal_weight_mean(13 development-standardised "
            "and [-3,+3]-clipped signed components)"
        ),
        "primary_target": "log(close_of_second_future_bar / first_post_trigger_bar_open)",
        "pre_entry_displacement": "log(entry_price / close_at_marker_T_minus_1)",
        "remaining_fraction": (
            "abs(post_entry_return)/(abs(pre_entry_signed_return)+abs(post_entry_return)+epsilon)"
        ),
    }


def feature_manifest() -> dict[str, Any]:
    return {
        "primary_window_bars": 5,
        "primary_window_minutes": 25,
        "secondary_windows_bars": [3, 9],
        "secondary_windows_minutes": [15, 45],
        "secondary_windows_role": "descriptive_only",
        "group_P": list(GROUP_P),
        "group_A": list(GROUP_A),
        "group_C": list(GROUP_C),
        "quiet_signed_components": list(QUIET_SIGNED_COMPONENTS),
        "Q0_numeric": list(Q0_NUMERIC_FEATURES),
        "QS_numeric": list(QS_NUMERIC_FEATURES),
        "Q1_numeric": list(Q1_NUMERIC_FEATURES),
        "categorical_fixed_effects": list(MODEL_CATEGORICAL_FEATURES),
        "missing_indicators": "one_per_numeric_model_feature",
        "route_orientation_features": [],
        "trigger_bar_direction_features": [],
        "post_trigger_features": [],
    }


def attach_safe_episode_metadata(
    raw_features: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only contemporaneous gate/context fields, never frozen outcomes."""

    metadata_columns = [
        "symbol",
        "session",
        "checkpoint",
        "atm_iv",
        "transition_probability",
        "M1_probability",
    ]
    metadata = panel.loc[:, metadata_columns].rename(
        columns={
            "symbol": "stock",
            "atm_iv": "_panel_atm_iv",
            "transition_probability": "_panel_transition_probability",
            "M1_probability": "_panel_m1_probability",
        }
    )
    metadata["stock"] = metadata["stock"].astype(str)
    metadata["session"] = metadata["session"].astype(str)
    output = raw_features.merge(
        metadata,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    for column in ("atm_iv", "transition_probability"):
        panel_column = f"_panel_{column}"
        if column in output.columns:
            existing = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
            attached = pd.to_numeric(output[panel_column], errors="coerce").to_numpy(float)
            valid = np.isfinite(existing) & np.isfinite(attached)
            if valid.any() and float(np.max(np.abs(existing[valid] - attached[valid]))) > 1e-12:
                raise ScreenBlocked(
                    "blocked_chronology_or_leakage_failure",
                    f"safe episode metadata drifted: {column}",
                )
            output[column] = output[column].where(output[column].notna(), output[panel_column])
        else:
            output[column] = output[panel_column]
    if output[["atm_iv", "transition_probability", "_panel_m1_probability"]].isna().all(axis=None):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "safe episode metadata failed to attach",
        )
    probability_difference = np.abs(
        pd.to_numeric(output["m1_probability"], errors="raise").to_numpy(float)
        - pd.to_numeric(output["_panel_m1_probability"], errors="raise").to_numpy(float)
    )
    if float(np.max(probability_difference)) > 1e-12:
        raise ScreenBlocked(
            "blocked_movement_episode_reconstruction_failure",
            "episode M1 probability drifted during metadata attachment",
        )
    return output.drop(
        columns=[
            "_panel_atm_iv",
            "_panel_transition_probability",
            "_panel_m1_probability",
        ]
    )


def build_pretrigger_audit(
    features: pd.DataFrame,
    episodes_before_history: int,
) -> dict[str, Any]:
    primary_present = features["primary_window_bars_present"].astype(int).eq(5)
    timestamps = pd.to_datetime(features["pretrigger_marker_timestamp"], utc=True, errors="raise")
    triggers = pd.to_datetime(features["trigger_timestamp"], utc=True, errors="raise")
    maximum = pd.to_datetime(
        features["maximum_direction_feature_timestamp"], utc=True, errors="raise"
    )
    expected_markers = triggers - pd.Timedelta(minutes=5)
    marker_mismatches = int((timestamps != expected_markers).sum())
    trigger_intrusions = int((maximum >= triggers).sum())
    window_mismatches = int((~primary_present).sum())
    if marker_mismatches or trigger_intrusions or window_mismatches:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "T-1 marker or trigger-bar exclusion audit failed",
        )
    return {
        "passed": True,
        "episodes_before_pretrigger_history_requirement": episodes_before_history,
        "episodes_retained": int(len(features)),
        "episodes_dropped": episodes_before_history - int(len(features)),
        "marker_equals_T_minus_1_close": True,
        "pretrigger_timestamp_mismatches": marker_mismatches,
        "trigger_bar_excluded": True,
        "trigger_bar_intrusions": trigger_intrusions,
        "primary_window_bars": 5,
        "primary_window_minutes": 25,
        "primary_window_mismatches": window_mismatches,
        "secondary_45m_missing_rows": int(features["net_return_45"].isna().sum()),
        "development_retained": int(features["partition"].astype(str).eq("development").sum()),
        "assessment_retained": int(features["partition"].astype(str).eq("assessment").sum()),
        "pressure_complete_primary_windows": int(
            features["signed_pressure_bars_present_25"].astype(int).eq(5).sum()
        ),
        "development_pressure_complete_primary_windows": int(
            (
                features["partition"].astype(str).eq("development")
                & features["signed_pressure_bars_present_25"].astype(int).eq(5)
            ).sum()
        ),
        "assessment_pressure_complete_primary_windows": int(
            (
                features["partition"].astype(str).eq("assessment")
                & features["signed_pressure_bars_present_25"].astype(int).eq(5)
            ).sum()
        ),
        "pressure_on_non_snapshot_bars_is_missing": True,
        "pressure_interpolation_or_forward_fill": False,
    }


def run_oof_stack(
    development_raw_targets: pd.DataFrame,
    *,
    target_values: pd.Series | None = None,
    model_prefix: str = "",
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Four complete-session folds with fold-specific score preprocessing."""

    development = development_raw_targets.copy().reset_index(drop=True)
    development["_row_order"] = np.arange(len(development), dtype=int)
    if target_values is not None:
        development["direction_up_10m"] = pd.to_numeric(
            target_values.reset_index(drop=True), errors="coerce"
        )
    development["oof_fold"] = assign_contiguous_session_folds(development["session"], folds=4)
    held_rows: list[pd.DataFrame] = []
    fold_manifests: dict[int, dict[str, Any]] = {}
    for fold in range(4):
        training_raw = development.loc[development["oof_fold"].ne(fold)].copy()
        held_raw = development.loc[development["oof_fold"].eq(fold)].copy()
        if set(training_raw["session"]).intersection(set(held_raw["session"])):
            raise ScreenBlocked(
                "blocked_chronology_or_leakage_failure",
                "an OOF fold split a complete session",
            )
        parameters = fit_quiet_score_parameters(training_raw)
        training = apply_quiet_score_parameters(training_raw, parameters)
        held = apply_quiet_score_parameters(held_raw, parameters)
        models: dict[str, FrozenDirectionModel] = {}
        for model_id, numeric in model_features().items():
            try:
                models[model_id] = fit_direction_model(
                    training,
                    target_column="direction_up_10m",
                    numeric_features=numeric,
                    categorical_features=MODEL_CATEGORICAL_FEATURES,
                    model_id=f"{model_prefix}{model_id}_fold_{fold}",
                )
            except (RuntimeError, ValueError) as error:
                raise ScreenBlocked(
                    "blocked_model_convergence_failure",
                    f"{model_id} fold {fold} failed: {error}",
                ) from error
            held[f"{model_id}_p_up"] = models[model_id].predict(held)
        held_rows.append(held)
        fold_manifests[fold] = {
            "held_sessions": sorted(held["session"].astype(str).unique().tolist()),
            "training_sessions": sorted(training["session"].astype(str).unique().tolist()),
            "training_rows": int(len(training)),
            "held_rows": int(len(held)),
            "score_parameters": parameters.as_dict(),
            "models": {model_id: model.as_dict() for model_id, model in models.items()},
        }
    output = pd.concat(held_rows, ignore_index=True).sort_values("_row_order", kind="mergesort")
    if len(output) != len(development) or output["_row_order"].duplicated().any():
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "development OOF coverage is not one prediction per episode",
        )
    output = output.drop(columns=["_row_order"]).reset_index(drop=True)
    return output, fold_manifests


def fit_full_stack(
    development: pd.DataFrame,
    score_parameters: QuietScoreParameters,
    *,
    model_prefix: str = "",
) -> tuple[pd.DataFrame, dict[str, FrozenDirectionModel]]:
    scored = apply_quiet_score_parameters(development, score_parameters)
    models: dict[str, FrozenDirectionModel] = {}
    for model_id, numeric in model_features().items():
        try:
            models[model_id] = fit_direction_model(
                scored,
                target_column="direction_up_10m",
                numeric_features=numeric,
                categorical_features=MODEL_CATEGORICAL_FEATURES,
                model_id=f"{model_prefix}{model_id}",
            )
        except (RuntimeError, ValueError) as error:
            raise ScreenBlocked(
                "blocked_model_convergence_failure",
                f"full {model_id} fit failed: {error}",
            ) from error
    return scored, models


def freeze_score_boundaries(scored_development: pd.DataFrame) -> dict[str, float]:
    score = pd.to_numeric(scored_development["quiet_absorption_score_25"], errors="raise").to_numpy(
        float
    )
    q20, q40, q60, q80 = np.quantile(score, [0.20, 0.40, 0.60, 0.80])
    return {
        "q20": float(q20),
        "q40": float(q40),
        "q60": float(q60),
        "q80": float(q80),
    }


def freeze_subgroup_boundaries(
    scored_development: pd.DataFrame,
) -> dict[str, float]:
    nonzero = scored_development.loc[scored_development["direction_up_10m"].notna()]

    def median(column: str) -> float:
        values = pd.to_numeric(scored_development[column], errors="coerce").dropna()
        return float(values.median())

    movement = pd.to_numeric(nonzero["absolute_log_return_10m"], errors="coerce").dropna()
    return {
        "prior_close_atm_iv_median": median("atm_iv"),
        "m1_probability_median": median("m1_probability"),
        "transition_probability_median": median("transition_probability"),
        "largest_absolute_movement_q75": float(movement.quantile(0.75, interpolation="linear")),
    }


def assign_score_bins(
    frame: pd.DataFrame,
    boundaries: Mapping[str, float],
) -> pd.Series:
    return pd.cut(
        pd.to_numeric(frame["quiet_absorption_score_25"], errors="coerce"),
        bins=[
            -math.inf,
            float(boundaries["q20"]),
            float(boundaries["q40"]),
            float(boundaries["q60"]),
            float(boundaries["q80"]),
            math.inf,
        ],
        labels=list(SCORE_BIN_LABELS),
        right=True,
        include_lowest=True,
        ordered=True,
    ).astype(str)


def prepare_frozen_experiment() -> FrozenPreparation:
    (
        contract,
        predecessor,
        sources,
        bars,
        panel,
        episodes,
        identity_comparison,
        movement_audit,
        episode_audit,
        source_manifest,
    ) = load_and_reconstruct()
    try:
        raw_features = build_pretrigger_feature_rows(episodes, bars)
    except ValueError as error:
        raise ScreenBlocked(
            "blocked_insufficient_pretrigger_history",
            str(error),
        ) from error
    raw_features = attach_safe_episode_metadata(raw_features, panel)
    pretrigger_audit = build_pretrigger_audit(raw_features, len(episodes))
    development_raw = raw_features.loc[
        raw_features["partition"].astype(str).eq("development")
    ].copy()
    development_raw_targets = attach_pretrigger_direction_targets(
        development_raw,
        bars,
    )
    development_oof, fold_manifests = run_oof_stack(development_raw_targets)
    score_parameters = fit_quiet_score_parameters(development_raw_targets)
    scored_development, full_models = fit_full_stack(
        development_raw_targets,
        score_parameters,
    )
    scored_features = apply_quiet_score_parameters(raw_features, score_parameters)
    thresholds = {
        model_id: freeze_confidence_boundary(
            development_oof[f"{model_id}_p_up"].to_numpy(float),
            target_coverage=0.35,
            minimum_actions=100,
        )
        for model_id in MODEL_IDS
    }
    score_boundaries = freeze_score_boundaries(scored_development)
    subgroup_boundaries = freeze_subgroup_boundaries(scored_development)
    preprocessing_manifest = {
        "fit_sessions_start": str(scored_development["session"].min()),
        "fit_sessions_end": str(scored_development["session"].max()),
        "fit_rows": int(len(scored_development)),
        "assessment_rows_used": 0,
        "oof_folds": fold_manifests,
        "full_score_parameters": score_parameters.as_dict(),
        "full_models": {model_id: model.as_dict() for model_id, model in full_models.items()},
        "score_bin_boundaries": score_boundaries,
        "subgroup_boundaries": subgroup_boundaries,
        "confidence_thresholds": thresholds,
    }
    return FrozenPreparation(
        contract=contract,
        predecessor=predecessor,
        sources=sources,
        bars=bars,
        panel=panel,
        episodes=episodes,
        identity_comparison=identity_comparison,
        raw_features=raw_features,
        development_raw_targets=development_raw_targets,
        development_oof=development_oof,
        score_parameters=score_parameters,
        scored_features=scored_features,
        full_models=full_models,
        thresholds=thresholds,
        score_boundaries=score_boundaries,
        subgroup_boundaries=subgroup_boundaries,
        movement_audit=movement_audit,
        episode_audit=episode_audit,
        pretrigger_audit=pretrigger_audit,
        source_manifest=source_manifest,
        preprocessing_manifest=preprocessing_manifest,
    )


def write_freeze_artifacts(preparation: FrozenPreparation) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONTRACT_PATH, PRIMARY / "contract.json")
    write_json(PRIMARY / "source_manifest.json", preparation.source_manifest)
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            "passed": True,
            "development_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
            "assessment_start": ASSESSMENT_START,
            "assessment_end": ASSESSMENT_END,
            "excluded_start": "2025-09-01",
            "excluded_end": "2025-12-31",
            "protected_start": "2026-01-01",
            "maximum_materialized_session": str(preparation.bars["session"].max()),
            "excluded_rows_materialized": 0,
            "protected_rows_read": 0,
            "protected_rows_materialized": 0,
        },
    )
    write_json(
        PRIMARY / "movement_model_reconstruction.json",
        {
            **preparation.movement_audit,
            "movement_probability_max_difference": preparation.movement_audit[
                "maximum_direct_probability_difference"
            ],
            "required_tolerance": 1e-12,
        },
    )
    write_json(PRIMARY / "episode_reconstruction.json", preparation.episode_audit)
    write_csv(
        PRIMARY / "episode_identity_comparison.csv",
        preparation.identity_comparison,
    )
    write_json(
        PRIMARY / "pretrigger_timestamp_audit.json",
        preparation.pretrigger_audit,
    )
    write_json(
        PRIMARY / "pretrigger_feature_manifest.json",
        feature_manifest(),
    )
    write_json(
        PRIMARY / "feature_formula_manifest.json",
        feature_formula_manifest(),
    )
    write_json(
        PRIMARY / "development_preprocessing.json",
        preparation.preprocessing_manifest,
    )
    write_parquet(
        PRIMARY / "development_oof_predictions.parquet",
        preparation.development_oof,
    )
    write_json(
        PRIMARY / "quiet_absorption_score_parameters.json",
        preparation.score_parameters.as_dict(),
    )
    write_json(
        PRIMARY / "frozen_direction_thresholds.json",
        {
            "source": "2024_complete_session_OOF_only",
            "target_action_coverage": 0.35,
            "minimum_development_actions": 100,
            "same_boundary_for_call_and_put": True,
            "boundaries": preparation.thresholds,
            "Q1_binding_boundary": preparation.thresholds["Q1"],
            "descriptive_probability_thresholds": [0.55, 0.60, 0.65],
        },
    )
    write_json(
        PRIMARY / "primary_candidate_freeze.json",
        {
            "primary_candidate": "Q1",
            "primary_window_bars": 5,
            "primary_window_minutes": 25,
            "primary_horizon_minutes": 10,
            "trigger_bar_excluded": True,
            "composite_weights": "equal",
            "model_family": "l2_logistic",
            "C": 0.25,
            "assessment_model_switching_allowed": False,
            "written_before_assessment_target_construction": True,
            "score_bin_boundaries": preparation.score_boundaries,
            "subgroup_boundaries": preparation.subgroup_boundaries,
        },
    )


def safe_binary_metrics(
    frame: pd.DataFrame,
    probability_column: str,
) -> dict[str, float | int]:
    valid = frame.loc[frame["direction_up_10m"].notna() & frame[probability_column].notna()].copy()
    if not len(valid):
        return {
            "log_loss": math.nan,
            "brier_score": math.nan,
            "auc": math.nan,
            "average_precision": math.nan,
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "matthews_correlation_coefficient": math.nan,
            "up_base_rate": math.nan,
            "predicted_up_rate": math.nan,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
            "expected_calibration_error": math.nan,
            "episodes": 0,
        }
    target = valid["direction_up_10m"].to_numpy(int)
    probabilities = np.clip(valid[probability_column].to_numpy(float), 1e-12, 1.0 - 1e-12)
    if len(np.unique(target)) == 2:
        try:
            return binary_direction_metrics(target, probabilities)
        except ValueError:
            pass
    hard = (probabilities >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "auc": (
            float(roc_auc_score(target, probabilities)) if len(np.unique(target)) == 2 else math.nan
        ),
        "average_precision": math.nan,
        "accuracy": float(accuracy_score(target, hard)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, hard))
            if len(np.unique(target)) == 2
            else math.nan
        ),
        "matthews_correlation_coefficient": math.nan,
        "up_base_rate": float(np.mean(target)),
        "predicted_up_rate": float(np.mean(hard)),
        "calibration_intercept": math.nan,
        "calibration_slope": math.nan,
        "expected_calibration_error": math.nan,
        "episodes": int(len(valid)),
    }


def selective_metrics(
    frame: pd.DataFrame,
    *,
    action_column: str,
    horizon_minutes: int,
) -> dict[str, float | int]:
    actions = frame[action_column].astype(str)
    selected = frame.loc[actions.ne("ABSTAIN")].copy()
    action_values = selected[action_column].astype(str).to_numpy()
    returns = pd.to_numeric(
        selected[f"signed_log_return_{horizon_minutes}m"], errors="coerce"
    ).to_numpy(float)
    sides = np.where(action_values == "CALL", 1.0, -1.0)
    aligned = sides * returns
    valid_direction = np.isfinite(returns) & (returns != 0.0)
    truth = (returns[valid_direction] > 0.0).astype(int)
    predictions = (sides[valid_direction] > 0.0).astype(int)
    accuracy = float(accuracy_score(truth, predictions)) if len(truth) else math.nan
    balanced = (
        float(balanced_accuracy_score(truth, predictions))
        if len(truth) and len(np.unique(truth)) == 2
        else math.nan
    )
    finite_aligned = aligned[np.isfinite(aligned)]
    ordered = np.sort(finite_aligned)
    trim = int(math.floor(0.10 * len(ordered)))
    trimmed = ordered[trim : len(ordered) - trim] if trim and len(ordered) > 2 * trim else ordered
    wins = finite_aligned[finite_aligned > 0.0]
    losses = finite_aligned[finite_aligned < 0.0]
    mean_win = float(np.mean(wins)) if len(wins) else math.nan
    mean_loss = float(np.mean(losses)) if len(losses) else math.nan
    calls = int(np.sum(action_values == "CALL"))
    puts = int(np.sum(action_values == "PUT"))
    call_mfe = pd.to_numeric(selected[f"call_mfe_{horizon_minutes}m"], errors="coerce").to_numpy(
        float
    )
    put_mfe = pd.to_numeric(selected[f"put_mfe_{horizon_minutes}m"], errors="coerce").to_numpy(
        float
    )
    call_mae = pd.to_numeric(selected[f"call_mae_{horizon_minutes}m"], errors="coerce").to_numpy(
        float
    )
    put_mae = pd.to_numeric(selected[f"put_mae_{horizon_minutes}m"], errors="coerce").to_numpy(
        float
    )
    favourable = np.where(sides > 0.0, call_mfe, put_mfe)
    adverse = np.where(sides > 0.0, call_mae, put_mae)
    mean_favourable = float(np.nanmean(favourable)) if len(favourable) else math.nan
    mean_adverse = float(np.nanmean(adverse)) if len(adverse) else math.nan
    return {
        "horizon_minutes": horizon_minutes,
        "total_episodes": int(len(frame)),
        "actions": int(len(selected)),
        "abstentions": int(len(frame) - len(selected)),
        "action_coverage": (float(len(selected) / len(frame)) if len(frame) else math.nan),
        "call_count": calls,
        "put_count": puts,
        "call_put_balance": (float(calls / (calls + puts)) if calls + puts else math.nan),
        "directional_accuracy": accuracy,
        "balanced_accuracy": balanced,
        "mean_aligned_return": (
            float(np.mean(finite_aligned)) if len(finite_aligned) else math.nan
        ),
        "median_aligned_return": (
            float(np.median(finite_aligned)) if len(finite_aligned) else math.nan
        ),
        "positive_aligned_return_rate": (
            float(np.mean(finite_aligned > 0.0)) if len(finite_aligned) else math.nan
        ),
        "trimmed_mean_aligned_return": (float(np.mean(trimmed)) if len(trimmed) else math.nan),
        "mean_winning_aligned_return": mean_win,
        "mean_losing_aligned_return": mean_loss,
        "payoff_ratio": (
            mean_win / abs(mean_loss)
            if math.isfinite(mean_win) and math.isfinite(mean_loss) and mean_loss < 0.0
            else math.nan
        ),
        "mean_maximum_favourable_excursion": mean_favourable,
        "mean_maximum_adverse_excursion": mean_adverse,
        "favourable_adverse_excursion_ratio": (
            mean_favourable / mean_adverse
            if math.isfinite(mean_favourable) and math.isfinite(mean_adverse) and mean_adverse > 0.0
            else math.nan
        ),
    }


def add_assessment_subgroups(
    assessment: pd.DataFrame,
    preparation: FrozenPreparation,
) -> pd.DataFrame:
    output = assessment.copy()
    output["month_group"] = output["session"].astype(str).str.slice(0, 7)
    output["score_bin"] = assign_score_bins(output, preparation.score_boundaries)
    checkpoint = output["checkpoint"].astype(int)
    output["checkpoint_group_frozen"] = np.select(
        [
            checkpoint.between(6, 14),
            checkpoint.between(16, 24),
            checkpoint.between(26, 34),
        ],
        ["early_6_14", "middle_16_24", "late_26_34"],
        default="outside_frozen_groups",
    )
    boundaries = preparation.subgroup_boundaries
    output["atm_iv_group_frozen"] = np.where(
        output["atm_iv"].astype(float) <= boundaries["prior_close_atm_iv_median"],
        "low_prior_close_atm_iv",
        "high_prior_close_atm_iv",
    )
    output["m1_probability_group_frozen"] = np.where(
        output["m1_probability"].astype(float) <= boundaries["m1_probability_median"],
        "low_m1_probability",
        "high_m1_probability",
    )
    output["transition_probability_group_frozen"] = np.where(
        output["transition_probability"].astype(float)
        <= boundaries["transition_probability_median"],
        "low_transition_probability",
        "high_transition_probability",
    )
    output["largest_absolute_movement_quartile"] = (
        output["absolute_log_return_10m"].astype(float)
        >= boundaries["largest_absolute_movement_q75"]
    )
    return output


def score_assessment(preparation: FrozenPreparation) -> pd.DataFrame:
    assessment_features = preparation.scored_features.loc[
        preparation.scored_features["partition"].astype(str).eq("assessment")
    ].copy()
    assessment = attach_pretrigger_direction_targets(
        assessment_features,
        preparation.bars,
    )
    for model_id, model in preparation.full_models.items():
        probability_column = f"{model_id}_p_up"
        action_column = f"{model_id}_action"
        assessment[probability_column] = model.predict(assessment)
        assessment[action_column] = selective_actions(
            assessment[probability_column].to_numpy(float),
            preparation.thresholds[model_id],
        )
    return add_assessment_subgroups(assessment, preparation)


def direction_model_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    model_rows: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        metrics = safe_binary_metrics(assessment, f"{model_id}_p_up")
        model_rows[model_id] = {
            "row_type": "model",
            "model_id": model_id,
            **metrics,
            "sessions": int(assessment["session"].nunique()),
            "stocks": int(assessment["stock"].nunique()),
            "month_groups": int(assessment["month_group"].nunique()),
            "exact_zero_returns_excluded": int(assessment["zero_return_10m"].astype(int).sum()),
        }
    rows = list(model_rows.values())
    for candidate, baseline in (("QS", "Q0"), ("Q1", "Q0"), ("Q1", "QS")):
        candidate_row = model_rows[candidate]
        baseline_row = model_rows[baseline]
        delta: dict[str, Any] = {
            "row_type": "increment",
            "model_id": f"{candidate}_minus_{baseline}",
            "sessions": candidate_row["sessions"],
            "stocks": candidate_row["stocks"],
            "month_groups": candidate_row["month_groups"],
            "episodes": candidate_row["episodes"],
        }
        for metric in (
            "log_loss",
            "brier_score",
            "expected_calibration_error",
        ):
            delta[metric] = float(baseline_row[metric]) - float(candidate_row[metric])
        for metric in (
            "auc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "matthews_correlation_coefficient",
        ):
            delta[metric] = float(candidate_row[metric]) - float(baseline_row[metric])
        rows.append(delta)
    return pd.DataFrame(rows)


def selective_policy_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for horizon in (5, 10, 15, 30):
            rows.append(
                {
                    "model_id": model_id,
                    "policy_id": "frozen_development_oof_35pct",
                    **selective_metrics(
                        assessment,
                        action_column=f"{model_id}_action",
                        horizon_minutes=horizon,
                    ),
                }
            )
    for threshold in (0.55, 0.60, 0.65):
        sensitivity = assessment.copy()
        sensitivity["_sensitivity_action"] = selective_actions(
            sensitivity["Q1_p_up"].to_numpy(float),
            threshold - 0.5,
        )
        rows.append(
            {
                "model_id": "Q1",
                "policy_id": f"descriptive_probability_{threshold:.2f}",
                **selective_metrics(
                    sensitivity,
                    action_column="_sensitivity_action",
                    horizon_minutes=10,
                ),
            }
        )
    return pd.DataFrame(rows)


def baseline_metric_table(
    assessment: pd.DataFrame,
    development: pd.DataFrame,
) -> pd.DataFrame:
    development_prior = float(development["direction_up_10m"].dropna().astype(float).mean())
    frame = assessment.copy()
    definitions: dict[str, tuple[FloatArray, FloatArray]] = {}
    episodes = len(frame)
    definitions["B0_development_prior"] = (
        np.full(episodes, development_prior, dtype=float),
        np.full(episodes, 1 if development_prior >= 0.5 else -1, dtype=float),
    )
    definitions["B1_always_up"] = (
        np.ones(episodes, dtype=float),
        np.ones(episodes, dtype=float),
    )
    definitions["B2_five_minute_momentum"] = (
        (frame["stock_return_5m_tminus1"].to_numpy(float) > 0.0).astype(float),
        np.where(frame["stock_return_5m_tminus1"].to_numpy(float) > 0.0, 1.0, -1.0),
    )
    definitions["B3_ten_minute_momentum"] = (
        (frame["stock_return_10m_tminus1"].to_numpy(float) > 0.0).astype(float),
        np.where(frame["stock_return_10m_tminus1"].to_numpy(float) > 0.0, 1.0, -1.0),
    )
    definitions["B4_market_direction"] = (
        (frame["market_return_10m_tminus1"].to_numpy(float) > 0.0).astype(float),
        np.where(frame["market_return_10m_tminus1"].to_numpy(float) > 0.0, 1.0, -1.0),
    )
    relative_10 = frame["stock_minus_market_return_10m_tminus1"].to_numpy(float)
    definitions["B5_relative_strength_direction"] = (
        (relative_10 > 0.0).astype(float),
        np.where(relative_10 > 0.0, 1.0, -1.0),
    )
    rows: list[dict[str, Any]] = []
    target_valid = frame["direction_up_10m"].notna().to_numpy()
    returns = frame["signed_log_return_10m"].to_numpy(float)
    for baseline_id, (probabilities, sides) in definitions.items():
        temporary = frame.assign(_baseline_probability=probabilities)
        metrics = safe_binary_metrics(temporary, "_baseline_probability")
        aligned = sides * returns
        rows.append(
            {
                "baseline_id": baseline_id,
                **metrics,
                "directional_accuracy": float(
                    np.mean(
                        (sides[target_valid] > 0.0)
                        == frame.loc[target_valid, "direction_up_10m"].to_numpy(bool)
                    )
                ),
                "balanced_accuracy_direction": float(
                    balanced_accuracy_score(
                        frame.loc[target_valid, "direction_up_10m"].to_numpy(int),
                        (sides[target_valid] > 0.0).astype(int),
                    )
                ),
                "mean_aligned_return": float(np.mean(aligned)),
                "median_aligned_return": float(np.median(aligned)),
                "positive_aligned_return_rate": float(np.mean(aligned > 0.0)),
                "development_up_prior": development_prior,
            }
        )
    return pd.DataFrame(rows)


def action_correct_mask(frame: pd.DataFrame, action_column: str) -> pd.Series:
    side = np.where(frame[action_column].astype(str).eq("CALL"), 1, -1)
    returns = frame["signed_log_return_10m"].to_numpy(float)
    return pd.Series(
        (side == np.sign(returns).astype(int)) & (returns != 0.0),
        index=frame.index,
    )


def remaining_movement_table(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    actioned = assessment.loc[assessment["Q1_action"].astype(str).ne("ABSTAIN")].copy()
    actioned["_correct"] = action_correct_mask(actioned, "Q1_action")
    partitions: list[tuple[str, pd.DataFrame]] = [
        ("all_actions", actioned),
        ("CALL", actioned.loc[actioned["Q1_action"].astype(str).eq("CALL")]),
        ("PUT", actioned.loc[actioned["Q1_action"].astype(str).eq("PUT")]),
        ("correct_direction", actioned.loc[actioned["_correct"]]),
        ("incorrect_direction", actioned.loc[~actioned["_correct"]]),
        (
            "largest_absolute_movement_quartile",
            actioned.loc[actioned["largest_absolute_movement_quartile"].astype(bool)],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for subgroup, subset in partitions:
        row: dict[str, Any] = {
            "subgroup": subgroup,
            "episodes": int(len(subset)),
            "mean_pre_entry_signed_return": float(subset["pre_entry_signed_return"].mean())
            if len(subset)
            else math.nan,
            "median_pre_entry_signed_return": float(subset["pre_entry_signed_return"].median())
            if len(subset)
            else math.nan,
            "mean_absolute_pre_entry_displacement": float(
                subset["pre_entry_signed_return"].abs().mean()
            )
            if len(subset)
            else math.nan,
            "median_absolute_pre_entry_displacement": float(
                subset["pre_entry_signed_return"].abs().median()
            )
            if len(subset)
            else math.nan,
        }
        for horizon in (10, 30):
            values = pd.to_numeric(subset[f"remaining_fraction_{horizon}m"], errors="coerce")
            row[f"mean_remaining_fraction_{horizon}m"] = (
                float(values.mean()) if len(values) else math.nan
            )
            row[f"median_remaining_fraction_{horizon}m"] = (
                float(values.median()) if len(values) else math.nan
            )
            row[f"episodes_at_least_50pct_remaining_{horizon}m"] = (
                float(values.ge(0.50).mean()) if len(values) else math.nan
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    all_mean = float(
        table.loc[
            table["subgroup"].eq("all_actions"),
            "mean_remaining_fraction_10m",
        ].iloc[0]
    )
    return table, bool(all_mean < 0.50)


def material_move_table(assessment: pd.DataFrame) -> pd.DataFrame:
    actioned = assessment.loc[assessment["Q1_action"].astype(str).ne("ABSTAIN")].copy()
    actioned["_correct"] = action_correct_mask(actioned, "Q1_action")
    partitions = [
        ("ten_minute_iv_excess", actioned["iv_excess_10m"].eq(1)),
        ("ten_minute_non_iv_excess", actioned["iv_excess_10m"].eq(0)),
        (
            "largest_absolute_movement_quartile",
            actioned["largest_absolute_movement_quartile"].astype(bool),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for subgroup, mask in partitions:
        subset = actioned.loc[mask].copy()
        metrics = selective_metrics(
            subset,
            action_column="Q1_action",
            horizon_minutes=10,
        )
        rows.append(
            {
                "subgroup": subgroup,
                "episodes": int(len(subset)),
                "accuracy": metrics["directional_accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "mean_aligned_return": metrics["mean_aligned_return"],
                "median_aligned_return": metrics["median_aligned_return"],
                "positive_aligned_return_rate": metrics["positive_aligned_return_rate"],
                "mean_remaining_fraction": float(subset["remaining_fraction_10m"].mean())
                if len(subset)
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def score_monotonic_slope(frame: pd.DataFrame) -> float:
    bin_order = {label: index for index, label in enumerate(SCORE_BIN_LABELS)}
    x_values = frame["score_bin"].astype(str).map(bin_order).to_numpy(float)
    y_values = frame["signed_log_return_10m"].to_numpy(float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2 or np.var(x_values[valid]) <= 0.0:
        return math.nan
    return float(np.polyfit(x_values[valid], y_values[valid], 1)[0])


def score_bin_metric_table(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    means: list[float] = []
    for order, label in enumerate(SCORE_BIN_LABELS):
        subset = assessment.loc[assessment["score_bin"].astype(str).eq(label)]
        mean_return = float(subset["signed_log_return_10m"].mean()) if len(subset) else math.nan
        means.append(mean_return)
        rows.append(
            {
                "score_bin": label,
                "bin_order": order,
                "episodes": int(len(subset)),
                "mean_future_signed_return_10m": mean_return,
                "median_future_signed_return_10m": float(subset["signed_log_return_10m"].median())
                if len(subset)
                else math.nan,
                "up_rate": float(subset["direction_up_10m"].mean())
                if subset["direction_up_10m"].notna().any()
                else math.nan,
                "mean_absolute_movement_10m": float(subset["absolute_log_return_10m"].mean())
                if len(subset)
                else math.nan,
                "iv_excess_rate": float(subset["iv_excess_10m"].mean())
                if subset["iv_excess_10m"].notna().any()
                else math.nan,
                "mean_remaining_fraction_10m": float(subset["remaining_fraction_10m"].mean())
                if len(subset)
                else math.nan,
            }
        )
    finite_means = np.asarray(means, dtype=float)
    strict_ordering = bool(np.isfinite(finite_means).all() and np.all(np.diff(finite_means) >= 0.0))
    diagnostics = {
        "monotonic_slope": score_monotonic_slope(assessment),
        "strict_non_decreasing_bin_means": strict_ordering,
        "strong_distribution_mean_less_than_strong_accumulation": bool(
            np.isfinite(finite_means[[0, 4]]).all() and finite_means[0] < finite_means[4]
        ),
        "correct_signed_monotonic_direction": strict_ordering,
    }
    return pd.DataFrame(rows), diagnostics


def grouped_permutation_table(
    assessment: pd.DataFrame,
    q1_model: FrozenDirectionModel,
    boundary: float,
) -> pd.DataFrame:
    real_primary = safe_binary_metrics(assessment, "Q1_p_up")
    real_selective = selective_metrics(
        assessment,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    groups: dict[str, tuple[str, ...]] = {
        "Group_P_persistent_pressure": GROUP_P,
        "Group_A_absorption_response": GROUP_A,
        "Group_C_compression_context": GROUP_C,
        "quiet_absorption_score_25": ("quiet_absorption_score_25",),
    }
    rows: list[dict[str, Any]] = []
    for group_id, columns in groups.items():
        for permutation_number, seed in enumerate(PERMUTATION_SEEDS, start=1):
            permuted = grouped_feature_permutation(
                assessment,
                feature_columns=columns,
                group_columns=("session", "checkpoint"),
                seed=seed,
            )
            permuted["permuted_p_up"] = q1_model.predict(permuted)
            permuted["permuted_action"] = selective_actions(
                permuted["permuted_p_up"].to_numpy(float),
                boundary,
            )
            primary = safe_binary_metrics(permuted, "permuted_p_up")
            selective = selective_metrics(
                permuted,
                action_column="permuted_action",
                horizon_minutes=10,
            )
            rows.append(
                {
                    "row_type": "permutation",
                    "group_id": group_id,
                    "permutation": permutation_number,
                    "seed": seed,
                    "log_loss_deterioration": float(primary["log_loss"])
                    - float(real_primary["log_loss"]),
                    "brier_deterioration": float(primary["brier_score"])
                    - float(real_primary["brier_score"]),
                    "auc_deterioration": float(real_primary["auc"]) - float(primary["auc"]),
                    "selective_accuracy_deterioration": float(
                        real_selective["directional_accuracy"]
                    )
                    - float(selective["directional_accuracy"]),
                    "mean_aligned_return_deterioration": float(
                        real_selective["mean_aligned_return"]
                    )
                    - float(selective["mean_aligned_return"]),
                    "median_aligned_return_deterioration": float(
                        real_selective["median_aligned_return"]
                    )
                    - float(selective["median_aligned_return"]),
                }
            )
    table = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for group_id, subset in table.groupby("group_id", sort=False):
        summary: dict[str, Any] = {
            "row_type": "mean_over_20",
            "group_id": group_id,
            "permutation": 20,
            "seed": math.nan,
        }
        for column in (
            "log_loss_deterioration",
            "brier_deterioration",
            "auc_deterioration",
            "selective_accuracy_deterioration",
            "mean_aligned_return_deterioration",
            "median_aligned_return_deterioration",
        ):
            summary[column] = float(subset[column].mean())
        summaries.append(summary)
    return pd.concat([table, pd.DataFrame(summaries)], ignore_index=True)


def combined_primary_selective_metrics(
    subset: pd.DataFrame,
    model_id: str,
) -> dict[str, Any]:
    primary = safe_binary_metrics(subset, f"{model_id}_p_up")
    selective = selective_metrics(
        subset,
        action_column=f"{model_id}_action",
        horizon_minutes=10,
    )
    return {
        **{f"primary_{key}": value for key, value in primary.items()},
        **{f"selective_{key}": value for key, value in selective.items()},
    }


def monthly_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    month_labels = (
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
    )
    rows: list[dict[str, Any]] = []
    for month in month_labels:
        subset = assessment.loc[assessment["month_group"].astype(str).eq(month)]
        for model_id in MODEL_IDS:
            rows.append(
                {
                    "month_group": month,
                    "model_id": model_id,
                    **combined_primary_selective_metrics(subset, model_id),
                }
            )
    return pd.DataFrame(rows)


def subgroup_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    subgroups: list[tuple[str, str, pd.Series]] = []
    for value in ("early_6_14", "middle_16_24", "late_26_34"):
        subgroups.append(
            (
                "checkpoint",
                value,
                assessment["checkpoint_group_frozen"].astype(str).eq(value),
            )
        )
    for column, dimension in (
        ("atm_iv_group_frozen", "prior_close_atm_iv"),
        ("m1_probability_group_frozen", "m1_probability"),
        ("transition_probability_group_frozen", "transition_probability"),
    ):
        for value in sorted(assessment[column].astype(str).unique()):
            subgroups.append((dimension, value, assessment[column].astype(str).eq(value)))
    for value in (
        "strong_accumulation",
        "strong_distribution",
        "neutral",
    ):
        subgroups.append(("quiet_score", value, assessment["score_bin"].astype(str).eq(value)))
    subgroups.extend(
        [
            (
                "Q1_decision",
                "CALL",
                assessment["Q1_action"].astype(str).eq("CALL"),
            ),
            (
                "Q1_decision",
                "PUT",
                assessment["Q1_action"].astype(str).eq("PUT"),
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    for dimension, value, mask in subgroups:
        subset = assessment.loc[mask]
        for model_id in MODEL_IDS:
            rows.append(
                {
                    "dimension": dimension,
                    "subgroup": value,
                    "model_id": model_id,
                    **combined_primary_selective_metrics(subset, model_id),
                }
            )
    return pd.DataFrame(rows)


def stock_and_concentration_tables(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actioned = assessment.loc[assessment["Q1_action"].astype(str).ne("ABSTAIN")].copy()
    actioned["_side"] = np.where(actioned["Q1_action"].astype(str).eq("CALL"), 1.0, -1.0)
    actioned["_aligned"] = actioned["_side"] * actioned["signed_log_return_10m"].astype(float)
    actioned["_correct"] = action_correct_mask(actioned, "Q1_action")
    total_positive = float(actioned["_aligned"].clip(lower=0.0).sum())
    total_negative = float(abs(actioned["_aligned"].clip(upper=0.0).sum()))
    rows: list[dict[str, Any]] = []
    for stock, all_stock in assessment.groupby("stock", sort=True):
        selected = actioned.loc[actioned["stock"].astype(str).eq(str(stock))]
        aligned = selected["_aligned"].to_numpy(float)
        rows.append(
            {
                "stock": stock,
                "episodes": int(len(all_stock)),
                "actions": int(len(selected)),
                "mean_quiet_absorption_score": float(all_stock["quiet_absorption_score_25"].mean()),
                "call_count": int(selected["Q1_action"].astype(str).eq("CALL").sum()),
                "put_count": int(selected["Q1_action"].astype(str).eq("PUT").sum()),
                "direction_accuracy": float(selected["_correct"].mean())
                if len(selected)
                else math.nan,
                "mean_aligned_return": float(np.mean(aligned)) if len(aligned) else math.nan,
                "median_aligned_return": float(np.median(aligned)) if len(aligned) else math.nan,
                "positive_aligned_return_rate": float(np.mean(aligned > 0.0))
                if len(aligned)
                else math.nan,
                "mean_remaining_fraction": float(selected["remaining_fraction_10m"].mean())
                if len(selected)
                else math.nan,
                "contribution_to_positive_aligned_return": (
                    float(np.maximum(aligned, 0.0).sum()) / total_positive
                    if total_positive > 0.0
                    else math.nan
                ),
                "contribution_to_negative_aligned_return": (
                    float(abs(np.minimum(aligned, 0.0).sum())) / total_negative
                    if total_negative > 0.0
                    else math.nan
                ),
            }
        )
    stock_table = pd.DataFrame(rows)
    leave_rows: list[dict[str, Any]] = []
    for stock in sorted(actioned["stock"].astype(str).unique()):
        subset = actioned.loc[actioned["stock"].astype(str).ne(stock)]
        aligned = subset["_aligned"].to_numpy(float)
        leave_rows.append(
            {
                "metric": "leave_one_stock_out",
                "excluded_stock": stock,
                "value": math.nan,
                "accuracy": float(subset["_correct"].mean()) if len(subset) else math.nan,
                "mean_aligned_return": float(np.mean(aligned)) if len(aligned) else math.nan,
                "median_aligned_return": float(np.median(aligned)) if len(aligned) else math.nan,
            }
        )

    def maximum_share(frame: pd.DataFrame, column: str) -> float:
        counts = frame[column].astype(str).value_counts()
        return float(counts.max() / len(frame)) if len(frame) else math.nan

    summary_rows = [
        {
            "metric": "maximum_stock_share_of_episodes",
            "excluded_stock": "",
            "value": maximum_share(assessment, "stock"),
        },
        {
            "metric": "maximum_stock_share_of_actions",
            "excluded_stock": "",
            "value": maximum_share(actioned, "stock"),
        },
        {
            "metric": "maximum_month_share_of_actions",
            "excluded_stock": "",
            "value": maximum_share(actioned, "month_group"),
        },
        {
            "metric": "maximum_session_share_of_actions",
            "excluded_stock": "",
            "value": maximum_share(actioned, "session"),
        },
    ]
    concentration = pd.concat(
        [pd.DataFrame(summary_rows), pd.DataFrame(leave_rows)],
        ignore_index=True,
    )
    return stock_table, concentration


def bootstrap_metric_values(sample: pd.DataFrame) -> dict[str, float]:
    q0 = safe_binary_metrics(sample, "Q0_p_up")
    q1 = safe_binary_metrics(sample, "Q1_p_up")
    selective = selective_metrics(
        sample,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    actioned = sample.loc[sample["Q1_action"].astype(str).ne("ABSTAIN")]
    iv_subset = sample.loc[sample["iv_excess_10m"].eq(1)]
    largest_subset = sample.loc[sample["largest_absolute_movement_quartile"].astype(bool)]
    iv_metrics = selective_metrics(
        iv_subset,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    largest_metrics = selective_metrics(
        largest_subset,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    return {
        "q1_minus_q0_log_loss_improvement": float(q0["log_loss"]) - float(q1["log_loss"]),
        "q1_minus_q0_brier_improvement": float(q0["brier_score"]) - float(q1["brier_score"]),
        "q1_minus_q0_auc_improvement": float(q1["auc"]) - float(q0["auc"]),
        "q1_selective_action_coverage": float(selective["action_coverage"]),
        "q1_selective_accuracy": float(selective["directional_accuracy"]),
        "q1_selective_balanced_accuracy": float(selective["balanced_accuracy"]),
        "mean_aligned_ten_minute_return": float(selective["mean_aligned_return"]),
        "median_aligned_ten_minute_return": float(selective["median_aligned_return"]),
        "positive_aligned_return_rate": float(selective["positive_aligned_return_rate"]),
        "mean_remaining_fraction": float(actioned["remaining_fraction_10m"].mean())
        if len(actioned)
        else math.nan,
        "quiet_absorption_score_monotonic_slope": score_monotonic_slope(sample),
        "iv_excess_subgroup_accuracy": float(iv_metrics["directional_accuracy"]),
        "largest_movement_quartile_accuracy": float(largest_metrics["directional_accuracy"]),
    }


def bootstrap_table_and_plan(
    assessment: pd.DataFrame,
    *,
    frozen_draw_sessions: Sequence[Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sessions = np.asarray(sorted(assessment["session"].astype(str).unique()), dtype=object)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw_sessions: list[list[str]] = []
    rows: list[dict[str, Any]] = []
    for draw in range(1, 101):
        if frozen_draw_sessions is None:
            chosen = generator.choice(sessions, size=len(sessions), replace=True)
            labels = [str(value) for value in chosen.tolist()]
        else:
            labels = [str(value) for value in frozen_draw_sessions[draw - 1]]
            if len(labels) != len(sessions):
                raise RuntimeError("frozen bootstrap draw has the wrong session count")
        draw_sessions.append(labels)
        parts = [
            assessment.loc[assessment["session"].astype(str).eq(session)].copy()
            for session in labels
        ]
        sample = pd.concat(parts, ignore_index=True)
        rows.append(
            {
                "row_type": "draw",
                "draw": draw,
                "interval_level": math.nan,
                "bound": "",
                "metric": "",
                "value": math.nan,
                **bootstrap_metric_values(sample),
            }
        )
    draw_table = pd.DataFrame(rows)
    interval_rows: list[dict[str, Any]] = []
    metric_columns = list(bootstrap_metric_values(assessment))
    for level in (0.80, 0.90, 0.95):
        alpha = (1.0 - level) / 2.0
        for metric in metric_columns:
            values = pd.to_numeric(draw_table[metric], errors="coerce").dropna()
            lower = float(values.quantile(alpha, interpolation="linear"))
            upper = float(values.quantile(1.0 - alpha, interpolation="linear"))
            interval_rows.extend(
                [
                    {
                        "row_type": "interval",
                        "draw": math.nan,
                        "interval_level": level,
                        "bound": "lower",
                        "metric": metric,
                        "value": lower,
                    },
                    {
                        "row_type": "interval",
                        "draw": math.nan,
                        "interval_level": level,
                        "bound": "upper",
                        "metric": metric,
                        "value": upper,
                    },
                ]
            )
    plan = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": 100,
        "session_universe": sessions.tolist(),
        "draw_sessions": draw_sessions,
    }
    return pd.concat([draw_table, pd.DataFrame(interval_rows)], ignore_index=True), plan


def run_oof_q1(
    development_raw_targets: pd.DataFrame,
    *,
    target_values: pd.Series | None = None,
    model_prefix: str,
) -> pd.DataFrame:
    development = development_raw_targets.copy().reset_index(drop=True)
    development["_row_order"] = np.arange(len(development), dtype=int)
    if target_values is not None:
        development["direction_up_10m"] = target_values.reset_index(drop=True)
    development["oof_fold"] = assign_contiguous_session_folds(development["session"], folds=4)
    outputs: list[pd.DataFrame] = []
    for fold in range(4):
        training_raw = development.loc[development["oof_fold"].ne(fold)]
        held_raw = development.loc[development["oof_fold"].eq(fold)].copy()
        parameters = fit_quiet_score_parameters(training_raw)
        training = apply_quiet_score_parameters(training_raw, parameters)
        held = apply_quiet_score_parameters(held_raw, parameters)
        model = fit_direction_model(
            training,
            target_column="direction_up_10m",
            numeric_features=Q1_NUMERIC_FEATURES,
            categorical_features=MODEL_CATEGORICAL_FEATURES,
            model_id=f"{model_prefix}_fold_{fold}",
        )
        held["Q1_p_up"] = model.predict(held)
        outputs.append(held)
    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values("_row_order", kind="mergesort")
        .drop(columns=["_row_order"])
        .reset_index(drop=True)
    )


def temporal_placebo_table(
    preparation: FrozenPreparation,
    real_assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    placebo_raw = temporal_placebo_bundle(
        preparation.raw_features,
        RAW_QUIET_BUNDLE_COLUMNS,
    )
    placebo_development_raw = placebo_raw.loc[
        placebo_raw["partition"].astype(str).eq("development")
    ]
    placebo_development = attach_pretrigger_direction_targets(
        placebo_development_raw,
        preparation.bars,
    )
    placebo_oof = run_oof_q1(
        placebo_development,
        model_prefix="temporal_placebo_Q1",
    )
    placebo_boundary = freeze_confidence_boundary(
        placebo_oof["Q1_p_up"].to_numpy(float),
        target_coverage=0.35,
        minimum_actions=100,
    )
    placebo_parameters = fit_quiet_score_parameters(placebo_development)
    placebo_scored_development = apply_quiet_score_parameters(
        placebo_development,
        placebo_parameters,
    )
    placebo_model = fit_direction_model(
        placebo_scored_development,
        target_column="direction_up_10m",
        numeric_features=Q1_NUMERIC_FEATURES,
        categorical_features=MODEL_CATEGORICAL_FEATURES,
        model_id="temporal_placebo_Q1",
    )
    placebo_assessment_raw = placebo_raw.loc[placebo_raw["partition"].astype(str).eq("assessment")]
    placebo_assessment = apply_quiet_score_parameters(
        placebo_assessment_raw,
        placebo_parameters,
    )
    placebo_assessment = attach_pretrigger_direction_targets(
        placebo_assessment,
        preparation.bars,
    )
    placebo_assessment["Q1_p_up"] = placebo_model.predict(placebo_assessment)
    placebo_assessment["Q1_action"] = selective_actions(
        placebo_assessment["Q1_p_up"].to_numpy(float),
        placebo_boundary,
    )
    real_primary = safe_binary_metrics(real_assessment, "Q1_p_up")
    placebo_primary = safe_binary_metrics(placebo_assessment, "Q1_p_up")
    real_selective = selective_metrics(
        real_assessment,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    placebo_selective = selective_metrics(
        placebo_assessment,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    rows = [
        {
            "model": "real_Q1",
            **real_primary,
            **{f"selective_{key}": value for key, value in real_selective.items()},
            "confidence_boundary": preparation.thresholds["Q1"],
        },
        {
            "model": "temporally_misaligned_placebo_Q1",
            **placebo_primary,
            **{f"selective_{key}": value for key, value in placebo_selective.items()},
            "confidence_boundary": placebo_boundary,
        },
    ]
    comparison = {
        "real_log_loss_improvement": float(placebo_primary["log_loss"])
        - float(real_primary["log_loss"]),
        "real_auc_improvement": float(real_primary["auc"]) - float(placebo_primary["auc"]),
        "real_mean_aligned_return_improvement": float(real_selective["mean_aligned_return"])
        - float(placebo_selective["mean_aligned_return"]),
    }
    comparison["real_q1_outperforms_temporal_placebo"] = bool(
        comparison["real_log_loss_improvement"] > 0.0
        and comparison["real_auc_improvement"] > 0.0
        and comparison["real_mean_aligned_return_improvement"] > 0.0
    )
    return pd.DataFrame(rows), comparison


def prediction_monotonic_slope(frame: pd.DataFrame, probability_column: str) -> float:
    order = {label: index for index, label in enumerate(SCORE_BIN_LABELS)}
    x_values = frame["score_bin"].astype(str).map(order).to_numpy(float)
    y_values = frame[probability_column].to_numpy(float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2 or np.var(x_values[valid]) <= 0.0:
        return math.nan
    return float(np.polyfit(x_values[valid], y_values[valid], 1)[0])


def direction_null_table(
    preparation: FrozenPreparation,
    assessment: pd.DataFrame,
    *,
    frozen_null_targets: Mapping[str, Sequence[object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    real_primary = safe_binary_metrics(assessment, "Q1_p_up")
    real_selective = selective_metrics(
        assessment,
        action_column="Q1_action",
        horizon_minutes=10,
    )
    real_monotonicity = prediction_monotonic_slope(assessment, "Q1_p_up")
    rows: list[dict[str, Any]] = []
    assignment_hashes: dict[str, str] = {}
    serialized_targets: dict[str, list[float | None]] = {}
    for null_number, seed in enumerate(NULL_SEEDS, start=1):
        if frozen_null_targets is None:
            null_development = label_null_within_slates(
                preparation.development_raw_targets,
                target_column="direction_up_10m",
                seed=seed,
            )
        else:
            null_development = preparation.development_raw_targets.copy()
            frozen = pd.Series(frozen_null_targets[str(null_number)], dtype=float)
            if len(frozen) != len(null_development):
                raise RuntimeError("frozen null assignment has the wrong row count")
            null_development["direction_up_10m"] = frozen.to_numpy(float)
        serialized_targets[str(null_number)] = [
            None if not math.isfinite(float(value)) else float(value)
            for value in pd.to_numeric(
                null_development["direction_up_10m"], errors="coerce"
            ).to_numpy(float)
        ]
        assignment_hashes[str(null_number)] = frame_identity(
            null_development.assign(null_target=null_development["direction_up_10m"]),
            ["stock", "session", "checkpoint", "null_target"],
        )
        null_oof, _ = run_oof_stack(
            preparation.development_raw_targets,
            target_values=null_development["direction_up_10m"],
            model_prefix=f"null_{null_number}_",
        )
        null_scored = apply_quiet_score_parameters(
            null_development,
            preparation.score_parameters,
        )
        for model_id, numeric in model_features().items():
            null_model = fit_direction_model(
                null_scored,
                target_column="direction_up_10m",
                numeric_features=numeric,
                categorical_features=MODEL_CATEGORICAL_FEATURES,
                model_id=f"null_{null_number}_{model_id}",
            )
            probability_column = f"null_{model_id}_p_up"
            action_column = f"null_{model_id}_action"
            null_assessment = assessment.copy()
            null_assessment[probability_column] = null_model.predict(null_assessment)
            boundary = freeze_confidence_boundary(
                null_oof[f"{model_id}_p_up"].to_numpy(float),
                target_coverage=0.35,
                minimum_actions=100,
            )
            null_assessment[action_column] = selective_actions(
                null_assessment[probability_column].to_numpy(float),
                boundary,
            )
            primary = safe_binary_metrics(
                null_assessment,
                probability_column,
            )
            selective = selective_metrics(
                null_assessment,
                action_column=action_column,
                horizon_minutes=10,
            )
            rows.append(
                {
                    "null_number": null_number,
                    "seed": seed,
                    "model_id": model_id,
                    **primary,
                    "selective_accuracy": selective["directional_accuracy"],
                    "mean_aligned_return": selective["mean_aligned_return"],
                    "quiet_score_prediction_monotonicity": (
                        prediction_monotonic_slope(null_assessment, probability_column)
                    ),
                    "confidence_boundary": boundary,
                }
            )
    table = pd.DataFrame(rows)
    q1 = table.loc[table["model_id"].eq("Q1")].copy()
    counts = {
        "real_exceeds_log_loss": int((float(real_primary["log_loss"]) < q1["log_loss"]).sum()),
        "real_exceeds_brier": int((float(real_primary["brier_score"]) < q1["brier_score"]).sum()),
        "real_exceeds_auc": int((float(real_primary["auc"]) > q1["auc"]).sum()),
        "real_exceeds_selective_accuracy": int(
            (float(real_selective["directional_accuracy"]) > q1["selective_accuracy"]).sum()
        ),
        "real_exceeds_mean_aligned_return": int(
            (float(real_selective["mean_aligned_return"]) > q1["mean_aligned_return"]).sum()
        ),
        "real_exceeds_quiet_score_monotonicity": int(
            (real_monotonicity > q1["quiet_score_prediction_monotonicity"]).sum()
        ),
    }
    logloss_or_auc = (float(real_primary["log_loss"]) < q1["log_loss"]) | (
        float(real_primary["auc"]) > q1["auc"]
    )
    counts["real_exceeds_log_loss_or_auc"] = int(logloss_or_auc.sum())
    counts["null_gate_passed"] = bool(
        counts["real_exceeds_log_loss_or_auc"] >= 4
        and counts["real_exceeds_mean_aligned_return"] >= 4
    )
    plan = {
        "null_seeds": list(NULL_SEEDS),
        "label_assignment_hashes": assignment_hashes,
        "null_targets": serialized_targets,
        "within": "development_session_x_checkpoint",
    }
    return table, counts, plan


def support_gates(
    preparation: FrozenPreparation,
    assessment: pd.DataFrame,
) -> dict[str, Any]:
    development = preparation.development_raw_targets
    development_labels = development["direction_up_10m"].dropna().astype(int)
    assessment_labels = assessment["direction_up_10m"].dropna().astype(int)
    development_months = development["session"].astype(str).str.slice(0, 7)
    assessment_months = assessment["month_group"].astype(str)
    assessment_stock_share = assessment["stock"].astype(str).value_counts(normalize=True)
    assessment_month_share = assessment_months.value_counts(normalize=True)
    development_gate_items = {
        "episodes_at_least_220": len(development) >= 220,
        "sessions_at_least_60": development["session"].nunique() >= 60,
        "stocks_at_least_15": development["stock"].nunique() >= 15,
        "months_at_least_10": development_months.nunique() >= 10,
        "up_at_least_90": int(development_labels.sum()) >= 90,
        "down_at_least_90": int((1 - development_labels).sum()) >= 90,
    }
    assessment_gate_items = {
        "episodes_at_least_180": len(assessment) >= 180,
        "sessions_at_least_45": assessment["session"].nunique() >= 45,
        "stocks_at_least_15": assessment["stock"].nunique() >= 15,
        "all_eight_month_groups": assessment_months.nunique() == 8,
        "up_at_least_75": int(assessment_labels.sum()) >= 75,
        "down_at_least_75": int((1 - assessment_labels).sum()) >= 75,
        "no_stock_above_15pct": bool(
            len(assessment_stock_share) and assessment_stock_share.max() <= 0.15
        ),
        "no_month_above_25pct": bool(
            len(assessment_month_share) and assessment_month_share.max() <= 0.25
        ),
    }
    actioned = assessment.loc[assessment["Q1_action"].astype(str).ne("ABSTAIN")]
    action_stock_share = actioned["stock"].astype(str).value_counts(normalize=True)
    action_month_share = actioned["month_group"].astype(str).value_counts(normalize=True)
    action_session_share = actioned["session"].astype(str).value_counts(normalize=True)
    selective_gate_items = {
        "actions_at_least_80": len(actioned) >= 80,
        "sessions_at_least_30": actioned["session"].nunique() >= 30,
        "stocks_at_least_12": actioned["stock"].nunique() >= 12,
        "month_groups_at_least_6": actioned["month_group"].nunique() >= 6,
        "calls_at_least_25": int(actioned["Q1_action"].astype(str).eq("CALL").sum()) >= 25,
        "puts_at_least_25": int(actioned["Q1_action"].astype(str).eq("PUT").sum()) >= 25,
        "no_stock_above_20pct": bool(len(action_stock_share) and action_stock_share.max() <= 0.20),
        "no_month_above_30pct": bool(len(action_month_share) and action_month_share.max() <= 0.30),
        "no_session_above_8pct": bool(
            len(action_session_share) and action_session_share.max() <= 0.08
        ),
    }
    return {
        "development": {
            "passed": all(development_gate_items.values()),
            "items": development_gate_items,
            "episodes": int(len(development)),
            "sessions": int(development["session"].nunique()),
            "stocks": int(development["stock"].nunique()),
            "months": int(development_months.nunique()),
            "up": int(development_labels.sum()),
            "down": int((1 - development_labels).sum()),
        },
        "assessment": {
            "passed": all(assessment_gate_items.values()),
            "items": assessment_gate_items,
            "episodes": int(len(assessment)),
            "sessions": int(assessment["session"].nunique()),
            "stocks": int(assessment["stock"].nunique()),
            "months": int(assessment_months.nunique()),
            "up": int(assessment_labels.sum()),
            "down": int((1 - assessment_labels).sum()),
            "maximum_stock_share": float(assessment_stock_share.max()),
            "maximum_month_share": float(assessment_month_share.max()),
        },
        "selective": {
            "passed": all(selective_gate_items.values()),
            "items": selective_gate_items,
            "actions": int(len(actioned)),
            "sessions": int(actioned["session"].nunique()),
            "stocks": int(actioned["stock"].nunique()),
            "months": int(actioned["month_group"].nunique()),
            "calls": int(actioned["Q1_action"].astype(str).eq("CALL").sum()),
            "puts": int(actioned["Q1_action"].astype(str).eq("PUT").sum()),
            "maximum_stock_share": float(action_stock_share.max())
            if len(action_stock_share)
            else math.nan,
            "maximum_month_share": float(action_month_share.max())
            if len(action_month_share)
            else math.nan,
            "maximum_session_share": float(action_session_share.max())
            if len(action_session_share)
            else math.nan,
        },
    }


def interval_value(
    bootstrap: pd.DataFrame,
    *,
    metric: str,
    level: float,
    bound: str,
) -> float:
    values = bootstrap.loc[
        bootstrap["row_type"].eq("interval")
        & bootstrap["metric"].eq(metric)
        & bootstrap["interval_level"].eq(level)
        & bootstrap["bound"].eq(bound),
        "value",
    ]
    if len(values) != 1:
        raise RuntimeError(f"bootstrap interval missing: {metric}|{level}|{bound}")
    return float(values.iloc[0])


def permutation_component_status(
    permutation: pd.DataFrame,
    group_id: str,
    *,
    require_auc_or_accuracy: bool,
) -> tuple[bool, dict[str, float]]:
    summary = permutation.loc[
        permutation["row_type"].eq("mean_over_20") & permutation["group_id"].eq(group_id)
    ]
    if len(summary) != 1:
        raise RuntimeError(f"permutation summary missing: {group_id}")
    values = {
        column: float(summary.iloc[0][column])
        for column in (
            "log_loss_deterioration",
            "brier_deterioration",
            "auc_deterioration",
            "selective_accuracy_deterioration",
            "mean_aligned_return_deterioration",
            "median_aligned_return_deterioration",
        )
    }
    proper = values["log_loss_deterioration"] > 0.0 or values["brier_deterioration"] > 0.0
    discrimination = values["auc_deterioration"] > 0.0 or (
        require_auc_or_accuracy and values["selective_accuracy_deterioration"] > 0.0
    )
    supported = bool(
        proper and discrimination and values["mean_aligned_return_deterioration"] > 0.0
    )
    return supported, values


def coefficient_for(
    model: FrozenDirectionModel,
    feature: str,
) -> float:
    try:
        index = model.design_feature_names.index(feature)
    except ValueError:
        return math.nan
    return float(model.coefficients[index])


def build_decision(
    preparation: FrozenPreparation,
    assessment: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    selective_metrics_table: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    score_diagnostics: Mapping[str, Any],
    score_bins: pd.DataFrame,
    permutation: pd.DataFrame,
    placebo_comparison: Mapping[str, Any],
    monthly_metrics: pd.DataFrame,
    concentration_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null_summary: Mapping[str, Any],
    support: Mapping[str, Any],
    late_direction_problem: bool,
) -> dict[str, Any]:
    q0 = direction_metrics.loc[
        direction_metrics["row_type"].eq("model") & direction_metrics["model_id"].eq("Q0")
    ].iloc[0]
    qs = direction_metrics.loc[
        direction_metrics["row_type"].eq("model") & direction_metrics["model_id"].eq("QS")
    ].iloc[0]
    q1 = direction_metrics.loc[
        direction_metrics["row_type"].eq("model") & direction_metrics["model_id"].eq("Q1")
    ].iloc[0]
    q1_selective = selective_metrics_table.loc[
        selective_metrics_table["model_id"].eq("Q1")
        & selective_metrics_table["policy_id"].eq("frozen_development_oof_35pct")
        & selective_metrics_table["horizon_minutes"].eq(10)
    ].iloc[0]
    required_baselines = baseline_metrics.loc[
        baseline_metrics["baseline_id"].isin(
            [
                "B3_ten_minute_momentum",
                "B4_market_direction",
                "B5_relative_strength_direction",
            ]
        )
    ]
    beats_baselines = bool(
        float(q1_selective["directional_accuracy"])
        > float(required_baselines["directional_accuracy"].max())
    )
    q1_months = monthly_metrics.loc[monthly_metrics["model_id"].eq("Q1")]
    positive_months = int(
        (pd.to_numeric(q1_months["selective_mean_aligned_return"], errors="coerce") > 0.0).sum()
    )
    concentration_values = concentration_metrics.loc[
        concentration_metrics["metric"].str.startswith("maximum_")
    ].set_index("metric")["value"]
    concentration_gates = bool(
        float(concentration_values["maximum_stock_share_of_episodes"]) <= 0.15
        and float(concentration_values["maximum_stock_share_of_actions"]) <= 0.20
        and float(concentration_values["maximum_month_share_of_actions"]) <= 0.30
        and float(concentration_values["maximum_session_share_of_actions"]) <= 0.08
    )
    pressure_supported, pressure_effects = permutation_component_status(
        permutation,
        "Group_P_persistent_pressure",
        require_auc_or_accuracy=False,
    )
    absorption_supported, absorption_effects = permutation_component_status(
        permutation,
        "Group_A_absorption_response",
        require_auc_or_accuracy=True,
    )
    context_supported, context_effects = permutation_component_status(
        permutation,
        "Group_C_compression_context",
        require_auc_or_accuracy=True,
    )
    _, score_effects = permutation_component_status(
        permutation,
        "quiet_absorption_score_25",
        require_auc_or_accuracy=True,
    )
    strong_support = bool(
        int(score_bins.loc[score_bins["score_bin"].eq("strong_distribution"), "episodes"].iloc[0])
        >= 25
        and int(
            score_bins.loc[score_bins["score_bin"].eq("strong_accumulation"), "episodes"].iloc[0]
        )
        >= 25
    )
    qs_improves = bool(
        float(qs["log_loss"]) < float(q0["log_loss"])
        or float(qs["accuracy"]) > float(baseline_metrics["directional_accuracy"].max())
    )
    composite_supported = bool(
        qs_improves
        and bool(score_diagnostics["correct_signed_monotonic_direction"])
        and strong_support
    )
    pressure_complete_assessment = int(
        assessment["signed_pressure_bars_present_25"].astype(int).eq(5).sum()
    )
    pressure_complete_development = int(
        preparation.pretrigger_audit["development_pressure_complete_primary_windows"]
    )
    full_pressure_history_supported = bool(
        pressure_complete_development >= 220 and pressure_complete_assessment >= 180
    )
    pressure_support_adequate = pressure_complete_assessment >= 25
    persistent_pressure_status = (
        "supported"
        if pressure_supported and pressure_support_adequate
        else "promising"
        if pressure_supported
        else "not_supported"
    )
    absorption_status = (
        "supported"
        if absorption_supported
        else "promising"
        if (
            absorption_effects["mean_aligned_return_deterioration"] > 0.0
            and (
                absorption_effects["auc_deterioration"] > 0.0
                or absorption_effects["selective_accuracy_deterioration"] > 0.0
            )
        )
        else "not_supported"
    )
    relative_coefficient = coefficient_for(preparation.full_models["Q1"], "relative_resilience_25")
    vwap_coefficients = [
        coefficient_for(preparation.full_models["Q1"], feature)
        for feature in (
            "mean_vwap_distance_25",
            "vwap_side_balance_25",
            "vwap_reclaim_balance_25",
        )
    ]
    relative_status = (
        "supported"
        if context_supported and relative_coefficient > 0.0
        else "promising"
        if relative_coefficient > 0.0
        else "not_supported"
    )
    vwap_status = (
        "supported"
        if context_supported and sum(value > 0.0 for value in vwap_coefficients) >= 2
        else "promising"
        if sum(value > 0.0 for value in vwap_coefficients) >= 2
        else "not_supported"
    )
    bootstrap_accuracy_lower = interval_value(
        bootstrap,
        metric="q1_selective_accuracy",
        level=0.80,
        bound="lower",
    )
    bootstrap_return_lower = interval_value(
        bootstrap,
        metric="mean_aligned_ten_minute_return",
        level=0.80,
        bound="lower",
    )
    primary_gates = {
        "1_q1_improves_log_loss_vs_q0": float(q1["log_loss"]) < float(q0["log_loss"]),
        "2_q1_improves_brier_vs_q0": float(q1["brier_score"]) < float(q0["brier_score"]),
        "3_q1_auc_at_least_0_55": float(q1["auc"]) >= 0.55,
        "4_q1_balanced_accuracy_above_0_52": float(q1["balanced_accuracy"]) > 0.52,
        "5_selective_coverage_between_20_and_50pct": 0.20
        <= float(q1_selective["action_coverage"])
        <= 0.50,
        "6_selective_accuracy_at_least_57pct": float(q1_selective["directional_accuracy"]) >= 0.57,
        "7_selective_accuracy_beats_required_baselines": beats_baselines,
        "8_mean_aligned_return_positive": float(q1_selective["mean_aligned_return"]) > 0.0,
        "9_median_aligned_return_positive": float(q1_selective["median_aligned_return"]) > 0.0,
        "10_bootstrap_80_accuracy_lower_above_50pct": (bootstrap_accuracy_lower > 0.50),
        "11_bootstrap_80_mean_return_lower_nonnegative": (bootstrap_return_lower >= 0.0),
        "12_positive_mean_return_in_six_of_eight_months": positive_months >= 6,
        "13_real_exceeds_four_of_five_nulls": bool(null_summary["null_gate_passed"]),
        "14_real_q1_outperforms_temporal_placebo": bool(
            placebo_comparison["real_q1_outperforms_temporal_placebo"]
        ),
        "15_score_bins_correct_signed_monotonic_direction": bool(
            score_diagnostics["correct_signed_monotonic_direction"]
        ),
        "16_assessment_and_selective_support": bool(
            cast(Mapping[str, Any], support["assessment"])["passed"]
            and cast(Mapping[str, Any], support["selective"])["passed"]
        ),
        "17_concentration_gates": concentration_gates,
        "18_late_direction_problem_false": not late_direction_problem,
    }
    evidence = {
        "development_support_passed": cast(Mapping[str, Any], support["development"])["passed"],
        "assessment_support_passed": cast(Mapping[str, Any], support["assessment"])["passed"],
        "selective_support_passed": cast(Mapping[str, Any], support["selective"])["passed"],
        "concentration_gates_passed": concentration_gates,
        "q1_log_loss_improves": primary_gates["1_q1_improves_log_loss_vs_q0"],
        "q1_brier_improves": primary_gates["2_q1_improves_brier_vs_q0"],
        "q1_auc": float(q1["auc"]),
        "q1_balanced_accuracy": float(q1["balanced_accuracy"]),
        "action_coverage": float(q1_selective["action_coverage"]),
        "selective_accuracy": float(q1_selective["directional_accuracy"]),
        "beats_required_baselines": beats_baselines,
        "mean_aligned_return_10m": float(q1_selective["mean_aligned_return"]),
        "median_aligned_return_10m": float(q1_selective["median_aligned_return"]),
        "bootstrap_80_accuracy_lower": bootstrap_accuracy_lower,
        "bootstrap_80_mean_return_lower": bootstrap_return_lower,
        "positive_month_groups": positive_months,
        "null_gate_passed": bool(null_summary["null_gate_passed"]),
        "temporal_placebo_gate_passed": bool(
            placebo_comparison["real_q1_outperforms_temporal_placebo"]
        ),
        "score_monotonic_direction_correct": bool(
            score_diagnostics["correct_signed_monotonic_direction"]
        ),
        "late_direction_problem": late_direction_problem,
        "persistent_pressure_supported": pressure_supported,
        "absorption_response_supported": absorption_supported,
        "absorption_response_promising": absorption_status == "promising",
        "score_descriptive_only": (
            score_effects["mean_aligned_return_deterioration"] > 0.0
            or score_diagnostics["strong_distribution_mean_less_than_strong_accumulation"]
        )
        and not composite_supported,
        "stability_failed": positive_months < 6,
    }
    if not full_pressure_history_supported:
        evidence["blocker"] = "blocked_insufficient_pretrigger_history"
    overall_decision = decide_pretrigger_candidate(evidence)
    if all(primary_gates.values()) and overall_decision != (
        "pretrigger_quiet_accumulation_direction_candidate_supported"
    ):
        raise RuntimeError("decision logic failed an all-pass primary gate")
    selective_status = (
        "supported"
        if primary_gates["6_selective_accuracy_at_least_57pct"]
        and primary_gates["8_mean_aligned_return_positive"]
        and primary_gates["9_median_aligned_return_positive"]
        else "promising"
        if primary_gates["8_mean_aligned_return_positive"]
        else "not_supported"
    )
    component_statuses = {
        "movement_gate_status": "supported",
        "episode_reconstruction_status": "supported",
        "pretrigger_history_status": (
            "supported" if full_pressure_history_supported else "insufficient_support"
        ),
        "quietness_status": "supported" if context_supported else "not_supported",
        "persistent_pressure_status": (
            persistent_pressure_status
            if full_pressure_history_supported
            else "insufficient_support"
        ),
        "absorption_response_status": absorption_status,
        "relative_resilience_status": relative_status,
        "vwap_defence_status": vwap_status,
        "composite_score_status": (
            "supported" if composite_supported else "promising" if qs_improves else "not_supported"
        )
        if full_pressure_history_supported
        else "insufficient_support",
        "selective_direction_status": (
            selective_status if full_pressure_history_supported else "insufficient_support"
        ),
        "remaining_movement_status": "not_supported" if late_direction_problem else "supported",
        "prospective_recorder_priority": (
            "not_supported"
            if not full_pressure_history_supported
            else "supported"
            if overall_decision == "pretrigger_quiet_accumulation_direction_candidate_supported"
            else "promising"
            if any(
                value in {"supported", "promising"}
                for value in (
                    persistent_pressure_status,
                    absorption_status,
                    relative_status,
                    vwap_status,
                )
            )
            else "not_supported"
        ),
    }
    return {
        **SAFETY_FLAGS,
        "overall_decision": overall_decision,
        **component_statuses,
        "all_primary_pass_gates_passed": all(primary_gates.values()),
        "primary_pass_gates": primary_gates,
        "support_gates": support,
        "component_evidence": {
            "persistent_pressure": pressure_effects,
            "absorption_response": absorption_effects,
            "compression_context": context_effects,
            "composite_score": score_effects,
            "pressure_complete_development_windows": pressure_complete_development,
            "pressure_complete_assessment_windows": pressure_complete_assessment,
            "full_hypothesis_pressure_history_supported": full_pressure_history_supported,
            "relative_resilience_q1_coefficient": relative_coefficient,
            "vwap_q1_coefficients": vwap_coefficients,
            "score_bin_diagnostics": dict(score_diagnostics),
        },
        "late_direction_problem": late_direction_problem,
        "positive_assessment_month_groups": positive_months,
        "bootstrap_80_accuracy_lower": bootstrap_accuracy_lower,
        "bootstrap_80_mean_aligned_return_lower": bootstrap_return_lower,
        "null_comparisons": dict(null_summary),
        "temporal_placebo_comparison": dict(placebo_comparison),
        "claims_boundary": {
            "institutional_accumulation_observed": False,
            "direct_order_flow_measured": False,
            "activity_field_is_confirmed_exchange_volume": False,
            "option_profitability_tested": False,
            "prospective_validation": False,
            "paper_readiness": False,
            "live_readiness": False,
            "deployable_strategy": False,
        },
    }


@dataclass
class AnalysisResults:
    assessment: pd.DataFrame
    direction_metrics: pd.DataFrame
    selective_metrics: pd.DataFrame
    baseline_metrics: pd.DataFrame
    score_bins: pd.DataFrame
    score_diagnostics: dict[str, Any]
    permutation_metrics: pd.DataFrame
    temporal_placebo_metrics: pd.DataFrame
    temporal_placebo_comparison: dict[str, Any]
    material_move_metrics: pd.DataFrame
    remaining_movement_metrics: pd.DataFrame
    late_direction_problem: bool
    monthly_metrics: pd.DataFrame
    checkpoint_metrics: pd.DataFrame
    stock_metrics: pd.DataFrame
    concentration_metrics: pd.DataFrame
    bootstrap_metrics: pd.DataFrame
    direction_null_metrics: pd.DataFrame
    null_summary: dict[str, Any]
    support_gates: dict[str, Any]
    decision: dict[str, Any]
    resampling_plan: dict[str, Any]


def run_analysis(
    preparation: FrozenPreparation,
    *,
    frozen_resampling_plan: Mapping[str, Any] | None = None,
) -> AnalysisResults:
    assessment = score_assessment(preparation)
    direction_metrics = direction_model_metric_table(assessment)
    selective_table = selective_policy_metric_table(assessment)
    scored_development = apply_quiet_score_parameters(
        preparation.development_raw_targets,
        preparation.score_parameters,
    )
    baseline_metrics = baseline_metric_table(assessment, scored_development)
    score_bins, score_diagnostics = score_bin_metric_table(assessment)
    permutation_metrics = grouped_permutation_table(
        assessment,
        preparation.full_models["Q1"],
        preparation.thresholds["Q1"],
    )
    temporal_metrics, temporal_comparison = temporal_placebo_table(
        preparation,
        assessment,
    )
    material_metrics = material_move_table(assessment)
    remaining_metrics, late_direction_problem = remaining_movement_table(assessment)
    monthly_metrics = monthly_metric_table(assessment)
    checkpoint_metrics = subgroup_metric_table(assessment)
    stock_metrics, concentration_metrics = stock_and_concentration_tables(assessment)
    frozen_draws = (
        cast(Sequence[Sequence[str]], frozen_resampling_plan["draw_sessions"])
        if frozen_resampling_plan is not None
        else None
    )
    bootstrap_metrics, bootstrap_plan = bootstrap_table_and_plan(
        assessment,
        frozen_draw_sessions=frozen_draws,
    )
    frozen_null_targets = (
        cast(
            Mapping[str, Sequence[object]],
            frozen_resampling_plan["null_targets"],
        )
        if frozen_resampling_plan is not None
        else None
    )
    null_metrics, null_summary, null_plan = direction_null_table(
        preparation,
        assessment,
        frozen_null_targets=frozen_null_targets,
    )
    support = support_gates(preparation, assessment)
    decision = build_decision(
        preparation,
        assessment,
        direction_metrics,
        selective_table,
        baseline_metrics,
        score_diagnostics,
        score_bins,
        permutation_metrics,
        temporal_comparison,
        monthly_metrics,
        concentration_metrics,
        bootstrap_metrics,
        null_summary,
        support,
        late_direction_problem,
    )
    resampling_plan = {
        "bootstrap_seed": bootstrap_plan["bootstrap_seed"],
        "bootstrap_draws": bootstrap_plan["bootstrap_draws"],
        "session_universe": bootstrap_plan["session_universe"],
        "draw_sessions": bootstrap_plan["draw_sessions"],
        "null_seeds": null_plan["null_seeds"],
        "null_targets": null_plan["null_targets"],
        "label_assignment_hashes": null_plan["label_assignment_hashes"],
        "permutation_seeds": list(PERMUTATION_SEEDS),
        "samples_reused_for_determinism": frozen_resampling_plan is not None,
    }
    return AnalysisResults(
        assessment=assessment,
        direction_metrics=direction_metrics,
        selective_metrics=selective_table,
        baseline_metrics=baseline_metrics,
        score_bins=score_bins,
        score_diagnostics=score_diagnostics,
        permutation_metrics=permutation_metrics,
        temporal_placebo_metrics=temporal_metrics,
        temporal_placebo_comparison=temporal_comparison,
        material_move_metrics=material_metrics,
        remaining_movement_metrics=remaining_metrics,
        late_direction_problem=late_direction_problem,
        monthly_metrics=monthly_metrics,
        checkpoint_metrics=checkpoint_metrics,
        stock_metrics=stock_metrics,
        concentration_metrics=concentration_metrics,
        bootstrap_metrics=bootstrap_metrics,
        direction_null_metrics=null_metrics,
        null_summary=null_summary,
        support_gates=support,
        decision=decision,
        resampling_plan=resampling_plan,
    )


def create_plots(results: AnalysisResults) -> None:
    assessment = results.assessment
    score_bins = results.score_bins

    figure, first_axis = plt.subplots(figsize=(10, 5.5))
    order = np.arange(len(score_bins))
    first_axis.bar(
        order - 0.18,
        score_bins["mean_future_signed_return_10m"].to_numpy(float) * 10_000.0,
        width=0.36,
        color="#355070",
        label="Mean signed return (bp)",
    )
    first_axis.set_ylabel("Mean future signed return (bp)")
    first_axis.set_xticks(order)
    first_axis.set_xticklabels(
        [
            "Strong\ndistribution",
            "Moderate\ndistribution",
            "Neutral",
            "Moderate\naccumulation",
            "Strong\naccumulation",
        ]
    )
    second_axis = first_axis.twinx()
    second_axis.plot(
        order,
        score_bins["up_rate"].to_numpy(float),
        marker="o",
        linewidth=2.0,
        color="#e56b6f",
        label="UP rate",
    )
    second_axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    second_axis.set_ylabel("UP rate")
    first_axis.set_title("Frozen quiet-absorption score bins")
    handles_one, labels_one = first_axis.get_legend_handles_labels()
    handles_two, labels_two = second_axis.get_legend_handles_labels()
    first_axis.legend(handles_one + handles_two, labels_one + labels_two, loc="best")
    figure.tight_layout()
    figure.savefig(PRIMARY / "quiet_absorption_score_bins.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {"Q0": "#6c757d", "QS": "#d17a22", "Q1": "#2a9d8f"}
    for model_id in MODEL_IDS:
        valid = assessment["direction_up_10m"].notna()
        target = assessment.loc[valid, "direction_up_10m"].to_numpy(int)
        probability = assessment.loc[valid, f"{model_id}_p_up"].to_numpy(float)
        false_positive, true_positive, _ = roc_curve(target, probability)
        axes[0].plot(
            false_positive,
            true_positive,
            label=model_id,
            color=colors[model_id],
        )
        bins = pd.qcut(
            pd.Series(probability),
            q=min(10, len(np.unique(probability))),
            duplicates="drop",
        )
        calibration = (
            pd.DataFrame({"probability": probability, "target": target, "bin": bins})
            .groupby("bin", observed=True)[["probability", "target"]]
            .mean()
        )
        axes[1].plot(
            calibration["probability"],
            calibration["target"],
            marker="o",
            label=model_id,
            color=colors[model_id],
        )
    axes[0].plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--")
    axes[0].set_title("Assessment ROC")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--")
    axes[1].set_title("Assessment calibration")
    axes[1].set_xlabel("Mean predicted UP")
    axes[1].set_ylabel("Observed UP rate")
    selective_rows = results.selective_metrics.loc[
        results.selective_metrics["policy_id"].eq("frozen_development_oof_35pct")
        & results.selective_metrics["horizon_minutes"].eq(10)
    ]
    axes[2].bar(
        selective_rows["model_id"],
        selective_rows["directional_accuracy"],
        color=[colors[str(value)] for value in selective_rows["model_id"]],
    )
    axes[2].axhline(0.5, color="#bbbbbb", linestyle="--")
    axes[2].set_ylim(0.35, 0.70)
    axes[2].set_title("Frozen selective accuracy")
    axes[2].set_ylabel("Ten-minute accuracy")
    figure.tight_layout()
    figure.savefig(PRIMARY / "q0_qs_q1_diagnostics.png", dpi=160)
    plt.close(figure)

    actioned = assessment.loc[assessment["Q1_action"].astype(str).ne("ABSTAIN")].copy()
    correct = action_correct_mask(actioned, "Q1_action")
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(
        actioned["pre_entry_signed_return"].abs() * 10_000.0,
        actioned["remaining_fraction_10m"],
        c=np.where(correct, "#2a9d8f", "#e76f51"),
        alpha=0.7,
        s=28,
    )
    axis.axhline(0.5, color="#555555", linestyle="--")
    axis.set_xlabel("Absolute marker-to-entry displacement (bp)")
    axis.set_ylabel("Remaining fraction at ten minutes")
    axis.set_title("Pre-entry displacement versus remaining movement")
    figure.tight_layout()
    figure.savefig(
        PRIMARY / "preentry_displacement_remaining_movement.png",
        dpi=160,
    )
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    display = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.8g}"
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [
            "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    preparation: FrozenPreparation,
    results: AnalysisResults,
) -> str:
    assessment = results.assessment
    q_models = results.direction_metrics.loc[results.direction_metrics["row_type"].eq("model")]
    q1_selective = results.selective_metrics.loc[
        results.selective_metrics["model_id"].eq("Q1")
        & results.selective_metrics["policy_id"].eq("frozen_development_oof_35pct")
    ]
    remaining_all = results.remaining_movement_metrics.loc[
        results.remaining_movement_metrics["subgroup"].eq("all_actions")
    ]
    permutation_summary = results.permutation_metrics.loc[
        results.permutation_metrics["row_type"].eq("mean_over_20")
    ]
    interval_summary = results.bootstrap_metrics.loc[
        results.bootstrap_metrics["row_type"].eq("interval")
        & results.bootstrap_metrics["interval_level"].isin([0.80, 0.90, 0.95])
    ]
    lines = [
        "# Pre-Trigger Quiet Accumulation / Distribution Direction Screen V0",
        "",
        f"**Overall decision:** `{results.decision['overall_decision']}`",
        "",
        "This is retrospective directional candidate evidence based on underlying-stock "
        "returns and bar-derived behaviour. It does not observe institutional "
        "accumulation, direct order flow, exchange-verified volume, option P&L, or "
        "prospective execution.",
        "",
        "## Chronology and frozen movement gate",
        "",
        f"- Development: {DEVELOPMENT_START} through {DEVELOPMENT_END}.",
        f"- Retrospective assessment: {ASSESSMENT_START} through {ASSESSMENT_END}.",
        "- Excluded opened holdout: 2025-09-01 through 2025-12-31.",
        "- Protected: 2026-01-01 onward; no protected outcomes were read or materialised.",
        f"- Frozen M1 threshold: `{M1_THRESHOLD:.17f}`.",
        (
            "- Frozen reconstruction: "
            f"{preparation.episode_audit['raw_above_threshold_checkpoint_rows']:,} "
            "raw above-threshold rows and "
            f"{preparation.episode_audit['fresh_episodes']:,} fresh episodes "
            f"({preparation.episode_audit['development_episodes']} development, "
            f"{preparation.episode_audit['assessment_episodes']} assessment)."
        ),
        "",
        "## Pre-trigger construction",
        "",
        "- Binding window: five completed five-minute bars (25 minutes), T-5 through T-1.",
        "- Direction marker: close of T-1.",
        "- The complete M1 trigger bar T was excluded from every direction feature.",
        "- Entry remained the first post-trigger completed bar open.",
        (
            "- Signed pressure: exact audited even-checkpoint snapshots only; "
            "intervening bars remain missing and are never interpolated, carried, "
            "or redefined."
        ),
        (
            "- Activity: repository causal activity proxy "
            "`historical_relative_activity`; it is not asserted to be exchange volume."
        ),
        (
            f"- Complete five-bar pressure windows: "
            f"{int(assessment['signed_pressure_bars_present_25'].eq(5).sum())}/"
            f"{len(assessment)} assessment episodes."
        ),
        "",
        "The composite is `quietness_25 × mean(13 clipped signed component z-scores)`, "
        "with development-only median/IQR preprocessing, development-median missing "
        "imputation, clipping to [-3,+3], and equal weights.",
        "",
        "## Assessment model metrics",
        "",
        _markdown_table(
            q_models,
            [
                "model_id",
                "episodes",
                "log_loss",
                "brier_score",
                "auc",
                "average_precision",
                "accuracy",
                "balanced_accuracy",
                "calibration_intercept",
                "calibration_slope",
            ],
        ),
        "",
        "## Frozen selective policy",
        "",
        f"Q1 confidence boundary: `{preparation.thresholds['Q1']:.12g}`.",
        "",
        _markdown_table(
            q1_selective,
            [
                "horizon_minutes",
                "actions",
                "abstentions",
                "action_coverage",
                "call_count",
                "put_count",
                "directional_accuracy",
                "balanced_accuracy",
                "mean_aligned_return",
                "median_aligned_return",
                "positive_aligned_return_rate",
            ],
        ),
        "",
        "Aligned return is the underlying-stock log return multiplied by +1 for CALL "
        "and -1 for PUT. It is not option P&L.",
        "",
        "## Baselines",
        "",
        _markdown_table(
            results.baseline_metrics,
            [
                "baseline_id",
                "directional_accuracy",
                "balanced_accuracy_direction",
                "mean_aligned_return",
                "median_aligned_return",
            ],
        ),
        "",
        "## Timing and remaining movement",
        "",
        _markdown_table(
            remaining_all,
            [
                "subgroup",
                "mean_pre_entry_signed_return",
                "median_pre_entry_signed_return",
                "mean_absolute_pre_entry_displacement",
                "mean_remaining_fraction_10m",
                "median_remaining_fraction_10m",
                "episodes_at_least_50pct_remaining_10m",
                "mean_remaining_fraction_30m",
            ],
        ),
        "",
        f"`late_direction_problem = {str(results.late_direction_problem).lower()}`.",
        "",
        "## Material-movement diagnostics",
        "",
        _markdown_table(
            results.material_move_metrics,
            [
                "subgroup",
                "episodes",
                "accuracy",
                "balanced_accuracy",
                "mean_aligned_return",
                "median_aligned_return",
                "positive_aligned_return_rate",
                "mean_remaining_fraction",
            ],
        ),
        "",
        "## Frozen score bins",
        "",
        _markdown_table(
            results.score_bins,
            [
                "score_bin",
                "episodes",
                "mean_future_signed_return_10m",
                "median_future_signed_return_10m",
                "up_rate",
                "mean_absolute_movement_10m",
                "iv_excess_rate",
                "mean_remaining_fraction_10m",
            ],
        ),
        "",
        (
            "Correct signed monotonic ordering: "
            f"`{str(results.score_diagnostics['correct_signed_monotonic_direction']).lower()}`; "
            f"slope `{results.score_diagnostics['monotonic_slope']:.8g}`."
        ),
        "",
        "## Grouped permutation attribution",
        "",
        _markdown_table(
            permutation_summary,
            [
                "group_id",
                "log_loss_deterioration",
                "brier_deterioration",
                "auc_deterioration",
                "selective_accuracy_deterioration",
                "mean_aligned_return_deterioration",
                "median_aligned_return_deterioration",
            ],
        ),
        "",
        "## Temporal placebo and nulls",
        "",
        _markdown_table(
            results.temporal_placebo_metrics,
            [
                "model",
                "log_loss",
                "brier_score",
                "auc",
                "selective_directional_accuracy",
                "selective_mean_aligned_return",
                "confidence_boundary",
            ],
        ),
        "",
        (
            "Real Q1 outperformed the temporally misaligned placebo under the frozen "
            "comparison: "
            f"`{
                str(
                    results.temporal_placebo_comparison['real_q1_outperforms_temporal_placebo']
                ).lower()
            }`."
        ),
        "",
        f"Five-null comparison: `{json.dumps(results.null_summary, sort_keys=True)}`.",
        "",
        "## Whole-session bootstrap intervals",
        "",
        _markdown_table(
            interval_summary,
            ["interval_level", "metric", "bound", "value"],
        ),
        "",
        "## Stability, concentration, and support",
        "",
        (
            f"Positive Q1 selective mean aligned return occurred in "
            f"{results.decision['positive_assessment_month_groups']} of eight month groups."
        ),
        "",
        _markdown_table(
            results.concentration_metrics.loc[
                results.concentration_metrics["metric"].str.startswith("maximum_")
            ],
            ["metric", "value"],
        ),
        "",
        f"Support gates: `{json.dumps(results.support_gates, sort_keys=True)}`.",
        "",
        "## Component statuses",
        "",
    ]
    for key in (
        "movement_gate_status",
        "episode_reconstruction_status",
        "pretrigger_history_status",
        "quietness_status",
        "persistent_pressure_status",
        "absorption_response_status",
        "relative_resilience_status",
        "vwap_defence_status",
        "composite_score_status",
        "selective_direction_status",
        "remaining_movement_status",
        "prospective_recorder_priority",
    ):
        lines.append(f"- `{key}`: `{results.decision[key]}`")
    lines.extend(
        [
            "",
            "## Primary pass gates",
            "",
        ]
    )
    for gate, passed in cast(Mapping[str, bool], results.decision["primary_pass_gates"]).items():
        lines.append(f"- `{gate}`: `{str(bool(passed)).lower()}`")
    lines.extend(
        [
            "",
            "## Claims boundary",
            "",
            "This experiment is not institutional-accumulation observation, direct "
            "order-flow research, an option profitability study, realistic bid/ask "
            "execution, prospective validation, paper readiness, live readiness, or "
            "a deployable strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def maximum_numeric_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    maximum = 0.0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if len(left_values) != len(right_values):
            return math.inf
        both_missing = ~np.isfinite(left_values) & ~np.isfinite(right_values)
        one_missing = np.isfinite(left_values) ^ np.isfinite(right_values)
        if bool(one_missing.any()):
            return math.inf
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        difference = np.zeros(len(left_values), dtype=float)
        difference[finite] = np.abs(left_values[finite] - right_values[finite])
        difference[both_missing] = 0.0
        maximum = max(maximum, float(np.max(difference)) if len(difference) else 0.0)
    return maximum


def determinism_check(
    first_preparation: FrozenPreparation,
    first_results: AnalysisResults,
) -> dict[str, Any]:
    second_preparation = prepare_frozen_experiment()
    second_results = run_analysis(
        second_preparation,
        frozen_resampling_plan=first_results.resampling_plan,
    )
    first_episodes = first_preparation.episodes.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    second_episodes = second_preparation.episodes.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    episode_mismatches = int(
        (
            first_episodes.loc[:, list(PRIMARY_IDENTITY_COLUMNS)].astype(str)
            != second_episodes.loc[:, list(PRIMARY_IDENTITY_COLUMNS)].astype(str)
        )
        .any(axis=1)
        .sum()
    )
    first_features = first_preparation.scored_features.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    second_features = second_preparation.scored_features.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    timestamp_mismatches = int(
        (
            pd.to_datetime(
                first_features["pretrigger_marker_timestamp"],
                utc=True,
                errors="raise",
            )
            != pd.to_datetime(
                second_features["pretrigger_marker_timestamp"],
                utc=True,
                errors="raise",
            )
        ).sum()
    )
    feature_columns = list(
        dict.fromkeys(
            (
                *PRIMARY_RAW_FEATURES,
                *Q0_NUMERIC_FEATURES,
                *GROUP_P,
                *GROUP_A,
                *GROUP_C,
                *(
                    column
                    for column in first_features.columns
                    if column.endswith("_15") or column.endswith("_45")
                ),
            )
        )
    )
    maximum_feature_difference = maximum_numeric_difference(
        first_features,
        second_features,
        feature_columns,
    )
    score_difference = maximum_numeric_difference(
        first_features,
        second_features,
        ["quiet_absorption_score_25"],
    )
    first_oof = first_preparation.development_oof.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    second_oof = second_preparation.development_oof.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    oof_probability_difference = maximum_numeric_difference(
        first_oof,
        second_oof,
        [f"{model_id}_p_up" for model_id in MODEL_IDS],
    )
    first_assessment = first_results.assessment.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    second_assessment = second_results.assessment.sort_values(
        list(PRIMARY_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    probability_difference = maximum_numeric_difference(
        first_assessment,
        second_assessment,
        [f"{model_id}_p_up" for model_id in MODEL_IDS],
    )
    action_mismatches = int(
        sum(
            (
                first_assessment[f"{model_id}_action"].astype(str)
                != second_assessment[f"{model_id}_action"].astype(str)
            ).sum()
            for model_id in MODEL_IDS
        )
    )
    target_columns = [f"signed_log_return_{horizon}m" for horizon in (5, 10, 15, 30)]
    target_columns.extend(
        ["pre_entry_signed_return", "remaining_fraction_10m", "remaining_fraction_30m"]
    )
    target_difference = maximum_numeric_difference(
        first_assessment,
        second_assessment,
        target_columns,
    )
    first_side = np.where(
        first_assessment["Q1_action"].astype(str).eq("CALL"),
        1.0,
        np.where(first_assessment["Q1_action"].astype(str).eq("PUT"), -1.0, 0.0),
    )
    second_side = np.where(
        second_assessment["Q1_action"].astype(str).eq("CALL"),
        1.0,
        np.where(second_assessment["Q1_action"].astype(str).eq("PUT"), -1.0, 0.0),
    )
    aligned_difference = float(
        np.max(
            np.abs(
                first_side * first_assessment["signed_log_return_10m"].to_numpy(float)
                - second_side * second_assessment["signed_log_return_10m"].to_numpy(float)
            )
        )
    )
    threshold_difference = max(
        abs(first_preparation.thresholds[model_id] - second_preparation.thresholds[model_id])
        for model_id in MODEL_IDS
    )
    first_panel = first_preparation.panel.sort_values("row_id", kind="mergesort").reset_index(
        drop=True
    )
    second_panel = second_preparation.panel.sort_values("row_id", kind="mergesort").reset_index(
        drop=True
    )
    movement_probability_difference = maximum_numeric_difference(
        first_panel,
        second_panel,
        ["M1_probability"],
    )
    metric_difference = maximum_numeric_difference(
        first_results.direction_metrics.sort_values(
            ["row_type", "model_id"], kind="mergesort"
        ).reset_index(drop=True),
        second_results.direction_metrics.sort_values(
            ["row_type", "model_id"], kind="mergesort"
        ).reset_index(drop=True),
        [
            "log_loss",
            "brier_score",
            "auc",
            "accuracy",
            "balanced_accuracy",
        ],
    )
    null_assignments_reused = (
        first_results.resampling_plan["label_assignment_hashes"]
        == second_results.resampling_plan["label_assignment_hashes"]
    )
    bootstrap_draws_reused = (
        first_results.resampling_plan["draw_sessions"]
        == second_results.resampling_plan["draw_sessions"]
    )
    decision_match = (
        first_results.decision["overall_decision"] == second_results.decision["overall_decision"]
    )
    passed = bool(
        episode_mismatches == 0
        and timestamp_mismatches == 0
        and movement_probability_difference <= 1e-12
        and maximum_feature_difference <= 1e-12
        and score_difference <= 1e-12
        and oof_probability_difference <= 1e-12
        and probability_difference <= 1e-12
        and action_mismatches == 0
        and target_difference <= 1e-12
        and aligned_difference <= 1e-12
        and threshold_difference <= 1e-12
        and metric_difference <= 1e-12
        and null_assignments_reused
        and bootstrap_draws_reused
        and decision_match
    )
    return {
        "passed": passed,
        "episode_identity_mismatches": episode_mismatches,
        "pretrigger_timestamp_mismatches": timestamp_mismatches,
        "movement_probability_max_difference": movement_probability_difference,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_score_difference": score_difference,
        "maximum_development_oof_probability_difference": (oof_probability_difference),
        "maximum_probability_difference": probability_difference,
        "action_decision_mismatches": action_mismatches,
        "maximum_target_difference": target_difference,
        "maximum_aligned_return_difference": aligned_difference,
        "maximum_threshold_difference": threshold_difference,
        "maximum_metric_difference": metric_difference,
        "bootstrap_draws_reused_without_redraw": bootstrap_draws_reused,
        "null_assignments_reused_without_redraw": null_assignments_reused,
        "final_decision_match": decision_match,
    }


def write_analysis_artifacts(
    preparation: FrozenPreparation,
    results: AnalysisResults,
    determinism: Mapping[str, Any],
) -> None:
    write_parquet(
        PRIMARY / "assessment_predictions.parquet",
        results.assessment,
    )
    write_csv(PRIMARY / "direction_model_metrics.csv", results.direction_metrics)
    write_csv(PRIMARY / "selective_policy_metrics.csv", results.selective_metrics)
    write_csv(PRIMARY / "baseline_metrics.csv", results.baseline_metrics)
    write_csv(PRIMARY / "score_bin_metrics.csv", results.score_bins)
    write_csv(
        PRIMARY / "grouped_permutation_metrics.csv",
        results.permutation_metrics,
    )
    write_csv(
        PRIMARY / "temporal_placebo_metrics.csv",
        results.temporal_placebo_metrics,
    )
    write_csv(
        PRIMARY / "material_move_metrics.csv",
        results.material_move_metrics,
    )
    write_csv(
        PRIMARY / "remaining_movement_metrics.csv",
        results.remaining_movement_metrics,
    )
    write_csv(PRIMARY / "monthly_metrics.csv", results.monthly_metrics)
    write_csv(PRIMARY / "checkpoint_metrics.csv", results.checkpoint_metrics)
    write_csv(PRIMARY / "stock_metrics.csv", results.stock_metrics)
    write_csv(
        PRIMARY / "concentration_metrics.csv",
        results.concentration_metrics,
    )
    write_csv(PRIMARY / "bootstrap_metrics.csv", results.bootstrap_metrics)
    write_csv(
        PRIMARY / "direction_null_metrics.csv",
        results.direction_null_metrics,
    )
    write_json(
        PRIMARY / "frozen_resampling_plan.json",
        results.resampling_plan,
    )
    decision = {
        **results.decision,
        "independent_audit_result": "pending",
        "determinism_result": "passed" if bool(determinism["passed"]) else "failed",
    }
    write_json(PRIMARY / "decision.json", decision)
    write_json(PRIMARY / "determinism_check.json", cast(dict[str, Any], determinism))
    create_plots(results)
    report = build_report(preparation, results)
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")


def execute(*, run_determinism: bool) -> dict[str, Any]:
    preparation = prepare_frozen_experiment()
    write_freeze_artifacts(preparation)
    results = run_analysis(preparation)
    if run_determinism:
        determinism = determinism_check(preparation, results)
    else:
        determinism = {
            "passed": False,
            "skipped": True,
            "reason": "explicit --skip-determinism",
        }
    if not bool(determinism["passed"]) and not bool(determinism.get("skipped", False)):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "determinism rebuild exceeded a frozen tolerance",
        )
    write_analysis_artifacts(preparation, results, determinism)
    q1 = results.direction_metrics.loc[
        results.direction_metrics["row_type"].eq("model")
        & results.direction_metrics["model_id"].eq("Q1")
    ].iloc[0]
    q1_selective = results.selective_metrics.loc[
        results.selective_metrics["model_id"].eq("Q1")
        & results.selective_metrics["policy_id"].eq("frozen_development_oof_35pct")
        & results.selective_metrics["horizon_minutes"].eq(10)
    ].iloc[0]
    return {
        "overall_decision": results.decision["overall_decision"],
        "raw_above_threshold_rows": preparation.episode_audit[
            "raw_above_threshold_checkpoint_rows"
        ],
        "fresh_episodes": preparation.episode_audit["fresh_episodes"],
        "development_episodes": preparation.episode_audit["development_episodes"],
        "assessment_episodes": preparation.episode_audit["assessment_episodes"],
        "q1_auc": float(q1["auc"]),
        "q1_balanced_accuracy": float(q1["balanced_accuracy"]),
        "q1_actions": int(q1_selective["actions"]),
        "q1_action_coverage": float(q1_selective["action_coverage"]),
        "q1_selective_accuracy": float(q1_selective["directional_accuracy"]),
        "q1_mean_aligned_return": float(q1_selective["mean_aligned_return"]),
        "late_direction_problem": results.late_direction_problem,
        "determinism_passed": bool(determinism["passed"]),
        "report": str(REPORTS / "report.md"),
    }


def write_blocked_decision(error: ScreenBlocked) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    blocked_statuses = {
        "movement_gate_status": "blocked",
        "episode_reconstruction_status": "blocked",
        "pretrigger_history_status": "blocked",
        "quietness_status": "blocked",
        "persistent_pressure_status": "blocked",
        "absorption_response_status": "blocked",
        "relative_resilience_status": "blocked",
        "vwap_defence_status": "blocked",
        "composite_score_status": "blocked",
        "selective_direction_status": "blocked",
        "remaining_movement_status": "blocked",
        "prospective_recorder_priority": "blocked",
    }
    write_json(
        PRIMARY / "decision.json",
        {
            **SAFETY_FLAGS,
            "overall_decision": error.decision,
            **blocked_statuses,
            "blocker": error.detail,
            "independent_audit_result": "not_run",
            "determinism_result": "not_run",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Run one build only; intended for local debugging, not final research.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        summary = execute(run_determinism=not arguments.skip_determinism)
    except ScreenBlocked as error:
        write_blocked_decision(error)
        print(
            json.dumps(
                {
                    "overall_decision": error.decision,
                    "blocker": error.detail,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
