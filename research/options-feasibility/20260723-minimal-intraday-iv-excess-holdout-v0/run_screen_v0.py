#!/usr/bin/env python3
"""Run Minimal Intraday Stock to IV-Excess Holdout Validation V0."""

from __future__ import annotations

# ruff: noqa: E402 -- one-process numerical limits must precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-minimal-intraday-iv-excess-mpl")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
from matplotlib import pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
DEFAULT_PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
DEFAULT_OPTIONS_CACHE = (
    REPO_ROOT
    / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v0"
    / "canonical/exact_holdout_options.parquet"
)
DEFAULT_STOCK_CACHE = (
    REPO_ROOT / "data/cache/minimal-intraday-iv-excess-holdout-v0/frozen_h0_stock_surface.parquet"
)
DEFAULT_STATE_CACHE = DEFAULT_STOCK_CACHE.with_name("frozen_state_surface.parquet")
ATTRIBUTION_PRIMARY = (
    REPO_ROOT
    / "research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0"
    / "artifacts/primary"
)
FRONT_PRIMARY = (
    REPO_ROOT
    / "research/cross-market-context/20260723-daily-stock-front-options-context-v01"
    / "artifacts/primary"
)
DENSE_PRIMARY = (
    REPO_ROOT
    / "research/route-competition/20260722-broad-conflict-advance-hazard-v02"
    / "artifacts/primary"
)
V2_RUNNER = (
    REPO_ROOT
    / "research/loop-funnel/20260721-emotion-regime-coarse-loop-family-v0/run_screen_v0.py"
)
HAZARD_RUNNER = (
    REPO_ROOT
    / "research/route-competition/20260722-route-competition-hazard-quick-v0/run_screen_v0.py"
)
ADVANCE_RUNNER = (
    REPO_ROOT
    / "research/route-competition/20260722-broad-conflict-advance-hazard-v02/run_screen_v02.py"
)
DAILY_RUNNER = (
    REPO_ROOT
    / "research/cross-market-context/20260723-daily-stock-options-regime-context-v0"
    / "run_screen_v0.py"
)
DICTIONARY_PATH = (
    REPO_ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts"
    / "20260718-loop-event-semantics-v2/primary/semantic_loop_dictionary_v2.csv"
)

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.broad_conflict_advance_hazard_v02 import (
    DENSE_H0_FEATURES,
    candidate_normalized_weights,
)
from stocker_research.daily_soft_regimes_v0 import (
    FrozenDimensionParameters,
    RobustValueScale,
)
from stocker_research.daily_stock_options_context_v0 import (
    calculate_daily_stock_raw_features,
    previous_us_trading_session,
)
from stocker_research.front_options_soft_regimes_v01 import (
    FRONT_OPTIONS_DIMENSIONS,
    FRONT_OPTIONS_MISSING_INDICATORS,
    apply_front_options_dimensions,
    apply_serialized_diag_regime,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    BOOTSTRAP_SEED,
    EXCLUDED_FEATURES,
    GROUP_I,
    GROUP_O,
    HOLDOUT_END,
    HOLDOUT_START,
    HORIZONS,
    NULL_SEEDS,
    PROTECTED_START,
    SAFETY_FLAGS,
    TARGET_COLUMN,
    ModelGateInputs,
    TailGateInputs,
    add_movement_outcomes,
    assert_safety_flags,
    build_group_i,
    build_group_o,
    decide_experiment,
    fit_minimal_models,
    freeze_tail_thresholds,
    frozen_tail_membership,
    frozen_tail_support,
    intraday_h0_bundle_null,
    joined_support,
    minimal_feature_manifest,
    model_increment,
    model_metric_row,
    model_specification,
    movement_timing_metrics,
    select_minimal_front_options_surface,
    tail_comparison_metrics,
    tail_metrics,
    tail_overlap_metrics,
    validate_exact_previous_session_options,
    validate_holdout_dates,
    weighted_quantile,
)
from stocker_research.stock_layer_iv_excess_attribution_v0 import (
    CHECKPOINTS,
    FROZEN_COHORT,
)
from stocker_research.stock_options_cross_market_quick_v0 import fit_cross_market_model

EXPECTED_HISTORICAL_PANEL_SHA256 = (
    "f62ef0144c12c813cbc665ba6d5ba1a235a6f77101a04b9f491c77b24c295529"
)
EXPECTED_DENSE_PANEL_SHA256 = "a916b792e15e8630dadc09bed64d71be5533ce9f3b2bd93af06605d0faaa0cc3"


class ScreenBlocked(RuntimeError):
    """One authorized fail-closed experiment blocker."""

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
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path, date)):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"cannot load frozen runner: {path}",
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def maximum_numeric_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    for column in columns:
        first = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        second = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        both_nan = np.isnan(first) & np.isnan(second)
        finite = np.isfinite(first) & np.isfinite(second)
        if bool((~both_nan & ~finite).any()):
            return math.inf
        if bool(finite.any()):
            maximum = max(maximum, float(np.max(np.abs(first[finite] - second[finite]))))
    return maximum


def row_identity_mismatches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str = "row_id",
) -> int:
    first = left[key].astype(str).sort_values(kind="mergesort").reset_index(drop=True)
    second = right[key].astype(str).sort_values(kind="mergesort").reset_index(drop=True)
    return abs(len(first) - len(second)) + sum(a != b for a, b in zip(first, second, strict=False))


def load_contract() -> dict[str, Any]:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    assert_safety_flags(contract)
    limits = cast(Mapping[str, Any], contract["hard_limits"])
    expected = {
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "primary_logistic_models": 2,
        "binding_tails": 1,
        "session_bootstrap_draws": 10,
        "intraday_h0_bundle_null_refits": 3,
        "maximum_new_eodhd_records": 350_000,
        "maximum_new_raw_bytes": 1_000_000_000,
        "maximum_plots": 2,
        "full_repository_test_suite": False,
    }
    if any(limits.get(key) != value for key, value in expected.items()):
        raise ScreenBlocked("blocked_quick_resource_limit", "contract hard limits differ")
    if tuple(int(value) for value in contract["checkpoints"]) != CHECKPOINTS:
        raise ScreenBlocked("blocked_reproducibility_or_audit_failure", "checkpoints differ")
    return contract


def historical_panel_path() -> Path:
    source = read_json(ATTRIBUTION_PRIMARY / "source_manifest.json")
    path = Path(str(cast(Mapping[str, Any], source["sources"])["frozen_branch_c_panel"]))
    if not path.is_file() or sha256_file(path) != EXPECTED_HISTORICAL_PANEL_SHA256:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            f"frozen Branch C panel is missing or drifted: {path}",
        )
    return path


def canonical_route_metadata() -> pd.DataFrame:
    table = pd.read_csv(DICTIONARY_PATH)
    rows: list[dict[str, object]] = []
    for row in table.itertuples(index=False):
        semantic_loop_id = str(row.semantic_loop_id)
        motif_type = str(row.motif_type)
        paths = cast(list[list[int]], json.loads(str(row.all_valid_oriented_paths)))
        for path in paths:
            rows.append(
                {
                    "semantic_loop_id": semantic_loop_id,
                    "orientation_id": (
                        f"{semantic_loop_id}__o_{'-'.join(str(value) for value in path)}"
                    ),
                    "dictionary_motif_type": motif_type,
                    "canonical_total_transitions": len(path) - 1,
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["semantic_loop_id", "orientation_id"]).any():
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "canonical route identities are not unique",
        )
    return result


def build_frozen_states(provider_root: Path) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    v2 = load_module(V2_RUNNER, "minimal_holdout_v2_runner")
    v2.PROTECTED_START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
    v2.READ_END_INCLUSIVE = pd.Timestamp("2025-12-31T23:59:59.999999Z")
    v2.MAX_TARGET_BAR_ORDINAL = 39
    preprocessing, parameters = v2.load_frozen_model()
    dictionary, dictionary_manifest = v2.load_loop_dictionary()
    built_states, built_source = v2.build_v2_state_panel(
        provider_root.expanduser().resolve(), preprocessing, parameters
    )
    states = cast(pd.DataFrame, built_states)
    states = states.loc[states["symbol"].isin(FROZEN_COHORT)].copy()
    states["posterior_entropy"] = states["posterior_entropy_reproduced"].astype(float)
    probabilities = states.loc[:, [f"state_p_{state}" for state in range(8)]].to_numpy(float)
    ordered = np.sort(probabilities, axis=1)
    states["top_state_probability"] = ordered[:, -1]
    states["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    states["historical_relative_activity"] = states["volume"] / states[
        "historical_volume_baseline_at_bar"
    ].replace(0.0, np.nan)
    source: dict[str, Any] = {
        **cast(dict[str, Any], built_source),
        "state_cache_reused": False,
        "provider_root": str(provider_root.expanduser().resolve()),
    }
    timestamps = pd.to_datetime(states["bar_start_timestamp"], errors="raise", utc=True)
    if bool(timestamps.ge(pd.Timestamp("2026-01-01T00:00:00Z")).any()):
        raise ScreenBlocked(
            "blocked_protected_boundary_failure",
            "protected state row was materialised",
        )
    states = states.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    manifest = {
        **source,
        "loop_dictionary": dictionary_manifest,
        "maximum_target_bar_ordinal": 39,
        "protected_rows_materialised": 0,
    }
    return states, dictionary, manifest


def attach_movement_prices(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Attach only prices at the frozen post-threshold outcome stage."""

    output = panel.copy()
    state_index = states.set_index(["symbol", "session", "bar_ordinal"])
    for column, offset, price_column in (
        ("entry_price", 0, "open"),
        ("close_5m", 0, "close"),
        ("close_10m", 1, "close"),
        ("close_15m", 2, "close"),
        ("close_30m", 5, "close"),
    ):
        keys = pd.MultiIndex.from_arrays(
            [
                output["symbol"].astype(str),
                output["session"].astype(str),
                output["checkpoint"].astype(int) + offset,
            ],
            names=["symbol", "session", "bar_ordinal"],
        )
        output[column] = state_index[price_column].reindex(keys).to_numpy(float)
    required = ["entry_price", *(f"close_{horizon}m" for horizon in HORIZONS)]
    finite = np.isfinite(output[required].to_numpy(float)).all(axis=1)
    positive = output[required].gt(0.0).all(axis=1).to_numpy()
    return output.loc[finite & positive].reset_index(drop=True)


def reconstruct_frozen_h0_surface(
    provider_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebuild the exact development/reference/holdout H0 candidate population."""

    states, dictionary, state_manifest = build_frozen_states(provider_root)
    hazard = load_module(HAZARD_RUNNER, "minimal_holdout_hazard_runner")
    hazard.CHECKPOINTS = CHECKPOINTS
    hazard.BASELINE_FEATURES = DENSE_H0_FEATURES
    ledger = hazard.build_structural_ledger(states, dictionary)
    raw, exclusions, possible_rows = hazard.build_raw_decision_panel(states, ledger)
    panel, component_scaling, local_scaling = hazard.add_development_frozen_baseline_features(raw)
    advance = load_module(ADVANCE_RUNNER, "minimal_holdout_advance_runner")
    panel = advance.reconstruct_fixed_lead_labels(
        panel,
        ledger,
        canonical_route_metadata(),
        checkpoints=CHECKPOINTS,
    )
    sessions = pd.to_datetime(panel["session"], errors="raise")
    panel["period"] = np.select(
        [
            sessions.dt.year.eq(2024),
            sessions.between("2025-01-01", "2025-08-22"),
            sessions.between(HOLDOUT_START, HOLDOUT_END),
        ],
        ["development", "prior_reference", "holdout"],
        default="causal_history_only",
    )
    if bool(sessions.ge(PROTECTED_START).any()):
        raise ScreenBlocked(
            "blocked_protected_boundary_failure",
            "protected session entered H0 surface",
        )
    panel = panel.loc[~panel["period"].eq("causal_history_only")].copy()
    weighted = candidate_normalized_weights(panel)
    eligible = weighted.loc[weighted["advance_eligible"].astype(int).eq(1)].copy()
    build_group_i(eligible)
    eligible = eligible.sort_values(
        ["period", "session", "checkpoint", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    dense_path = DENSE_PRIMARY / "dense_advance_panel.parquet"
    if not dense_path.is_file() or sha256_file(dense_path) != EXPECTED_DENSE_PANEL_SHA256:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "frozen dense H0 predecessor panel is missing or drifted",
        )
    predecessor = pd.read_parquet(dense_path)
    predecessor_dates = pd.to_datetime(predecessor["session"], errors="raise")
    predecessor = predecessor.loc[
        predecessor_dates.le("2025-08-22") & predecessor["advance_eligible"].astype(int).eq(1)
    ].copy()
    reconstructed = eligible.loc[
        pd.to_datetime(eligible["session"], errors="raise").le("2025-08-22")
    ].copy()
    left = predecessor.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    right = reconstructed.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    identity_mismatches = row_identity_mismatches(left, right)
    maximum_difference = maximum_numeric_difference(left, right, (*GROUP_I, "row_weight"))
    if identity_mismatches or maximum_difference > 1e-12:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "frozen H0 surface failed predecessor reconstruction",
        )
    cache_manifest = {
        **SAFETY_FLAGS,
        "possible_rows": int(possible_rows),
        "raw_rows": int(len(raw)),
        "eligible_rows": int(len(eligible)),
        "development_rows": int(eligible["period"].eq("development").sum()),
        "prior_reference_rows": int(eligible["period"].eq("prior_reference").sum()),
        "holdout_rows_before_options": int(eligible["period"].eq("holdout").sum()),
        "source_exclusions": int(len(exclusions)),
        "historical_row_identity_mismatches": identity_mismatches,
        "maximum_historical_h0_feature_difference": maximum_difference,
        "state_source": state_manifest,
        "component_scaling_fitted_on_2024_only": True,
        "component_scaling": component_scaling,
        "local_scaling_fitted_on_2024_only": True,
        "local_scaling_keys": len(local_scaling),
    }
    return eligible, states, cache_manifest


def load_or_build_h0_surface(
    provider_root: Path,
    cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct H0 once and keep only ignored deterministic caches."""

    panel, states, manifest = reconstruct_frozen_h0_surface(provider_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(cache_path, panel)
    state_path = cache_path.with_name("frozen_state_surface.parquet")
    write_parquet(state_path, states)
    manifest.update(
        {
            "stock_cache_path": str(cache_path),
            "stock_cache_sha256": sha256_file(cache_path),
            "state_cache_path": str(state_path),
            "state_cache_sha256": sha256_file(state_path),
        }
    )
    return panel, states, manifest


def frozen_dimension_parameters() -> FrozenDimensionParameters:
    manifest = read_json(FRONT_PRIMARY / "front_options_feature_manifest.json")
    scales = {
        name: RobustValueScale(
            center=float(cast(Mapping[str, Any], value)["center"]),
            scale=float(cast(Mapping[str, Any], value)["scale"]),
        )
        for name, value in cast(Mapping[str, Any], manifest["scales"]).items()
    }
    return FrozenDimensionParameters(
        kind="front_options",
        scales=scales,
        imputation_medians={
            str(name): float(value)
            for name, value in cast(Mapping[str, Any], manifest["imputation_medians"]).items()
        },
    )


def build_daily_stock_raw(provider_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = load_module(DAILY_RUNNER, "minimal_holdout_daily_runner")
    daily.ASSESSMENT_END = date(2025, 12, 31)
    daily.PROTECTED_START = date(2026, 1, 1)
    bars, manifest = daily.load_full_regular_session_bars(provider_root)
    aggregated = daily.aggregate_daily_bars(bars)
    raw = calculate_daily_stock_raw_features(aggregated)
    sessions = pd.to_datetime(raw["session"], errors="raise")
    if bool(sessions.ge(PROTECTED_START).any()):
        raise ScreenBlocked(
            "blocked_protected_boundary_failure",
            "protected daily underlying row was materialised",
        )
    return raw, cast(dict[str, Any], manifest)


def build_holdout_option_context(
    stock_surface: pd.DataFrame,
    *,
    provider_root: Path,
    options_cache: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select exact-D-1 pairs and apply only frozen 2024 front transforms."""

    if not options_cache.is_file():
        raise ScreenBlocked(
            "blocked_holdout_options_download_failure",
            f"holdout exact-date options cache is missing: {options_cache}",
        )
    chain = pd.read_parquet(options_cache)
    chain["trade_date"] = pd.to_datetime(chain["trade_date"], errors="raise").dt.date
    if any(value >= PROTECTED_START.date() for value in chain["trade_date"]):
        raise ScreenBlocked(
            "blocked_protected_boundary_failure",
            "protected option row was materialised",
        )
    chain_groups = {
        (str(symbol), cast(date, trade_date)): group.copy()
        for (symbol, trade_date), group in chain.groupby(
            ["underlying_symbol", "trade_date"], sort=False, observed=True
        )
    }
    daily_raw, daily_manifest = build_daily_stock_raw(provider_root)
    daily_raw["session"] = daily_raw["session"].astype(str)
    daily_index = daily_raw.set_index(["symbol", "session"])
    candidates = (
        stock_surface.loc[stock_surface["period"].eq("holdout"), ["symbol", "session"]]
        .drop_duplicates()
        .sort_values(["symbol", "session"], kind="mergesort")
    )
    rows: list[dict[str, object]] = []
    join_audit: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        symbol = str(candidate.symbol)
        signal = date.fromisoformat(str(candidate.session))
        required = previous_us_trading_session(signal)
        base: dict[str, object] = {
            "symbol": symbol,
            "session": signal.isoformat(),
            "required_options_date": required.isoformat(),
        }
        daily_key = (symbol, required.isoformat())
        if daily_key not in daily_index.index:
            result: dict[str, object] = {
                "pair_available": False,
                "pair_reason": "missing_clean_previous_session_underlying",
            }
        else:
            context = cast(pd.Series, daily_index.loc[daily_key])
            previous_close = float(context["unadjusted_close"])
            realised = float(context["realised_volatility_20d"])
            exact_chain = chain_groups.get((symbol, required))
            if exact_chain is None:
                result = {
                    "pair_available": False,
                    "pair_reason": "missing_exact_chain",
                }
            elif not math.isfinite(previous_close) or not math.isfinite(realised):
                result = {
                    "pair_available": False,
                    "pair_reason": "missing_daily_stock_volatility_or_close",
                }
            else:
                result = select_minimal_front_options_surface(
                    exact_chain,
                    previous_close=previous_close,
                    realised_volatility_20d=realised,
                )
        rows.append({**base, **result})
        join_audit.append(
            {
                **base,
                "pair_available": bool(result.get("pair_available", False)),
                "pair_reason": str(result.get("pair_reason", "unknown")),
                "chain_records": (
                    0
                    if chain_groups.get((symbol, required)) is None
                    else len(cast(pd.DataFrame, chain_groups[(symbol, required)]))
                ),
            }
        )
    raw_context = pd.DataFrame(rows)
    available = raw_context.loc[raw_context["pair_available"].astype(bool)].copy()
    if available.empty:
        raise ScreenBlocked(
            "blocked_insufficient_holdout_options_coverage",
            "no valid holdout front-option pairs were selected",
        )
    for row in available.itertuples(index=False):
        validate_exact_previous_session_options(
            signal_date=date.fromisoformat(str(row.session)),
            required_options_date=date.fromisoformat(str(row.required_options_date)),
            actual_options_date=cast(date, row.options_observation_date),
        )
    dimensioned = apply_front_options_dimensions(available, frozen_dimension_parameters())
    regime_mapping = read_json(FRONT_PRIMARY / "front_options_regime_mapping.json")
    context = apply_serialized_diag_regime(
        dimensioned,
        regime_mapping,
        prefix="front_options_regime",
    )
    missing = sorted(
        set((*FRONT_OPTIONS_DIMENSIONS, *FRONT_OPTIONS_MISSING_INDICATORS)).difference(
            context.columns
        )
    )
    if missing:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            f"frozen front-options transform missing: {missing}",
        )
    audit_frame = pd.DataFrame(join_audit)
    audit_frame["month"] = audit_frame["session"].str[:7]
    coverage = (
        audit_frame.groupby(["symbol", "month"], sort=True, observed=True)
        .agg(
            stock_sessions=("session", "size"),
            exact_chain_sessions=("chain_records", lambda values: int((values > 0).sum())),
            selected_pair_sessions=("pair_available", "sum"),
            chain_records=("chain_records", "sum"),
        )
        .reset_index()
    )
    manifest = {
        **SAFETY_FLAGS,
        "stock_sessions_requested": int(len(candidates)),
        "exact_chain_sessions": int(audit_frame["chain_records"].gt(0).sum()),
        "selected_pair_sessions": int(len(context)),
        "selected_pair_stocks": int(context["symbol"].nunique()),
        "selected_pair_months": int(pd.to_datetime(context["session"]).dt.to_period("M").nunique()),
        "same_day_or_future_observations": int(
            (
                pd.to_datetime(context["options_observation_date"])
                >= pd.to_datetime(context["session"])
            ).sum()
        ),
        "non_exact_previous_session_observations": int(
            (
                context["required_options_date"].astype(str)
                != context["options_observation_date"].astype(str)
            ).sum()
        ),
        "back_expiry_records_requested": 0,
        "daily_underlying_source": daily_manifest,
    }
    return context, coverage, audit_frame, manifest


def reconstruct_historical_models(
    historical: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit M0/M1 on 2024 and require exact M0/G0 predecessor compatibility."""

    historical = historical.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    development = historical.loc[historical["period"].astype(str).eq("development")].copy()
    reference = historical.loc[historical["period"].astype(str).eq("assessment")].copy()
    models = fit_minimal_models(development)
    development["M0_probability"] = models.m0.predict(development)
    development["M1_probability"] = models.m1.predict(development)
    reference["M0_probability"] = models.m0.predict(reference)
    reference["M1_probability"] = models.m1.predict(reference)
    predecessor = pd.read_parquet(
        ATTRIBUTION_PRIMARY / "assessment_predictions.parquet",
        columns=["row_id", "G0_probability"],
    )
    joined = reference[["row_id", "M0_probability"]].merge(
        predecessor,
        on="row_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    mismatches = int(joined["_merge"].ne("both").sum())
    maximum_probability_difference = (
        math.inf
        if mismatches
        else float(
            np.max(
                np.abs(
                    joined["M0_probability"].to_numpy(float)
                    - joined["G0_probability"].to_numpy(float)
                )
            )
        )
    )
    thresholds = freeze_tail_thresholds(
        m0_probabilities=development["M0_probability"].to_numpy(float),
        m1_probabilities=development["M1_probability"].to_numpy(float),
        weights=development["row_weight"].to_numpy(float),
    )
    thresholds["atm_iv_2024_median"] = weighted_quantile(
        development["atm_iv"].to_numpy(float),
        development["row_weight"].to_numpy(float),
        0.50,
    )
    thresholds["transition_probability_2024_median"] = weighted_quantile(
        development["transition_probability"].to_numpy(float),
        development["row_weight"].to_numpy(float),
        0.50,
    )
    thresholds["M1_probability_2024_median"] = weighted_quantile(
        development["M1_probability"].to_numpy(float),
        development["row_weight"].to_numpy(float),
        0.50,
    )
    reference_metrics = pd.DataFrame(
        [
            model_metric_row(reference, model="M0", probability_column="M0_probability"),
            model_metric_row(reference, model="M1", probability_column="M1_probability"),
        ]
    )
    reconstruction = {
        **SAFETY_FLAGS,
        "frozen_branch_c_panel_sha256": EXPECTED_HISTORICAL_PANEL_SHA256,
        "historical_rows": int(len(historical)),
        "development_rows": int(len(development)),
        "prior_reference_rows": int(len(reference)),
        "M0_predecessor_model": "G0",
        "M0_row_identity_mismatches": mismatches,
        "M0_maximum_probability_difference": maximum_probability_difference,
        "M0_exactly_reproduced": bool(mismatches == 0 and maximum_probability_difference <= 1e-12),
        "M1_feature_manifest_exact_group_O_plus_group_I": True,
        "M1_changed_after_reference_inspection": False,
        "prior_reference_binding": False,
        "prior_reference_used_for_tuning": False,
        "prior_reference_metrics": reference_metrics.to_dict(orient="records"),
    }
    reconstruction["passed"] = bool(reconstruction["M0_exactly_reproduced"])
    if not reconstruction["passed"]:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            f"M0 failed predecessor G0 reconstruction: {reconstruction}",
        )
    return models, development, reference, {**reconstruction, "thresholds": thresholds}


def join_holdout_panel(
    h0_surface: pd.DataFrame,
    option_context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = h0_surface.loc[h0_surface["period"].astype(str).eq("holdout")].copy()
    context_columns = [
        "symbol",
        "session",
        "required_options_date",
        "options_observation_date",
        "front_expiration_date",
        "front_strike",
        "front_call_contract_id",
        "front_put_contract_id",
        "skew_put_contract_id",
        "skew_call_contract_id",
        "atm_iv",
        *FRONT_OPTIONS_DIMENSIONS,
        *(f"front_options_regime_p_{index}" for index in range(4)),
        "front_options_regime_entropy",
        "front_options_regime_margin",
        *FRONT_OPTIONS_MISSING_INDICATORS,
    ]
    joined = holdout.merge(
        option_context.loc[:, context_columns],
        on=["symbol", "session"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    audit = (
        joined.assign(month=joined["session"].astype(str).str[:7])
        .groupby(["symbol", "month", "_merge"], sort=True, observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    joined = joined.loc[joined["_merge"].eq("both")].drop(columns="_merge")
    group_o = build_group_o(joined)
    joined.loc[:, list(GROUP_O)] = group_o
    build_group_i(joined)
    if set(EXCLUDED_FEATURES).intersection([*GROUP_O, *GROUP_I]):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "excluded feature entered minimal model surface",
        )
    for row in joined[["session", "required_options_date", "options_observation_date"]].itertuples(
        index=False
    ):
        validate_exact_previous_session_options(
            signal_date=date.fromisoformat(str(row.session)),
            required_options_date=date.fromisoformat(str(row.required_options_date)),
            actual_options_date=cast(date, row.options_observation_date),
        )
    return joined.reset_index(drop=True), audit


def prediction_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    m0 = model_metric_row(frame, model="M0", probability_column="M0_probability")
    m1 = model_metric_row(frame, model="M1", probability_column="M1_probability")
    increment = model_increment(m0, m1)
    rows = pd.DataFrame([m0, m1, increment])
    return rows, {key: float(value) for key, value in increment.items() if key != "comparison"}


def subgroup_metric_records(
    frame: pd.DataFrame,
    *,
    scope: str,
    labels: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in sorted(labels.astype(str).unique()):
        subset = frame.loc[labels.astype(str).eq(group)].copy()
        if subset.empty:
            continue
        m0 = model_metric_row(subset, model="M0", probability_column="M0_probability")
        m1 = model_metric_row(subset, model="M1", probability_column="M1_probability")
        increment = model_increment(m0, m1)
        for metrics in (m0, m1):
            rows.append({"scope": scope, "group": group, "record_type": "model", **metrics})
        rows.append(
            {
                "scope": scope,
                "group": group,
                "record_type": "increment",
                **increment,
                "rows": len(subset),
                "sessions": subset["session"].nunique(),
                "stocks": subset["symbol"].nunique(),
            }
        )
        m1_tail = subset.loc[subset["M1_top_5pct"].astype(bool)]
        m0_tail = subset.loc[subset["M0_top_5pct"].astype(bool)]
        if not m1_tail.empty:
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "record_type": "tail",
                    **tail_metrics(m1_tail, model="M1"),
                }
            )
        if not m0_tail.empty:
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "record_type": "tail",
                    **tail_metrics(m0_tail, model="M0"),
                }
            )
    return pd.DataFrame(rows)


def stability_tables(
    frame: pd.DataFrame,
    thresholds: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = pd.to_datetime(frame["session"], errors="raise")
    monthly = subgroup_metric_records(
        frame,
        scope="month",
        labels=sessions.dt.to_period("M").astype(str),
    )
    checkpoint_labels = pd.Series(
        np.select(
            [
                frame["checkpoint"].astype(int).between(6, 14),
                frame["checkpoint"].astype(int).between(16, 24),
                frame["checkpoint"].astype(int).between(26, 34),
            ],
            ["early_6_14", "middle_16_24", "late_26_34"],
            default="invalid",
        ),
        index=frame.index,
        dtype="string",
    )
    if checkpoint_labels.eq("invalid").any():
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "holdout contains an invalid checkpoint",
        )
    tables = [
        subgroup_metric_records(
            frame,
            scope="checkpoint_group",
            labels=checkpoint_labels,
        )
    ]
    for scope, column, threshold_name in (
        ("prior_close_atm_iv", "atm_iv", "atm_iv_2024_median"),
        (
            "intraday_transition_probability",
            "transition_probability",
            "transition_probability_2024_median",
        ),
        (
            "intraday_h0_model_probability",
            "M1_probability",
            "M1_probability_2024_median",
        ),
    ):
        threshold = float(cast(Any, thresholds[threshold_name]))
        labels = pd.Series(
            np.where(
                pd.to_numeric(frame[column], errors="raise").le(threshold),
                "low",
                "high",
            ),
            index=frame.index,
            dtype="string",
        )
        tables.append(subgroup_metric_records(frame, scope=scope, labels=labels))
    return monthly, pd.concat(tables, ignore_index=True)


def bootstrap_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Run exactly ten fixed-prediction whole-session bootstrap draws."""

    from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
        fixed_session_bootstrap_multiplicities,
    )

    multiplicities = fixed_session_bootstrap_multiplicities(
        frame["session"],
        draws=10,
        seed=BOOTSTRAP_SEED,
    )
    session_labels = frame["session"].astype(str).to_numpy()
    unique_sessions = tuple(sorted(set(session_labels)))
    rows: list[dict[str, object]] = []
    statistic_names = (
        "M1_minus_M0_log_loss_improvement",
        "M1_minus_M0_brier_improvement",
        "M1_minus_M0_auc_improvement",
        "M1_minus_M0_average_precision_improvement",
        "M1_top_5pct_mean_iv_residual",
        "M1_top_5pct_median_iv_residual",
        "M1_top_5pct_exceed_iv_rate",
        "M1_minus_M0_top_5pct_mean_iv_residual",
        "M1_minus_M0_top_5pct_median_iv_residual",
        "M1_minus_M0_top_5pct_exceed_iv_rate",
    )
    for draw, multiplicity in enumerate(multiplicities):
        selected = multiplicity > 0
        sample = frame.loc[selected].copy()
        sample["row_weight"] = sample["row_weight"].to_numpy(float) * multiplicity[selected].astype(
            float
        )
        m0 = model_metric_row(sample, model="M0", probability_column="M0_probability")
        m1 = model_metric_row(sample, model="M1", probability_column="M1_probability")
        increment = model_increment(m0, m1)
        m1_tail = tail_metrics(
            sample.loc[sample["M1_top_5pct"].astype(bool)],
            model="M1",
        )
        m0_tail = tail_metrics(
            sample.loc[sample["M0_top_5pct"].astype(bool)],
            model="M0",
        )
        comparison = tail_comparison_metrics(m1_tail, m0_tail)
        session_multiplicities = {
            session: int(multiplicity[np.flatnonzero(session_labels == session)[0]])
            for session in unique_sessions
        }
        rows.append(
            {
                "record_type": "draw",
                "draw": draw,
                "seed": BOOTSTRAP_SEED,
                "session_multiplicities_json": json.dumps(
                    session_multiplicities,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "M1_minus_M0_log_loss_improvement": increment["log_loss_improvement"],
                "M1_minus_M0_brier_improvement": increment["brier_improvement"],
                "M1_minus_M0_auc_improvement": increment["auc_improvement"],
                "M1_minus_M0_average_precision_improvement": increment[
                    "average_precision_improvement"
                ],
                "M1_top_5pct_mean_iv_residual": m1_tail["mean_iv_residual"],
                "M1_top_5pct_median_iv_residual": m1_tail["median_iv_residual"],
                "M1_top_5pct_exceed_iv_rate": m1_tail["exceed_iv_rate"],
                "M1_minus_M0_top_5pct_mean_iv_residual": comparison["mean_iv_residual_difference"],
                "M1_minus_M0_top_5pct_median_iv_residual": comparison[
                    "median_iv_residual_difference"
                ],
                "M1_minus_M0_top_5pct_exceed_iv_rate": comparison["exceed_iv_rate_difference"],
            }
        )
    draws = pd.DataFrame(rows)
    intervals: list[dict[str, object]] = []
    for statistic in statistic_names:
        values = pd.to_numeric(draws[statistic], errors="raise").to_numpy(float)
        for level in (0.80, 0.90, 0.95):
            alpha = (1.0 - level) / 2.0
            intervals.append(
                {
                    "record_type": "interval",
                    "statistic": statistic,
                    "interval_level": level,
                    "lower": float(np.quantile(values, alpha)),
                    "upper": float(np.quantile(values, 1.0 - alpha)),
                    "draws": 10,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    return pd.concat([draws, pd.DataFrame(intervals)], ignore_index=True)


def bootstrap_lower(
    bootstrap: pd.DataFrame,
    statistic: str,
    *,
    level: float = 0.80,
) -> float:
    row = bootstrap.loc[
        bootstrap["record_type"].eq("interval")
        & bootstrap["statistic"].eq(statistic)
        & bootstrap["interval_level"].eq(level)
    ]
    if len(row) != 1:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"bootstrap interval missing: {statistic}/{level}",
        )
    return float(row.iloc[0]["lower"])


def h0_null_refits(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    real_increment: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exactly three complete-H0-bundle null refits."""

    training = development.copy()
    training["period"] = "development"
    evaluation = holdout.copy()
    evaluation["period"] = "holdout"
    combined = pd.concat([training, evaluation], ignore_index=True, sort=False)
    combined["__null_source_ordinal"] = np.arange(len(combined), dtype=np.int64)
    source_row_ids = combined["row_id"].astype(str).to_numpy()
    m0_metrics = model_metric_row(
        evaluation,
        model="M0",
        probability_column="M0_probability",
    )
    rows: list[dict[str, object]] = []
    evidence = evaluation.loc[:, ["row_id"]].reset_index(drop=True)
    for null_index, seed in enumerate(NULL_SEEDS):
        permuted = intraday_h0_bundle_null(
            combined,
            group_i_columns=(*GROUP_I, "__null_source_ordinal"),
            seed=seed,
        )
        permuted_training = permuted.loc[permuted["period"].eq("development")].copy()
        permuted_holdout = permuted.loc[permuted["period"].eq("holdout")].copy()
        if (
            not permuted_holdout["row_id"]
            .astype(str)
            .reset_index(drop=True)
            .equals(evidence["row_id"].astype(str))
        ):
            raise AssertionError("null holdout evidence row order drifted")
        null_model = fit_cross_market_model(
            permuted_training,
            model_id=f"M1_null_{null_index}",
            numeric_features=(*GROUP_O, *GROUP_I),
            category_control_names=("stock",),
            target_column=TARGET_COLUMN,
            kind="logistic",
        )
        permuted_holdout["null_probability"] = null_model.predict(permuted_holdout)
        source_ordinals = pd.to_numeric(
            permuted_holdout["__null_source_ordinal"],
            errors="raise",
        ).astype(int)
        evidence[f"M1_null_{null_index}_source_row_id"] = source_row_ids[source_ordinals.to_numpy()]
        evidence[f"M1_null_{null_index}_probability"] = permuted_holdout[
            "null_probability"
        ].to_numpy(float)
        null_metrics = model_metric_row(
            permuted_holdout,
            model=f"M1_null_{null_index}",
            probability_column="null_probability",
        )
        increment = model_increment(m0_metrics, null_metrics)
        rows.append(
            {
                "null_refit": null_index,
                "seed": seed,
                "bundle": "complete_Group_I",
                "strata": "training_or_holdout_x_session_x_checkpoint",
                "M0_refit": False,
                "M1_refit": True,
                "model_specification_json": json.dumps(
                    model_specification(null_model),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **increment,
                "real_exceeds_null_log_loss_improvement": float(
                    real_increment["log_loss_improvement"]
                )
                > float(cast(Any, increment["log_loss_improvement"])),
                "real_exceeds_null_brier_improvement": float(real_increment["brier_improvement"])
                > float(cast(Any, increment["brier_improvement"])),
                "real_exceeds_null_auc_improvement": float(real_increment["auc_improvement"])
                > float(cast(Any, increment["auc_improvement"])),
                "real_exceeds_null_average_precision_improvement": float(
                    real_increment["average_precision_improvement"]
                )
                > float(cast(Any, increment["average_precision_improvement"])),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 3:
        raise AssertionError("intraday-H0 null must contain exactly three refits")
    return result, evidence


def monthly_positive_counts(monthly: pd.DataFrame) -> tuple[int, int, int]:
    increments = monthly.loc[monthly["record_type"].eq("increment")]
    positive_log_loss = int(increments["log_loss_improvement"].gt(0.0).sum())
    tail = monthly.loc[monthly["record_type"].eq("tail") & monthly["model"].eq("M1")]
    positive_mean = int(tail["mean_iv_residual"].gt(0.0).sum())
    positive_median = int(tail["median_iv_residual"].gt(0.0).sum())
    return positive_log_loss, positive_mean, positive_median


def adverse_checkpoint_groups(checkpoint: pd.DataFrame) -> int:
    rows = checkpoint.loc[
        checkpoint["record_type"].eq("increment") & checkpoint["scope"].eq("checkpoint_group")
    ]
    return int(
        (rows["log_loss_improvement"].lt(-1e-12) | rows["brier_improvement"].lt(-1e-12)).sum()
    )


def create_plots(
    frame: pd.DataFrame,
    m0_tail: pd.DataFrame,
    m1_tail: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    target = frame[TARGET_COLUMN].to_numpy(int)
    weights = frame["row_weight"].to_numpy(float)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for model, color in (("M0", "#777777"), ("M1", "#006d77")):
        probability = frame[f"{model}_probability"].to_numpy(float)
        fpr, tpr, _ = roc_curve(target, probability, sample_weight=weights)
        precision, recall, _ = precision_recall_curve(
            target,
            probability,
            sample_weight=weights,
        )
        axes[0].plot(fpr, tpr, label=model, color=color)
        axes[1].plot(recall, precision, label=model, color=color)
    metrics, _ = prediction_metrics(frame)
    proper = metrics.loc[metrics["model"].isin(["M0", "M1"])]
    x = np.arange(2)
    width = 0.35
    axes[2].bar(
        x - width / 2,
        proper["log_loss"].to_numpy(float),
        width,
        label="log loss",
    )
    axes[2].bar(
        x + width / 2,
        proper["brier_score"].to_numpy(float),
        width,
        label="Brier",
    )
    axes[2].set_xticks(x, proper["model"].astype(str))
    axes[0].set(title="Holdout ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Holdout precision-recall", xlabel="Recall", ylabel="Precision")
    axes[2].set(title="Proper scores", ylabel="Lower is better")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(REPORTS / "m0_m1_holdout_forecast.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(
        m0_tail["iv_absolute_residual_15m"],
        bins=30,
        alpha=0.55,
        label="M0",
        color="#777777",
    )
    axes[0].hist(
        m1_tail["iv_absolute_residual_15m"],
        bins=30,
        alpha=0.55,
        label="M1",
        color="#006d77",
    )
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set(title="Frozen top-5% IV residual", xlabel="15-minute residual")
    axes[0].legend()
    axes[1].plot(
        timing["horizon_minutes"],
        timing["mean_iv_residual"],
        marker="o",
        label="mean residual",
    )
    axes[1].plot(
        timing["horizon_minutes"],
        timing["median_iv_residual"],
        marker="s",
        label="median residual",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(
        title="M1 frozen-tail movement timing",
        xlabel="Minutes",
        ylabel="IV absolute residual",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORTS / "frozen_tail_iv_residual_timing.png", dpi=160)
    plt.close(figure)


def concentration_table(frame: pd.DataFrame) -> pd.DataFrame:
    tail = frame.loc[frame["M1_top_5pct"].astype(bool)].copy()
    tail["month"] = pd.to_datetime(tail["session"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for scope, column in (
        ("stock", "symbol"),
        ("month", "month"),
        ("session", "session"),
    ):
        counts = tail[column].astype(str).value_counts()
        for group, count in counts.items():
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "rows": int(count),
                    "row_share": float(count / len(tail)),
                }
            )
    return pd.DataFrame(rows)


def report_text(
    *,
    decision: Mapping[str, object],
    reconstruction: Mapping[str, object],
    download: Mapping[str, object],
    coverage: Mapping[str, object],
    support: Mapping[str, object],
    tail_support_value: Mapping[str, object],
    model_metrics: pd.DataFrame,
    increment: Mapping[str, float],
    tail_rows: pd.DataFrame,
    comparison: Mapping[str, float],
    overlap: Mapping[str, object],
    timing: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
) -> str:
    indexed = model_metrics.loc[model_metrics["model"].isin(["M0", "M1"])].set_index("model")
    m1_tail = tail_rows.loc[tail_rows["model"].eq("M1")].iloc[0]
    thresholds = cast(Mapping[str, Any], reconstruction["thresholds"])
    interval_rows = bootstrap.loc[
        bootstrap["record_type"].eq("interval") & bootstrap["interval_level"].eq(0.80)
    ]
    lines = [
        "# Minimal Intraday Stock → IV-Excess Holdout Validation V0",
        "",
        f"Overall decision: `{decision['overall_decision']}`.",
        "",
        "This is a retrospective, research-only frozen holdout validation. It is not an "
        "option P&L backtest, prospective validation, or deployable trading strategy.",
        "",
        "## Frozen design",
        "",
        "- Training: 2024-01-01 through 2024-12-31.",
        "- Prior reference: 2025-01-01 through 2025-08-22, compatibility only.",
        "- Binding holdout: actual XNYS sessions 2025-09-01 through 2025-12-31.",
        "- M0: Group O; M1: Group O + Group I; stock fixed effects in both.",
        "- Excluded: daily stock, route competition, route state, and mismatch features.",
        f"- Historical M0/G0 reconstruction passed: `{reconstruction['passed']}`.",
        f"- Frozen M0 top-5% threshold: {float(thresholds['M0_top_5_percent_threshold']):.12g}.",
        f"- Frozen M1 top-5% threshold: {float(thresholds['M1_top_5_percent_threshold']):.12g}.",
        "",
        "## Holdout support and options",
        "",
        f"- Download requests: {int(download['requests_completed'])}; returned records: "
        f"{int(download['records_returned'])}; canonical exact-date records: "
        f"{int(download['canonical_rows'])}.",
        f"- Selected exact previous-close pair sessions: "
        f"{int(coverage['selected_pair_sessions'])}.",
        f"- Joined rows/sessions/stocks: {int(support['rows'])}/"
        f"{int(support['sessions'])}/{int(support['stocks'])}.",
        f"- Joined support passed: `{support['passed']}`.",
        f"- Frozen M1 tail support passed: `{tail_support_value['passed']}`.",
        "",
        "## Holdout forecast",
        "",
    ]
    for model in ("M0", "M1"):
        row = indexed.loc[model]
        lines.append(
            f"- {model}: log loss {float(row['log_loss']):.8f}, Brier "
            f"{float(row['brier_score']):.8f}, AUC {float(row['auc']):.8f}, AP "
            f"{float(row['average_precision']):.8f}."
        )
    lines.extend(
        [
            f"- M1−M0: log-loss improvement {increment['log_loss_improvement']:.8f}, "
            f"Brier improvement {increment['brier_improvement']:.8f}, AUC improvement "
            f"{increment['auc_improvement']:.8f}, AP improvement "
            f"{increment['average_precision_improvement']:.8f}.",
            "",
            "## Frozen M1 top-5% tail",
            "",
            f"- Rows/sessions/stocks/months: {int(m1_tail['rows'])}/"
            f"{int(m1_tail['sessions'])}/{int(m1_tail['stocks'])}/"
            f"{int(m1_tail['months'])}.",
            f"- Mean IV residual: {float(m1_tail['mean_iv_residual']):.8f}.",
            f"- Median IV residual: {float(m1_tail['median_iv_residual']):.8f}.",
            f"- Exceed-IV rate: {float(m1_tail['exceed_iv_rate']):.6f}.",
            f"- M1−M0 tail mean-residual difference: "
            f"{comparison['mean_iv_residual_difference']:.8f}.",
            f"- Tail Jaccard overlap: {float(cast(Any, overlap['jaccard_overlap'])):.6f}.",
            "",
            "## Movement timing",
            "",
        ]
    )
    for row in timing.itertuples(index=False):
        lines.append(
            f"- {int(row.horizon_minutes)}m: mean residual "
            f"{float(row.mean_iv_residual):.8f}, median residual "
            f"{float(row.median_iv_residual):.8f}, exceed rate "
            f"{float(row.exceed_iv_rate):.6f}."
        )
    lines.extend(["", "## Coarse resampling", ""])
    for row in interval_rows.itertuples(index=False):
        lines.append(f"- {row.statistic} 80%: [{float(row.lower):.8f}, {float(row.upper):.8f}].")
    lines.extend(
        [
            "",
            f"The real log-loss increment exceeded "
            f"{int(nulls['real_exceeds_null_log_loss_improvement'].sum())}/3 H0 nulls; "
            f"the real Brier increment exceeded "
            f"{int(nulls['real_exceeds_null_brier_improvement'].sum())}/3.",
            "",
            "Ten bootstrap draws are a coarse whole-session diagnostic. The 15-minute "
            "target remains binding regardless of the timing diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    *,
    provider_root: Path,
    options_cache: Path,
    stock_cache: Path,
) -> dict[str, Any]:
    load_contract()
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    feature_manifest = minimal_feature_manifest()
    write_json(PRIMARY / "minimal_feature_manifest.json", feature_manifest)

    historical_path = historical_panel_path()
    historical = pd.read_parquet(historical_path)
    models, development, _reference, reconstruction = reconstruct_historical_models(historical)
    thresholds = cast(dict[str, Any], reconstruction.pop("thresholds"))
    write_json(
        PRIMARY / "historical_model_reconstruction.json",
        cast(Mapping[str, Any], reconstruction),
    )
    write_json(
        PRIMARY / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "M0": {
                "numeric_features": list(GROUP_O),
                "categorical_controls": ["stock"],
                "penalty": "l2",
                "C": 0.25,
                "solver": "liblinear",
                "max_iter": 300,
                "class_weight": None,
                "n_jobs": 1,
                "fitted_period": "2024_only",
            },
            "M1": {
                "numeric_features": [*GROUP_O, *GROUP_I],
                "categorical_controls": ["stock"],
                "penalty": "l2",
                "C": 0.25,
                "solver": "liblinear",
                "max_iter": 300,
                "class_weight": None,
                "n_jobs": 1,
                "fitted_period": "2024_only",
            },
            "primary_models_fitted": 2,
        },
    )
    coefficients = {
        **SAFETY_FLAGS,
        "M0": model_specification(models.m0),
        "M1": model_specification(models.m1),
    }
    write_json(PRIMARY / "model_coefficients.json", coefficients)
    write_json(
        PRIMARY / "frozen_tail_thresholds.json",
        {**SAFETY_FLAGS, **thresholds, "written_before_holdout_outcomes": True},
    )
    threshold_hash_before_outcomes = sha256_file(PRIMARY / "frozen_tail_thresholds.json")

    h0_surface, states, h0_manifest = load_or_build_h0_surface(
        provider_root,
        stock_cache,
    )
    option_context, option_coverage, option_audit, coverage_manifest = build_holdout_option_context(
        h0_surface,
        provider_root=provider_root,
        options_cache=options_cache,
    )
    write_parquet(PRIMARY / "holdout_selected_option_pairs.parquet", option_context)
    write_csv(PRIMARY / "holdout_options_coverage.csv", option_coverage)
    feature_panel, structural_join_audit = join_holdout_panel(h0_surface, option_context)
    feature_panel["M0_probability"] = models.m0.predict(feature_panel)
    feature_panel["M1_probability"] = models.m1.predict(feature_panel)
    feature_panel["M0_top_5pct"] = frozen_tail_membership(
        feature_panel["M0_probability"].to_numpy(float),
        float(thresholds["M0_top_5_percent_threshold"]),
    )
    feature_panel["M1_top_5pct"] = frozen_tail_membership(
        feature_panel["M1_probability"].to_numpy(float),
        float(thresholds["M1_top_5_percent_threshold"]),
    )
    if sha256_file(PRIMARY / "frozen_tail_thresholds.json") != threshold_hash_before_outcomes:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "frozen threshold artifact changed before holdout outcome construction",
        )

    holdout = attach_movement_prices(feature_panel, states)
    holdout = add_movement_outcomes(holdout)
    validate_holdout_dates(holdout["session"])
    if len(holdout) != len(feature_panel):
        raise ScreenBlocked(
            "blocked_insufficient_holdout_support",
            "a structurally eligible joined row lacked a frozen movement horizon",
        )
    holdout = holdout.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    write_parquet(PRIMARY / "holdout_panel.parquet", holdout)
    prediction_columns = [
        "row_id",
        "symbol",
        "session",
        "checkpoint",
        "row_weight",
        "M0_probability",
        "M1_probability",
        "M0_top_5pct",
        "M1_top_5pct",
        TARGET_COLUMN,
        *(f"absolute_log_return_{horizon}m" for horizon in HORIZONS),
        *(f"iv_expected_absolute_{horizon}m" for horizon in HORIZONS),
        *(f"iv_absolute_residual_{horizon}m" for horizon in HORIZONS),
    ]
    write_parquet(
        PRIMARY / "holdout_predictions.parquet",
        holdout.loc[:, prediction_columns],
    )

    model_metrics, increment = prediction_metrics(holdout)
    write_csv(PRIMARY / "holdout_model_metrics.csv", model_metrics)
    monthly, checkpoint = stability_tables(holdout, thresholds)
    write_csv(PRIMARY / "holdout_monthly_metrics.csv", monthly)
    write_csv(PRIMARY / "holdout_checkpoint_metrics.csv", checkpoint)

    m1_tail_frame = holdout.loc[holdout["M1_top_5pct"].astype(bool)].copy()
    m0_tail_frame = holdout.loc[holdout["M0_top_5pct"].astype(bool)].copy()
    if m1_tail_frame.empty or m0_tail_frame.empty:
        raise ScreenBlocked(
            "blocked_insufficient_frozen_tail_support",
            "a frozen top-five-percent threshold selected no holdout rows",
        )
    m1_tail = tail_metrics(m1_tail_frame, model="M1")
    m0_tail = tail_metrics(m0_tail_frame, model="M0")
    tail_table = pd.DataFrame([m0_tail, m1_tail])
    comparison = tail_comparison_metrics(m1_tail, m0_tail)
    overlap = tail_overlap_metrics(
        holdout["M1_top_5pct"].to_numpy(bool),
        holdout["M0_top_5pct"].to_numpy(bool),
    )
    write_csv(PRIMARY / "tail_metrics.csv", tail_table)
    write_csv(PRIMARY / "tail_comparison_metrics.csv", pd.DataFrame([comparison]))
    write_csv(PRIMARY / "tail_overlap_metrics.csv", pd.DataFrame([overlap]))
    timing = movement_timing_metrics(m1_tail_frame)
    write_csv(PRIMARY / "movement_timing_metrics.csv", timing)
    concentration = concentration_table(holdout)
    write_csv(PRIMARY / "concentration_metrics.csv", concentration)

    bootstrap = bootstrap_statistics(holdout)
    write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap)
    nulls, null_evidence = h0_null_refits(
        development,
        holdout,
        real_increment=increment,
    )
    if not null_evidence["row_id"].astype(str).equals(holdout["row_id"].astype(str)):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "null evidence row order differs from the frozen holdout",
        )
    null_evidence_columns = [name for name in null_evidence.columns if name != "row_id"]
    for name in null_evidence_columns:
        holdout[name] = null_evidence[name].to_numpy()
    prediction_columns.extend(null_evidence_columns)
    write_parquet(
        PRIMARY / "holdout_predictions.parquet",
        holdout.loc[:, prediction_columns],
    )
    write_csv(PRIMARY / "intraday_h0_null_metrics.csv", nulls)

    joined_gate = joined_support(holdout)
    tail_gate_support = frozen_tail_support(m1_tail_frame)
    positive_log_months, positive_mean_months, positive_median_months = monthly_positive_counts(
        monthly
    )
    adverse = adverse_checkpoint_groups(checkpoint)
    real_exceeds_null = bool(
        nulls["real_exceeds_null_log_loss_improvement"].sum() == 3
        or nulls["real_exceeds_null_brier_improvement"].sum() == 3
    )
    model_gate = ModelGateInputs(
        log_loss_improvement=increment["log_loss_improvement"],
        brier_improvement=increment["brier_improvement"],
        auc_improvement=increment["auc_improvement"],
        average_precision_improvement=increment["average_precision_improvement"],
        bootstrap_80_log_loss_lower=bootstrap_lower(
            bootstrap,
            "M1_minus_M0_log_loss_improvement",
        ),
        bootstrap_80_brier_lower=bootstrap_lower(
            bootstrap,
            "M1_minus_M0_brier_improvement",
        ),
        bootstrap_80_average_precision_lower=bootstrap_lower(
            bootstrap,
            "M1_minus_M0_average_precision_improvement",
        ),
        positive_log_loss_months=positive_log_months,
        materially_adverse_checkpoint_groups=adverse,
        real_exceeds_all_nulls=real_exceeds_null,
        support_passed=bool(joined_gate["passed"]),
    )
    concentration_passed = bool(
        tail_gate_support["passed"]
        and math.isfinite(float(m1_tail["top_5pct_positive_residual_contribution"]))
        and float(m1_tail["top_5pct_positive_residual_contribution"]) <= 0.50
    )
    tail_gate = TailGateInputs(
        mean_iv_residual=float(m1_tail["mean_iv_residual"]),
        median_iv_residual=float(m1_tail["median_iv_residual"]),
        exceed_iv_rate=float(m1_tail["exceed_iv_rate"]),
        bootstrap_80_mean_lower=bootstrap_lower(
            bootstrap,
            "M1_top_5pct_mean_iv_residual",
        ),
        bootstrap_80_median_lower=bootstrap_lower(
            bootstrap,
            "M1_top_5pct_median_iv_residual",
        ),
        positive_mean_months=positive_mean_months,
        positive_median_months=positive_median_months,
        m1_minus_m0_mean=comparison["mean_iv_residual_difference"],
        bootstrap_80_difference_lower=bootstrap_lower(
            bootstrap,
            "M1_minus_M0_top_5pct_mean_iv_residual",
        ),
        concentration_passed=concentration_passed,
        support_passed=bool(tail_gate_support["passed"]),
    )
    decision = decide_experiment(model=model_gate, tail=tail_gate)
    coverage_supported = bool(
        coverage_manifest["selected_pair_stocks"] >= 15
        and coverage_manifest["selected_pair_months"] == 4
        and joined_gate["passed"]
    )
    decision["holdout_options_coverage_status"] = (
        "supported" if coverage_supported else "insufficient_support"
    )
    if not coverage_supported:
        decision["overall_decision"] = "blocked_insufficient_holdout_options_coverage"
    elif not joined_gate["passed"]:
        decision["overall_decision"] = "blocked_insufficient_holdout_support"
    decision["joined_holdout_support"] = joined_gate
    decision["frozen_tail_support"] = tail_gate_support
    decision["threshold_artifact_sha256_before_outcomes"] = threshold_hash_before_outcomes
    assert_safety_flags(decision)
    write_json(PRIMARY / "decision.json", decision)

    combined_join_audit = structural_join_audit.copy()
    combined_join_audit["audit_type"] = "structural_rows"
    write_csv(PRIMARY / "holdout_join_audit.csv", combined_join_audit)
    download_manifest = read_json(PRIMARY / "holdout_options_download_manifest.json")
    source_manifest = {
        **SAFETY_FLAGS,
        "starting_branch": "agent/stock-layer-iv-excess-attribution-quick-v0",
        "starting_sha": "aad974a4e2c12fe3aeb5290540f84ed61480036e",
        "final_branch": "agent/minimal-intraday-iv-excess-holdout-v0",
        "historical_panel": str(historical_path),
        "historical_panel_sha256": sha256_file(historical_path),
        "provider_root": str(provider_root),
        "options_cache": str(options_cache),
        "options_cache_sha256": sha256_file(options_cache),
        "h0_reconstruction": h0_manifest,
        "options_coverage": coverage_manifest,
        "download": download_manifest,
        "raw_vendor_data_committed": False,
        "canonical_vendor_data_committed": False,
        "protected_rows_materialised": 0,
    }
    write_json(PRIMARY / "source_manifest.json", source_manifest)
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "protected_start": "2026-01-01",
            "protected_underlying_rows_materialised": 0,
            "protected_option_rows_materialised": 0,
            "holdout_rows_outside_authorized_dates": 0,
            "same_day_or_future_option_observations": int(
                coverage_manifest["same_day_or_future_observations"]
            ),
            "non_exact_previous_session_option_observations": int(
                coverage_manifest["non_exact_previous_session_observations"]
            ),
            "passed": True,
        },
    )
    write_json(
        PRIMARY / "holdout_data_authorisation.json",
        {
            **SAFETY_FLAGS,
            "authorized_calendar_range": ["2025-09-01", "2025-12-31"],
            "actual_xnys_sessions": int(holdout["session"].nunique()),
            "protected_start": "2026-01-01",
            "other_protected_period_opened": False,
            "economic_option_pnl_outcome_opened": False,
            "holdout_outcomes_attached_after_threshold_artifact": True,
            "threshold_artifact_sha256": threshold_hash_before_outcomes,
            "passed": True,
        },
    )
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            **SAFETY_FLAGS,
            "status": "pending_independent_audit",
            "runner_self_checks_passed": True,
            "historical_reconstruction_passed": True,
            "protected_boundary_passed": True,
            "exact_previous_session_chronology_passed": True,
        },
    )
    create_plots(holdout, m0_tail_frame, m1_tail_frame, timing)
    report = report_text(
        decision=decision,
        reconstruction={**reconstruction, "thresholds": thresholds},
        download=download_manifest,
        coverage=coverage_manifest,
        support=joined_gate,
        tail_support_value=tail_gate_support,
        model_metrics=model_metrics,
        increment=increment,
        tail_rows=tail_table,
        comparison=comparison,
        overlap=overlap,
        timing=timing,
        bootstrap=bootstrap,
        nulls=nulls,
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    return {
        "decision": decision,
        "rows": len(holdout),
        "sessions": int(holdout["session"].nunique()),
        "stocks": int(holdout["symbol"].nunique()),
        "M0_threshold": thresholds["M0_top_5_percent_threshold"],
        "M1_threshold": thresholds["M1_top_5_percent_threshold"],
        "M1_tail": m1_tail,
        "increment": increment,
    }


def blocked_decision(error: ScreenBlocked) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "overall_decision": error.decision,
        "minimal_model_status": "blocked",
        "frozen_top_5pct_status": "blocked",
        "options_only_tail_comparison_status": "blocked",
        "movement_timing_status": "blocked",
        "holdout_options_coverage_status": "blocked",
        "blocker_detail": error.detail,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    parser.add_argument("--options-cache", type=Path, default=DEFAULT_OPTIONS_CACHE)
    parser.add_argument("--stock-cache", type=Path, default=DEFAULT_STOCK_CACHE)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        result = execute(
            provider_root=arguments.provider_root,
            options_cache=arguments.options_cache,
            stock_cache=arguments.stock_cache,
        )
    except ScreenBlocked as error:
        PRIMARY.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        decision = blocked_decision(error)
        write_json(PRIMARY / "decision.json", decision)
        text = (
            "# Minimal Intraday Stock → IV-Excess Holdout Validation V0\n\n"
            f"Overall decision: `{error.decision}`.\n\n{error.detail}\n"
        )
        (PRIMARY / "report.md").write_text(text, encoding="utf-8")
        (REPORTS / "report.md").write_text(text, encoding="utf-8")
        print(f"{error.decision}: {error.detail}")
        return 2
    print(
        f"{result['decision']['overall_decision']}: {result['rows']} rows, "
        f"{result['sessions']} sessions, {result['stocks']} stocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
