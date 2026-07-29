"""Independent pre-score audit for factor-conditioned loop occurrence V1.

The auditor deliberately does not import the production core, evaluator, or
runner.  It reconstructs the 2024 population, causal factors, labels, path
offsets, residual designs, fits, predictions, and statistical decision from
the frozen contract and immutable 2024 sources.  It has no later-period path
surface and can only authorize the separate production score-only phase.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260711-factor-conditioned-loop-occurrence-v1.json"
CONTRACT_SHA256 = "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d"
RUNNER_PATH = HERE / "run_factor_conditioned_loop_occurrence_v1.py"
CORE_PATH = HERE / "factor_conditioned_loop_occurrence_core.py"
EVALUATOR_PATH = HERE / "factor_conditioned_loop_occurrence_eval.py"
DEFAULT_ROOT = Path("/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711")

END_STATE = 8
STATE_COUNT = 8
DESTINATION_COUNT = 9
TOKEN_WIDTH = 648
CYCLE_COUNT = 20
ROUTE_COUNT = 44
CYCLE_CONTRAST_WIDTH = 19
ROUTE_CONTRAST_WIDTH = 24
PATTERN_WIDTH = 44
LIMITED_WIDTH = 2812
FULL_WIDTH = 6272
EPSILON = 1e-12
RIDGE_GRID = (0.0001, 0.0003, 0.001, 0.003)
HEADS = ("qpattern", "qlimited4", "qfull9")
FACTOR_COUNT = {"qpattern": 0, "qlimited4": 4, "qfull9": 9}
OUTER_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
GRID_MONTHS = tuple(f"2024-{month:02d}" for month in range(4, 13))
INNER_SCHEDULE = {
    outer: tuple(f"2024-{month:02d}" for month in range(4, int(outer[-2:])))
    for outer in OUTER_MONTHS
}
LIMITED4 = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
)
NEW5 = (
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
FULL9 = LIMITED4 + NEW5
QUARTILE_COLUMNS = ("entry_minutes", *NEW5)
MODEL_COLUMNS = (
    "qhistory",
    "qpattern",
    "qlimited4",
    "qold_limited_path",
    "qfull9",
)
PRIMARY_BASELINES = ("qhistory", "qpattern", "qlimited4")
LINEAGE_BASELINE = "qold_limited_path"
PRIMARY_CANDIDATE = "qfull9"
IRREGULAR_DATE = "2025-04-10"  # Contract metadata only; no path is resolved.
OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close")
NATURAL_KEY = ("symbol_norm", "session_date", "start_timestamp")
SAFETY = {
    "research_only": True,
    "live_ordering_enabled": False,
    "order_placement": "disabled",
}

REQUIRED_REJECTION_ARTIFACTS = frozenset(
    {
        "fit_source_manifest_2024.json",
        "gates_2024.json",
        "grid_replay_inputs_2024.npz",
        "lambda_grid_scores_2024.csv",
        "oof_predictions_2024.parquet",
        "outer_fit_audit_2024.json",
        "outer_lambda_selection_2024.csv",
        "outer_replay_inputs_2024.npz",
        "population_validation_2024.json",
        "fold_schedule.json",
        "hyperparameter_grid.json",
        "provisional_decision.json",
        "support_2024.json",
        "overall_2024.csv",
        "ranking_2024.csv",
        "calibration_2024.json",
        "comparisons_2024.csv",
        "bootstrap_2024.json",
        "slices_2024.csv",
        "falsification_2024.json",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


@dataclass
class Audit:
    checks: list[dict[str, Any]]

    def __init__(self) -> None:
        self.checks = []

    def check(self, name: str, passed: bool, details: Any = None) -> None:
        self.checks.append(
            {"name": name, "passed": bool(passed), "details": json_safe(details)}
        )

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(row["passed"] for row in self.checks)


@dataclass(frozen=True)
class Kernel:
    classes: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    numeric_width: int
    iterations: int


@dataclass(frozen=True)
class Transform:
    medians: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True)
class Contrasts:
    cycle_weights: np.ndarray
    route_weights: np.ndarray
    route_references: np.ndarray


@dataclass(frozen=True)
class RidgeFit:
    coefficients: np.ndarray
    ridge_lambda: float
    factor_count: int
    objective: float
    gradient_max_abs: float
    iterations: int
    message: str


@dataclass
class Prepared2024:
    anchors: pd.DataFrame
    expanded: pd.DataFrame
    cycles: pd.DataFrame
    routes: pd.DataFrame
    factor_hash: str
    source_audit: dict[str, Any]


def load_contract() -> dict[str, Any]:
    if sha256(CONTRACT_PATH) != CONTRACT_SHA256:
        raise AssertionError("frozen factor-conditioned contract changed")
    contract = json.loads(CONTRACT_PATH.read_text())
    if any(contract.get(key) != value for key, value in SAFETY.items()):
        raise AssertionError("research safety labels changed")
    if contract["periods"]["forbidden"] != [2026, "prospective_shadow"]:
        raise AssertionError("forbidden-period boundary changed")
    if tuple(contract["models"]["ridge_lambda_grid"]) != RIDGE_GRID:
        raise AssertionError("ridge grid changed")
    return contract


def _artifact_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise AssertionError("artifact name must be a nonempty string")
    relative = Path(name)
    if relative.is_absolute() or relative.name != name or name in {".", ".."}:
        raise AssertionError(f"unsafe artifact name: {name}")
    return name


def verify_fit_lock(audit: Audit, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        audit.check("fit_root_is_real_directory", False, str(root))
        return {}, {}
    fit_path = root / "fit_complete.json"
    manifest_path = root / "complete_fit_artifact_manifest.json"
    if any(path.is_symlink() or not path.is_file() for path in (fit_path, manifest_path)):
        audit.check("fit_markers_present_and_not_symlinks", False)
        return {}, {}
    fit = json.loads(fit_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("artifacts")
    if not isinstance(entries, Mapping) or not entries:
        audit.check("fit_manifest_nonempty", False)
        return fit, manifest
    names = {_artifact_name(name) for name in entries}
    later_like = sorted(
        name
        for name in names
        if "_2025" in Path(name).stem
        or "_2023" in Path(name).stem
        or name.startswith("scoring_")
        or name.startswith("predictions_") and name != "oof_predictions_2024.parquet"
    )
    rejection = fit.get("development_2024_primary_pass") is False
    expected_status = (
        "stopped_2024_primary_gates_failed"
        if rejection
        else "fit_frozen_pending_independent_pre_score_audit"
    )
    full_only = {
        "model_bundle.npz",
        "full_fit_audit_2024.json",
        "full_lambda_selection_2024.csv",
        "full_replay_inputs_2024.npz",
    }
    marker_checks = {
        "status": fit.get("status") == expected_status,
        "contract": fit.get("contract_sha256") == CONTRACT_SHA256,
        "runner": fit.get("runner_sha256") == sha256(RUNNER_PATH),
        "primary_boolean": type(fit.get("development_2024_primary_pass")) is bool,
        "selection_state": (
            fit.get("selected_full_lambdas") is None
            if rejection
            else isinstance(fit.get("selected_full_lambdas"), Mapping)
        ),
        "scoring_closed": fit.get("scoring_authorized") is False,
        "later_closed": fit.get("later_period_paths_resolved") is False
        and fit.get("later_period_rows_read") is False,
        "shadow_closed": fit.get("shadow_tree_read") is False
        and fit.get("shadow_tree_written") is False,
        "safety": all(fit.get(key) == value for key, value in SAFETY.items()),
        "manifest_hash": fit.get("complete_fit_artifact_manifest_sha256")
        == sha256(manifest_path),
        "manifest_contract": manifest.get("contract_sha256") == CONTRACT_SHA256,
        "manifest_runner": manifest.get("runner_sha256") == sha256(RUNNER_PATH),
        "manifest_later_closed": manifest.get("later_period_paths_resolved") is False,
        "manifest_safety": all(manifest.get(key) == value for key, value in SAFETY.items()),
        "required": REQUIRED_REJECTION_ARTIFACTS.issubset(names),
        "rejection_manifest_exact": (
            names == REQUIRED_REJECTION_ARTIFACTS if rejection else True
        ),
        "full_artifact_state": (
            full_only.isdisjoint(names) if rejection else full_only.issubset(names)
        ),
        "no_later": not later_like,
    }
    audit.check("fit_lock_semantics_exact", all(marker_checks.values()), marker_checks)
    source_hashes = fit.get("production_source_hashes", {})
    expected_source_hashes = {
        CORE_PATH.name: sha256(CORE_PATH),
        EVALUATOR_PATH.name: sha256(EVALUATOR_PATH),
    }
    audit.check(
        "production_sources_unchanged_after_fit",
        source_hashes == expected_source_hashes,
        {"stored": source_hashes, "expected": expected_source_hashes},
    )
    artifact_errors: list[str] = []
    for name, metadata in entries.items():
        name = _artifact_name(name)
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(metadata, Mapping)
            or sha256(path) != metadata.get("sha256")
            or path.stat().st_size != int(metadata.get("size", -1))
        ):
            artifact_errors.append(name)
    audit.check("every_fit_artifact_hash_and_size_exact", not artifact_errors, artifact_errors)
    allowed = names | {"fit_complete.json", "complete_fit_artifact_manifest.json"}
    unexpected = sorted(
        path.name
        for path in root.iterdir()
        if path.name not in allowed or path.is_dir() or path.is_symlink()
    )
    audit.check("pre_score_namespace_pristine", not unexpected, unexpected)
    return fit, manifest


def canonical_cycle(values: Iterable[int]) -> tuple[int, ...]:
    core = tuple(int(value) for value in values)
    if not core:
        raise ValueError("empty cycle")
    return min(core[index:] + core[:index] for index in range(len(core)))


def compatible_rotations(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def load_cycles(contract: Mapping[str, Any]) -> pd.DataFrame:
    spec = contract["frozen_sources"]["cycles"]
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise AssertionError("cycle source hash changed")
    source = pd.read_csv(path)
    if len(source) != int(spec["count"]):
        raise AssertionError("cycle count changed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for position, text in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in text.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError("invalid closed cycle")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError("invalid or duplicate cycle")
        seen.add(core)
        rows.append(
            {
                "cycle_index": position - 1,
                "cycle_id": f"cycle_{position:02d}",
                "cycle": "->".join(str(value) for value in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    return pd.DataFrame(rows)


def build_routes(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        states = sorted(set(int(value) for value in cycle.core))
        for state in states:
            rows.append(
                {
                    "route_index": len(rows),
                    "cycle_index": int(cycle.cycle_index),
                    "cycle_id": str(cycle.cycle_id),
                    "current_state": state,
                    "is_reference": state == max(states),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != ROUTE_COUNT or int((~result["is_reference"]).sum()) != ROUTE_CONTRAST_WIDTH:
        raise AssertionError("route mapping width changed")
    return result


def history_token(p2: Any, p1: Any, current: Any) -> np.ndarray:
    a = np.asarray(p2, dtype=int)
    b = np.asarray(p1, dtype=int)
    c = np.asarray(current, dtype=int)
    if a.shape != b.shape or b.shape != c.shape:
        raise ValueError("history arrays have different shapes")
    if (
        a.min(initial=0) < 0
        or a.max(initial=0) > END_STATE
        or b.min(initial=0) < 0
        or b.max(initial=0) > END_STATE
        or c.min(initial=0) < 0
        or c.max(initial=0) >= STATE_COUNT
    ):
        raise AssertionError("history state outside range")
    token = ((a * 9 + b) * 8 + c).astype(np.int64)
    if token.max(initial=0) >= TOKEN_WIDTH:
        raise AssertionError("history token outside width")
    return token


def load_runs_2024(contract: Mapping[str, Any]) -> pd.DataFrame:
    spec = contract["frozen_sources"]["runs_2024"]
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise AssertionError("2024 run source hash changed")
    frame = pd.read_csv(path)
    required = {
        "run_id", "symbol_norm", "session_date", "state", "start_pos",
        "start_timestamp", "previous_state_1", "previous_state_2",
        "b0_state_numeric", "b0_high_stress", "next_state", "has_next_state",
    }
    if required.difference(frame.columns):
        raise AssertionError("2024 run source lacks columns")
    output = frame.copy()
    output["symbol_norm"] = output["symbol_norm"].astype(str)
    output["session_date"] = output["session_date"].astype(str)
    output["state"] = pd.to_numeric(output["state"], errors="raise").astype(int)
    output["start_pos"] = pd.to_numeric(output["start_pos"], errors="raise").astype(int)
    output["start_timestamp"] = pd.to_datetime(output["start_timestamp"], utc=True, errors="raise")
    output = output.sort_values(["symbol_norm", "session_date", "start_pos"], kind="stable").reset_index(drop=True)
    if len(output) != int(spec["rows"]) or output["symbol_norm"].nunique() != int(spec["stocks"]):
        raise AssertionError("2024 run population changed")
    if output.duplicated(list(NATURAL_KEY)).any():
        raise AssertionError("duplicate 2024 run natural key")
    if set(pd.to_datetime(output["session_date"]).dt.year) != {2024} or output["start_timestamp"].dt.year.ne(2024).any():
        raise AssertionError("non-2024 run entered audit")
    grouped = output.groupby(["symbol_norm", "session_date"], sort=False)["state"]
    p1 = grouped.shift(1).fillna(END_STATE).astype(int)
    p2 = grouped.shift(2).fillna(END_STATE).astype(int)
    if not np.array_equal(p1, output["previous_state_1"].astype(int)) or not np.array_equal(p2, output["previous_state_2"].astype(int)):
        raise AssertionError("stored previous-state history changed")
    next_state = grouped.shift(-1)
    has_next = output["has_next_state"].astype(bool)
    if not np.array_equal(next_state.notna(), has_next):
        raise AssertionError("stored terminal flag changed")
    stored_next = pd.to_numeric(output["next_state"], errors="coerce")
    if not np.array_equal(next_state.loc[has_next].astype(int), stored_next.loc[has_next].astype(int)):
        raise AssertionError("stored next state changed")
    output["next_outcome"] = next_state.fillna(END_STATE).astype(int)
    output["terminal"] = ~has_next
    output["anchor_id"] = np.arange(len(output), dtype=np.int64)
    output["month"] = output["session_date"].str[:7]
    dates = pd.to_datetime(output["session_date"])
    output["quarter"] = dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
    output["period"] = "2024"
    for step in range(1, 5):
        output[f"future_state_{step}"] = grouped.shift(-step).fillna(END_STATE).astype(int)
    output["history_token"] = history_token(output["previous_state_2"], output["previous_state_1"], output["state"])
    return output


def provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def canonical_factor_hash(frame: pd.DataFrame) -> str:
    columns = (
        "symbol_norm", "timestamp_ns_utc", "session_date", "bar_ordinal", *NEW5
    )
    ordered = frame.loc[:, columns].sort_values(["symbol_norm", "timestamp_ns_utc"], kind="stable")
    digest = hashlib.sha256()
    digest.update((",".join(columns) + "\n").encode())
    for row in ordered.itertuples(index=False, name=None):
        fields = [
            str(row[0]), str(int(row[1])), str(row[2]), str(int(row[3])),
            *(format(float(value), ".17g") for value in row[4:]),
        ]
        digest.update((",".join(fields) + "\n").encode())
    return digest.hexdigest()


def canonical_discarded_hash(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, ["symbol_norm", "timestamp_ns_utc"]].sort_values(
        ["symbol_norm", "timestamp_ns_utc"], kind="stable"
    )
    content = "symbol_norm,timestamp_ns_utc\n" + "".join(
        f"{row.symbol_norm},{int(row.timestamp_ns_utc)}\n"
        for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(content.encode()).hexdigest()


def reconstruct_provider_2024(
    symbols: Sequence[str], contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    source_spec = contract["frozen_sources"]
    manifest_path = Path(source_spec["provider_hash_manifest"]["path"])
    if sha256(manifest_path) != source_spec["provider_hash_manifest"]["sha256"]:
        raise AssertionError("provider provenance manifest changed")
    manifest = json.loads(manifest_path.read_text())
    expected_provenance = {
        key[len("provider_2024_") : -len(".parquet")]: str(value)
        for key, value in manifest.items()
        if key.startswith("provider_2024_") and key.endswith(".parquet")
    }
    if set(expected_provenance) != set(symbols):
        raise AssertionError("provider provenance symbol set changed")
    root = Path(source_spec["provider_root_2024_2025"])
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01", tz="UTC")
    panels: list[pd.DataFrame] = []
    discarded_parts: list[pd.DataFrame] = []
    per_symbol: dict[str, int] = {}
    for symbol in sorted(symbols):
        table = pq.read_table(
            provider_path(root, symbol),
            columns=list(OHLC_COLUMNS),
            partitioning=None,
            filters=[
                ("timestamp", ">=", start.to_pydatetime()),
                ("timestamp", "<", end.to_pydatetime()),
            ],
        )
        bars = table.to_pandas()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        if bars.empty or bars["timestamp"].isna().any() or bars["timestamp"].lt(start).any() or bars["timestamp"].ge(end).any():
            raise AssertionError("year predicate admitted invalid provider rows")
        for column in ("open", "high", "low", "close"):
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
        local = bars["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        bars = bars.loc[minute.ge(570) & minute.lt(960)].copy()
        nulls = bars.loc[:, ["open", "high", "low", "close"]].isna()
        all_null = nulls.all(axis=1)
        if (nulls.any(axis=1) & ~all_null).any():
            raise AssertionError("partial provider OHLC null")
        discarded = bars.loc[all_null, ["timestamp"]].copy()
        discarded["symbol_norm"] = symbol
        discarded["timestamp_ns_utc"] = discarded["timestamp"].astype("int64")
        discarded_parts.append(discarded[["symbol_norm", "timestamp_ns_utc"]])
        per_symbol[symbol] = len(discarded)
        bars = bars.loc[~all_null].copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
        values = bars[["open", "high", "low", "close"]].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0.0).any() or bars["timestamp"].duplicated().any():
            raise AssertionError("invalid provider OHLC")
        if (
            (bars["high"] < bars[["open", "close"]].max(axis=1)).any()
            or (bars["low"] > bars[["open", "close"]].min(axis=1)).any()
            or (bars["high"] < bars["low"]).any()
        ):
            raise AssertionError("inconsistent provider OHLC")
        local = bars["timestamp"].dt.tz_convert("America/New_York")
        bars["symbol_norm"] = symbol
        bars["session_date"] = local.dt.strftime("%Y-%m-%d")
        bars["bar_ordinal"] = bars.groupby("session_date", sort=False).cumcount()
        grouped = bars.groupby("session_date", sort=False)
        previous = grouped["close"].shift(1)
        first = bars["bar_ordinal"].eq(0)
        bars["current_bar_log_return"] = np.log(
            bars["close"] / previous.where(~first, bars["open"])
        )
        bars["return_sum_6"] = grouped["current_bar_log_return"].transform(
            lambda series: series.rolling(6, min_periods=1).sum()
        )
        bars["mean_abs_return_12"] = grouped["current_bar_log_return"].transform(
            lambda series: series.abs().rolling(12, min_periods=1).mean()
        )
        bars["session_return"] = np.log(bars["close"] / grouped["open"].transform("first"))
        bars["bar_range_pct"] = (bars["high"] - bars["low"]) / bars["open"]
        if not np.isfinite(bars[list(NEW5)].to_numpy(float)).all():
            raise AssertionError("nonfinite causal provider factor")
        bars["timestamp_ns_utc"] = bars["timestamp"].astype("int64")
        panels.append(
            bars[
                [
                    "symbol_norm", "timestamp", "timestamp_ns_utc", "session_date",
                    "bar_ordinal", *NEW5,
                ]
            ]
        )
    panel = pd.concat(panels, ignore_index=True).sort_values(
        ["symbol_norm", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    discarded = pd.concat(discarded_parts, ignore_index=True).sort_values(
        ["symbol_norm", "timestamp_ns_utc"], kind="stable"
    ).reset_index(drop=True)
    factor_hash = canonical_factor_hash(panel)
    expected_cleanup = contract["feature_construction"]["provider_scan_and_factor_table"]["fit_2024_placeholder_audit"]
    cleanup = {
        "discarded_rows": len(discarded),
        "all_four_OHLC_null_rows": len(discarded),
        "partial_null_rows": 0,
        "canonical_discarded_key_sha256": canonical_discarded_hash(discarded),
        "per_symbol": per_symbol,
    }
    for key in cleanup:
        if cleanup[key] != expected_cleanup[key]:
            raise AssertionError(f"provider placeholder audit changed: {key}")
    expected_factor_hash = contract["feature_construction"]["provider_scan_and_factor_table"]["fit_2024_canonical_retained_factor_table_sha256"]
    if factor_hash != expected_factor_hash:
        raise AssertionError("canonical retained 2024 factor table changed")
    return panel, factor_hash, {
        "rows": len(panel),
        "canonical_table_sha256": factor_hash,
        "provider_file_hashes": expected_provenance,
        "placeholder_cleanup": cleanup,
        "historical_volume_used": False,
        "volume_label": "historical_volume_not_used",
    }


def merge_entry_factors(runs: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    right = factors.rename(columns={"timestamp": "start_timestamp"})
    if right.duplicated(list(NATURAL_KEY)).any() or runs.duplicated(list(NATURAL_KEY)).any():
        raise AssertionError("duplicate factor merge key")
    merged = runs.merge(
        right.drop(columns="timestamp_ns_utc"),
        on=list(NATURAL_KEY), how="left", sort=False, validate="one_to_one", indicator=True,
    )
    if len(merged) != len(runs) or not merged["_merge"].eq("both").all():
        raise AssertionError("run/factor merge is not exact one-to-one")
    merged = merged.drop(columns="_merge")
    if not np.isfinite(merged[list(NEW5)].to_numpy(float)).all():
        raise AssertionError("nonfinite price factor at run entry")
    timestamp = pd.to_datetime(merged["start_timestamp"], utc=True)
    local = timestamp.dt.tz_convert("America/New_York")
    seconds = (
        local.dt.hour.to_numpy(float) * 3600.0
        + local.dt.minute.to_numpy(float) * 60.0
        + local.dt.second.to_numpy(float)
        + local.dt.microsecond.to_numpy(float) / 1_000_000.0
        - 570.0 * 60.0
    )
    minutes = seconds / 60.0
    if minutes.min(initial=0.0) < 0.0 or minutes.max(initial=0.0) >= 390.0:
        raise AssertionError("run entry outside regular session")
    phase = 2.0 * np.pi * minutes / 390.0
    merged["entry_minutes"] = minutes
    merged["entry_time_sin"] = np.sin(phase)
    merged["entry_time_cos"] = np.cos(phase)
    raw_b0 = pd.to_numeric(merged["b0_state_numeric"], errors="coerce")
    merged["b0_unknown"] = raw_b0.isna()
    merged["b0_entry_numeric"] = raw_b0.fillna(0.0)
    merged["b0_entry_high_stress"] = pd.to_numeric(
        merged["b0_high_stress"], errors="coerce"
    ).fillna(0.0)
    if not np.isfinite(merged[list(FULL9)].to_numpy(float)).all():
        raise AssertionError("nonfinite entry factor")
    return merged


def expand_labels(anchors: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    routes = build_routes(cycles)
    compatible = {
        state: sum(state in set(cycle.core) for cycle in cycles.itertuples(index=False))
        for state in range(STATE_COUNT)
    }
    anchor_counts = anchors["state"].map(compatible).astype(int)
    meta = [
        "anchor_id", "period", "symbol_norm", "session_date", "start_timestamp",
        "month", "quarter", "state", "history_token", "next_outcome", "terminal",
        "bar_ordinal", "b0_unknown", "entry_minutes", "future_state_1",
        "future_state_2", "future_state_3", "future_state_4", *FULL9,
    ]
    parts: list[pd.DataFrame] = []
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(value) for value in cycle.core)
        selected = anchors.loc[anchors["state"].isin(set(core)), meta].copy()
        target = np.zeros(len(selected), dtype=bool)
        for current in sorted(set(core)):
            mask = selected["state"].eq(current).to_numpy()
            local = np.zeros(int(mask.sum()), dtype=bool)
            subset = selected.loc[mask]
            for path in compatible_rotations(core, current):
                match = np.ones(len(subset), dtype=bool)
                for step, destination in enumerate(path[1:], start=1):
                    match &= subset[f"future_state_{step}"].to_numpy(int) == destination
                local |= match
            target[mask] = local
        selected["cycle_index"] = int(cycle.cycle_index)
        selected["cycle_id"] = str(cycle.cycle_id)
        selected["cycle"] = str(cycle.cycle)
        selected["transition_length"] = int(cycle.transition_length)
        selected["current_state"] = selected["state"].astype(int)
        selected["target"] = target.astype(np.int8)
        selected["compatible_cycle_count"] = selected["anchor_id"].map(anchor_counts)
        selected["inverse_compatible_weight"] = 1.0 / selected["compatible_cycle_count"]
        parts.append(selected)
    expanded = pd.concat(parts, ignore_index=True).sort_values(
        ["anchor_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    expanded = expanded.merge(
        routes[["route_index", "cycle_index", "current_state"]],
        on=["cycle_index", "current_state"], how="left", sort=False, validate="many_to_one",
    )
    expanded["route_index"] = expanded["route_index"].astype(int)
    if expanded.duplicated(["anchor_id", "cycle_index"]).any():
        raise AssertionError("duplicate expanded label row")
    if expanded.loc[expanded["terminal"].astype(bool), "target"].sum() != 0:
        raise AssertionError("terminal anchor has a positive loop label")
    return expanded


def reconstruct_2024(contract: Mapping[str, Any]) -> Prepared2024:
    cycles = load_cycles(contract)
    routes = build_routes(cycles)
    runs = load_runs_2024(contract)
    factors, factor_hash, provider_audit = reconstruct_provider_2024(
        sorted(runs["symbol_norm"].unique()), contract
    )
    anchors = merge_entry_factors(runs, factors)
    expanded = expand_labels(anchors, cycles)
    expected_rows = int(contract["population_and_target"]["compatible_anchor_cycle_rows_expected"]["2024"])
    expected_positive = int(contract["population_and_target"]["positive_rows_expected"]["2024"])
    if len(expanded) != expected_rows or int(expanded["target"].sum()) != expected_positive:
        raise AssertionError("expanded 2024 population changed")
    return Prepared2024(
        anchors=anchors,
        expanded=expanded,
        cycles=cycles,
        routes=routes,
        factor_hash=factor_hash,
        source_audit={
            "period": "2024",
            "year": 2024,
            "run_rows": len(runs),
            "compatible_rows": len(expanded),
            "positive_rows": int(expanded["target"].sum()),
            "terminal_run_entries": int(anchors["terminal"].sum()),
            "terminal_rows_in_fit_population": int(expanded["terminal"].sum()),
            "b0_unknown_run_entries": int(anchors["b0_unknown"].sum()),
            "factor_table": provider_audit,
            "later_period_paths_resolved": False,
            "later_period_rows_read": False,
            "shadow_tree_read": False,
            "shadow_tree_written": False,
            **SAFETY,
        },
    )


def subset_population(
    anchors: pd.DataFrame, expanded: pd.DataFrame, mask: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(anchors),):
        raise AssertionError("fold mask has wrong shape")
    old_ids = anchors.loc[selected, "anchor_id"].to_numpy(int)
    source_ids = (
        anchors["source_anchor_id"].to_numpy(int)
        if "source_anchor_id" in anchors
        else anchors["anchor_id"].to_numpy(int)
    )
    remap = np.full(len(anchors), -1, dtype=np.int64)
    remap[old_ids] = np.arange(len(old_ids), dtype=np.int64)
    left = anchors.loc[selected].copy().reset_index(drop=True)
    left["source_anchor_id"] = source_ids[old_ids]
    left["anchor_id"] = np.arange(len(left), dtype=np.int64)
    right = expanded.loc[expanded["anchor_id"].isin(set(old_ids))].copy()
    right["source_anchor_id"] = source_ids[right["anchor_id"].to_numpy(int)]
    right["anchor_id"] = remap[right["anchor_id"].to_numpy(int)]
    right = right.sort_values(["anchor_id", "cycle_index"], kind="stable").reset_index(drop=True)
    if (right["anchor_id"] < 0).any():
        raise AssertionError("fold anchor remap failed")
    return left, right


def token_matrix(tokens: Any) -> sparse.csr_matrix:
    values = np.asarray(tokens, dtype=int)
    return sparse.csr_matrix(
        (
            np.ones(len(values), dtype=np.float32),
            (np.arange(len(values)), values),
        ),
        shape=(len(values), TOKEN_WIDTH),
        dtype=np.float32,
    )


def raw_limited4(anchors: pd.DataFrame) -> np.ndarray:
    frame = anchors.loc[:, list(LIMITED4)].apply(pd.to_numeric, errors="coerce")
    frame["b0_entry_numeric"] = frame["b0_entry_numeric"].fillna(0.0)
    frame["b0_entry_high_stress"] = frame["b0_entry_high_stress"].fillna(0.0)
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise AssertionError("nonfinite limited transition input")
    return values


def fit_kernel(anchors: pd.DataFrame, *, with_numeric: bool) -> Kernel:
    matrix: sparse.csr_matrix = token_matrix(anchors["history_token"])
    numeric_width = 0
    if with_numeric:
        numeric = raw_limited4(anchors)
        matrix = sparse.hstack(
            (matrix, sparse.csr_matrix(numeric)), format="csr", dtype=np.float32
        )
        numeric_width = numeric.shape[1]
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=500,
        tol=0.0001,
        random_state=20260710,
    )
    model.fit(matrix, anchors["next_outcome"].to_numpy(int))
    if not np.array_equal(model.classes_, np.arange(DESTINATION_COUNT)):
        raise AssertionError("transition destination class missing")
    iterations = int(model.n_iter_[0])
    if iterations >= 500:
        raise RuntimeError("transition kernel failed convergence")
    return Kernel(
        classes=model.classes_.astype(np.int64, copy=True),
        coefficients=model.coef_.astype(np.float64, copy=True),
        intercepts=model.intercept_.astype(np.float64, copy=True),
        numeric_width=numeric_width,
        iterations=iterations,
    )


def load_pinned_kernels(contract: Mapping[str, Any]) -> tuple[Kernel, Kernel]:
    spec = contract["frozen_sources"]["retained_path_parameters"]
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise AssertionError("retained path parameters changed")
    with np.load(path, allow_pickle=False) as stored:
        history = Kernel(
            stored["history_classes"].copy(),
            stored["history_coef"].astype(np.float64, copy=True),
            stored["history_intercept"].astype(np.float64, copy=True),
            0,
            int(stored["history_n_iter"][0]),
        )
        limited = Kernel(
            stored["context_classes"].copy(),
            stored["context_coef"].astype(np.float64, copy=True),
            stored["context_intercept"].astype(np.float64, copy=True),
            4,
            int(stored["context_n_iter"][0]),
        )
    expected = ((history, TOKEN_WIDTH), (limited, TOKEN_WIDTH + 4))
    for kernel, width in expected:
        if (
            not np.array_equal(kernel.classes, np.arange(DESTINATION_COUNT))
            or kernel.coefficients.shape != (DESTINATION_COUNT, width)
            or kernel.intercepts.shape != (DESTINATION_COUNT,)
        ):
            raise AssertionError("retained transition kernel shape changed")
    return history, limited


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def destination_probability(
    kernel: Kernel,
    p2: np.ndarray,
    p1: np.ndarray,
    current: np.ndarray,
    destination: int,
    numeric: np.ndarray | None,
) -> np.ndarray:
    tokens = history_token(p2, p1, current)
    logits = kernel.intercepts[None, :] + kernel.coefficients[:, tokens].T
    if kernel.numeric_width:
        if numeric is None or numeric.shape != (len(tokens), kernel.numeric_width):
            raise AssertionError("limited transition numeric shape changed")
        logits = logits + numeric.astype(np.float64) @ kernel.coefficients[:, TOKEN_WIDTH:].T
    elif numeric is not None:
        raise AssertionError("history kernel received numeric factors")
    index = int(np.flatnonzero(kernel.classes == destination)[0])
    return np.clip(softmax(logits)[:, index], EPSILON, 1.0 - EPSILON)


def route_probability(
    anchors: pd.DataFrame,
    path: tuple[int, ...],
    kernel: Kernel,
    *,
    repeat_limited: bool,
) -> np.ndarray:
    probability = np.ones(len(anchors), dtype=np.float64)
    p2 = anchors["previous_state_2"].to_numpy(int)
    p1 = anchors["previous_state_1"].to_numpy(int)
    current = np.full(len(anchors), path[0], dtype=int)
    numeric = raw_limited4(anchors) if repeat_limited else None
    for destination in path[1:]:
        probability *= destination_probability(
            kernel, p2, p1, current, int(destination), numeric
        )
        p2, p1, current = p1, current, np.full(len(anchors), destination, dtype=int)
    return probability


def path_offsets(
    anchors: pd.DataFrame,
    expanded: pd.DataFrame,
    cycles: pd.DataFrame,
    history: Kernel,
    limited: Kernel,
) -> pd.DataFrame:
    if not np.array_equal(anchors["anchor_id"].to_numpy(int), np.arange(len(anchors))):
        raise AssertionError("local anchor ids must be contiguous")
    qhistory = np.zeros(len(expanded), dtype=np.float64)
    qold = np.zeros(len(expanded), dtype=np.float64)
    cycle_values = expanded["cycle_index"].to_numpy(int)
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(value) for value in cycle.core)
        positions = np.flatnonzero(cycle_values == int(cycle.cycle_index))
        states = expanded.iloc[positions]["current_state"].to_numpy(int)
        for current in sorted(set(core)):
            local = positions[states == current]
            selected = anchors.iloc[expanded.iloc[local]["anchor_id"].to_numpy(int)]
            history_sum = np.zeros(len(selected), dtype=float)
            limited_sum = np.zeros(len(selected), dtype=float)
            for path in compatible_rotations(core, current):
                history_sum += route_probability(selected, path, history, repeat_limited=False)
                limited_sum += route_probability(selected, path, limited, repeat_limited=True)
            qhistory[local] = history_sum
            qold[local] = limited_sum
    qhistory = np.clip(qhistory, EPSILON, 1.0 - EPSILON)
    qold = np.clip(qold, EPSILON, 1.0 - EPSILON)
    return pd.DataFrame(
        {
            "qhistory": qhistory,
            "qold_limited_path": qold,
            "eta_history": np.log(qhistory) - np.log1p(-qhistory),
            "eta_old_limited_path": np.log(qold) - np.log1p(-qold),
        },
        index=expanded.index,
    )


def numeric_frame(anchors: pd.DataFrame) -> pd.DataFrame:
    frame = anchors.loc[:, list(FULL9)].apply(pd.to_numeric, errors="coerce")
    frame["b0_entry_numeric"] = frame["b0_entry_numeric"].fillna(0.0)
    frame["b0_entry_high_stress"] = frame["b0_entry_high_stress"].fillna(0.0)
    if not np.isfinite(frame[list(NEW5)].to_numpy(float)).all():
        raise AssertionError("price factor nonfinite before imputation")
    return frame


def fit_transform(anchors: pd.DataFrame) -> Transform:
    values = numeric_frame(anchors).to_numpy(np.float64)
    medians = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, medians[None, :])
    scales = np.sqrt(np.mean(np.square(filled - medians[None, :]), axis=0))
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
    if not np.isfinite(medians).all():
        raise AssertionError("numeric median nonfinite")
    return Transform(medians=medians, scales=scales)


def apply_transform(anchors: pd.DataFrame, transform: Transform) -> np.ndarray:
    values = numeric_frame(anchors).to_numpy(np.float64)
    filled = np.where(np.isfinite(values), values, transform.medians[None, :])
    output = (filled - transform.medians[None, :]) / transform.scales[None, :]
    if output.shape != (len(anchors), 9) or not np.isfinite(output).all():
        raise AssertionError("invalid transformed factors")
    return output


def fit_contrasts(expanded: pd.DataFrame, routes: pd.DataFrame) -> Contrasts:
    weights = expanded["inverse_compatible_weight"].to_numpy(np.float64)
    cycle_weights = np.bincount(
        expanded["cycle_index"].to_numpy(int), weights=weights, minlength=CYCLE_COUNT
    )
    route_weights = np.bincount(
        expanded["route_index"].to_numpy(int), weights=weights, minlength=ROUTE_COUNT
    )
    if (cycle_weights <= 0).any() or (route_weights <= 0).any():
        raise AssertionError("contrast level lacks prefix support")
    references = routes.loc[routes["is_reference"]].sort_values(
        "cycle_index", kind="stable"
    )["route_index"].to_numpy(int)
    return Contrasts(cycle_weights, route_weights, references)


def contrast_blocks(
    expanded: pd.DataFrame, contrasts: Contrasts, routes: pd.DataFrame
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rows = len(expanded)
    cycles = expanded["cycle_index"].to_numpy(int)
    route_ids = expanded["route_index"].to_numpy(int)
    cycle_block = sparse.lil_matrix((rows, CYCLE_CONTRAST_WIDTH), dtype=np.float64)
    reference = CYCLE_COUNT - 1
    ref_rows = np.flatnonzero(cycles == reference)
    for level in range(CYCLE_CONTRAST_WIDTH):
        cycle_block[np.flatnonzero(cycles == level), level] = 1.0
        cycle_block[ref_rows, level] = -contrasts.cycle_weights[level] / contrasts.cycle_weights[reference]
    route_block = sparse.lil_matrix((rows, ROUTE_CONTRAST_WIDTH), dtype=np.float64)
    nonreference = routes.loc[~routes["is_reference"]].sort_values("route_index", kind="stable")
    for column, mapping in enumerate(nonreference.itertuples(index=False)):
        direct = np.flatnonzero(route_ids == int(mapping.route_index))
        reference_route = int(contrasts.route_references[int(mapping.cycle_index)])
        ref = np.flatnonzero(route_ids == reference_route)
        route_block[direct, column] = 1.0
        route_block[ref, column] = -contrasts.route_weights[int(mapping.route_index)] / contrasts.route_weights[reference_route]
    return cycle_block.tocsr(), route_block.tocsr()


def token_support(anchors: pd.DataFrame, expanded: pd.DataFrame) -> np.ndarray:
    anchor_count = np.bincount(anchors["history_token"].to_numpy(int), minlength=TOKEN_WIDTH)
    positive_count = np.bincount(
        expanded["history_token"].to_numpy(int),
        weights=expanded["target"].to_numpy(float),
        minlength=TOKEN_WIDTH,
    )
    return (anchor_count >= 200) & (positive_count >= 50.0)


def build_design(
    expanded: pd.DataFrame,
    anchor_z: np.ndarray,
    factor_count: int,
    contrasts: Contrasts,
    routes: pd.DataFrame,
    token_mask: np.ndarray,
) -> sparse.csr_matrix:
    row_anchor = expanded["anchor_id"].to_numpy(int)
    row_factors = np.asarray(anchor_z, float)[row_anchor, :factor_count]
    cycle, route = contrast_blocks(expanded, contrasts, routes)
    structural = sparse.hstack(
        (sparse.csr_matrix(np.ones((len(expanded), 1))), cycle, route),
        format="csr", dtype=np.float64,
    )
    if factor_count == 0:
        return structural
    blocks: list[sparse.spmatrix] = [structural, sparse.csr_matrix(row_factors)]
    blocks.extend(cycle.multiply(row_factors[:, factor][:, None]) for factor in range(factor_count))
    blocks.extend(route.multiply(row_factors[:, factor][:, None]) for factor in range(factor_count))
    tokens = expanded["history_token"].to_numpy(int)
    for factor in range(factor_count):
        values = row_factors[:, factor]
        supported = token_mask[tokens]
        selected = np.flatnonzero(supported & (values != 0.0))
        blocks.append(
            sparse.csr_matrix(
                (values[selected], (selected, tokens[selected])),
                shape=(len(expanded), TOKEN_WIDTH), dtype=np.float64,
            )
        )
    design = sparse.hstack(blocks, format="csr", dtype=np.float64)
    expected = LIMITED_WIDTH if factor_count == 4 else FULL_WIDTH
    if design.shape != (len(expanded), expected) or not np.isfinite(design.data).all():
        raise AssertionError("residual design width or values changed")
    return design


def build_designs(
    expanded: pd.DataFrame,
    anchor_z: np.ndarray,
    contrasts: Contrasts,
    routes: pd.DataFrame,
    token_mask: np.ndarray,
) -> dict[str, sparse.csr_matrix]:
    return {
        head: build_design(
            expanded, anchor_z, FACTOR_COUNT[head], contrasts, routes, token_mask
        )
        for head in HEADS
    }


def penalties(factor_count: int) -> np.ndarray:
    values = [0.0, *([4.0] * 19), *([8.0] * 24)]
    if factor_count:
        values.extend([1.0] * factor_count)
        values.extend([4.0] * (factor_count * 19))
        values.extend([8.0] * (factor_count * 24))
        values.extend([32.0] * (factor_count * TOKEN_WIDTH))
    result = np.asarray(values, dtype=np.float64)
    expected = {0: PATTERN_WIDTH, 4: LIMITED_WIDTH, 9: FULL_WIDTH}[factor_count]
    if len(result) != expected:
        raise AssertionError("penalty width changed")
    return result


def objective_gradient(
    coefficients: np.ndarray,
    design: sparse.csr_matrix,
    target: np.ndarray,
    offset: np.ndarray,
    ridge_lambda: float,
    penalty: np.ndarray,
) -> tuple[float, np.ndarray]:
    beta = np.asarray(coefficients, np.float64)
    matrix = sparse.csr_matrix(design, dtype=np.float64)
    y = np.asarray(target, np.float64)
    eta = np.asarray(offset, np.float64) + matrix @ beta
    loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    loss += 0.5 * ridge_lambda * float(np.dot(penalty, np.square(beta)))
    gradient = np.asarray(matrix.T @ (expit(eta) - y)).ravel() / len(y)
    gradient += ridge_lambda * penalty * beta
    return loss, gradient


def fit_ridge(
    design: sparse.csr_matrix,
    target: np.ndarray,
    offset: np.ndarray,
    ridge_lambda: float,
    factor_count: int,
) -> RidgeFit:
    matrix = sparse.csr_matrix(design, dtype=np.float64)
    penalty = penalties(factor_count)

    def wrapped(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return objective_gradient(beta, matrix, target, offset, ridge_lambda, penalty)

    result = minimize(
        wrapped,
        np.zeros(matrix.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    objective, gradient = wrapped(np.asarray(result.x, float))
    if (
        not bool(result.success)
        or int(result.nit) >= 1000
        or not math.isfinite(objective)
        or not np.isfinite(gradient).all()
        or not np.isfinite(result.x).all()
    ):
        raise RuntimeError(f"independent ridge fit failed: {result.message}")
    return RidgeFit(
        coefficients=np.asarray(result.x, np.float64),
        ridge_lambda=float(ridge_lambda),
        factor_count=factor_count,
        objective=objective,
        gradient_max_abs=float(np.max(np.abs(gradient))),
        iterations=int(result.nit),
        message=str(result.message),
    )


def predict_ridge(
    design: sparse.csr_matrix, offset: np.ndarray, fit: RidgeFit
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.asarray(offset, float) + sparse.csr_matrix(design) @ fit.coefficients
    return np.clip(expit(eta), EPSILON, 1.0 - EPSILON), np.asarray(eta, float)


def fit_quartiles(anchors: pd.DataFrame) -> np.ndarray:
    values = anchors.loc[:, list(QUARTILE_COLUMNS)].to_numpy(np.float64)
    return np.quantile(values, [0.25, 0.5, 0.75], axis=0, method="linear").T


def apply_quartiles(anchors: pd.DataFrame, cutpoints: np.ndarray) -> dict[str, np.ndarray]:
    return {
        column: np.searchsorted(
            cutpoints[index], anchors[column].to_numpy(float), side="right"
        ).astype(np.int8)
        for index, column in enumerate(QUARTILE_COLUMNS)
    }


@dataclass
class FoldData:
    train_anchors: pd.DataFrame
    train_expanded: pd.DataFrame
    validation_anchors: pd.DataFrame
    validation_expanded: pd.DataFrame
    history: Kernel
    old_limited: Kernel
    train_offsets: pd.DataFrame
    validation_offsets: pd.DataFrame
    transform: Transform
    contrasts: Contrasts
    token_mask: np.ndarray
    quartile_cutpoints: np.ndarray
    train_designs: dict[str, sparse.csr_matrix]
    validation_designs: dict[str, sparse.csr_matrix]


def build_fold(prepared: Prepared2024, validation_month: str) -> FoldData:
    train_mask = prepared.anchors["month"].astype(str).lt(validation_month).to_numpy()
    validation_mask = prepared.anchors["month"].astype(str).eq(validation_month).to_numpy()
    train_anchors, train_expanded = subset_population(
        prepared.anchors, prepared.expanded, train_mask
    )
    validation_anchors, validation_expanded = subset_population(
        prepared.anchors, prepared.expanded, validation_mask
    )
    if train_anchors.empty or validation_anchors.empty:
        raise AssertionError("causal fold has empty side")
    history = fit_kernel(train_anchors, with_numeric=False)
    old_limited = fit_kernel(train_anchors, with_numeric=True)
    train_offsets = path_offsets(
        train_anchors, train_expanded, prepared.cycles, history, old_limited
    )
    validation_offsets = path_offsets(
        validation_anchors, validation_expanded, prepared.cycles, history, old_limited
    )
    transform = fit_transform(train_anchors)
    contrasts = fit_contrasts(train_expanded, prepared.routes)
    mask = token_support(train_anchors, train_expanded)
    train_designs = build_designs(
        train_expanded,
        apply_transform(train_anchors, transform),
        contrasts,
        prepared.routes,
        mask,
    )
    validation_designs = build_designs(
        validation_expanded,
        apply_transform(validation_anchors, transform),
        contrasts,
        prepared.routes,
        mask,
    )
    return FoldData(
        train_anchors,
        train_expanded,
        validation_anchors,
        validation_expanded,
        history,
        old_limited,
        train_offsets,
        validation_offsets,
        transform,
        contrasts,
        mask,
        fit_quartiles(train_anchors),
        train_designs,
        validation_designs,
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def replay_prefix(payload: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    marker = prefix + "__"
    selected = {
        key[len(marker) :]: value
        for key, value in payload.items()
        if key.startswith(marker)
    }
    if not selected:
        raise AssertionError(f"missing replay prefix: {prefix}")
    return selected


def csr_from_replay(payload: Mapping[str, np.ndarray], head: str) -> sparse.csr_matrix:
    shape = tuple(int(value) for value in payload[f"{head}__shape"])
    return sparse.csr_matrix(
        (
            np.asarray(payload[f"{head}__data"], np.float64),
            np.asarray(payload[f"{head}__indices"]),
            np.asarray(payload[f"{head}__indptr"]),
        ),
        shape=shape,
        dtype=np.float64,
    )


def compare_replay(
    stored: Mapping[str, np.ndarray],
    fold: FoldData,
    *,
    tolerance: float = 1e-12,
) -> tuple[bool, dict[str, Any]]:
    expected_vectors = {
        "target": fold.train_expanded["target"].to_numpy(np.int8),
        "offset": fold.train_offsets["eta_history"].to_numpy(np.float64),
        "anchor_id": fold.train_expanded["anchor_id"].to_numpy(np.int64),
        "cycle_index": fold.train_expanded["cycle_index"].to_numpy(np.int16),
    }
    details: dict[str, Any] = {}
    passed = True
    for name, expected in expected_vectors.items():
        observed = np.asarray(stored[name])
        exact = np.array_equal(observed, expected) if name != "offset" else False
        error = 0.0
        if name == "offset" and observed.shape == expected.shape:
            error = float(np.max(np.abs(observed.astype(float) - expected), initial=0.0))
            exact = error <= tolerance
        details[name] = {"shape": list(observed.shape), "error": error, "pass": exact}
        passed &= exact
    for head in HEADS:
        observed = csr_from_replay(stored, head)
        expected = sparse.csr_matrix(fold.train_designs[head], dtype=np.float64)
        structure = (
            observed.shape == expected.shape
            and np.array_equal(observed.indptr, expected.indptr)
            and np.array_equal(observed.indices, expected.indices)
        )
        error = math.inf
        if structure:
            error = float(np.max(np.abs(observed.data - expected.data), initial=0.0))
        ok = structure and error <= tolerance
        details[head] = {
            "shape": list(observed.shape),
            "structure": structure,
            "maximum_data_error": error,
            "pass": ok,
        }
        passed &= ok
    return passed, details


def binary_log_loss(target: Any, probability: Any) -> np.ndarray:
    y = np.asarray(target, np.float64)
    p = np.clip(np.asarray(probability, np.float64), EPSILON, 1.0 - EPSILON)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def select_lambdas(
    scores: pd.DataFrame, months: Sequence[str]
) -> tuple[dict[str, float], pd.DataFrame]:
    selected: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for head in HEADS:
        objectives: dict[float, float] = {}
        for ridge_lambda in RIDGE_GRID:
            subset = scores.loc[
                scores["head"].eq(head)
                & np.isclose(scores["lambda"], ridge_lambda)
                & scores["validation_month"].astype(str).isin(months)
            ]
            if len(subset) != len(months) or set(subset["validation_month"].astype(str)) != set(months):
                raise AssertionError("lambda selection lacks a validation month")
            objectives[ridge_lambda] = float(
                subset.set_index("validation_month").loc[list(months), "log_loss"].mean()
            )
        minimum = min(objectives.values())
        ties = [
            value for value in RIDGE_GRID
            if objectives[value] <= minimum + 1e-12
        ]
        choice = max(ties)
        selected[head] = choice
        for ridge_lambda in RIDGE_GRID:
            rows.append(
                {
                    "head": head,
                    "lambda": ridge_lambda,
                    "selection_months": json.dumps(list(months), separators=(",", ":")),
                    "equal_month_mean_log_loss": objectives[ridge_lambda],
                    "selected": ridge_lambda == choice,
                }
            )
    return selected, pd.DataFrame(rows)


def quartile_columns(
    anchors: pd.DataFrame, expanded: pd.DataFrame, cutpoints: np.ndarray
) -> dict[str, np.ndarray]:
    values = apply_quartiles(anchors, cutpoints)
    row_anchor = expanded["anchor_id"].to_numpy(int)
    output = {"entry_clock_quartile": values["entry_minutes"][row_anchor]}
    for factor in NEW5:
        output[f"factor_quartile__{factor}"] = values[factor][row_anchor]
    return output


def predict_fold(
    fold: FoldData,
    fits: Mapping[str, RidgeFit],
) -> pd.DataFrame:
    output = fold.validation_expanded.copy()
    output["n_compatible"] = output["compatible_cycle_count"].astype(int)
    for name, values in quartile_columns(
        fold.validation_anchors,
        fold.validation_expanded,
        fold.quartile_cutpoints,
    ).items():
        output[name] = values
    for column in fold.validation_offsets:
        output[column] = fold.validation_offsets[column].to_numpy()
    offset = fold.validation_offsets["eta_history"].to_numpy(float)
    for head in HEADS:
        probability, eta = predict_ridge(fold.validation_designs[head], offset, fits[head])
        output[head] = probability
        output[f"eta_{head}"] = eta
    output["local_anchor_id"] = output["anchor_id"].astype(int)
    output["anchor_id"] = output["source_anchor_id"].astype(int)
    return output


def frame_difference(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: Sequence[str],
    *,
    tolerance: float = 1e-10,
) -> tuple[bool, dict[str, Any]]:
    left = observed.sort_values(list(keys), kind="stable").reset_index(drop=True)
    right = expected.sort_values(list(keys), kind="stable").reset_index(drop=True)
    if len(left) != len(right) or set(left.columns) != set(right.columns):
        return False, {
            "rows": [len(left), len(right)],
            "observed_only": sorted(set(left.columns) - set(right.columns)),
            "expected_only": sorted(set(right.columns) - set(left.columns)),
        }
    right = right.loc[:, left.columns]
    details: dict[str, Any] = {}
    passed = True
    for column in left.columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(float)
            bv = b.to_numpy(float)
            finite = np.isfinite(av) == np.isfinite(bv)
            error = math.inf
            if finite.all() and np.array_equal(np.isnan(av), np.isnan(bv)):
                both = np.isfinite(av) & np.isfinite(bv)
                error = float(np.max(np.abs(av[both] - bv[both]), initial=0.0))
            ok = error <= tolerance
            if not ok:
                details[column] = error
            passed &= ok
        else:
            ok = np.array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy())
            if not ok:
                details[column] = "categorical_mismatch"
            passed &= ok
    return passed, {"maximum_errors": details, "rows": len(left)}


def nested_differences(
    observed: Any,
    expected: Any,
    *,
    path: str = "root",
    tolerance: float = 1e-10,
) -> list[str]:
    left = json_safe(observed)
    right = json_safe(expected)
    differences: list[str] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            differences.append(
                f"{path}:keys:{sorted(set(left)-set(right))}:{sorted(set(right)-set(left))}"
            )
            return differences
        for key in sorted(left):
            differences.extend(
                nested_differences(left[key], right[key], path=f"{path}.{key}", tolerance=tolerance)
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}:length:{len(left)}:{len(right)}"]
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(
                nested_differences(a, b, path=f"{path}[{index}]", tolerance=tolerance)
            )
        return differences
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if type(left) is bool or type(right) is bool:
            if type(left) is not type(right) or left is not right:
                differences.append(f"{path}:{left!r}:{right!r}")
        elif not math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=tolerance):
            differences.append(f"{path}:{left!r}:{right!r}")
    elif left != right:
        differences.append(f"{path}:{left!r}:{right!r}")
    return differences


def reconstruct_grid_and_oof(
    audit: Audit,
    root: Path,
    prepared: Prepared2024,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stored_scores = pd.read_csv(root / "lambda_grid_scores_2024.csv")
    grid_replay = load_npz(root / "grid_replay_inputs_2024.npz")
    outer_replay = load_npz(root / "outer_replay_inputs_2024.npz")
    stored_outer_selection = pd.read_csv(root / "outer_lambda_selection_2024.csv")
    stored_outer_audits = json.loads((root / "outer_fit_audit_2024.json").read_text())
    expected_score_rows: list[dict[str, Any]] = []
    expected_selection_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    expected_outer_audits: list[dict[str, Any]] = []
    replay_checks: dict[str, Any] = {}
    for month in GRID_MONTHS:
        fold = build_fold(prepared, month)
        prefix = month.replace("-", "_")
        stored_replay = replay_prefix(grid_replay, prefix)
        replay_pass, replay_detail = compare_replay(stored_replay, fold)
        replay_checks[f"grid_{month}"] = replay_detail
        audit.check(f"grid_{month}_serialized_design_independently_exact", replay_pass, replay_detail)
        target = fold.train_expanded["target"].to_numpy(int)
        validation_target = fold.validation_expanded["target"].to_numpy(int)
        validation_offset = fold.validation_offsets["eta_history"].to_numpy(float)
        fitted: dict[tuple[str, float], RidgeFit] = {}
        for head in HEADS:
            stored_design = csr_from_replay(stored_replay, head)
            for ridge_lambda in RIDGE_GRID:
                fit = fit_ridge(
                    stored_design,
                    np.asarray(stored_replay["target"], int),
                    np.asarray(stored_replay["offset"], float),
                    ridge_lambda,
                    FACTOR_COUNT[head],
                )
                probability, _ = predict_ridge(
                    fold.validation_designs[head], validation_offset, fit
                )
                expected_score_rows.append(
                    {
                        "validation_month": month,
                        "head": head,
                        "lambda": ridge_lambda,
                        "validation_rows": len(fold.validation_expanded),
                        "validation_positives": int(validation_target.sum()),
                        "log_loss": float(binary_log_loss(validation_target, probability).mean()),
                        "optimizer_objective": fit.objective,
                        "optimizer_gradient_max_abs": fit.gradient_max_abs,
                        "optimizer_iterations": fit.iterations,
                    }
                )
                fitted[(head, ridge_lambda)] = fit
        current_scores = pd.DataFrame(expected_score_rows)
        if month in OUTER_MONTHS:
            selected, selection = select_lambdas(current_scores, INNER_SCHEDULE[month])
            selection.insert(0, "outer_month", month)
            expected_selection_parts.append(selection)
            outer_payload = replay_prefix(outer_replay, prefix)
            outer_pass, outer_detail = compare_replay(outer_payload, fold)
            replay_checks[f"outer_{month}"] = outer_detail
            audit.check(f"outer_{month}_serialized_design_independently_exact", outer_pass, outer_detail)
            selected_fits = {
                head: fit_ridge(
                    csr_from_replay(outer_payload, head),
                    np.asarray(outer_payload["target"], int),
                    np.asarray(outer_payload["offset"], float),
                    selected[head],
                    FACTOR_COUNT[head],
                )
                for head in HEADS
            }
            outer_predictions = predict_fold(fold, selected_fits)
            outer_predictions["validation_month"] = month
            oof_parts.append(outer_predictions)
            fit_audit = {
                "anchors": len(fold.train_anchors),
                "compatible_rows": len(fold.train_expanded),
                "positives": int(target.sum()),
                "terminal_anchors_included": int(fold.train_anchors["terminal"].sum()),
                "selected_lambdas": selected,
                "history_iterations": fold.history.iterations,
                "old_limited_iterations": fold.old_limited.iterations,
                "supported_tokens": int(fold.token_mask.sum()),
                "numeric_medians": fold.transform.medians,
                "numeric_scales": fold.transform.scales,
                "embedding_invariants": embedding_errors(
                    fold.train_designs,
                    fold.train_offsets["eta_history"].to_numpy(float),
                    selected_fits["qpattern"].coefficients,
                    selected_fits["qlimited4"].coefficients,
                ),
                "optimizer": {
                    head: {
                        "objective": selected_fits[head].objective,
                        "gradient_max_abs": selected_fits[head].gradient_max_abs,
                        "iterations": selected_fits[head].iterations,
                        "message": selected_fits[head].message,
                    }
                    for head in HEADS
                },
                **SAFETY,
                "validation_month": month,
                "validation_anchors": len(fold.validation_anchors),
                "validation_rows": len(fold.validation_expanded),
                "session_overlap": False,
            }
            expected_outer_audits.append(
                {"outer_month": month, "selected_lambdas": selected, "audit": fit_audit}
            )
    expected_scores = pd.DataFrame(expected_score_rows)
    passed, details = frame_difference(
        stored_scores, expected_scores, ["validation_month", "head", "lambda"]
    )
    audit.check("all_108_lambda_grid_scores_independently_exact", passed, details)
    expected_selection = pd.concat(expected_selection_parts, ignore_index=True)
    passed, details = frame_difference(
        stored_outer_selection,
        expected_selection,
        ["outer_month", "head", "lambda"],
    )
    audit.check("all_outer_lambda_selections_independently_exact", passed, details)
    outer_differences = nested_differences(stored_outer_audits, expected_outer_audits)
    audit.check("all_outer_fit_audits_independently_exact", not outer_differences, outer_differences[:30])
    expected_oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["anchor_id", "cycle_id"], kind="stable"
    ).reset_index(drop=True)
    stored_oof = pd.read_parquet(root / "oof_predictions_2024.parquet")
    passed, details = frame_difference(
        stored_oof, expected_oof, ["anchor_id", "cycle_id"], tolerance=1e-10
    )
    audit.check("all_outer_oof_probabilities_and_metadata_independently_exact", passed, details)
    return expected_scores, expected_oof, replay_checks


def factor_layout(factor_count: int) -> dict[str, Any]:
    start = PATTERN_WIDTH
    global_slice = slice(start, start + factor_count)
    start = global_slice.stop
    cycle = [
        slice(start + factor * 19, start + (factor + 1) * 19)
        for factor in range(factor_count)
    ]
    start += factor_count * 19
    route = [
        slice(start + factor * 24, start + (factor + 1) * 24)
        for factor in range(factor_count)
    ]
    start += factor_count * 24
    token = [
        slice(start + factor * TOKEN_WIDTH, start + (factor + 1) * TOKEN_WIDTH)
        for factor in range(factor_count)
    ]
    start += factor_count * TOKEN_WIDTH
    return {
        "structural": slice(0, PATTERN_WIDTH),
        "global": global_slice,
        "cycle": cycle,
        "route": route,
        "token": token,
        "width": start,
    }


def embed(coefficients: np.ndarray, source_count: int, target_count: int) -> np.ndarray:
    source = factor_layout(source_count)
    target = factor_layout(target_count)
    output = np.zeros(target["width"], dtype=float)
    output[target["structural"]] = coefficients[source["structural"]]
    if source_count:
        output[target["global"].start : target["global"].start + source_count] = coefficients[source["global"]]
        for block in ("cycle", "route", "token"):
            for factor in range(source_count):
                output[target[block][factor]] = coefficients[source[block][factor]]
    return output


def embedding_errors(
    designs: Mapping[str, sparse.csr_matrix],
    offset: np.ndarray,
    pattern_coefficients: np.ndarray,
    limited_coefficients: np.ndarray,
) -> dict[str, float]:
    history = np.clip(expit(offset), EPSILON, 1.0 - EPSILON)
    zero = np.clip(expit(offset + designs["qfull9"] @ np.zeros(FULL_WIDTH)), EPSILON, 1.0 - EPSILON)
    pattern = offset + designs["qpattern"] @ pattern_coefficients
    limited = offset + designs["qlimited4"] @ limited_coefficients
    errors = {
        "zero_residual_to_history": float(np.max(np.abs(zero - history), initial=0.0)),
        "pattern_to_qlimited4": float(np.max(np.abs(offset + designs["qlimited4"] @ embed(pattern_coefficients, 0, 4) - pattern), initial=0.0)),
        "pattern_to_qfull9": float(np.max(np.abs(offset + designs["qfull9"] @ embed(pattern_coefficients, 0, 9) - pattern), initial=0.0)),
        "limited4_to_full9": float(np.max(np.abs(offset + designs["qfull9"] @ embed(limited_coefficients, 4, 9) - limited), initial=0.0)),
    }
    if max(errors.values()) > 1e-12:
        raise AssertionError("coefficient embedding failed")
    return errors


def inverse_weights(counts: Any) -> np.ndarray:
    values = np.asarray(counts, float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("invalid compatible counts")
    return 1.0 / values


def binary_losses(target: Any, probability: Any) -> dict[str, np.ndarray]:
    y = np.asarray(target, float)
    p = np.asarray(probability, float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid binary outcome/probability")
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)),
        "brier": np.square(p - y),
    }


def weighted_mean(values: Any, weights: Any) -> float:
    array = np.asarray(values, float)
    weight = np.asarray(weights, float)
    return float(np.dot(array, weight) / weight.sum())


def loss_metrics(
    target: Any, probability: Any, counts: Any | None = None
) -> dict[str, Any]:
    y = np.asarray(target, float)
    weights = np.ones(len(y)) if counts is None else inverse_weights(counts)
    losses = binary_losses(y, probability)
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "weight_sum": float(weights.sum()),
        "log_loss": weighted_mean(losses["log_loss"], weights),
        "brier": weighted_mean(losses["brier"], weights),
    }


def loss_comparison(
    target: Any,
    candidate: Any,
    baseline: Any,
    counts: Any | None = None,
) -> dict[str, float]:
    y = np.asarray(target, float)
    weights = np.ones(len(y)) if counts is None else inverse_weights(counts)
    candidate_loss = binary_losses(y, candidate)
    baseline_loss = binary_losses(y, baseline)
    output: dict[str, float] = {}
    for name in ("log_loss", "brier"):
        candidate_mean = weighted_mean(candidate_loss[name], weights)
        baseline_mean = weighted_mean(baseline_loss[name], weights)
        difference = candidate_mean - baseline_mean
        output[f"candidate_{name}"] = candidate_mean
        output[f"baseline_{name}"] = baseline_mean
        output[f"{name}_difference"] = difference
        if name == "log_loss":
            output["relative_log_loss_improvement"] = -difference / baseline_mean
    return output


def ranking_metrics(frame: pd.DataFrame, probability: str) -> dict[str, Any]:
    ranked = frame.loc[:, ["anchor_id", "cycle_id", "target", probability]].sort_values(
        ["anchor_id", probability, "cycle_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("anchor_id", sort=False).cumcount() + 1
    positive = ranked["target"].eq(1)
    top_three = ranked["rank"].le(3)
    top_one = ranked["rank"].eq(1)
    per_anchor = ranked.assign(
        top_three_hit=(positive & top_three).astype(np.int8),
        reciprocal=np.where(positive, 1.0 / ranked["rank"], 0.0),
    ).groupby("anchor_id", sort=False).agg(
        positive=("target", "max"),
        top_three_hit=("top_three_hit", "max"),
        reciprocal=("reciprocal", "max"),
    )
    positive_anchor = per_anchor["positive"].eq(1)
    positive_labels = int(positive.sum())
    hits = int((positive & top_three).sum())
    selected = int(top_three.sum())
    return {
        "anchors": int(ranked["anchor_id"].nunique()),
        "positive_labels": positive_labels,
        "selected_labels": selected,
        "hits": hits,
        "recall": float(hits / positive_labels),
        "precision": float(hits / selected),
        "positive_anchors": int(positive_anchor.sum()),
        "positive_anchor_hit_rate": float(per_anchor.loc[positive_anchor, "top_three_hit"].mean()),
        "top_one_recall": float((positive & top_one).sum() / positive_labels),
        "mean_reciprocal_rank": float(per_anchor.loc[positive_anchor, "reciprocal"].mean()),
    }


def calibration(frame: pd.DataFrame, probability: str) -> dict[str, Any]:
    y = frame["target"].to_numpy(float)
    p = frame[probability].to_numpy(float)
    index = np.minimum(np.floor(10.0 * p).astype(int), 9)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    supported: list[float] = []
    for bin_index in range(10):
        mask = index == bin_index
        count = int(mask.sum())
        mean_probability = float(p[mask].mean()) if count else math.nan
        event_rate = float(y[mask].mean()) if count else math.nan
        error = abs(mean_probability - event_rate) if count else math.nan
        is_supported = count >= 500
        if count:
            ece += count / len(y) * error
        if is_supported:
            supported.append(error)
        rows.append(
            {
                "bin": bin_index,
                "lower": bin_index / 10.0,
                "upper": (bin_index + 1) / 10.0,
                "rows": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_error": error,
                "supported": is_supported,
            }
        )
    return {
        "rows": pd.DataFrame(rows),
        "ece": float(ece),
        "maximum_supported_bin_error": float(max(supported)) if supported else math.nan,
        "has_supported_bin": bool(supported),
    }


def support_payload(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame.groupby("cycle_id", sort=True)["target"].sum()
    payload = {
        "compatible_rows": len(frame),
        "positive_rows": int(frame["target"].sum()),
        "cycles": int(frame["cycle_id"].nunique()),
        "minimum_positive_rows_per_cycle": int(positive.min()),
        "stocks": int(frame["symbol_norm"].nunique()),
        "quarters": int(frame["quarter"].nunique()),
        "current_states": int(frame["state"].nunique()),
    }
    payload["pass"] = bool(
        payload["compatible_rows"] >= 300_000
        and payload["positive_rows"] >= 10_000
        and payload["cycles"] == 20
        and payload["minimum_positive_rows_per_cycle"] >= 100
        and payload["stocks"] >= 20
        and payload["quarters"] == 2
        and payload["current_states"] == 8
    )
    return payload


def common_block_positions(
    date_count: int, seed: int, draws: int = 999
) -> np.ndarray:
    needed = int(math.ceil(date_count / 5))
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, date_count, size=(draws, needed))
    positions = (starts[:, :, None] + np.arange(5)[None, None, :]) % date_count
    return positions.reshape(draws, -1)[:, :date_count]


def bootstrap_payload(frame: pd.DataFrame) -> dict[str, Any]:
    dates = np.asarray(sorted(frame["session_date"].astype(str).unique()))
    endpoints = [
        (baseline, loss)
        for baseline in PRIMARY_BASELINES
        for loss in ("log_loss", "brier")
    ]
    target = frame["target"].to_numpy(int)
    candidate_loss = binary_losses(target, frame[PRIMARY_CANDIDATE])
    daily = np.empty((len(dates), len(endpoints)), dtype=float)
    for column, (baseline, loss) in enumerate(endpoints):
        difference = candidate_loss[loss] - binary_losses(target, frame[baseline])[loss]
        grouped = pd.DataFrame(
            {"session_date": frame["session_date"].astype(str), "difference": difference}
        ).groupby("session_date", sort=True)["difference"].mean()
        daily[:, column] = grouped.reindex(dates).to_numpy(float)
    positions = common_block_positions(len(dates), 20260711)
    samples = daily[positions].mean(axis=1)
    upper = np.quantile(samples, 0.9916666666666667, axis=0, method="linear")
    rows = [
        {
            "baseline": baseline,
            "loss": loss,
            "daily_mean_difference": float(daily[:, index].mean()),
            "upper_bound": float(upper[index]),
            "pass": bool(upper[index] < 0.0),
        }
        for index, (baseline, loss) in enumerate(endpoints)
    ]
    return {
        "rows": pd.DataFrame(rows),
        "block_positions": positions,
        "bootstrap_means": samples,
        "upper_quantile": 0.9916666666666667,
        "pass": bool(all(row["pass"] for row in rows)),
    }


def slice_rows(
    frame: pd.DataFrame,
    family: str,
    groups: Any,
    baselines: Sequence[str],
    minimum_rows: int,
    minimum_positives: int,
    *,
    losses: Sequence[str] = ("log_loss", "brier"),
    allow_zero: bool = False,
    gate_required: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = pd.Series(groups, index=frame.index, dtype="object")
    strings = grouped.astype(str)
    for value in sorted(grouped.dropna().astype(str).unique()):
        subset = frame.loc[strings.eq(value)].reset_index(drop=True)
        supported = len(subset) >= minimum_rows and int(subset["target"].sum()) >= minimum_positives
        for baseline in baselines:
            comparison = loss_comparison(
                subset["target"], subset[PRIMARY_CANDIDATE], subset[baseline]
            )
            for loss in losses:
                difference = float(comparison[f"{loss}_difference"])
                rows.append(
                    {
                        "family": family,
                        "value": value,
                        "baseline": baseline,
                        "loss": loss,
                        "rows": len(subset),
                        "positives": int(subset["target"].sum()),
                        "supported": bool(supported),
                        "difference": difference,
                        "gate_required": gate_required,
                        "pass": bool(
                            not supported
                            or (difference <= 0.0 if allow_zero else difference < 0.0)
                        ),
                    }
                )
    return rows


def subset_slice_rows(
    frame: pd.DataFrame,
    family: str,
    mask: Any,
    baselines: Sequence[str],
    minimum_rows: int,
    minimum_positives: int,
    *,
    gate_required: bool = True,
) -> list[dict[str, Any]]:
    groups = pd.Series(np.where(np.asarray(mask, bool), "selected", None), index=frame.index)
    return slice_rows(
        frame, family, groups, baselines, minimum_rows, minimum_positives,
        gate_required=gate_required,
    )


def slices_payload(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, groups in (
        ("time", frame["session_date"].astype(str).str[:7]),
        ("current_state", frame["state"]),
        ("transition_length", frame["transition_length"]),
    ):
        rows.extend(slice_rows(frame, family, groups, PRIMARY_BASELINES, 5_000, 100))
    rows.extend(subset_slice_rows(frame, "nonterminal", ~frame["terminal"].astype(bool), PRIMARY_BASELINES, 5_000, 100))
    rows.extend(subset_slice_rows(frame, "early_entry", frame["bar_ordinal"].to_numpy(int) <= 53, PRIMARY_BASELINES, 5_000, 100))
    rows.extend(slice_rows(frame, "cycle", frame["cycle_id"], PRIMARY_BASELINES, 500, 40, losses=("log_loss",)))
    orientation = frame["cycle_id"].astype(str) + "__s" + frame["state"].astype(str)
    rows.extend(slice_rows(frame, "cycle_current_state_orientation", orientation, ("qlimited4",), 500, 40, allow_zero=True))
    for column in sorted(name for name in frame.columns if name.startswith("factor_quartile__")):
        rows.extend(slice_rows(frame, column, frame[column], ("qlimited4",), 5_000, 100, allow_zero=True))
    for column in ("b0_entry_high_stress", "b0_unknown", "entry_clock_quartile"):
        rows.extend(slice_rows(frame, column, frame[column], PRIMARY_BASELINES, 5_000, 100, gate_required=False))
    rows.extend(subset_slice_rows(frame, "terminal", frame["terminal"].astype(bool), PRIMARY_BASELINES, 5_000, 0, gate_required=False))
    rows.extend(subset_slice_rows(frame, "late_entry", frame["bar_ordinal"].to_numpy(int) > 53, PRIMARY_BASELINES, 5_000, 100, gate_required=False))
    for symbol in sorted(frame["symbol_norm"].astype(str).unique()):
        subset = frame.loc[frame["symbol_norm"].astype(str).ne(symbol)]
        for baseline in PRIMARY_BASELINES:
            comparison = loss_comparison(subset["target"], subset[PRIMARY_CANDIDATE], subset[baseline])
            for loss in ("log_loss", "brier"):
                difference = float(comparison[f"{loss}_difference"])
                rows.append(
                    {
                        "family": "leave_one_stock_out",
                        "value": symbol,
                        "baseline": baseline,
                        "loss": loss,
                        "rows": len(subset),
                        "positives": int(subset["target"].sum()),
                        "supported": True,
                        "difference": difference,
                        "gate_required": True,
                        "pass": bool(difference < 0.0),
                    }
                )
    return pd.DataFrame(rows)


def stable_logit(probability: Any) -> np.ndarray:
    p = np.clip(np.asarray(probability, float), EPSILON, 1.0 - EPSILON)
    return np.log(p) - np.log1p(-p)


def stable_sigmoid(values: Any) -> np.ndarray:
    eta = np.asarray(values, float)
    output = np.empty_like(eta)
    positive = eta >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exponential = np.exp(eta[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def left_rotate(blocks: Sequence[np.ndarray], boundary: int) -> np.ndarray:
    if boundary < 1 or boundary >= len(blocks):
        raise ValueError("invalid session boundary")
    return np.concatenate(list(blocks[boundary:]) + list(blocks[:boundary]))


def holm_payload(p_values: Mapping[str, float]) -> dict[str, Any]:
    order = (
        "unweighted_log_loss",
        "inverse_compatible_weighted_log_loss",
        "top_three_recall",
    )
    order_index = {name: index for index, name in enumerate(order)}
    ranked = sorted(order, key=lambda name: (p_values[name], order_index[name]))
    rows: list[dict[str, Any]] = []
    passed = True
    for rank, name in enumerate(ranked, start=1):
        threshold = 0.01 / (len(ranked) - rank + 1)
        row_pass = p_values[name] <= threshold
        passed &= row_pass
        rows.append(
            {
                "name": name,
                "rank": rank,
                "p_value": p_values[name],
                "threshold": threshold,
                "pass": row_pass,
            }
        )
    return {"rows": pd.DataFrame(rows), "pass": bool(passed)}


def direct_falsification_statistics(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    unweighted = loss_comparison(
        frame["target"], probability, frame["qlimited4"]
    )["relative_log_loss_improvement"]
    weighted = loss_comparison(
        frame["target"], probability, frame["qlimited4"], frame["n_compatible"]
    )["relative_log_loss_improvement"]
    scratch = frame.copy()
    scratch["__candidate"] = probability
    recall = ranking_metrics(scratch, "__candidate")["recall"] - ranking_metrics(
        scratch, "qlimited4"
    )["recall"]
    return np.asarray([unweighted, weighted, recall], float)


def falsification_payload(predictions: pd.DataFrame) -> dict[str, Any]:
    metadata = predictions[
        ["anchor_id", "symbol_norm", "session_date", "start_timestamp", "state"]
    ].drop_duplicates("anchor_id")
    metadata = metadata.sort_values(
        ["symbol_norm", "session_date", "start_timestamp", "anchor_id"], kind="stable"
    ).reset_index(drop=True)
    metadata["month"] = metadata["session_date"].astype(str).str[:7]
    strata_ids: list[np.ndarray] = []
    for _, indices in metadata.groupby(["symbol_norm", "month", "state"], sort=True).groups.items():
        selected = metadata.loc[np.asarray(indices, int)]
        if selected["session_date"].nunique() >= 2:
            strata_ids.append(selected["anchor_id"].to_numpy())
    eligible_ids = {value for group in strata_ids for value in group.tolist()}
    frame = predictions.loc[predictions["anchor_id"].isin(eligible_ids)].copy()
    frame = frame.sort_values(["anchor_id", "cycle_id"], kind="stable").reset_index(drop=True)
    cycles = sorted(frame["cycle_id"].astype(str).unique())
    anchors = sorted(frame["anchor_id"].unique())
    anchor_position = {value: index for index, value in enumerate(anchors)}
    cycle_position = {value: index for index, value in enumerate(cycles)}
    row_anchor = frame["anchor_id"].map(anchor_position).to_numpy(int)
    row_cycle = frame["cycle_id"].astype(str).map(cycle_position).to_numpy(int)
    limited_eta = stable_logit(frame["qlimited4"])
    full_eta = stable_logit(frame["qfull9"])
    residual = np.full((len(anchors), len(cycles)), np.nan)
    residual[row_anchor, row_cycle] = full_eta - limited_eta
    limited_matrix = np.full_like(residual, np.nan)
    limited_matrix[row_anchor, row_cycle] = limited_eta
    target_matrix = np.zeros_like(residual, dtype=np.int8)
    target_matrix[row_anchor, row_cycle] = frame["target"].to_numpy(np.int8)
    replay_error = float(
        np.max(
            np.abs(stable_sigmoid(limited_eta + full_eta - limited_eta) - frame["qfull9"].to_numpy(float))
        )
    )
    target = frame["target"].to_numpy(float)
    weights = inverse_weights(frame["n_compatible"])
    limited_loss = np.logaddexp(0.0, limited_eta) - target * limited_eta
    limited_mean = float(limited_loss.mean())
    limited_weighted = weighted_mean(limited_loss, weights)
    positive_anchor, positive_cycle = np.nonzero(target_matrix)

    def recall(logits: np.ndarray) -> float:
        positive_scores = logits[positive_anchor]
        own = logits[positive_anchor, positive_cycle]
        higher = np.sum(positive_scores > own[:, None], axis=1)
        cycle_indices = np.arange(logits.shape[1])
        earlier = np.sum(
            (positive_scores == own[:, None])
            & (cycle_indices[None, :] < positive_cycle[:, None]),
            axis=1,
        )
        return float(np.mean(1 + higher + earlier <= 3))

    limited_recall = recall(limited_matrix)

    def statistics(logits: np.ndarray) -> np.ndarray:
        long = logits[row_anchor, row_cycle]
        loss = np.logaddexp(0.0, long) - target * long
        return np.asarray(
            [
                (limited_mean - float(loss.mean())) / limited_mean,
                (limited_weighted - weighted_mean(loss, weights)) / limited_weighted,
                recall(logits) - limited_recall,
            ]
        )

    observed = statistics(limited_matrix + residual)
    direct = direct_falsification_statistics(frame, frame["qfull9"].to_numpy(float))
    statistic_error = float(np.max(np.abs(observed - direct)))
    anchor_meta = frame[["anchor_id", "session_date", "start_timestamp"]].drop_duplicates("anchor_id").set_index("anchor_id")
    strata: list[list[np.ndarray]] = []
    for ids in strata_ids:
        ordered = anchor_meta.loc[list(ids)].reset_index().sort_values(
            ["session_date", "start_timestamp", "anchor_id"], kind="stable"
        )
        signatures = frame.loc[frame["anchor_id"].isin(ids)].groupby("anchor_id", sort=False)["cycle_id"].agg(
            lambda values: tuple(sorted(values.astype(str)))
        )
        if signatures.nunique() != 1:
            raise AssertionError("falsification stratum cycle support changed")
        strata.append(
            [
                np.asarray([anchor_position[value] for value in group["anchor_id"]], int)
                for _, group in ordered.groupby("session_date", sort=True)
            ]
        )
    rng = np.random.Generator(np.random.PCG64(20260711))
    null = np.empty((999, 3), dtype=float)
    for draw in range(999):
        shifted = residual.copy()
        for sessions in strata:
            boundary = int(rng.integers(1, len(sessions)))
            target_positions = np.concatenate(sessions)
            donor_positions = left_rotate(sessions, boundary)
            shifted[target_positions] = residual[donor_positions]
        null[draw] = statistics(limited_matrix + shifted)
    p_array = (1 + (null >= observed[None, :]).sum(axis=0)) / 1000
    names = (
        "unweighted_log_loss",
        "inverse_compatible_weighted_log_loss",
        "top_three_recall",
    )
    p_values = {name: float(p_array[index]) for index, name in enumerate(names)}
    holm = holm_payload(p_values)
    return {
        "draws": 999,
        "seed": 20260711,
        "eligible_rows": len(frame),
        "eligible_anchors": int(frame["anchor_id"].nunique()),
        "strata": len(strata),
        "full_logit_replay_max_error": replay_error,
        "statistic_replay_max_error": statistic_error,
        "statistics": pd.DataFrame(
            {
                "name": names,
                "observed": observed,
                "null_mean": null.mean(axis=0),
                "null_q99": np.quantile(null, 0.99, axis=0, method="linear"),
                "p_value": p_array,
            }
        ),
        "null_statistics": null,
        "holm": holm,
        "pass": bool(holm["pass"]),
    }


def stability_gates(slices: pd.DataFrame) -> dict[str, Any]:
    supported = slices.loc[slices["supported"].astype(bool)]
    required_families = (
        "time", "leave_one_stock_out", "current_state", "transition_length",
        "nonterminal", "early_entry",
    )
    family = {
        name: bool(
            not supported.loc[supported["family"].eq(name)].empty
            and supported.loc[supported["family"].eq(name), "pass"].all()
        )
        for name in required_families
    }
    observed_times = set(supported.loc[supported["family"].eq("time"), "value"].astype(str))
    family["time_expected_values"] = observed_times == set(OUTER_MONTHS)
    cycle = supported.loc[
        supported["family"].eq("cycle") & supported["loss"].eq("log_loss")
    ]
    counts = {
        baseline: int(
            cycle.loc[cycle["baseline"].eq(baseline) & cycle["pass"].astype(bool), "value"].nunique()
        )
        for baseline in PRIMARY_BASELINES
    }
    cycle_pass = all(value >= 15 for value in counts.values())
    orientation = supported.loc[supported["family"].eq("cycle_current_state_orientation")]
    orientation_pass = bool(not orientation.empty and orientation["pass"].all())
    quartile = {
        f"factor_quartile__{factor}": bool(
            not supported.loc[supported["family"].eq(f"factor_quartile__{factor}")].empty
            and supported.loc[supported["family"].eq(f"factor_quartile__{factor}"), "pass"].all()
        )
        for factor in NEW5
    }
    return {
        "required_families": family,
        "cycle_improvement_counts": counts,
        "cycle_pass": cycle_pass,
        "orientation_pass": orientation_pass,
        "factor_quartile_pass": quartile,
        "pass": bool(all(family.values()) and cycle_pass and orientation_pass and all(quartile.values())),
    }


def evaluate_gates(
    support: Mapping[str, Any],
    comparisons: pd.DataFrame,
    ranking: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    slices: pd.DataFrame,
    falsification: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_map = {
        surface: {
            row["baseline"]: row
            for row in comparisons.loc[comparisons["surface"].eq(surface)].to_dict(orient="records")
        }
        for surface in ("unweighted", "inverse_compatible")
    }
    rank = {row["model"]: row for row in ranking.to_dict(orient="records")}
    calibration_map = {row["model"]: row for row in calibration_summary.to_dict(orient="records")}
    bootstrap_map = {
        (row["baseline"], row["loss"]): row
        for row in bootstrap["rows"].to_dict(orient="records")
    }
    requirements = {
        "qhistory": (0.01, 0.005),
        "qpattern": (0.005, 0.002),
        "qlimited4": (0.0025, 0.002),
    }
    comparison_gates: dict[str, Any] = {}
    for baseline, (minimum_loss, minimum_recall) in requirements.items():
        pooled = comparison_map["unweighted"][baseline]
        checks = {
            "relative_log_loss": pooled["relative_log_loss_improvement"] >= minimum_loss,
            "brier": pooled["brier_difference"] < 0.0,
            "bootstrap_log_loss": bool(bootstrap_map[(baseline, "log_loss")]["pass"]),
            "bootstrap_brier": bool(bootstrap_map[(baseline, "brier")]["pass"]),
            "top_three_recall": rank[PRIMARY_CANDIDATE]["recall"] - rank[baseline]["recall"] >= minimum_recall,
            "top_three_precision": rank[PRIMARY_CANDIDATE]["precision"] - rank[baseline]["precision"] >= 0.0,
            "positive_anchor_hit_rate": rank[PRIMARY_CANDIDATE]["positive_anchor_hit_rate"] - rank[baseline]["positive_anchor_hit_rate"] >= -0.002,
            "inverse_log_loss": comparison_map["inverse_compatible"][baseline]["log_loss_difference"] <= 0.0,
            "inverse_brier": comparison_map["inverse_compatible"][baseline]["brier_difference"] <= 0.0,
        }
        comparison_gates[baseline] = {"checks": checks, "pass": bool(all(checks.values()))}
    lineage = comparison_map["unweighted"][LINEAGE_BASELINE]
    lineage_checks = {
        "log_loss": lineage["log_loss_difference"] <= 0.0,
        "brier": lineage["brier_difference"] <= 0.0,
        "top_three_recall": rank[PRIMARY_CANDIDATE]["recall"] - rank[LINEAGE_BASELINE]["recall"] >= 0.0,
    }
    candidate = calibration_map[PRIMARY_CANDIDATE]
    calibration_checks = {
        "supported": bool(candidate["has_supported_bin"]),
        "ece": all(candidate["ece"] <= calibration_map[baseline]["ece"] for baseline in PRIMARY_BASELINES),
        "absolute_maximum": candidate["maximum_supported_bin_error"] <= 0.02,
        "history_margin": candidate["maximum_supported_bin_error"] <= calibration_map["qhistory"]["maximum_supported_bin_error"] + 0.005,
    }
    stability = stability_gates(slices)
    pooled = bool(
        support["pass"]
        and all(value["pass"] for value in comparison_gates.values())
        and all(lineage_checks.values())
        and all(calibration_checks.values())
        and bootstrap["pass"]
    )
    return {
        "support_pass": bool(support["pass"]),
        "comparisons": comparison_gates,
        "lineage_baseline": {"checks": lineage_checks, "pass": bool(all(lineage_checks.values()))},
        "calibration": {"checks": calibration_checks, "pass": bool(all(calibration_checks.values()))},
        "stability": stability,
        "bootstrap_pass": bool(bootstrap["pass"]),
        "falsification_pass": bool(falsification["pass"]),
        "pooled_primary_pass": pooled,
        "primary_pass": bool(pooled and stability["pass"] and falsification["pass"]),
    }


def independently_evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    support = support_payload(frame)
    overall_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for surface in ("unweighted", "inverse_compatible"):
        counts = None if surface == "unweighted" else frame["n_compatible"]
        for model in MODEL_COLUMNS:
            overall_rows.append(
                {"surface": surface, "model": model, **loss_metrics(frame["target"], frame[model], counts)}
            )
        for baseline in (*PRIMARY_BASELINES, LINEAGE_BASELINE):
            comparison_rows.append(
                {
                    "surface": surface,
                    "candidate": PRIMARY_CANDIDATE,
                    "baseline": baseline,
                    **loss_comparison(frame["target"], frame[PRIMARY_CANDIDATE], frame[baseline], counts),
                }
            )
    overall = pd.DataFrame(overall_rows)
    comparisons = pd.DataFrame(comparison_rows)
    ranking = pd.DataFrame(
        [{"model": model, **ranking_metrics(frame, model)} for model in MODEL_COLUMNS]
    )
    calibration_summaries: list[dict[str, Any]] = []
    calibration_bins: list[pd.DataFrame] = []
    for model in (*PRIMARY_BASELINES, PRIMARY_CANDIDATE):
        result = calibration(frame, model)
        calibration_summaries.append(
            {
                "model": model,
                "ece": result["ece"],
                "maximum_supported_bin_error": result["maximum_supported_bin_error"],
                "has_supported_bin": result["has_supported_bin"],
            }
        )
        bins = result["rows"].copy()
        bins.insert(0, "model", model)
        calibration_bins.append(bins)
    calibration_payload = {
        "summary": pd.DataFrame(calibration_summaries),
        "bins": pd.concat(calibration_bins, ignore_index=True),
    }
    bootstrap = bootstrap_payload(frame)
    slices = slices_payload(frame)
    falsification = falsification_payload(frame)
    gates = evaluate_gates(
        support, comparisons, ranking, calibration_payload["summary"], bootstrap, slices, falsification
    )
    return {
        "support": support,
        "overall": overall,
        "ranking": ranking,
        "calibration": calibration_payload,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "slices": slices,
        "falsification": falsification,
        "gates": gates,
    }


def verify_contract_and_sources(audit: Audit, contract: Mapping[str, Any]) -> dict[str, str]:
    semantics = {
        "objective": contract.get("objective", "").startswith("Test whether nine causal run-entry factors"),
        "population": contract["population_and_target"]["anchor_universe"]
        == "every_filtered_state_run_entry_in_the_frozen_period_run_file",
        "terminal": "remains_in_residual_fitting" in contract["population_and_target"]["terminal_anchor_semantics"],
        "softmax_forbidden": contract["population_and_target"]["softmax_across_cycles_forbidden"] is True,
        "features": tuple(contract["feature_construction"]["full9"]) == FULL9,
        "design_widths": (
            contract["models"]["exact_design"]["qpattern_width"] == PATTERN_WIDTH
            and contract["models"]["exact_design"]["qlimited4_width"] == LIMITED_WIDTH
            and contract["models"]["exact_design"]["qfull9_width"] == FULL_WIDTH
        ),
        "raw_primary": contract["models"]["raw_probabilities_are_primary"] is True,
        "calibration_forbidden": contract["models"]["post_hoc_calibration_allowed"] is False,
        "outer_schedule": tuple(contract["periods"]["outer_oof_months"]) == OUTER_MONTHS,
        "full_schedule": tuple(contract["causal_nested_selection"]["full_fit_lambda_validation_months"]) == GRID_MONTHS,
        "later_cannot_promote": contract["decision_rule"]["later_periods_can_promote"] is False,
        "grade_unchanged": contract["decision_rule"]["good_or_high_movement_quality_grade_changed"] is False,
    }
    audit.check("frozen_contract_semantics_exact", all(semantics.values()), semantics)
    source_names = (
        "runs_2024",
        "cycles",
        "retained_path_parameters",
        "retained_path_gates",
        "provider_hash_manifest",
    )
    hashes: dict[str, str] = {"contract": sha256(CONTRACT_PATH)}
    for name in source_names:
        spec = contract["frozen_sources"][name]
        path = Path(spec["path"])
        hashes[name] = sha256(path)
        audit.check(f"source_hash_{name}_exact", hashes[name] == spec["sha256"])
    runner_source = RUNNER_PATH.read_text()
    runner_tree = ast.parse(runner_source)
    imports_auditor = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "audit_factor_conditioned_loop_occurrence_v1" in (ast.get_source_segment(runner_source, node) or "")
        for node in ast.walk(runner_tree)
    )
    functions = {
        node.name for node in runner_tree.body if isinstance(node, ast.FunctionDef)
    }
    audit.check(
        "runner_and_auditor_are_phase_separated",
        not imports_auditor and {"run_fit_only", "run_score_only"}.issubset(functions),
        {"imports_auditor": imports_auditor, "functions": sorted(functions)},
    )
    return hashes


def compare_json_artifact(
    audit: Audit, name: str, observed: Any, expected: Any, *, tolerance: float = 1e-10
) -> None:
    differences = nested_differences(observed, expected, tolerance=tolerance)
    audit.check(name, not differences, differences[:50])


def verify_static_fit_artifacts(
    audit: Audit,
    root: Path,
    prepared: Prepared2024,
    evaluation: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> None:
    expected_population = {
        "period": "2024",
        "anchors": len(prepared.anchors),
        "compatible_rows": len(prepared.expanded),
        "positive_rows": int(prepared.expanded["target"].sum()),
        "cycles": len(prepared.cycles),
        "pass": True,
    }
    compare_json_artifact(
        audit,
        "population_validation_independently_exact",
        json.loads((root / "population_validation_2024.json").read_text()),
        expected_population,
    )
    expected_schedule = {
        "outer": {key: list(value) for key, value in INNER_SCHEDULE.items()},
        "full_selection_months": list(GRID_MONTHS),
        "heads_select_independently": list(HEADS),
        "training_rule": "month strictly before validation month",
    }
    compare_json_artifact(
        audit,
        "fold_schedule_independently_exact",
        json.loads((root / "fold_schedule.json").read_text()),
        expected_schedule,
    )
    expected_grid = {
        "lambdas": list(RIDGE_GRID),
        "tie_tolerance": 1e-12,
        "tie_break": "largest_lambda",
        "selection_objective": "equal_mean_of_validation_month_log_loss",
    }
    compare_json_artifact(
        audit,
        "hyperparameter_grid_independently_exact",
        json.loads((root / "hyperparameter_grid.json").read_text()),
        expected_grid,
    )
    stored_source = json.loads((root / "fit_source_manifest_2024.json").read_text())
    source_checks = {
        "factor_hash": stored_source.get("factor_table_sha256") == prepared.factor_hash,
        "period": stored_source.get("period") == "2024" and stored_source.get("year") == 2024,
        "population": stored_source.get("run_rows") == len(prepared.anchors)
        and stored_source.get("compatible_rows") == len(prepared.expanded)
        and stored_source.get("positive_rows") == int(prepared.expanded["target"].sum()),
        "provider_hashes": stored_source.get("provider_file_hashes")
        == prepared.source_audit["factor_table"]["provider_file_hashes"],
        "placeholder": stored_source.get("placeholder_cleanup")
        == {
            **prepared.source_audit["factor_table"]["placeholder_cleanup"],
            "run_entry_natural_key_intersection": 0,
        },
        "phase": stored_source.get("later_period_paths_resolved") is False
        and stored_source.get("later_period_rows_read") is False
        and stored_source.get("shadow_tree_read") is False
        and stored_source.get("shadow_tree_written") is False,
        "safety": all(stored_source.get(key) == value for key, value in SAFETY.items()),
        "contract": stored_source.get("contract_sha256") == CONTRACT_SHA256,
        "sources": stored_source.get("production_source_hashes")
        == {CORE_PATH.name: sha256(CORE_PATH), EVALUATOR_PATH.name: sha256(EVALUATOR_PATH)},
        "year": stored_source.get("fit_phase_year") == 2024,
    }
    audit.check("fit_source_manifest_independently_exact", all(source_checks.values()), source_checks)

    frame_artifacts = (
        ("overall_2024.csv", evaluation["overall"], ["surface", "model"], "all_pooled_losses_independently_exact"),
        ("ranking_2024.csv", evaluation["ranking"], ["model"], "all_ranking_metrics_independently_exact"),
        ("comparisons_2024.csv", evaluation["comparisons"], ["surface", "baseline"], "all_loss_comparisons_independently_exact"),
        ("slices_2024.csv", evaluation["slices"], ["family", "value", "baseline", "loss"], "all_slice_diagnostics_independently_exact"),
    )
    for filename, expected, keys, check_name in frame_artifacts:
        observed = pd.read_csv(root / filename)
        passed, details = frame_difference(observed, expected, keys)
        audit.check(check_name, passed, details)
    for filename, key, check_name in (
        ("support_2024.json", "support", "support_gates_independently_exact"),
        ("calibration_2024.json", "calibration", "calibration_bins_independently_exact"),
        ("bootstrap_2024.json", "bootstrap", "common_bootstrap_draws_independently_exact"),
        ("falsification_2024.json", "falsification", "all_falsification_draws_and_holm_independently_exact"),
        ("gates_2024.json", "gates", "all_primary_gates_independently_exact"),
    ):
        compare_json_artifact(
            audit,
            check_name,
            json.loads((root / filename).read_text()),
            evaluation[key],
        )
    expected_rejection = {
        "development_2024_primary_pass": False,
        "label": "factor_conditioned_loop_occurrence_rejected_2024_and_do_not_score_later_periods",
        "later_scoring_authorized": False,
        "prospective_validated": False,
        "movement_quality_grade_changed": False,
    }
    provisional = json.loads((root / "provisional_decision.json").read_text())
    compare_json_artifact(
        audit, "rejection_decision_independently_exact", provisional, expected_rejection
    )
    gate = evaluation["gates"]
    rejection_checks = {
        "independent_primary_failed": gate["primary_pass"] is False,
        "independent_pooled_passed": gate["pooled_primary_pass"] is True,
        "independent_orientation_failed": gate["stability"]["orientation_pass"] is False,
        "marker_failed": fit.get("development_2024_primary_pass") is False,
        "marker_no_full_fit": fit.get("selected_full_lambdas") is None,
        "marker_no_authorization": fit.get("scoring_authorized") is False,
    }
    audit.check(
        "2024_orientation_failure_rejection_is_independently_reproduced",
        all(rejection_checks.values()),
        rejection_checks,
    )


def data_free_self_test() -> dict[str, Any]:
    contract = load_contract()
    checks = {
        "contract_hash": sha256(CONTRACT_PATH) == CONTRACT_SHA256,
        "safety": all(contract[key] == value for key, value in SAFETY.items()),
        "token_boundary": int(history_token([8], [8], [7])[0]) == 647,
        "design_widths": factor_layout(0)["width"] == PATTERN_WIDTH
        and factor_layout(4)["width"] == LIMITED_WIDTH
        and factor_layout(9)["width"] == FULL_WIDTH,
        "penalty_widths": all(
            len(penalties(count)) == factor_layout(count)["width"] for count in (0, 4, 9)
        ),
        "causal_schedule": all(
            all(inner < outer for inner in months)
            for outer, months in INNER_SCHEDULE.items()
        ),
        "no_production_import": "factor_conditioned_loop_occurrence_core" not in sys.modules
        and "factor_conditioned_loop_occurrence_eval" not in sys.modules
        and "run_factor_conditioned_loop_occurrence_v1" not in sys.modules,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "status": "independent_auditor_self_tests_passed",
        "checks": checks,
        "passed": len(checks),
        "total": len(checks),
        "later_period_paths_resolved": False,
        **SAFETY,
    }


def audit_rejection(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    audit = Audit()
    contract = load_contract()
    source_hashes = verify_contract_and_sources(audit, contract)
    fit, manifest = verify_fit_lock(audit, root)
    if not fit or not manifest or not audit.all_passed:
        return {
            "phase": "factor_conditioned_loop_occurrence_v1_independent_2024_audit",
            "all_passed": False,
            "checks": audit.checks,
            "scoring_authorized": False,
            "rejection_verified": False,
            "later_period_paths_resolved": False,
            "later_period_rows_read": False,
            "shadow_tree_read": False,
            "shadow_tree_written": False,
            **SAFETY,
        }
    prepared = reconstruct_2024(contract)
    audit.check(
        "full_2024_population_features_labels_and_terminal_rows_exact",
        len(prepared.anchors) == 110_949
        and len(prepared.expanded) == 759_212
        and int(prepared.expanded["target"].sum()) == 46_630
        and prepared.factor_hash
        == contract["feature_construction"]["provider_scan_and_factor_table"]["fit_2024_canonical_retained_factor_table_sha256"],
        {
            "anchors": len(prepared.anchors),
            "compatible_rows": len(prepared.expanded),
            "positives": int(prepared.expanded["target"].sum()),
            "factor_hash": prepared.factor_hash,
        },
    )
    _, expected_oof, replay = reconstruct_grid_and_oof(audit, root, prepared)
    evaluation = independently_evaluate(expected_oof)
    verify_static_fit_artifacts(audit, root, prepared, evaluation, fit)
    final_manifest_hash = sha256(root / "complete_fit_artifact_manifest.json")
    final_fit_hash = sha256(root / "fit_complete.json")
    audit.check(
        "fit_bundle_unchanged_through_independent_audit",
        fit.get("complete_fit_artifact_manifest_sha256") == final_manifest_hash
        and fit.get("runner_sha256") == sha256(RUNNER_PATH),
        {"manifest_sha256": final_manifest_hash, "fit_complete_sha256": final_fit_hash},
    )
    result = {
        "phase": "factor_conditioned_loop_occurrence_v1_independent_2024_audit",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(RUNNER_PATH),
        "auditor_source_sha256": sha256(Path(__file__)),
        "fit_complete_sha256": final_fit_hash,
        "complete_fit_artifact_manifest_sha256": final_manifest_hash,
        "source_hashes": source_hashes,
        "replay_folds": sorted(replay),
        "development_2024_primary_pass": False,
        "rejection_verified": audit.all_passed,
        "scoring_authorized": False,
        "authorization_marker_written": False,
        "later_period_paths_resolved": False,
        "later_period_rows_read": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "prospective_validated": False,
        "movement_quality_grade_changed": False,
        **SAFETY,
    }
    if audit.all_passed:
        output = root / "pre_score_audit.json"
        if output.exists() or (root / "pre_score_authorization.json").exists():
            raise FileExistsError("stale audit or authorization marker exists")
        write_json(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test-only", action="store_true")
    modes.add_argument("--audit-rejection", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = data_free_self_test() if args.self_test_only else audit_rejection(args.root)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    if not result.get("all_passed", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
