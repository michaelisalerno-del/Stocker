"""Reusable causal core for factor-conditioned fixed-loop occurrence research.

This module contains data validation, causal feature construction, overlapping
loop labels, fold-local path offsets, the frozen direct residual design, and
the custom offset-ridge optimizer.  Phase orchestration and scientific gates
belong in the separate runner.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
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
OUT = Path("/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711")

RUN_2024 = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/train_2024_filtered_runs.csv")
CYCLE_PATH = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/fixed_cycle_shuffled_nulls.csv")
PATH_PARAMETERS = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710/model_parameters.npz")
PATH_GATES = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710/gates.json")
PROVIDER_HASH_MANIFEST = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710/source_hashes.json")
PROVIDER_HASH_MANIFEST_SHA256 = "b63b51dc41c7868fc54cb0a9b1114e9095b2fdd89a9510baaec7c375db1da619"
CYCLE_SHA256 = "5695f09a7573a110034d251b5abdc40c2f37a11cc7198b196636a624c7d1ad22"
PROVIDER_ROOT_2024_2025 = Path("/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock")

K = 8
END_STATE = 8
DESTINATIONS = 9
TOKEN_WIDTH = 648
CYCLE_COUNT = 20
CYCLE_CONTRAST_WIDTH = 19
ROUTE_COUNT = 44
ROUTE_CONTRAST_WIDTH = 24
PATTERN_WIDTH = 44
LIMITED_WIDTH = 2812
FULL_WIDTH = 6272
EPSILON = 1e-12
SEED = 20260711
LAMBDA_GRID = (0.0001, 0.0003, 0.001, 0.003)
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
IRREGULAR_DATE = "2025-04-10"
IRREGULAR_SYMBOLS = ("CIFR", "IREN", "NVTS", "RIVN", "WULF")
OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close")
NATURAL_KEY = ("symbol_norm", "session_date", "start_timestamp")
SAFETY = {
    "research_only": True,
    "live_ordering_enabled": False,
    "order_placement": "disabled",
}


@dataclass(frozen=True)
class TransitionKernel:
    classes: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    numeric_width: int
    n_iter: int


@dataclass(frozen=True)
class NumericTransform:
    columns: tuple[str, ...]
    medians: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True)
class ContrastSpec:
    cycle_weights: np.ndarray
    route_weights: np.ndarray
    cycle_reference: int
    route_reference_by_cycle: np.ndarray


@dataclass(frozen=True)
class ResidualFit:
    coefficients: np.ndarray
    ridge_lambda: float
    factor_count: int
    objective: float
    gradient_max_abs: float
    iterations: int
    optimizer_message: str


@dataclass
class PreparedPeriod:
    anchors: pd.DataFrame
    expanded: pd.DataFrame
    cycles: pd.DataFrame
    route_map: pd.DataFrame
    factor_table_hash: str
    audit: dict[str, Any]

    @property
    def source_manifest(self) -> dict[str, Any]:
        return {
            "factor_table_sha256": self.factor_table_hash,
            "period": self.audit.get("period"),
            "year": self.audit.get("year"),
            "run_rows": self.audit.get("run_rows"),
            "compatible_rows": self.audit.get("compatible_rows"),
            "positive_rows": self.audit.get("positive_rows"),
            "provider_file_hashes": self.audit.get("factor_table", {}).get(
                "provider_file_hashes", {}
            ),
            "placeholder_cleanup": self.audit.get("factor_table", {}).get(
                "placeholder_cleanup", {}
            ),
            "later_period_paths_resolved": self.audit.get(
                "later_period_paths_resolved"
            ),
            "later_period_rows_read": self.audit.get("later_period_rows_read"),
            "shadow_tree_read": self.audit.get("shadow_tree_read"),
            "shadow_tree_written": self.audit.get("shadow_tree_written"),
            **SAFETY,
        }


@dataclass
class ModelBundle:
    history_kernel: TransitionKernel
    old_limited_kernel: TransitionKernel
    numeric_transform: NumericTransform
    contrast_spec: ContrastSpec
    token_mask: np.ndarray
    quartile_cutpoints: np.ndarray
    qpattern: ResidualFit
    qlimited4: ResidualFit
    qfull9: ResidualFit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def validate_contract(contract_path: Path = CONTRACT_PATH) -> dict[str, Any]:
    if sha256(contract_path) != CONTRACT_SHA256:
        raise AssertionError("factor-conditioned occurrence contract changed")
    contract = json.loads(contract_path.read_text())
    for key, expected in SAFETY.items():
        if contract.get(key) != expected:
            raise AssertionError(f"safety label changed: {key}")
    if contract["periods"]["forbidden"] != [2026, "prospective_shadow"]:
        raise AssertionError("forbidden-period declaration changed")
    if tuple(contract["models"]["ridge_lambda_grid"]) != LAMBDA_GRID:
        raise AssertionError("ridge grid changed")
    return contract


def phase_source_paths(
    phase: str,
    authorization: Mapping[str, Any] | None = None,
    *,
    guard_token: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    common = {
        "runs_2024": RUN_2024,
        "cycles": CYCLE_PATH,
        "retained_path_parameters": PATH_PARAMETERS,
        "retained_path_gates": PATH_GATES,
        "provider_hash_manifest": PROVIDER_HASH_MANIFEST,
        "provider_root_2024": PROVIDER_ROOT_2024_2025,
    }
    if phase in ("validate", "fit"):
        return common
    if phase != "score":
        raise ValueError("phase must be validate, fit, or score")
    if guard_token is not None:
        if authorization is not None:
            raise ValueError("provide authorization or guard_token, not both")
        authorization = guard_token
    if authorization is None:
        raise PermissionError("later paths require hash-bound scoring authorization")
    required = {
        "core_guard_token": True,
        "auditor_all_passed": True,
        "development_2024_primary_pass": True,
        "scoring_authorized": True,
        **SAFETY,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise PermissionError(f"invalid scoring authorization field: {key}")
    return {
        **common,
        "runs_2025": Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/test_2025_filtered_runs.csv"),
        "runs_2023": Path("/private/tmp/stocker_sealed_backward_2023_complete_detector_20260710/backward_2023_filtered_runs.parquet"),
        "provider_root_2025": PROVIDER_ROOT_2024_2025,
        "provider_root_2023": Path("/private/tmp/stocker_eodhd_pre2024_intraday_20260710/source=eodhd/instrument_type=stock"),
        "retained_scoring_2025": Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710/scoring_2025.parquet"),
        "retained_scoring_2023": Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710/scoring_2023.parquet"),
    }


def authorize_later_phase(
    authorization: Mapping[str, Any],
    fit_complete: Mapping[str, Any],
    fit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an orchestrator/auditor marker without resolving later paths."""

    required_fields = tuple(
        validate_contract()["independent_audit_contract"][
            "authorization_marker_required_fields"
        ]
    )
    missing = sorted(set(required_fields).difference(authorization))
    if missing:
        raise PermissionError(f"authorization marker is incomplete: {missing}")
    required_values = {
        "contract_sha256": CONTRACT_SHA256,
        "auditor_all_passed": True,
        "development_2024_primary_pass": True,
        "scoring_authorized": True,
        **SAFETY,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
    }
    for key, expected in required_values.items():
        if authorization.get(key) != expected:
            raise PermissionError(f"authorization marker failed closed: {key}")
    for key in ("contract_sha256", "runner_sha256"):
        if fit_complete.get(key) != authorization.get(key):
            raise PermissionError(f"fit/authorization binding mismatch: {key}")
    # The orchestrator binds this value to the exact serialized manifest-file
    # bytes.  Never reserialize a parsed mapping here because whitespace would
    # change the hash.
    manifest_hash = fit_complete.get("complete_fit_artifact_manifest_sha256")
    if authorization.get("complete_fit_artifact_manifest_sha256") != manifest_hash:
        raise PermissionError("authorization does not bind the fit artifact manifest")
    if fit_manifest.get("manifest_sha256") not in (None, manifest_hash):
        raise PermissionError("parsed fit manifest reports a different file hash")
    return {**dict(authorization), "core_guard_token": True}


def validate_2024_source_integrity() -> dict[str, str]:
    contract = validate_contract()
    sources = contract["frozen_sources"]
    expected = {
        "runs_2024": sources["runs_2024"]["sha256"],
        "cycles": sources["cycles"]["sha256"],
        "retained_path_parameters": sources["retained_path_parameters"]["sha256"],
        "retained_path_gates": sources["retained_path_gates"]["sha256"],
        "provider_hash_manifest": sources["provider_hash_manifest"]["sha256"],
    }
    paths = phase_source_paths("validate")
    actual = {key: sha256(paths[key]) for key in expected}
    if actual != expected:
        raise AssertionError({"expected": expected, "actual": actual})
    return actual


def canonical_cycle(values: Iterable[int]) -> tuple[int, ...]:
    core = tuple(int(value) for value in values)
    if not core:
        raise ValueError("empty cycle")
    return min(core[index:] + core[:index] for index in range(len(core)))


def oriented_paths(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def load_cycles(path: Path = CYCLE_PATH) -> pd.DataFrame:
    if path == CYCLE_PATH and sha256(path) != CYCLE_SHA256:
        raise AssertionError("frozen cycle source hash changed")
    source = pd.read_csv(path)
    if len(source) != CYCLE_COUNT or "cycle" not in source:
        raise AssertionError("frozen cycle source changed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for index, text in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in text.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid closed cycle: {text}")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError(f"invalid or duplicate cycle: {text}")
        if min(core) < 0 or max(core) >= K:
            raise AssertionError(f"cycle state outside frozen range: {text}")
        seen.add(core)
        rows.append(
            {
                "cycle_index": index - 1,
                "cycle_id": f"cycle_{index:02d}",
                "cycle": "->".join(str(state) for state in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    return pd.DataFrame(rows)


def route_mapping(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for cycle in cycles.itertuples(index=False):
        states = sorted(set(int(state) for state in cycle.core))
        reference = max(states)
        for state in states:
            rows.append(
                {
                    "route_index": len(rows),
                    "cycle_index": int(cycle.cycle_index),
                    "cycle_id": str(cycle.cycle_id),
                    "current_state": state,
                    "is_reference": state == reference,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != ROUTE_COUNT:
        raise AssertionError(f"compatible route width changed: {len(result)}")
    if int((~result["is_reference"]).sum()) != ROUTE_CONTRAST_WIDTH:
        raise AssertionError("route contrast width changed")
    return result


def history_tokens(
    previous_state_2: Sequence[int] | np.ndarray,
    previous_state_1: Sequence[int] | np.ndarray,
    current_state: Sequence[int] | np.ndarray,
) -> np.ndarray:
    p2 = np.asarray(previous_state_2, dtype=int)
    p1 = np.asarray(previous_state_1, dtype=int)
    current = np.asarray(current_state, dtype=int)
    if p2.shape != p1.shape or p1.shape != current.shape:
        raise ValueError("history arrays differ in shape")
    if (
        p2.min(initial=0) < 0
        or p2.max(initial=0) > END_STATE
        or p1.min(initial=0) < 0
        or p1.max(initial=0) > END_STATE
        or current.min(initial=0) < 0
        or current.max(initial=0) >= K
    ):
        raise AssertionError("history token contains an invalid state")
    token = ((p2 * 9 + p1) * 8 + current).astype(np.int64)
    if token.min(initial=0) < 0 or token.max(initial=0) >= TOKEN_WIDTH:
        raise AssertionError("history token outside frozen width")
    return token


def load_runs(
    path: Path,
    year: int,
    period: str,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    expected_stocks: int | None = None,
) -> pd.DataFrame:
    if year >= 2026:
        raise PermissionError("2026 and later are forbidden")
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise AssertionError(f"run source hash changed for {period}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "run_id", "symbol_norm", "session_date", "state", "start_pos",
        "start_timestamp", "previous_state_1", "previous_state_2",
        "b0_state_numeric", "b0_high_stress", "next_state", "has_next_state",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"missing run columns: {missing}")
    output = frame.copy()
    output["symbol_norm"] = output["symbol_norm"].astype(str)
    output["session_date"] = output["session_date"].astype(str)
    output["state"] = pd.to_numeric(output["state"], errors="raise").astype(int)
    output["start_pos"] = pd.to_numeric(output["start_pos"], errors="raise").astype(int)
    output["start_timestamp"] = pd.to_datetime(output["start_timestamp"], utc=True, errors="raise")
    output = output.sort_values(["symbol_norm", "session_date", "start_pos"], kind="stable").reset_index(drop=True)
    if expected_rows is not None and len(output) != expected_rows:
        raise AssertionError(f"run row count changed for {period}")
    if expected_stocks is not None and output["symbol_norm"].nunique() != expected_stocks:
        raise AssertionError(f"stock count changed for {period}")
    dates = pd.to_datetime(output["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {year} or output["start_timestamp"].dt.year.ne(year).any():
        raise AssertionError(f"out-of-year run entered {period}")
    if output.duplicated(list(NATURAL_KEY)).any():
        raise AssertionError(f"duplicate run natural key in {period}")
    if output["state"].min() < 0 or output["state"].max() >= K:
        raise AssertionError("state outside frozen eight-state range")
    grouped = output.groupby(["symbol_norm", "session_date"], sort=False)["state"]
    expected_p1 = grouped.shift(1).fillna(END_STATE).astype(int)
    expected_p2 = grouped.shift(2).fillna(END_STATE).astype(int)
    if not np.array_equal(expected_p1, output["previous_state_1"].astype(int)):
        raise AssertionError(f"previous_state_1 mismatch in {period}")
    if not np.array_equal(expected_p2, output["previous_state_2"].astype(int)):
        raise AssertionError(f"previous_state_2 mismatch in {period}")
    expected_next = grouped.shift(-1)
    stored_has_next = output["has_next_state"].astype(bool)
    if not np.array_equal(expected_next.notna(), stored_has_next):
        raise AssertionError(f"has_next_state mismatch in {period}")
    stored_next = pd.to_numeric(output["next_state"], errors="coerce")
    if not np.array_equal(expected_next.loc[stored_has_next].astype(int), stored_next.loc[stored_has_next].astype(int)):
        raise AssertionError(f"next_state mismatch in {period}")
    output["next_outcome"] = expected_next.fillna(END_STATE).astype(int)
    output["terminal"] = ~stored_has_next
    output["anchor_id"] = np.arange(len(output), dtype=np.int64)
    output["month"] = output["session_date"].str[:7]
    output["quarter"] = dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
    output["period"] = period
    for step in range(1, 5):
        output[f"future_state_{step}"] = grouped.shift(-step).fillna(END_STATE).astype(int)
    output["history_token"] = history_tokens(output["previous_state_2"], output["previous_state_1"], output["state"])
    return output


def provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def provider_hashes_for_year(year: int) -> dict[str, str]:
    if year != 2024:
        raise PermissionError("fit-phase provider manifest access is restricted to 2024")
    if sha256(PROVIDER_HASH_MANIFEST) != json.loads(CONTRACT_PATH.read_text())["frozen_sources"]["provider_hash_manifest"]["sha256"]:
        raise AssertionError("provider hash manifest changed")
    payload = json.loads(PROVIDER_HASH_MANIFEST.read_text())
    prefix = f"provider_{year}_"
    result = {
        key[len(prefix) : -len(".parquet")]: str(value)
        for key, value in payload.items()
        if key.startswith(prefix) and key.endswith(".parquet")
    }
    if len(result) != 22:
        raise AssertionError("2024 provider symbol manifest changed")
    return result


def _canonical_factor_table_hash(frame: pd.DataFrame) -> str:
    columns = (
        "symbol_norm",
        "timestamp_ns_utc",
        "session_date",
        "bar_ordinal",
        *NEW5,
    )
    ordered = frame.loc[:, columns].sort_values(
        ["symbol_norm", "timestamp_ns_utc"], kind="stable"
    )
    digest = hashlib.sha256()
    digest.update((",".join(columns) + "\n").encode("utf-8"))
    for row in ordered.itertuples(index=False, name=None):
        fields = [
            str(row[0]),
            str(int(row[1])),
            str(row[2]),
            str(int(row[3])),
            *(format(float(value), ".17g") for value in row[4:]),
        ]
        digest.update((",".join(fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _canonical_discarded_key_hash(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, ["symbol_norm", "timestamp_ns_utc"]].sort_values(
        ["symbol_norm", "timestamp_ns_utc"], kind="stable"
    )
    content = "symbol_norm,timestamp_ns_utc\n" + "".join(
        f"{row.symbol_norm},{int(row.timestamp_ns_utc)}\n"
        for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def scan_provider_factors(
    symbols: Sequence[str],
    root: Path,
    year: int,
    *,
    expected_file_hashes: Mapping[str, str] | None = None,
    enforce_frozen_2024_placeholder_audit: bool = True,
    enforce_irregular_2025_audit: bool = True,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Read only a predicate-bounded year and construct the five causal factors."""

    if year >= 2026:
        raise PermissionError("2026 and later provider rows are forbidden")
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    parts: list[pd.DataFrame] = []
    discarded_parts: list[pd.DataFrame] = []
    discarded_per_symbol: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    for symbol in sorted(set(str(value) for value in symbols)):
        path = provider_path(root, symbol)
        if expected_file_hashes is not None:
            if symbol not in expected_file_hashes:
                raise AssertionError(f"missing provider hash for {symbol}")
            # The current physical file is shared across calendar years.  A
            # pre-score whole-file hash would read later-year bytes and violate
            # phase isolation.  The already hash-pinned manifest entry is
            # provenance only; the selected year is bound independently by the
            # canonical factor-table hash below.
            file_hashes[symbol] = str(expected_file_hashes[symbol])
        table = pq.read_table(
            path,
            columns=list(OHLC_COLUMNS),
            # The file lives below Hive-like source/symbol directories whose
            # dictionary encodings differ from in-file columns.  Disable
            # partition inference while retaining the row predicate pushdown.
            partitioning=None,
            filters=[
                ("timestamp", ">=", start.to_pydatetime()),
                ("timestamp", "<", end.to_pydatetime()),
            ],
        )
        bars = table.to_pandas()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        if bars["timestamp"].isna().any():
            raise AssertionError(f"null provider timestamp for {symbol} {year}")
        if bars.empty or bars["timestamp"].lt(start).any() or bars["timestamp"].ge(end).any():
            raise AssertionError(f"predicate pushdown admitted an invalid year for {symbol}")
        for column in ("open", "high", "low", "close"):
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
        local = bars["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        bars = bars.loc[minute.ge(570) & minute.lt(960)].copy()
        if bars.empty:
            raise AssertionError(f"empty regular-session provider table for {symbol}")
        nulls = bars.loc[:, ["open", "high", "low", "close"]].isna()
        all_null = nulls.all(axis=1)
        partial_null = nulls.any(axis=1) & ~all_null
        if partial_null.any():
            raise AssertionError(f"partial-null regular-session OHLC for {symbol}")
        discarded = bars.loc[all_null, ["timestamp"]].copy()
        discarded["symbol_norm"] = symbol
        discarded["timestamp_ns_utc"] = discarded["timestamp"].astype("int64")
        discarded_parts.append(
            discarded[["symbol_norm", "timestamp", "timestamp_ns_utc"]]
        )
        discarded_per_symbol[symbol] = len(discarded)
        bars = bars.loc[~all_null].copy()
        if bars.empty:
            raise AssertionError(f"placeholder cleanup emptied {symbol} {year}")
        ohlc = bars.loc[:, ["open", "high", "low", "close"]].to_numpy(float)
        if not np.isfinite(ohlc).all() or (ohlc <= 0.0).any():
            raise AssertionError(f"invalid nonpositive or nonfinite regular-session OHLC for {symbol}")
        if (
            (bars["high"] < bars[["open", "close"]].max(axis=1)).any()
            or (bars["low"] > bars[["open", "close"]].min(axis=1)).any()
            or (bars["high"] < bars["low"]).any()
        ):
            raise AssertionError(f"internally inconsistent regular-session OHLC for {symbol}")
        bars = bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if bars["timestamp"].duplicated().any():
            raise AssertionError(f"duplicate provider timestamp for {symbol}")
        local = bars["timestamp"].dt.tz_convert("America/New_York")
        bars["symbol_norm"] = symbol
        bars["session_date"] = local.dt.strftime("%Y-%m-%d")
        bars["bar_ordinal"] = bars.groupby("session_date", sort=False).cumcount()
        grouped = bars.groupby("session_date", sort=False)
        previous_close = grouped["close"].shift(1)
        first = bars["bar_ordinal"].eq(0)
        bars["current_bar_log_return"] = np.log(
            bars["close"] / previous_close.where(~first, bars["open"])
        )
        bars["return_sum_6"] = grouped["current_bar_log_return"].transform(
            lambda values: values.rolling(6, min_periods=1).sum()
        )
        bars["mean_abs_return_12"] = grouped["current_bar_log_return"].transform(
            lambda values: values.abs().rolling(12, min_periods=1).mean()
        )
        bars["session_return"] = np.log(
            bars["close"] / grouped["open"].transform("first")
        )
        bars["bar_range_pct"] = (bars["high"] - bars["low"]) / bars["open"]
        values = bars.loc[:, list(NEW5)].to_numpy(float)
        if not np.isfinite(values).all():
            raise AssertionError(f"nonfinite causal price factor for {symbol}")
        bars["timestamp_ns_utc"] = bars["timestamp"].astype("int64")
        parts.append(
            bars.loc[
                :,
                [
                    "symbol_norm",
                    "timestamp",
                    "timestamp_ns_utc",
                    "session_date",
                    "bar_ordinal",
                    *NEW5,
                ],
            ]
        )
    panel = pd.concat(parts, ignore_index=True).sort_values(
        ["symbol_norm", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if panel.duplicated(["symbol_norm", "timestamp"]).any():
        raise AssertionError("duplicate symbol timestamp in provider factor table")
    discarded_keys = pd.concat(discarded_parts, ignore_index=True).sort_values(
        ["symbol_norm", "timestamp_ns_utc"], kind="stable"
    ).reset_index(drop=True)
    discarded_hash = _canonical_discarded_key_hash(discarded_keys)
    if year == 2024 and enforce_frozen_2024_placeholder_audit:
        expected_cleanup = validate_contract()["feature_construction"][
            "provider_scan_and_factor_table"
        ]["fit_2024_placeholder_audit"]
        observed_cleanup = {
            "discarded_rows": len(discarded_keys),
            "all_four_OHLC_null_rows": len(discarded_keys),
            "partial_null_rows": 0,
            "canonical_discarded_key_sha256": discarded_hash,
            "per_symbol": discarded_per_symbol,
        }
        for key in (
            "discarded_rows",
            "all_four_OHLC_null_rows",
            "partial_null_rows",
            "canonical_discarded_key_sha256",
            "per_symbol",
        ):
            if observed_cleanup[key] != expected_cleanup[key]:
                raise AssertionError(
                    f"2024 placeholder cleanup audit changed: {key}"
                )
    canonical_hash = _canonical_factor_table_hash(panel)
    panel.attrs["discarded_placeholder_keys"] = discarded_keys
    irregular_audit: dict[str, Any] | None = None
    if year == 2025 and enforce_irregular_2025_audit:
        local = panel["timestamp"].dt.tz_convert("America/New_York")
        exact_time = (
            local.dt.hour.eq(9)
            & local.dt.minute.eq(33)
            & local.dt.second.eq(19)
        )
        on_date = panel["session_date"].eq(IRREGULAR_DATE)
        selected = panel.loc[on_date & exact_time]
        counts = selected.groupby("symbol_norm", sort=True).size().to_dict()
        expected_counts = {symbol: 1 for symbol in IRREGULAR_SYMBOLS}
        if counts != expected_counts:
            raise AssertionError(
                f"2025 irregular provider cohort changed: {counts}"
            )
        irregular_audit = {
            "date": IRREGULAR_DATE,
            "timestamp_new_york": "09:33:19",
            "provider_rows": len(selected),
            "per_symbol": counts,
            "no_other_symbol_at_exact_timestamp": True,
        }
    audit = {
        "year": year,
        "symbols": len(set(symbols)),
        "rows": len(panel),
        "minimum_timestamp": panel["timestamp"].min(),
        "maximum_timestamp": panel["timestamp"].max(),
        "canonical_table_sha256": canonical_hash,
        "provider_file_hashes": file_hashes,
        "placeholder_cleanup": {
            "discarded_rows": len(discarded_keys),
            "all_four_OHLC_null_rows": len(discarded_keys),
            "partial_null_rows": 0,
            "canonical_discarded_key_sha256": discarded_hash,
            "per_symbol": discarded_per_symbol,
        },
        "historical_volume_used": False,
        "volume_label": "historical_volume_not_used",
    }
    if irregular_audit is not None:
        audit["irregular_provider_cohort"] = irregular_audit
    return panel, canonical_hash, audit


def merge_entry_factors(runs: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    discarded = factors.attrs.get("discarded_placeholder_keys")
    discarded_intersection = 0
    if isinstance(discarded, pd.DataFrame) and len(discarded):
        discarded_keys = discarded.rename(columns={"timestamp": "start_timestamp"})
        discarded_keys["session_date"] = (
            pd.to_datetime(discarded_keys["start_timestamp"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
        )
        discarded_intersection = len(
            runs.loc[:, list(NATURAL_KEY)].merge(
                discarded_keys.loc[:, list(NATURAL_KEY)],
                on=list(NATURAL_KEY),
                how="inner",
            )
        )
        if discarded_intersection:
            raise AssertionError("a frozen run entry references an all-null placeholder")
    right = factors.rename(columns={"timestamp": "start_timestamp"})
    if right.duplicated(list(NATURAL_KEY)).any() or runs.duplicated(list(NATURAL_KEY)).any():
        raise AssertionError("duplicate natural key before factor merge")
    merged = runs.merge(
        right.drop(columns="timestamp_ns_utc"),
        on=list(NATURAL_KEY),
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(runs) or not merged["_merge"].eq("both").all():
        raise AssertionError("every run entry must have exactly one provider factor row")
    merged = merged.drop(columns="_merge")
    if not np.isfinite(merged.loc[:, list(NEW5)].to_numpy(float)).all():
        raise AssertionError("a declared price factor is nonfinite before imputation")
    timestamp = pd.to_datetime(merged["start_timestamp"], utc=True, errors="raise")
    local = timestamp.dt.tz_convert("America/New_York")
    seconds = (
        local.dt.hour.to_numpy(float) * 3600.0
        + local.dt.minute.to_numpy(float) * 60.0
        + local.dt.second.to_numpy(float)
        + local.dt.microsecond.to_numpy(float) / 1_000_000.0
        - 570.0 * 60.0
    )
    entry_minutes = seconds / 60.0
    if entry_minutes.min(initial=0.0) < 0.0 or entry_minutes.max(initial=0.0) >= 390.0:
        raise AssertionError("run entry lies outside the regular session")
    phase = 2.0 * np.pi * entry_minutes / 390.0
    merged["entry_minutes"] = entry_minutes
    merged["entry_time_sin"] = np.sin(phase)
    merged["entry_time_cos"] = np.cos(phase)
    raw_b0 = pd.to_numeric(merged["b0_state_numeric"], errors="coerce")
    merged["b0_unknown"] = raw_b0.isna()
    merged["b0_entry_numeric"] = raw_b0.fillna(0.0)
    merged["b0_entry_high_stress"] = pd.to_numeric(
        merged["b0_high_stress"], errors="coerce"
    ).fillna(0.0)
    if not np.isfinite(merged.loc[:, list(FULL9)].to_numpy(float)).all():
        raise AssertionError("nonfinite causal entry factor after declared imputation")
    merged.attrs["placeholder_run_entry_intersection"] = discarded_intersection
    return merged


def oriented_path_label(anchors: pd.DataFrame, path: tuple[int, ...]) -> np.ndarray:
    label = np.ones(len(anchors), dtype=bool)
    for step, destination in enumerate(path[1:], start=1):
        label &= anchors[f"future_state_{step}"].to_numpy(int) == int(destination)
    return label


def expand_compatible_labels(
    anchors: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_positives: int | None = None,
) -> pd.DataFrame:
    route_map = route_mapping(cycles)
    compatible_by_state = {
        state: sum(state in set(cycle.core) for cycle in cycles.itertuples(index=False))
        for state in range(K)
    }
    anchor_counts = anchors["state"].map(compatible_by_state).astype(int)
    if (anchor_counts <= 0).any():
        raise AssertionError("a state has no compatible frozen cycle")
    rows: list[pd.DataFrame] = []
    meta = [
        "anchor_id", "period", "symbol_norm", "session_date", "start_timestamp",
        "month", "quarter", "state", "history_token", "next_outcome", "terminal",
        "bar_ordinal", "b0_unknown", "entry_minutes",
        "future_state_1", "future_state_2", "future_state_3", "future_state_4",
        *FULL9,
    ]
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(state) for state in cycle.core)
        selected = anchors.loc[anchors["state"].isin(set(core)), meta].copy()
        target = np.zeros(len(selected), dtype=bool)
        for current in sorted(set(core)):
            mask = selected["state"].eq(current).to_numpy()
            state_anchors = selected.loc[mask]
            local = np.zeros(len(state_anchors), dtype=bool)
            for path in oriented_paths(core, current):
                local |= oriented_path_label(state_anchors, path)
            target[mask] = local
        selected["cycle_index"] = int(cycle.cycle_index)
        selected["cycle_id"] = str(cycle.cycle_id)
        selected["cycle"] = str(cycle.cycle)
        selected["transition_length"] = int(cycle.transition_length)
        selected["current_state"] = selected["state"].astype(int)
        selected["target"] = target.astype(np.int8)
        selected["compatible_cycle_count"] = selected["anchor_id"].map(anchor_counts)
        selected["inverse_compatible_weight"] = 1.0 / selected["compatible_cycle_count"]
        rows.append(selected)
    expanded = pd.concat(rows, ignore_index=True).sort_values(
        ["anchor_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    expanded = expanded.merge(
        route_map[["route_index", "cycle_index", "current_state"]],
        on=["cycle_index", "current_state"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if expanded["route_index"].isna().any() or expanded.duplicated(["anchor_id", "cycle_index"]).any():
        raise AssertionError("invalid compatible anchor-cycle expansion")
    expanded["route_index"] = expanded["route_index"].astype(int)
    if expected_rows is not None and len(expanded) != expected_rows:
        raise AssertionError("compatible anchor-cycle row count changed")
    if expected_positives is not None and int(expanded["target"].sum()) != expected_positives:
        raise AssertionError("compatible loop positive count changed")
    terminal = expanded["terminal"].to_numpy(bool)
    if expanded.loc[terminal, "target"].sum() != 0:
        raise AssertionError("terminal anchors must have zero compatible loop labels")
    return expanded


def prepare_period_anchors(
    run_path: Path,
    provider_root: Path,
    year: int,
    period: str,
    *,
    expected_run_sha256: str | None = None,
    expected_rows: int | None = None,
    expected_stocks: int | None = None,
    expected_compatible_rows: int | None = None,
    expected_positives: int | None = None,
    expected_provider_hashes: Mapping[str, str] | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> PreparedPeriod:
    if year != 2024:
        phase_source_paths("score", authorization)
    cycles = load_cycles()
    runs = load_runs(
        run_path,
        year,
        period,
        expected_sha256=expected_run_sha256,
        expected_rows=expected_rows,
        expected_stocks=expected_stocks,
    )
    factors, factor_hash, factor_audit = scan_provider_factors(
        sorted(runs["symbol_norm"].unique()),
        provider_root,
        year,
        expected_file_hashes=expected_provider_hashes,
    )
    if year == 2024:
        expected_factor_hash = validate_contract()["feature_construction"][
            "provider_scan_and_factor_table"
        ]["fit_2024_canonical_retained_factor_table_sha256"]
        if factor_hash != expected_factor_hash:
            raise AssertionError("canonical 2024 causal factor table hash changed")
    anchors = merge_entry_factors(runs, factors)
    expanded = expand_compatible_labels(
        anchors,
        cycles,
        expected_rows=expected_compatible_rows,
        expected_positives=expected_positives,
    )
    audit = {
        "period": period,
        "year": year,
        "run_rows": len(runs),
        "anchors": len(anchors),
        "compatible_rows": len(expanded),
        "positive_rows": int(expanded["target"].sum()),
        "terminal_run_entries": int(anchors["terminal"].sum()),
        "terminal_rows_in_fit_population": int(expanded["terminal"].sum()),
        "b0_unknown_run_entries": int(anchors["b0_unknown"].sum()),
        "natural_key_unmatched": 0,
        "factor_table": factor_audit,
        "later_period_paths_resolved": year != 2024,
        "later_period_rows_read": year != 2024,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        **SAFETY,
    }
    audit["factor_table"]["placeholder_cleanup"][
        "run_entry_natural_key_intersection"
    ] = int(anchors.attrs.get("placeholder_run_entry_intersection", 0))
    if year == 2024:
        expected_intersection = validate_contract()["feature_construction"][
            "provider_scan_and_factor_table"
        ]["fit_2024_placeholder_audit"]["run_entry_natural_key_intersection"]
        if (
            audit["factor_table"]["placeholder_cleanup"][
                "run_entry_natural_key_intersection"
            ]
            != expected_intersection
        ):
            raise AssertionError("2024 placeholder/run intersection changed")
    if year == 2025:
        audit["irregular_provider_cohort"] = dict(
            audit["factor_table"]["irregular_provider_cohort"]
        )
        affected_anchor = anchors["session_date"].eq(IRREGULAR_DATE) & anchors[
            "symbol_norm"
        ].isin(IRREGULAR_SYMBOLS)
        affected_expanded = expanded["session_date"].eq(IRREGULAR_DATE) & expanded[
            "symbol_norm"
        ].isin(IRREGULAR_SYMBOLS)
        irregular = {
            "symbol_date_cells": int(
                anchors.loc[affected_anchor, ["symbol_norm", "session_date"]]
                .drop_duplicates()
                .shape[0]
            ),
            "run_entries": int(affected_anchor.sum()),
            "compatible_rows": int(affected_expanded.sum()),
        }
        expected_irregular = validate_contract()["irregular_provider_cohort"]
        if irregular != {
            "symbol_date_cells": int(expected_irregular["expected_symbol_date_cells"]),
            "run_entries": int(expected_irregular["expected_run_entries"]),
            "compatible_rows": int(expected_irregular["expected_compatible_rows"]),
        }:
            raise AssertionError(f"2025 irregular run cohort changed: {irregular}")
        audit["irregular_provider_cohort"].update(irregular)
    return PreparedPeriod(anchors, expanded, cycles, route_mapping(cycles), factor_hash, audit)


def prepare_2024() -> PreparedPeriod:
    contract = validate_contract()
    validate_2024_source_integrity()
    source = contract["frozen_sources"]["runs_2024"]
    return prepare_period_anchors(
        RUN_2024,
        PROVIDER_ROOT_2024_2025,
        2024,
        "2024",
        expected_run_sha256=source["sha256"],
        expected_rows=int(source["rows"]),
        expected_stocks=int(source["stocks"]),
        expected_compatible_rows=int(contract["population_and_target"]["compatible_anchor_cycle_rows_expected"]["2024"]),
        expected_positives=int(contract["population_and_target"]["positive_rows_expected"]["2024"]),
        expected_provider_hashes=provider_hashes_for_year(2024),
    )


def _token_matrix(tokens: np.ndarray) -> sparse.csr_matrix:
    values = np.asarray(tokens, dtype=int)
    return sparse.csr_matrix(
        (
            np.ones(len(values), dtype=np.float32),
            (np.arange(len(values)), values),
        ),
        shape=(len(values), TOKEN_WIDTH),
        dtype=np.float32,
    )


def _raw_limited4(anchors: pd.DataFrame) -> np.ndarray:
    numeric = anchors.loc[:, list(LIMITED4)].apply(pd.to_numeric, errors="coerce")
    numeric["b0_entry_numeric"] = numeric["b0_entry_numeric"].fillna(0.0)
    numeric["b0_entry_high_stress"] = numeric["b0_entry_high_stress"].fillna(0.0)
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise AssertionError("nonfinite old-limited path factor")
    return numeric.to_numpy(dtype=np.float32)


def _fit_transition_kernel(
    anchors: pd.DataFrame, *, numeric: np.ndarray | None
) -> TransitionKernel:
    tokens = anchors["history_token"].to_numpy(int)
    matrix: sparse.csr_matrix = _token_matrix(tokens)
    numeric_width = 0
    if numeric is not None:
        numeric = np.asarray(numeric, dtype=np.float32)
        if numeric.shape != (len(anchors), len(LIMITED4)):
            raise AssertionError("old-limited transition numeric shape changed")
        matrix = sparse.hstack((matrix, sparse.csr_matrix(numeric)), format="csr", dtype=np.float32)
        numeric_width = numeric.shape[1]
    target = anchors["next_outcome"].to_numpy(int)
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=500,
        tol=0.0001,
        random_state=20260710,
    )
    model.fit(matrix, target)
    required = np.arange(DESTINATIONS)
    if not np.array_equal(model.classes_, required):
        raise AssertionError("fold-local transition kernel lacks a destination class")
    iterations = int(model.n_iter_[0])
    if iterations >= 500:
        raise AssertionError("fold-local transition kernel failed convergence")
    return TransitionKernel(
        classes=model.classes_.astype(np.int64, copy=True),
        coef=model.coef_.astype(np.float64, copy=True),
        intercept=model.intercept_.astype(np.float64, copy=True),
        numeric_width=numeric_width,
        n_iter=iterations,
    )


def fit_transition_kernels(
    train_anchors: pd.DataFrame,
) -> tuple[TransitionKernel, TransitionKernel]:
    """Fit the exact fold-local retained-history and rejected limited kernels."""

    history = _fit_transition_kernel(train_anchors, numeric=None)
    old_limited = _fit_transition_kernel(
        train_anchors, numeric=_raw_limited4(train_anchors)
    )
    return history, old_limited


def load_pinned_transition_kernels(
    path: Path = PATH_PARAMETERS,
) -> tuple[TransitionKernel, TransitionKernel]:
    contract = validate_contract()
    expected = contract["frozen_sources"]["retained_path_parameters"]["sha256"]
    if sha256(path) != expected:
        raise AssertionError("retained full-2024 transition parameters changed")
    with np.load(path) as stored:
        history = TransitionKernel(
            classes=stored["history_classes"].copy(),
            coef=stored["history_coef"].copy(),
            intercept=stored["history_intercept"].copy(),
            numeric_width=0,
            n_iter=int(stored["history_n_iter"][0]),
        )
        old_limited = TransitionKernel(
            classes=stored["context_classes"].copy(),
            coef=stored["context_coef"].copy(),
            intercept=stored["context_intercept"].copy(),
            numeric_width=len(LIMITED4),
            n_iter=int(stored["context_n_iter"][0]),
        )
    for kernel, width in ((history, TOKEN_WIDTH), (old_limited, TOKEN_WIDTH + len(LIMITED4))):
        if (
            not np.array_equal(kernel.classes, np.arange(DESTINATIONS))
            or kernel.coef.shape != (DESTINATIONS, width)
            or kernel.intercept.shape != (DESTINATIONS,)
        ):
            raise AssertionError("pinned transition parameter shape changed")
    return history, old_limited


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def destination_probability(
    kernel: TransitionKernel,
    previous_state_2: np.ndarray,
    previous_state_1: np.ndarray,
    current_state: np.ndarray,
    destination: int,
    numeric: np.ndarray | None = None,
) -> np.ndarray:
    tokens = history_tokens(previous_state_2, previous_state_1, current_state)
    logits = kernel.intercept[None, :] + kernel.coef[:, tokens].T
    if kernel.numeric_width:
        if numeric is None:
            raise AssertionError("numeric transition kernel requires limited4")
        values = np.asarray(numeric, dtype=np.float32)
        if values.shape != (len(tokens), kernel.numeric_width):
            raise AssertionError("numeric transition values have the wrong shape")
        logits = logits + values.astype(np.float64) @ kernel.coef[:, TOKEN_WIDTH:].T
    elif numeric is not None:
        raise AssertionError("history-only transition kernel received numeric values")
    column = int(np.flatnonzero(kernel.classes == int(destination))[0])
    return np.clip(_softmax(logits)[:, column], EPSILON, 1.0 - EPSILON)


def route_path_probability(
    anchors: pd.DataFrame,
    path: tuple[int, ...],
    kernel: TransitionKernel,
    *,
    repeat_limited4: bool,
) -> np.ndarray:
    probability = np.ones(len(anchors), dtype=np.float64)
    p2 = anchors["previous_state_2"].to_numpy(int)
    p1 = anchors["previous_state_1"].to_numpy(int)
    current = np.full(len(anchors), int(path[0]), dtype=int)
    numeric = _raw_limited4(anchors) if repeat_limited4 else None
    for destination in path[1:]:
        probability *= destination_probability(
            kernel, p2, p1, current, int(destination), numeric
        )
        p2, p1, current = (
            p1,
            current,
            np.full(len(anchors), int(destination), dtype=int),
        )
    return probability


def score_path_offsets(
    anchors: pd.DataFrame,
    expanded: pd.DataFrame,
    cycles: pd.DataFrame,
    history_kernel: TransitionKernel,
    old_limited_kernel: TransitionKernel,
) -> pd.DataFrame:
    if not np.array_equal(anchors["anchor_id"].to_numpy(int), np.arange(len(anchors))):
        raise AssertionError("anchors must retain contiguous positional anchor_id")
    anchor_ids = expanded["anchor_id"].to_numpy(int)
    if anchor_ids.min(initial=0) < 0 or anchor_ids.max(initial=0) >= len(anchors):
        raise AssertionError("expanded row references an unknown anchor")
    qhistory = np.zeros(len(expanded), dtype=np.float64)
    qold = np.zeros(len(expanded), dtype=np.float64)
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(state) for state in cycle.core)
        cycle_positions = np.flatnonzero(
            expanded["cycle_index"].to_numpy(int) == int(cycle.cycle_index)
        )
        current_values = expanded.iloc[cycle_positions]["current_state"].to_numpy(int)
        for current in sorted(set(core)):
            local_positions = cycle_positions[current_values == current]
            selected = anchors.iloc[expanded.iloc[local_positions]["anchor_id"].to_numpy(int)]
            history_values = np.zeros(len(selected), dtype=np.float64)
            old_values = np.zeros(len(selected), dtype=np.float64)
            for path in oriented_paths(core, current):
                history_values += route_path_probability(
                    selected, path, history_kernel, repeat_limited4=False
                )
                old_values += route_path_probability(
                    selected, path, old_limited_kernel, repeat_limited4=True
                )
            qhistory[local_positions] = history_values
            qold[local_positions] = old_values
    qhistory = np.clip(qhistory, EPSILON, 1.0 - EPSILON)
    qold = np.clip(qold, EPSILON, 1.0 - EPSILON)
    if not np.isfinite(qhistory).all() or not np.isfinite(qold).all():
        raise AssertionError("nonfinite path offset probability")
    return pd.DataFrame(
        {
            "qhistory": qhistory,
            "qold_limited_path": qold,
            "eta_history": np.log(qhistory) - np.log1p(-qhistory),
            "eta_old_limited_path": np.log(qold) - np.log1p(-qold),
        },
        index=expanded.index,
    )


def verify_pinned_later_offsets(
    expanded: pd.DataFrame,
    offsets: pd.DataFrame,
    period: str,
    guard_token: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify and substitute the exact retained later-period path baselines."""

    if period not in ("2025", "2023"):
        raise ValueError("pinned later offset verification is only for 2025/2023")
    paths = phase_source_paths("score", guard_token=guard_token)
    pinned = pd.read_parquet(
        paths[f"retained_scoring_{period}"],
        columns=[
            "anchor_id",
            "cycle_id",
            "target",
            "probability_history",
            "probability_context",
        ],
    ).sort_values(["anchor_id", "cycle_id"], kind="stable").reset_index(drop=True)
    observed = expanded.loc[:, ["anchor_id", "cycle_id", "target"]].copy()
    observed = observed.sort_values(["anchor_id", "cycle_id"], kind="stable")
    order = observed.index.to_numpy(int)
    observed = observed.reset_index(drop=True)
    if len(pinned) != len(observed):
        raise AssertionError("retained later path panel row count changed")
    if (
        not np.array_equal(pinned["anchor_id"].to_numpy(int), observed["anchor_id"].to_numpy(int))
        or not np.array_equal(pinned["cycle_id"].astype(str), observed["cycle_id"].astype(str))
        or not np.array_equal(pinned["target"].to_numpy(int), observed["target"].to_numpy(int))
    ):
        raise AssertionError("retained later path keys or labels changed")
    calculated_history = offsets.iloc[order]["qhistory"].to_numpy(float)
    calculated_old = offsets.iloc[order]["qold_limited_path"].to_numpy(float)
    pinned_history = pinned["probability_history"].to_numpy(float)
    pinned_old = pinned["probability_context"].to_numpy(float)
    errors = {
        "qhistory_max_abs_error": float(np.max(np.abs(calculated_history - pinned_history))),
        "qold_limited_path_max_abs_error": float(np.max(np.abs(calculated_old - pinned_old))),
    }
    if any(value > tolerance for value in errors.values()):
        raise AssertionError(f"pinned later path replay failed: {errors}")
    replacement = offsets.copy()
    replacement.iloc[order, replacement.columns.get_loc("qhistory")] = pinned_history
    replacement.iloc[order, replacement.columns.get_loc("qold_limited_path")] = pinned_old
    replacement["eta_history"] = np.log(replacement["qhistory"]) - np.log1p(-replacement["qhistory"])
    replacement["eta_old_limited_path"] = np.log(replacement["qold_limited_path"]) - np.log1p(-replacement["qold_limited_path"])
    return replacement, {
        "period": period,
        "rows": len(pinned),
        "tolerance": tolerance,
        **errors,
        "pinned_probabilities_substituted_before_candidate_scoring": True,
    }


def _numeric_factor_frame(anchors: pd.DataFrame) -> pd.DataFrame:
    frame = anchors.loc[:, list(FULL9)].apply(pd.to_numeric, errors="coerce")
    frame["b0_entry_numeric"] = frame["b0_entry_numeric"].fillna(0.0)
    frame["b0_entry_high_stress"] = frame["b0_entry_high_stress"].fillna(0.0)
    if not np.isfinite(frame.loc[:, list(NEW5)].to_numpy(float)).all():
        raise AssertionError("a price factor is nonfinite before imputation")
    return frame


def fit_numeric_transform(train_anchors: pd.DataFrame) -> NumericTransform:
    frame = _numeric_factor_frame(train_anchors)
    values = frame.to_numpy(dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        raise AssertionError("training-prefix factor median is nonfinite")
    filled = np.where(np.isfinite(values), values, medians[None, :])
    scales = np.sqrt(np.mean(np.square(filled - medians[None, :]), axis=0))
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
    return NumericTransform(tuple(FULL9), medians, scales)


def transform_numeric(
    anchors: pd.DataFrame, transform: NumericTransform
) -> np.ndarray:
    if transform.columns != tuple(FULL9):
        raise AssertionError("numeric transform column order changed")
    frame = _numeric_factor_frame(anchors)
    values = frame.to_numpy(dtype=np.float64)
    filled = np.where(
        np.isfinite(values), values, np.asarray(transform.medians)[None, :]
    )
    output = (filled - np.asarray(transform.medians)[None, :]) / np.asarray(
        transform.scales
    )[None, :]
    if output.shape != (len(anchors), len(FULL9)) or not np.isfinite(output).all():
        raise AssertionError("invalid transformed factor matrix")
    return output


def fit_contrast_spec(
    train_expanded: pd.DataFrame, route_map: pd.DataFrame
) -> ContrastSpec:
    weights = train_expanded["inverse_compatible_weight"].to_numpy(np.float64)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise AssertionError("invalid inverse-compatible training weight")
    cycles = train_expanded["cycle_index"].to_numpy(int)
    routes = train_expanded["route_index"].to_numpy(int)
    cycle_weights = np.bincount(cycles, weights=weights, minlength=CYCLE_COUNT)
    route_weights = np.bincount(routes, weights=weights, minlength=ROUTE_COUNT)
    if (cycle_weights <= 0.0).any() or (route_weights <= 0.0).any():
        raise AssertionError("every cycle and compatible route requires training-prefix support")
    references = (
        route_map.loc[route_map["is_reference"]]
        .sort_values("cycle_index", kind="stable")["route_index"]
        .to_numpy(int)
    )
    if len(references) != CYCLE_COUNT:
        raise AssertionError("each cycle must have one route reference")
    return ContrastSpec(
        cycle_weights=cycle_weights.astype(np.float64),
        route_weights=route_weights.astype(np.float64),
        cycle_reference=CYCLE_COUNT - 1,
        route_reference_by_cycle=references,
    )


def contrast_blocks(
    expanded: pd.DataFrame,
    spec: ContrastSpec,
    route_map: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    count = len(expanded)
    cycles = expanded["cycle_index"].to_numpy(int)
    routes = expanded["route_index"].to_numpy(int)
    cycle_rows: list[np.ndarray] = []
    cycle_columns: list[np.ndarray] = []
    cycle_data: list[np.ndarray] = []
    for level in range(CYCLE_CONTRAST_WIDTH):
        direct = np.flatnonzero(cycles == level)
        reference = np.flatnonzero(cycles == spec.cycle_reference)
        if len(direct):
            cycle_rows.append(direct)
            cycle_columns.append(np.full(len(direct), level, dtype=int))
            cycle_data.append(np.ones(len(direct), dtype=np.float64))
        if len(reference):
            ratio = -spec.cycle_weights[level] / spec.cycle_weights[spec.cycle_reference]
            cycle_rows.append(reference)
            cycle_columns.append(np.full(len(reference), level, dtype=int))
            cycle_data.append(np.full(len(reference), ratio, dtype=np.float64))
    cycle_block = sparse.coo_matrix(
        (
            np.concatenate(cycle_data),
            (np.concatenate(cycle_rows), np.concatenate(cycle_columns)),
        ),
        shape=(count, CYCLE_CONTRAST_WIDTH),
        dtype=np.float64,
    ).tocsr()

    nonreference = route_map.loc[~route_map["is_reference"]].sort_values(
        "route_index", kind="stable"
    )
    if len(nonreference) != ROUTE_CONTRAST_WIDTH:
        raise AssertionError("route contrast mapping changed")
    route_rows: list[np.ndarray] = []
    route_columns: list[np.ndarray] = []
    route_data: list[np.ndarray] = []
    for column, mapping in enumerate(nonreference.itertuples(index=False)):
        direct = np.flatnonzero(routes == int(mapping.route_index))
        reference_route = int(spec.route_reference_by_cycle[int(mapping.cycle_index)])
        reference = np.flatnonzero(routes == reference_route)
        if len(direct):
            route_rows.append(direct)
            route_columns.append(np.full(len(direct), column, dtype=int))
            route_data.append(np.ones(len(direct), dtype=np.float64))
        if len(reference):
            ratio = -spec.route_weights[int(mapping.route_index)] / spec.route_weights[reference_route]
            route_rows.append(reference)
            route_columns.append(np.full(len(reference), column, dtype=int))
            route_data.append(np.full(len(reference), ratio, dtype=np.float64))
    route_block = sparse.coo_matrix(
        (
            np.concatenate(route_data),
            (np.concatenate(route_rows), np.concatenate(route_columns)),
        ),
        shape=(count, ROUTE_CONTRAST_WIDTH),
        dtype=np.float64,
    ).tocsr()
    return cycle_block, route_block


def token_support_mask(
    train_anchors: pd.DataFrame, train_expanded: pd.DataFrame
) -> np.ndarray:
    token = train_anchors["history_token"].to_numpy(int)
    anchor_counts = np.bincount(token, minlength=TOKEN_WIDTH)
    positive_counts = np.bincount(
        train_expanded["history_token"].to_numpy(int),
        weights=train_expanded["target"].to_numpy(float),
        minlength=TOKEN_WIDTH,
    )
    return ((anchor_counts >= 200) & (positive_counts >= 50.0)).astype(bool)


def fit_quartile_cutpoints(train_anchors: pd.DataFrame) -> np.ndarray:
    values = train_anchors.loc[:, list(QUARTILE_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("quartile input is nonfinite")
    cutpoints = np.quantile(
        values,
        np.asarray([0.25, 0.5, 0.75]),
        axis=0,
        method="linear",
    ).T
    if cutpoints.shape != (len(QUARTILE_COLUMNS), 3):
        raise AssertionError("quartile cutpoint shape changed")
    return cutpoints


def apply_quartile_cutpoints(
    anchors: pd.DataFrame, cutpoints: np.ndarray
) -> dict[str, np.ndarray]:
    points = np.asarray(cutpoints, dtype=np.float64)
    if points.shape != (len(QUARTILE_COLUMNS), 3):
        raise AssertionError("quartile cutpoint shape changed")
    output: dict[str, np.ndarray] = {}
    for index, column in enumerate(QUARTILE_COLUMNS):
        values = pd.to_numeric(anchors[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise AssertionError(f"quartile scoring input is nonfinite: {column}")
        output[column] = np.searchsorted(points[index], values, side="right").astype(np.int8)
    return output


def _token_factor_block(
    tokens: np.ndarray, values: np.ndarray, token_mask: np.ndarray
) -> sparse.csr_matrix:
    supported = np.asarray(token_mask, bool)[tokens]
    rows = np.flatnonzero(supported & (values != 0.0))
    return sparse.csr_matrix(
        (values[rows], (rows, tokens[rows])),
        shape=(len(tokens), TOKEN_WIDTH),
        dtype=np.float64,
    )


def build_design(
    expanded: pd.DataFrame,
    anchor_factors_z: np.ndarray,
    factor_count: int,
    contrast_spec: ContrastSpec,
    route_map: pd.DataFrame,
    token_mask: np.ndarray,
) -> sparse.csr_matrix:
    if factor_count not in (0, 4, 9):
        raise ValueError("factor_count must be 0, 4, or 9")
    factors = np.asarray(anchor_factors_z, dtype=np.float64)
    if factors.ndim != 2 or factors.shape[1] != len(FULL9):
        raise AssertionError("anchor factor matrix must have the full9 column order")
    anchor_ids = expanded["anchor_id"].to_numpy(int)
    if anchor_ids.max(initial=0) >= len(factors):
        raise AssertionError("expanded row has no anchor factor vector")
    row_factors = factors[anchor_ids, :factor_count]
    cycle_block, route_block = contrast_blocks(expanded, contrast_spec, route_map)
    structural = sparse.hstack(
        (
            sparse.csr_matrix(np.ones((len(expanded), 1), dtype=np.float64)),
            cycle_block,
            route_block,
        ),
        format="csr",
        dtype=np.float64,
    )
    if structural.shape[1] != PATTERN_WIDTH:
        raise AssertionError("qpattern width changed")
    if factor_count == 0:
        return structural
    blocks: list[sparse.spmatrix] = [
        structural,
        sparse.csr_matrix(row_factors, dtype=np.float64),
    ]
    blocks.extend(
        cycle_block.multiply(row_factors[:, factor][:, None])
        for factor in range(factor_count)
    )
    blocks.extend(
        route_block.multiply(row_factors[:, factor][:, None])
        for factor in range(factor_count)
    )
    tokens = expanded["history_token"].to_numpy(int)
    blocks.extend(
        _token_factor_block(tokens, row_factors[:, factor], token_mask)
        for factor in range(factor_count)
    )
    design = sparse.hstack(blocks, format="csr", dtype=np.float64)
    expected = LIMITED_WIDTH if factor_count == 4 else FULL_WIDTH
    if design.shape != (len(expanded), expected):
        raise AssertionError(f"direct residual design width changed: {design.shape}")
    if design.dtype != np.float64 or not np.isfinite(design.data).all():
        raise AssertionError("direct residual design must be finite float64 CSR")
    return design


def build_all_designs(
    expanded: pd.DataFrame,
    anchor_factors_z: np.ndarray,
    contrast_spec: ContrastSpec,
    route_map: pd.DataFrame,
    token_mask: np.ndarray,
) -> dict[str, sparse.csr_matrix]:
    return {
        "qpattern": build_design(expanded, anchor_factors_z, 0, contrast_spec, route_map, token_mask),
        "qlimited4": build_design(expanded, anchor_factors_z, 4, contrast_spec, route_map, token_mask),
        "qfull9": build_design(expanded, anchor_factors_z, 9, contrast_spec, route_map, token_mask),
    }


def penalty_multipliers(factor_count: int) -> np.ndarray:
    if factor_count not in (0, 4, 9):
        raise ValueError("factor_count must be 0, 4, or 9")
    values = [0.0, *([4.0] * CYCLE_CONTRAST_WIDTH), *([8.0] * ROUTE_CONTRAST_WIDTH)]
    if factor_count:
        values.extend([1.0] * factor_count)
        values.extend([4.0] * (factor_count * CYCLE_CONTRAST_WIDTH))
        values.extend([8.0] * (factor_count * ROUTE_CONTRAST_WIDTH))
        values.extend([32.0] * (factor_count * TOKEN_WIDTH))
    result = np.asarray(values, dtype=np.float64)
    expected = {0: PATTERN_WIDTH, 4: LIMITED_WIDTH, 9: FULL_WIDTH}[factor_count]
    if len(result) != expected:
        raise AssertionError("penalty-vector width changed")
    return result


def offset_ridge_objective_gradient(
    coefficients: np.ndarray,
    design: sparse.csr_matrix,
    target: np.ndarray,
    offset: np.ndarray,
    ridge_lambda: float,
    penalty: np.ndarray,
) -> tuple[float, np.ndarray]:
    matrix = sparse.csr_matrix(design, dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    eta0 = np.asarray(offset, dtype=np.float64)
    multipliers = np.asarray(penalty, dtype=np.float64)
    if matrix.shape != (len(y), len(beta)) or eta0.shape != y.shape:
        raise AssertionError("offset-ridge objective shapes differ")
    if multipliers.shape != beta.shape or not np.isfinite(multipliers).all():
        raise AssertionError("offset-ridge penalty shape changed")
    eta = eta0 + matrix @ beta
    loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    gradient = np.asarray(matrix.T @ (expit(eta) - y)).ravel() / float(len(y))
    loss += 0.5 * float(ridge_lambda) * float(
        np.dot(multipliers, np.square(beta))
    )
    gradient += float(ridge_lambda) * multipliers * beta
    return loss, gradient


def fit_offset_ridge(
    design: sparse.csr_matrix,
    target: np.ndarray,
    offset: np.ndarray,
    ridge_lambda: float,
    penalty: np.ndarray | None = None,
    *,
    factor_count: int | None = None,
) -> ResidualFit:
    matrix = sparse.csr_matrix(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    eta0 = np.asarray(offset, dtype=np.float64)
    if matrix.shape[0] != len(y) or y.shape != eta0.shape:
        raise AssertionError("offset-ridge arrays differ in length")
    if not np.isfinite(matrix.data).all() or not np.isfinite(y).all() or not np.isfinite(eta0).all():
        raise AssertionError("offset-ridge input is nonfinite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise AssertionError("offset-ridge target must be binary")
    if ridge_lambda not in LAMBDA_GRID:
        raise AssertionError("ridge lambda is outside the frozen grid")
    if factor_count is None:
        factor_count = {PATTERN_WIDTH: 0, LIMITED_WIDTH: 4, FULL_WIDTH: 9}.get(matrix.shape[1])
    if factor_count not in (0, 4, 9):
        raise AssertionError("cannot infer residual-head factor count")
    multipliers = penalty_multipliers(factor_count) if penalty is None else np.asarray(penalty, float)
    if multipliers.shape != (matrix.shape[1],):
        raise AssertionError("ridge penalty multiplier shape changed")
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return offset_ridge_objective_gradient(
            beta, matrix, y, eta0, ridge_lambda, multipliers
        )

    result = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    final_objective, final_gradient = objective(np.asarray(result.x, float))
    if (
        not bool(result.success)
        or int(result.nit) >= 1000
        or not math.isfinite(final_objective)
        or not np.isfinite(final_gradient).all()
        or not np.isfinite(result.x).all()
    ):
        raise RuntimeError(
            f"offset-ridge optimizer failed: success={result.success}, nit={result.nit}, message={result.message}"
        )
    return ResidualFit(
        coefficients=np.asarray(result.x, dtype=np.float64),
        ridge_lambda=float(ridge_lambda),
        factor_count=int(factor_count),
        objective=final_objective,
        gradient_max_abs=float(np.max(np.abs(final_gradient))),
        iterations=int(result.nit),
        optimizer_message=str(result.message),
    )


def predict_offset_ridge(
    design: sparse.csr_matrix,
    offset: np.ndarray,
    fit: ResidualFit | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = fit.coefficients if isinstance(fit, ResidualFit) else np.asarray(fit, float)
    eta = np.asarray(offset, float) + sparse.csr_matrix(design, dtype=np.float64) @ coefficients
    probability = np.clip(expit(eta), EPSILON, 1.0 - EPSILON)
    if not np.isfinite(probability).all():
        raise AssertionError("nonfinite residual probability")
    return probability, np.asarray(eta, dtype=np.float64)


def _factor_layout(factor_count: int) -> dict[str, Any]:
    start = PATTERN_WIDTH
    global_slice = slice(start, start + factor_count)
    start = global_slice.stop
    cycle_slices = [
        slice(start + factor * CYCLE_CONTRAST_WIDTH, start + (factor + 1) * CYCLE_CONTRAST_WIDTH)
        for factor in range(factor_count)
    ]
    start += factor_count * CYCLE_CONTRAST_WIDTH
    route_slices = [
        slice(start + factor * ROUTE_CONTRAST_WIDTH, start + (factor + 1) * ROUTE_CONTRAST_WIDTH)
        for factor in range(factor_count)
    ]
    start += factor_count * ROUTE_CONTRAST_WIDTH
    token_slices = [
        slice(start + factor * TOKEN_WIDTH, start + (factor + 1) * TOKEN_WIDTH)
        for factor in range(factor_count)
    ]
    start += factor_count * TOKEN_WIDTH
    return {
        "structural": slice(0, PATTERN_WIDTH),
        "global": global_slice,
        "cycle": cycle_slices,
        "route": route_slices,
        "token": token_slices,
        "width": start,
    }


def embed_coefficients(
    coefficients: np.ndarray, source_factor_count: int, target_factor_count: int
) -> np.ndarray:
    if source_factor_count not in (0, 4, 9) or target_factor_count not in (0, 4, 9):
        raise ValueError("factor count must be 0, 4, or 9")
    if source_factor_count > target_factor_count:
        raise ValueError("embedding may only add zero factor coordinates")
    source = _factor_layout(source_factor_count)
    target = _factor_layout(target_factor_count)
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (source["width"],):
        raise AssertionError("source coefficient width changed")
    output = np.zeros(target["width"], dtype=np.float64)
    output[target["structural"]] = values[source["structural"]]
    if source_factor_count:
        output[target["global"].start : target["global"].start + source_factor_count] = values[source["global"]]
        for block in ("cycle", "route", "token"):
            for factor in range(source_factor_count):
                output[target[block][factor]] = values[source[block][factor]]
    return output


def embedding_invariants(
    designs: Mapping[str, sparse.csr_matrix],
    offset: np.ndarray,
    *,
    pattern_coefficients: np.ndarray | None = None,
    limited_coefficients: np.ndarray | None = None,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    history_probability = np.clip(expit(np.asarray(offset, float)), EPSILON, 1.0 - EPSILON)
    zero_probability = np.clip(expit(np.asarray(offset, float) + designs["qfull9"] @ np.zeros(FULL_WIDTH)), EPSILON, 1.0 - EPSILON)
    errors = {"zero_residual_to_history": float(np.max(np.abs(zero_probability - history_probability)))}
    if pattern_coefficients is not None:
        base = np.asarray(offset, float) + designs["qpattern"] @ np.asarray(pattern_coefficients, float)
        for name, factors in (("qlimited4", 4), ("qfull9", 9)):
            replay = np.asarray(offset, float) + designs[name] @ embed_coefficients(pattern_coefficients, 0, factors)
            errors[f"pattern_to_{name}"] = float(np.max(np.abs(replay - base)))
    if limited_coefficients is not None:
        base = np.asarray(offset, float) + designs["qlimited4"] @ np.asarray(limited_coefficients, float)
        replay = np.asarray(offset, float) + designs["qfull9"] @ embed_coefficients(limited_coefficients, 4, 9)
        errors["limited4_to_full9"] = float(np.max(np.abs(replay - base)))
    if any(error > tolerance for error in errors.values()):
        raise AssertionError(f"zero-block embedding invariant failed: {errors}")
    return errors


def subset_anchors_expanded(
    anchors: pd.DataFrame,
    expanded: pd.DataFrame,
    anchor_mask: np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = np.asarray(anchor_mask, dtype=bool)
    if mask.shape != (len(anchors),):
        raise AssertionError("anchor subset mask has the wrong length")
    old_ids = anchors.loc[mask, "anchor_id"].to_numpy(int)
    source_ids_all = (
        anchors["source_anchor_id"].to_numpy(int)
        if "source_anchor_id" in anchors
        else anchors["anchor_id"].to_numpy(int)
    )
    mapping = np.full(len(anchors), -1, dtype=np.int64)
    mapping[old_ids] = np.arange(len(old_ids), dtype=np.int64)
    selected_anchors = anchors.loc[mask].copy().reset_index(drop=True)
    selected_anchors["source_anchor_id"] = source_ids_all[old_ids]
    selected_anchors["anchor_id"] = np.arange(len(selected_anchors), dtype=np.int64)
    selected_rows = expanded["anchor_id"].isin(set(old_ids)).to_numpy()
    selected_expanded = expanded.loc[selected_rows].copy()
    selected_expanded["source_anchor_id"] = source_ids_all[
        selected_expanded["anchor_id"].to_numpy(int)
    ]
    selected_expanded["anchor_id"] = mapping[selected_expanded["anchor_id"].to_numpy(int)]
    selected_expanded = selected_expanded.sort_values(
        ["anchor_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    if (selected_expanded["anchor_id"] < 0).any():
        raise AssertionError("anchor remapping failed")
    return selected_anchors, selected_expanded


def _head_factor_count(head: str) -> int:
    mapping = {"qpattern": 0, "qlimited4": 4, "qfull9": 9}
    if head not in mapping:
        raise KeyError(f"unknown residual head: {head}")
    return mapping[head]


def binary_log_loss(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _fit_selected_heads_low_level(
    anchors: pd.DataFrame,
    expanded: pd.DataFrame,
    cycles: pd.DataFrame,
    selected_lambdas: Mapping[str, float],
    *,
    transition_kernels: tuple[TransitionKernel, TransitionKernel] | None = None,
) -> tuple[ModelBundle, dict[str, Any], dict[str, Any]]:
    required_heads = ("qpattern", "qlimited4", "qfull9")
    if set(selected_lambdas) != set(required_heads):
        raise AssertionError("selected lambdas must cover exactly three residual heads")
    history, old_limited = (
        fit_transition_kernels(anchors)
        if transition_kernels is None
        else transition_kernels
    )
    offsets = score_path_offsets(anchors, expanded, cycles, history, old_limited)
    transform = fit_numeric_transform(anchors)
    factors = transform_numeric(anchors, transform)
    route_map = route_mapping(cycles)
    contrast = fit_contrast_spec(expanded, route_map)
    support_mask = token_support_mask(anchors, expanded)
    designs = build_all_designs(expanded, factors, contrast, route_map, support_mask)
    target = expanded["target"].to_numpy(int)
    offset = offsets["eta_history"].to_numpy(float)
    fits: dict[str, ResidualFit] = {}
    for head in required_heads:
        fits[head] = fit_offset_ridge(
            designs[head],
            target,
            offset,
            float(selected_lambdas[head]),
            factor_count=_head_factor_count(head),
        )
    invariants = embedding_invariants(
        designs,
        offset,
        pattern_coefficients=fits["qpattern"].coefficients,
        limited_coefficients=fits["qlimited4"].coefficients,
    )
    bundle = ModelBundle(
        history_kernel=history,
        old_limited_kernel=old_limited,
        numeric_transform=transform,
        contrast_spec=contrast,
        token_mask=support_mask,
        quartile_cutpoints=fit_quartile_cutpoints(anchors),
        qpattern=fits["qpattern"],
        qlimited4=fits["qlimited4"],
        qfull9=fits["qfull9"],
    )
    audit = {
        "anchors": len(anchors),
        "compatible_rows": len(expanded),
        "positives": int(target.sum()),
        "terminal_anchors_included": int(anchors["terminal"].sum()),
        "selected_lambdas": {key: float(value) for key, value in selected_lambdas.items()},
        "history_iterations": history.n_iter,
        "old_limited_iterations": old_limited.n_iter,
        "supported_tokens": int(support_mask.sum()),
        "numeric_medians": transform.medians,
        "numeric_scales": transform.scales,
        "embedding_invariants": invariants,
        "optimizer": {
            head: {
                "objective": fit.objective,
                "gradient_max_abs": fit.gradient_max_abs,
                "iterations": fit.iterations,
                "message": fit.optimizer_message,
            }
            for head, fit in fits.items()
        },
        **SAFETY,
    }
    replay_inputs = design_replay_inputs(
        designs,
        target,
        offset,
        expanded["anchor_id"].to_numpy(int),
        expanded["cycle_index"].to_numpy(int),
    )
    return bundle, audit, replay_inputs


def predict_bundle(
    bundle: ModelBundle,
    anchors: pd.DataFrame | PreparedPeriod,
    expanded: pd.DataFrame | None = None,
    cycles: pd.DataFrame | None = None,
    *,
    period: str | None = None,
    phase_guard: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if isinstance(anchors, PreparedPeriod):
        prepared = anchors
        anchors = prepared.anchors
        expanded = prepared.expanded if expanded is None else expanded
        cycles = prepared.cycles if cycles is None else cycles
    if expanded is None or cycles is None:
        raise ValueError("expanded labels and cycles are required")
    if period in ("2025", "2023"):
        phase_source_paths("score", guard_token=phase_guard)
    offsets = score_path_offsets(
        anchors,
        expanded,
        cycles,
        bundle.history_kernel,
        bundle.old_limited_kernel,
    )
    pinned_replay: dict[str, Any] | None = None
    if period in ("2025", "2023"):
        if phase_guard is None:
            raise PermissionError("later prediction requires a scoring guard")
        offsets, pinned_replay = verify_pinned_later_offsets(
            expanded, offsets, period, phase_guard
        )
    factors = transform_numeric(anchors, bundle.numeric_transform)
    designs = build_all_designs(
        expanded,
        factors,
        bundle.contrast_spec,
        route_mapping(cycles),
        bundle.token_mask,
    )
    output = expanded.copy()
    output["n_compatible"] = output["compatible_cycle_count"].astype(int)
    quartiles = apply_quartile_cutpoints(anchors, bundle.quartile_cutpoints)
    row_anchor = output["anchor_id"].to_numpy(int)
    output["entry_clock_quartile"] = quartiles["entry_minutes"][row_anchor]
    for factor in NEW5:
        output[f"factor_quartile__{factor}"] = quartiles[factor][row_anchor]
    for column in offsets:
        output[column] = offsets[column].to_numpy()
    for head in ("qpattern", "qlimited4", "qfull9"):
        probability, eta = predict_offset_ridge(
            designs[head], offsets["eta_history"].to_numpy(float), getattr(bundle, head)
        )
        output[head] = probability
        output[f"eta_{head}"] = eta
    if "source_anchor_id" in output:
        output["local_anchor_id"] = output["anchor_id"].astype(int)
        output["anchor_id"] = output["source_anchor_id"].astype(int)
    output.attrs["embedding_invariants"] = embedding_invariants(
        designs,
        offsets["eta_history"].to_numpy(float),
        pattern_coefficients=bundle.qpattern.coefficients,
        limited_coefficients=bundle.qlimited4.coefficients,
    )
    if pinned_replay is not None:
        output.attrs["pinned_path_replay"] = pinned_replay
    return output


def fit_full_bundle(
    prepared: PreparedPeriod,
    expanded: pd.DataFrame | Mapping[str, float] | None = None,
    cycles: pd.DataFrame | None = None,
    selected_lambdas: Mapping[str, float] | None = None,
    *,
    transition_kernels: tuple[TransitionKernel, TransitionKernel] | None = None,
    use_pinned_offsets: bool = True,
) -> dict[str, Any]:
    if selected_lambdas is None and isinstance(expanded, Mapping):
        selected_lambdas = expanded
        expanded = None
    if selected_lambdas is None:
        raise ValueError("selected_lambdas are required")
    source_expanded = prepared.expanded if expanded is None else expanded
    source_cycles = prepared.cycles if cycles is None else cycles
    kernels = transition_kernels
    if kernels is None and use_pinned_offsets:
        kernels = load_pinned_transition_kernels()
    bundle, audit, replay_inputs = _fit_selected_heads_low_level(
        prepared.anchors,
        source_expanded,
        source_cycles,
        selected_lambdas,
        transition_kernels=kernels,
    )
    audit["full_2024_pinned_offsets"] = bool(
        transition_kernels is None and use_pinned_offsets
    )
    return {"bundle": bundle, "audit": audit, "replay_inputs": replay_inputs}


def fit_predict_fold(
    prepared: PreparedPeriod,
    expanded: pd.DataFrame | Mapping[str, float] | None = None,
    cycles: pd.DataFrame | None = None,
    selected_lambdas: Mapping[str, float] | None = None,
    *,
    validation_month: str,
) -> dict[str, Any]:
    if selected_lambdas is None and isinstance(expanded, Mapping):
        selected_lambdas = expanded
        expanded = None
    if selected_lambdas is None:
        raise ValueError("selected_lambdas are required")
    source_expanded = prepared.expanded if expanded is None else expanded
    source_cycles = prepared.cycles if cycles is None else cycles
    train_mask = prepared.anchors["month"].astype(str).lt(validation_month).to_numpy()
    validation_mask = prepared.anchors["month"].astype(str).eq(validation_month).to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise AssertionError("causal fold has an empty train or validation side")
    train_anchors, train_expanded = subset_anchors_expanded(
        prepared.anchors, source_expanded, train_mask
    )
    validation_anchors, validation_expanded = subset_anchors_expanded(
        prepared.anchors, source_expanded, validation_mask
    )
    bundle, audit, replay_inputs = _fit_selected_heads_low_level(
        train_anchors,
        train_expanded,
        source_cycles,
        selected_lambdas,
    )
    predictions = predict_bundle(
        bundle, validation_anchors, validation_expanded, source_cycles
    )
    audit.update(
        {
            "validation_month": validation_month,
            "validation_anchors": len(validation_anchors),
            "validation_rows": len(validation_expanded),
            "session_overlap": bool(
                set(zip(train_anchors["symbol_norm"], train_anchors["session_date"]))
                & set(zip(validation_anchors["symbol_norm"], validation_anchors["session_date"]))
            ),
        }
    )
    if audit["session_overlap"]:
        raise AssertionError("whole-session causal fold separation failed")
    return {
        "predictions": predictions,
        "bundle": bundle,
        "audit": audit,
        "replay_inputs": replay_inputs,
    }


def score_lambda_grid(
    prepared: PreparedPeriod,
    expanded: pd.DataFrame | None = None,
    cycles: pd.DataFrame | None = None,
    validation_month: str | None = None,
    lambdas: Sequence[float] = LAMBDA_GRID,
    heads: Sequence[str] = ("qpattern", "qlimited4", "qfull9"),
) -> dict[str, Any]:
    if validation_month is None:
        raise ValueError("validation_month is required")
    source_expanded = prepared.expanded if expanded is None else expanded
    source_cycles = prepared.cycles if cycles is None else cycles
    train_mask = prepared.anchors["month"].astype(str).lt(validation_month).to_numpy()
    validation_mask = prepared.anchors["month"].astype(str).eq(validation_month).to_numpy()
    train_anchors, train_expanded = subset_anchors_expanded(
        prepared.anchors, source_expanded, train_mask
    )
    validation_anchors, validation_expanded = subset_anchors_expanded(
        prepared.anchors, source_expanded, validation_mask
    )
    history, old_limited = fit_transition_kernels(train_anchors)
    train_offsets = score_path_offsets(
        train_anchors, train_expanded, source_cycles, history, old_limited
    )
    validation_offsets = score_path_offsets(
        validation_anchors, validation_expanded, source_cycles, history, old_limited
    )
    transform = fit_numeric_transform(train_anchors)
    contrast = fit_contrast_spec(train_expanded, route_mapping(source_cycles))
    mask = token_support_mask(train_anchors, train_expanded)
    train_designs = build_all_designs(
        train_expanded,
        transform_numeric(train_anchors, transform),
        contrast,
        route_mapping(source_cycles),
        mask,
    )
    validation_designs = build_all_designs(
        validation_expanded,
        transform_numeric(validation_anchors, transform),
        contrast,
        route_mapping(source_cycles),
        mask,
    )
    train_target = train_expanded["target"].to_numpy(int)
    validation_target = validation_expanded["target"].to_numpy(int)
    rows: list[dict[str, Any]] = []
    fits: dict[tuple[str, float], ResidualFit] = {}
    for head in heads:
        factor_count = _head_factor_count(head)
        for ridge_lambda in lambdas:
            fit = fit_offset_ridge(
                train_designs[head],
                train_target,
                train_offsets["eta_history"].to_numpy(float),
                float(ridge_lambda),
                factor_count=factor_count,
            )
            probability, _ = predict_offset_ridge(
                validation_designs[head],
                validation_offsets["eta_history"].to_numpy(float),
                fit,
            )
            rows.append(
                {
                    "validation_month": validation_month,
                    "head": head,
                    "lambda": float(ridge_lambda),
                    "validation_rows": len(validation_expanded),
                    "validation_positives": int(validation_target.sum()),
                    "log_loss": float(binary_log_loss(validation_target, probability).mean()),
                    "optimizer_objective": fit.objective,
                    "optimizer_gradient_max_abs": fit.gradient_max_abs,
                    "optimizer_iterations": fit.iterations,
                }
            )
            fits[(head, float(ridge_lambda))] = fit
    return {
        "scores": pd.DataFrame(rows),
        "cache": {
            "history_kernel": history,
            "old_limited_kernel": old_limited,
            "numeric_transform": transform,
            "contrast_spec": contrast,
            "token_mask": mask,
            "fits": fits,
        },
        "replay_inputs": design_replay_inputs(
            train_designs,
            train_target,
            train_offsets["eta_history"].to_numpy(float),
            train_expanded["anchor_id"].to_numpy(int),
            train_expanded["cycle_index"].to_numpy(int),
        ),
    }


def choose_lambda(scores: pd.DataFrame, head: str) -> float:
    selected = scores.loc[scores["head"].eq(head)].groupby(
        "lambda", sort=True
    )["log_loss"].mean()
    if set(float(value) for value in selected.index) != set(LAMBDA_GRID):
        raise AssertionError("lambda selection lacks the frozen grid")
    minimum = float(selected.min())
    ties = [
        float(value)
        for value, objective in selected.items()
        if float(objective) <= minimum + 1e-12
    ]
    return max(ties)


def design_replay_inputs(
    designs: Mapping[str, sparse.csr_matrix],
    target: np.ndarray,
    offset: np.ndarray,
    anchor_id: np.ndarray,
    cycle_index: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact arrays the runner can serialize for independent replay."""

    payload: dict[str, np.ndarray] = {
        "target": np.asarray(target, dtype=np.int8),
        "offset": np.asarray(offset, dtype=np.float64),
        "anchor_id": np.asarray(anchor_id, dtype=np.int64),
        "cycle_index": np.asarray(cycle_index, dtype=np.int16),
    }
    for name in ("qpattern", "qlimited4", "qfull9"):
        matrix = sparse.csr_matrix(designs[name], dtype=np.float64)
        payload[f"{name}__data"] = matrix.data.copy()
        payload[f"{name}__indices"] = matrix.indices.copy()
        payload[f"{name}__indptr"] = matrix.indptr.copy()
        payload[f"{name}__shape"] = np.asarray(matrix.shape, dtype=np.int64)
    return payload


def replay_design_from_payload(
    payload: Mapping[str, np.ndarray], name: str
) -> sparse.csr_matrix:
    if name not in ("qpattern", "qlimited4", "qfull9"):
        raise KeyError(name)
    shape = tuple(int(value) for value in payload[f"{name}__shape"])
    return sparse.csr_matrix(
        (
            np.asarray(payload[f"{name}__data"], dtype=np.float64),
            np.asarray(payload[f"{name}__indices"]),
            np.asarray(payload[f"{name}__indptr"]),
        ),
        shape=shape,
        dtype=np.float64,
    )


def _fit_payload(prefix: str, fit: ResidualFit) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__coefficients": fit.coefficients,
        f"{prefix}__ridge_lambda": np.asarray([fit.ridge_lambda], dtype=np.float64),
        f"{prefix}__factor_count": np.asarray([fit.factor_count], dtype=np.int8),
        f"{prefix}__objective": np.asarray([fit.objective], dtype=np.float64),
        f"{prefix}__gradient_max_abs": np.asarray([fit.gradient_max_abs], dtype=np.float64),
        f"{prefix}__iterations": np.asarray([fit.iterations], dtype=np.int32),
        f"{prefix}__optimizer_message": np.asarray([fit.optimizer_message]),
    }


def _kernel_payload(prefix: str, kernel: TransitionKernel) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__classes": kernel.classes,
        f"{prefix}__coef": kernel.coef,
        f"{prefix}__intercept": kernel.intercept,
        f"{prefix}__numeric_width": np.asarray([kernel.numeric_width], dtype=np.int8),
        f"{prefix}__n_iter": np.asarray([kernel.n_iter], dtype=np.int32),
    }


def bundle_arrays(bundle: ModelBundle) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    payload.update(_kernel_payload("history", bundle.history_kernel))
    payload.update(_kernel_payload("old_limited", bundle.old_limited_kernel))
    payload.update(
        {
            "numeric_medians": bundle.numeric_transform.medians,
            "numeric_scales": bundle.numeric_transform.scales,
            "numeric_columns": np.asarray(bundle.numeric_transform.columns),
            "cycle_weights": bundle.contrast_spec.cycle_weights,
            "route_weights": bundle.contrast_spec.route_weights,
            "cycle_reference": np.asarray([bundle.contrast_spec.cycle_reference], dtype=np.int8),
            "route_reference_by_cycle": bundle.contrast_spec.route_reference_by_cycle,
            "token_mask": bundle.token_mask.astype(np.uint8),
            "quartile_cutpoints": bundle.quartile_cutpoints,
            "quartile_columns": np.asarray(QUARTILE_COLUMNS),
            "contract_sha256": np.asarray([CONTRACT_SHA256]),
        }
    )
    for head in ("qpattern", "qlimited4", "qfull9"):
        payload.update(_fit_payload(head, getattr(bundle, head)))
    return payload


def save_bundle(path: Path, bundle: ModelBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle_arrays(bundle))


def _load_kernel(stored: Mapping[str, np.ndarray], prefix: str) -> TransitionKernel:
    return TransitionKernel(
        classes=np.asarray(stored[f"{prefix}__classes"]).copy(),
        coef=np.asarray(stored[f"{prefix}__coef"], dtype=np.float64).copy(),
        intercept=np.asarray(stored[f"{prefix}__intercept"], dtype=np.float64).copy(),
        numeric_width=int(stored[f"{prefix}__numeric_width"][0]),
        n_iter=int(stored[f"{prefix}__n_iter"][0]),
    )


def _load_fit(stored: Mapping[str, np.ndarray], prefix: str) -> ResidualFit:
    return ResidualFit(
        coefficients=np.asarray(stored[f"{prefix}__coefficients"], dtype=np.float64).copy(),
        ridge_lambda=float(stored[f"{prefix}__ridge_lambda"][0]),
        factor_count=int(stored[f"{prefix}__factor_count"][0]),
        objective=float(stored[f"{prefix}__objective"][0]),
        gradient_max_abs=float(stored[f"{prefix}__gradient_max_abs"][0]),
        iterations=int(stored[f"{prefix}__iterations"][0]),
        optimizer_message=str(stored[f"{prefix}__optimizer_message"][0]),
    )


def load_bundle(path: Path) -> ModelBundle:
    with np.load(path, allow_pickle=False) as stored_file:
        stored = {name: stored_file[name].copy() for name in stored_file.files}
    if str(stored["contract_sha256"][0]) != CONTRACT_SHA256:
        raise AssertionError("serialized bundle contract hash changed")
    columns = tuple(str(value) for value in stored["numeric_columns"])
    bundle = ModelBundle(
        history_kernel=_load_kernel(stored, "history"),
        old_limited_kernel=_load_kernel(stored, "old_limited"),
        numeric_transform=NumericTransform(
            columns,
            np.asarray(stored["numeric_medians"], dtype=np.float64),
            np.asarray(stored["numeric_scales"], dtype=np.float64),
        ),
        contrast_spec=ContrastSpec(
            np.asarray(stored["cycle_weights"], dtype=np.float64),
            np.asarray(stored["route_weights"], dtype=np.float64),
            int(stored["cycle_reference"][0]),
            np.asarray(stored["route_reference_by_cycle"], dtype=int),
        ),
        token_mask=np.asarray(stored["token_mask"], dtype=bool),
        quartile_cutpoints=np.asarray(stored["quartile_cutpoints"], dtype=np.float64),
        qpattern=_load_fit(stored, "qpattern"),
        qlimited4=_load_fit(stored, "qlimited4"),
        qfull9=_load_fit(stored, "qfull9"),
    )
    expected_widths = (PATTERN_WIDTH, LIMITED_WIDTH, FULL_WIDTH)
    observed_widths = tuple(
        len(getattr(bundle, head).coefficients)
        for head in ("qpattern", "qlimited4", "qfull9")
    )
    quartile_columns = tuple(str(value) for value in stored["quartile_columns"])
    if (
        observed_widths != expected_widths
        or columns != tuple(FULL9)
        or quartile_columns != tuple(QUARTILE_COLUMNS)
        or bundle.quartile_cutpoints.shape != (len(QUARTILE_COLUMNS), 3)
    ):
        raise AssertionError("serialized bundle design manifest changed")
    return bundle


def provider_hashes_for_scoring_year(
    year: int, guard_token: Mapping[str, Any]
) -> dict[str, str]:
    phase_source_paths("score", guard_token=guard_token)
    if year not in (2023, 2025):
        raise ValueError("later scoring year must be 2025 or 2023")
    if sha256(PROVIDER_HASH_MANIFEST) != PROVIDER_HASH_MANIFEST_SHA256:
        raise AssertionError("pinned provider hash manifest changed")
    payload = json.loads(PROVIDER_HASH_MANIFEST.read_text())
    prefix = f"provider_{year}_"
    result = {
        key[len(prefix) : -len(".parquet")]: str(value)
        for key, value in payload.items()
        if key.startswith(prefix) and key.endswith(".parquet")
    }
    expected = 22 if year == 2025 else 20
    if len(result) != expected:
        raise AssertionError(f"provider manifest symbol count changed for {year}")
    return result


def validate_2024() -> dict[str, Any]:
    prepared = prepare_2024()
    return {
        "status": "validated_2024_without_fit",
        "contract_sha256": CONTRACT_SHA256,
        "source_manifest": prepared.source_manifest,
        "audit": prepared.audit,
        "route_rows": len(prepared.route_map),
        "later_period_panels_read": False,
        **SAFETY,
    }


def self_tests() -> dict[str, Any]:
    contract = validate_contract()
    layouts = {factor: _factor_layout(factor)["width"] for factor in (0, 4, 9)}
    checks = {
        "contract_hash": sha256(CONTRACT_PATH) == CONTRACT_SHA256,
        "safety": all(contract[key] == value for key, value in SAFETY.items()),
        "lambda_grid": tuple(contract["models"]["ridge_lambda_grid"]) == LAMBDA_GRID,
        "design_widths": layouts == {0: PATTERN_WIDTH, 4: LIMITED_WIDTH, 9: FULL_WIDTH},
        "penalty_widths": all(
            len(penalty_multipliers(factor)) == layouts[factor]
            for factor in (0, 4, 9)
        ),
        "history_token_boundary": int(history_tokens([8], [8], [7])[0]) == 647,
        "phase_isolation": "runs_2025" not in phase_source_paths("fit"),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks": checks, "passed": len(checks), "total": len(checks)}
