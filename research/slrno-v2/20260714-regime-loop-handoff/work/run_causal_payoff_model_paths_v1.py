"""Research-only prequential tests of causal SLRNO payoff-model paths."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-causal-payoff-model-paths-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260714-causal-payoff-model-paths-v1-pre-score.json"
BASE_PATH = HERE / "run_loop_payoff_phase_path_v1.py"
SPEC = importlib.util.spec_from_file_location("loop_phase_path_v1_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

PERIOD = 2024
SEED = 20260714
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
WARMUP_SESSIONS = 60
ROLLING_SESSIONS = 60
ROUTE_MIN_SESSIONS = 20
ROUND_TRIP_COST_BPS = 10.0
Z_90 = 1.6448536269514722
ROUTE_UNIFORM_MIX = 0.02
ROUTE_CLASSES = (
    "no_transition",
    "expected_leg_partial",
    "exact_parent_completion",
    "incompatible_first_transition",
    "expected_leg_then_diversion",
)
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)

ADMISSION_NUMERIC = (
    "top_loop_probability",
    "top_loop_margin",
    "entry_step",
    "entry_session_fraction",
    "anchor_body_fraction",
    "anchor_upper_wick_fraction",
    "anchor_lower_wick_fraction",
    "anchor_range_fraction",
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
    "atr14_prior_fraction",
    "anchor_range_atr",
    "prior_regime_age_bars",
)
ADMISSION_CATEGORICAL = (
    "candidate",
    "direction_label",
    "previous_state_1_label",
    "pre_entry_path_status",
    "clock_quartile_label",
    "prior_completed_state_label",
)
ROUTE_PROBABILITY_COLUMNS = tuple(f"route_probability__{name}" for name in ROUTE_CLASSES)
ROUTE_PRIOR_COLUMNS = tuple(f"route_prior__{name}" for name in ROUTE_CLASSES)
SEQUENTIAL_ROUTE_NUMERIC = (
    "checkpoint_fraction",
    "bars_elapsed",
    "bars_remaining",
    "current_regime_age_bars",
)
SEQUENTIAL_ROUTE_CATEGORICAL = (
    "causal_route_status",
    "current_completed_state_label",
)
SEQUENTIAL_PRICE_NUMERIC = (
    "directional_close_return_bps",
    "running_post_entry_mfe_bps",
    "running_post_entry_mae_bps",
    "running_close_peak_bps",
    "causal_close_retracement_bps",
    "mfe_prior_atr",
    "mae_prior_atr",
    "retracement_prior_atr",
    "mean_completed_bar_range_prior_atr",
    "current_bar_body_fraction",
    "current_bar_upper_wick_fraction",
    "current_bar_lower_wick_fraction",
)


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    inputs = contract["inputs"]
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "auditor": HERE / "audit_causal_payoff_model_paths_v1.py",
        "tests": HERE / "tests/test_causal_payoff_model_paths_v1.py",
        "base_loader_runner": Path(inputs["base_loader_runner"]),
        "accepted_setup_signals_2024": Path(inputs["accepted_setup_signals_2024"]),
        "anchor_panel_2024": Path(inputs["anchor_panel_2024"]),
        "state_runs_2024": Path(inputs["state_runs_2024"]),
        "fixed_cycles": Path(inputs["fixed_cycles"]),
        "parent_report": Path(inputs["parent_report"]),
        "parent_handoff": Path(inputs["parent_handoff"]),
        "prospective_log_contract_v2": Path(inputs["prospective_log_contract_v2"]),
    }
    root = Path(inputs["provider_root_2024"])
    for symbol in contract["population"]["symbols"]:
        paths[f"provider_2024_{symbol}"] = provider_path(root, symbol)
    return paths


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text())
    safety = contract["safety"]
    seal = contract["sealed_data_status"]
    if not (
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["broker_connection_enabled"] is False
        and safety["paper_or_demo_execution_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["position_or_order_functionality_allowed"] is False
        and safety["application_code_modification_allowed"] is False
        and safety["repository_write_allowed"] is False
        and seal["genuinely_unseen_sessions_available"] is False
        and seal["validation_claim_allowed"] is False
        and seal["diversion_specific_hypothesis_test_allowed"] is False
    ):
        raise AssertionError("research-only or data-seal boundary drift")
    return contract


def freeze_manifest(contract: dict[str, Any]) -> None:
    if PRE_SCORE_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {PRE_SCORE_PATH}")
    paths = source_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze; missing sources: {missing}")
    write_json(
        PRE_SCORE_PATH,
        {
            "contract_id": contract["contract_id"],
            "frozen_before_model_scoring": True,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "sha256": {name: sha256(path) for name, path in sorted(paths.items())},
        },
    )


def verify_frozen_sources(contract: dict[str, Any]) -> dict[str, str]:
    expected = json.loads(PRE_SCORE_PATH.read_text())
    actual = {name: sha256(path) for name, path in sorted(source_paths(contract).items())}
    if actual != expected["sha256"]:
        changed = sorted(
            name
            for name in set(actual) | set(expected["sha256"])
            if actual.get(name) != expected["sha256"].get(name)
        )
        raise AssertionError(f"pre-score source hash mismatch: {changed}")
    return actual


def topology_from_transitions(
    transitions: list[tuple[int, int]], anchor_state: int, alternate_state: int
) -> str:
    if not transitions:
        return "no_transition"
    if transitions[0][0] != alternate_state:
        return "incompatible_first_transition"
    if len(transitions) == 1:
        return "expected_leg_partial"
    if transitions[1][0] == anchor_state:
        return "exact_parent_completion"
    return "expected_leg_then_diversion"


def pre_entry_status(
    transitions: list[tuple[int, int]],
    entry_state_position: int,
    anchor_state: int,
    alternate_state: int,
) -> str:
    known = [
        (state, position) for state, position in transitions if position < entry_state_position
    ]
    topology = topology_from_transitions(known, anchor_state, alternate_state)
    return {
        "no_transition": "orientation_intact",
        "expected_leg_partial": "expected_leg_active",
        "exact_parent_completion": "completed_before_entry",
        "incompatible_first_transition": "invalidated_before_entry",
        "expected_leg_then_diversion": "invalidated_before_entry",
    }[topology]


def causal_route_status(
    transitions: list[tuple[int, int]],
    checkpoint_position: int,
    anchor_state: int,
    alternate_state: int,
) -> str:
    known = [
        (state, position) for state, position in transitions if position <= checkpoint_position
    ]
    topology = topology_from_transitions(known, anchor_state, alternate_state)
    return {
        "no_transition": "orientation_intact",
        "expected_leg_partial": "expected_leg_active",
        "exact_parent_completion": "exact_parent_completion_detected",
        "incompatible_first_transition": "incompatible_first_transition_detected",
        "expected_leg_then_diversion": "expected_leg_then_diversion_detected",
    }[topology]


def checkpoint_offsets(holding_bars: int) -> list[tuple[float, int]]:
    result: dict[int, float] = {}
    for fraction in CHECKPOINT_FRACTIONS:
        offset = int(math.ceil(fraction * holding_bars))
        if 1 <= offset < holding_bars:
            result.setdefault(offset, fraction)
    return [(fraction, offset) for offset, fraction in sorted(result.items())]


def interval_class(mean: np.ndarray, std: np.ndarray, target: str) -> np.ndarray:
    lower = mean - Z_90 * std
    upper = mean + Z_90 * std
    if target == "admission":
        return np.where(
            lower > 0.0, "positive", np.where(upper < 0.0, "negative", "unknown_abstain")
        )
    if target == "sequential":
        return np.where(
            lower > 0.0,
            "positive_hold",
            np.where(upper < 0.0, "negative_exit", "unknown_abstain"),
        )
    raise ValueError(target)


def body_wicks(
    open_price: float, high: float, low: float, close: float
) -> tuple[float, float, float]:
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return 0.0, 0.0, 0.0
    body = abs(close - open_price) / span
    upper = (high - max(open_price, close)) / span
    lower = (min(open_price, close) - low) / span
    return float(body), float(upper), float(lower)


def gross_bps(direction: int, exit_price: float, entry_price: float) -> float:
    if direction not in (-1, 1):
        raise AssertionError("invalid direction")
    return 10000.0 * direction * (exit_price / entry_price - 1.0)


def load_surface(
    contract: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], pd.DataFrame],
    list[dict[str, Any]],
]:
    signals = pd.read_parquet(Path(contract["inputs"]["accepted_setup_signals_2024"]))
    symbols = list(contract["population"]["symbols"])
    selected_surface = signals.loc[
        signals["setup"].eq(contract["population"]["setup"])
        & signals["family"].eq(contract["population"]["family"])
        & signals["horizon"].eq(24)
        & signals["status"].eq("filled")
        & signals["symbol_norm"].isin(symbols)
    ].copy()
    selected_surface["session_date"] = selected_surface["session_date"].astype(str)
    calendar = sorted(selected_surface["session_date"].unique())
    if len(calendar) != int(contract["population"]["surface_sessions"]):
        raise AssertionError("surface calendar drift")

    score_columns = [f"loop_score_{index:02d}" for index in range(1, 21)]
    anchor_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "state",
        "start_pos",
        "start_timestamp",
        "previous_state_1",
        "current_bar_log_return",
        "return_sum_6",
        "mean_abs_return_12",
        "session_return",
        "bar_range_pct",
        *score_columns,
    ]
    anchors = pd.read_parquet(Path(contract["inputs"]["anchor_panel_2024"]), columns=anchor_columns)
    anchors["session_date"] = anchors["session_date"].astype(str)
    cycles = pd.read_csv(Path(contract["inputs"]["fixed_cycles"]))
    if len(cycles) != 20:
        raise AssertionError("cycle dictionary drift")
    scores = anchors[score_columns].to_numpy(float)
    top_index = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    anchors["top_loop"] = cycles["cycle_id"].to_numpy(str)[top_index]
    anchors["top_loop_probability"] = scores[np.arange(len(anchors)), top_index]
    anchors["top_loop_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    anchors = anchors.rename(columns={"state": "anchor_state", "start_pos": "anchor_start_pos"})
    joined = selected_surface.merge(
        anchors[
            [
                "anchor_id",
                "symbol_norm",
                "session_date",
                "anchor_state",
                "anchor_start_pos",
                "start_timestamp",
                "previous_state_1",
                "current_bar_log_return",
                "return_sum_6",
                "mean_abs_return_12",
                "session_return",
                "bar_range_pct",
                "top_loop",
                "top_loop_probability",
                "top_loop_margin",
            ]
        ],
        on=["anchor_id", "symbol_norm", "session_date", "start_timestamp"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_anchor"),
    )
    if joined["top_loop"].isna().any() or not joined["state"].eq(joined["anchor_state"]).all():
        raise AssertionError("anchor join drift")
    specs = {item["candidate"]: item for item in contract["population"]["primary_candidates"]}
    candidates: list[pd.DataFrame] = []
    for candidate, item in specs.items():
        loop = candidate.split("|")[0]
        cell = joined.loc[
            joined["top_loop"].eq(loop) & joined["anchor_state"].eq(int(item["anchor_state"]))
        ].copy()
        if len(cell) != int(item["full_surface_expected_rows"]):
            raise AssertionError(f"candidate population drift {candidate}: {len(cell)}")
        cell["candidate"] = candidate
        cell["expected_alternate_state"] = int(item["expected_alternate_state"])
        candidates.append(cell)
    candidate_frame = pd.concat(candidates, ignore_index=True)

    tape, coverage = BASE.load_tape(Path(contract["inputs"]["provider_root_2024"]), symbols, PERIOD)
    runs = BASE.load_runs(Path(contract["inputs"]["state_runs_2024"]), PERIOD)
    run_groups = BASE.run_lookup(runs)
    return candidate_frame, calendar, tape, run_groups, coverage


def enrich_surface(
    frame: pd.DataFrame,
    calendar: list[str],
    tape: dict[tuple[str, str], pd.DataFrame],
    run_groups: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    calendar_index = {date: index for index, date in enumerate(calendar)}
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        key = (str(row.symbol_norm), str(row.session_date))
        bars = tape[key]
        state_runs = run_groups[key]
        anchor_timestamp = pd.Timestamp(row.start_timestamp)
        matches = np.flatnonzero(bars["timestamp"].eq(anchor_timestamp).to_numpy(bool))
        if len(matches) != 1:
            raise AssertionError("provider anchor timestamp mismatch")
        tape_anchor = int(matches[0])
        entry_ordinal = tape_anchor + int(row.entry_step)
        frozen_exit_ordinal = tape_anchor + 24
        if not (0 <= tape_anchor < entry_ordinal <= frozen_exit_ordinal < len(bars)):
            raise AssertionError("invalid execution ordinals")
        anchor_position = int(row.anchor_start_pos)
        entry_position = anchor_position + int(row.entry_step)
        frozen_position = anchor_position + 24
        anchor_run = BASE.covering_run(state_runs, anchor_position)
        if (
            int(anchor_run.state) != int(row.anchor_state)
            or int(anchor_run.start_pos) != anchor_position
        ):
            raise AssertionError("anchor state-run mismatch")
        transitions_frame = state_runs.loc[
            state_runs["start_pos"].gt(anchor_position)
            & state_runs["start_pos"].le(frozen_position)
        ].sort_values("start_pos", kind="stable")
        transitions = [
            (int(item.state), int(item.start_pos))
            for item in transitions_frame.itertuples(index=False)
        ]
        prior_position = entry_position - 1
        prior_run = BASE.covering_run(state_runs, prior_position)
        prior_age = prior_position - int(prior_run.start_pos) + 1
        topology = topology_from_transitions(
            transitions, int(row.anchor_state), int(row.expected_alternate_state)
        )
        fixed_price = float(bars.iloc[frozen_exit_ordinal]["close"])
        replay_gross = gross_bps(int(row.direction), fixed_price, float(row.entry_price))
        if not np.isclose(replay_gross, float(row.gross_return_bps), atol=1e-8, rtol=1e-8):
            raise AssertionError("2024 frozen payoff replay mismatch")
        anchor_bar = bars.iloc[tape_anchor]
        body, upper, lower = body_wicks(
            float(anchor_bar["open"]),
            float(anchor_bar["high"]),
            float(anchor_bar["low"]),
            float(anchor_bar["close"]),
        )
        atr = float(bars.iloc[entry_ordinal]["atr14_prior"])
        atr_valid = math.isfinite(atr) and atr > 0
        anchor_range = float(anchor_bar["high"] - anchor_bar["low"])
        payload = {
            "signal_id": str(row.anchor_id),
            "anchor_id": row.anchor_id,
            "candidate": str(row.candidate),
            "symbol_norm": str(row.symbol_norm),
            "session_date": str(row.session_date),
            "month": str(row.session_date)[:7],
            "calendar_index": calendar_index[str(row.session_date)],
            "score_eligible": calendar_index[str(row.session_date)] >= WARMUP_SESSIONS,
            "start_timestamp": anchor_timestamp,
            "anchor_state": int(row.anchor_state),
            "expected_alternate_state": int(row.expected_alternate_state),
            "anchor_state_position": anchor_position,
            "entry_state_position": entry_position,
            "frozen_exit_state_position": frozen_position,
            "tape_anchor_ordinal": tape_anchor,
            "entry_ordinal": entry_ordinal,
            "frozen_exit_ordinal": frozen_exit_ordinal,
            "entry_timestamp": pd.Timestamp(bars.iloc[entry_ordinal]["timestamp"]),
            "entry_price": float(row.entry_price),
            "fixed_exit_price": fixed_price,
            "direction": int(row.direction),
            "direction_label": str(int(row.direction)),
            "entry_step": int(row.entry_step),
            "entry_session_fraction": entry_ordinal / max(1.0, len(bars) - 1.0),
            "top_loop_probability": float(row.top_loop_probability),
            "top_loop_margin": float(row.top_loop_margin),
            "anchor_body_fraction": body,
            "anchor_upper_wick_fraction": upper,
            "anchor_lower_wick_fraction": lower,
            "anchor_range_fraction": anchor_range / float(anchor_bar["close"]),
            "current_bar_log_return": float(row.current_bar_log_return),
            "return_sum_6": float(row.return_sum_6),
            "mean_abs_return_12": float(row.mean_abs_return_12),
            "session_return": float(row.session_return),
            "bar_range_pct": float(row.bar_range_pct),
            "atr14_prior": atr if atr_valid else math.nan,
            "atr14_prior_fraction": atr / float(row.entry_price) if atr_valid else math.nan,
            "anchor_range_atr": anchor_range / atr if atr_valid else math.nan,
            "prior_regime_age_bars": prior_age,
            "previous_state_1_label": str(int(row.previous_state_1)),
            "pre_entry_path_status": pre_entry_status(
                transitions,
                entry_position,
                int(row.anchor_state),
                int(row.expected_alternate_state),
            ),
            "clock_quartile_label": str(row.clock_quartile),
            "prior_completed_state_label": str(int(prior_run.state)),
            "route_topology_outcome_only": topology,
            "fixed_gross_bps": replay_gross,
            "fixed_net_bps": replay_gross - ROUND_TRIP_COST_BPS,
            "holding_bars_after_entry": frozen_exit_ordinal - entry_ordinal,
        }
        rows.append(payload)
    result = (
        pd.DataFrame(rows)
        .sort_values(["calendar_index", "candidate", "symbol_norm", "entry_ordinal"], kind="stable")
        .reset_index(drop=True)
    )
    if result["signal_id"].duplicated().any():
        raise AssertionError("duplicate signal id")
    return result


def make_preprocessor(numeric: Iterable[str], categorical: Iterable[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(numeric)),
            ("categorical", categorical_pipeline, list(categorical)),
        ],
        sparse_threshold=0.0,
    )


def fit_bayesian_ridge(
    train: pd.DataFrame,
    score: pd.DataFrame,
    target: str,
    numeric: Iterable[str],
    categorical: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    numeric = tuple(numeric)
    categorical = tuple(categorical)
    transformer = make_preprocessor(numeric, categorical)
    train_matrix = transformer.fit_transform(train[list(numeric) + list(categorical)])
    score_matrix = transformer.transform(score[list(numeric) + list(categorical)])
    model = BayesianRidge(
        max_iter=300,
        tol=1e-6,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=False,
        fit_intercept=True,
    )
    model.fit(train_matrix, train[target].to_numpy(float))
    mean, std = model.predict(score_matrix, return_std=True)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def fit_route_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, bool]:
    transformer = make_preprocessor(ADMISSION_NUMERIC, ADMISSION_CATEGORICAL)
    train_matrix = transformer.fit_transform(
        train[list(ADMISSION_NUMERIC) + list(ADMISSION_CATEGORICAL)]
    )
    score_matrix = transformer.transform(
        score[list(ADMISSION_NUMERIC) + list(ADMISSION_CATEGORICAL)]
    )
    labels = train["route_topology_outcome_only"].astype(str).to_numpy()
    fitted = len(np.unique(labels)) >= 2
    raw = np.zeros((len(score), len(ROUTE_CLASSES)), dtype=float)
    if fitted:
        model = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=None
        )
        model.fit(train_matrix, labels)
        predicted = model.predict_proba(score_matrix)
        class_index = {name: index for index, name in enumerate(ROUTE_CLASSES)}
        for source_index, label in enumerate(model.classes_):
            raw[:, class_index[str(label)]] = predicted[:, source_index]
    else:
        counts = pd.Series(labels).value_counts()
        raw[:] = np.array([counts.get(name, 0) + 1 for name in ROUTE_CLASSES], dtype=float)
        raw /= raw.sum(axis=1, keepdims=True)
    probabilities = (1.0 - ROUTE_UNIFORM_MIX) * raw + ROUTE_UNIFORM_MIX / len(ROUTE_CLASSES)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    priors = np.empty_like(probabilities)
    for candidate, indices in score.groupby("candidate", sort=False).groups.items():
        candidate_labels = train.loc[
            train["candidate"].eq(candidate), "route_topology_outcome_only"
        ]
        if candidate_labels.empty:
            candidate_labels = train["route_topology_outcome_only"]
        counts = candidate_labels.value_counts()
        prior = np.array([counts.get(name, 0) + 1.0 for name in ROUTE_CLASSES], dtype=float)
        prior /= prior.sum()
        positions = score.index.get_indexer(indices)
        priors[positions, :] = prior
    return probabilities, priors, fitted


def prequential_route_predictions(surface: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for score_index in range(ROUTE_MIN_SESSIONS, len(calendar)):
        train_start = max(0, score_index - ROLLING_SESSIONS)
        train_dates = calendar[train_start:score_index]
        score_date = calendar[score_index]
        train = surface.loc[surface["session_date"].isin(train_dates)].copy()
        score = surface.loc[surface["session_date"].eq(score_date)].copy().reset_index(drop=True)
        if score.empty:
            continue
        probabilities, priors, fitted = fit_route_model(train, score)
        output = score[
            ["signal_id", "candidate", "symbol_norm", "session_date", "month", "calendar_index"]
        ].copy()
        output["actual_route"] = score["route_topology_outcome_only"].to_numpy(str)
        for index, name in enumerate(ROUTE_CLASSES):
            output[f"route_probability__{name}"] = probabilities[:, index]
            output[f"route_prior__{name}"] = priors[:, index]
        actual_index = np.array([ROUTE_CLASSES.index(name) for name in output["actual_route"]])
        output["model_log_loss"] = -np.log(probabilities[np.arange(len(output)), actual_index])
        output["prior_log_loss"] = -np.log(priors[np.arange(len(output)), actual_index])
        output["log_loss_improvement"] = output["prior_log_loss"] - output["model_log_loss"]
        one_hot = np.eye(len(ROUTE_CLASSES))[actual_index]
        output["model_brier"] = np.sum((probabilities - one_hot) ** 2, axis=1)
        output["prior_brier"] = np.sum((priors - one_hot) ** 2, axis=1)
        output["model_correct"] = np.argmax(probabilities, axis=1) == actual_index
        output["prior_correct"] = np.argmax(priors, axis=1) == actual_index
        output["train_first_session"] = train_dates[0]
        output["train_last_session"] = train_dates[-1]
        output["train_sessions"] = len(train_dates)
        output["train_rows"] = len(train)
        output["route_model_fitted"] = fitted
        rows.append(output)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["calendar_index", "candidate", "symbol_norm"], kind="stable")
        .reset_index(drop=True)
    )


def attach_route_probabilities(surface: pd.DataFrame, route: pd.DataFrame) -> pd.DataFrame:
    columns = ["signal_id", "train_last_session", *ROUTE_PROBABILITY_COLUMNS]
    rename = {"train_last_session": "route_probability_train_last_session"}
    result = surface.merge(
        route[columns].rename(columns=rename), on="signal_id", how="left", validate="one_to_one"
    )
    result["route_probabilities_available"] = (
        result[list(ROUTE_PROBABILITY_COLUMNS)].notna().all(axis=1)
    )
    return result


def rolling_shrunk_mean(train: pd.DataFrame, score: pd.DataFrame, target: str) -> np.ndarray:
    pooled = float(train[target].mean())
    result = np.empty(len(score), dtype=float)
    for candidate, positions in score.groupby("candidate", sort=False).groups.items():
        values = train.loc[train["candidate"].eq(candidate), target]
        n = len(values)
        mean = float(values.mean()) if n else pooled
        shrunk = (n * mean + 20.0 * pooled) / (n + 20.0)
        result[score.index.get_indexer(positions)] = shrunk
    return result


def prequential_admission_predictions(surface: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    variants = {
        "admission_only": (ADMISSION_NUMERIC, ADMISSION_CATEGORICAL),
        "admission_plus_causal_route_probabilities": (
            (*ADMISSION_NUMERIC, *ROUTE_PROBABILITY_COLUMNS),
            ADMISSION_CATEGORICAL,
        ),
    }
    for score_index in range(WARMUP_SESSIONS, len(calendar)):
        train_dates = calendar[score_index - ROLLING_SESSIONS : score_index]
        score_date = calendar[score_index]
        base_train = surface.loc[surface["session_date"].isin(train_dates)].copy()
        base_score = (
            surface.loc[surface["session_date"].eq(score_date)].copy().reset_index(drop=True)
        )
        if base_score.empty:
            continue
        baseline = rolling_shrunk_mean(base_train, base_score, "fixed_net_bps")
        for variant, (numeric, categorical) in variants.items():
            train = base_train
            score = base_score
            if variant.endswith("route_probabilities"):
                train = train.loc[train["route_probabilities_available"]].copy()
                if not score["route_probabilities_available"].all():
                    raise AssertionError("route probability unavailable on admission score row")
            mean, std = fit_bayesian_ridge(train, score, "fixed_net_bps", numeric, categorical)
            lower = mean - Z_90 * std
            upper = mean + Z_90 * std
            output = score[
                [
                    "signal_id",
                    "candidate",
                    "symbol_norm",
                    "session_date",
                    "month",
                    "calendar_index",
                    "fixed_gross_bps",
                    "fixed_net_bps",
                ]
            ].copy()
            output["model"] = variant
            output["predicted_net_bps"] = mean
            output["predictive_std_bps"] = std
            output["lower_90_bps"] = lower
            output["upper_90_bps"] = upper
            output["uncertainty_class"] = interval_class(mean, std, "admission")
            output["point_positive"] = mean > 0.0
            output["rolling_shrunk_mean_bps"] = baseline
            output["train_first_session"] = train_dates[0]
            output["train_last_session"] = train_dates[-1]
            output["train_sessions"] = len(train_dates)
            output["train_rows"] = len(train)
            rows.append(output)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["model", "calendar_index", "candidate", "symbol_norm"], kind="stable")
        .reset_index(drop=True)
    )


def build_snapshots(
    surface: pd.DataFrame,
    tape: dict[tuple[str, str], pd.DataFrame],
    run_groups: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in surface.itertuples(index=False):
        key = (str(row.symbol_norm), str(row.session_date))
        bars = tape[key]
        state_runs = run_groups[key]
        anchor_position = int(row.anchor_state_position)
        frozen_position = int(row.frozen_exit_state_position)
        transitions_frame = state_runs.loc[
            state_runs["start_pos"].gt(anchor_position)
            & state_runs["start_pos"].le(frozen_position)
        ].sort_values("start_pos", kind="stable")
        transitions = [
            (int(item.state), int(item.start_pos))
            for item in transitions_frame.itertuples(index=False)
        ]
        for fraction, offset in checkpoint_offsets(int(row.holding_bars_after_entry)):
            checkpoint_ordinal = int(row.entry_ordinal) + offset
            next_open_ordinal = checkpoint_ordinal + 1
            checkpoint_position = int(row.entry_state_position) + offset
            if next_open_ordinal > int(row.frozen_exit_ordinal):
                raise AssertionError("checkpoint lacks next open before frozen close")
            current_run = BASE.covering_run(state_runs, checkpoint_position)
            current_age = checkpoint_position - int(current_run.start_pos) + 1
            completed = bars.iloc[int(row.entry_ordinal) + 1 : checkpoint_ordinal + 1]
            if completed.empty:
                raise AssertionError("checkpoint without completed post-entry bar")
            direction = int(row.direction)
            entry_price = float(row.entry_price)
            if direction == 1:
                favorable = 10000.0 * (completed["high"].to_numpy(float) / entry_price - 1.0)
                adverse = 10000.0 * (completed["low"].to_numpy(float) / entry_price - 1.0)
                close_path = 10000.0 * (completed["close"].to_numpy(float) / entry_price - 1.0)
            else:
                favorable = -10000.0 * (completed["low"].to_numpy(float) / entry_price - 1.0)
                adverse = -10000.0 * (completed["high"].to_numpy(float) / entry_price - 1.0)
                close_path = -10000.0 * (completed["close"].to_numpy(float) / entry_price - 1.0)
            current_close_return = float(close_path[-1])
            close_peak = float(np.max(close_path))
            retracement = current_close_return - close_peak
            checkpoint_bar = bars.iloc[checkpoint_ordinal]
            body, upper, lower = body_wicks(
                float(checkpoint_bar["open"]),
                float(checkpoint_bar["high"]),
                float(checkpoint_bar["low"]),
                float(checkpoint_bar["close"]),
            )
            atr = float(row.atr14_prior)
            atr_bps = 10000.0 * atr / entry_price if math.isfinite(atr) and atr > 0 else math.nan
            mean_range = float(completed["true_range"].mean())
            next_open_price = float(bars.iloc[next_open_ordinal]["open"])
            next_open_gross = gross_bps(direction, next_open_price, entry_price)
            payload = {name: getattr(row, name) for name in ADMISSION_NUMERIC}
            payload.update({name: getattr(row, name) for name in ADMISSION_CATEGORICAL})
            payload.update({name: getattr(row, name) for name in ROUTE_PROBABILITY_COLUMNS})
            payload.update(
                {
                    "signal_id": str(row.signal_id),
                    "candidate": str(row.candidate),
                    "symbol_norm": str(row.symbol_norm),
                    "session_date": str(row.session_date),
                    "month": str(row.month),
                    "calendar_index": int(row.calendar_index),
                    "fixed_gross_bps": float(row.fixed_gross_bps),
                    "fixed_net_bps": float(row.fixed_net_bps),
                    "checkpoint_fraction": fraction,
                    "checkpoint_offset": offset,
                    "checkpoint_ordinal": checkpoint_ordinal,
                    "checkpoint_timestamp": pd.Timestamp(checkpoint_bar["timestamp"]),
                    "next_open_ordinal": next_open_ordinal,
                    "next_open_timestamp": pd.Timestamp(bars.iloc[next_open_ordinal]["timestamp"]),
                    "next_open_price": next_open_price,
                    "next_open_gross_bps": next_open_gross,
                    "next_open_net_bps": next_open_gross - ROUND_TRIP_COST_BPS,
                    "hold_advantage_bps": float(row.fixed_gross_bps) - next_open_gross,
                    "bars_elapsed": offset,
                    "bars_remaining": int(row.frozen_exit_ordinal) - next_open_ordinal,
                    "current_regime_age_bars": current_age,
                    "causal_route_status": causal_route_status(
                        transitions,
                        checkpoint_position,
                        int(row.anchor_state),
                        int(row.expected_alternate_state),
                    ),
                    "current_completed_state_label": str(int(current_run.state)),
                    "directional_close_return_bps": current_close_return,
                    "running_post_entry_mfe_bps": float(np.max(favorable)),
                    "running_post_entry_mae_bps": float(np.min(adverse)),
                    "running_close_peak_bps": close_peak,
                    "causal_close_retracement_bps": retracement,
                    "mfe_prior_atr": float(np.max(favorable)) / atr_bps
                    if atr_bps > 0
                    else math.nan,
                    "mae_prior_atr": float(np.min(adverse)) / atr_bps if atr_bps > 0 else math.nan,
                    "retracement_prior_atr": retracement / atr_bps if atr_bps > 0 else math.nan,
                    "mean_completed_bar_range_prior_atr": mean_range / atr if atr > 0 else math.nan,
                    "current_bar_body_fraction": body,
                    "current_bar_upper_wick_fraction": upper,
                    "current_bar_lower_wick_fraction": lower,
                }
            )
            rows.append(payload)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["calendar_index", "candidate", "symbol_norm", "checkpoint_ordinal"], kind="stable"
        )
        .reset_index(drop=True)
    )


def prequential_snapshot_predictions(snapshots: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    variants = {
        "sequential_route_state_only": (
            (*ADMISSION_NUMERIC, *ROUTE_PROBABILITY_COLUMNS, *SEQUENTIAL_ROUTE_NUMERIC),
            (*ADMISSION_CATEGORICAL, *SEQUENTIAL_ROUTE_CATEGORICAL),
        ),
        "sequential_route_plus_price_path": (
            (
                *ADMISSION_NUMERIC,
                *ROUTE_PROBABILITY_COLUMNS,
                *SEQUENTIAL_ROUTE_NUMERIC,
                *SEQUENTIAL_PRICE_NUMERIC,
            ),
            (*ADMISSION_CATEGORICAL, *SEQUENTIAL_ROUTE_CATEGORICAL),
        ),
    }
    for score_index in range(WARMUP_SESSIONS, len(calendar)):
        train_dates = calendar[score_index - ROLLING_SESSIONS : score_index]
        score_date = calendar[score_index]
        train = snapshots.loc[snapshots["session_date"].isin(train_dates)].copy()
        score = (
            snapshots.loc[snapshots["session_date"].eq(score_date)].copy().reset_index(drop=True)
        )
        if score.empty:
            continue
        train = train.loc[train[list(ROUTE_PROBABILITY_COLUMNS)].notna().all(axis=1)].copy()
        if not score[list(ROUTE_PROBABILITY_COLUMNS)].notna().all(axis=1).all():
            raise AssertionError("snapshot score row lacks causal route probability")
        for variant, (numeric, categorical) in variants.items():
            mean, std = fit_bayesian_ridge(train, score, "hold_advantage_bps", numeric, categorical)
            output = score[
                [
                    "signal_id",
                    "candidate",
                    "symbol_norm",
                    "session_date",
                    "month",
                    "calendar_index",
                    "checkpoint_fraction",
                    "checkpoint_offset",
                    "checkpoint_ordinal",
                    "checkpoint_timestamp",
                    "next_open_ordinal",
                    "next_open_timestamp",
                    "next_open_gross_bps",
                    "next_open_net_bps",
                    "fixed_gross_bps",
                    "fixed_net_bps",
                    "hold_advantage_bps",
                ]
            ].copy()
            output["model"] = variant
            output["predicted_hold_advantage_bps"] = mean
            output["predictive_std_bps"] = std
            output["lower_90_bps"] = mean - Z_90 * std
            output["upper_90_bps"] = mean + Z_90 * std
            output["uncertainty_class"] = interval_class(mean, std, "sequential")
            output["point_negative_exit"] = mean < 0.0
            output["train_first_session"] = train_dates[0]
            output["train_last_session"] = train_dates[-1]
            output["train_sessions"] = len(train_dates)
            output["train_rows"] = len(train)
            rows.append(output)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            ["model", "calendar_index", "candidate", "symbol_norm", "checkpoint_ordinal"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_sequential_policies(
    score_surface: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    prediction_groups = {
        (str(signal_id), str(model)): group.sort_values("checkpoint_ordinal", kind="stable")
        for (signal_id, model), group in predictions.groupby(["signal_id", "model"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    models = ("sequential_route_state_only", "sequential_route_plus_price_path")
    for signal in score_surface.itertuples(index=False):
        for model in models:
            group = prediction_groups.get((str(signal.signal_id), model), pd.DataFrame())
            policies = {
                "uncertainty_aware": (
                    group.loc[group["uncertainty_class"].eq("negative_exit")]
                    if not group.empty
                    else group
                ),
                "point_mean_diagnostic": (
                    group.loc[group["point_negative_exit"].eq(True)] if not group.empty else group
                ),
            }
            for policy_type, actions in policies.items():
                acted = not actions.empty
                action = actions.iloc[0] if acted else None
                gross = (
                    float(action["next_open_gross_bps"]) if acted else float(signal.fixed_gross_bps)
                )
                net = gross - ROUND_TRIP_COST_BPS
                rows.append(
                    {
                        "signal_id": str(signal.signal_id),
                        "candidate": str(signal.candidate),
                        "symbol_norm": str(signal.symbol_norm),
                        "session_date": str(signal.session_date),
                        "month": str(signal.month),
                        "model": model,
                        "policy_type": policy_type,
                        "policy": f"{model}__{policy_type}",
                        "action": acted,
                        "action_checkpoint_fraction": (
                            float(action["checkpoint_fraction"]) if acted else math.nan
                        ),
                        "action_checkpoint_ordinal": (
                            int(action["checkpoint_ordinal"]) if acted else -1
                        ),
                        "action_next_open_ordinal": int(action["next_open_ordinal"])
                        if acted
                        else -1,
                        "action_prediction_bps": (
                            float(action["predicted_hold_advantage_bps"]) if acted else math.nan
                        ),
                        "fixed_gross_bps": float(signal.fixed_gross_bps),
                        "fixed_net_bps": float(signal.fixed_net_bps),
                        "policy_gross_bps": gross,
                        "policy_net_bps": net,
                        "paired_difference_bps": net - float(signal.fixed_net_bps),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["policy", "session_date", "candidate", "symbol_norm"], kind="stable")
        .reset_index(drop=True)
    )


def group_slices(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "pooled", frame
    for candidate in ("cycle_04|state4", "cycle_07|state5"):
        yield candidate, frame.loc[frame["candidate"].eq(candidate)]


def spearman(actual: pd.Series, predicted: pd.Series) -> float:
    if len(actual) < 2 or actual.nunique() < 2 or predicted.nunique() < 2:
        return math.nan
    return float(actual.rank(method="average").corr(predicted.rank(method="average")))


def calibration(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    if len(actual) < 2 or float(np.var(predicted)) <= 0:
        return math.nan, math.nan
    slope, intercept = np.polyfit(predicted.to_numpy(float), actual.to_numpy(float), 1)
    return float(slope), float(intercept)


def moving_block_sample(sessions: list[str], rng: np.random.Generator) -> list[str]:
    count = len(sessions)
    blocks = int(math.ceil(count / BOOTSTRAP_BLOCK))
    starts = rng.integers(0, count, size=blocks)
    return [
        sessions[(int(start) + offset) % count]
        for start in starts
        for offset in range(BOOTSTRAP_BLOCK)
    ][:count]


def paired_bootstrap(frame: pd.DataFrame, value_column: str, seed_offset: int) -> dict[str, Any]:
    sessions = sorted(frame["session_date"].astype(str).unique())
    by_session = {
        date: group[value_column].to_numpy(float)
        for date, group in frame.groupby("session_date", sort=False)
    }
    rng = np.random.default_rng(SEED + seed_offset)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = moving_block_sample(sessions, rng)
        values[draw] = float(np.mean(np.concatenate([by_session[date] for date in sampled])))
    return {
        "observed_mean": float(frame[value_column].mean()),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_one_sided": float((1 + np.sum(values <= 0.0)) / (len(values) + 1)),
        "draws": BOOTSTRAP_DRAWS,
        "sessions": len(sessions),
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    p = output["p_one_sided"].to_numpy(float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    output["holm_adjusted_p"] = adjusted
    output["passes_holm_0_05"] = output["holm_adjusted_p"].lt(0.05) & output["ci_lower"].gt(0.0)
    return output


def route_evaluation(
    route: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score = route.loc[route["calendar_index"].ge(WARMUP_SESSIONS)].copy()
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    for seed_offset, (group_name, group) in enumerate(group_slices(score)):
        metric_rows.append(
            {
                "group": group_name,
                "rows": len(group),
                "model_log_loss": float(group["model_log_loss"].mean()),
                "prior_log_loss": float(group["prior_log_loss"].mean()),
                "log_loss_improvement": float(group["log_loss_improvement"].mean()),
                "model_brier": float(group["model_brier"].mean()),
                "prior_brier": float(group["prior_brier"].mean()),
                "model_accuracy": float(group["model_correct"].mean()),
                "prior_accuracy": float(group["prior_correct"].mean()),
            }
        )
        bootstrap_rows.append(
            {
                "family": "route_log_loss",
                "group": group_name,
                "rows": len(group),
                **paired_bootstrap(group, "log_loss_improvement", seed_offset),
            }
        )
        for month, month_group in group.groupby("month", sort=True):
            month_rows.append(
                {
                    "group": group_name,
                    "month": month,
                    "rows": len(month_group),
                    "log_loss_improvement": float(month_group["log_loss_improvement"].mean()),
                }
            )
        for symbol in sorted(score["symbol_norm"].unique()):
            deleted = group.loc[~group["symbol_norm"].eq(symbol)]
            deletion_rows.append(
                {
                    "group": group_name,
                    "deleted_symbol": symbol,
                    "rows": len(deleted),
                    "log_loss_improvement": float(deleted["log_loss_improvement"].mean()),
                }
            )
    return (
        pd.DataFrame(metric_rows),
        holm_adjust(pd.DataFrame(bootstrap_rows)),
        pd.DataFrame(month_rows),
        pd.DataFrame(deletion_rows),
    )


def admission_evaluation(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    seed_offset = 100
    working = predictions.copy()
    working["uncertainty_selected"] = working["uncertainty_class"].eq("positive")
    working["point_selected"] = working["point_positive"].astype(bool)
    for model, model_frame in working.groupby("model", sort=True):
        for group_name, group in group_slices(model_frame):
            error = group["predicted_net_bps"] - group["fixed_net_bps"]
            slope, intercept = calibration(group["fixed_net_bps"], group["predicted_net_bps"])
            model_rows.append(
                {
                    "model": model,
                    "group": group_name,
                    "rows": len(group),
                    "mae_bps": float(error.abs().mean()),
                    "rmse_bps": float(np.sqrt(np.mean(error**2))),
                    "baseline_rmse_bps": float(
                        np.sqrt(
                            np.mean(
                                (group["rolling_shrunk_mean_bps"] - group["fixed_net_bps"]) ** 2
                            )
                        )
                    ),
                    "spearman": spearman(group["fixed_net_bps"], group["predicted_net_bps"]),
                    "calibration_slope": slope,
                    "calibration_intercept_bps": intercept,
                    "positive_share": float(group["uncertainty_class"].eq("positive").mean()),
                    "negative_share": float(group["uncertainty_class"].eq("negative").mean()),
                    "unknown_share": float(group["uncertainty_class"].eq("unknown_abstain").mean()),
                }
            )
            for policy_type, selected_column in (
                ("uncertainty_aware", "uncertainty_selected"),
                ("point_mean_diagnostic", "point_selected"),
            ):
                evaluated = group.copy()
                selected = evaluated[selected_column].astype(bool)
                evaluated["selector_return_bps"] = np.where(
                    selected, evaluated["fixed_net_bps"], 0.0
                )
                evaluated["paired_difference_bps"] = (
                    evaluated["selector_return_bps"] - evaluated["fixed_net_bps"]
                )
                selector_rows.append(
                    {
                        "model": model,
                        "policy_type": policy_type,
                        "group": group_name,
                        "rows": len(evaluated),
                        "selected_rows": int(selected.sum()),
                        "coverage": float(selected.mean()),
                        "unfiltered_mean_net_bps": float(evaluated["fixed_net_bps"].mean()),
                        "selector_mean_per_opportunity_bps": float(
                            evaluated["selector_return_bps"].mean()
                        ),
                        "selected_mean_net_bps": (
                            float(evaluated.loc[selected, "fixed_net_bps"].mean())
                            if selected.any()
                            else math.nan
                        ),
                        "paired_difference_bps": float(evaluated["paired_difference_bps"].mean()),
                    }
                )
                if policy_type == "uncertainty_aware":
                    family = (
                        "direct_admission"
                        if model == "admission_only"
                        else "route_augmented_admission"
                    )
                    bootstrap_rows.append(
                        {
                            "family": family,
                            "model": model,
                            "group": group_name,
                            "rows": len(evaluated),
                            **paired_bootstrap(evaluated, "paired_difference_bps", seed_offset),
                        }
                    )
                    seed_offset += 1
                    for month, month_group in evaluated.groupby("month", sort=True):
                        month_rows.append(
                            {
                                "model": model,
                                "group": group_name,
                                "month": month,
                                "rows": len(month_group),
                                "coverage": float(month_group[selected_column].mean()),
                                "paired_difference_bps": float(
                                    np.where(
                                        month_group[selected_column],
                                        month_group["fixed_net_bps"],
                                        0.0,
                                    ).mean()
                                    - month_group["fixed_net_bps"].mean()
                                ),
                            }
                        )
                    for symbol in sorted(working["symbol_norm"].unique()):
                        deleted = evaluated.loc[~evaluated["symbol_norm"].eq(symbol)].copy()
                        selected_deleted = deleted[selected_column].astype(bool)
                        selector_return = np.where(selected_deleted, deleted["fixed_net_bps"], 0.0)
                        deletion_rows.append(
                            {
                                "model": model,
                                "group": group_name,
                                "deleted_symbol": symbol,
                                "rows": len(deleted),
                                "coverage": float(selected_deleted.mean()),
                                "paired_difference_bps": float(
                                    np.mean(selector_return - deleted["fixed_net_bps"])
                                ),
                            }
                        )
                    for cost in (2.5, 5.0, 7.5, 10.0):
                        net = evaluated["fixed_gross_bps"] - 2.0 * cost
                        selector_return = np.where(selected, net, 0.0)
                        cost_rows.append(
                            {
                                "model": model,
                                "group": group_name,
                                "cost_bps_per_side": cost,
                                "coverage": float(selected.mean()),
                                "unfiltered_mean_net_bps": float(net.mean()),
                                "selector_mean_per_opportunity_bps": float(selector_return.mean()),
                                "selected_mean_net_bps": (
                                    float(net.loc[selected].mean()) if selected.any() else math.nan
                                ),
                            }
                        )

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    adjusted = pd.concat(
        [
            holm_adjust(bootstrap_frame.loc[bootstrap_frame["family"].eq(family)])
            for family in ("direct_admission", "route_augmented_admission")
        ],
        ignore_index=True,
    )
    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(selector_rows),
        adjusted,
        pd.DataFrame(month_rows),
        pd.DataFrame(deletion_rows),
        pd.DataFrame(cost_rows),
    )


def sequential_evaluation(
    snapshot_predictions: pd.DataFrame,
    policies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    seed_offset = 200
    for model, model_frame in snapshot_predictions.groupby("model", sort=True):
        for group_name, group in group_slices(model_frame):
            error = group["predicted_hold_advantage_bps"] - group["hold_advantage_bps"]
            slope, intercept = calibration(
                group["hold_advantage_bps"], group["predicted_hold_advantage_bps"]
            )
            model_rows.append(
                {
                    "model": model,
                    "group": group_name,
                    "rows": len(group),
                    "signals": int(group["signal_id"].nunique()),
                    "mae_bps": float(error.abs().mean()),
                    "rmse_bps": float(np.sqrt(np.mean(error**2))),
                    "spearman": spearman(
                        group["hold_advantage_bps"], group["predicted_hold_advantage_bps"]
                    ),
                    "calibration_slope": slope,
                    "calibration_intercept_bps": intercept,
                    "positive_hold_share": float(
                        group["uncertainty_class"].eq("positive_hold").mean()
                    ),
                    "negative_exit_share": float(
                        group["uncertainty_class"].eq("negative_exit").mean()
                    ),
                    "unknown_share": float(group["uncertainty_class"].eq("unknown_abstain").mean()),
                }
            )
    for (model, policy_type), policy_frame in policies.groupby(["model", "policy_type"], sort=True):
        for group_name, group in group_slices(policy_frame):
            policy_rows.append(
                {
                    "model": model,
                    "policy_type": policy_type,
                    "group": group_name,
                    "rows": len(group),
                    "action_rows": int(group["action"].sum()),
                    "action_coverage": float(group["action"].mean()),
                    "fixed_mean_net_bps": float(group["fixed_net_bps"].mean()),
                    "policy_mean_net_bps": float(group["policy_net_bps"].mean()),
                    "paired_difference_bps": float(group["paired_difference_bps"].mean()),
                }
            )
            if policy_type == "uncertainty_aware":
                family = (
                    "sequential_route_plus_price"
                    if model == "sequential_route_plus_price_path"
                    else "sequential_route_only"
                )
                bootstrap_rows.append(
                    {
                        "family": family,
                        "model": model,
                        "group": group_name,
                        "rows": len(group),
                        **paired_bootstrap(group, "paired_difference_bps", seed_offset),
                    }
                )
                seed_offset += 1
                for month, month_group in group.groupby("month", sort=True):
                    month_rows.append(
                        {
                            "model": model,
                            "group": group_name,
                            "month": month,
                            "rows": len(month_group),
                            "action_coverage": float(month_group["action"].mean()),
                            "paired_difference_bps": float(
                                month_group["paired_difference_bps"].mean()
                            ),
                        }
                    )
                for symbol in sorted(policies["symbol_norm"].unique()):
                    deleted = group.loc[~group["symbol_norm"].eq(symbol)]
                    deletion_rows.append(
                        {
                            "model": model,
                            "group": group_name,
                            "deleted_symbol": symbol,
                            "rows": len(deleted),
                            "action_coverage": float(deleted["action"].mean()),
                            "paired_difference_bps": float(deleted["paired_difference_bps"].mean()),
                        }
                    )
                for cost in (2.5, 5.0, 7.5, 10.0):
                    fixed_net = group["fixed_gross_bps"] - 2.0 * cost
                    policy_net = group["policy_gross_bps"] - 2.0 * cost
                    cost_rows.append(
                        {
                            "model": model,
                            "group": group_name,
                            "cost_bps_per_side": cost,
                            "action_coverage": float(group["action"].mean()),
                            "fixed_mean_net_bps": float(fixed_net.mean()),
                            "policy_mean_net_bps": float(policy_net.mean()),
                            "paired_difference_bps": float((policy_net - fixed_net).mean()),
                        }
                    )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    adjusted = pd.concat(
        [
            holm_adjust(bootstrap_frame.loc[bootstrap_frame["family"].eq(family)])
            for family in ("sequential_route_plus_price", "sequential_route_only")
        ],
        ignore_index=True,
    )
    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(policy_rows),
        adjusted,
        pd.DataFrame(month_rows),
        pd.DataFrame(deletion_rows),
        pd.DataFrame(cost_rows),
    )


def positive_month_majority(frame: pd.DataFrame, model: str, group: str) -> bool:
    subset = frame.loc[frame["model"].eq(model) & frame["group"].eq(group)]
    return bool(len(subset) > 0 and subset["paired_difference_bps"].gt(0).sum() > len(subset) / 2)


def positive_deletions(frame: pd.DataFrame, model: str, group: str) -> int:
    subset = frame.loc[frame["model"].eq(model) & frame["group"].eq(group)]
    return int(subset["paired_difference_bps"].gt(0).sum())


def artifact_manifest(out: Path) -> dict[str, Any]:
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(
                item
                for item in out.iterdir()
                if item.is_file()
                and item.name not in {"artifact_manifest.json", "independent_audit.json"}
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-manifest", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    if args.freeze_manifest:
        if args.out is not None:
            raise ValueError("--out cannot accompany --freeze-manifest")
        freeze_manifest(contract)
        print(json.dumps({"frozen": str(PRE_SCORE_PATH)}, indent=2))
        return
    if args.out is None:
        raise ValueError("--out is required")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    source_hashes = verify_frozen_sources(contract)
    args.out.mkdir(parents=True)

    raw_surface, calendar, tape, run_groups, coverage = load_surface(contract)
    surface = enrich_surface(raw_surface, calendar, tape, run_groups)
    route = prequential_route_predictions(surface, calendar)
    surface = attach_route_probabilities(surface, route)
    admission = prequential_admission_predictions(surface, calendar)
    snapshots = build_snapshots(surface, tape, run_groups)
    snapshot_predictions = prequential_snapshot_predictions(snapshots, calendar)
    score_surface = surface.loc[surface["score_eligible"]].copy()
    sequential_policies = build_sequential_policies(score_surface, snapshot_predictions)

    route_metrics, route_bootstraps, route_months, route_deletions = route_evaluation(route)
    (
        admission_metrics,
        admission_selectors,
        admission_bootstraps,
        admission_months,
        admission_deletions,
        admission_costs,
    ) = admission_evaluation(admission)
    (
        sequential_metrics,
        sequential_policy_metrics,
        sequential_bootstraps,
        sequential_months,
        sequential_deletions,
        sequential_costs,
    ) = sequential_evaluation(snapshot_predictions, sequential_policies)

    population_rows = []
    for group_name, group in group_slices(surface):
        population_rows.append(
            {
                "group": group_name,
                "full_rows": len(group),
                "full_sessions": group["session_date"].nunique(),
                "score_rows": int(group["score_eligible"].sum()),
                "score_sessions": group.loc[group["score_eligible"], "session_date"].nunique(),
                "symbols": group["symbol_norm"].nunique(),
            }
        )
    population = pd.DataFrame(population_rows)

    route_metric_map = route_metrics.set_index("group")
    route_boot_map = route_bootstraps.set_index("group")
    direct_selector = admission_selectors.loc[
        admission_selectors["model"].eq("admission_only")
        & admission_selectors["policy_type"].eq("uncertainty_aware")
    ].set_index("group")
    direct_boot = admission_bootstraps.loc[
        admission_bootstraps["family"].eq("direct_admission")
    ].set_index("group")
    admission_metric_map = admission_metrics.set_index(["model", "group"])
    plus_selector = admission_selectors.loc[
        admission_selectors["model"].eq("admission_plus_causal_route_probabilities")
        & admission_selectors["policy_type"].eq("uncertainty_aware")
    ].set_index("group")
    sequential_primary = sequential_policy_metrics.loc[
        sequential_policy_metrics["model"].eq("sequential_route_plus_price_path")
        & sequential_policy_metrics["policy_type"].eq("uncertainty_aware")
    ].set_index("group")
    sequential_route_only = sequential_policy_metrics.loc[
        sequential_policy_metrics["model"].eq("sequential_route_state_only")
        & sequential_policy_metrics["policy_type"].eq("uncertainty_aware")
    ].set_index("group")
    sequential_boot_map = sequential_bootstraps.loc[
        sequential_bootstraps["family"].eq("sequential_route_plus_price")
    ].set_index("group")

    groups = ("pooled", "cycle_04|state4", "cycle_07|state5")
    route_checks = {
        "support_at_least_50_all_groups": all(
            route_metric_map.loc[group, "rows"] >= 50 for group in groups
        ),
        "log_loss_improvement_positive_all_groups": all(
            route_metric_map.loc[group, "log_loss_improvement"] > 0 for group in groups
        ),
        "block_lower_positive_and_holm_all_groups": all(
            bool(route_boot_map.loc[group, "passes_holm_0_05"]) for group in groups
        ),
        "brier_lower_than_prior_all_groups": all(
            route_metric_map.loc[group, "model_brier"] < route_metric_map.loc[group, "prior_brier"]
            for group in groups
        ),
        "positive_month_majority_both_candidates": all(
            (
                route_months.loc[route_months["group"].eq(group), "log_loss_improvement"]
                .gt(0)
                .sum()
                > len(route_months.loc[route_months["group"].eq(group)]) / 2
            )
            for group in groups[1:]
        ),
    }
    direct_checks = {
        "support_at_least_50_all_groups": all(
            direct_selector.loc[group, "rows"] >= 50 for group in groups
        ),
        "positive_coverage_at_least_5pct_all_groups": all(
            direct_selector.loc[group, "coverage"] >= 0.05 for group in groups
        ),
        "selected_mean_net_positive_all_groups": all(
            direct_selector.loc[group, "selected_mean_net_bps"] > 0 for group in groups
        ),
        "paired_mean_positive_all_groups": all(
            direct_selector.loc[group, "paired_difference_bps"] > 0 for group in groups
        ),
        "block_lower_positive_and_holm_all_groups": all(
            bool(direct_boot.loc[group, "passes_holm_0_05"]) for group in groups
        ),
        "positive_month_majority_both_candidates": all(
            positive_month_majority(admission_months, "admission_only", group)
            for group in groups[1:]
        ),
        "at_least_16_positive_stock_deletions_pooled": positive_deletions(
            admission_deletions, "admission_only", "pooled"
        )
        >= 16,
        "forecast_spearman_positive_all_groups": all(
            admission_metric_map.loc[("admission_only", group), "spearman"] > 0 for group in groups
        ),
    }
    route_increment_checks = {
        "plus_route_rmse_lower_both_candidates": all(
            admission_metric_map.loc[
                ("admission_plus_causal_route_probabilities", group), "rmse_bps"
            ]
            < admission_metric_map.loc[("admission_only", group), "rmse_bps"]
            for group in groups[1:]
        ),
        "plus_route_selector_higher_both_candidates": all(
            plus_selector.loc[group, "selector_mean_per_opportunity_bps"]
            > direct_selector.loc[group, "selector_mean_per_opportunity_bps"]
            for group in groups[1:]
        ),
    }
    sequential_checks = {
        "support_at_least_50_all_groups": all(
            sequential_primary.loc[group, "rows"] >= 50 for group in groups
        ),
        "exit_coverage_at_least_5pct_all_groups": all(
            sequential_primary.loc[group, "action_coverage"] >= 0.05 for group in groups
        ),
        "paired_mean_positive_all_groups": all(
            sequential_primary.loc[group, "paired_difference_bps"] > 0 for group in groups
        ),
        "block_lower_positive_and_holm_all_groups": all(
            bool(sequential_boot_map.loc[group, "passes_holm_0_05"]) for group in groups
        ),
        "route_plus_price_higher_than_route_only_both_candidates": all(
            sequential_primary.loc[group, "policy_mean_net_bps"]
            > sequential_route_only.loc[group, "policy_mean_net_bps"]
            for group in groups[1:]
        ),
        "absolute_policy_mean_net_positive_all_groups": all(
            sequential_primary.loc[group, "policy_mean_net_bps"] > 0 for group in groups
        ),
        "positive_month_majority_both_candidates": all(
            positive_month_majority(sequential_months, "sequential_route_plus_price_path", group)
            for group in groups[1:]
        ),
        "at_least_16_positive_stock_deletions_pooled": positive_deletions(
            sequential_deletions, "sequential_route_plus_price_path", "pooled"
        )
        >= 16,
    }
    decisions = {
        "direct_payoff_state": (
            "supported_for_prospective_logging_only"
            if all(direct_checks.values())
            else "rejected_or_unknown"
        ),
        "route_branch_forecast": (
            "supported_as_auxiliary_only" if all(route_checks.values()) else "rejected_or_unknown"
        ),
        "predicted_route_increment": (
            "supported_as_auxiliary_only"
            if all(route_increment_checks.values())
            else "rejected_or_unknown"
        ),
        "sequential_route_plus_price": (
            "supported_for_prospective_logging_only"
            if all(sequential_checks.values())
            else "rejected_or_unknown"
        ),
        "diversion_specific_payoff": "deferred_no_sealed_data",
    }
    decision = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "application_modified": False,
        "sealed_validation_performed": False,
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "development_surface": contract["population"]["surface"],
        "primary_cost_bps_per_side": 5.0,
        "route_checks": route_checks,
        "direct_checks": direct_checks,
        "route_increment_checks": route_increment_checks,
        "sequential_checks": sequential_checks,
        "decisions": decisions,
        "maximum_allowed_action": "prospective immutable research logging only",
    }

    population.to_csv(args.out / "population.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.out / "data_coverage.csv", index=False)
    surface.to_parquet(args.out / "research_surface.parquet", index=False)
    route.to_parquet(args.out / "route_predictions.parquet", index=False)
    route_metrics.to_csv(args.out / "route_forecast_metrics.csv", index=False)
    route_bootstraps.to_csv(args.out / "route_bootstraps.csv", index=False)
    route_months.to_csv(args.out / "route_month_metrics.csv", index=False)
    route_deletions.to_csv(args.out / "route_stock_deletions.csv", index=False)
    admission.to_parquet(args.out / "admission_predictions.parquet", index=False)
    admission_metrics.to_csv(args.out / "admission_model_metrics.csv", index=False)
    admission_selectors.to_csv(args.out / "admission_selector_metrics.csv", index=False)
    admission_bootstraps.to_csv(args.out / "admission_bootstraps.csv", index=False)
    admission_months.to_csv(args.out / "admission_month_metrics.csv", index=False)
    admission_deletions.to_csv(args.out / "admission_stock_deletions.csv", index=False)
    admission_costs.to_csv(args.out / "admission_cost_sensitivity.csv", index=False)
    snapshots.to_parquet(args.out / "causal_snapshots.parquet", index=False)
    snapshot_predictions.to_parquet(args.out / "snapshot_predictions.parquet", index=False)
    sequential_policies.to_parquet(args.out / "sequential_policy_rows.parquet", index=False)
    sequential_metrics.to_csv(args.out / "sequential_model_metrics.csv", index=False)
    sequential_policy_metrics.to_csv(args.out / "sequential_policy_metrics.csv", index=False)
    sequential_bootstraps.to_csv(args.out / "sequential_bootstraps.csv", index=False)
    sequential_months.to_csv(args.out / "sequential_month_metrics.csv", index=False)
    sequential_deletions.to_csv(args.out / "sequential_stock_deletions.csv", index=False)
    sequential_costs.to_csv(args.out / "sequential_cost_sensitivity.csv", index=False)
    write_json(args.out / "decision.json", decision)
    write_json(
        args.out / "summary.json",
        {
            "contract_id": contract["contract_id"],
            "scientific_status": contract["scientific_status"],
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "sealed_data_status": contract["sealed_data_status"],
            "provider_volume_label": contract["provenance"]["volume"],
            "population": population.to_dict("records"),
            "route_metrics": route_metrics.to_dict("records"),
            "direct_model_metrics": admission_metrics.to_dict("records"),
            "direct_selector_metrics": admission_selectors.to_dict("records"),
            "sequential_model_metrics": sequential_metrics.to_dict("records"),
            "sequential_policy_metrics": sequential_policy_metrics.to_dict("records"),
            "decision": decision,
        },
    )
    write_json(
        args.out / "source_hashes.json",
        {
            "contract_id": contract["contract_id"],
            "frozen_before_model_scoring": True,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "sha256": source_hashes,
        },
    )
    write_json(args.out / "artifact_manifest.json", artifact_manifest(args.out))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "full_rows": len(surface),
                "score_rows": len(score_surface),
                "route_prediction_rows": len(route),
                "admission_prediction_rows": len(admission),
                "snapshot_prediction_rows": len(snapshot_predictions),
                "decisions": decisions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
