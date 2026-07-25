"""Independent auditor for Frozen Named-Loop T0 Execution Realism V1."""

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
CONTRACT_PATH = WORK / "contracts/20260717-frozen-named-loop-t0-execution-realism-v1.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260717-frozen-named-loop-t0-execution-realism-v1/primary"
DEFAULT_EXACT = WORK / "artifacts/20260717-frozen-named-loop-t0-execution-realism-v1/exact_rerun"
FROZEN_HANDOFF_COMMIT = "9b0fcf7"
EXCLUSIONS = {"independent_audit.json", "exact_rerun_identity.json"}
STRESSES = {"F0": 0.0, "F5": 5.0, "F10": 10.0, "F15": 15.0, "F20": 20.0}
FAMILIES = {
    "cycle_04|state_4": ("named", "cycle_04", "state_4", 4),
    "cycle_04|state_2": ("control", "cycle_04", "state_2", 2),
    "cycle_07|state_5": ("named", "cycle_07", "state_5", 5),
    "cycle_07|state_6": ("control", "cycle_07", "state_6", 6),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _resolved(value: object) -> Path:
    return (CONTRACT_PATH.parent / str(value)).resolve()


def _close(left: object, right: object, tolerance: float = 1e-8) -> bool:
    return math.isclose(
        float(cast(Any, left)), float(cast(Any, right)), rel_tol=0.0, abs_tol=tolerance
    )


def _performance(frame: pd.DataFrame) -> dict[str, float | int]:
    values = pd.to_numeric(frame["net_payoff_bps"], errors="coerce").dropna()
    positive = float(values.loc[values.gt(0.0)].sum())
    negative = float(-values.loc[values.lt(0.0)].sum())
    cumulative = values.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "opportunities": len(values),
        "total_net_payoff_bps": float(values.sum()),
        "mean_net_payoff_bps": float(values.mean()),
        "median_net_payoff_bps": float(values.median()),
        "positive_payoff_rate": float(values.gt(0.0).mean()),
        "profit_factor": positive / negative if negative else float("nan"),
        "maximum_drawdown_bps": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def _cohort(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all_reference":
        return frame
    if name == "primary_valid_fill":
        return frame.loc[frame["primary_valid_fill_evidence"]]
    if name == "bounded_or_timing_ambiguous":
        return frame.loc[frame["fill_evidence_classification"].eq("BOUNDED_BUT_NOT_EXACT")]
    raise ValueError(f"unknown evidence cohort: {name}")


def _check_contract_and_artifact_identity(
    primary: Path, contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[bool, str]:
    if sha256(CONTRACT_PATH) != str(metadata["contract_hash"]):
        return False, "run metadata contract hash differs"
    mapping_path = _resolved(contract["inputs"]["family_mapping"]["path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    observed = {
        str(row["family"]): (
            str(row["classification"]),
            str(row["loop_id"]),
            str(row["orientation"]),
            int(row["current_state"]),
        )
        for row in mapping["families"]
    }
    if observed != FAMILIES:
        return False, "frozen family mapping differs"
    checked = 0
    for name, specification in contract["inputs"].items():
        if not isinstance(specification, Mapping) or "path" not in specification:
            continue
        path = _resolved(specification["path"])
        actual = sha256(path)
        if actual != str(specification["sha256"]):
            return False, f"frozen input drift: {name}"
        if str(metadata["input_hashes"].get(name)) != actual:
            return False, f"metadata input hash mismatch: {name}"
        checked += 1
    for relative, expected in cast(Mapping[str, str], metadata["code_hashes"]).items():
        path = REPO / relative
        if not path.is_file() or sha256(path) != expected:
            return False, f"recorded code identity differs: {relative}"
    manifest = json.loads((primary / "artifact_manifest.json").read_text())
    for item in manifest["files"]:
        path = primary / str(item["path"])
        if not path.is_file() or sha256(path) != str(item["sha256"]):
            return False, f"artifact manifest mismatch: {item['path']}"
    return True, f"hashed contract, mapping, {checked} inputs, code, and artifact manifest"


def _check_provider_hashes(
    contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[bool, str]:
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_2025_hash_manifest"]["path"]).read_text()
    )["sha256"]
    root = Path(str(contract["inputs"]["provider_2025_root"]))
    checked = 0
    for key, expected in sorted(manifest.items()):
        if not str(key).startswith("provider_2025_"):
            continue
        symbol = str(key).removeprefix("provider_2025_")
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        actual = sha256(path)
        if actual != str(expected) or actual != str(metadata["provider_hashes"].get(symbol)):
            return False, f"provider identity mismatch: {symbol}"
        checked += 1
    return checked == 20, f"independently hashed {checked} provider tapes"


def _check_population(primary: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    actual = pd.concat(
        [
            pd.read_parquet(primary / "historical_named_reference_ledger.parquet"),
            pd.read_parquet(primary / "historical_control_reference_ledger.parquet"),
        ],
        ignore_index=True,
    )
    policy = pd.read_parquet(
        _resolved(contract["inputs"]["sequential_veto_population_identity"]["path"])
    )
    named = policy.loc[
        policy["track"].eq("track_a_named_family")
        & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
        & policy["period"].eq(2025)
    ]
    controls = policy.loc[
        policy["track"].eq("track_b_prior_only")
        & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
        & policy["population_role"].isin(["neutral_control", "negative_control"])
        & policy["period"].eq(2025)
    ]
    expected = pd.concat([named, controls], ignore_index=True)
    if set(actual["opportunity_id"].astype(str)) != set(expected["opportunity_id"].astype(str)):
        return False, "historical opportunity population differs from frozen policy"
    joined = actual.set_index("opportunity_id").join(
        expected.set_index("opportunity_id")[
            ["event_lineage_id", "session_date", "stock", "target_loop", "orientation"]
        ],
        rsuffix="_policy",
        validate="one_to_one",
    )
    checks = [
        joined["event_lineage_id"].astype(str).eq(joined["event_lineage_id_policy"].astype(str)),
        joined["session"].astype(str).eq(joined["session_date"].astype(str)),
        joined["symbol"].astype(str).eq(joined["stock"].astype(str)),
        joined["loop_id"].astype(str).eq(joined["target_loop"].astype(str)),
        joined["orientation"].astype(str).eq(joined["orientation_policy"].astype(str)),
    ]
    if not all(bool(check.all()) for check in checks):
        return False, "policy lineage, symbol, session, loop, or orientation differs"
    counts = actual.groupby("family").size().to_dict()
    if counts != {
        "cycle_04|state_2": 6,
        "cycle_04|state_4": 96,
        "cycle_07|state_5": 713,
        "cycle_07|state_6": 296,
    }:
        return False, f"frozen 2025 family count drift: {counts}"
    return True, "all 1,111 family, pair, lineage, and policy identities match"


def _provider(contract: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    path = (
        Path(str(contract["inputs"]["provider_2025_root"]))
        / f"symbol={symbol}"
        / "timeframe=5m/data.parquet"
    )
    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    return frame


def _first_oco(
    provider: pd.DataFrame, anchor: pd.Timestamp, upper: float, lower: float
) -> dict[str, object]:
    for step in range(1, 25):
        timestamp = anchor + pd.Timedelta(minutes=5 * step)
        rows = provider.loc[provider["timestamp"].eq(timestamp)]
        if len(rows) != 1:
            raise AssertionError(f"missing exact provider bar at {timestamp}")
        bar = rows.iloc[0]
        opening_long = float(bar["open"]) >= upper
        opening_short = float(bar["open"]) <= lower
        if opening_long and opening_short:
            raise AssertionError("impossible dual opening gap")
        if opening_long or opening_short:
            direction = 1 if opening_long else -1
            return {
                "step": step,
                "timestamp": timestamp,
                "direction": direction,
                "price": max(upper, float(bar["open"]))
                if direction == 1
                else min(lower, float(bar["open"])),
                "type": "opening_gap_through_threshold",
                "evidence": "GAP_FILL_OBSERVABLE",
                "signal_known": timestamp,
                "available": timestamp,
                "bar": bar,
            }
        long_hit = float(bar["high"]) >= upper
        short_hit = float(bar["low"]) <= lower
        if long_hit and short_hit:
            raise AssertionError("ambiguous dual-side OCO preceded a stored source fill")
        if long_hit or short_hit:
            direction = 1 if long_hit else -1
            return {
                "step": step,
                "timestamp": timestamp,
                "direction": direction,
                "price": upper if direction == 1 else lower,
                "type": "intrabar_threshold_cross",
                "evidence": "BOUNDED_BUT_NOT_EXACT",
                "signal_known": None,
                "available": timestamp + pd.Timedelta(minutes=5),
                "bar": bar,
            }
    raise AssertionError("stored source opportunity has no trigger in 24 bars")


def _check_trigger_fill_terminal_and_payoffs(
    primary: Path, contract: Mapping[str, Any]
) -> tuple[bool, str]:
    reference = pd.concat(
        [
            pd.read_parquet(primary / "historical_named_reference_ledger.parquet"),
            pd.read_parquet(primary / "historical_control_reference_ledger.parquet"),
        ],
        ignore_index=True,
    )
    payoff = pd.read_parquet(primary / "payoff_envelope_ledger.parquet")
    v2 = pd.read_parquet(_resolved(contract["inputs"]["v2_trade_decisions"]["path"]))
    v2 = v2.loc[v2["model_name"].eq("no_payoff_state_filter")]
    v2_by_opportunity = {
        str(raw["opportunity_id"]): cast(dict[str, Any], raw)
        for raw in v2.to_dict(orient="records")
    }
    payoff_by_key = {
        (str(raw["opportunity_id"]), str(raw["fill_model"])): cast(dict[str, Any], raw)
        for raw in payoff.to_dict(orient="records")
    }
    providers: dict[str, pd.DataFrame] = {}
    for raw in reference.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        opportunity_id = str(row["opportunity_id"])
        source = v2_by_opportunity[opportunity_id]
        symbol = str(row["symbol"])
        providers.setdefault(symbol, _provider(contract, symbol))
        anchor = pd.Timestamp(cast(Any, row["anchor_timestamp"]))
        event = _first_oco(
            providers[symbol], anchor, float(row["anchor_high"]), float(row["anchor_low"])
        )
        if (
            int(cast(Any, event["direction"])) != int(row["direction"])
            or int(cast(Any, event["step"])) != int(row["entry_step"])
            or pd.Timestamp(cast(Any, event["timestamp"]))
            != pd.Timestamp(cast(Any, row["reference_entry_timestamp"]))
            or not _close(event["price"], row["reference_entry_price"], 1e-12)
            or str(event["type"]) != str(row["trigger_type"])
            or str(event["evidence"]) != str(row["fill_evidence_classification"])
        ):
            return False, f"independent trigger or fill mismatch: {opportunity_id}"
        event_bar = cast(pd.Series, event["bar"])
        for field, source_field in (
            ("trigger_bar_open", "open"),
            ("trigger_bar_high", "high"),
            ("trigger_bar_low", "low"),
            ("trigger_bar_close", "close"),
        ):
            if not _close(row[field], event_bar[source_field], 1e-12):
                return False, f"trigger OHLC mismatch: {opportunity_id}:{field}"
        signal = event["signal_known"]
        observed_signal = row["signal_known_timestamp"]
        if signal is None:
            if not pd.isna(observed_signal):
                return False, f"bounded fill incorrectly has an exact signal time: {opportunity_id}"
        elif pd.Timestamp(cast(Any, signal)) != pd.Timestamp(cast(Any, observed_signal)):
            return False, f"gap signal-known timestamp mismatch: {opportunity_id}"
        if pd.Timestamp(cast(Any, event["available"])) != pd.Timestamp(
            cast(Any, row["market_data_availability_timestamp"])
        ):
            return False, f"market-data availability clock mismatch: {opportunity_id}"
        terminal_timestamp = anchor + pd.Timedelta(minutes=125)
        terminal_bar = providers[symbol].loc[
            providers[symbol]["timestamp"].eq(terminal_timestamp - pd.Timedelta(minutes=5))
        ]
        if len(terminal_bar) != 1:
            return False, f"terminal provider bar missing: {opportunity_id}"
        terminal_price = float(terminal_bar.iloc[0]["close"])
        if (
            terminal_timestamp != pd.Timestamp(cast(Any, row["original_terminal_timestamp"]))
            or not _close(terminal_price, row["terminal_price"], 1e-12)
            or not _close(row["reference_entry_price"], source["entry_price"], 1e-12)
        ):
            return False, f"source entry or unchanged terminal mismatch: {opportunity_id}"
        for model, stress in STRESSES.items():
            observed = payoff_by_key[(opportunity_id, model)]
            direction = int(row["direction"])
            factor = 1.0 + stress / 10_000.0 if direction == 1 else 1.0 - stress / 10_000.0
            stressed = float(row["reference_entry_price"]) * factor
            gross = 10_000.0 * direction * (terminal_price / stressed - 1.0)
            expected = (stressed, gross, 10.0, gross - 10.0)
            actual = (
                observed["stressed_entry_price"],
                observed["gross_payoff_bps"],
                observed["cost_bps"],
                observed["net_payoff_bps"],
            )
            if not all(_close(left, right) for left, right in zip(expected, actual, strict=True)):
                return False, f"independent {model} payoff mismatch: {opportunity_id}"
            if (
                pd.Timestamp(cast(Any, observed["original_terminal_timestamp"]))
                != terminal_timestamp
            ):
                return False, f"stress model changed terminal: {opportunity_id}:{model}"
    evidence = reference["fill_evidence_classification"].value_counts().to_dict()
    return True, f"reconstructed all 1,111 triggers/terminals and 5,555 fills; evidence={evidence}"


def _check_family_metrics(primary: Path) -> tuple[bool, str]:
    payoff = pd.read_parquet(primary / "payoff_envelope_ledger.parquet")
    tables = pd.concat(
        [
            pd.read_csv(primary / "named_family_metrics.csv"),
            pd.read_csv(primary / "control_family_metrics.csv"),
        ],
        ignore_index=True,
    )
    checked = 0
    for raw in tables.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = _cohort(payoff, str(row["evidence_cohort"]))
        selected = selected.loc[
            selected["family"].eq(str(row["family"]))
            & selected["fill_model"].eq(str(row["fill_model"]))
        ]
        expected = _performance(selected)
        for key, value in expected.items():
            observed = row[key]
            if isinstance(value, int):
                if int(observed) != value:
                    return False, f"family count mismatch: {row['family']}:{row['fill_model']}"
            elif not (_close(observed, value) or (pd.isna(observed) and math.isnan(value))):
                return False, f"family metric mismatch: {row['family']}:{row['fill_model']}:{key}"
        if int(row["independent_stocks"]) != selected["symbol"].nunique():
            return False, "independent stock count mismatch"
        if (
            int(row["sessions"]) != selected["session"].nunique()
            or int(row["months"]) != selected["month"].nunique()
        ):
            return False, "session or month count mismatch"
        checked += 1
    comparisons = pd.read_csv(primary / "named_versus_control_comparisons.csv")
    for raw in comparisons.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = _cohort(payoff, str(row["evidence_cohort"]))
        named = selected.loc[
            selected["family"].eq(str(row["named_family"]))
            & selected["fill_model"].eq(str(row["fill_model"]))
        ]
        control = selected.loc[
            selected["family"].eq(str(row["control_family"]))
            & selected["fill_model"].eq(str(row["fill_model"]))
        ]
        if str(row["comparison"]).startswith("combined_"):
            named = selected.loc[
                selected["classification"].eq("named")
                & selected["fill_model"].eq(str(row["fill_model"]))
            ]
            control = selected.loc[
                selected["classification"].eq("control")
                & selected["fill_model"].eq(str(row["fill_model"]))
            ]
        if not _close(
            float(named["net_payoff_bps"].mean()) - float(control["net_payoff_bps"].mean()),
            row["mean_difference_bps"],
        ):
            return False, f"named-control comparison mismatch: {row['comparison']}"
        checked += 1
    return True, f"independently reconstructed {checked} family and comparison aggregates"


def _check_reference_decay_and_quality(primary: Path) -> tuple[bool, str]:
    payoff = pd.read_parquet(primary / "payoff_envelope_ledger.parquet")
    decay = pd.read_csv(primary / "execution_decay_curve.csv")
    checked = 0
    for raw in decay.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = _cohort(payoff, str(row["evidence_cohort"]))
        selected = selected.loc[
            selected["classification"].eq(str(row["classification"]))
            & selected["family"].eq(str(row["family"]))
            & selected["fill_model"].eq(str(row["fill_model"]))
        ]
        expected = _performance(selected)
        if not _close(row["total_net_payoff_bps"], expected["total_net_payoff_bps"]) or not _close(
            row["mean_net_payoff_bps"], expected["mean_net_payoff_bps"]
        ):
            return False, f"execution decay mismatch: {row['family']}:{row['fill_model']}"
        if not _close(row["adverse_entry_slippage_bps"], STRESSES[str(row["fill_model"])]):
            return False, f"execution stress label mismatch: {row['fill_model']}"
        checked += 1
    reference = pd.read_csv(primary / "historical_reference_metrics.csv")
    f0 = payoff.loc[payoff["fill_model"].eq("F0")]
    for classification in ("named", "control"):
        reference_row = cast(
            dict[str, Any],
            reference.loc[
                reference["reference"].eq("full_2025_t0_source")
                & reference["classification"].eq(classification)
            ]
            .iloc[0]
            .to_dict(),
        )
        expected = _performance(f0.loc[f0["classification"].eq(classification)])
        if int(reference_row["opportunities"]) != expected["opportunities"] or not _close(
            reference_row["total_net_payoff_bps"], expected["total_net_payoff_bps"]
        ):
            return False, f"historical reference metric mismatch: {classification}"
        checked += 1
    quality = json.loads((primary / "data_quality_report.json").read_text())
    triggers = pd.read_parquet(primary / "trigger_reconstruction_ledger.parquet")
    counts = {
        str(key): int(value)
        for key, value in triggers["fill_evidence_classification"]
        .value_counts()
        .sort_index()
        .items()
    }
    if quality["fill_evidence_counts"] != counts:
        return False, "fill-evidence quality counts differ from trigger ledger"
    if not _close(
        quality["primary_valid_fill_coverage"],
        triggers["primary_valid_fill_evidence"].mean(),
        1e-12,
    ):
        return False, "valid-fill coverage mismatch"
    reconciliation = json.loads((primary / "historical_reconciliation_checks.json").read_text())
    if reconciliation["exact_t0_reconstructions"] != 1111 or not _close(
        reconciliation["f0_total_net_payoff_bps"], f0["net_payoff_bps"].sum()
    ):
        return False, "historical reconciliation summary mismatch"
    return True, f"reconstructed {checked} reference/decay rows and all quality counts"


def _session_interval(frame: pd.DataFrame) -> dict[str, float | int]:
    values = frame.groupby("session", sort=True)["net_payoff_bps"].mean().to_numpy(float)
    rng = np.random.default_rng(20260717)
    draws = np.empty(2000)
    blocks = int(np.ceil(len(values) / 5))
    for draw in range(2000):
        starts = rng.integers(0, len(values), size=blocks)
        rebuilt = np.asarray(
            [values[(int(start) + offset) % len(values)] for start in starts for offset in range(5)]
        )[: len(values)]
        draws[draw] = rebuilt.mean()
    return {
        "sessions": len(values),
        "observed_session_mean_bps": float(values.mean()),
        "sessions_positive_percentage": float(100.0 * np.mean(values > 0.0)),
        "bootstrap_lower_95_bps": float(np.quantile(draws, 0.025)),
        "bootstrap_upper_95_bps": float(np.quantile(draws, 0.975)),
    }


def _mean_at_stress(frame: pd.DataFrame, stress: float) -> float:
    direction = frame["direction"].to_numpy(int)
    factor = np.where(direction == 1, 1.0 + stress / 10_000.0, 1.0 - stress / 10_000.0)
    net = 10_000.0 * direction * (
        frame["terminal_price"].to_numpy(float)
        / (frame["reference_entry_price"].to_numpy(float) * factor)
        - 1.0
    ) - frame["cost_bps"].to_numpy(float)
    return float(net.mean())


def _break_even(frame: pd.DataFrame) -> float:
    if frame.empty or _mean_at_stress(frame, 0.0) <= 0.0:
        return 0.0
    low, high = 0.0, 25.0
    while high < 9_000.0 and _mean_at_stress(frame, high) > 0.0:
        high *= 2.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _mean_at_stress(frame, midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= 1e-9:
            break
    return (low + high) / 2.0


def _break_even_interval(frame: pd.DataFrame) -> tuple[float, float, float]:
    sessions = sorted(frame["session"].astype(str).unique())
    groups = {
        session: frame.loc[frame["session"].astype(str).eq(session)].copy() for session in sessions
    }
    rng = np.random.default_rng(20260717)
    draws = np.empty(2000)
    blocks = int(np.ceil(len(sessions) / 5))
    for draw in range(2000):
        starts = rng.integers(0, len(sessions), size=blocks)
        sampled_sessions = [
            sessions[(int(start) + offset) % len(sessions)]
            for start in starts
            for offset in range(5)
        ][: len(sessions)]
        sampled = pd.concat([groups[session] for session in sampled_sessions], ignore_index=True)
        draws[draw] = _break_even(sampled)
    return (
        _break_even(frame),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _check_inference_and_break_even(primary: Path) -> tuple[bool, str]:
    payoff = pd.read_parquet(primary / "payoff_envelope_ledger.parquet")
    intervals = pd.read_csv(primary / "session_block_intervals.csv")
    checked = 0
    for raw in intervals.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = _cohort(payoff, str(row["evidence_cohort"]))
        selected = selected.loc[
            selected["family"].eq(str(row["family"]))
            & selected["fill_model"].eq(str(row["fill_model"]))
        ]
        interval_expected = _session_interval(selected)
        for key, value in interval_expected.items():
            if not _close(row[key], value):
                return (
                    False,
                    f"session interval mismatch: {row['family']}:{row['fill_model']}:{key}",
                )
        checked += 1
    f0 = payoff.loc[payoff["fill_model"].eq("F0")]
    break_even = pd.read_csv(primary / "break_even_slippage_results.csv")
    for raw in break_even.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = f0.loc[f0["family"].eq(str(row["family"]))]
        if str(row["evidence_cohort"]) == "primary_valid_fill":
            selected = selected.loc[selected["primary_valid_fill_evidence"]]
        if str(row["scope"]) in {"symbol", "month"}:
            selected = selected.loc[
                selected[str(row["scope"])].astype(str).eq(str(row["scope_value"]))
            ]
        point = _break_even(selected)
        if not _close(point, row["break_even_adverse_slippage_bps"], 1e-7):
            return False, f"break-even point mismatch: {row['family']}:{row['evidence_cohort']}"
        if str(row["scope"]) == "overall":
            _, lower, upper = _break_even_interval(selected)
            if not _close(lower, row["bootstrap_lower_95_bps"], 1e-7) or not _close(
                upper, row["bootstrap_upper_95_bps"], 1e-7
            ):
                return (
                    False,
                    f"break-even bootstrap mismatch: {row['family']}:{row['evidence_cohort']}",
                )
        checked += 1
    return True, f"reconstructed {checked} session-block and break-even rows"


def _check_breakdowns_and_concentration(primary: Path) -> tuple[bool, str]:
    payoff = pd.read_parquet(primary / "payoff_envelope_ledger.parquet")
    breakdowns = pd.read_csv(primary / "stock_and_month_breakdowns.csv")
    checked = 0
    for raw in breakdowns.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = payoff.loc[
            payoff["family"].eq(str(row["family"]))
            & payoff["fill_model"].eq(str(row["fill_model"]))
            & payoff[str(row["slice_type"])].astype(str).eq(str(row["slice_value"]))
        ]
        metrics_expected = _performance(selected)
        if int(row["opportunities"]) != metrics_expected["opportunities"] or not _close(
            row["total_net_payoff_bps"], metrics_expected["total_net_payoff_bps"]
        ):
            return (
                False,
                f"breakdown mismatch: {row['family']}:{row['slice_type']}:{row['slice_value']}",
            )
        checked += 1
    f10 = payoff.loc[payoff["fill_model"].eq("F10")]
    concentration = pd.read_csv(primary / "concentration_results.csv")
    for raw in concentration.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = f10.loc[f10["family"].eq(str(row["family"]))]
        grouped = (
            selected.groupby(str(row["dimension"]), dropna=False, sort=True)["net_payoff_bps"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )
        shares = grouped / grouped.sum()
        concentration_expected = (
            float(shares.iloc[:1].sum()),
            float(shares.iloc[:5].sum()),
            float(np.square(shares).sum()),
        )
        actual = (
            row["top_one_absolute_contribution_share"],
            row["top_five_absolute_contribution_share"],
            row["herfindahl_index"],
        )
        if not all(
            _close(left, right, 1e-10)
            for left, right in zip(concentration_expected, actual, strict=True)
        ):
            return False, f"concentration mismatch: {row['family']}:{row['dimension']}"
        checked += 1
    deletions = pd.read_csv(primary / "dominant_contributor_removals.csv")
    for raw in deletions.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = f10.loc[f10["family"].eq(str(row["family"]))]
        contributions = (
            selected.groupby(str(row["dimension"]), dropna=False, sort=True)["net_payoff_bps"]
            .sum()
            .sort_values(ascending=False)
        )
        removed = [str(value) for value in contributions.index[: int(row["top_n"])]]
        remaining = selected.loc[~selected[str(row["dimension"])].astype(str).isin(removed)]
        if not _close(remaining["net_payoff_bps"].sum(), row["remaining_total_net_payoff_bps"]):
            return (
                False,
                f"contributor deletion mismatch: {row['family']}:{row['dimension']}:{row['top_n']}",
            )
        checked += 1
    loo = pd.read_csv(primary / "leave_one_stock_out.csv")
    for raw in loo.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        selected = f10.loc[
            f10["family"].eq(str(row["family"]))
            & ~f10["symbol"].astype(str).eq(str(row["removed_symbol"]))
        ]
        if not _close(selected["net_payoff_bps"].sum(), row["remaining_total_net_payoff_bps"]):
            return False, f"leave-one-stock-out mismatch: {row['family']}:{row['removed_symbol']}"
        checked += 1
    return True, f"reconstructed {checked} breakdown, concentration, deletion, and LOO rows"


def _check_nulls(primary: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    payoff = pd.read_parquet(primary / "payoff_envelope_ledger.parquet")
    flipped = pd.read_parquet(primary / "direction_flipped_null_ledger.parquet")
    f0 = payoff.loc[payoff["fill_model"].eq("F0")]
    f0_by_opportunity = {
        str(raw["opportunity_id"]): cast(dict[str, Any], raw)
        for raw in f0.to_dict(orient="records")
    }
    for raw in flipped.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        source = f0_by_opportunity[str(row["opportunity_id"])]
        flipped_expected = (
            10_000.0
            * -int(source["direction"])
            * (float(source["terminal_price"]) / float(source["reference_entry_price"]) - 1.0)
            - 10.0
        )
        if int(row["direction"]) != -int(source["direction"]) or not _close(
            row["net_payoff_bps"], flipped_expected
        ):
            return False, f"direction-flipped null mismatch: {row['opportunity_id']}"
    shifted = pd.read_parquet(primary / "timestamp_shift_null_ledger.parquet")
    providers: dict[str, pd.DataFrame] = {}
    reference = pd.concat(
        [
            pd.read_parquet(primary / "historical_named_reference_ledger.parquet"),
            pd.read_parquet(primary / "historical_control_reference_ledger.parquet"),
        ],
        ignore_index=True,
    )
    reference_by_opportunity = {
        str(raw["opportunity_id"]): cast(dict[str, Any], raw)
        for raw in reference.to_dict(orient="records")
    }
    available_shifted = shifted.loc[shifted["status"].eq("available")]
    for raw in available_shifted.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        source = reference_by_opportunity[str(row["opportunity_id"])]
        symbol = str(source["symbol"])
        providers.setdefault(symbol, _provider(contract, symbol))
        timestamp = pd.Timestamp(cast(Any, source["reference_entry_timestamp"])) + pd.Timedelta(
            minutes=5
        )
        entry = float(
            providers[symbol].loc[providers[symbol]["timestamp"].eq(timestamp), "open"].iloc[0]
        )
        shifted_expected = (
            10_000.0 * int(source["direction"]) * (float(source["terminal_price"]) / entry - 1.0)
            - 10.0
        )
        if not _close(row["shifted_net_payoff_bps"], shifted_expected):
            return False, f"timestamp-shift null mismatch: {row['opportunity_id']}"
    nulls = pd.read_csv(primary / "null_test_results.csv")
    if (
        nulls.loc[
            nulls["null_name"].eq("timestamp_shift_plus_one_bar"), "may_replace_primary_clock"
        ]
        .astype(bool)
        .any()
    ):
        return False, "timestamp-shift null was allowed to replace primary clock"
    for family, group in flipped.groupby("family", sort=True):
        observed = nulls.loc[
            nulls["null_name"].eq("direction_flipped_same_fill_and_terminal")
            & nulls["family"].eq(str(family))
        ]
        if len(observed) != 1 or not _close(
            observed.iloc[0]["observed_statistic_bps"], group["net_payoff_bps"].mean()
        ):
            return False, f"direction-flip summary mismatch: {family}"
    for family, group in shifted.loc[shifted["status"].eq("available")].groupby(
        "family", sort=True
    ):
        observed = nulls.loc[
            nulls["null_name"].eq("timestamp_shift_plus_one_bar") & nulls["family"].eq(str(family))
        ]
        if len(observed) != 1 or not _close(
            observed.iloc[0]["observed_statistic_bps"],
            group["shifted_minus_primary_bps"].mean(),
        ):
            return False, f"timestamp-shift summary mismatch: {family}"
    f10 = payoff.loc[payoff["fill_model"].eq("F10")].copy()
    rng = np.random.default_rng(20260717)
    values = f10["net_payoff_bps"].to_numpy(float)
    families = f10["family"].astype(str).to_numpy()
    for family in sorted(f10["family"].astype(str).unique()):
        mask = families == family
        observed_statistic = float(values[mask].mean())
        draws = np.asarray([float(rng.permutation(values)[mask].mean()) for _ in range(500)])
        permutation_row = cast(
            dict[str, Any],
            nulls.loc[
                nulls["null_name"].eq("random_opportunity_label_permutation_within_period")
                & nulls["family"].eq(family)
            ]
            .iloc[0]
            .to_dict(),
        )
        permutation_expected = (
            observed_statistic,
            float(draws.mean()),
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
            float((1 + np.sum(np.abs(draws) >= abs(observed_statistic))) / 501),
        )
        actual = (
            permutation_row["observed_statistic_bps"],
            permutation_row["null_mean_bps"],
            permutation_row["null_lower_95_bps"],
            permutation_row["null_upper_95_bps"],
            permutation_row["empirical_pvalue"],
        )
        if not all(
            _close(left, right) for left, right in zip(permutation_expected, actual, strict=True)
        ):
            return False, f"opportunity-label permutation mismatch: {family}"
    for loop_id, group in f10.groupby("loop_id", sort=True):
        named = group["classification"].eq("named").to_numpy()
        group_values = group["net_payoff_bps"].to_numpy(float)
        observed_statistic = float(group_values[named].mean() - group_values[~named].mean())
        draws = np.empty(500)
        for draw in range(500):
            permuted = rng.permutation(named)
            draws[draw] = float(group_values[permuted].mean() - group_values[~permuted].mean())
        label_row = cast(
            dict[str, Any],
            nulls.loc[
                nulls["null_name"].eq("random_named_control_label_permutation_within_parent_loop")
                & nulls["family"].eq(str(loop_id))
            ]
            .iloc[0]
            .to_dict(),
        )
        label_expected = (
            observed_statistic,
            float(draws.mean()),
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
            float((1 + np.sum(draws >= observed_statistic)) / 501),
        )
        actual = (
            label_row["observed_statistic_bps"],
            label_row["null_mean_bps"],
            label_row["null_lower_95_bps"],
            label_row["null_upper_95_bps"],
            label_row["empirical_pvalue"],
        )
        if not all(_close(left, right) for left, right in zip(label_expected, actual, strict=True)):
            return False, f"named-control label permutation mismatch: {loop_id}"
    return True, (
        f"reconstructed {len(flipped)} direction flips, {len(shifted)} shifted clocks, "
        "and all predeclared permutation nulls"
    )


def _record_hash(record: Mapping[str, object]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit_live_prospective_collection(
    contract: Mapping[str, Any], contract_hash: str
) -> tuple[bool, str]:
    root = WORK / "prospective/frozen-named-loop-t0-execution-realism-v1"
    if not root.exists():
        return True, "no prospective collection has begun"
    identity_path = root / "collection_identity.json"
    if not identity_path.is_file():
        return False, "prospective root exists without immutable collection identity"
    identity = cast(dict[str, object], json.loads(identity_path.read_text()))
    if _record_hash(identity) != str(identity.get("record_sha256")):
        return False, "prospective collection identity hash mismatch"
    if str(identity["contract_hash"]) != contract_hash or str(
        identity["completion_rule_hash"]
    ) != stable_hash(contract["prospective_completion_rule"]):
        return False, "prospective contract or completion-rule identity changed"
    opportunities: dict[str, dict[str, object]] = {}
    triggers: dict[str, dict[str, object]] = {}
    settlements: dict[str, dict[str, object]] = {}
    for stage, destination in (
        ("opportunities", opportunities),
        ("triggers", triggers),
        ("settlements", settlements),
    ):
        for path in sorted((root / stage).glob("*.json")):
            record = cast(dict[str, object], json.loads(path.read_text()))
            if _record_hash(record) != str(record.get("record_sha256")):
                return False, f"prospective record hash mismatch: {path}"
            opportunity_id = str(record["opportunity_id"])
            if opportunity_id in destination:
                return False, f"duplicate prospective identity: {stage}:{opportunity_id}"
            destination[opportunity_id] = record
    forbidden = ("payoff", "outcome", "hindsight", "episode", "terminal_price")
    for opportunity_id, record in opportunities.items():
        if any(token in str(key).lower() for key in record for token in forbidden):
            return False, f"outcome field entered opportunity record: {opportunity_id}"
        if record.get("research_only") is not True or record.get("execution_enabled") is not False:
            return False, f"prospective safety flag drift: {opportunity_id}"
    for opportunity_id, record in triggers.items():
        precursor = opportunities.get(opportunity_id)
        if precursor is None or str(record["opportunity_record_sha256"]) != str(
            precursor["record_sha256"]
        ):
            return False, f"trigger precursor chain mismatch: {opportunity_id}"
    for opportunity_id, record in settlements.items():
        opportunity = opportunities.get(opportunity_id)
        trigger = triggers.get(opportunity_id)
        if opportunity is None or trigger is None:
            return False, f"settlement precursor missing: {opportunity_id}"
        if str(record["opportunity_record_sha256"]) != str(opportunity["record_sha256"]) or str(
            record["trigger_record_sha256"]
        ) != str(trigger["record_sha256"]):
            return False, f"settlement precursor chain mismatch: {opportunity_id}"
        direction = int(cast(Any, opportunity["frozen_direction"]))
        entry = float(cast(Any, trigger["reference_entry_price"]))
        terminal = float(cast(Any, record["terminal_price"]))
        for model, stress in STRESSES.items():
            stressed = entry * (
                1.0 + stress / 10_000.0 if direction == 1 else 1.0 - stress / 10_000.0
            )
            gross = 10_000.0 * direction * (terminal / stressed - 1.0)
            if not _close(record[f"{model}_stressed_entry_price"], stressed) or not _close(
                record[f"{model}_net_payoff_bps"], gross - 10.0
            ):
                return (
                    False,
                    f"prospective settlement arithmetic mismatch: {opportunity_id}:{model}",
                )
    return True, (
        f"verified {len(opportunities)} opportunity, {len(triggers)} trigger, and "
        f"{len(settlements)} settlement records with immutable precursor hashes"
    )


def _check_prospective_and_2023(primary: Path, contract: Mapping[str, Any]) -> tuple[bool, str]:
    for filename in (
        "prospective_opportunity_ledger.parquet",
        "prospective_trigger_fill_append_ledger.parquet",
        "prospective_outcome_settlement_ledger.parquet",
    ):
        if len(pd.read_parquet(primary / filename)) != 0:
            return False, f"reference run contaminated prospective ledger: {filename}"
    status = json.loads((primary / "prospective_completion_status.json").read_text())
    flattened = json.dumps(status).lower()
    if status["completion_rule_reached"] or any(
        token in flattened for token in ("payoff", "profit", "f10_")
    ):
        return False, "incomplete prospective administration exposed economics"
    if status["minimums"] != contract["prospective_completion_rule"]:
        return False, "prospective completion rule identity differs"
    missing = json.loads((primary / "missing_2023_archival_report.json").read_text())
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_2023_hash_manifest"]["path"]).read_text()
    )["sha256"]
    if len([key for key in manifest if str(key).startswith("provider_2023_")]) != 20:
        return False, "registered 2023 manifest does not contain twenty symbols"
    if missing["all_registered_hashes_match"] or missing["economic_scoring_performed"]:
        return False, "missing 2023 input was scored or treated as restored"
    registered_root = Path(str(contract["inputs"]["expired_provider_2023_root"]))
    matched: list[str] = []
    mismatched: list[str] = []
    for key, digest in sorted(manifest.items()):
        if not str(key).startswith("provider_2023_"):
            continue
        symbol = str(key).removeprefix("provider_2023_")
        path = registered_root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        if path.is_file():
            (matched if sha256(path) == str(digest) else mismatched).append(symbol)
    if matched != missing["matched_symbols"] or mismatched != missing["mismatched_symbols"]:
        return False, "2023 registered-root hash report does not reconstruct"
    ledger_source = (
        REPO / "packages/stocker_research/src/stocker_research/"
        "frozen_named_loop_t0_execution/immutable_ledger.py"
    ).read_text()
    required_tokens = (
        "fcntl.flock",
        "os.O_EXCL",
        "record_sha256",
        "prospective sample is incomplete",
    )
    if not all(token in ledger_source for token in required_tokens):
        return False, "append-only or blinding implementation token missing"
    prospective_ok, detail = _audit_live_prospective_collection(contract, sha256(CONTRACT_PATH))
    if not prospective_ok:
        return False, detail
    return True, (
        "prospective templates are empty/blinded/create-only; 2023 remains hash-unavailable; "
        + detail
    )


def prohibited_changed_paths(paths: Iterable[str]) -> list[str]:
    allowed = ("research/slrno-v2/", "packages/stocker_research/", "tests/")
    return sorted(path for path in paths if not path.startswith(allowed))


def _check_safety(contract: Mapping[str, Any]) -> tuple[bool, str]:
    start = str(contract["frozen_lineage"]["inspected_head_before_editing"])
    changed = sorted(
        subprocess.check_output(
            ["git", "diff", "--name-only", start, FROZEN_HANDOFF_COMMIT],
            cwd=REPO,
            text=True,
        ).splitlines()
    )
    prohibited = prohibited_changed_paths(changed)
    if prohibited:
        return False, f"non-research/runtime path changed: {prohibited}"
    safety = contract["safety"]
    if not (
        safety["research_only"] is True
        and safety["execution_enabled"] is False
        and safety["broker_connection_enabled"] is False
        and safety["ig_order_placement_enabled"] is False
        and safety["paper_or_demo_ordering_enabled"] is False
        and safety["position_management_changed"] is False
        and safety["existing_strategy_exits_changed"] is False
        and safety["deployment_enabled"] is False
        and safety["application_runtime_changed"] is False
    ):
        return False, "contract safety boundary drift"
    return True, (
        f"all {len(changed)} frozen-handoff paths remain confined to research/package/tests"
    )


def verify_exact_identity(primary: Path, exact: Path) -> dict[str, object]:
    left = {
        path.relative_to(primary): path
        for path in primary.rglob("*")
        if path.is_file() and path.name not in EXCLUSIONS
    }
    right = {
        path.relative_to(exact): path
        for path in exact.rglob("*")
        if path.is_file() and path.name not in EXCLUSIONS
    }
    mismatches = [
        name.as_posix()
        for name in sorted(set(left) & set(right))
        if sha256(left[name]) != sha256(right[name])
    ]
    return {
        "byte_identical": set(left) == set(right) and not mismatches,
        "files_compared": len(left),
        "missing": sorted(name.as_posix() for name in set(left) - set(right)),
        "extra": sorted(name.as_posix() for name in set(right) - set(left)),
        "hash_mismatches": mismatches,
    }


def run_audit(primary: Path, exact: Path, output: Path) -> dict[str, object]:
    primary = Path(primary).resolve()
    exact = Path(exact).resolve()
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    metadata = cast(dict[str, Any], json.loads((primary / "run_metadata.json").read_text()))
    checks: dict[str, tuple[bool, str]] = {
        "contract_code_data_and_artifact_identity": _check_contract_and_artifact_identity(
            primary, contract, metadata
        ),
        "provider_hash_identity": _check_provider_hashes(contract, metadata),
        "frozen_pair_family_and_opportunity_identity": _check_population(primary, contract),
        "signal_trigger_fill_terminal_cost_and_payoff": _check_trigger_fill_terminal_and_payoffs(
            primary, contract
        ),
        "named_control_metrics_and_comparisons": _check_family_metrics(primary),
        "historical_reference_decay_and_data_quality": _check_reference_decay_and_quality(primary),
        "session_block_metrics_and_break_even": _check_inference_and_break_even(primary),
        "breakdowns_concentration_and_deletions": _check_breakdowns_and_concentration(primary),
        "direction_flip_and_timestamp_shift_nulls": _check_nulls(primary, contract),
        "prospective_append_only_blinding_and_2023_hashes": _check_prospective_and_2023(
            primary, contract
        ),
        "research_only_changed_paths": _check_safety(contract),
    }
    exact_result = verify_exact_identity(primary, exact)
    checks["historical_primary_exact_rerun_identity"] = (
        bool(exact_result["byte_identical"]),
        f"compared {exact_result['files_compared']} files including plots",
    )
    passed = all(result[0] for result in checks.values())
    result = {
        "audit_id": "frozen_named_loop_t0_execution_realism_v1_independent_audit",
        "auditor_version": "1.0.0",
        "passed": passed,
        "contract_hash": sha256(CONTRACT_PATH),
        "audited_run_id": metadata["run_id"],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "checks": {
            name: {"passed": check[0], "detail": check[1]} for name, check in checks.items()
        },
        "exact_identity": exact_result,
        "research_only": True,
        "execution_enabled": False,
        "broker_connection_enabled": False,
        "order_placement_enabled": False,
        "paper_or_demo_ordering_enabled": False,
        "position_management_enabled": False,
        "existing_exit_management_enabled": False,
        "deployment_enabled": False,
    }
    write_json(Path(output), result)
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


def main() -> int:
    args = parse_args()
    output = args.output or args.primary / "independent_audit.json"
    result = run_audit(args.primary, args.exact, output)
    print(json.dumps({"passed": result["passed"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
