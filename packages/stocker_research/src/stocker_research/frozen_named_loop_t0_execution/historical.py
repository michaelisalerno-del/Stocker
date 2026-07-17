"""Hash-pinned historical reconstruction for the frozen T0 experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .execution import (
    FillEvidence,
    TriggerType,
    gross_payoff_bps,
    reconstruct_frozen_oco_trigger,
    score_fill_envelope,
)
from .families import FROZEN_FAMILIES, family_spec


class HistoricalReproductionError(RuntimeError):
    """The frozen source, trigger, terminal, or payoff did not reconcile."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolved(contract_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (contract_path.parent / path).resolve()


def load_and_verify_contract(
    contract_path: Path,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Load the versioned contract and verify every file-backed input."""

    path = Path(contract_path).resolve()
    contract = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if not contract["registered_before_prospective_collection"]:
        raise IntegrityError("contract was not registered before prospective collection")
    if contract["adverse_entry_envelope"]["primary"] != "F10":
        raise IntegrityError("primary execution stress drift")
    if contract["populations"]["replacement_family_allowed"]:
        raise IntegrityError("replacement family enabled")
    safety = contract["safety"]
    if safety["research_only"] is not True or safety["execution_enabled"] is not False:
        raise IntegrityError("research-only safety boundary drift")
    hashes: dict[str, str] = {}
    for name, specification in contract["inputs"].items():
        if not isinstance(specification, Mapping) or "path" not in specification:
            continue
        source = _resolved(path, specification["path"])
        if not source.is_file():
            raise FileNotFoundError(f"frozen input is unavailable: {name}: {source}")
        actual = sha256_file(source)
        if actual != str(specification["sha256"]):
            raise IntegrityError(f"frozen input hash drift: {name}")
        hashes[str(name)] = actual
    mapping = cast(
        dict[str, Any],
        json.loads(_resolved(path, contract["inputs"]["family_mapping"]["path"]).read_text()),
    )
    mapped = {str(row["family"]): row for row in mapping["families"]}
    if set(mapped) != set(FROZEN_FAMILIES):
        raise IntegrityError("frozen mapping family set drift")
    for family, specification in FROZEN_FAMILIES.items():
        row = mapped[family]
        observed = (
            str(row["classification"]),
            str(row["loop_id"]),
            str(row["cycle"]),
            str(row["orientation"]),
            int(row["current_state"]),
            str(row["role"]),
        )
        expected = (
            specification.classification,
            specification.loop_id,
            specification.cycle,
            specification.orientation,
            specification.current_state,
            specification.role,
        )
        if observed != expected:
            raise IntegrityError(f"frozen mapping drift: {family}")
    return contract, sha256_file(path), hashes


class IntegrityError(RuntimeError):
    """A frozen contract or source identity changed."""


def _source_path(contract: Mapping[str, Any], contract_path: Path, classification: str) -> Path:
    key = (
        "named_source_reconciliation"
        if classification == "named"
        else "control_source_reconciliation"
    )
    return _resolved(contract_path, contract["inputs"][key]["path"])


def _build_population(
    contract: Mapping[str, Any], *, contract_path: Path, classification: str
) -> pd.DataFrame:
    if classification not in {"named", "control"}:
        raise ValueError(f"unsupported classification: {classification}")
    track = "track_a_named_family" if classification == "named" else "track_b_prior_only"
    reference = pd.read_parquet(_source_path(contract, contract_path, classification))
    policy = pd.read_parquet(
        _resolved(
            contract_path,
            contract["inputs"]["sequential_veto_population_identity"]["path"],
        )
    )
    policy = policy.loc[
        policy["track"].eq(track) & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
    ].copy()
    if classification == "control":
        policy = policy.loc[
            policy["population_role"].isin(["neutral_control", "negative_control"])
        ].copy()
    v2 = pd.read_parquet(_resolved(contract_path, contract["inputs"]["v2_trade_decisions"]["path"]))
    v2 = v2.loc[v2["model_name"].eq("no_payoff_state_filter")].copy()
    if policy["opportunity_id"].duplicated().any() or v2["opportunity_id"].duplicated().any():
        raise HistoricalReproductionError("ambiguous source opportunity identity")
    if set(policy["opportunity_id"].astype(str)) != set(reference["opportunity_id"].astype(str)):
        raise HistoricalReproductionError("reconciliation source opportunity set drift")
    trade_columns = [
        "opportunity_id",
        "anchor_id",
        "symbol_norm",
        "session_date",
        "period",
        "start_timestamp",
        "anchor_open",
        "anchor_high",
        "anchor_low",
        "anchor_close",
        "direction",
        "entry_step",
        "entry_timestamp",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "gross_payoff_bps",
        "primary_total_cost_bps",
        "primary_net_payoff_bps",
        "loop_id",
        "orientation",
        "state",
        "history_token",
        "dollar_volume_proxy",
        "liquidity_proxy_status",
        "sector",
        "month",
        "quarter",
        "run_id",
        "configuration_hash",
        "status",
        "horizon",
        "strategy",
        "decision_timestamp",
        "feature_availability_timestamp",
    ]
    source = policy.loc[
        :,
        [
            "experiment_run_id",
            "opportunity_id",
            "event_lineage_id",
            "period",
            "session_date",
            "stock",
            "target_loop",
            "orientation",
            "population_role",
        ],
    ].merge(
        v2.loc[:, trade_columns],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
        suffixes=("_policy", "_v2"),
    )
    if source["anchor_id"].isna().any():
        raise HistoricalReproductionError("policy opportunity lacks exact V2 row")
    checks = {
        "symbol": source["stock"].astype(str).eq(source["symbol_norm"].astype(str)),
        "period": source["period_policy"].astype(str).eq(source["period_v2"].astype(str)),
        "loop": source["target_loop"].astype(str).eq(source["loop_id"].astype(str)),
        "orientation": source["orientation_policy"]
        .astype(str)
        .eq(source["orientation_v2"].astype(str)),
    }
    if failed := [name for name, values in checks.items() if not bool(values.all())]:
        raise HistoricalReproductionError(f"source identity mismatch: {failed}")
    source = source.rename(
        columns={
            "symbol_norm": "symbol",
            "period_policy": "period",
            "session_date_policy": "session",
            "target_loop": "frozen_loop_id",
            "orientation_policy": "orientation",
            "start_timestamp": "anchor_timestamp",
            "entry_timestamp": "original_entry_timestamp",
            "entry_price": "original_entry_price",
            "exit_timestamp": "original_terminal_timestamp",
            "exit_price": "original_terminal_price",
            "gross_payoff_bps": "original_gross_payoff_bps",
            "primary_total_cost_bps": "original_total_cost_bps",
            "primary_net_payoff_bps": "original_net_payoff_bps",
            "run_id": "source_run_id",
        }
    )
    source["loop_id"] = source.pop("frozen_loop_id")
    source["classification"] = classification
    source["family"] = source["loop_id"].astype(str) + "|" + source["orientation"].astype(str)
    source["role"] = source["family"].map(lambda value: family_spec(str(value)).role)
    source["cycle"] = source["family"].map(lambda value: family_spec(str(value)).cycle)
    source["current_state"] = source["family"].map(
        lambda value: family_spec(str(value)).current_state
    )
    expected_classification = source["family"].map(
        lambda value: family_spec(str(value)).classification
    )
    if not expected_classification.eq(classification).all():
        raise HistoricalReproductionError("named and control population crossed")
    if not source["state"].astype(int).eq(source["current_state"].astype(int)).all():
        raise HistoricalReproductionError("orientation state differs from frozen current state")
    if not source["status"].eq("filled").all() or not source["horizon"].eq(24).all():
        raise HistoricalReproductionError("source fill status or terminal horizon drift")
    if not source["strategy"].eq("breakout_loop_scores_range_p75").all():
        raise HistoricalReproductionError("source strategy drift")
    for column in (
        "anchor_timestamp",
        "original_entry_timestamp",
        "original_terminal_timestamp",
        "decision_timestamp",
        "feature_availability_timestamp",
    ):
        source[column] = pd.to_datetime(source[column], utc=True, errors="raise")
    if not source["direction"].isin([-1, 1]).all():
        raise HistoricalReproductionError("ambiguous source direction survived")
    source_indexed = source.set_index("opportunity_id")
    reference_indexed = reference.set_index("opportunity_id").loc[source_indexed.index]
    reconciliations = {
        "anchor_id": source_indexed["anchor_id"]
        .astype(str)
        .eq(reference_indexed["anchor_id"].astype(str)),
        "direction": source_indexed["direction"]
        .astype(int)
        .eq(reference_indexed["direction"].astype(int)),
        "entry_timestamp": source_indexed["original_entry_timestamp"].eq(
            pd.to_datetime(reference_indexed["original_entry_timestamp"], utc=True)
        ),
        "terminal_timestamp": source_indexed["original_terminal_timestamp"].eq(
            pd.to_datetime(reference_indexed["original_terminal_timestamp"], utc=True)
        ),
        "entry_price": np.isclose(
            source_indexed["original_entry_price"].to_numpy(float),
            reference_indexed["original_entry_price"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ),
        "terminal_price": np.isclose(
            source_indexed["original_terminal_price"].to_numpy(float),
            reference_indexed["original_exit_price"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ),
    }
    if failed := [
        name for name, values in reconciliations.items() if not bool(np.asarray(values).all())
    ]:
        raise HistoricalReproductionError(f"source field reconciliation failed: {failed}")
    source["source_artifact_hash"] = stable_hash(
        {
            "v2": contract["inputs"]["v2_trade_decisions"]["sha256"],
            "policy": contract["inputs"]["sequential_veto_population_identity"]["sha256"],
        }
    )
    source["source_opportunity_hash"] = [
        stable_hash(
            {
                "opportunity_id": row.opportunity_id,
                "anchor_id": row.anchor_id,
                "direction": row.direction,
                "entry_timestamp": row.original_entry_timestamp,
                "entry_price": row.original_entry_price,
                "terminal_timestamp": row.original_terminal_timestamp,
                "terminal_price": row.original_terminal_price,
            }
        )
        for row in source.itertuples(index=False)
    ]
    source["period"] = source["period"].astype(int)
    source["month"] = source["session"].astype(str).str[:7]
    source["quarter_label"] = pd.to_datetime(source["session"]).dt.to_period("Q").astype(str)
    source["anchor_regime"] = source["orientation"]
    source = source.drop(
        columns=[
            "period_v2",
            "session_date_v2",
            "orientation_v2",
            "stock",
            "experiment_run_id",
        ]
    )
    return source


def build_source_populations(contract: Mapping[str, Any], *, contract_path: Path) -> pd.DataFrame:
    """Build named and control populations without any replacement or filter."""

    source = pd.concat(
        [
            _build_population(contract, contract_path=contract_path, classification="named"),
            _build_population(contract, contract_path=contract_path, classification="control"),
        ],
        ignore_index=True,
    )
    expected = contract["populations"]["expected_historical_source_counts"]
    counts = source.groupby(["period", "family"]).size().to_dict()
    expected_counts = {
        (int(period), family): int(count)
        for period, families in expected.items()
        for family, count in families.items()
    }
    if counts != expected_counts:
        raise HistoricalReproductionError(
            f"frozen source counts differ: observed={counts}, expected={expected_counts}"
        )
    return source.sort_values(
        ["period", "session", "symbol", "anchor_timestamp", "opportunity_id"], kind="stable"
    ).reset_index(drop=True)


def _provider_manifest(
    contract: Mapping[str, Any], contract_path: Path, period: int
) -> dict[str, str]:
    key = f"provider_{period}_hash_manifest"
    document = json.loads(_resolved(contract_path, contract["inputs"][key]["path"]).read_text())
    prefix = f"provider_{period}_"
    return {
        name.removeprefix(prefix): str(digest)
        for name, digest in document["sha256"].items()
        if name.startswith(prefix)
    }


def load_provider_frames(
    contract: Mapping[str, Any],
    symbols: Iterable[object],
    *,
    contract_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load and hash-verify the exact 2025 five-minute provider files."""

    manifest = _provider_manifest(contract, contract_path, 2025)
    root = Path(str(contract["inputs"]["provider_2025_root"]))
    frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted({str(value) for value in symbols}):
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        if symbol not in manifest or not path.is_file():
            raise HistoricalReproductionError(f"missing registered provider file: {symbol}")
        if sha256_file(path) != manifest[symbol]:
            raise HistoricalReproductionError(f"provider hash drift: {symbol}")
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
            raise HistoricalReproductionError(f"invalid provider timestamps: {symbol}")
        if not frame["timestamp"].is_monotonic_increasing:
            frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        frames[symbol] = frame
    return frames


def reconstruct_historical_2025(
    source: pd.DataFrame,
    providers: Mapping[str, pd.DataFrame],
    *,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Reconstruct every source trigger, reference fill, and frozen terminal."""

    if not source["period"].eq(2025).all():
        raise ValueError("historical execution reconstruction only accepts 2025 rows")
    rows: list[dict[str, object]] = []
    for raw in source.to_dict(orient="records"):
        values = cast(dict[str, Any], raw)
        anchor = pd.Timestamp(cast(Any, values["anchor_timestamp"]))
        provider = providers[str(values["symbol"])]
        window = provider.loc[
            provider["timestamp"].between(
                anchor + pd.Timedelta(minutes=5),
                anchor + pd.Timedelta(minutes=120),
                inclusive="both",
            )
        ].copy()
        reconstruction = reconstruct_frozen_oco_trigger(
            window,
            anchor_timestamp=anchor,
            long_threshold=float(values["anchor_high"]),
            short_threshold=float(values["anchor_low"]),
            horizon_bars=24,
        )
        if reconstruction.status != "triggered":
            raise HistoricalReproductionError(
                "source fill does not reconstruct: "
                f"{values['opportunity_id']}: {reconstruction.status}"
            )
        reference_exact = bool(
            reconstruction.direction == int(values["direction"])
            and reconstruction.entry_step == int(values["entry_step"])
            and reconstruction.reference_entry_timestamp
            == pd.Timestamp(cast(Any, values["original_entry_timestamp"]))
            and np.isclose(
                float(cast(float, reconstruction.reference_entry_price)),
                float(values["original_entry_price"]),
                rtol=0.0,
                atol=tolerance,
            )
        )
        terminal_timestamp = anchor + pd.Timedelta(minutes=125)
        terminal_rows = provider.loc[
            provider["timestamp"].eq(terminal_timestamp - pd.Timedelta(minutes=5))
        ]
        terminal_exact = bool(
            len(terminal_rows) == 1
            and terminal_timestamp == pd.Timestamp(cast(Any, values["original_terminal_timestamp"]))
            and np.isclose(
                float(terminal_rows.iloc[0]["close"]),
                float(values["original_terminal_price"]),
                rtol=0.0,
                atol=tolerance,
            )
        )
        source_gross = gross_payoff_bps(
            int(values["direction"]),
            float(values["original_entry_price"]),
            float(values["original_terminal_price"]),
        )
        payoff_exact = bool(
            np.isclose(
                source_gross,
                float(values["original_gross_payoff_bps"]),
                rtol=0.0,
                atol=tolerance,
            )
            and np.isclose(
                source_gross - float(values["original_total_cost_bps"]),
                float(values["original_net_payoff_bps"]),
                rtol=0.0,
                atol=tolerance,
            )
            and np.isclose(float(values["original_total_cost_bps"]), 10.0, rtol=0.0, atol=tolerance)
        )
        if not (reference_exact and terminal_exact and payoff_exact):
            raise HistoricalReproductionError(
                f"source trigger/terminal/payoff discrepancy: {values['opportunity_id']}"
            )
        evidence = reconstruction.fill_evidence
        rows.append(
            {
                **values,
                "long_threshold": float(values["anchor_high"]),
                "short_threshold": float(values["anchor_low"]),
                "threshold_known_timestamp": reconstruction.threshold_known_timestamp,
                "signal_known_timestamp": reconstruction.signal_known_timestamp,
                "trigger_timestamp": reconstruction.trigger_bar_timestamp,
                "trigger_bar": reconstruction.entry_step,
                "trigger_type": reconstruction.trigger_type.value,
                "trigger_price": reconstruction.reference_entry_price,
                "trigger_bar_open": reconstruction.trigger_bar_open,
                "trigger_bar_high": reconstruction.trigger_bar_high,
                "trigger_bar_low": reconstruction.trigger_bar_low,
                "trigger_bar_close": reconstruction.trigger_bar_close,
                "reference_entry_timestamp": reconstruction.reference_entry_timestamp,
                "reference_entry_price": reconstruction.reference_entry_price,
                "reference_fill_convention": "frozen_open_or_threshold",
                "opening_gap_fill": reconstruction.trigger_type
                is TriggerType.OPENING_GAP_THROUGH_THRESHOLD,
                "threshold_fill": reconstruction.trigger_type
                is TriggerType.INTRABAR_THRESHOLD_CROSS,
                "fill_evidence_classification": evidence.value,
                "primary_valid_fill_evidence": evidence
                in {FillEvidence.EXACTLY_OBSERVABLE, FillEvidence.GAP_FILL_OBSERVABLE},
                "signal_fill_time_status": reconstruction.signal_fill_time_status,
                "market_data_availability_timestamp": (
                    reconstruction.market_data_availability_timestamp
                ),
                "exact_or_bounded_evidence": (
                    "observed_trigger_bar_open"
                    if evidence is FillEvidence.GAP_FILL_OBSERVABLE
                    else "five_minute_high_low_bounds_only"
                ),
                "terminal_price": float(values["original_terminal_price"]),
                "reference_reconstruction_exact": reference_exact,
                "terminal_reconstruction_exact": terminal_exact,
                "payoff_reconciliation_exact": payoff_exact,
                "reference_reconstruction_status": "exact_match",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["session", "symbol", "anchor_timestamp", "opportunity_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_payoff_ledger(reconstructed: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed non-cumulative execution envelope to every source row."""

    rows: list[dict[str, object]] = []
    identity_columns = [
        "opportunity_id",
        "anchor_id",
        "event_lineage_id",
        "source_run_id",
        "source_artifact_hash",
        "source_opportunity_hash",
        "symbol",
        "period",
        "session",
        "loop_id",
        "orientation",
        "family",
        "classification",
        "role",
        "cycle",
        "current_state",
        "direction",
        "anchor_timestamp",
        "anchor_close",
        "entry_step",
        "trigger_timestamp",
        "trigger_type",
        "fill_evidence_classification",
        "primary_valid_fill_evidence",
        "reference_entry_timestamp",
        "reference_entry_price",
        "original_terminal_timestamp",
        "terminal_price",
        "month",
        "quarter_label",
        "anchor_regime",
        "hindsight_episode_id",
    ]
    for raw in reconstructed.to_dict(orient="records"):
        values = cast(dict[str, Any], raw)
        envelope = score_fill_envelope(
            opportunity_id=str(values["opportunity_id"]),
            direction=int(values["direction"]),
            reference_entry_price=float(values["reference_entry_price"]),
            terminal_timestamp=pd.Timestamp(cast(Any, values["original_terminal_timestamp"])),
            terminal_price=float(values["terminal_price"]),
            cost_bps=float(values["original_total_cost_bps"]),
        )
        identity = {column: values.get(column) for column in identity_columns}
        for payoff in envelope:
            rows.append(
                {
                    **identity,
                    "fill_model": payoff.fill_model,
                    "adverse_entry_slippage_bps": payoff.adverse_entry_slippage_bps,
                    "stressed_entry_price": payoff.stressed_entry_price,
                    "gross_payoff_bps": payoff.gross_payoff_bps,
                    "cost_bps": payoff.cost_bps,
                    "net_payoff_bps": payoff.net_payoff_bps,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["session", "symbol", "anchor_timestamp", "opportunity_id", "fill_model"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def attach_hindsight_episodes(
    frame: pd.DataFrame, contract: Mapping[str, Any], *, contract_path: Path
) -> pd.DataFrame:
    """Attach opened-data episodes strictly after scoring for attribution."""

    result = frame.copy()
    result["hindsight_episode_id"] = "outside_hindsight_episode"
    episodes = pd.read_parquet(
        _resolved(contract_path, contract["inputs"]["hindsight_episode_diagnostics"]["path"])
    )
    episodes = episodes.loc[episodes["horizon"].eq(24)].copy()
    episodes["onset"] = pd.to_datetime(episodes["hindsight_estimated_onset"]).dt.date
    episodes["end"] = pd.to_datetime(episodes["hindsight_estimated_end"]).dt.date
    sessions = pd.to_datetime(result["session"]).dt.date
    for raw in episodes.to_dict(orient="records"):
        values = cast(dict[str, Any], raw)
        mask = (
            result["period"].eq(int(values["period"]))
            & result["loop_id"].eq(str(values["loop_id"]))
            & result["orientation"].eq(str(values["orientation"]))
            & sessions.between(values["onset"], values["end"])
        )
        result.loc[mask, "hindsight_episode_id"] = str(values["episode_id"])
    return result


def verify_2023_archive(root: Path, manifest: Mapping[str, Any]) -> dict[str, object]:
    """Accept 2023 only if all twenty original files match their registered hashes."""

    expected = {
        key.removeprefix("provider_2023_"): str(value)
        for key, value in cast(Mapping[str, object], manifest["sha256"]).items()
        if str(key).startswith("provider_2023_")
    }
    matched: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    base = Path(root)
    for symbol, digest in sorted(expected.items()):
        path = base / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        if not path.is_file():
            missing.append(symbol)
        elif sha256_file(path) == digest:
            matched.append(symbol)
        else:
            mismatched.append(symbol)
    complete = len(matched) == len(expected) == 20
    return {
        "status": "available_all_registered_hashes_match"
        if complete
        else "unavailable_hash_mismatch_or_missing",
        "required_symbols": len(expected),
        "matched_symbols": matched,
        "missing_symbols": missing,
        "mismatched_symbols": mismatched,
        "all_registered_hashes_match": complete,
        "fresh_download_allowed": False,
        "imputation_allowed": False,
    }
