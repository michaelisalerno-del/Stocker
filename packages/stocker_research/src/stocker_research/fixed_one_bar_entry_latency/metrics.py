"""Paired-population accounting and uncertainty for fixed entry latency."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

IDENTITY_COLUMNS = (
    "opportunity_id",
    "anchor_id",
    "event_lineage_id",
    "period",
    "session_date",
    "symbol",
    "loop_id",
    "orientation",
    "direction",
    "original_terminal_timestamp",
)


def build_exact_paired_population(
    source: pd.DataFrame,
    latency: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pair exact T0/T1 rows without replacement or silent missing-row loss."""

    required_source = {
        *IDENTITY_COLUMNS,
        "original_entry_timestamp",
        "original_gross_payoff_bps",
        "original_total_cost_bps",
        "original_net_payoff_bps",
    }
    required_latency = {
        *IDENTITY_COLUMNS,
        "t1_status",
        "t1_entry_timestamp",
        "t1_gross_return_bps",
        "t1_total_cost_bps",
        "t1_net_return_bps",
        "paired_difference_bps",
    }
    missing_source = sorted(required_source - set(source))
    missing_latency = sorted(required_latency - set(latency))
    if missing_source or missing_latency:
        raise ValueError(
            f"missing pairing fields: source={missing_source}, latency={missing_latency}"
        )
    if source["opportunity_id"].duplicated().any() or latency["opportunity_id"].duplicated().any():
        raise ValueError("opportunity identities must be unique")
    if set(source["opportunity_id"].astype(str)) != set(latency["opportunity_id"].astype(str)):
        raise ValueError("source and latency populations differ")
    source_by_id = source.set_index("opportunity_id", drop=False)
    latency_by_id = latency.set_index("opportunity_id", drop=False)
    aligned = latency_by_id.loc[source_by_id.index]
    for column in IDENTITY_COLUMNS[1:]:
        left = source_by_id[column]
        right = aligned[column]
        if column.endswith("timestamp"):
            equal = pd.to_datetime(left, utc=True).eq(pd.to_datetime(right, utc=True))
        else:
            equal = left.astype(str).eq(right.astype(str))
        if not bool(equal.all()):
            raise ValueError(f"identity mismatch: {column}")
    latency_values = latency.drop(columns=list(IDENTITY_COLUMNS[1:]))
    all_rows = source.merge(
        latency_values,
        on="opportunity_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    all_rows["t0_available"] = (
        pd.to_datetime(all_rows["original_entry_timestamp"], utc=True, errors="coerce").notna()
        & pd.to_numeric(all_rows["original_net_payoff_bps"], errors="coerce").notna()
    )
    all_rows["t1_available"] = (
        all_rows["t1_status"].eq("available")
        & pd.to_datetime(all_rows["t1_entry_timestamp"], utc=True, errors="coerce").notna()
        & pd.to_numeric(all_rows["t1_net_return_bps"], errors="coerce").notna()
    )
    all_rows["paired_available"] = all_rows["t0_available"] & all_rows["t1_available"]
    all_rows["pairing_status"] = np.where(
        all_rows["paired_available"], "paired_available", all_rows["t1_status"]
    )
    all_rows["replacement_opportunity_id"] = None
    all_rows["overlap_or_capacity_refilled"] = False
    all_rows["existing_position_action"] = "unchanged"
    all_rows["t0_net_return_bps"] = pd.to_numeric(
        all_rows["original_net_payoff_bps"], errors="coerce"
    )
    all_rows["t0_gross_return_bps"] = pd.to_numeric(
        all_rows["original_gross_payoff_bps"], errors="coerce"
    )
    all_rows["t0_total_cost_bps"] = pd.to_numeric(
        all_rows["original_total_cost_bps"], errors="coerce"
    )
    paired = all_rows.loc[all_rows["paired_available"]].copy()
    unavailable = all_rows.loc[~all_rows["paired_available"]].copy()
    return (
        all_rows.reset_index(drop=True),
        paired.reset_index(drop=True),
        unavailable.reset_index(drop=True),
    )


def _profit_factor(values: pd.Series) -> float:
    net = pd.to_numeric(values, errors="coerce").dropna()
    loss = float(-net.loc[net.lt(0.0)].sum())
    return float(net.loc[net.gt(0.0)].sum() / loss) if loss > 0.0 else float("nan")


def _drawdown(values: pd.Series) -> float:
    cumulative = pd.to_numeric(values, errors="coerce").fillna(0.0).cumsum()
    return float((cumulative - cumulative.cummax()).min()) if len(cumulative) else 0.0


def paired_summary(paired: pd.DataFrame) -> dict[str, float | int]:
    """Summarise exact paired T0/T1 levels and the incremental endpoint."""

    required = {
        "t0_net_return_bps",
        "t1_net_return_bps",
        "paired_difference_bps",
    }
    missing = sorted(required - set(paired))
    if missing:
        raise ValueError(f"missing paired summary fields: {missing}")
    t0 = pd.to_numeric(paired["t0_net_return_bps"], errors="coerce")
    t1 = pd.to_numeric(paired["t1_net_return_bps"], errors="coerce")
    delta = pd.to_numeric(paired["paired_difference_bps"], errors="coerce")
    observable = t0.notna() & t1.notna() & delta.notna()
    t0 = t0.loc[observable]
    t1 = t1.loc[observable]
    delta = delta.loc[observable]
    nonzero = delta.loc[~np.isclose(delta, 0.0, rtol=0.0, atol=1e-12)]
    improved = int(nonzero.gt(0.0).sum())
    sign_pvalue = (
        float(binomtest(improved, len(nonzero), p=0.5, alternative="two-sided").pvalue)
        if len(nonzero)
        else float("nan")
    )
    return {
        "paired_opportunities": int(len(delta)),
        "t0_net_payoff_bps": float(t0.sum()),
        "t1_net_payoff_bps": float(t1.sum()),
        "t0_mean_net_payoff_bps": float(t0.mean()),
        "t1_mean_net_payoff_bps": float(t1.mean()),
        "t0_median_net_payoff_bps": float(t0.median()),
        "t1_median_net_payoff_bps": float(t1.median()),
        "t0_positive_rate": float(t0.gt(0.0).mean()),
        "t1_positive_rate": float(t1.gt(0.0).mean()),
        "t0_profit_factor": _profit_factor(t0),
        "t1_profit_factor": _profit_factor(t1),
        "t0_maximum_drawdown_bps": _drawdown(t0),
        "t1_maximum_drawdown_bps": _drawdown(t1),
        "paired_total_difference_bps": float(delta.sum()),
        "paired_mean_difference_bps": float(delta.mean()),
        "paired_median_difference_bps": float(delta.median()),
        "opportunities_improved_fraction": float(delta.gt(0.0).mean()),
        "opportunities_worsened_fraction": float(delta.lt(0.0).mean()),
        "opportunities_unchanged_fraction": float(np.isclose(delta, 0.0).mean()),
        "paired_sign_test_pvalue": sign_pvalue,
    }


def session_block_bootstrap(
    paired: pd.DataFrame,
    *,
    resamples: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    """Moving five-session block interval over session-mean paired deltas."""

    if resamples <= 0 or block_length <= 0:
        raise ValueError("resamples and block_length must be positive")
    required = {"period", "session_date", "paired_difference_bps"}
    missing = sorted(required - set(paired))
    if missing:
        raise ValueError(f"missing bootstrap fields: {missing}")
    session = (
        paired.groupby(["period", "session_date"], sort=True)["paired_difference_bps"]
        .mean()
        .reset_index()
    )
    observed = float(session["paired_difference_bps"].mean())
    arrays = [
        group["paired_difference_bps"].to_numpy(float)
        for _, group in session.groupby("period", sort=True)
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        rebuilt: list[float] = []
        for values in arrays:
            count = len(values)
            if count == 0:
                continue
            starts = rng.integers(0, count, size=int(np.ceil(count / block_length)))
            sampled = [
                float(values[(int(start) + offset) % count])
                for start in starts
                for offset in range(block_length)
            ]
            rebuilt.extend(sampled[:count])
        draws[draw] = float(np.mean(rebuilt)) if rebuilt else np.nan
    return {
        "observed_session_mean_delta_bps": observed,
        "bootstrap_lower_95_bps": float(np.nanquantile(draws, 0.025)),
        "bootstrap_upper_95_bps": float(np.nanquantile(draws, 0.975)),
    }


def paired_breakdowns(
    paired: pd.DataFrame,
    *,
    dimensions: Sequence[str],
) -> pd.DataFrame:
    """Return paired summaries for frozen descriptive dimensions."""

    records: list[dict[str, object]] = [
        {"slice_type": "all", "slice_value": "all", **paired_summary(paired)}
    ]
    for dimension in dimensions:
        if dimension not in paired:
            continue
        for value, group in paired.groupby(dimension, dropna=False, sort=True):
            records.append(
                {
                    "slice_type": dimension,
                    "slice_value": str(value),
                    **paired_summary(group),
                }
            )
    return pd.DataFrame(records)
