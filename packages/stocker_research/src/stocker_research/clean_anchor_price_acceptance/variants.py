"""Frozen A--E admission variants on one immutable source population."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

VARIANT_RULES: dict[str, tuple[str, ...]] = {
    "A_same_clock_base": (),
    "B_anchor_veto_only": ("static_anchor_veto_pass",),
    "C_price_acceptance_only": ("price_acceptance_pass",),
    "D_anchor_veto_plus_price_acceptance": (
        "static_anchor_veto_pass",
        "price_acceptance_pass",
    ),
    "E_anchor_veto_plus_price_acceptance_plus_range": (
        "static_anchor_veto_pass",
        "price_acceptance_pass",
        "range_permission_pass",
    ),
}


def build_variant_decisions(source: pd.DataFrame) -> pd.DataFrame:
    """Cross the same immutable source rows with exactly the registered variants."""

    required = {
        "opportunity_id",
        "source_available",
        "static_anchor_veto_pass",
        "price_acceptance_pass",
        "range_permission_available",
        "range_permission_pass",
        "entry_timestamp",
        "original_terminal_timestamp",
        "net_payoff_bps",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"missing variant source columns: {missing}")
    if source["opportunity_id"].duplicated().any():
        raise ValueError("duplicate source opportunity identity")
    records: list[dict[str, object]] = []
    for raw_row in source.to_dict("records"):
        row = {str(key): value for key, value in raw_row.items()}
        available = bool(row["source_available"])
        for variant, gates in VARIANT_RULES.items():
            if not available:
                decision = "unavailable"
                reasons = [str(row.get("availability_status", "source_unavailable"))]
                admitted = False
            elif "range_permission_pass" in gates and not bool(row["range_permission_available"]):
                decision = "unavailable"
                reasons = ["range_permission_unavailable"]
                admitted = False
            else:
                failed = [gate for gate in gates if not bool(row[gate])]
                admitted = not failed
                decision = "admitted" if admitted else "rejected"
                reasons = [] if admitted else [f"{gate}_failed" for gate in failed]
            record = dict(row)
            record.update(
                {
                    "variant": variant,
                    "decision": decision,
                    "admitted": admitted,
                    "reason_codes": "|".join(reasons) if reasons else "admission_conditions_met",
                    "replacement_opportunity_id": None,
                    "overlap_or_capacity_refilled": False,
                    "existing_position_action": "unchanged",
                    "policy_net_payoff_bps": (float(row["net_payoff_bps"]) if admitted else 0.0)
                    if available and decision != "unavailable"
                    else None,
                }
            )
            records.append(record)
    result = pd.DataFrame(records)
    expected = len(source) * len(VARIANT_RULES)
    if len(result) != expected:
        raise AssertionError("variant population expansion changed source identity")
    return result


def variant_population_identity(decisions: pd.DataFrame) -> Mapping[str, tuple[str, ...]]:
    """Expose sorted immutable identities for independent pairing checks."""

    return {
        str(variant): tuple(sorted(group["opportunity_id"].astype(str)))
        for variant, group in decisions.groupby("variant", sort=True)
    }
