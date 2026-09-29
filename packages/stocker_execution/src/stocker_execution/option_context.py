"""Receipt-timed provider context; chain prices never become executable quotes.

ETO documentation does not specify all analytics scales. Preserve signed raw
values without inventing percentage, per-day theta or probability conversions.
"""

import math
from typing import Any

GREEKS = (
    "Delta",
    "Gamma",
    "Theta",
    "Vega",
    "Rho",
    "AskVolatility",
    "BidVolatility",
    "MidVolatility",
    "TheoreticalPrice",
    "InTheMoneyProbability",
)
CONTEXT_FIELDS = {
    "Greeks": (*GREEKS, "MidVol"),
    "PriceInfoDetails": ("LastTraded", "LastTradedSize", "Volume"),
    "InstrumentPriceDetails": ("OpenInterest",),
}


def fields(update: dict[str, Any], at: float, source: str) -> dict[str, Any]:
    result = {}
    for group, names in CONTEXT_FIELDS.items():
        if group not in update:
            continue
        values = update[group]
        for name in names:
            if values is not None and (not isinstance(values, dict) or name not in values):
                continue
            raw = values.get(name) if isinstance(values, dict) else None
            valid = (
                isinstance(raw, (float, int)) and not isinstance(raw, bool) and math.isfinite(raw)
            )
            if (
                name in ("OpenInterest", "Volume", "LastTradedSize")
                and isinstance(raw, (int, float))
                and raw < 0
            ):
                valid = False
            result[group + "." + name] = {
                "raw": raw,
                "value": raw if valid else None,
                "received_at": at,
                "effective_at": None,
                "effective_status": "NOT_PROVIDED",
                "provider_timestamp": update.get("LastUpdated"),
                "source": source,
                "availability": "AVAILABLE" if valid else "MISSING" if raw is None else "INVALID",
                "unit": "contracts"
                if name == "OpenInterest"
                else "annual fraction"
                if name == "MidVol"
                else "PROVIDER_NATIVE",
                "normalised": raw if valid and name in ("OpenInterest", "MidVol") else None,
                # Price-subscription MidVol scale verified against live SIM on 2026-09-29.
                "scaling": "DOCUMENTED"
                if name == "OpenInterest"
                else "VERIFIED_LIVE"
                if name == "MidVol"
                else "UNVERIFIED",
            }
    return result


def view(observations: dict[str, Any], at: float) -> dict[str, Any]:
    result = {}
    for name, row in observations.items():
        age = at - row["received_at"]
        status = row["availability"]
        if age < 0:
            continue  # Current cache is not an earlier decision snapshot.
        if status == "AVAILABLE":
            status = (
                ("STALE" if age > 86400 else "AS_OF_EFFECTIVE_TIME_UNKNOWN")
                if name.endswith("OpenInterest")
                else "STALE"
                if age > 60
                else "OBSERVED_UNVERIFIED"
            )
        result[name] = {**row, "age_seconds": age, "status": status}
    return result


def chain_update(side: dict[str, Any], provider_time: Any = None) -> dict[str, Any]:
    # DeltaPct and ContractId belong to FX options, not FuturesOption.
    result: dict[str, Any] = {"LastUpdated": provider_time}
    if "Greeks" in side:
        result["Greeks"] = side["Greeks"]
    for group, names in CONTEXT_FIELDS.items():
        if group == "Greeks":
            continue
        values = {name: side[name] for name in names if name in side}
        if values:
            result[group] = values
    return result
