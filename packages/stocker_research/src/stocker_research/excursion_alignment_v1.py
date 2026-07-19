"""Time-window and family alignment for cluster-invariant excursion events V1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_REQUIRED = {
    "event_id",
    "symbol",
    "session",
    "segment_id",
    "event_family",
    "onset_bar_ordinal",
    "resolution_bar_ordinal",
}


def _validated(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"{role} event ledger lacks columns: {missing}")
    result = frame.copy().reset_index(drop=True)
    if result["event_id"].duplicated().any():
        raise ValueError(f"{role} event IDs must be unique")
    for column in ("onset_bar_ordinal", "resolution_bar_ordinal"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    if result["resolution_bar_ordinal"].lt(result["onset_bar_ordinal"]).any():
        raise ValueError(f"{role} event resolves before onset")
    return result


def _overlap(left: pd.Series, right: pd.Series, tolerance: int) -> bool:
    if str(left["symbol"]) != str(right["symbol"]) or str(left["session"]) != str(right["session"]):
        return False
    left_start = int(left["onset_bar_ordinal"])
    left_end = int(left["resolution_bar_ordinal"])
    right_start = int(right["onset_bar_ordinal"])
    right_end = int(right["resolution_bar_ordinal"])
    window_overlap = left_start <= right_end and right_start <= left_end
    bounded_anchor = abs(left_start - right_start) <= tolerance
    bounded_resolution = abs(left_end - right_end) <= tolerance
    return bool(window_overlap or bounded_anchor or bounded_resolution)


def _row(
    reference: pd.Series,
    candidate: pd.Series | None,
    *,
    alignment_class: str,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    if candidate is None:
        onset_shift: float = math.nan
        resolution_shift: float = math.nan
        candidate_id = ""
        candidate_family = ""
    else:
        onset_shift = float(
            int(candidate["onset_bar_ordinal"]) - int(reference["onset_bar_ordinal"])
        )
        resolution_shift = float(
            int(candidate["resolution_bar_ordinal"]) - int(reference["resolution_bar_ordinal"])
        )
        candidate_id = str(candidate["event_id"])
        candidate_family = str(candidate["event_family"])
    return {
        "reference_event_id": str(reference["event_id"]),
        "candidate_event_id": candidate_id,
        "candidate_event_ids": "|".join(candidate_ids or ([candidate_id] if candidate_id else [])),
        "symbol": str(reference["symbol"]),
        "session": str(reference["session"]),
        "reference_segment_id": str(reference["segment_id"]),
        "candidate_segment_id": "" if candidate is None else str(candidate["segment_id"]),
        "reference_family": str(reference["event_family"]),
        "candidate_family": candidate_family,
        "reference_onset_bar_ordinal": int(reference["onset_bar_ordinal"]),
        "candidate_onset_bar_ordinal": (
            math.nan if candidate is None else int(candidate["onset_bar_ordinal"])
        ),
        "reference_resolution_bar_ordinal": int(reference["resolution_bar_ordinal"]),
        "candidate_resolution_bar_ordinal": (
            math.nan if candidate is None else int(candidate["resolution_bar_ordinal"])
        ),
        "onset_shift_bars": onset_shift,
        "resolution_shift_bars": resolution_shift,
        "alignment_class": alignment_class,
    }


def align_event_ledgers(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    tolerance_bars: int = 2,
) -> pd.DataFrame:
    """Classify event matches without reading or assuming numeric latent-state IDs."""

    if tolerance_bars < 0:
        raise ValueError("event alignment tolerance cannot be negative")
    left = _validated(reference, role="reference")
    right = _validated(candidate, role="candidate")
    left_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in left.groupby(["symbol", "session"], sort=False)
    }
    right_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in right.groupby(["symbol", "session"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for _, reference_row in left.iterrows():
        key = (str(reference_row["symbol"]), str(reference_row["session"]))
        candidate_group = right_groups.get(key)
        if candidate_group is None:
            rows.append(_row(reference_row, None, alignment_class="MISSING_EVENT"))
            continue
        overlap_mask = np.asarray(
            [
                _overlap(reference_row, candidate_row, tolerance_bars)
                for _, candidate_row in candidate_group.iterrows()
            ],
            dtype=bool,
        )
        overlaps = candidate_group.loc[overlap_mask]
        if overlaps.empty:
            rows.append(_row(reference_row, None, alignment_class="MISSING_EVENT"))
            continue
        if len(overlaps) > 1:
            first = overlaps.iloc[0]
            rows.append(
                _row(
                    reference_row,
                    first,
                    alignment_class="SPLIT_EVENT",
                    candidate_ids=overlaps["event_id"].astype(str).tolist(),
                )
            )
            continue
        candidate_row = overlaps.iloc[0]
        reverse_overlap_count = sum(
            _overlap(other_reference, candidate_row, tolerance_bars)
            for _, other_reference in left_groups[key].iterrows()
        )
        if reverse_overlap_count > 1:
            rows.append(_row(reference_row, candidate_row, alignment_class="MERGED_EVENT"))
            continue
        same_family = str(reference_row["event_family"]) == str(candidate_row["event_family"])
        onset_shift = abs(
            int(reference_row["onset_bar_ordinal"]) - int(candidate_row["onset_bar_ordinal"])
        )
        resolution_shift = abs(
            int(reference_row["resolution_bar_ordinal"])
            - int(candidate_row["resolution_bar_ordinal"])
        )
        if not same_family:
            classification = "DIFFERENT_FAMILY"
        elif onset_shift == 0 and resolution_shift == 0:
            classification = "EXACT_FAMILY_AND_TIME"
        elif onset_shift <= tolerance_bars and resolution_shift <= tolerance_bars:
            classification = "SAME_FAMILY_BOUNDED_SHIFT"
        else:
            classification = "MISSING_EVENT"
        rows.append(_row(reference_row, candidate_row, alignment_class=classification))
    return pd.DataFrame(rows)


def event_alignment_summary(alignment: pd.DataFrame) -> dict[str, float]:
    """Summarize same-family and bounded timing agreement."""

    if alignment.empty:
        return {
            "reference_event_count": 0.0,
            "same_family_fraction": 0.0,
            "exact_family_and_time_fraction": 0.0,
            "median_timing_disagreement_bars": math.nan,
        }
    classifications = alignment["alignment_class"].astype(str)
    same = classifications.isin(["EXACT_FAMILY_AND_TIME", "SAME_FAMILY_BOUNDED_SHIFT"])
    timing = alignment.loc[same, ["onset_shift_bars", "resolution_shift_bars"]].abs().max(axis=1)
    return {
        "reference_event_count": float(len(alignment)),
        "same_family_fraction": float(same.mean()),
        "exact_family_and_time_fraction": float(classifications.eq("EXACT_FAMILY_AND_TIME").mean()),
        "median_timing_disagreement_bars": (
            float(np.median(timing.to_numpy(dtype=float))) if len(timing) else math.nan
        ),
    }


__all__ = ["align_event_ledgers", "event_alignment_summary"]
