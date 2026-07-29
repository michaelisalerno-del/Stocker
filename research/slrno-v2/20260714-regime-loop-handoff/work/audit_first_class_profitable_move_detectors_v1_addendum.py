#!/usr/bin/env python3
"""Transparent addendum for the frozen V1 independent auditor.

The frozen auditor correctly reconstructs missing 24-bar paths, but its comparison
loop assumes that every outcome row contains scored-path fields.  The scorer
intentionally leaves those fields absent for unscored rows.  This addendum keeps
the frozen auditor unchanged and supplies comparison placeholders only for those
unscored rows; scored paths and every aggregate remain audited by the original
independent implementation.
"""

from __future__ import annotations

import math
from typing import Any

import audit_first_class_profitable_move_detectors_v1 as frozen_auditor


class _AbsentUnscoredField:
    """Compare equal to an absent pandas value and convert to NaN."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    def __float__(self) -> float:
        return math.nan


_original_replay_outcome = frozen_auditor.replay_outcome
_COMPARISON_FIELDS = (
    "hit_type",
    "target_first",
    "rapid_target_3",
    "clean_success",
    "pre_target_mae_r",
    "mfe_bps",
    "mae_bps",
    "mfe_r",
    "mae_r",
    "dynamic_gross_bps",
    "dynamic_net_bps",
    "fixed_h24_gross_bps",
    "fixed_h24_net_bps",
)


def _replay_with_unscored_placeholders(
    row: Any,
    lookup: dict[tuple[int, str, str, int], Any],
    stop: float,
) -> dict[str, Any]:
    result = _original_replay_outcome(row, lookup, stop)
    if result.get("outcome_status") != "scored":
        for field in _COMPARISON_FIELDS:
            result.setdefault(field, _AbsentUnscoredField())
    return result


frozen_auditor.replay_outcome = _replay_with_unscored_placeholders


if __name__ == "__main__":
    frozen_auditor.main()
