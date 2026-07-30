"""Deterministic ordering policy for prospective SQLite migrations."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_MIGRATION_NAME = re.compile(r"^(?P<sequence>[0-9]{4})_[a-z0-9_]+\.sql$")

# These filenames were already deployed before the unique-prefix policy existed.
# Their full names remain migration identities and their historical order is frozen.
LEGACY_DUPLICATE_ORDER: dict[int, tuple[str, ...]] = {
    11: (
        "0011_m1c_checkpoint_completion_v0.sql",
        "0011_m1c_tail_phase_v1.sql",
    ),
    12: (
        "0012_m1c_signed_market_shock_v1.sql",
        "0012_option_schedule_degradation_v0.sql",
    ),
}


class MigrationOrderError(ValueError):
    """Raised when package migration identities are ambiguous."""


@dataclass(frozen=True)
class Migration:
    sequence: int
    path: Path


def migration_plan(root: Path) -> tuple[Migration, ...]:
    """Return a validated plan with explicit legacy duplicate ordering."""

    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in root.glob("*.sql"):
        match = _MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise MigrationOrderError(f"invalid migration filename: {path.name}")
        sequence = int(match.group("sequence"))
        if sequence < 1:
            raise MigrationOrderError("migration sequences must start at 0001 or later")
        grouped[sequence].append(path)

    plan: list[Migration] = []
    for sequence in sorted(grouped):
        paths_by_name = {path.name: path for path in grouped[sequence]}
        expected_legacy = LEGACY_DUPLICATE_ORDER.get(sequence)
        if expected_legacy is not None:
            if set(paths_by_name) != set(expected_legacy):
                actual = ", ".join(sorted(paths_by_name))
                expected = ", ".join(expected_legacy)
                raise MigrationOrderError(
                    f"legacy migration sequence {sequence:04d} must remain "
                    f"exactly [{expected}]; found [{actual}]"
                )
            ordered_paths = tuple(paths_by_name[name] for name in expected_legacy)
        elif len(paths_by_name) > 1:
            names = ", ".join(sorted(paths_by_name))
            raise MigrationOrderError(
                f"duplicate migration sequence {sequence:04d}: {names}"
            )
        else:
            ordered_paths = tuple(paths_by_name.values())
        plan.extend(Migration(sequence=sequence, path=path) for path in ordered_paths)

    return tuple(plan)


__all__ = [
    "LEGACY_DUPLICATE_ORDER",
    "Migration",
    "MigrationOrderError",
    "migration_plan",
]
