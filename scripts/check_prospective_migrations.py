"""Repository check for unambiguous prospective migration identities."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = ROOT / "packages" / "stocker_prospective" / "src"
sys.path.insert(0, str(PACKAGE_ROOT))

from stocker_prospective.migration_order import migration_plan  # noqa: E402

MIGRATION_ROOT = (
    ROOT / "packages" / "stocker_prospective" / "src" / "stocker_prospective" / "migrations"
)


def main() -> None:
    plan = migration_plan(MIGRATION_ROOT)
    print(f"validated {len(plan)} prospective migrations through {plan[-1].sequence:04d}")


if __name__ == "__main__":
    main()
