"""Repository check for unambiguous prospective migration identities."""

from __future__ import annotations

from pathlib import Path

from stocker_prospective.migration_order import migration_plan

ROOT = Path(__file__).parents[1]
MIGRATION_ROOT = (
    ROOT
    / "packages"
    / "stocker_prospective"
    / "src"
    / "stocker_prospective"
    / "migrations"
)


def main() -> None:
    plan = migration_plan(MIGRATION_ROOT)
    print(f"validated {len(plan)} prospective migrations through {plan[-1].sequence:04d}")


if __name__ == "__main__":
    main()
