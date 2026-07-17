"""Create-only logger for frozen named/control T0 opportunities and trigger evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
PACKAGE_SOURCE = REPO / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.frozen_named_loop_t0_execution.prospective import (  # noqa: E402
    append_payloads,
    load_payloads,
    open_collection_ledger,
)

CONTRACT_PATH = WORK / "contracts/20260717-frozen-named-loop-t0-execution-realism-v1.json"
DEFAULT_LEDGER = WORK / "prospective/frozen-named-loop-t0-execution-realism-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("opportunity", "trigger", "status"))
    parser.add_argument("--record", type=Path)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "status":
        if args.record is not None or args.dry_run:
            raise ValueError("status does not accept --record or --dry-run")
        status = open_collection_ledger(
            args.ledger_root, contract_path=args.contract
        ).administrative_status()
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    if args.record is None:
        raise ValueError("--record is required for opportunity and trigger stages")
    paths = append_payloads(
        ledger_root=args.ledger_root,
        contract_path=args.contract,
        stage=args.stage,
        records=load_payloads(args.record),
        dry_run=args.dry_run,
    )
    result = {
        "status": "validated_dry_run" if args.dry_run else "created",
        "stage": args.stage,
        "records": len(paths),
        "paths": [str(path) for path in paths],
        "research_only": True,
        "execution_enabled": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
