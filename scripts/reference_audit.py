"""Write the reference-session selection audit for every pinned market. Read-only Saxo access.

It is for today's CME session, or, from 17:00 New York (after the close), for the next weekday's,
which opens at 18:00 (2026-10-04: clocks run through the whole session).

Rule (the user's standing approval, 2026-09-30): for each of the previous five completed sessions,
the reference contract is the standard nearby future with the greatest Saxo daily-chart volume on
the session before it, among contracts not yet expired on that session. The runtime still fetches
and checks every selected session's one-minute bars itself.

The running service owns the OAuth grant. This script only reads its current access token and never
refreshes or rotates it; if that token is not current it stops and the runtime stays blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from stocker_execution.config import ContractSelection, FuturesConfig, load
from stocker_execution.contracts import future_identity
from stocker_execution.reference_sessions import SelectedSession, SelectionAudit
from stocker_execution.rules import next_weekday

NY = ZoneInfo("America/New_York")
BASES = {
    "SAXO_SIM": "https://gateway.saxobank.com/sim/openapi",
    "SAXO_LIVE": "https://gateway.saxobank.com/openapi",
}
NEARBY = 4


def audit_day(now: datetime) -> str:
    """The session the audit prepares: today's, or from 17:00 New York (the close) and at weekends
    the next. This runs an hour before `rules.session_day` turns over at the 18:00 open on
    purpose, so the 17:10 timer's file names the coming session. The runtime reads the file only
    once its own day matches, so a restart between 17:10 and 18:00 sees a mismatch and retries
    15 minutes later, in time for the first evening clock at 19:00."""
    local = now.astimezone(NY)
    later = local.hour >= 17 or local.weekday() > 4
    return (next_weekday(local.date()) if later else local.date()).isoformat()


def build_audit(
    market: str,
    environment: str,
    pinned: ContractSelection,
    nearby: list[dict[str, Any]],
    volumes: dict[int, dict[str, float]],
    today: str,
    approval: str,
) -> SelectionAudit:
    """Pure selection: nearby identities (with expiry) and daily volumes by UIC and session date."""
    days = sorted(d for d in volumes.get(pinned.uic, {}) if d < today)[-6:]
    if len(days) < 6:
        raise ValueError(f"{market}_FEWER_THAN_SIX_COMPLETED_DAILY_SAMPLES")
    sessions = []
    for before, day in zip(days, days[1:], strict=False):
        alive = [
            f for f in nearby if f["expiry"][:10] > day and before in volumes.get(f["uic"], {})
        ]
        if not alive:
            raise ValueError(f"{market}_NO_CONTRACT_WITH_VOLUME_ON_{before}")
        chosen = max(alive, key=lambda f: volumes[f["uic"]][before])
        evidence = f"Saxo daily chart volume on {before}: " + "; ".join(
            f"{f['symbol']} {volumes[f['uic']][before]:,.0f}" for f in alive
        )
        sessions.append(
            SelectedSession(
                day=day,
                contract=ContractSelection(
                    environment=environment,
                    uic=chosen["uic"],
                    symbol=chosen["symbol"],
                    exchange=chosen["exchange"],
                    contract_month=chosen["contract_month"],
                    approval=approval,
                ),
                prior_session_volume_evidence=evidence,
            )
        )
    return SelectionAudit(
        provider="SAXO",
        environment=environment,
        market=market,
        as_of=today,
        current_uic=pinned.uic,
        selection_rule="PRIOR_COMPLETED_EXCHANGE_SESSION_VOLUME",
        approval=approval,
        sessions=sessions,
    )


def fetch(config: FuturesConfig, token: str, account_key: str, market: str) -> tuple[list, dict]:
    headers = {"Authorization": "Bearer " + token}
    with httpx.Client(base_url=BASES[config.data_environment], headers=headers, timeout=30) as http:

        def get(path: str, **params: Any) -> dict[str, Any]:
            response = http.get(path, params=params)
            response.raise_for_status()
            return response.json()

        found = get(
            "/ref/v1/instruments", Keywords=market, AssetTypes="ContractFutures", **{"$top": 100}
        )
        nearby = []
        # The standard family only (contracts.future_identity's pattern), before any details GET:
        # minis, micros and spreads share the keyword and the RefData minute limit is the service's.
        family = re.compile(market + r"[FGHJKMNQUVXZ][0-9]{1,4}(?::[A-Za-z0-9_-]+)?")
        for row in found.get("Data", []):
            if row.get("AssetType") != "ContractFutures" or not family.fullmatch(
                str(row.get("Symbol", ""))
            ):
                continue
            raw = get(
                f"/ref/v1/instruments/details/{row['Identifier']}/ContractFutures",
                AccountKey=account_key,
            )
            try:
                nearby.append(future_identity(market, config.data_environment, raw))
            except ValueError:
                continue  # mini, micro and spread products are not the standard family
        today = datetime.now(NY).date().isoformat()
        nearby = sorted(
            (f for f in nearby if f["expiry"][:10] >= today), key=lambda f: f["expiry"]
        )[:NEARBY]
        volumes = {}
        for f in nearby:
            chart = get(
                "/chart/v3/charts",
                Uic=f["uic"],
                AssetType="ContractFutures",
                Horizon=1440,
                Count=10,
                FieldGroups="Data",
            )
            volumes[f["uic"]] = {
                str(c["Time"])[:10]: float(c["Volume"])
                for c in chart.get("Data", [])
                if c.get("Volume") is not None
            }
        return nearby, volumes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    config = load(args.config)
    if not config.reference_selections_file or not config.saxo.credentials_file:
        raise SystemExit("REFERENCE_SELECTIONS_FILE_AND_CREDENTIALS_REQUIRED")
    tokens = json.loads(args.tokens.read_text())
    if (
        tokens.get("environment") != config.data_environment
        or float(tokens.get("expires_at", 0)) <= time.time() + 60
    ):
        raise SystemExit("SERVICE_ACCESS_TOKEN_NOT_CURRENT")
    account_key = json.loads(config.saxo.credentials_file.read_text())["account_key"]
    today = audit_day(datetime.now(NY))
    books = []
    for market, pinned in config.contracts.items():
        nearby, volumes = fetch(config, tokens["access_token"], account_key, market)
        books.append(
            build_audit(
                market, config.data_environment, pinned, nearby, volumes, today, args.approval
            )
        )
    target = config.reference_selections_file
    tmp = target.with_suffix(".new")
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640), "w") as f:
        json.dump([json.loads(b.model_dump_json()) for b in books], f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    print(
        json.dumps(
            {
                "as_of": today,
                "markets": [b.market for b in books],
                "latest": {b.market: b.sessions[-1].contract.symbol for b in books},
            }
        )
    )


if __name__ == "__main__":
    main()
