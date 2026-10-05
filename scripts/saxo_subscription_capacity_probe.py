"""Read-only probe of how many streaming subscriptions Saxo lets this session hold.

Saxo documents no subscription-count limit (only REST call quotas), so this measures it: in its
own streaming context it opens cheap one-minute chart subscriptions on the pinned futures, one
per second, until Saxo refuses one or the bound is reached, printing the rate-limit headers Saxo
returns and the refusal, then deletes every subscription it opened. It reuses the service's
current access token without refreshing it and never touches the service's subscriptions. Run
when the service is settled (never during its start-up), as the service user:
  runuser -u stocker -- .venv/bin/python scripts/saxo_subscription_capacity_probe.py \\
      --config /etc/stocker/v1/saxo.live.paper.yaml --state /var/lib/stocker/v1 --max 40
"""

import argparse
import asyncio
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from stocker_execution.config import load
from stocker_execution.saxo_auth import OAuth, SaxoError, private_read
from stocker_execution.saxo_client import SaxoClient

PATH = "/chart/v3/charts/subscriptions"


class CurrentToken(OAuth):
    """Read the service's latest access token; never refresh or rotate it."""

    async def access_token(self) -> str:
        tokens = private_read(self.token_file)
        if tokens.get("environment") != self.environment:
            raise ValueError("TOKEN_ENVIRONMENT_MISMATCH")
        if float(tokens.get("expires_at", 0)) - time.time() < 60:
            raise ValueError("ACCESS_TOKEN_NEAR_EXPIRY_RETRY_SHORTLY")
        return str(tokens["access_token"])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max", type=int, default=40)
    args = parser.parse_args()
    config = load(args.config)
    client = SaxoClient(CurrentToken(config.data_environment, config.saxo, args.state))
    uics = [c.uic for c in config.contracts.values()]
    if not uics:
        raise SystemExit("No contracts pinned in the config: nothing to subscribe to")
    context = "SLRNOCAPPROBE" + secrets.token_hex(6)
    opened: list[str] = []
    try:
        token = await client.oauth.access_token()
        async with connect(
            client.oauth.urls["stream"] + "/connect?" + urlencode({"contextId": context}),
            additional_headers={"Authorization": "Bearer " + token},
            max_size=256 * 1024,
            open_timeout=15,
            close_timeout=5,
        ):
            for i in range(args.max):
                ref = "K" + secrets.token_hex(6)
                try:
                    result = await client.request(
                        "POST",
                        PATH,
                        body={
                            "ContextId": context,
                            "ReferenceId": ref,
                            "RefreshRate": 1000,
                            "Format": "application/json",
                            "Arguments": {
                                "Uic": uics[i % len(uics)],
                                "AssetType": "ContractFutures",
                                "Horizon": 1,
                                "Count": 2,
                                "FieldGroups": ["Data"],
                            },
                        },
                    )
                except SaxoError as exc:
                    print(f"refused at extra subscription #{i + 1}: {exc}")
                    print(f"rate headers: {client.rate_headers}")
                    break
                opened.append(ref)
                limits = {
                    k.replace("x-ratelimit-", ""): v
                    for k, v in client.rate_headers.items()
                    if "remaining" in k or "limit" in k
                }
                print(f"#{i + 1} ok state={result.get('State')} headers={limits}")
                await asyncio.sleep(1)
            else:
                print(f"opened {len(opened)} extra subscriptions without a refusal")
    finally:
        for ref in opened:
            try:
                await client.request("DELETE", f"{PATH}/{context}/{ref}")
            except SaxoError as exc:
                print(f"delete failed: {exc}")
        await client.close()
        print(f"deleted {len(opened)} probe subscriptions; service lines untouched")


if __name__ == "__main__":
    asyncio.run(main())
