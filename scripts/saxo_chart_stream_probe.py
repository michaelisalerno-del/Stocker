"""Read-only probe of Saxo's chart streaming: what a /chart/v3/charts subscription delivers.

Safe beside the running service: it reuses the service's current access token without refreshing
it (the service owns the rotating refresh token), opens its own streaming context, subscribes to
one-minute chart samples for the pinned futures, prints every message it receives for a bounded
time (sample times, OHLCV, DataVersion, heartbeats; never tokens or account keys), compares the
streamed state with one REST GET, then deletes its subscriptions and closes. No order endpoint.

Run as the service user, for example:
  runuser -u stocker -- .venv/bin/python scripts/saxo_chart_stream_probe.py \\
      --config /etc/stocker/v1/saxo.live.paper.yaml --state /var/lib/stocker/v1 --seconds 150
"""

import argparse
import asyncio
import json
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from stocker_execution.config import MARKETS, load
from stocker_execution.saxo_auth import OAuth, SaxoError, private_read
from stocker_execution.saxo_client import SaxoClient
from stocker_execution.saxo_stream import Frames

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


def stamp() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]


def sample_line(sample: dict[str, Any]) -> str:
    keys = ("Time", "Open", "High", "Low", "Close", "Volume", "Interest", "MarketTradingState")
    return " ".join(f"{k}={sample.get(k)}" for k in keys if k in sample) + (
        f" extra={sorted(set(sample) - set(keys))}" if set(sample) - set(keys) else ""
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="Directory holding the ledger")
    parser.add_argument("--markets", default=",".join(MARKETS))
    parser.add_argument("--seconds", type=int, default=150)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    config = load(args.config)
    client = SaxoClient(CurrentToken(config.data_environment, config.saxo, args.state))
    context = "SLRNOCHARTPROBE" + secrets.token_hex(6)
    refs: dict[str, str] = {}
    try:
        token = await client.oauth.access_token()
        async with connect(
            client.oauth.urls["stream"] + "/connect?" + urlencode({"contextId": context}),
            additional_headers={"Authorization": "Bearer " + token},
            max_size=256 * 1024,
            open_timeout=15,
            close_timeout=5,
        ) as socket:
            print(f"{stamp()} stream connected, context {context[:12]}…")
            for market in args.markets.split(","):
                selection = config.contracts.get(market)
                if not selection:
                    print(f"{market}: no pinned contract")
                    continue
                ref = "C" + secrets.token_hex(6)
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
                                "Uic": selection.uic,
                                "AssetType": "ContractFutures",
                                "Horizon": 1,
                                "Count": args.count,
                                "FieldGroups": ["Data", "ChartInfo"],
                            },
                        },
                    )
                except SaxoError as exc:
                    print(f"{stamp()} {market} subscription refused: {exc}")
                    continue
                refs[ref] = market
                snapshot = result.get("Snapshot") or {}
                print(
                    f"{stamp()} {market} {selection.symbol} subscribed: "
                    f"RefreshRate={result.get('RefreshRate')} "
                    f"InactivityTimeout={result.get('InactivityTimeout')} "
                    f"State={result.get('State')} snapshot keys={sorted(snapshot)} "
                    f"DataVersion={snapshot.get('DataVersion')} "
                    f"ChartInfo={snapshot.get('ChartInfo')}"
                )
                for sample in (snapshot.get("Data") or [])[-3:]:
                    print(f"    snapshot sample: {sample_line(sample)}")
            frames = Frames(256 * 1024)
            deadline = time.monotonic() + args.seconds
            print(f"{stamp()} listening for {args.seconds}s …")
            while time.monotonic() < deadline:
                try:
                    payload = await asyncio.wait_for(socket.recv(), timeout=1)
                except TimeoutError:
                    continue
                if not isinstance(payload, bytes):
                    print(f"{stamp()} non-binary frame: {type(payload).__name__}")
                    continue
                for message in frames.feed(payload):
                    ref, body = message["reference"], message["payload"]
                    if ref == "_heartbeat":
                        beats = [
                            (h.get("OriginatingReferenceId", "")[:3], h.get("Reason"))
                            for e in (body if isinstance(body, list) else [body])
                            for h in e.get("Heartbeats", [])
                        ]
                        print(f"{stamp()} heartbeat {beats}")
                        continue
                    if ref.startswith("_"):
                        print(f"{stamp()} control {ref}: {json.dumps(body)[:300]}")
                        continue
                    market = refs.get(ref, ref)
                    envelopes = body if isinstance(body, list) else [body]
                    for envelope in envelopes:
                        data = (
                            envelope.get("Data", envelope)
                            if isinstance(envelope, dict)
                            else envelope
                        )
                        meta = {
                            k: envelope.get(k)
                            for k in ("Timestamp", "PartitionNumber", "TotalPartitions")
                            if isinstance(envelope, dict) and k in envelope
                        }
                        if isinstance(data, dict):
                            samples = data.get("Data") or []
                            other = {k: v for k, v in data.items() if k != "Data"}
                            print(
                                f"{stamp()} {market} update meta={meta} "
                                f"other={json.dumps(other)[:200]} samples={len(samples)}"
                            )
                            for sample in samples:
                                print(f"    {sample_line(sample)}")
                        else:
                            print(
                                f"{stamp()} {market} update meta={meta} "
                                f"payload={json.dumps(data)[:300]}"
                            )
            # One REST read for comparison with the streamed state.
            for market in refs.values():
                selection = config.contracts[market]
                rest = await client.request(
                    "GET",
                    "/chart/v3/charts",
                    params={
                        "Uic": selection.uic,
                        "AssetType": "ContractFutures",
                        "Horizon": 1,
                        "Count": 3,
                        "FieldGroups": "Data,ChartInfo",
                    },
                )
                print(
                    f"{stamp()} {market} REST DataVersion={rest.get('DataVersion')} "
                    f"ChartInfo={rest.get('ChartInfo')}"
                )
                for sample in rest.get("Data") or []:
                    print(f"    rest sample: {sample_line(sample)}")
    finally:
        for ref in refs:
            try:
                await client.request("DELETE", f"{PATH}/{context}/{ref}")
            except SaxoError as exc:
                print(f"{stamp()} delete {ref[:3]} failed: {exc}")
        await client.close()
        print(f"{stamp()} done; subscriptions deleted")


if __name__ == "__main__":
    asyncio.run(main())
