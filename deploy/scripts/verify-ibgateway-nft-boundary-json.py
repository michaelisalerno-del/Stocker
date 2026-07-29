#!/usr/bin/python3
"""Verify the exact high-priority nftables guard for the Gateway API port."""

from __future__ import annotations

import json
import sys
from typing import NoReturn


def fail(reason: str) -> NoReturn:
    print(f"ibgateway_nft_boundary:{reason}", file=sys.stderr)
    raise SystemExit(78)


def tcp_port_match(port: int) -> dict:
    return {
        "match": {
            "op": "==",
            "left": {
                "payload": {
                    "protocol": "tcp",
                    "field": "dport",
                }
            },
            "right": port,
        }
    }


def main() -> None:
    if len(sys.argv) != 2:
        fail("expected_upstream_port")
    try:
        port = int(sys.argv[1])
    except ValueError:
        fail("upstream_port_must_be_numeric")
    if not 1 <= port <= 65535:
        fail("upstream_port_out_of_range")

    try:
        document = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        fail("invalid_nft_json")
    items = document.get("nftables")
    if not isinstance(items, list):
        fail("invalid_nft_json")

    chains = [
        item["chain"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("chain"), dict)
        and item["chain"].get("family") == "inet"
        and item["chain"].get("table") == "stocker_ibgateway"
    ]
    if len(chains) != 1:
        fail("exact_guard_chain_required")
    chain = chains[0]
    expected_chain = {
        "name": "input",
        "type": "filter",
        "hook": "input",
        "prio": -300,
        "policy": "accept",
    }
    if any(chain.get(key) != value for key, value in expected_chain.items()):
        fail("high_priority_input_chain_required")

    rules = [
        item["rule"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("rule"), dict)
        and item["rule"].get("family") == "inet"
        and item["rule"].get("table") == "stocker_ibgateway"
    ]
    if len(rules) != 2 or any(rule.get("chain") != "input" for rule in rules):
        fail("exact_guard_rules_required")

    expected_allow = [
        {
            "match": {
                "op": "==",
                "left": {"meta": {"key": "iifname"}},
                "right": "lo",
            }
        },
        tcp_port_match(port),
        {"accept": None},
    ]
    expected_drop = [
        tcp_port_match(port),
        {"drop": None},
    ]
    if rules[0].get("expr") != expected_allow:
        fail("exact_guard_rules_required")
    if rules[1].get("expr") != expected_drop:
        fail("exact_guard_rules_required")

    print(f"ibgateway_nft_boundary:verified:{port}")


if __name__ == "__main__":
    main()
