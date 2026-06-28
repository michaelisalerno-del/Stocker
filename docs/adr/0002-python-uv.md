# ADR 0002: Python 3.12 And uv

## Status

Accepted.

## Context

Quant Python packages often lag the newest Python release. The project needs stable
research dependencies, reproducible local environments, and simple server bootstraps.

## Decision

Target Python 3.12 and manage environments with `uv`.

## Consequences

- Python 3.12 gives the research stack a conservative compatibility target.
- `uv` provides fast syncs and dependency groups for Mac research, server execution,
  and dev tooling.
- Python 3.13 can be revisited after the full dependency set is proven stable there.
