# ADR 0002: Python 3.12 And uv

## Status

Accepted.

## Context

The runtime needs a stable interpreter target, reproducible local environments and a simple,
locked server bootstrap.

## Decision

Target Python 3.12 and manage environments with `uv` (`uv sync --locked`; the server installs
with `--no-default-groups` so only runtime dependencies ship).

## Consequences

- One lock file covers the runtime, dashboard and dev tooling.
- Python 3.13 can be revisited once every runtime dependency is proven stable there.
