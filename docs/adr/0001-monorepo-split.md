# ADR 0001: Monorepo Split

## Status

Accepted.

## Context

Stocker needs heavy research on macOS and lightweight future execution on Linux. The
codebase also needs shared types, config, and safety logic without copying code between
machines.

## Decision

Use one repository with separate `apps/desktop`, `apps/server`, and focused shared
packages under `packages/`.

## Consequences

- Research can use heavy dependencies without forcing the server to install every
  research tool.
- Shared package boundaries make risk and execution code testable.
- The server can stay small by syncing only the `server` dependency group.
- The repo remains simple enough for early-stage development.
