# Stocker agent workflow

## Purpose

This workflow separates architecture, implementation, review, and adversarial testing
so that a single agent does not silently design, implement, and approve a
trading-sensitive change. `AGENTS.md` remains the repository constitution; an accepted
plan under `docs/plans/` defines the bounded work for a particular change.

The repository includes project-scoped custom roles under `.codex/agents/` for Codex
clients that support them. The files intentionally omit model identifiers, so each role
inherits the parent session's verified model. Select the strongest available model and
reasoning level for Architect and Reviewer tasks in the parent session. If a client
does not surface custom roles, run the same stages as separate Codex tasks using the
prompts below.

Codex loads project-local `.codex` configuration only for a trusted checkout. If the
roles do not appear, trust the exact repository path through the client's normal trust
flow; do not copy these settings into a global config as a workaround.

## Architect

The Architect is read-only. It uses the strongest available reasoning model, inspects
the actual repository, and challenges the requested design before accepting it. It
identifies unnecessary complexity and deletion opportunities; defines mode, trust, and
data-ownership boundaries; and analyses failure, restart, uncertain submission, and
reconciliation behavior. It does not implement production code.

Required Architect output:

1. Current-state findings.
2. Target design.
3. Mode and trust boundaries.
4. Schema and API impacts.
5. Migration and rollback.
6. Failure modes.
7. Phased implementation plan.
8. Acceptance tests.
9. Explicit non-goals.
10. Unresolved decisions.

The owner reviews and accepts the plan before implementation. Store the accepted
version under `docs/plans/` with enough context to remain intelligible after the
original conversation is gone.

## Implementer

The Implementer may write to the workspace, but implements one accepted phase only. It
reads `AGENTS.md` and the accepted plan, inspects the current implementation, and does
not redesign neighboring phases.

The Implementer must not change live enablement, credentials, account allowlists,
paper/live selection, or risk limits unless the accepted phase explicitly authorises
that exact change. It runs focused tests, keeps the diff reviewable, commits one bounded
phase, and reports any behavior it could not verify.

## Reviewer

The Reviewer is independent from the Implementer and works read-only with strong
reasoning. It reviews the actual diff and the accepted plan, prioritising correctness
over style.

The review checks, where relevant:

- research/shadow/paper/live separation
- paper configuration reaching a live account
- plugin or web paths bypassing risk
- missing or reusable order-intent identity
- retry-created duplicate orders
- stale or competing sources of broker state
- submission uncertainty and reconciliation gaps
- unsafe migration/cutover behavior
- protected-data contamination
- unbounded operational storage or backups
- architectural drift and unnecessary compatibility layers

Findings are reported in severity order with file references and realistic failure
sequences. The Reviewer does not edit the implementation. If there are no findings, it
says so and lists remaining test or evidence gaps.

## Test Engineer

The Test Engineer owns failure-oriented and adversarial tests. It may write tests and
test support only within the accepted phase; it does not weaken production behavior to
make tests easier.

It uses fake, replay, simulated, or explicitly isolated paper adapters. It never sends
live orders. Depending on the affected subsystem, it tests disconnects, duplicate
callbacks, duplicate intents, submission uncertainty, partial fills, rejection,
cancellation races, stale state, process restart, open-order and position
reconciliation, and emergency stops.

Tests should prove both the expected action and the required non-action: no duplicate
submission, no stale approval, no paper-to-live fall-through, and no inference of fill
or position from intent alone.

## Mechanical Fixer

The Mechanical Fixer handles precise, repetitive work only: scoped renames, confirmed
deletions, documentation corrections, formatting, and explicitly specified test
updates. It makes no architecture, execution, account, risk, mode, or trading decision.
It stops when ambiguity requires owner judgment.

## Recommended sequence

User request
→ Architect
→ user/owner approval
→ accepted plan in `docs/plans/`
→ Implementer for one phase
→ focused tests
→ read-only Reviewer
→ blocking fixes
→ Test Engineer
→ final Reviewer

Central schemas, risk engines, execution paths, broker adapters, reconciliation state
machines, and migration cutovers are implemented sequentially. Do not run multiple
write-capable agents against the same critical subsystem at once.

Parallel agents are better suited to independent, read-heavy work such as repository
exploration, test-gap analysis, performance analysis, documentation verification, and
security review. The project config caps concurrent subagent threads conservatively,
and the project roles do not recursively delegate.

## Copy-paste task examples

### Run the Architect only

```text
Use the project Architect role only. Work read-only and do not implement anything.
Read AGENTS.md and inspect the actual repository before proposing a design for:

<request>

Return the ten required Architect sections from docs/agent-workflow.md. Distinguish
current capability from intended capability, identify all affected operating modes and
trust boundaries, challenge unnecessary complexity, and include phased commits and
acceptance tests. Stop for owner approval; do not create or modify production files.
```

### Implement one approved phase

```text
Use the project Implementer role. Read AGENTS.md and this accepted plan:

docs/plans/<accepted-plan>.md

Implement phase <number/name> only. Do not redesign adjacent phases or change trading
mode enablement, credentials, account allowlists, or risk limits unless that exact
change is explicitly in the phase. Run focused tests, inspect the diff, commit one
reviewable phase, and report unverified behavior and paper/live implications.
```

### Review a completed phase

```text
Use the project Reviewer role and work read-only. Review the actual branch diff for
phase <number/name> against AGENTS.md and docs/plans/<accepted-plan>.md.

Lead with severity-ordered findings and file references. Explain realistic failure
sequences. Focus on mode leakage, risk bypass, duplicate orders, stale state,
reconciliation, migration safety, protected data, storage growth, and architectural
drift. Do not edit files. If there are no findings, say so and list remaining evidence
gaps.
```

### Add adversarial tests

```text
Use the project Test Engineer role. Read AGENTS.md, the accepted plan at
docs/plans/<accepted-plan>.md, and the reviewed implementation diff.

Add failure-oriented tests for phase <number/name> only. Use fake, replay, simulation,
or explicitly isolated paper adapters; never transmit a live order. Cover the relevant
disconnect, duplicate, uncertainty, partial-fill, rejection, cancellation, stale-state,
restart, reconciliation, and emergency-stop cases. Do not weaken production behavior
or expand the phase. Run the focused tests and report what remains untested.
```

### Perform a final review

```text
Use the project Reviewer role and work read-only. Perform the final review of this
branch against AGENTS.md and docs/plans/<accepted-plan>.md after implementation,
blocking fixes, and adversarial tests.

Review the full diff and actual test evidence. Report severity-ordered findings,
paper/live implications, migration implications, and any unverified acceptance
criteria. Do not edit files and do not approve based only on the Implementer's summary.
```

## Stage handoff checklist

Every handoff states:

- the accepted plan and phase
- files changed or reviewed
- operating modes affected
- whether market data, risk, execution, or reconciliation is affected
- tests actually run and tests not run
- paper/live implications
- migration and rollback implications
- known limitations and unresolved decisions

The next role verifies this evidence against the repository rather than trusting the
summary alone.
