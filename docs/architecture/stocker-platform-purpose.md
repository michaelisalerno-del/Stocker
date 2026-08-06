# Stocker platform purpose

## Purpose and scope

Stocker is a personal algorithmic-trading platform intended to carry an idea through a
controlled research-to-live lifecycle. Its long-term purpose includes historical
research, prospective evidence collection, strategy evaluation, shadow operation,
paper execution, and eventually tightly controlled live execution.

That purpose is a direction, not a statement that every stage is implemented or safe.
The stages are separate operating modes with different permissions, data rules, and
failure consequences. Promotion from one mode to another requires explicit evidence
and operator authorisation; it must never happen because a component happens to expose
a similarly named method.

## Current capability versus intended capability

At the time this document was introduced, the repository implements:

- historical data ingestion, research, and backtesting tools
- a record-only/shadow prospective evidence recorder
- an optional market-data-only IBKR adapter for prospective recording
- deterministic replay and virtual/shadow outcome projections
- a read-only prospective web application
- authority-free V2 domain and first-party idea-plugin contracts

It does **not** implement an order-capable IBKR adapter, a safely reconciled paper
execution service, or live trading. The prior in-memory paper broker, direct submission
interface, stateless risk/execution placeholders, and executor dry run were removed in
V2 Phase 1 because they did not satisfy the approval, idempotency, account-validation,
restart, or reconciliation requirements below.

## Operating modes

### Research

Research consumes historical or deliberately admitted live data to produce analyses,
observations, signals, and proposed trades. It may reject, compare, or refine ideas.
It has no authority to submit, modify, or cancel an order.

Historical research data may come from archived sources, including EODHD datasets.
Those datasets remain research inputs and do not define the active broker or runtime
market-data architecture.

### Shadow

Shadow mode evaluates ideas against live or replayed market data and records virtual
positions and outcomes. It exists to expose timing, data-quality, operational, and
strategy failures without broker-order risk. Shadow records are evidence, not fills or
broker positions, and shadow mode has no order authority.

### Paper

Paper mode is future order-capable operation against an explicitly identified IBKR
paper account. A paper order must pass the same durable intent, approval, idempotency,
and reconciliation discipline expected of live operation.

Paper is not merely a configuration label. The execution subsystem must validate that
the connected account is an allowed paper account before transmission. Unknown,
missing, conflicting, or live account identity must fail closed.

### Live

Live mode is a legitimate future capability with real capital consequences. It remains
disabled by default until an accepted implementation plan and tests demonstrate all
required controls, including explicit operator authorisation, live-account allowlisting,
server-side risk approval, durable idempotent intents, order and position
reconciliation, emergency stop, and unmistakable mode display.

No current research result, shadow outcome, paper placeholder, or broker connection
authorises live trading.

## Trust boundaries and data ownership

The intended flow is:

Market-data ingestion
→ idea plugin
→ signal or proposed trade
→ portfolio construction
→ risk approval
→ durable approved order intent
→ paper or live execution adapter
→ broker acknowledgement and fills
→ reconciliation
→ persisted projections
→ operator web application

Each boundary narrows authority.

### Ingestion

Ingestion owns market-data acquisition, source identity, ordering, timestamps, gaps,
duplicates, staleness, and durable raw evidence. It may publish validated market events.
It cannot create an approved order or infer broker state.

IBKR is the intended source for active prospective, paper, and live market data.
Historical research datasets remain separately usable, but active-runtime correctness
must not depend on EODHD/IBKR equality, cross-vendor matching, or source-transfer gates.

### Idea plugins

Idea plugins own idea-specific interpretation of admitted data. They may emit
observations, signals, target positions, or proposed trades through a generic contract.
They cannot approve their own proposals.

An idea plugin must not possess an IBKR client, broker credentials, account identity,
mode-selection authority, or order methods. Otherwise a strategy bug, compromised
plugin, or innocent refactor could bypass portfolio and risk policy and directly affect
capital. Keeping broker access outside plugins also makes an idea portable across
research, replay, shadow, paper, and live evaluation without changing its authority.

### Portfolio construction

Portfolio construction combines eligible proposals with current portfolio objectives
and constraints. It owns desired exposure, not broker submission. Its output remains a
request for risk approval.

### Risk approval

Risk is authoritative for allow/deny decisions based on mode, account, exposure,
limits, freshness, emergency state, and the latest reconciled broker state. Approval
must be explicit, attributable, and bound to the exact intent it assessed. A missing or
stale prerequisite rejects the request.

### Execution

Execution is the only subsystem allowed to hold an order-capable broker connection.
It accepts only durable approved intents, validates mode and account, assigns stable
idempotency identity, transmits, and records the broker response. Submission is not a
fill, and a timeout is an uncertain state rather than permission to retry blindly.

Paper and live use distinct account validation. Paper configuration must prove that it
targets an allowlisted paper account. Live configuration must separately prove an
allowlisted live account and explicit operator authorisation. No default or fallback
may translate one into the other.

### Reconciliation

Reconciliation owns broker-facing reads for account identity, open/completed orders,
executions, fills, cash, and positions. These reads belong beside execution because
they interpret broker identities and repair uncertainty created by transmission,
disconnects, callbacks, restarts, and manual broker-side actions.

Market-data ingestion and idea code do not need these privileges. Duplicating
broker-state reads elsewhere would create competing truths and make it possible for the
web, a plugin, or a local projection to overrule the actual broker.

Reconciliation compares broker truth with durable local intents and events. It must
surface mismatches and block potentially duplicative action until uncertainty is
resolved. Local desired or intended positions never substitute for broker positions.

### Storage

Storage persists immutable evidence, approvals, intents, broker identities,
acknowledgements, fills, reconciliation results, and bounded operator projections. Each
operational database has one authoritative writer. Migrations create a new safe state
or copy; they do not mutate a legacy operational database in place without an explicit
cutover and rollback plan.

Permanent evidence and backups require measured bounds. Backups are integrity checked,
compressed, rotated, and size limited. Retention policy must preserve required audit
and recovery evidence without silently allowing disks to grow without limit.

### Web

The web application is an operator interface. It reads bounded persisted projections
produced by authoritative subsystems. It does not connect directly to IBKR or become a
second owner of account, order, fill, position, or risk state.

This boundary keeps browser availability, polling, authentication, and presentation
bugs from changing broker state. Future operator actions initiated in the web layer
must still pass through authenticated commands and server-side policy; the browser
never becomes the policy authority.

## Active market-data direction and EODHD

EODHD remains present in the repository for historical ingestion, existing research,
and reproducibility of prior experiments. The architectural direction is to remove it
from the active prospective/paper/live runtime, together with cross-vendor matching,
source-transfer gates, and provider-equality requirements.

This direction does not claim that all legacy prospective code has already been
removed. Existing EODHD-related active-runtime paths must not be expanded or preserved
through speculative compatibility layers. Their eventual deletion belongs in an
accepted, tested implementation plan. Historical EODHD datasets and research code may
remain separate.

## Protected-data boundaries

Development, retrospective assessment, stress, prospective, shadow, paper, and live
datasets have different epistemic roles. Prospective or operational results must not be
silently opened for fitting, feature selection, threshold selection, parameter tuning,
or repeated hypothesis repair.

Any reuse of protected data requires an explicit authorised research protocol that
names the period, purpose, permitted analysis, resulting contamination, and future
holdout. Audit records should make it possible to tell which idea/version and data
boundary produced every decision.

## Simplicity and deletion

Stocker should use the smallest dependable architecture that can demonstrate its
safety properties. A future requirement alone does not justify a service, message bus,
workflow engine, duplicated projection, compatibility wrapper, or generic framework.

Prefer deep, narrow modules with one source of authority. Keep idea-specific behavior
behind a generic plugin contract. When an accepted plan replaces an active path, delete
the superseded runtime code rather than preserving dead architecture indefinitely.
Do not remove historical research code merely because the active runtime changes.
