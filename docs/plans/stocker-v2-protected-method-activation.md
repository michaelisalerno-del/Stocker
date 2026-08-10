# Stocker V2 protected method activation

Status: **Accepted by the owner on 2026-08-09.**

This plan authorises a new, bounded post-Phase-8 capability: port the genuine frozen
prospective methods into the generic Stocker V2 idea-plugin system, then activate them
in a fresh `shadow` run. It is not Phase 9 and does not authorise portfolio, risk,
paper, live, account, order, fill, or reconciliation work.

The approved public test seams are:

1. `IdeaPlugin.requirements()` and `IdeaPlugin.evaluate()` with frozen golden fixtures.
2. `plan_market_data(requirements, interests, capacity) -> MarketDataPlan`.
3. `InstrumentResolver.resolve(interest) -> DiscoveryReceipt`.
4. `SubscriptionController.apply(plan) -> lifecycle results`.
5. Fake or replay callbacks through the recorder into the unchanged generic Ideas and
   Results read model.

## 1. Current-state findings

- Phase 8 is deployed far enough for the compact V2 web application and imported
  evidence to be inspected, but the V2 recorder and backup timers remain stopped.
- The only configured V2 plugin is Opening Leader Continuation V0.
- The frozen legacy evidence contains three additional genuine prospective methods:
  Frozen M1C Signal V0, M1C Quiet State Options V0, and M1C Opening Reversal V1.1.
- The legacy aggregate database was retired under separate owner authorisation. The
  checked 2026-08-05 import snapshot, the 2026-08-06 V1 rollback database, release,
  configuration, units, and Phase-8 recovery evidence remain protected through the
  owner-controlled seven-day rollback window.
- Frozen M1C evidence includes 2,240 checkpoints, 2,160 completion rows, 600 validity
  rows, and 156 high-tail classifications. It produced no eligible M1C episodes.
- Frozen Quiet evidence includes 2,240 checkpoints and 233 bottom-ten classifications,
  but no observations, option contracts, or outcomes.
- Opening Reversal evidence includes 240 predictions and eight barriers. Its legacy
  cross-vendor and transfer gates made every prediction abstain, ineligible, or
  incomplete; those retired EODHD gates must not return.
- The current V2 plugin contract can declare fixed subscriptions only. Quiet and
  Opening Reversal require bounded, causal option discovery and subscription changes.
- The V2 market-data line limit is 100. The frozen Quiet contract can require as many
  as 54 option contracts for one observation, so the legacy eight-line operational
  reduction would not be scientifically complete.

## 2. Target design

Activate exactly four generic plugin cards:

1. Opening Leader Continuation V0.
2. Frozen M1C Signal V0.
3. M1C Quiet State Options V0.
4. M1C Opening Reversal V1.1.

M1C-derived plugins share a pinned, pure calculation module. Quiet and Opening
Reversal recompute their inputs independently and do not consume another plugin's
outputs. A1, C1, R1, tail strata, transition state, capacity rehearsals, and
microstructure diagnostics remain labelled outputs or controls within the applicable
method; they are not separate plugin cards.

Dynamic market data follows one generic path:

```text
plugin evaluation
  -> bounded durable MarketDataInterest
  -> deterministic core planner
  -> core-owned IBKR instrument resolver
  -> core-owned subscription controller
  -> normal generation-fenced market events
  -> independently checkpointed plugin evaluation
```

An interest identifies the underlying, asset kind, DTE or expiry rule, option right,
strike-selection rule, feed, causal event, expiry, required/optional priority, and
maximum contracts. Plugins never see an IBKR client, request ID allocator, broker
credentials, account identity, or order methods.

The Quiet plugin requests the complete bounded frozen option set. The core may pace
sequential snapshots and deduplicate shared contracts, but it must fail closed and
record incomplete evidence if entitlements, pacing, or the 100-line ceiling cannot
support the declared set. It must never silently substitute the old eight-line subset
or claim scientific completeness for partial coverage.

## 3. Mode and trust boundaries

- Development and tests use fake, replay, or isolated market-data adapters only.
- Attended activation creates a fresh `shadow` run with `shadow_protected` data.
- Shadow positions, marks, and outcomes are virtual only. They are never broker
  positions, fills, approvals, or orders.
- The recorder remains the sole V2 SQLite writer and the sole owner of market-data
  planning, instrument resolution, request IDs, subscriptions, cancellation, pacing,
  callback fencing, and gaps.
- Plugins interpret market events and emit bounded evidence or unapproved proposals.
- The web remains query-only and generic. It receives no broker client or mutation
  capability.
- IBKR remains the sole active prospective source. No EODHD parity, provider equality,
  source transfer, or replay promotion gate may be reintroduced.
- No protected result may be used for threshold selection, feature selection, fitting,
  or hypothesis repair. Frozen thresholds and parameters are identities, not tuning
  candidates.

## 4. Schema and API impacts

The Phase-1 migration adds only the generic operational tables needed for dynamic
interests and discovery receipts:

- `market_data_interests`
- `instrument_discovery_receipts`

Existing instruments, subscriptions, market events, idea outputs, shadow positions,
marks, and outcomes remain authoritative. Add indexes only for measured bounded query
or lifecycle plans. Do not add method-specific tables or a second operational
database.

Owner-approved amendment on 2026-08-10: Phase 2 may add one further sequential,
generic migration that generalises `market_events` and `market_event_derivations` for
bounded derived market-data receipts. It may add no domain table, method-specific
schema, route, mutation surface, or paper/live authority. The existing 64 KiB plugin
state and 256-event retained-lineage bounds remain unchanged. The authorised receipt
kinds are generic session-volume baselines, five-minute session prefixes, and
per-contract option-snapshot captures with exact causal input mappings.

The JSON API remains exactly the existing seven GET routes. No method-specific route,
schema, screen, renderer, poller, or mutation is added. Unknown plugins continue to
render through the generic Ideas and Results views.

Phase 2 also uses a schema-neutral, generic runner continuation for bounded causal
rehydration. A plugin checkpoint may request either a 64-event page of declared-kind
ancestors from at most 64 retained roots, or at most 64 exact events discovered by
that traversal. Traversal follows only the generic `prior_receipt` role and is limited
to 78 derivation edges; rehydrated input is limited to 512 KiB, and every root must
remain retained evidence from the same run at or before the committed watermark. The
core persists the continuation beside plugin state in the existing 64 KiB checkpoint
envelope. Continuations run before new input, never advance the ordinary input
watermark, and may not create dynamic interests.
This permits one canonical 20-stock-plus-VTI M1C checkpoint cohort per evaluation
without increasing the 128-output M1C manifest limit or the central 256-event bound.

## 5. Migration, activation, and rollback

- Do not mutate or re-import either V1 source.
- Before the schema migration, create and restore-check a bounded V2 backup.
- Apply the migration with the recorder and web stopped and with the existing writer
  lease rules intact.
- Rollback requires the matching prior release, configuration, and pre-migration V2
  backup; code rollback alone is insufficient after the schema changes.
- Preserve all Phase-8/V1 recovery material through Michael's seven-day rollback
  window unless he separately authorises its retirement.
- Activation uses a new frozen configuration and a new protected run. It never
  backfills the operational V2 database from legacy outputs.
- The recorder stays stopped until all four plugins, the dynamic market-data path,
  focused tests, independent reviews, capacity and entitlement checks are complete.
- After activation, prove a genuine IBKR callback, durable discovery/subscription
  evidence, generic API visibility, and bounded backup restore before declaring the
  method set operational.

## 6. Failure modes

- Unsupported option capacity or entitlements: record a bounded denial/incomplete
  result; do not substitute contracts or start partial Quiet evaluation as complete.
- Duplicate interests or shared contracts: canonical identity and deterministic
  planning deduplicate them without losing per-instance provenance.
- Restart during discovery or subscription change: durable lifecycle identity resumes
  idempotently; it must not allocate duplicate active request IDs.
- Interest expiry or plugin disable: cancel only the generation-fenced subscriptions
  no longer required by any active instance.
- Late callbacks after cancellation/reconnect: tombstones and generation fences reject
  stale authority.
- Pacing denial or farm outage: preserve the causal gap and retry only according to
  bounded core policy.
- Missing, stale, incomplete, or crossed option quotes: produce incomplete evidence,
  not a fabricated mark or outcome.
- Plugin failure: isolate the instance without stopping ingestion or other plugins.
- Full storage or failed backup: retain existing fail-stop and bounded backup policy;
  never delete protected evidence merely to continue recording.
- Mode/configuration mismatch: fail before connecting. Paper/live/unknown modes remain
  unavailable.

## 7. Sequential implementation phases

Each phase is one bounded Implementer commit followed by focused tests and an
independent read-only Reviewer. Blocking findings are fixed and freshly reviewed
before the next phase.

1. **Generic dynamic market data.** Upgrade the plugin contract, add the two generic
   tables in one migration, and implement durable interests, deterministic planning,
   resolution, subscription lifecycle, restart, expiry, pacing, capacity, and callback
   fencing. Activate no new method.
2. **Frozen M1C Signal V0.** Add the pinned pure M1C helper and plugin with frozen
   threshold `0.488333710794033`; retain A1/C1/R1 as labelled generic evidence.
3. **M1C Quiet State Options V0.** Add frozen quiet threshold
   `0.135896965695626`, complete bounded option interests, long-premium observations,
   and defined-risk short-premium virtual proposals. Naked short proposals are
   forbidden.
4. **M1C Opening Reversal V1.1.** Add the frozen checkpoint-six severe negative
   transition to CALL, severe positive transition to PUT, otherwise ABSTAIN contract,
   with the primary 1-DTE pair and no retired cross-vendor gate.
5. **Attended activation.** Freeze hashes/configuration, create the new `shadow` run,
   verify entitlements/capacity/callbacks/restart/API evidence, then enable bounded
   backups. Stop rather than silently reducing the approved method set.

## 8. Acceptance tests

- Frozen golden parity, determinism, parameter identity, ordering, and no backfill for
  every method.
- Dynamic interest canonical identity, bounded persistence, filterable provenance,
  restart recovery, expiry, cancellation races, late callbacks, reconnects, pacing,
  line capacity, and hard 100-line ceiling.
- Exact option DTE/right/strike selection; no full-chain subscription, substitution,
  or hidden eight-line fallback.
- Shared-contract deduplication, priority, required/optional behavior, and per-instance
  provenance.
- Plugin isolation and bounded state/output/interest counts.
- Continuation cursor/filter binding, ancestry and run/watermark validation, restart
  recovery, skewed source arrival, no-new-event draining, canonical cohort ordering,
  and exact 64-event/512-KiB/64-KiB/128-output bounds.
- Gap and staleness blocking, incomplete/crossed quote handling, and virtual outcome
  lineage to exact proposal and market events.
- Shadow-only results conspicuously remain `shadow=true`, `broker_position=false`, and
  `fill=false`.
- The OpenAPI remains GET-only with exactly seven JSON routes; generic rendering works
  for all four and for an unknown plugin.
- No EODHD, transfer, account, position, order, fill, execution, approval, or broker
  mutation surface appears.
- Mode-boundary and no-live-order tests remain green.
- Attended evidence proves real callback/discovery/capacity/restart and a checked V2
  backup restore before the recorder is left running.

## 9. Explicit non-goals

- Phase 9 or later portfolio, risk, intent, paper, live, execution, or reconciliation
  work.
- Broker account reads, positions, orders, fills, or credentials in plugins or web.
- Parameter tuning, fitting, retrospective promotion, or use of protected outcomes.
- Recreating legacy idea-specific tables, routes, screens, reports, renderers, or
  source-transfer paths.
- Reintroducing EODHD or cross-vendor parity into the active runtime.
- Treating A1/C1/R1, strata, capacity rehearsals, or diagnostics as separate methods.
- Restoring Opening Leader P20/P30/BPS20 option diagnostics.
- Migrating all historical experiments or backfilling legacy method outputs.
- Automatically closing the Phase-8 rollback window or deleting its recovery set.

## 10. Resolved owner decisions and remaining attended gates

Resolved on 2026-08-09:

- activate the recommended four-card set;
- use a fresh protected `shadow` run rather than capture-only prospective mode;
- require full bounded Quiet capture, paced if necessary, and fail closed rather than
  silently using an incomplete eight-line subset;
- leave Opening Leader P20/P30/BPS20 diagnostics disabled;
- perform no backfill, tuning, or protected-data fitting;
- keep the recorder stopped until all four methods are reviewed and deployable; and
- accept the five public TDD seams listed at the top of this plan.

Resolved on 2026-08-10:

- authorise the additional generic Phase-2 derived-market-event migration described
  in section 4, after the existing schema proved unable to represent the required
  causal receipt chains without payload-only pseudo-lineage or exceeding the retained
  event bound.

Remaining attended evidence gates are operational facts, not design discretion:

- IB Gateway manual authentication and Read-Only API mode;
- option market-data entitlements and the verified account line allowance;
- measured ability to satisfy the full Quiet capture within the 100-line hard bound;
- first genuine callbacks and causal plugin outputs;
- restart/reconnect recovery; and
- a checked post-activation backup restore.
