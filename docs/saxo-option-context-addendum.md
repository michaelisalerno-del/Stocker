# Saxo option context addendum — 28 September 2026

## Evidence status

- **IMPLEMENTED:** incremental changes on the existing `codex/saxo-only` branch, inspected baseline `ee193ee`. No second pipeline, recorder, strategy, provider, ledger or navigation tab.
- **OFFLINE_TESTED:** 483 Python tests passed (26 new cases), strict type checks on 117 source files, lint/format on the repository targets, desktop/mobile dashboard and OAuth browser checks, and locked server-only installation smoke. Seven existing library/research warnings remain; none are test failures. Synthetic fixtures are not feed evidence.
- **FEED_VERIFIED:** authenticated SIM account and connected stream were directly observed at 17:43 UTC. This does **not** verify futures or option field delivery: no exact underlying selections or option mappings are configured. See the per-market table.
- **DEPLOYED:** pending final verification and the existing immutable-release upgrade. The directly inspected running release before this change was `2ff57756e272b53b900a7c6e1291a97372c62af8`, not the newer local branch baseline.

Execution remains DISABLED and unarmed. No orders/precheck test orders, purchases, agreements or session upgrades were performed. Saxo is the sole active external provider; IBKR remains parked. The five markets, £10 premium-plus-entry-and-reserved-exit policy, one-contract rule, four concurrent positions including reservations, GC monitor-only behavior, and frozen strategy sources remain intact.

## Reused components and actual gaps

Reused OAuth/account isolation, the one websocket/subscription manager, reference discovery, completed-bar/session caches, frozen model-delta ranking, paper broker, durable signal/reservation/fill store, rolling recorder and its writer, authenticated dashboard, Markets page and event/trade details. No dependency was added.

New behavior addresses gaps found in code: per-field analytics receipts; ETO-specific chain analytics; contract-specific condition costs; exact expiry-time evidence; bounded candidate warming/rotation and event pinning; versioned slow metadata; honest option prehistory; and compact option context. Reviewing the working code also exposed a chain-window mismatch, optional-chain failure affecting an underlying subscription, and stale metadata/size ambiguity; these are corrected and regression tested.

`option_context.py` is a small parsing/quality helper used by the existing stream and recorder. It is not a data source or archive. `option_subscription_budget` defaults to 16 (configurable 4–16) within the unchanged total 32 subscriptions; `option_candidate_window` defaults to three strikes (configurable 1–3). Candidate warming leaves four regular-price lines available for owned/pending/event contracts. Existing chain subscriptions each expose one relevant expiry/window; the option budget counts **regular option price subscriptions**, not chain-side indications. Owned/pinned contracts take precedence when one chain window cannot cover every candidate. Missing chain analytics remain missing.

Every option is keyed by environment, asset type and UIC and tied to a verified underlying UIC and approved root. Candidate changes record old/new UIC, receipt time, ranking inputs and reason. Old windows expire under the ordinary rolling policy. An event pins its selected UIC; later delta/candidate changes cannot re-strike it. Warming generates no strategy events. Optional chain PATCHes run in the existing background history worker, outside the decision loop.

## Field and timing conventions

ETO parsing uses `OptionSidePutCall.Uic` and `ContractOptionsGreeks`; FX-only `ContractId`/`DeltaPct` are not used. Bid/ask/sizes/status and actual execution checks use the regular Prices subscription. Chain quotes, last trades and theoretical values remain labelled context and cannot price a paper fill. Explicit null fields clear prior analytics; absent fields retain their own old receipt and become stale. Trade sizes are never summed into volume.

Raw signed delta/gamma/theta/vega and bid/ask/mid volatility are retained. The public ETO schema does not establish all scales or per-time/per-vol-point conventions, so those normalized values remain null with `UNVERIFIED` scaling; no percent multiplication, per-day theta conversion or probability calibration is invented. Regular Prices `MidVol` and chain `MidVolatility` remain distinguishable by source. Rho/theoretical price/ITM probability are retained only if already delivered and are never displayed as strategy win probability. The existing approved rule is **NEAREST_FROZEN_MODEL_DELTA**; provider delta is optional context for that rule, not a replacement input or newly introduced model.

OI is documented in contracts. It has a separate local receipt, raw value, normalized contract count, and unknown effective date. `LastUpdated` is retained as a provider timestamp but is not repurposed as OI publication/effective time. Later receipts cannot be used in an earlier decision snapshot. OI is slow as-of context; after 24 hours it is explicitly stale. Other analytics age after 60 seconds; executable quote/size checks retain the existing five-second limits. These freshness labels do not introduce strategy filters. Unknown values remain null/missing, not zero.

Slow reference/conditions payloads are sanitized, content-versioned, bounded and referenced from event records. Their versions and receipt times are preserved separately from option updates. Refreshes revalidate economic conventions; a changed multiplier, lot, currency or tick rule blocks the affected option rather than silently repricing it. Optional metadata failures preserve quote observation and other workers. Event decision evidence is detached from later cache mutations.

## Contract safety and costs

InstrumentDetails, option-space underlying relationships, expiry, call/put, strike, root, contract size, price conversion factor, quantity/lot rules, currency, ticks, sessions, notice date, settlement and exercise information are retained where returned. The option-space expiry and InstrumentDetails expiry must agree.

`ExpiryDate`/space `Expiry` are dates, not safe trading/model instants. Approved mappings can now supply `expiry_instants` keyed by date plus `expiry_time_evidence`, specific to that root/date and matching the strategy's New York date. Missing evidence blocks selection (`NO_VERIFIED_REAL_0DTE_EXPIRY_TIME`). `LastTradeDate` or the documented conditions `ExpirationTime` establish a separate last-trading deadline only when explicitly dated/time-zoned. Conflicts block. Exchange-local exercise clocks are preserved separately; ambiguous/unknown zones and DST instants are not guessed. Safety checks retain the strategy exit deadline, verified open exit session and existing 120-second margin before the earliest established cutoff. No universal expiry time is supplied.

Costs use `GET /cs/v1/tradingconditions/ContractOptionSpaces/{AccountKey}/{OptionRootId}?Uic=...`, with safely encoded account keys. The generic instrument-conditions endpoint is not used. Fixed/base/per-lot commission schedules and separately returned exchange fees are supported, including explicit zero fees and minimum/maximum fee bounds. Ambiguous tiers, variable tick schemes, percentage scaling, taxes/carrying/holding schedules and unverified FX markups block execution with a precise reason; none become a zero-cost assumption.

One whole contract premium is price × **PriceToContractFactor**, without multiplying ContractSize a second time. The current regular ask plus entry charges yields minimum purchase cost. The existing ask-plus-one-tick conservative purchase limit drives budget premium; entry charges and estimated exit charges are shown separately, converted with contemporaneous GBP/USD evidence when needed. Existing component totals are not added a second time. Budget totals round upward to pennies but quantities never round upward; totals above £10 reject. Conditions are estimates, not booked broker charges. Existing SIM-broker, internal-paper and observation evidence remain separate.

## Recorder and UI

The same recorder preserves qualifying skipped events and reasons, available option history, context, metadata versions and contemporaneous cost assumptions. Pre-trigger coverage excludes observations collected between the strategy timestamp and later subscription/detection. Captures last at least 60 minutes and through a linked trade closure plus five minutes; overlapping events share segments. Raw unpinned data still expire at 900 seconds. No permanent full-session option/Greek archive exists.

The original 32 MiB rolling, 8 MiB/512-item queue, 2 GiB archive and 2 GiB free-disk limits are unchanged. Initial measurements exposed rolling-memory and burst-queue pressure. Lossless compression was added to the existing RAM rows and queued batches; the existing worker still writes the same JSONL.gz archives. Reconstruction, row order, overlap deduplication, interruption recovery and failure caps are tested. Memory/queue accounting uses the actual compressed resident bytes; this is not an increased quota.

Markets shows the owned/event-selected contract or current candidate, actual underlying identity, expiry/trading/strategy exit times, regular quote/spread/sizes and their states, provider delta/IV, volume/OI ages, minimum purchase/budget result, subscription start and actual latest-event prehistory. Secondary fields and candidate/metadata history remain in an expansion. Existing signal/trade detail JSON carries the related option context.

## Directly verified availability before deployment

Read-only application inspection at **2026-09-28 17:43 UTC**, SIM session `Authenticated / Standard / OrdersOnly`, two subscriptions (session and FX), zero positions/reservations:

| Market | Futures search candidates | Selected future quote / volume / OI | Option identity / quote / Greeks / IV / volume / OI / costs | Recorded option prehistory |
| --- | ---: | --- | --- | ---: |
| CL | 16 | Not verified | Not verified | 0 s |
| GC | 6 | Not verified; monitor-only | Not verified | 0 s |
| NG | 7 | Not verified | Not verified | 0 s |
| NQ | 0 | Not verified; search returned none | Not verified | 0 s |
| SI | 10 before the exact-prefix discovery filter | Not verified | Not verified | 0 s |

All five report `REFERENCE_CONTRACT_SELECTION_REQUIRED`. No field is called unavailable by entitlement merely because it has not been requested/observed. The SI prefix filter excludes unrelated search results; post-deployment counts are recorded separately. Authentication is verified; exact future/root selection, approved tolerance/expiry evidence, safety-critical contract fields, relevant quote/analytics entitlements and recording permission remain unresolved. Persistent recording is disabled with `RECORDING_PERMISSION_NOT_VERIFIED`. Optional Greeks/IV do not delay the connected core service.

## Verification and resource evidence

Focused fixtures cover ETO fields/signs/units, missing and stale analytics, chain-price rejection, OI receipt timing/no look-ahead, candidate rotation, pinned selection, separate histories, partial prehistory, detached metadata, sizes, reference changes, cost factors for all five markets, zero/duplicate fees, FX, fractional-contract rejection, £10 rejection, DST/deadline separation, bounded subscriptions and isolated provider failures. Existing tests cover frozen decisions, capacity/reservations, live blocking, recording limits and overlap deduplication. Browser fixtures are labelled as fixtures and assert no additional navigation tab.

Reproducible commands are in `scripts/check.sh`; the bundled Node runtime was used directly for the two browser scripts because `npm` was not on this shell's PATH. Type checks use the repository's `mypy packages apps` target (117 sources). Standards and addendum reviews found no remaining blocking issues after the fixes and regression tests.

Sequential two-hour generated workloads, same final recorder and machine: [baseline](saxo-option-baseline.json), [five options](saxo-option-benchmark.json). Each has five futures at one second; the latter adds five regular option streams at one second plus chain context at two seconds. Three events share two segments. Both finish with 900 seconds in every window, zero gaps, zero queue backlog and no paused reason.

| Measurement | No options | Five options | Increment |
| --- | ---: | ---: | ---: |
| Rolling high water, bytes | 8,048,936 | 13,631,838 | 5,582,902 |
| Queue peak, bytes | 449,500 | 567,546 | 118,046 |
| Peak process RSS, bytes | 89,686,016 | 100,319,232 | 10,633,216 |
| Event archive bytes | 5,029,784 | 7,291,332 | 2,261,548 |
| Accelerated run elapsed, seconds | 171.710 | 256.361 | 84.651 |
| Ingest p99, milliseconds | 9.344 | 10.530 | 1.186 |
| Maximum event-loop delay, milliseconds | 132.190 | 176.637 | 44.447 |

These are measured fixture workloads, including compression and durable writer work, not live-field availability. Process RSS includes transient expanded archive batches and is not the separately enforced rolling-byte limit. Machine background load and batching affect timing/RSS/compressed archive size.

Full-budget burst evidence: [saxo-option-burst.json](saxo-option-burst.json), generated 1,200-second workload, 16 options and five simultaneous market captures. All 21 windows retained 900 seconds; five shared segments; no gaps or paused reason. Rolling high water 25,858,811 bytes, queue peak 3,675,707 bytes, peak process RSS 144,637,952 bytes, ingest p99 7.204 ms and maximum event-loop delay 469.429 ms. This short stress run deliberately ends while captures are open and is not evidence of completed 60-minute captures. The two-hour comparative runs exercise completed windows. All timings are desktop offline measurements, not a production Saxo throughput guarantee.

Initial failing measurements are retained in `saxo-option-uncompressed-*.json` to explain the bounded compression changes; they are not final acceptance results.

## Official references consulted

Checked current official Saxo documentation on 28 September 2026. Documentation example numbers are not instrument observations or fixtures copied as facts.

- [Options Chain guide](https://www.developer.saxo/openapi/learn/options-chain) and [current Options Chain endpoint reference](https://www.developer.saxo/openapi/referencedocs/trade/v1/optionschain). The current PATCH subscription endpoint, not the older guide's `/active` spelling, is used.
- [OptionSidePutCall](https://www.developer.saxo/openapi/referencedocs/trade/v1/optionschain/post__trade__subscriptions/schema-optionsideputcall) and [ContractOptionsGreeks](https://www.developer.saxo/openapi/referencedocs/trade/v1/optionschain/post__trade__subscriptions/schema-contractoptionsgreeks).
- [Regular Prices subscriptions](https://www.developer.saxo/openapi/referencedocs/trade/v1/prices/post__trade__subscriptions), including its Greeks, PriceInfoDetails, InstrumentPriceDetails and timestamps schemas.
- [InstrumentDetails](https://www.developer.saxo/openapi/referencedocs/ref/v1/instruments/get__ref__details/schema-instrumentdetails) and [ContractOptionSpace](https://www.developer.saxo/openapi/referencedocs/ref/v1/instruments/get__ref__contractoptionspaces_optionrootid).
- [Trading Conditions — Contract Option](https://www.developer.saxo/openapi/referencedocs/cs/v1/tradingconditions-contractoption/get__cs_tradingconditions_contractoptionspaces_accountkey_optionrootid), including CommissionLimit, ExchangeFeeRules and CurrencyConversion schemas.
