# Candidate availability fix — 14 September 2026

## Outcome

The new selectable PAPER method is `SESSION_HARD_CAUSAL_Q1_AVAILABLE_V10`, with candidate
recipe `SESSION_HARD_RANGE5_250_RV10_50_RV15_30_AVAILABLE_V2`.
It skips stocks whose exact opening prefix is absent, incomplete or invalid and proceeds
with the valid stocks. The existing maximum capacities remain 250, then 50, then 30;
fewer valid stocks produce a smaller set, never fabricated entries or missing-score fillers.

This addresses the observed LSE failure and the same coverage risk found in Korea. It does
not guarantee a completed session: an empty valid population, permission/connection/pacing
failure, or expired processing window still degrades the session and prevents new entries.

## Behavior and evidence

- Same exact one-minute TRADES/RTH prefixes and unchanged Range/RV formulas and tie ordering.
- Stock-local missing/invalid scores remain in the immutable stage audit with `selected=false`,
  their reason and actual returned partial bars. Subsequent stages request only survivors.
- No post-selection refill, reintroduction of excluded stocks, later-data repair, bar filling,
  changed scanner coverage, increased subscription limits or changed trading/risk rules.
- The run details show the count excluded at each stage; paginated diagnostics retain identities.
- The delayed oracle uses the saved availability policy, and its output includes the candidate
  recipe and missing-data policy. Old oracle sessions without these fields retain old behavior.
- V10 is labelled `UNVALIDATED_AVAILABILITY_POLICY`. Frozen research mathematics remain a
  reference; the changed availability policy has not acquired performance validation.

All 14 V9 specification hashes were captured before changing the catalogue and match afterward.
V7/V8 hash fixtures also remain valid. V9 is still runnable through the same method composition;
the incomplete-prefix regression confirms that its original strict behavior is preserved.
New run creation selects V10. The new version is PAPER-only at the admission boundary.

## Replay using today's original LSE inputs

The [replay summary](candidate-availability-replay-20260914.json) uses the persisted first-stage
inputs for all 458 acquired stocks from the failed LSE session. No later bars were added and
no production state was written.

| Observation | Result |
| --- | ---: |
| Original population | 458 |
| Original selected slots | 250 |
| Missing-score stocks in those slots | 48 |
| New valid selected stocks | 202 |
| New excluded missing-score stocks | 256 |
| Scores matching the original saved values, including missingness | 458/458 |

The 202 valid stocks retain exactly the same order. Under the new policy these inputs can
advance through the first stage instead of failing for missing prefixes. This is first-stage
replay evidence, not proof of later-stage completion or trading outcomes. The separate later
recheck found 292 complete prefixes; those extra bars were intentionally not used here.

## Prepared configuration; not applied

`.stocker/available-candidates-runs.yaml` is a validated, separate configuration based on the
current server configuration. It preserves the exact US and ASX runs and creates disabled
V10 replacements only for LSE and Korea. Their risk settings and broad membership are identical
to the originals. The original LSE/Korea specifications and hashes remain intact, with those
old runs archived/disabled only in this proposed file.

| Market | Existing run suffix | Prepared V10 suffix |
| --- | --- | --- |
| LSE | `41e88f0f06cb` | `057e595a67cd` |
| Korea | `f2bc5e08b2e3` | `426a1231490c` |

The full mapping is `.stocker/available-candidates-mapping.json`. These files have not been
uploaded, applied or enabled. The server still runs form-only release `c8a3753`. Deployment
must include the prior tested FX and pause/frozen-write corrections as well as V10. After
deployment and configuration activation, new sessions must begin before their opening
selection windows; today's saved failed selections cannot be resumed as hypothetical winners.

## LSE scanner warning: narrowed, still unresolved at IBKR

The Gateway's saved capability XML advertises `STK.EU.LSE`, instrument `STOCK.EU`, display name
United Kingdom (LSE), route exchange LSE. No other advertised location code containing IOB
or AIM was found in the examined XML. Its access string is
`restricted;s=1321;s=180;s=181;s=183;s=184;s=185;s=6666`; this is broker capability metadata,
not a decoded statement of the account's paid subscriptions.

At 09:19:26 UTC, a fresh execution-disabled PAPER API client explicitly set market-data
type 1 and requested TOP_TRADE_RATE with the advertised LSE location/instrument. It returned
50 rows and the same warning 492 requiring United Kingdom (LSE) real-time permission for
precise scanner results. Together with the earlier five-family/filter matrix and successful
direct LSE quotes, this rules out the tested stock-type filter and an implicit delayed-data
request as explanations. It does not identify the missing entitlement or prove that a
fresh Gateway login will clear a cached scanner entitlement.

There is no justified application-side switch to suppress this broker warning or substitute
another scanner market. The existing warning remains visible in acquisition diagnostics.
IBKR's pricing page lists LSE UK L1 separately, but pricing alone does not establish which
entitlement the scanner requires. [Official market-data pricing](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php).
The unsent support draft in [the earlier report](market-readiness-20260914.md) contains the
exact affected families. A fresh authenticated Gateway session or IBKR clarification is the
remaining broker-side step; this investigation has not logged out the functioning Gateway,
purchased subscriptions, sent a support message or submitted orders.

## Validation

The actual missing-prefix failure was reproduced before the fix through AcquiredCandidates,
the real opening source/history cache and the candidate pipeline. Regression tests cover
stock-local empty/partial responses, retained audit evidence, shrinkage at later stages,
restart without replenishment, zero valid candidates, permission/connection/pacing failures,
deadlines, V9 behavior and hashes, PAPER-only admission, and matching oracle exclusion rules.
Final validation: **1,233 passed, 14 skipped, 10 existing warnings** in the full Python suite
(109.46 seconds). All three browser suites passed, including the visible excluded-stock
counts. Ruff lint and formatting (281 files), mypy (146 source files), fresh server-only
installation/offline startup and `git diff --check` passed. The final read-only server check
still reports deployed `c8a3753`, PAPER connected/reconciled/ready, zero positions and zero
open orders. No debug instrumentation was added to production paths.
