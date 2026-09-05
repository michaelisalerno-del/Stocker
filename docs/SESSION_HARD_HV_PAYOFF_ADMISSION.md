# Session HARD · HV pooled payoff admission

This updates SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D / SESSION_HARD_HV_V1
in place. No new strategy or broker execution path is installed. Original
SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D is unchanged.

## Authorized final rule

The user's later instructions supersede the original warmup and strategy scope:

- Use the existing HV strategy.
- With fewer than 20 completed pooled observations, continue baseline paper trading
  (TRADE_BASELINE_WARMUP). This includes zero history and unfamiliar symbols.
- With at least 20 observations, require estimated_net_R > 0.
- Record only baseline-qualified, DOWN-first-touch, baseline-fillable TOP5 opportunities.
  Market/cap selection remains dynamic. Unqualified stocks need no payoff record.

For each decision, prior observations satisfy completion_timestamp < signal_timestamp.
The current physical opportunity is explicitly excluded, regardless of supplied history.

    estimated_gross_R = sum(prior hypothetical gross_R) / count(prior)
    stop_distance_bps = abs(entry_reference - initial_stop) / entry_reference * 10000
    cost_R = estimated_round_trip_cost_bps / stop_distance_bps
    estimated_net_R = estimated_gross_R - cost_R
    take_trade = prior_count < 20 or estimated_net_R > 0

No rolling window, decay, stock-specific estimates, confidence bands, extra threshold,
or adaptive parameter is used. Changing ticker alone has no effect on the estimate.

## Placement and prices

StockerRuntime keeps the existing Stage 6 evaluation/first-touch/ranking, then freezes
admission before the existing Stage7StrategyRuntime execution/risk path. Rejections are
not replaced with lower-ranked candidates. Candidate ranking, account capacity and sizing
are preserved.

The entry reference is the existing lower trigger for a single intrabar DOWN touch,
or that minute's open for an opening DOWN gap. Signal time is unchanged: minute completion
for intrabar touches, minute start for gaps. Stop and target use the shared
nominal_exit_prices function, extracted without changing the existing Stage 7 formulas:
entry + 0.50M and entry - 1.00M. Broker tick rounding is unchanged.

RunsConfig.session_hard_hv_round_trip_cost_bps supplies cost independently of the rule.
The default is the existing frozen research assumption, 10 bps. The deployed Stage 8–10
run configuration previously had no transaction-cost input wired into strategy admission;
the unrelated ServerConfig/CostsConfig placeholder was not used by these runs.
The setting is preserved by normal runs-config serialization.

## Outcome accounting and persistence

RuntimeStore adds runtime_session_hard_payoffs to the application's existing SQLite database.
It retains the qualified signal and IBKR instrument identity, original run configuration,
nominal reference/stop, original cost, immutable assessment, completion time, gross R,
and actual-entry-fill flag. The assessment records count, gross estimate, cost bps,
stop bps, cost R, net estimate and explicit decision.

Every baseline-fillable TOP5 opportunity is tracked, whether admitted, skipped by cost,
or subsequently rejected by account risk/capacity. No hypothetical entry is manufactured
for an unfilled/expired/UP/ambiguous/nonselected signal. Actual broker fills are recorded
separately from hypothetical completion. A submission alone is not an actual fill.

The passive outcome calculation uses the frozen Structure D research accounting:
nominal -1R stop, +2R target, stop-first for ambiguous bars, and timeout at the T0+14
minute close (observable at T0+15). This preserves the existing nominal stop-gap
accounting; it does not substitute broker fill P&L. Gross R is short P&L divided by
nominal initial stop risk, without historical transaction-cost subtraction.

Broker exits are unchanged. The repository's pre-existing architecture explicitly
separates the research T0+15 accounting horizon from runtime broker brackets, which have
no forced T0+15 exit. These outcomes are therefore baseline hypothetical observations,
not a claim that broker realised P&L equals research P&L.

Only complete, causally available minute prefixes are used. Missing bars or missing
timeout closes never become fabricated observations. Ordinary entry acquisition receives
only waiting signals. Passive historical downloads run outside ready-order submission;
synchronous admission consumes completed outcomes from available cached bars.
Failed/missing old history cannot block current entries. Pending records survive process,
day, ticker and universe changes and retain their original qualified identities.

The pool spans all exact HV opportunities in this application database; it is not keyed by
ticker, market, cap, session or run. Identical economic opportunities shared by runs are
deduplicated using conId, strategy/version, T0 and geometry/feature lineage.

Initialization reuses already-persisted exact HV signals and IBKR cache identities.
Historical cost/admission estimates for pre-hurdle trades remain null and are labelled
TRADE_BASELINE_PRE_HURDLE. Their gross outcomes can be reconstructed through the same
cached-bar calculation. No research CSV is imported into production.

## Regression evidence

The test-only fixture is an extract of the 296-opportunity original IV-M research ledger.
Source SHA256: ca274ae2e153e5998483b91f87f950efd942ad41b09c97a051eb8a5a80a3fd00.
It verifies admission arithmetic; it is not an HV performance validation or a production seed.

With the originally requested abstaining warmup, ready decisions reproduce 224 passes,
baseline mean net R 0.2352, retained mean 0.3911, mean per opportunity 0.2959,
and total approximately 87.60R.

The later authorized baseline-trading warmup admits the 21 warmup opportunities too:
245 total, sum net R 91.2620223161, mean per taken 0.3724980503,
mean per original opportunity 0.3083176430. No runtime compatibility option was added
to force the old trade count.

Focused tests cover warmup continuation, pass/fail/zero, unseen stocks, pooled independence,
stop-cost sensitivity, strict completion boundaries, overlapping/future/current outcomes,
rejected-opportunity learning, replay gaps/ambiguity/timeouts, restart, cross-run deduplication,
HV isolation, missing-history nonblocking behavior and cached historical initialization.
