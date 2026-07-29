# Frozen loop-movement prospective shadow contract

## Scope and seal

This contract was frozen at `2026-07-10T18:29:16Z`, before any post-freeze
outcome was opened. Eligible prediction sessions satisfy `session_date >
2026-07-10`. The exact machine-readable contract is
`work/contracts/20260710-frozen-loop-movement-shadow-v1.json`.

The experiment is research-only. `research_only: true`,
`live_ordering_enabled: false`, and `order_placement: disabled`. It contains no
broker, order, position, deployment, direction, signed-return, P&L, cost, or
strategy-promotion surface.

## Frozen lineage

- Eight-state causal forward semi-Markov detector, fitted on 2024 only.
- Last-three-state fixed-loop identity probabilities for the twenty frozen
  cycles, fitted on 2024 only.
- Exact 2024-fitted `state_context` and `loop_scores` Ridge movement models.
- Targets are only 6-, 12-, and 24-bar absolute return and future range.
- No refit, recalibration, threshold tuning, feature change, or alternate model
  may enter this prospective decision.

EODHD bars are regular-session five-minute provider OHLCV. Provider volume is
called `historical_volume`; it is neither exchange-wide volume nor order flow.
The state detector retains its frozen historical-volume features. The movement
regressions do not use volume directly.

## Hypotheses considered

| Hypothesis | Target | Expected research benefit | Safety risk | Validation | Stop condition |
| --- | --- | --- | --- | --- | --- |
| H1: exact frozen inference can be issued causally | shadow runner and sealed ledger | establish a real forward record rather than another retrospective score | accidentally reading a future bar or refitting | exact frozen hashes, as-of slicing, run-entry-only anchors, 5–29 minute issuance window, stored feature digest | any hash, timestamp, feature, or duplicate-ID check fails |
| H2: loop scores retain absolute-movement information | absolute return at 6/12/24 bars | prospective test of the retained ~1% MSE signal | optional stopping or weakening a weak result | first eligible ledger prefix; MSE/MAE intervals, horizons, quarters, stock deletions, correlations | any frozen gate fails |
| H3: loop scores retain future-range information | intrahorizon range at 6/12/24 bars | prospective test of the retained ~3% MSE signal | inspecting range outcomes before adequate support | identical sealed cohort and robustness checks | any frozen gate fails |
| H4: an outcome embargo is enforceable in the harness | status/evaluation boundary | prevent repeated peeking while support accumulates | a status path loading provider prices or an early evaluation calculating losses | support is derived from prediction metadata only; evaluator checks support before provider access | any early metric or outcome access is possible |

The smallest retained experiment is H1–H4 as one shadow harness. Large-move
probabilities/quantiles, a joint history-duration kernel, and an alternative
regime-discovery V2 remain separate development work and cannot affect this
contract.

## Prediction issuance

Each ledger row is an exact causal state-run entry whose provider timestamp
equals the declared as-of timestamp. The provider snapshot is sliced at that
timestamp before feature calculation. Issuance must occur no earlier than five
minutes and no later than twenty-nine minutes after the bar timestamp, so the
completed anchor is available but the six-bar outcome is not. Each anchor
records both frozen representations for both movement targets at all three
horizons.

Prediction batches are immutable Parquet files referenced by an append-only,
SHA-256-chained JSON ledger. Duplicate prediction IDs are forbidden. Neither
`issue` nor `status` may load outcome bars or calculate performance.

## Cohort and embargo

The primary cohort closes at the first ledger prefix containing all of:

- at least 65,000 issued anchors;
- at least 200 distinct session dates;
- at least 18 distinct stocks;
- at least four calendar quarters;
- all eight frozen states.

The primary evaluator uses only that first qualifying prefix. Later
predictions cannot be substituted, and optional stopping is forbidden. Before
the prefix exists, evaluation must stop before reading provider data.

After the issuance gate closes, evaluation requires at least 60,000 anchors
with exact five-minute outcome support through all three horizons. A timestamp
support failure may be reported, but no loss, correlation, or pass/fail metric
may be calculated on an under-supported cohort. Evaluation cannot begin until
125 minutes after the closing prefix's latest anchor, when its 24-bar outcome
bar has completed. The first mature timestamp-support audit is sealed; a later
data correction cannot be used to replace a failed support decision.

## Frozen outcome gates

For `loop_scores` versus `state_context`:

- absolute-return MSE improvement must be at least 1% pooled, at each horizon,
  and under every leave-one-stock-out deletion;
- future-range MSE improvement must be at least 3% pooled, at each horizon, and
  under every leave-one-stock-out deletion;
- for each target, daily five-session moving-block 95% upper bounds must be
  below zero for both squared and absolute error differences;
- candidate MSE and MAE must be lower in every represented calendar quarter;
- candidate/outcome correlation must exceed baseline at every horizon.

Both targets must pass every gate. A pass is prospective predictive-information
evidence only. It is not an economic-edge, tradability, or strategy claim.

## Frozen support decision

At contract time no eligible post-freeze prediction exists and no post-freeze
outcome has been opened. The required action is therefore to accumulate the
sealed ledger without inspecting results until its deterministic support rule
closes the cohort.
