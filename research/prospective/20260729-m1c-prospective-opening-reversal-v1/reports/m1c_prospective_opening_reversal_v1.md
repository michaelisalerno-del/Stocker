# M1C Prospective Opening Reversal V1

**RESEARCH ONLY — SHADOW RECORDING ONLY — NO ORDERS**

Activation: `2026-07-29T06:39:44.466604+00:00`  
Rule hash: `7a2a40170fa1dffb148cb51144c1c9bfcb029c5d797ea9680164cfd9fbce1ea7`  
Configuration hash: `7d95baa2f7d56c8b82b639f7ee2bae459a879aa0b61200bc2786f0e3bf695c8e`

## Event accounting

The prior assessment labels mixed two populations. Its total of 43 counted
severe transition events with at least one eligible stock episode, while 28
negative and 30 positive counted all severe VTI sessions. On the common
eligible-event population, assessment is 23 negative plus 20 positive = 43.
Stress is 5 negative plus 8 positive = 13. Development is 18 negative plus
8 positive = 26. This is `event_label_ambiguity_corrected`, not an event-sign
duplication or aggregation bug. The prior scientific interpretation remains
`blocked_insufficient_support`.

## API and capacity operation

The deployment template has a 100-line local budget, but exact broker capacity
is treated as unknown until runtime. Twelve lines are reserved and unavailable
to optional research. Persistent priority protects VTI and the frozen 20-stock
five-minute universe. Only one deterministically promoted underlying and its
one 1DTE nearest-ATM call/put pair may be mandatory. Optional feeds fail closed
and are dropped in the frozen order recorded in the priority manifest. There
are no post-activation runtime snapshots or degradation events yet.

## Engineering transfer

The first 20 valid sessions are `engineering_transfer`. No future stock return,
M1C outcome, option P&L, or directional score from those sessions may influence
the rule. IBKR/EODHD comparison is predictor-only and does not require exact
OHLC equality. Zero valid sessions have been recorded.

There is one explicit timing blocker to resolve prospectively: the sixth bar
completes at 10:00, the frozen entry is also 10:00, and the receipt contract is
strictly before entry. The recorder therefore fails closed instead of inventing
an earlier timestamp.

## Prospective development

The frozen action is CALL after a negative severe VTI opening transition, PUT
after a positive severe transition, and ABSTAIN otherwise. No development
outcome has been opened. Support, baselines, cluster intervals, fixed-seed
nulls, placebo, and concentration results are pending.

## Prospective confirmation

Confirmation has not started. Its start and decision receipts remain
deliberately unissued.

## Option economics

No option-economics claim is made. Primary evidence requires actual first-valid
ask at/after entry and first-valid bid at/after +15 minutes for the frozen 1DTE
pair. Midpoints remain diagnostics. Optional expiries remain separate and
capacity-gated.

## Structural, movement, and execution evidence

No post-activation movement evidence exists. The activation opened no protected
pre-activation 2026 outcome. Order routing is disabled, no order method is
available in the experiment module, and no order was placed.
