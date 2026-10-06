# Option-quote lag: measured, and a lag-aware fill for futures-triggered rules (drafted 2026-10-05, frozen 2026-10-05 ~23:30 UTC on the user's "freeze", before any judged result)

Amends the paper fill of every rule whose trigger is the future's state: `L2-VETO-EXIT-PROTOCOL.md` exits X5
and X6 (sha256 in `L2-VETO-EXIT-PROTOCOL.sha256`), `EXIT-SET-2-PROTOCOL.md` exits E1 and E2 (sha256 in
`EXIT-SET-2-PROTOCOL.sha256`) and the pressure-progress entry and exit of `PRESSURE-PROGRESS-PROTOCOL.md` with
`PRESSURE-PROGRESS-AMENDMENT-1.md`. Nothing else in those protocols changes; nothing changes the runtime. No judged result
of any of them has been computed (they are judged once after 2026-11-27). The hash of this file is in
`QUOTE-LAG-AMENDMENT-20261005.sha256`.

## What was measured (hourly ledger `lag.py`, captures 1–5 October, every market)
`EXIT-SET-2-PROTOCOL.md` noted that "Saxo's option prices arrive about 2 s after the future's". That is now measured on the
one-second local-receipt grid of the recordings (Saxo sends nothing while a quote stands, so quotes are carried forward):

- Correlation of the option mid's one-second change with delta × the futures mid's change `lag` seconds earlier, pooled
  over the app's own options and the chain window's strikes: near zero at lag 0, peaks at **+2 s** in every market (ES
  0.13–0.18, NQ 0.11–0.19, GC 0.10–0.19, CL 0.05–0.15), still half its peak at +3 s, gone by +5 s. Nothing at negative
  lags: the option never leads the future in this feed.
- After an isolated futures move (at least 2 ticks in ES and NQ, 5 in CL and GC, following 10 s without one), the chain's
  strikes first move the implied way after a median **1–2 s** (quartiles 0–3 s); 50–85% respond within 15 s. The app's
  own 0.10-delta options respond less often (10–47%) because a move of that size is well under one of their ticks.
- Saxo's own `LastUpdated` stamp is sent only with trades. Futures trades reach the recorder a median 0.40–0.56 s after
  the stamp, option trades 0.85–1.28 s: about half a second of the lag is Saxo's pipeline, the rest is market makers
  re-quoting. The two subscriptions run on their own one-second refresh phases, so a single quote's lag is 1–3 s.

## Why the frozen fill is flattered
A rule that fires on the future's state and fills on the option's last quote **at or before** the firing time fills on a
quote from before the move it reacted to: an exit after an adverse move sells into a bid the market maker has already
lowered, an entry after a favourable move buys an ask already raised. Measured on the same isolated moves, as the
difference between the quote the frozen convention fills on and the first quote received after it, signed so that a
positive number flatters the rule (share of the option mid; mean, with the median in brackets):

| Market | Fill at the firing time t (X5, X6, pressure-progress) | Fill at t + 3 s (E1, E2) |
|---|---|---|
| CL, app's options | +5.3% adverse, +9.1% favourable (+5.0%, +7.1%) | +0.5%, +5.0% (0, 0); the stale quote still predates the move 31–32% of the time |
| ES, app's options | +2.0%, +2.1% (+1.7%, +1.7%) | +0.5%, +0.3% (0, 0); predates 11–13% |
| GC, app's options | +3.2%, +3.6% (+2.4%, +2.6%) | +0.2%, +0.7% (0, 0); predates 25% |
| NQ, app's options | +0.8%, +0.6% (+0.6%, +0.5%) | +0.2%, +0.1% (0, 0); predates 39–41% |

At the firing time the stale quote predates the move in 100% of cases by construction and differs from the next quote in
70–84% of them. Three seconds removes most of the bias in ES and GC; in CL, whose cheap options stand unchanged for a
median 6–8 s and tick in 10–20% steps, a 3-second allowance still leaves a flattering mean of 5% of mid on favourable
moves. The frozen verdicts are judged on mean improvements of a few per cent of premium at t > 2, so a systematic 1–9%
per futures-triggered fill can decide them.

Rules whose trigger is the clock, the calendar or the option's own quote (every V entry veto, X1–X4, the lean, trail,
IV, minute-early, release, pullback and delta-target rules) fill on quotes that do not depend on a futures move and are
not affected.

## Amendment
- **Lag-aware fill.** For X5, X6, E1, E2 and the pressure-progress entry and exit, the judged fill is the option's first
  two-sided quote (bid and ask both present, DelayedByMinutes 0) **received at or after t + 3 s**, where t is the firing
  time (the check, or the qualifying sample), and received within 30 s of it. If no such quote arrives within 30 s the
  rule does not fill at that check and goes on checking (as Exit Set 2 already does for a missing bid); for X5 and X6
  the exit falls to the next check at which the condition still holds. Sell at that quote's bid less one tick, buy at
  its ask plus one tick, exactly as each protocol's cost model says.
- **Both reported.** Every affected rule is also computed with its frozen convention ("the last quote at or before t",
  or "t + 3 s" for E1 and E2) and both results are shown side by side, with the mean difference between the two fills
  and the age of the frozen fill's quote at the firing time. The pass/fail verdict uses the lag-aware fill.
- **Where it is computed.** The pressure-progress entry and exit: `pressure_progress.py` in the hourly ledger now writes
  `*_lag` columns (entry and exit, both specifications) beside the frozen ones, which are unchanged; its sha256 at
  freezing is recorded in `LEDGER-NEXT-CONTRACT-AMENDMENT-20261005.md` (the same file carries that correction). X5, X6,
  E1 and E2 have no script yet; the analysis that judges them after 2026-11-27 implements this fill.
- **The pre-registration is kept.** No other threshold, grid, period, population or verdict bar changes. This amendment
  is made before any judged result exists (no pressure-progress return has been looked at); `lag.py` (sha256 cacb860f8e58d7de3427999c545ae5cb0aaad80d85897d467d2c83d155a9b8f9 at
  freezing) and its outputs (`lag.csv`, `lag_events.csv`, `lag_stale.csv`,
  `lag_report.txt`) are descriptive and never a judged column.

## Follow-ups this measurement points to (not part of this amendment)
- Saxo returns no depth for these options (`saxo-option-capability-probe-20261006.json`: MarketDepth absent on the GC,
  ES, CL and NQ candidates), so the note's option-depth ideas ("lift the lone offer" on small size) cannot be built; option
  roots have `CanParticipateInMultiLegOrder: false`, so a vertical cannot be one order; Limit orders allow DayOrder,
  GoodTillCancel and GoodTillDate, which is all a client-side stepped limit needs.
- The futures-burst hold-off and the fair-value-gated cross from the Level 2 research note can only be tested with the
  lag-aware fill; their protocols, if written, must cite this one.
- A true exchange-to-Saxo delay needs an exchange-timestamped reference for one recorded day (Databento GLBX.MDP3
  carries matching-engine timestamps; its pay-as-you-go history comes with USD 125 of credit).
