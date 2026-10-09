# Waiting out a thinning or lopsided futures book: two five-second deferrals — frozen protocol (2026-10-09, on the user's "Yes", before the data it is judged on)

Agreed between Claude and Codex on 2026-10-09 (discussion and signed plan: `~/Codex/2026-10-09-claude-codex-entries-exits/`,
`L2_PLAN.md`). Level 2 has not predicted direction in any test; its use here is execution only. Both rules delay an order
by five seconds when the futures book suggests the app's option quote is about to widen. They are replayed on the
recorded quotes; nothing changes the runtime and no order is sent. The goal is to lose less: neither rule can make a
method whose options move about zero mid to mid profitable on its own.
The hash of this file is in `L2-DEFERRAL-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such,
made before the results are looked at.

## Inputs (as recorded by the app; nothing new is recorded)
- The future's 10-level book, about once a second; the future's traded volume per second (differences of the
  cumulative volume, as `incidence.py`); the app's option top of book (bid, ask), updated about every 2 s.
- Option spread = ask − bid of the last option quote at or before a time, at most 30 s old.
- Session: 09:00–16:00 New York clocks vs the rest.

## Rule 1 — hedge-side depth withdrawal (entry and exit)
- Hedge side of an order: the futures ask for buying a call or selling a put; the futures bid for buying a put or
  selling a call.
- Depth = total size on the hedge side at prices within 3 futures ticks of that side's best price (best level included).
- Trigger at the action time a (entry t0 = clock + 3 s; exit e0 = clock + 60 min + 3 s): depth is below 35% of its
  median over the 300 s before, on both of the last two book snapshots at or before a, and the option spread at a is
  below 1.5 x its median over the 300 s before (sampled at each option quote).
- Action: defer once; trade at the visible quote at a + 5 s (ask on entry, bid on exit; no bid = 0). The app's exit
  deadlines are kept: an exit is never deferred past the option's expiry.

## Rule 2 — signed-volume blow-out forecast (entry only)
- Signing: each second's traded volume is signed by that second's change in the futures mid (up +, down −; no change
  takes the sign of the last non-zero change). Imbalance = |sum of signed volume| / total volume over the 20 s before t0.
- Trigger at t0: imbalance > 0.70, the 20 s volume is above the 95th percentile of all 20 s volumes in the 30 minutes
  before, and the option spread at t0 is below 1.5 x its median over the 300 s before.
- Action: defer the entry once; buy at the visible ask at t0 + 5 s. The exit is the baseline's.
- Volume-only control: the same trigger without the imbalance condition. Rule 2 is credited only with what it adds
  over this control.
- If the recorded futures volume cannot be signed on at least 80% of judged clocks (missing mids or volume), rule 2 is
  reported as "ineligible", not as a pass or fail.

## Baselines and outcomes
- Baseline A: `EXECUTION-PROTOCOL.md`'s baseline (ask at t0, bid at e0, visible quotes, at most 30 s old).
- Baseline B: `EXECUTION-SET-2-PROTOCOL.md` scheme C (futures-burst hold-off), on the same clocks.
- Saving per clock = (baseline entry − rule entry + rule exit − baseline exit) / baseline entry, in % and in option
  ticks; GBP after fees as the app's ledger computes them. Movement during the 5 s delay counts against the rule.
- Forecast check (each rule): at every 5 s sample of the recorded segments where the rule's trigger is evaluated, the
  outcome is whether the option spread reaches at least 2 x its median over the 300 s before within the next 10 s.
  Compared with non-triggered samples in the same market, session and starting-spread ratio band (< 1.0, 1.0–1.25,
  1.25–1.5 x the median).

## Verdict (each rule separately, once, after 2026-11-27)
- Judged clocks: from the first clock after the freezing commit to 2026-11-27 16:00 New York, every market and session;
  halves split at 2026-10-31. Clocks before that commit are never judged.
- A rule PASSES only if all of these hold:
  1. Forecast: triggered samples reach 2 x spread more often than matched non-triggered ones, t > 2.5 across days
     (rule 2: also more often than the volume-only control's triggered samples).
  2. Saving against baseline A AND against baseline B: mean positive, t > 2.5 across days (the daily mean over affected
     clocks), at least 40 affected clocks on at least 10 days, positive in both halves.
  The bar t > 2.5 covers the two rules being tested together.
- Reported regardless: entry and exit separately (rule 1); by market and session; trigger counts; the share of affected
  clocks where the rule paid more than the baseline.
- Paper caveat, as in `EXECUTION-PROTOCOL.md`: these fills are estimates from quotes. A pass earns a small randomised
  real test, not a change to the app.

## Script
The scoring script is written after this file and recorded with its sha256 in a dated amendment before any judged
result is computed (as `PRESSURE-PROGRESS-AMENDMENT-1.md`). Until then no saving or forecast figure on clocks after the
freezing commit is computed.

## What was seen before this was written
Nothing for these rules: no trigger, forecast or saving was computed on any data. The thresholds (35%, two snapshots,
3 ticks, 0.70, 95th percentile, 1.5 x, 5 s, 10 s) were set in the discussion from earlier findings (1–2 Oct: option
spread blow-outs come almost only with futures volume surges and clear in about 28 s; level-1 queue sees about 1 s ahead).
