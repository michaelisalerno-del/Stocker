# Paying less of the spread, set 2: four execution schemes on the recorded quotes — frozen protocol (2026-10-06 ~07:05 UTC, on the user's "Freeze", before the data it is judged on)

The Level 2 research note (2026-10-05) ranks execution first: the paper loss is the option's spread (mid to mid the
options barely moved), and its proposed schemes act on that cost directly. These four are replayed on the recorded
quotes of the app's option at every clock and judged once after 2026-11-27. Nothing changes the runtime and no order is
sent. They are one family, each judged at t > 2.5. They differ from `EXECUTION-PROTOCOL.md` (Q, queue-timed; L, a static
limit at the mid for 60 s): stepping, a fair value from the future, a futures-burst hold-off and a ten-minute maker exit.
The hash of this file is in `EXECUTION-SET-2-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are looked at.

## What the schemes can and cannot rest on (measured 2026-10-05/06)
- Saxo shows no option depth (`saxo-option-capability-probe-20261006.json`), so every scheme sees only the option's top
  of book. Limit orders may be Day, GTC or GTD; stepping and cancelling are done by the client (placement limit one per
  second per session: one repricing every 5 s for a few working orders fits).
- The option's quote trails the future by about 2 s (`QUOTE-LAG-AMENDMENT-20261005.md`), so every fill below that is
  triggered by the future or by the option's own quote is taken from the first quote received 3 s after the action.
- Spreads at the clocks are 2–10 ticks (median 5); one-tick spreads are almost absent, so there is room inside them.

## Data
`execution2.csv`, written by `execution2.py` in the hourly ledger after `build.py` (sha256
69addc28dd9f7c3d97f367b2a4509f3c371f2965979eb9a5d8deae35bdd813f2); it reads the futures' per-second volume from the cache
of `incidence.py`. At freezing: `execution2.py` f0e42de4bf3e54966c81d5145bc9d2449a53b1e00cfb3d29cfdff683a0cf1781,
`incidence.py` 8014007c1f699680fda2d5d2edcb8e7b1eeb78de3e6c5a194c207a8f9d8cc198. The module docstring of `execution2.py`
is the exact specification; this file states it in words.

## Baseline and fills
- Baseline (as `EXECUTION-PROTOCOL.md`): buy at the ask of the last quote at or before t0 = clock + 3 s, sell at the bid
  of the last quote at or before e0 = clock + 60 min + 3 s (both at most 30 s old; no bid = 0). An option settling within
  a minute of e0 is worth its intrinsic value and has no exit to improve.
- Per clock: saving = (base ask − entry + exit − base bid) / base ask; a side a scheme does not act on keeps the
  baseline's price.
- Visible quote: the last at or before a time, for clock- and deadline-timed actions (as the baseline). Lag-aware quote:
  the first received at or after the action + 3 s, for actions triggered by the future or the option's quote.
- A resting order fills only when a quote received at u shows the other side at or through the price the order had at
  u − 3 s, or a new trade prints strictly through it, with u from its placement + 3 s to its deadline. Otherwise it is
  cancelled and the deadline crosses at the visible quote. A trade at the order's price does not count.

## Schemes (one configuration each; nothing is tuned)
- **A — stepped limit, entry and exit.** At t0, S = the spread in ticks (CME schedules as Saxo states them). S <= 1:
  cross. Otherwise rest at the mid rounded toward the far side (ask − ⌊S/2⌋ ticks), step one tick toward the far side
  every 5 s without reaching it, cross at t0 + 20 s (the app's entry deadline). The exit mirrors this from e0.
- **B — fair-value-gated cross, entry and exit.** FV = Black-76 at the future's weighted mid 2 s before the option
  quote's receipt, with σ = the median of the option's own mid-implied volatility sampled every 30 s over the 30 minutes
  before (at least 5 samples; time to expiry floored at 5 minutes). Each second from t0 to t0 + 20 s, cross when the ask
  is at most 0.5 tick above FV (lag-aware fill); at t0 + 20 s cross at the visible ask. The exit mirrors this from e0.
  Placebo: cross at a random second of the same windows (seeded by the clock id), lag-aware fill.
- **C — futures-burst hold-off, entry and exit.** A burst at t0 when the future's traded volume or absolute mid change
  over the 3 s before is at or above its 95th percentile for that market and New York hour on earlier days (and at least
  1 contract or 1 tick). Then wait for 3 consecutive quiet seconds and an option re-quote since the burst (at most 30 s)
  and fill lag-aware. Clocks without a burst keep the baseline. Placebo: unaffected clocks acting at t0 + 10 s and
  e0 + 10 s, lag-aware.
- **D — maker exit with a hold-to-expiry floor.** From clock + 55 min: if the bid is at most one tick (or missing) and
  the option expires later that day, hold it to expiry (intrinsic value). Otherwise rest a sell at max(ask − 1 tick,
  bid + 1 tick) (at the ask when S = 1), one tick lower every 60 s but never at or below the visible bid, and hit the
  visible bid at clock + 65 min. Entry is the baseline's.

## Verdict (each scheme separately, once, after 2026-11-27)
- Judged clocks: from the first clock after the freezing commit (the 04:00 New York clock of 2026-10-06) to
  2026-11-27 16:00 New York, every market and session; halves split at 2026-10-31.
- A scheme PASSES only if the mean saving is positive with t > 2.5 across days (the daily mean), on at least 100 scored
  clocks on at least 15 days (C: at least 40 affected clocks on at least 10 days, its mean over affected clocks), and is
  positive in both halves. B must also beat its placebo's mean, and C its placebo's.
- Reported regardless: entry and exit separately; passive fill rates and waits; D's modes (filled, deadline, floor) and
  the in-the-money finishes of floor holds (CL and GC options are American and exercise into the future); by market, by
  session (09:00–16:00 New York clocks and the rest), and by spread in ticks at t0 (2, 3–4, 5+); B's FV − mid at t0 (its
  bias check: about 0 ticks on the look); the share of clocks where the scheme paid more than the baseline.
- Paper caveat, as in `EXECUTION-PROTOCOL.md`: a real order changes the market it rests in and joins the back of its
  price's queue, so these fills are a floor-style estimate. A pass earns a small randomised real test (one lot, per-clock
  coin flip between crossing and the scheme), not a change to the app.

## What was seen before this was written (the look, 1–5 October, not judged)
60 clocks, baseline ask→bid mean −26.9%. A: mean saving −1.1% (median 0), passive fills 44% at entry and 44% at exit.
B: +0.35% against its placebo's +0.25%, early crosses on 22% of entries; FV − mid at t0 averaged 0.0 ticks (quartiles
−0.44 and +0.65). C: 6 affected clocks (its norms need earlier days at the same hour), mean −11.5%, median 0. D: +3.7%
(median +1.8%), filled 43 of 59, deadline 10, floor 6, no in-the-money finish. These figures chose nothing above: every
parameter was set from the research note before the look was run.
