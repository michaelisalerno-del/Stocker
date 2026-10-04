# Paying less of the spread — two frozen execution tests (2026-10-04, before the data they are judged on)

Spreads were the whole loss on the first 24 paper trades (+£1,497 at mid prices, −£1,459 crossing spreads). These two
rules for buying and selling the same option more cheaply are fixed now and judged once on clocks the app records from
5 October; nothing here changes the runtime and no order is sent. The hash of this file is in
`EXECUTION-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at. The figures from 1–2 October seen while building the measurement (about 20 clocks) are not part of it.

## Data
The hourly ledger (`~/Documents/Codex/2026-10-03-hourly-ledger`, `build.py` sha256
4891e47066a94923378c6c84dcfbc6d06cfe5b3e5b8bf027c923f63fc5b01343 at freezing, function `execution`): the app's option at
each clock (traded or skipped), with its recorded quotes, sizes and trade prints (LastTraded, Volume) from the capture
segments. A quote counts only if it is at most 30 s old and not after a recording gap.

## Baseline
Buy at the ask at t0 = the clock + 3 s (when the app decides); sell at the bid at e0 = the hour's end + 3 s (no bid = 0).
An option that settles within a minute of e0 has no exit to improve.

## Rules
- Q, queue-timed: at entry, if the option's own queue leans against the buyer (ask size > bid size), wait up to 30 s for
  it to stop leaning, then buy at the ask (at 30 s regardless). At exit, the mirror: if bid size > ask size, wait up to 30 s
  for that to stop, then sell at the bid.
- L, limit at the mid: at entry, a buy limit at the mid rounded down to the option's tick (not below the bid). It counts
  as filled only if the ask reaches it or a trade prints strictly below it within 60 s (a trade at the limit does not
  count: the order would wait behind others there); otherwise buy at the ask at 60 s. At exit, the mirror (mid rounded
  up, not above the ask; filled if the bid reaches it or a trade prints strictly above it; otherwise the bid at 60 s).

## Outcome and verdict
- Per clock: saving = (baseline ask − rule's entry price + rule's exit price − baseline bid) / baseline ask
  (`q_saving_pct`, `l_saving_pct`); the day's figure is the mean over its scored clocks.
- Judged once after 2026-11-27 on clocks 2026-10-05 to 2026-11-27 (halves 5–30 October, 2–27 November). A rule PASSES
  only if the mean saving is positive with t > 2 across days, with at least 100 scored clocks on at least 15 days, and
  positive in both halves.
- Reported regardless: entry and exit separately, fill rates and waits, by market and by option price band, and the
  share of clocks where the rule paid more than the baseline.
- Paper caveat: the fill rule is deliberately strict because a real order joins the back of the queue at its price,
  but whether Saxo would fill a resting order the same way is only answered by real orders. A pass earns a small real
  test of fills, not a change to the app.
