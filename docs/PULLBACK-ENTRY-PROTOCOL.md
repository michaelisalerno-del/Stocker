# Buy on the first dip, or not at all — frozen entry test (2026-10-05, before the data it is judged on)

The user asked whether waiting improves an entry, counting every opportunity a waiting rule misses. Of four ways to wait
compared on 1–5 October, one stood out and is frozen here: after the clock, buy the same option only if its ask dips 10%
below the clock's ask within 15 minutes; otherwise do not buy. It is a paper test on the hourly ledger: the app keeps
buying at the clock, nothing changes the runtime and no order is sent. The hash of this file is in
`PULLBACK-ENTRY-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at.

## Where the idea came from, and what was seen
On the 48 clocks of 1–5 October with the app's option recorded through the hour (same option, same exit, a policy that
does not buy earns 0): buying at the clock averaged −15% per pound of premium; a fixed delay of 2 or 5 minutes −21% (worse);
buying once the future had moved a quarter of a normal hour the option's way −10%, but the trades it bought did worse than
buying at once (−17% against −9%); the 10% dip −4%. The dip came on 27 of the 48 clocks, where it improved the result from
−18% to −7%; the 21 clocks without a dip would have averaged −12% bought at once. Paired against buying at once it was
better on 41 clocks and worse on 6, by +12 points per clock (+8 without the three best clocks), positive in every market
(CL +17, ES +6, GC +4, NQ +15) and on every day, and similar with a 5% dip (+10); a 20% dip helped little (+3). It kept 2
of the 6 options that doubled, against 3 bought at once. The 10% and 15-minute values were set before the comparison was
run, but the rule was chosen as the best of four. The look also found that neighbouring strikes always moved with the
app's option over the 3 minutes before a clock (no case of the selected option's ask jumping alone), so no
neighbouring-strike test is frozen. 1–5 October is not used to judge it.

## Definition (parameters fixed here)
- Option and quotes: the app's selected option at each clock (`opt_uic`) and its recorded two-sided quotes in the capture
  segments (not gaps), as the hourly ledger parses them.
- Baseline: buy at the ask of the last quote at or before clock + 3 s (at most 30 s old); sell at the bid of the last quote
  at or before clock + 60 min + 3 s (at most 30 s old; no bid = 0). `immediate_ret` = exit bid / that ask − 1.
- Pullback policy: the first quote after clock + 3 s and within the next 15 minutes whose ask is at most 0.90 x the
  baseline's ask; buy at that ask, sell at the same exit; `pullback_ret` = exit bid / that ask − 1. Without such a quote it
  does not buy and `pullback_ret` = 0.
- Paired difference per clock: `pullback_delta` = `pullback_ret` − `immediate_ret`.
- Population: every clock with the app's selected option, traded or skipped, that has both quotes. Column: `pullback.csv`,
  written by `entry_pullback.py` in the hourly ledger (sha256
  2c3528352b4c52924c400f04669caa124700c218f90146c8eff6e9016f449935 at freezing) after `build.py`.

## Data
Judged clocks: 2026-10-06 to 2026-11-27. Halves: 6–30 October and 2–27 November. 1–5 October is the look and is not
judged.

## Verdict
The pullback entry PASSES only if, judged once after 2026-11-27: (a) the mean `pullback_delta` is positive with t > 2.5,
t across days of the daily mean (the stricter bar: the rule was picked as the best of four); (b) at least 200 clocks on at
least 15 distinct days; (c) the mean is positive in both halves.

Reported regardless: how often the dip came and its minute; on the clocks it bought, the baseline's and the pullback's
returns; on the clocks it skipped, what buying at once would have made (the cost of the missed trades); the options that
at least doubled from the clock's ask and how many the pullback kept; the share of all clocks it made the result better and
worse; by market; the original 09:00–16:00 clocks against the others; the frozen-scope clocks (CL, GC, NQ, 09:00–16:00) on
their own; the overlap with the frozen break-even and momentum-against vetoes; with the app's own one-tick-worse paper fill
applied to both entries; and, for information only, 5% and 20% dips and 5- and 30-minute waits.

A pass earns the pullback a place in the frozen method only by a dated rulebook amendment and a new rule version after
2026-11-27.
