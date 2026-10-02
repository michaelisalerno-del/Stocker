# Amendment 2 to the entry-veto and exit protocol (2026-10-02, before the data it is judged on)

Frozen 2026-10-02 at the user's request, before that day's first clock; its sha256 is in
`L2-VETO-EXIT-PROTOCOL.sha256`. It governs together with `L2-VETO-EXIT-PROTOCOL.md` and Amendment 1, which are
unchanged. Nothing below changes the runtime.

## Where the idea came from
On 2026-10-01 the candidate options' quotes blew out to at least twice their recent spread at 17 moments. At 16 of
them futures volume over the minute was at least 1.5x the day's median minute, and the spread came back within 1.5x
its normal after a median 28 seconds (75th percentile 105 s; 2 not within 10 minutes). Two fills that day crossed
such a quote, both at the 10:00 New York ISM release: the 09:00 NQ put was sold into 45.5 / 69 (20 seconds later
65.5 / 68.5) and the 10:00 GC put was bought from 1.1 / 2.5 (ten seconds earlier 1.4 / 1.7; 20 seconds later
1.6 / 2.2). Neither the futures spread nor volume predicted direction. That day suggested the rule, so it is not
used to judge it.

## Rule (W1: wait out a blown-out option quote; no free parameter)
- Quote at t: the option's last two-sided quote (bid and ask both present, DelayedByMinutes 0) recorded at or before
  t. Spread s = ask - bid. Tick: the option's tick at its ask (`contracts.tick_at`, side +1, which follows NQ's
  price-tiered scheme).
- Normal N(t): the median of the spread at each whole second in (t - 300 s, t], each second taking the quote at or
  before it. Seconds before the option's first quote in the capture are left out; with fewer than 60 seconds left
  there is no normal.
- A quote is blown out at a planned time P when s >= 2 x N(P) and s >= N(P) + 2 ticks. After that, a quote is back
  when s <= 1.5 x N(P).
- Entry: if the quote the baseline buys from (the first quote within the 20-second entry deadline after clock C) is
  blown out against N(C), buy instead at the ask plus one tick of the first quote recorded after it that is back, no
  later than C + 120 seconds; if none is back by then, at the last quote at or before C + 120 seconds. The exit time
  stays E = C + 60 minutes.
- Exit: if the quote the baseline sells into (the last quote at or before E) is blown out against N(E), sell instead
  at the bid less one tick of the first quote recorded after E that is back, no later than L = the earlier of
  E + 120 seconds and the option's cutoff (the earliest of its last trade, its expiry instant and the end of its
  trading session, as in `contracts.verified_cutoff`); if none is back by then, at the last quote at or before L.
- Otherwise nothing changes. A quote without a bid or an ask is never treated as blown out; the existing data rule
  handles it.

## Data
- Opportunities, quotes and P&L exactly as in the protocol's "Data": every recorded clock with a verified candidate
  option, traded or not; buy at the ask plus one tick, sell at the bid less one tick; fees per side and the
  conversion mark-up; as a share of the money paid in. For traded clocks the baseline uses the actual fills; a side
  W1 changes always uses the capture, and a side it leaves alone keeps the baseline price.
- Clocks 2026-10-02 to 2026-11-27, all weekdays. W1 has nothing to tune, so nothing is chosen on DISCOVERY.
- Affected opportunities: those where W1 changes the entry, the exit or both. One whose normal or wait cannot be
  worked out from the capture is excluded and counted, by side. The capture keeps an untraded clock's option only to
  about E, and a traded one 5 minutes past its close, so the exit side can mostly be judged on traded clocks only.
- Placebo: on unaffected opportunities, buy at the first quote at or after C + 30 seconds and sell at the last quote
  at or before E + 30 seconds (30 seconds is about the median blow-out on 2026-10-01), with the same exclusion rule.
  It separates waiting out a blow-out from simply trading later.

## Verdict
W1 PASSES only if, on affected opportunities: (a) the mean improvement over the baseline (per opportunity, after
costs) is positive with t > 2, t across days of the daily mean improvement (days with none left out); (b) there are
at least 40 of them on at least 10 distinct days; (c) the mean improvement is larger than the placebo's; and (d) the
mean improvement is positive both on DISCOVERY clocks (2026-10-02 to 2026-10-30) and on HOLDOUT clocks (2026-11-02
to 2026-11-27). Reported regardless: the entry and exit sides separately; at a listed release (the calendar as
hashed in `RELEASE-EXIT-PROTOCOL.sha256`, within 120 seconds of the planned time) and elsewhere; by market; the
share improved; the median; the improvement in GBP per contract; the wait lengths; futures volume over the minute
before each blow-out against the day's median minute; the placebo's own mean and t; and the exclusions. Analysis
runs once, after 2026-11-27, alongside the other protocols.

## Count
This adds one family, so the protocol now tests fourteen: about one false pass at t > 2 is still expected by chance,
and a pass at t < 3 is reported as weak. Combinations, including with X7, X8, N1 and N2, stay exploratory.

## Unchanged
Everything else in the protocol and Amendment 1, including the other thirteen families, their periods and their
verdict.
