# Which way a blown-out option quote leans — frozen protocol (2026-10-02, before the data it is judged on)

The user asked whether option spreads can be used "like a compass" and chose to test this one form of it. It is a
prediction test only: the app keeps trading the frozen method unchanged and nothing below changes the runtime. The
hash of this file is in `QUOTE-LEAN-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such,
made before the results are looked at.

## Where the idea came from
On 2026-10-01 the width of a blown-out option quote said nothing about direction (the option's mid rose after 8
of 17 blow-outs and fell after 8), and the futures order-book imbalance was inconsistent. Untested: which side of the
quote moves when it blows out, the bid dropping away or the ask jumping up. A count on 2026-10-01 with the
definitions below found 42 blow-out moments, 37 of them leaning one way; no lean was compared with any outcome.
That day is not used to judge it.

## Definitions (no free parameter)
- Quotes: every recorded change of a candidate option's two-sided quote (bid and ask both present,
  DelayedByMinutes 0) in the capture segments. Spread s = ask - bid; tick and normal N(t) exactly as in W1
  (`L2-VETO-EXIT-PROTOCOL-AMENDMENT-2.md`): the tick at the ask (`contracts.tick_at`, side +1); the median spread at
  each whole second in (t - 300 s, t], at least 60 seconds, else no normal.
- Onset: a quote with s >= 2 x N(t) and s >= N(t) + 2 ticks, with no such quote for that option in the 120 seconds
  before.
- Reference: the last quote before the onset, recorded at most 30 seconds before it, with s <= 1.5 x N(t) at the
  onset. With none, or with a reference bid below two ticks (the tick at that bid, `contracts.tick_at`, side +1),
  the onset is excluded and counted.
- Lean: the option's mid at the onset minus its mid at the reference, i.e. half of (ask rise - bid fall). A call
  leaning up or a put leaning down points the future up; a call leaning down or a put leaning up points it down;
  zero points nowhere.
- Moment: onsets in the same market within 90 seconds of the first form one moment, timed at the first onset. Its
  direction is the sign of the sum of its onsets' directions (+1 up, -1 down, 0 none). None is excluded and counted.
- Outcome: the future's mid (its last real-time quote) 5 minutes after the moment minus its mid at the moment, in
  futures ticks. Flat or missing is excluded and counted.
- Score: +1 when the moment's direction matches the outcome's sign, -1 when it does not.
- Comparison: the future's own last minute, i.e. the sign of its mid change over the 60 seconds before the moment,
  scored the same way where it is not zero.

## Data
- Every moment in the capture segments from 2026-10-02 to 2026-11-27, all weekdays (the capture runs from about 15
  minutes before each clock to at least 60 minutes after it). 2026-10-01 is not used.
- DISCOVERY / HOLDOUT halves for the consistency check: 2026-10-02 to 2026-10-30 and 2026-11-02 to 2026-11-27.

## Verdict
The lean PASSES as a compass only if: (a) the mean score is positive with t > 2, t across days of the daily mean
score; (b) at least 300 moments are scored, on at least 15 distinct days; (c) its hit rate is higher than the
comparison's on the moments both score; and (d) its hit rate is above 50% in both halves. Reported regardless: hit
rate and the mean futures move in the direction pointed (ticks) at 1, 5 and 15 minutes; by market; at a listed
release (the calendar as hashed in `RELEASE-EXIT-PROTOCOL.sha256`, within 120 seconds) and elsewhere; by lean size
in ticks (terciles); the lean net of the future's own move (option mid change minus Saxo's delta x the futures mid
change over the same span) as a second predictor; whether the mean 5-minute move in the direction pointed exceeds
the round trip of one futures contract (the market's median spread over the period plus $2 commission and the
exchange fee per side); and every exclusion. Analysis runs once, after 2026-11-27, alongside the other protocols;
it is a separate single test.

A pass shows direction only. Trading on it needs its own rule, frozen before the data it is judged on.
