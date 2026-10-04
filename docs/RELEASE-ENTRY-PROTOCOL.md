# Enter one minute before a scheduled release — frozen protocol (2026-10-01, before the data it is judged on)

The user chose to fix this candidate now and judge it later on the LIVE paper recordings. The app keeps trading
the frozen method unchanged; nothing below changes the runtime. The hash of this file is in
`RELEASE-ENTRY-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results
are looked at.

## Where the idea came from
On 2026-10-01 the 10:00 New York GC put was bought at 10:00:00, as the ISM release came out: its quote was
1.1 / 2.5 (spread 78% of the mid) and the trade lost 90%. One trade on one day suggested the rule, so that day is not
used to judge it.

## Rule (N1; no free parameter)
For a clock C at which the calendar lists a release for that market at a time R with C - 60 seconds <= R <= C + 120
seconds, the whole frozen decision runs one minute early, at C' = R - 60 seconds: the gates, rv15 and the strike
are computed with the frozen code from the completed bars before C' (the hourly references and the clock's veto
stay those of C), and the option is bought at its ask plus one tick (the first quote at or after C', within 20
seconds). The exit is unchanged: E = C + 60 minutes, bid less one tick. The decision at C' never uses a bar that
ends after C'; using C's choice at C''s price would let the entry see the minute it is meant to avoid.

## Data
- Opportunities, quotes and P&L exactly as in `L2-VETO-EXIT-PROTOCOL.md` ("Data"): buy at the ask plus one tick,
  sell at the bid less one tick (the last quote at or before E); fees per side and the conversion mark-up; as a share
  of the money paid in. Bars from the app's Saxo bar cache; option quotes from the capture segments.
- Clocks 2026-10-02 to 2026-11-27. With nothing to tune there is no discovery/holdout split.
- Calendar: `docs/release-calendar-2026-10-11.yaml` as hashed in `RELEASE-EXIT-PROTOCOL.sha256`, with that
  protocol's correction rule.
- Affected opportunities: release clocks where both the normal entry at C and the early entry at C' yield a verified
  option with quotes at entry and exit. Clocks eligible at only one of C and C', or whose C' option was not
  captured, are excluded and counted, with how often each happened.
- Placebo: the same clock hours on weekdays with no listed release near C, entered at C - 60 seconds the same way.
  It separates the release from simply entering a minute early.

## Verdict
N1 PASSES only if, on affected opportunities: (a) the mean improvement of the early entry over the normal entry
(per opportunity, after costs, both held to E) is positive with t > 2, t across days of the daily mean improvement;
(b) there are at least 40 of them on at least 10 distinct days; and (c) the mean improvement is larger than the
placebo's. Reported regardless: by market and by release, the share improved, the median, the option spread (as a
share of the mid) at C' and at C, and the placebo's own mean and t. Analysis runs once, after 2026-11-27, alongside
the other protocols; it is a separate single test.
