# Enter one minute after a scheduled release — frozen protocol (2026-10-01, before the data it is judged on)

The user chose to fix this candidate now and judge it later on the LIVE paper recordings. The app keeps trading
the frozen method unchanged; nothing below changes the runtime. The hash of this file is in
`RELEASE-LATE-ENTRY-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the
results are looked at.

## Where the idea came from
The same 2026-10-01 GC trade as `RELEASE-ENTRY-PROTOCOL.md` (N1): bought at 10:00:00 into the ISM release with a
spread of 78% of the mid; it lost 90%. N1 decides and buys a minute before the release; N2 keeps the normal decision
and buys a minute after it, once quotes have come back. That day is not used to judge either.

## Rule (N2; no free parameter)
For a clock C at which the calendar lists a release for that market at a time R with C - 60 seconds <= R <= C + 120
seconds, the frozen decision at C is unchanged (the same gates and the same option), but the option is bought at
R + 60 seconds at its ask plus one tick (the first quote at or after R + 60 seconds, within 20 seconds). The exit is
unchanged: E = C + 60 minutes, bid less one tick. Nothing after C changes which trades are taken or which option.

## Data
- Opportunities, quotes and P&L exactly as in `L2-VETO-EXIT-PROTOCOL.md` ("Data"): buy at the ask plus one tick,
  sell at the bid less one tick (the last quote at or before E); fees per side and the conversion mark-up; as a share
  of the money paid in. Option quotes from the capture segments; the option is the one recorded at C.
- Clocks 2026-10-02 to 2026-11-27. With nothing to tune there is no discovery/holdout split.
- Calendar: `docs/release-calendar-2026-10-11.yaml` as hashed in `RELEASE-EXIT-PROTOCOL.sha256`, with that
  protocol's correction rule.
- Affected opportunities: release clocks whose decision at C selected a verified option with quotes at C, at
  R + 60 seconds and at E. Others are excluded and counted.
- Placebo: the same clock hours on weekdays with no listed release near C, bought at C + 60 seconds the same way.
  It separates the release from simply buying a minute late.

## Verdict
N2 PASSES only if, on affected opportunities: (a) the mean improvement of the late entry over the normal entry
(per opportunity, after costs, both held to E) is positive with t > 2, t across days of the daily mean improvement;
(b) there are at least 40 of them on at least 10 distinct days; and (c) the mean improvement is larger than the
placebo's. Reported regardless: by market and by release, the share improved, the median, the option spread (as a
share of the mid) at C and at R + 60 seconds, the placebo's own mean and t, and N2 against N1 on the opportunities
both cover. Analysis runs once, after 2026-11-27, alongside the other protocols; it is a separate single test.
