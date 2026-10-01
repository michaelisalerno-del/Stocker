# Exit before a scheduled release — frozen protocol (2026-10-01, before the data it is judged on)

The user chose to fix this candidate now and judge it later on the LIVE paper recordings. The app keeps trading
the frozen method unchanged (hold 60 minutes); nothing below changes the runtime. The hashes of this file and of
its calendar are in `RELEASE-EXIT-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made
before the results are looked at.

## Where the idea came from
On 2026-10-01 the 09:00 New York NQ put's exit fell on the 10:00 ISM release. Five seconds before 10:00 its bid was
53.5; at 10:00:00 the quote was 45.5 bid / 69 ask while the future moved 12 points in the put's favour (about £120
lost to the quote alone). One trade on one day suggested the rule, so that day is not used to judge it.

## Rule (X7; no free parameter)
For an opportunity with clock C and planned exit E = C + 60 minutes: if the calendar lists a release for that market
at a time R with E - 120 seconds <= R <= E + 60 seconds, sell at R - 60 seconds instead of at E. Otherwise the exit
is unchanged. Entries are unchanged; this protocol does not test entering at a release.

## Data
- Opportunities, quotes and P&L exactly as in `L2-VETO-EXIT-PROTOCOL.md` ("Data"): every recorded clock with a
  verified candidate option, traded or not; buy at the ask plus one tick at the clock, sell at the bid less one tick
  (the last quote at or before the exit time); fees per side and the conversion mark-up; as a share of the money
  paid in. For traded clocks the baseline uses the actual fills; the X7 exit always uses the capture.
- Clocks 2026-10-02 to 2026-11-27, all weekdays. With nothing to tune there is no discovery/holdout split.
- Calendar: `docs/release-calendar-2026-10-11.yaml` as hashed here (also deployed as the app's event calendar).
  A release the publisher adds, confirms, moves or cancels may be corrected only by a dated commit made before the
  release time concerned; that commit says what changed and cites the publisher. Nothing is corrected afterwards.
  The calendar's two known gaps follow this rule: Conference Board on 2026-11-24 is added only once the publisher
  confirms it, and ISM Services on 2026-11-04 is checked against ISM's 2026-10-05 release.
- Affected opportunities: those whose exit the rule changes. One lacking a quote at either exit time is excluded
  and counted.
- Placebo: the same clocks on weekdays with no listed release near E, sold at E - 60 seconds instead of E. It
  separates the release from simply leaving a minute early.

## Verdict
X7 PASSES only if, on affected opportunities: (a) the mean improvement over the 60-minute exit (per opportunity,
after costs) is positive with t > 2, t across days of the daily mean improvement; (b) there are at least 40 of them
on at least 10 distinct days; and (c) the mean improvement is larger than the placebo's. Reported regardless: the
result by market and by release, the share of affected opportunities improved, the median, and the placebo's own
mean and t. Analysis runs once, after 2026-11-27, alongside the L2 protocol; it is a separate single test and
does not change that protocol's count of thirteen families.
