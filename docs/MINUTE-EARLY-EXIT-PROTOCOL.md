# Exit one minute before the hour — frozen protocol (2026-10-01, before the data it is judged on)

The user chose to fix this candidate now and judge it later on the LIVE paper recordings. The app keeps trading
the frozen method unchanged (hold 60 minutes); nothing below changes the runtime. The hash of this file is in
`MINUTE-EARLY-EXIT-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the
results are looked at.

## Where the idea came from
On the 80 cached days (08:00-16:59 New York) the minute starting on the hour moved 2.2x (CL) to 4.5x (SI) a typical
minute and the half hour 1.9x to 10.7x, with scheduled releases, the stock open and close, settlements and the Fed
all landing on round times. The frozen method buys and sells exactly on the hour, when option market makers widen.
Release-day exits are tested separately by `RELEASE-EXIT-PROTOCOL.md` (X7); this tests the ordinary hours.

## Rule (X8; no free parameter)
For an opportunity with clock C and planned exit E = C + 60 minutes, sell at E - 60 seconds instead of at E.
Entries are unchanged: a minute before the clock the signal does not exist (it uses the bar that ends at C).

## Data
- Opportunities, quotes and P&L exactly as in `L2-VETO-EXIT-PROTOCOL.md` ("Data"): every recorded clock with a
  verified candidate option, traded or not; buy at the ask plus one tick at the clock, sell at the bid less one tick
  (the last quote at or before the exit time); fees per side and the conversion mark-up; as a share of the money
  paid in. For traded clocks the baseline uses the actual fills; the X8 exit always uses the capture.
- Clocks 2026-10-02 to 2026-11-27, all weekdays. With nothing to tune there is no discovery/holdout split.
- Ordinary opportunities: all except those X7 affects (a release in `docs/release-calendar-2026-10-11.yaml`, as
  hashed in `RELEASE-EXIT-PROTOCOL.sha256`, within E - 120 seconds to E + 60 seconds). One lacking a quote at either
  exit time is excluded and counted.

## Verdict
X8 PASSES only if, on ordinary opportunities: (a) the mean improvement over the 60-minute exit (per opportunity,
after costs) is positive with t > 2, t across days of the daily mean improvement; and (b) there are at least 150 of
them. Reported regardless: the same over all opportunities including X7's, by market and by clock hour, the share
improved, the median, and the option spread (as a share of the mid) at E - 60 seconds and at E. Analysis runs once,
after 2026-11-27, alongside the other protocols; it is a separate single test.
