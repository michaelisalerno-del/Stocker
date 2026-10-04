# Amendment to every frozen protocol: markets, hours and expiries changed (2026-10-04)

Dated 2026-10-04, before any clock judged by these protocols (all start on 2026-10-05 or are judged later). The
hash of this file is in `MARKET-CHANGE-AMENDMENT-20261004.sha256`. The protocol files themselves are unchanged.

## What changed (the user's decisions, 2026-10-04)
- Markets: CL, ES, GC, NQ. Natural gas (NG) and silver (SI) are removed: part-week same-day options and the widest
  spreads. All their recorded data was deleted at the user's request (capture segments, bar cache and ledger rows,
  including 1–2 October). E-mini S&P (ES) is added and traded exactly as NQ (10-delta put, 60 minutes).
- Clocks: every hour of the CME session, 18:00 to 16:00 New York (rule version `CLOCK60_23H_ES_20261004`). The
  original 09:00–16:00 clocks keep their inherited definitions exactly: the eligibility gate's 08:00 session and
  08:00–08:30 opening, and reference sessions of 540 bars from 08:00 to 17:00. The other clocks use the CME session
  from its 18:00 open and reference medians from the same sessions' overnight minutes.
- Expiries: every market takes the nearest listed expiry still trading two minutes after the exit
  (`SAME_DAY_OR_NEXT_LISTED`), so overnight clocks buy the next day's option; at the original clocks this changes
  nothing where a same-day option trades past the exit, and replaces a skip with a next-day option where it does not
  (CL from its 14:00 clock, GC from 13:00).
- Recordings: the archive cap is 50 GiB (was 20 GiB) for the longer hours.

## How the frozen protocols are judged now
- Every verdict uses only what its protocol originally covered: the 09:00–16:00 New York clocks, the markets among
  CL, GC and NQ that the protocol named, and, where it concerns the app's own option, trades on same-day expiries
  (the clocks that were skipped before are excluded).
- NG and SI no longer exist in the data; their thresholds (for example PRICED-BELOW H1) are unused, and a protocol's
  minimum counts are judged on the remaining markets as written. The L2 protocol's discovery period (1–30 October)
  loses the 1–2 October NG and SI rows with the rest of their data.
- ES rows, the overnight clocks and next-day-expiry trades are reported alongside every verdict, never inside it.
