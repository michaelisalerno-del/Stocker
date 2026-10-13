# Hourly ledger: straddle outcomes valued on the wrong series — correction (2026-10-13, on the user's "Go ahead", before any verdict is computed)

Amends the straddle outcome columns `straddle_ret_ask_to_bid` and `straddle_ret_mid` used by `PRICED-BELOW-PROTOCOL.md`
and `PRICED-BELOW-AMENDMENT-1.md` (primary and reported outcomes), and the `build.py` hash cited by the protocols that
name it. Nothing changes in the rules, thresholds, periods or verdict bars, and nothing changes the runtime. The hash of
this file is in `LEDGER-STRADDLE-EXPIRY-AMENDMENT-20261013.sha256`.

## What went wrong
`build.py` values the clock's at-the-money straddle an hour later by looking up the same strike on the hour-end chain
board, without checking which expiry the quote belongs to. When the clock's series expires inside the hour (ES and NQ at
the 15:00 New York clock, GC at 13:00, CL at 14:00), the hour-end board lists only later series, so the "end value" is
the next expiry's straddle. Every such row showed a large fake gain, for example:

- ES, 5 October 15:00: straddle bought at 8.45, "sold" an hour later at the next day's straddle: +303%.
- CL, 14:00 on 5, 6, 7 and 8 October: +441%, +458%, +414%, +349%.
- GC, 13:00 on 5, 7 and 8 October: +487%, +622%, +402%.

The same lookup also took a later series on two clocks whose hour-end board no longer listed the clock's own series
(ES 7 October 14:00, +176%; GC 12 October 12:00, +364%), and valued one GC clock (6 October 14:00) whose board still
listed a series that had already expired at 13:30.

The app's own option has been valued correctly since the ledger began (`opt_end_source` = `EXPIRY_INTRINSIC` when it
expires inside the hour); only the straddle columns were affected.

## Correction
`straddle_expiry_fix.patch` in the ledger folder (sha256 5685b4e0c8c49d1183d540d755bb58ad6cc85bade9ffd7f8c519ef9b61ee4a18):

1. A straddle whose series expires inside the hour (`T < expiry <= hour end + 60 s`) is worth its intrinsic value,
   |futures − strike|, at the futures' close of the expiry minute, the same settlement the app's option uses. With no
   futures bar at that minute there is no outcome.
2. Otherwise the hour-end quote must come from the clock's own expiry; if the hour-end board does not list it, there is
   no outcome.
3. A clock whose board lists a series that has already expired gets no straddle outcome.

The 120-second hour-end quote rule in `PRICED-BELOW-PROTOCOL.md` applies to quoted rows only; rows settled under rule 1
need no hour-end quote (`chain_age_end` is left empty).

## Effect, checked on the rebuilt ledger (1–13 October, 524 rows)
- Only `straddle_ret_ask_to_bid` and `straddle_ret_mid` (21 rows each) and `chain_age_end` (16 rows) changed; every
  other column is identical. In `fairvar.csv`, written by `edges.py` (unchanged), only the same two straddle columns
  changed. The number of straddle outcomes went from 334 to 333.
- H4 (CL at the 13:00 and 14:00 clocks, GC at 14:00 and 15:00) included the CL 14:00 clock, which settles at the CL
  expiry at 14:30, so H4 would have passed on the fake gain alone. The pooled H1–H5 means also included the NQ 15:00,
  GC 13:00 and CL 14:00 clocks (ES is reported, never judged).

## Disclosure
Before this amendment, on 2026-10-13, Claude looked at judged-period data in two ways:
- The bug was found in a descriptive table of straddle outcomes by market and New York hour for 1–13 October
  (`~/Codex/2026-10-13-vix-level/beat_times.py`; 3–5 clocks per cell), where the expiry hours showed impossible gains.
- While checking the fix, the H4 cells' straddle `ask->bid` means for 5–13 October, before and after: CL 13:00 −30%
  (4 clocks, unchanged); CL 14:00 +415% (4) → −88% (3); GC 14:00 +13% (3) → −6% (2); GC 15:00 −5% (1, unchanged).

H1, H2, H3 and H5 and the combined test were not computed. The correction has no free choice in it: an expiring
straddle pays its intrinsic value, and a quote from another series is not the straddle's price. So the look cannot
steer it toward or away from a pass.

## Hashes at freezing (patch applied, ledger rebuilt)
- `build.py` 239a8ae900f5638a194b78bdcae3d87fd7f5987e8a131dbf3b5e51e860656f86 (was
  69addc28dd9f7c3d97f367b2a4509f3c371f2965979eb9a5d8deae35bdd813f2): this patch only.
- `edges.py`, `report.py` and every other ledger script: unchanged.
