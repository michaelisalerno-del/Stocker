# Clock-aligned rules: the last hour before the index expiry, exits after a release, and the roll — frozen protocol (2026-10-06 ~07:05 UTC, on the user's "Freeze", before the data it is judged on)

Three rules from the Level 2 research note (2026-10-05) that need no new data and accrue slowly, so they are fixed before
the clocks they are judged on. Paper comparisons on the hourly ledger; the app keeps trading its frozen method, nothing
changes the runtime and no order is sent. The hash of this file is in `CLOCK-RULES-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the
results are looked at.

## Common
- Population: every ledger clock with the app's option and an hour-end outcome (`opt_ret_ask_to_bid`: the ask at the
  clock to the bid at clock + 60 min + 3 s, no bid = 0, or the intrinsic value when the option settles as the hour ends),
  traded or not, in every market and session.
- Judged clocks: from the first clock after the freezing commit (the 04:00 New York clock of 2026-10-06) to
  2026-11-27 16:00 New York; halves split at 2026-10-31. Clocks before the freeze are not judged.
- Columns: `clock_rules.csv` (rules G and H), written by `clock_rules.py` in the hourly ledger, and `roll.csv` (rule R),
  written by `roll.py`, both after `build.py` (sha256 69addc28dd9f7c3d97f367b2a4509f3c371f2965979eb9a5d8deae35bdd813f2).
  At freezing: `clock_rules.py` c8eebaf6345c3018108ebc4b04e85ddd3948fc67fb37d1f2bb82e6875ebe909d, `roll.py`
  d5a192f8835de5d99b6a7e20957613f5b82bd1b63e4fd40f343d600c535b4e4f.
- Calendar for H: `release-calendar-2026-10-04-markets.yaml` (sha256
  ffa1ff257107b253ba23bc361f92f228521fb3ab48dbbff17e360a73dfcc7dfe), the app's deployed event calendar since the market
  change of 2026-10-04: the release times of `release-calendar-2026-10-11.yaml` (frozen with `RELEASE-EXIT-PROTOCOL.md`)
  with ES added to every macro release and NG/SI removed. Corrections follow the RELEASE-EXIT rule: only by a dated
  commit made before the release concerned, citing the publisher.

## G — the last hour before the E-mini expiry (ES and NQ, the 15:00 New York clock)
Source: Baltussen, Da, Lammers and Martens (equity index futures 1974–2020): the return from the prior close predicts
the last half hour (β 4.18, t 7.29), and mostly on negative net-gamma days (6.63, t 4.78; positive-gamma days 0.82).
- day_ret = the future's return from the prior weekday's 16:00 New York close to 15:00 (one-minute bar closes, the
  ledger's `bar_grid`).
- **G1 keep** the clock when day_ret agrees with the option's direction (a call after a rise, a put after a fall);
  otherwise veto it.
- **G2 keep** only when G1 keeps and `gex_net_share` < 0 at the clock (the ledger's naive open-interest gamma sign).
- **G3 exit**: at 15:30, if the return from the prior close to 15:30 disagrees with the option's direction, sell at the
  bid of the first option quote received at or after 15:30:03 (the futures-triggered fill of
  `QUOTE-LAG-AMENDMENT-20261005.md`); otherwise hold to 16:00.
- Verdict, each judged once after 2026-11-27: G1 and G2 PASS only if the mean outcome of kept clocks exceeds that of
  vetoed clocks with Welch t > 2 (clocks are one per market per day), at least 15 kept and 15 vetoed, the kept mean is
  itself positive, and the difference is positive in both halves. G2 with fewer than 15 kept clocks is "insufficient",
  not a fail. G3 PASSES only if the mean of `g3_delta` over the clocks where it fired is positive with t > 2, on at least
  15 of them. Reported regardless: `g_fut_hour` (the future's 15:00→16:00 move signed by the option's direction) for kept
  and vetoed clocks — the mechanism without the option — by market.
- Expectation stated now: the naive gamma share has been positive on most ES/NQ clocks (0.24–0.89 at the four 15:00
  clocks seen), so G2 will probably be insufficient by 27 November.

## H — exit 15 minutes after a release that falls early in the hour (all markets)
Source: Luo and Kang (CL, 2003–2011): around the EIA report the jump probability is 42 times normal in the release
interval and 6.6 times for the next 25 minutes, with adjustment mostly inside 30 minutes; after macro announcements
Treasury volatility stays high while spreads normalise in 5–15 minutes (Balduzzi, Elton and Green). The note proposed it
for the CL 10:00 clock on EIA days only (about one observation a week); it is widened here to every listed release for
the market, so it can be judged in this window. The EIA subset is reported separately.
- Affected: a clock C with a release R for its market listed with C < R <= C + 43 minutes (on the calendar this is any
  08:30, 10:30 or 14:30 release, so R − C = 30 minutes; it never overlaps X7 of `RELEASE-EXIT-PROTOCOL.md`, which
  concerns releases within two minutes of the exit).
- Rule: sell at R + 15 minutes, at the bid of the last option quote at or before R + 15 min + 3 s (at most 30 s old; no
  bid = 0), instead of at the hour's end. `h_delta` = that return − `opt_ret_ask_to_bid`. An option settling before then
  is excluded and counted.
- Placebo: clocks at the same New York hours (08:00, 10:00, 14:00) on weekdays with no release for that market in the
  hour, sold at C + 45 minutes (`h_placebo_delta`). It separates the release from simply leaving 15 minutes early.
- Verdict: PASSES only if the mean `h_delta` over affected clocks is positive with t > 2 across days (the daily mean), on
  at least 40 affected clocks on at least 10 days, the mean is larger than the placebo's, and it is positive in both
  halves. Reported regardless: by market, by release, and the EIA subset with its own mean and t (the note's original
  rule, judged on its own once 30 EIA releases have been recorded, about May 2027).

## R — veto the clock when the pinned contract has lost the book (any market with its next contract recorded)
Source: across 45 WTI contract pairs, liquidity at the touch moves to the next month 2–3 days before expiry (BMLL). The
app records the next contract beside the pinned one from about a week before each re-pin (CL from 2026-10-04; ES, GC
and NQ only once their `next_contracts` are configured).
- At each clock, over the 60 seconds before it: S = pinned 10-level depth / (pinned + next) (medians), and each
  contract's median spread in ticks against its median at that New York hour on earlier days.
- **Veto** when S < 0.5, or when the pinned spread is at least twice its normal while the next contract's is at or
  below its own normal (`roll_veto`).
- Verdict: PASSES only if the mean outcome of kept clocks (with a next contract recorded) exceeds that of vetoed clocks
  with Welch t > 2, on at least 30 vetoed clocks on at least 5 days. With fewer it is "insufficient" and carried to the
  next rolls (CL's December→January roll is due mid-November; GC's, ES's and NQ's fall after 27 November), and judged
  when the count is reached.
- Separate from the verdict and not a test: `roll.py` prints each market's US-session median S per day. The first
  session below 0.5 is an objective date for the operational re-pin, replacing the calendar estimate (CL about 12–15
  October). Re-pinning remains a configuration change made in the restart window.
- Expectation stated now: if the re-pin follows the trigger, S < 0.5 occurs on few clocks and R will probably be
  insufficient; its value is mostly the re-pin date.

## What was seen before this was written
Counts only for G, H and R: 4 G clocks (all puts on rising days, so G1 vetoed all four and G3 fired on all four; their
futures moves 15:00→16:00 were printed while checking the script, their option outcomes were not); no H clock yet (the
recorded days had no 08:00, 10:00 or 14:00 clock on a release day with a release inside the hour); 23 R clocks, none
vetoed (S 0.54–0.69).
