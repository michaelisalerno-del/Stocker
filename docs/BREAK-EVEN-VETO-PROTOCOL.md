# Don't buy an option an ordinary hour can't pay for — frozen entry veto (2026-10-05, before the data it is judged on)

The user asked whether an option is worth buying now, and chose to test the break-even move as an entry veto: skip a clock
when the future would need a bigger-than-normal hour in the option's direction just for the option's bid to get back to
the price paid. It is a paper test on the hourly ledger: the app keeps taking every clock, nothing changes the runtime and no
order is sent. The hash of this file is in `BREAK-EVEN-VETO-PROTOCOL.sha256`; any later change is an amendment, dated and
stated as such, made before the results are looked at.

## Where the idea came from, and what was seen
On the app ledger's 72 closed paper trades of 1–5 October with a recorded option path (computed at the actual fill): the
27 trades whose bid never rose above the price paid needed a median 1.02x a normal hour to break even, the others 0.65x.
By thirds of the break-even: 24%, 35% and 54% never above the price paid; median close −22%, −29%, −43%. By market it
ordered the losses (ES 0.37x, NQ 0.54x, GC 0.81x, CL 1.20x). With the implied volatility 10% lower, the median break-even
rose from 0.75x to 1.23x. It is mostly the option's spread (rank correlation +0.76), and half the trades that doubled had
a break-even in the top third: the veto removes the cheapest lottery tickets along with the dead trades. On the ledger's
clock-time definition below, 20 of 78 look-period clocks would have been vetoed. 1–5 October is not used to judge it.

## Definition (parameters fixed here)
- From the ledger's clock-time fields only: the app's selected option's quote at the clock (`opt_bid`, `opt_ask`), the
  futures mid (`fut_mid`), the option's strike, right and expiry instant (`opt_strike`, `opt_right`, `opt_expiry`).
- Normal hour: twice `rv15_hour_median`, the median 15-minute realised move (log) of that New York hour over the reference
  sessions.
- Black-76 on the future: the option's implied move from its mid at the clock; after 60 minutes with the implied volatility
  unchanged, the futures log move x in the option's direction (up for a call, down for a put) at which the model value less
  the clock's half-spread (mid − bid) equals the price paid (the ask). An option expiring inside the hour is valued at its
  intrinsic value. `be_ratio` = x / the normal hour.
- The veto skips the clock when `be_ratio` > 1.0, or when no move up to 50% would break even. Missing fields or no implied
  volatility: the clock is kept and counted as unknown.
- Population: every clock with the app's selected option and its hour outcome, traded or skipped by the app; the outcome is
  the ledger's `opt_ret_ask_to_bid`.
- Column: `veto_be.csv`, written by `veto_be.py` in the hourly ledger (sha256
  09a7b0f3b744f64ce5e566bed3fcb5494f033d0fce33698928e41d2159bf2219 at freezing) after `build.py`; it also records
  `be_ratio_iv_down` (implied volatility 10% lower) for the report.

## Data
Judged clocks: 2026-10-06 to 2026-11-27, every clock the app observes. Halves: 6–30 October and 2–27 November.
1–5 October is the look and is not judged.

## Verdict
The veto PASSES only if, judged once after 2026-11-27: (a) the kept clocks' mean `opt_ret_ask_to_bid` exceeds the vetoed
clocks' with t > 2, t across days of the daily difference (days with at least one clock of each); (b) at least 30 vetoed
clocks on at least 10 distinct days; (c) the difference is positive in both halves.

Reported regardless: the vetoed and kept clocks' mean and median returns and the share never above the price paid; the
pound result of the app's actual trades on vetoed clocks; how many clocks whose option at least doubled the veto removed
(big winners missed count against it); by market; the original 09:00–16:00 clocks against the others; the frozen-scope
clocks (CL, GC, NQ, 09:00–16:00) on their own; the overlap with `MOMENTUM-AGAINST-VETO-PROTOCOL.md` and with the strike
comparisons of `DELTA-TARGET-PROTOCOL.md`; the unknown clocks by cause; and, for information only, the limits 0.75x and
1.5x and the `be_ratio_iv_down` version.

A pass earns the veto a place in the frozen method only by a dated rulebook amendment and a new rule version after
2026-11-27.
