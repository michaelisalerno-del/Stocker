# Don't enter while the future runs against the option — frozen entry veto (2026-10-05, before the data it is judged on)

The user asked to test whether order-flow "pressure that fails to produce movement" (absorption) helps entries and exits,
and, when it turned out almost never to occur, chose to freeze the one related observation that did: entering just after
the future moved hard against the option. It is a paper test on the hourly ledger: the app keeps taking every clock,
nothing changes the runtime and no order is sent. The hash of this file is in `MOMENTUM-AGAINST-VETO-PROTOCOL.sha256`; any
later change is an amendment, dated and stated as such, made before the results are looked at.

## Where the idea came from, and what was seen
On the app ledger's closed paper trades of 1–5 October (72 with a recorded option path, 50 with a known futures state),
loud trading with no price progress over a minute (absorption: volume at least 1.5x normal, the mid within one tick) was
seen once at an entry and fired once as an exit warning: at Saxo's one-second sampling, heavy trading in CL, ES, GC and NQ
nearly always moves the price. What did separate entries was the futures moving against the option on heavy volume in the
minute before the fill: 10 trades, 60% never above the price paid, median close −40%, −£95 a trade, against 44%, −30% and
−£60 for quiet entries and +£42 for the 5 loud entries moving the option's way. An entry autopsy on the same trades found
the same pattern by another cut (future against by more than 3 ticks in the last minute: 47% never above the price paid,
−£86 a trade). The two share most of their trades: one weak signal, seen twice, chosen by looking. 1–5 October is not used
to judge it. This is a different test from the frozen L2 vetoes V1–V7 (`L2-VETO-EXIT-PROTOCOL.md`), which read the
displayed book, not the price's move on traded volume, and from `WALL-PROTOCOL.md`.

## Definition (parameters fixed here)
- Futures state: the clock's market's book-flow messages in the capture segments (CURRENT), the last at or before a time
  and at most 30 s old.
- LOUD: the traded volume over the 60 s before the clock (the book flow's 60-second volume change) is at least 1.5x its
  median over the hour before the clock, that hour's samples spanning at least 10 minutes.
- AGAINST: the futures' weighted mid moved at least 2 ticks of that future against the option's direction (a call down, a
  put up) over the 60 s before the clock.
- The veto skips the clock when LOUD and AGAINST both hold. With no state (no sample, too little history) the clock is
  kept and counted as unknown.
- Population: every clock with the app's selected option and its hour outcome, traded or skipped by the app; the outcome
  is the ledger's `opt_ret_ask_to_bid` (the option bought at the ask at the clock, sold at the bid an hour later).
- Column: `veto_ma.csv`, written by `veto_ma.py` in the hourly ledger (sha256
  ef4a57528659f50b1f147a2024b8b70950c2da5674440bc134c750dc7207194b at freezing) after `build.py`.

## Data
Judged clocks: 2026-10-06 to 2026-11-27, every clock the app observes. Halves: 6–30 October and 2–27 November.
1–5 October is the look and is not judged.

## Verdict
The veto PASSES only if, judged once after 2026-11-27: (a) the kept clocks' mean `opt_ret_ask_to_bid` exceeds the vetoed
clocks' with t > 2, t across days of the daily difference (days with at least one clock of each); (b) at least 30 vetoed
clocks on at least 10 distinct days; (c) the difference is positive in both halves. A veto that removes clocks that lose no
more than the rest has not earned its place, however many losing trades it removes.

Reported regardless: the vetoed and kept clocks' mean and median returns, the share never above the price paid, and the
pound result of the app's actual trades on vetoed clocks; how many clocks whose option at least doubled the veto would have
removed (big winners missed count against it); by market; the original 09:00–16:00 clocks against the others; the
frozen-scope clocks (CL, GC, NQ, 09:00–16:00) on their own; the unknown clocks by cause; and, for information only, the
grid LOUD 1.2x/1.5x/2x × AGAINST 1/2/4 ticks.

A pass earns the veto a place in the frozen method only by a dated rulebook amendment and a new rule version after
2026-11-27.
