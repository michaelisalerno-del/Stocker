# Three early exits for a profit — frozen protocol (2026-10-05, before the data they are judged on)

The user asked to test selling the app's option early once it is in profit, on three signals: its own quote leaning
against it, its bid coming off its high, and the future giving back its move. They are paper tests only: the app keeps
trading the frozen method unchanged, nothing below changes the runtime and no order is sent. The hash of this file is in
`LEAN-EXIT-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at.

## Where the idea came from, and what the look says
Of 44 paper trades closed by the 01:00 clock on 2026-10-05 with a recorded option quote path, 15 had a bid above the
entry ask at some point and closed below it. At those options' highs the futures book looked ordinary (depth, imbalance
and spread at the hour's median, the same as for the trades that held), but the option's own quote did not: the ask size
was 36 contracts against 5.5 for the trades that held, and the bid size had fallen to a fifth of its size at entry against
four fifths. The same shape of quote on a losing far-out-of-the-money option is that option's normal state and carried no
information, so every rule below reads its signal only while the position is in profit. The "move has stopped" rules were
added when the lean alone, replayed, fired within minutes and sold the largest winners.

The thresholds were chosen on 1–5 October by looking at those trades, so a replay on those days is not evidence. On the
definitions below (24 in-profit clocks with a paired baseline) all three were a wash on the mean (lean −0.7%, trail −0.3%,
retrace −1.2%), helped the median clock (+7%, +14%, +4%) and cut the one +450% winner (to +38%, +25% and +162%): a
one-tick dip on a cheap option is already a 10% move. Those days are not used to judge anything.

This is a different test from `QUOTE-LEAN-PROTOCOL.md` (a direction compass read from spread blow-outs) and sits beside
the frozen exit sets (`L2-VETO-EXIT-PROTOCOL.md` X1–X6, `MINUTE-EARLY-EXIT-PROTOCOL.md`, `EXIT-SET-2-PROTOCOL.md`); it
shares the entry and exit conventions of `EXECUTION-PROTOCOL.md`.

## Definitions (parameters fixed here; none are tuned on the judged data)
- Option and quotes: the app's selected option at a clock (`option_context.identity`) and every recorded two-sided quote
  of it (bid and ask present, DelayedByMinutes 0) in the capture segments, as the hourly ledger parses them.
- Entry: the last quote at or before clock + 3 s, at most 30 s old (`exe_ask0`): buy at its ask A0; its bid size is B0.
- Baseline exit: the last quote at or before clock + 60 min + 3 s, at most 30 s old: sell at its bid (no bid = 0). The
  baseline return is bid / A0 − 1 (`x_base_ret`). A clock without an entry quote, without such an exit quote, or whose
  option settles at expiry inside the hour, is excluded and counted.
- In profit: a quote after entry and before the baseline exit whose bid is above A0 (`x_in_profit`). Every rule acts only
  at such a quote, selling at its bid the first time the rule's condition holds; a rule that never fires has the
  baseline's exit.
- Rule L, lean: ask size ≥ 20 contracts and bid size ≤ 0.5 × B0 (no B0: never fires).
- Rule T, trail: the bid is at least 10% below the highest bid seen since entry.
- Rule R, retrace: the future's completed one-minute closes since the clock (the ledger's bar grid), signed in the
  option's direction (a call up, a put down), have given back at least 25% of their largest gain since the clock (no
  gain yet: never fires).
- Per rule: `<rule>_exit_ret` (the rule's return, bid / A0 − 1, or the baseline's), `<rule>_fired_min`, `<rule>_exit_bid`
  and `<rule>_delta` = `<rule>_exit_ret` − `x_base_ret`, the paired difference per clock.

## Data
The hourly ledger (`~/Documents/Codex/2026-10-03-hourly-ledger`, README.md; `build.py` sha256
db04206f2f53c19be20712bec838edd6eecc3fa1d46a87eb8eefc6521c1f9e14 at freezing, function `exit_rules`), one row per
market (CL, ES, GC, NQ) per clock, traded or not (a skipped clock whose selected option was recorded through the hour
counts as a paper path). Judged clocks: 2026-10-06 to 2026-11-27, every clock the app observes. Halves: 6–30 October and
2–27 November. 1–5 October is the look and is not judged.

## Verdict
The population is the clocks with a paired baseline that were ever in profit (`x_in_profit`); the rules can do nothing
elsewhere. Each rule is judged on its own, once, after 2026-11-27, and PASSES only if: (a) the mean of its `_delta` is
positive with t > 2.5, t across days of the daily mean (three tests: about a 2% chance that any passes by luck); (b) at
least 40 such clocks on at least 10 distinct days; (c) the mean is positive in both halves. The mean, not the median,
decides: a rule that helps most clocks while giving away the few large winners has not paid.

Reported regardless, for each rule: the median `_delta`, clocks helped and hurt, the single largest positive and negative
contribution and the verdict with each removed (tail dependence), the fire minute, `x_base_ret` against `_exit_ret` in
sum; by market; the original 09:00–16:00 clocks against the others; traded against skipped clocks; the exclusions by
cause; and, for information only, never for a verdict, a sensitivity grid: L at ask size 10/20/40 and bid fraction 1/3,
1/2, 2/3; T at 5%, 10%, 20% and at 2 and 3 ticks of the option; R at 25%, 50%, 75%. Analysis runs alongside the other
protocols; it is one family of three single tests.

A pass earns a paper trading rule frozen for a further test, not a change to the running method.
