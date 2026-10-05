# Clock set amendment (2026-10-04, evening)

Amends the clock list in `docs/MARKET-CHANGE-AMENDMENT-20261004.md` ("every hour of the CME
session, 18:00 to 16:00 New York"). The rule version `CLOCK60_23H_ES_20261004` is unchanged: no
clock that could ever have traded is added or removed, so every recorded signal keeps its meaning.

## What changed

Two of the 23 hourly clocks could never enter, by construction:

- **18:00 New York** (the open). The eligibility gate needs 31 completed minutes before the clock
  (`rules.prior_rv(..., 30)`) and a non-empty session since 18:00; 17:00–18:00 is the CME
  maintenance break, so every 18:00 clock was skipped `INCOMPLETE_COMPLETED_HISTORY`.
- **16:00 New York**. Its exit is 17:00, the CME close. `contracts.verified_cutoff` requires the
  exit inside an option trading session (`start <= exit < end`) and two minutes before the
  contract's cutoff; 17:00 is the session end, so every 16:00 clock was skipped
  `OPTION_EXIT_SESSION_UNVERIFIED` (the sibling case, NQ's 15:00 clock under `SAME_DAY`, is in
  the ledger as `UNSUPPORTED_EXIT_BEFORE_CONTRACT_CUTOFF` six times on 1–2 October).

`rules.clocks()` now lists 19:00–23:00 on Sunday–Thursday evenings and 00:00–15:00 on weekdays
(21 clocks a day). The skip rows those two clocks wrote each day looked like data problems in the
audit; nothing else changes.

## What did not change

- Every other clock, its exit anchor (+60 minutes), the eligibility gate, the references, the
  option selection and the frozen verdict plans (`CL/GC/NQ`, 09:00–16:00 clocks, same-day).
- `us_clock()` still names 09:00–16:00 as the inherited definitions; 16:00 simply never occurs.
