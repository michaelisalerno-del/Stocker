# Five-market frozen method

Runtime version: `CLOCK60_NG13_20260927`. Exact original paths and SHA-256 hashes are in
[saxo-frozen-sources.json](saxo-frozen-sources.json); the same hashes accompany every event.
`fixed_spec.json`, each `*_TAIL_FROZEN.json`, `build_features.py.txt`, and `tail.py.txt` remain
unchanged under `research/futures-integration-sources`. Historical providers and synthetic pricing
are not relabelled Saxo. Previous operating rules are archived in the parked-runtime directory.

| Market | Frozen primary | Absolute target delta | Historical attainment threshold |
|---|---|---:|---:|
| CL | Buy call, 0DTE | 0.10 | +500% |
| GC | Buy put, 0DTE | 0.10 | +500% |
| NG | Buy put, 0DTE; next listed expiry on days without one (2026-10-01) | 0.10 | +500% |
| NQ | Buy put, 0DTE | 0.10 | +500% |
| SI | Buy put, 0DTE; next listed expiry on days without one (2026-10-01) | 0.20 | +300% |

Triggers are the frozen weekday 09:00–16:00 hourly clocks in America/New_York. NG 13:00 is vetoed
but recorded. They are not every quote or near-clock update. The original opportunity time anchors
the 60-minute exit; recorder duration never changes it. Attainment thresholds above are research
outcome definitions, not newly added profit-taking rules. Secondary structures, GC16/G2 experiments,
FIRST4 stock slots, L2 filters and new volume vetoes are not execution rules.

The inherited feature-availability gate needs 31 completed closes, positive RV15, five preceding
source-session hourly medians, original opening/session windows and the underlying expiry guard.
Saxo historical samples supply actual OHLCV; missing volume is not zero. No VWAP, open interest or
volume is fabricated from last-price snapshots. Missing samples and mutable chart tails cannot enter
completed-bar calculations. Gap requests are bounded and DataVersion changes require a fresh fetch.
One dated exception (2026-10-01, the user's decision): Saxo's one-minute charts omit minutes in which
nothing traded, which blocked GC, NG and SI reference sessions (1-8 quiet minutes, mostly after 14:00 New
York). A run of at most five minutes for which Saxo sent no sample, between two real bars, is filled as
that many unchanged minutes (the previous close, zero volume). Longer runs (possible outages) and samples
Saxo sent but that fail validation (for example missing volume) stay missing and still block.

Actual display contracts must be explicitly pinned from the selected environment's ContractFutures
reference data. The standard CL/GC/NG/NQ/SI family, exchange, symbol/month and multiplier are verified.
A config change is audited; buffers and option positions retain their original provider/environment/UIC.
There is no unlabelled roll or continuous series. Contract month comes from the actual contract symbol,
not its sometimes-earlier last trading month (for example a November crude contract expiring in October).

The frozen reference-selection rule is greatest volume on the strictly preceding **completed exchange
session** among nearby actual futures, excluding expired contracts. Saxo daily chart boundaries have not
been verified as those exchange sessions. Therefore selection is not guessed. An explicit, current,
approved Saxo selection audit can be supplied via `reference_selections_file`; the implementation then
fetches each selected session's actual contract bars and validates complete 08:00–17:00 NY coverage.
See the schema in `reference_sessions.py`. Without that audit, the exact block is
`REFERENCE_SESSION_CONTRACT_SELECTION_UNVERIFIED`. This is a remaining data-verification requirement.

The research used continuous strikes and a hypothetical 17:00 NY expiry. It did **not** approve a
listed-product substitution, universal actual expiry time, delta tolerance, fee schedule or changed DTE.
All five execution mappings therefore start empty. Real FuturesOption reference relationships, actual
0DTE timestamp, approved nearest frozen-model-delta tolerance, whole quantity, account quote, fees,
GBP FX and an open session covering the original exit are required. Date-only/midnight expiry metadata
is insufficient. Exercise cutoff text is retained but is never silently interpreted as a UTC deadline.
Missing option-specific timing blocks admission. Never substitute a cheaper strike, micro contract,
next expiry, CFD, direct future or fractional quantity to manufacture an affordable trade. The one
approved exception is a mapping's `expiry_rule: SAME_DAY_OR_NEXT_LISTED` (below), never a fallback.

GC is treated like the other four markets (unblocked 2026-09-30 at the user's request): it warms and
records option candidates, and enters once its listed execution mapping is approved like any market. No
experimental GC management rule is adopted.

INTERNAL_PAPER consumes no broker orders. Entry fills use ask plus one tick, exits bid minus one tick,
with fresh displayed size and verified fee assumptions; no midpoint fills. Fresh means known current within
5 seconds: Saxo sends a price only when it changes, so since 2026-10-01 (the user's request, after cheap
options were measured unchanged for minutes) an unchanged quote or size stands while the socket has delivered
in the last 5 seconds and its own subscription has a Saxo heartbeat within its inactivity timeout, with no
pause or gap. A paused, gapped or silent subscription never stands. All fills/P&L are labelled
internally simulated. SAXO_SIM uses verified SIM account orders and confirmed broker evidence. Prechecks
are required, disclaimers block, timeouts are never blindly retried, and unknown exposure blocks entries.
Confirmed positions and terminal order/fill states govern capacity release. Costs/FX absent from SIM
execution evidence leave P&L provisional, never zero-filled. Failed closure is a prominent exception.

## Listed option families and daily expiries (2026-09-30)

CME lists a separate option root for each weekday of each week (crude "Mon Weekly (1)", "Tue Weekly (1)" ...,
gold, NQ likewise), each with one expiry. A mapping therefore approves the **family** of roots on the pinned
future (`option_root_ids`). The runtime loads every approved root's options, re-reads them once per New York
day (new weekly listings), points the observation chain at the root of the nearest expiry, and ranks only
options expiring today unless the mapping's expiry rule says otherwise (below). An expiry instant is either
listed in `expiry_instants`, or derived per day from Saxo's timestamped `LastTradeDate` only when it falls on
that expiry day at the approved `expiry_clock_new_york` (evidence required); conflicting timestamps leave the
day unverified, and a series expiring the same day at another time (for example AM-settled) is never a candidate. Live listings on
2026-09-30: CL, GC and NQ have a same-day expiry every weekday; NG only Mondays and Fridays; SI only Fridays.

Expiry rule (2026-10-01, the user's decision: "go longer for some"): a mapping may set
`expiry_rule: SAME_DAY_OR_NEXT_LISTED`. The runtime then ranks the nearest listed expiry whose verified
instant is more than two minutes after the 60-minute exit: today's when there is one, otherwise the next
listed day (for example silver on a Thursday buys Friday's options). Strike, target delta, the 60-minute
hold, costs and every other gate are unchanged; the frozen model prices the longer time to expiry. Each
plan records `expiry_rule`, and the option's expiry date shows how far out it was. NG and SI use it;
CL, GC and NQ stay `SAME_DAY`.

Reference sessions: `scripts/reference_audit.py` writes the daily selection audit before the session (a
systemd timer at 07:30 New York) under the user's standing approval of the prior-session volume rule, using
Saxo daily-chart volume. It reads the service's current access token and never refreshes it.
