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
| NG | Buy put, 0DTE | 0.10 | +500% |
| NQ | Buy put, 0DTE | 0.10 | +500% |
| SI | Buy put, 0DTE | 0.20 | +300% |

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
next expiry, CFD, direct future or fractional quantity to manufacture an affordable trade.

GC is treated like the other four markets (unblocked 2026-09-30 at the user's request): it warms and
records option candidates, and enters once its listed execution mapping is approved like any market. No
experimental GC management rule is adopted.

INTERNAL_PAPER consumes no broker orders. Entry fills use ask plus one tick, exits bid minus one tick,
with fresh displayed size and verified fee assumptions; no midpoint fills. All fills/P&L are labelled
internally simulated. SAXO_SIM uses verified SIM account orders and confirmed broker evidence. Prechecks
are required, disclaimers block, timeouts are never blindly retried, and unknown exposure blocks entries.
Confirmed positions and terminal order/fill states govern capacity release. Costs/FX absent from SIM
execution evidence leave P&L provisional, never zero-filled. Failed closure is a prominent exception.
