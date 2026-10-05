# Hourly ledger: the CL futures series mixed CLX6 and CLZ6 from 4 October — correction (drafted 2026-10-05, frozen 2026-10-05 ~23:30 UTC on the user's "freeze", before any judged result)

Amends the ledger scripts cited by `MOMENTUM-AGAINST-VETO-PROTOCOL.md`, `EXIT-SET-3-PROTOCOL.md`,
`PRESSURE-PROGRESS-AMENDMENT-1.md` and `LEAN-EXIT-PROTOCOL.md` (which cite `build.py`, `veto_ma.py`, `exits2.py` and
`pressure_progress.py` by sha256). Nothing changes in the rules, thresholds, periods or verdict bars; only how the
scripts identify the market's own future in a capture segment. Nothing changes the runtime. The hash of this file is
in `LEDGER-NEXT-CONTRACT-AMENDMENT-20261005.sha256`; the new script hashes are listed below.

## What went wrong
Release 93d188b (deployed 2026-10-04 17:41 UTC) streams the next CL contract, CLZ6, beside the pinned CLX6 ahead of the
re-pin, as role `NEXT_CONTRACT`, and the recorder writes its messages into CLX6's capture segment. The role is in the
service's configuration, not in the identity written on each record: a CLZ6 record carries `market: CL`,
`symbol: CLZ6`, `uic: 31051110` and no `role`. Three ledger parsers filter the next contract out by
`ident.get("role") != "NEXT_CONTRACT"`, which never matches, so from 2026-10-04 18:00 New York every CL futures series
in the ledger interleaves CLX6 (about 91.05) and CLZ6 (about 89.45) second by second:

- `build.py` `parse()` → the per-market futures series `fut` (weighted mid, depth, imbalances, displacement, 60-second
  volume, spread). Visible in `veto_ma.csv`: CL `ma_move_ticks` of −154, −151, +146 and +133 on 5 October (the
  CLX6−CLZ6 spread of about $1.50 is 150 ticks), and the 04:00 clock vetoed on it. `exits2.py` reads the same series
  for the futures mid behind the minute-10 IV exit, `timing.py` and `tops_for()` for the descriptive tables.
- `pressure_progress.py` `parse_books()` → the five-level books behind pressure, progress, range breakouts and the
  direction states for CL.
- `extras.py` → relative and average volume (descriptive).

Other markets have no next contract streamed yet (ES, GC and NQ get theirs a week before their rolls), so their
series are unaffected. The app's own `signals.detail` fields (book_flow, forecast, option context), the option series
and the chain series are per instrument and unaffected; so are the ledger columns built from them
(`opt_*`, `fut_*` from the bar cache, GEX, smile).

## Correction
Each parser takes the segment's own future from its manifest key (`SAXO:SAXO_LIVE:ContractFutures:<uic>`) and keeps only
futures records with that uic. The exact change is `next_contract_fix.patch` in the ledger folder (three files; it adds
`own_uic()` to `build.py`, replaces the role test in all three and bumps their `PARSE_VERSION` so the cached parses are
rebuilt; sha256 0852c2948035092155020b1856840b4aafd44d61db1a237fb6957f6431ea13e4). After it is applied every CL row from 4 October 18:00 New York on is rebuilt by `sync.sh`; rows before that
are unchanged by construction (one futures contract in the segment).

## Hashes at freezing (patch applied, ledger rebuilt)
- `build.py` 69addc28dd9f7c3d97f367b2a4509f3c371f2965979eb9a5d8deae35bdd813f2 (was
  db04206f2f53c19be20712bec838edd6eecc3fa1d46a87eb8eefc6521c1f9e14): this patch only.
- `pressure_progress.py` a9afb69f288201a44172d370528089f7b0186c3c8671d1aa608e0f1e5375b5b9 (was
  077ea4f45fe083c3f0873e9c863f51387ee9bebb509ee903540af83f1165d1e8): this patch plus the `*_lag` columns of
  `QUOTE-LAG-AMENDMENT-20261005.md`; the frozen columns' definitions are unchanged.
- `extras.py` (descriptive, not cited by a protocol) 89ab13ca37ec1b01bbced6194329cb566fdafbcdce977c6245dfdfff52424d6d.
- `veto_ma.py`, `exits2.py`, `delta_cond.py`, `edges.py`, `entry_pullback.py`, `veto_be.py`: unchanged.

## What was seen before this was written
Only counts and the contaminated `ma_move_ticks` values above; no judged return of any affected protocol was computed,
and the 1–5 October clocks are not judged by any of them. The momentum-against look recorded at freezing ("6 of 52 known
clocks vetoed") included the two CL clocks of 4–5 October that fired on the mixed series.
