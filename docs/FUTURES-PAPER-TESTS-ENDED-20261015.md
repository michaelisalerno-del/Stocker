# The futures paper tests ended early (15 October 2026)

On 15 October 2026, about 21:45 UTC, the user cancelled the CME, COMEX and NYMEX Level 2 real-time
data on Saxo ("pointless keeping the old futures if they are negative") and subscribed to OPRA and
Nasdaq Level 1 instead. A read-only check at 21:47 UTC found ES, NQ, CL and GC quotes and charts
10 minutes delayed (chart stream `SAXO_CHART_DATA_DELAYED`, option quotes `UNVERIFIED`). The same day
the user approved replacing the futures-options runtime with the CFD and QQQ paper record
([protocol](PAPER-RECORD-PROTOCOL.md)).

## What this ends

Every frozen test fed by the futures runtime's hourly ledger, judged on its recordings from
14 October to 27 November 2026, stops receiving data on 15 October:

- PRICED-BELOW (`PRICED-BELOW-PROTOCOL.md`, `PRICED-BELOW-AMENDMENT-1.md`, and the straddle-expiry
  amendment in PR #29);
- FEAR-RISING, V1-PAPER and the WALL-PROTOCOL scorer (PR #30);
- EMA12-FIRST-CANDLE and its amendment 1 (PR #31), whose option-cost verdict was the real-price
  check of the first-candle rule;
- PRESSURE-PROGRESS (`PRESSURE-PROGRESS-PROTOCOL.md`, `PRESSURE-PROGRESS-AMENDMENT-1.md`) and the
  other hourly-ledger protocols in this folder judged in November.

None of these will issue the verdict it registered: their samples are cut to the days already
recorded (from 1 to 15 October at most), far below the sizes they set. Nothing recorded is deleted;
the ledgers and recordings stay on the server (`/var/lib/stocker/v1/`) and in the hourly ledger
folder. Any later look at those few days is descriptive, not a verdict.

The first-candle option-cost question moves to the paper record, which prices it on the QQQ options
the user will actually trade instead of NQ futures options.

## Pull requests

The protocol pull requests that never merged (#25, #29, #30, #31), the option-root fix that was
deployed without merging (#28) and the deployment record #33 describe the retired runtime. They are
closed unmerged with a link to this note; their branches stay on GitHub.
