# PAPER Gateway scanner access check — 2026-09-10

Completed a read-only capability request followed by finite scanner snapshots across all 14 supported market profiles. The account subscription list was not supplied; this report records observed access and Gateway warnings, not confirmed paid subscriptions.

Observed window: 2026-09-10T21:23:47.844220+00:00 to 2026-09-10T21:25:35.010837+00:00. Gateway API server version **178**; **527 scan codes, 156 locations and 1153 filter-field codes** advertised.

## Material finding: production V1 opening-scan resolver is blocked

The actual Gateway advertises `TOP_OPEN_PERC_GAIN` (Top % Gainers Since Open) and `TOP_OPEN_PERC_LOSE` (Top % Losers Since Open). Both worked in the diagnostic snapshots. However, the current semantic resolver also matches `HIGH_OPEN_GAP` and `LOW_OPEN_GAP`, which refer to close-to-open movement. Its loose loss regex additionally matches the word "Close" in the gain-gap description. It therefore marks both opening families ambiguous. With the default all-components-required policy, a V9 acquisition run will degrade rather than complete. **The deployed resolver was not changed in this check.**

The current acquisition recipe must resolve that ambiguity explicitly before a prospective opening test. No gap scan was substituted. The diagnostic used the two exact advertised since-open codes under `SCANNER_ACCESS_DIAGNOSTIC_EXPLICIT_CODES`; it did not modify or validate production V1.

## Observed market matrix

Each full matrix is five exact scan codes × seven cap slices: UNCAPPED, BELOW_MICRO (<$50m), MICRO, SMALL, MID, LARGE, MEGA. No price/volume floor. The five codes were `TOP_TRADE_RATE`, `TOP_VOLUME_RATE`, `HOT_BY_VOLUME`, `TOP_OPEN_PERC_GAIN`, `TOP_OPEN_PERC_LOSE`.

| Market profile | Completed / planned | Unique raw conIds | Components with warning 492 | Result |
|---|---:|---:|---:|---|
| US_NASDAQ | 35/35 | 1,058 | 0 | No scanner precision warnings |
| US_NYSE | 35/35 | 1,058 | 0 | No scanner precision warnings |
| US_ALL | 35/35 | 1,058 | 0 | No scanner precision warnings |
| CANADA_TSX | 35/35 | 632 | 35 | Additional permissions needed for precise scanner results |
| UK_LSE | 35/35 | 669 | 35 | Additional permissions needed for precise scanner results |
| GERMANY_XETRA | 35/35 | 812 | 35 | Additional permissions needed for precise scanner results |
| FRANCE_PARIS | 35/35 | 521 | 35 | Additional permissions needed for precise scanner results |
| NETHERLANDS_AMSTERDAM | 35/35 | 372 | 35 | Additional permissions needed for precise scanner results |
| SWITZERLAND_SIX | 35/35 | 611 | 35 | Additional permissions needed for precise scanner results |
| AUSTRALIA_ASX | 35/35 | 0 | 35 | Additional permissions needed for precise scanner results |
| HONG_KONG_HKEX | 35/35 | 259 | 35 | Additional permissions needed for precise scanner results |
| JAPAN_TSE | 35/35 | 654 | 35 | Additional permissions needed for precise scanner results |
| SOUTH_KOREA_KRX | 5/35 | 0 | 0 | Uncapped empty; 30 cap components blocked by unusable USD/KRW quote |
| SOUTH_AFRICA_JSE | 0/35 | 0 | 0 | Configured STK.ZA.JSE not advertised |

US_NASDAQ, US_NYSE and US_ALL currently share the market profile scanner location STK.US.MAJOR. Their identical diagnostic requests were reused, not sent three times. These are raw scanner contracts before authoritative membership and security eligibility; the counts are not eligible acquisition-pool sizes. Canada uses STK.NA.CANADA in the current profile. ASX returned zero rows at this observation time; empty after-hours/before-open results do not demonstrate lack of a subscription or opening-session coverage.

Warning 492 explicitly requested additional permissions for precise results for Canada (TSE/VENTURE), LSE, Xetra, Paris, Amsterdam, SIX, ASX, HKEX and Japan. It is a request-scoped precision warning, not a connection failure. Korea returned no warning on its five empty uncapped responses; entitlement remains unconfirmed.

## Observed resource behavior

- Actual scanner requests: **355**, maximum concurrency **2**. **70** identical US-profile component requests reused.
- Raw scanner hits across actual requests: **8,492**.
- Requests reaching scannerDataEnd: **355**; cancellations: **355**.
- Total access-check elapsed time, including FX and setup: **107.167 seconds**.
- Scanner latency p50/p90/p95/p99: **581.9/752.6/860.2/1029.3 ms**.
- No scanner request failed; 65 planned components were unsupported/unready (30 Korean cap partitions, all 35 JSE components).
- Broker cancellation notifications 162 are retained in the log; these are not historical requests or request failures.
- Opening-history requests: **0**. No Range250, RV50, RV30, oracle recall, or opening-deadline throughput was measured. Those need prospective opening sessions and saved broad eligible membership.

## Persistence and operational safety

Server artifacts: `/var/lib/stocker/benchmarks/acquisition-access-20260910/` (`capabilities.sqlite`, `capabilities.scanner-check.json`, `access.sqlite`, `access.scanner-check.json`, `access.log`). Raw Gateway XML is stored in `acquisition_capabilities`; the JSON contains component filters, request timing, warnings, raw hits and scanner ranks. Local copies: `/tmp/stocker-acquisition-access-20260910/`.

Initial capability digest: `3d01200a211c5595491992d7007281b44f58ef460c9b937db4cf956e6e553054`.
Access report SHA256: `2d6eabff98501df96c9dd34f2e20bac9591c0d380a5efa6ccf7f34530bb1ac09`.

Used the updated `scripts/benchmark_scanner_acquisition.py` as an isolated `/tmp/benchmark_scanner_acquisition.py` copy against the deployed adapter, with PAPER client ID 292 and `execution_enabled=False`. This diagnostic CLI does not require or construct a strategy run, historical queue or execution service. The full opening benchmark retains its dedicated-Gateway requirement. The production release remains `64b6e89`; no service restart, run migration, activation or production recipe update was performed.

Post-check verification: PAPER connected/reconciled/ready; zero open positions and orders. Existing run IDs, specification hashes and configuration file hashes unchanged. Database execution plans 1, attempts 1, fills 28, signals 52,661, method runs 21, saved run configurations 53 and historical bars 136,005 all unchanged. No orders were placed.

Validation: three new diagnostic safety tests pass; the focused diagnostic/acquisition/broker suite passes (47 tests). Ruff and mypy pass for the changed CLI; Ruff passes for the new tests. The initial regression collection was missing the locked FastAPI dependency; `uv sync --frozen --group server` restored it, and the suite then passed.

RANGE5 HIGH TOP250 → RV10 HIGH TOP50 → RV15 HIGH TOP30 remains unchanged. Session HARD qualification, MODEL_T0, Q1, MID/cohort, entries, exits and execution rules were not changed.
