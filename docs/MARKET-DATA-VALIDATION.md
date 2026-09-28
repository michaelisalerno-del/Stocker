# Validation evidence

All tests here are offline; prices, UICs, OAuth responses and fills are sanitised/generated fixtures.
No evidence below proves a Saxo entitlement, broker fill or exchange recording permission.

- Focused Saxo tests: boundaries/OAuth, field/null/array/index merges, frame fragmentation,
  duplicate/reset/session downgrade, actual whole-contract budget, atomic four-way capacity,
  15-minute checkpoint expiry, overlap deduplication, skipped-event captures, late option history,
  gzip crash recovery, queue/disk limits, protected pruning, internal non-transmission, ambiguous
  SIM request retention, non-ordering preflight, dashboard API and parked provider transport.
- Provider-independent frozen source/DST/feature-availability and durable late-fill/provisional-P&L
  regressions were retained in `tests/test_futures_invariants.py`. Obsolete IB-specific suites remain
  hashed historical evidence outside test collection; they are not falsely claimed as Saxo tests.
- Full Python suite: final result recorded after review. The first full run exposed two old smoke
  tests requiring now-retired EODHD launchers; they were replaced by an archive/hash/non-active check.
- Ruff format/lint and Mypy over `packages apps` pass; 115 source files type-checked at this stage.
- Locked server-only install/import/dashboard smoke passes with all socket connections forbidden.
  It installs no `ib-async`, pytest, notebook or research-model dependencies.
- Browser tests with locked Playwright dependencies pass five fixed cards, provisional P&L, stable
  DOM identity, scroll/focus/filter/expanded-panel preservation, stale ladder clearing and mobile layout.
  Screenshots are labelled OFFLINE TEST FIXTURE under `docs/saxo-screenshots/`.

Representative recorder test: generated five underlying streams, ten depth levels, 1,000 ms
virtual cadence, 36,005 messages over two virtual hours, three events merged into two segments.
The real recorder, compression, fsync and manifest code ran in 11.107 seconds of accelerated wall time.
[Machine-readable measurement](saxo-recorder-benchmark.json):

| Measurement | Result |
|---|---:|
| Rolling coverage | 900 seconds for every market |
| Recorder accounted memory high-water | 5,794,570 bytes (5.53 MiB), below 32 MiB |
| Whole benchmark process peak RSS | 58,867,712 bytes (56.14 MiB) |
| Event archives | 878,040 bytes (0.84 MiB), below 2 GiB |
| Ingest p99 / maximum | 0.530 / 5.263 ms |
| Maximum event-loop scheduling delay | 13.685 ms |
| Queue remaining / gaps / storage failures | 0 / 0 / 0 |

This is an accelerated simulation on the local workstation, not a live-duration soak or a compression
forecast for actual Saxo payloads. Repeating values compress well. Deployment must verify server RAM,
disk, actual granted cadence, payload fields, downtime coverage and filesystem permissions.
Reproduce with `uv run --no-sync python scripts/saxo_recorder_benchmark.py --output /tmp/saxo-benchmark.json`.
