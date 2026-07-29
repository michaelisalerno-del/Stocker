# Frozen loop-movement shadow harness

## Outcome

The prospective movement contract is frozen and the offline shadow harness is
initialized. The durable runtime is
`work/shadow_validation/frozen_loop_movement_shadow_v1`. Its prediction ledger
is empty: zero post-freeze predictions and zero post-freeze outcomes have been
opened.

The original detector workspace remains unchanged. This continuation workspace
contains the prospective contract, inference code, tests, report, and a durable
copy of the small frozen model/provenance artifacts that previously existed
only under `/private/tmp`.

## Files

- Machine contract:
  `work/contracts/20260710-frozen-loop-movement-shadow-v1.json`
- Freeze manifest:
  `work/contracts/20260710-frozen-loop-movement-shadow-v1-manifest.json`
- Frozen inference core: `work/frozen_loop_movement_shadow_core.py`
- Two-stage harness: `work/run_frozen_loop_movement_shadow.py`
- Guardrail tests: `work/tests/test_frozen_loop_movement_shadow.py`
- Contract rationale:
  `work/reports/20260710-frozen-loop-movement-prospective-shadow-contract.md`

## Operating protocol

Run `issue` once after each completed five-minute provider bar that may contain
a new state-run entry. The command must begin at least five minutes and finish
before thirty minutes after the provider timestamp. It records nothing when no
frozen state run began at that exact bar.

```bash
rtk python3 work/run_frozen_loop_movement_shadow.py issue --as-of "$BAR_TIMESTAMP_UTC"
```

The provider snapshot is filtered at `--as-of` before any feature is
calculated. A second wall-clock check occurs immediately before the batch is
sealed, so a slow calculation cannot be backdated past the six-bar outcome.

Support can be inspected without provider or outcome access:

```bash
rtk python3 work/run_frozen_loop_movement_shadow.py status
```

Do not inspect movement results while support accumulates. The evaluator
enforces that rule: before the first qualifying ledger prefix, it exits before
resolving or reading the provider root. Once the prefix exists, it waits until
the closing cohort is mature, performs a timestamp-only exact-support audit,
and only then may load OHLC outcomes.

```bash
rtk python3 work/run_frozen_loop_movement_shadow.py evaluate
```

The primary cohort is the first hash-chained ledger prefix with at least 65,000
issued anchors, 200 session dates, 18 stocks, four quarters, and all eight
states. Evaluation additionally requires 60,000 anchors with exact outcomes at
all three horizons. The first mature timestamp-support decision is sealed.

## Frozen decision

The only primary comparison is frozen `loop_scores` versus frozen
`state_context` for absolute return and future range at 6, 12, and 24 bars.
Direction, signed return, raw-history model selection, quantile development,
P&L, costs, and trading rules are excluded.

Absolute-return MSE improvement must be at least 1% pooled, at every horizon,
and under every stock deletion. Future-range MSE improvement must be at least
3% under the same slices. Both targets also require negative daily
moving-block upper bounds for MSE and MAE differences, lower MSE and MAE in
every quarter, and better correlation at every horizon. Both targets must pass.

## Validation evidence

- Original frozen price runner self-test: passed.
- Python compilation for the new core, runner, and tests: passed.
- Four outcome-embargo/ledger unit tests: passed.
- Frozen 2025 inference reproduction: 70,883 anchors; maximum loop-score error
  `0.0`; maximum movement-prediction error `0.0`. No outcome, direction, or
  signed-return column was read.
- Full causal replay for 30 December 2025: 1,716 bar-level state/age rows and
  362 eligible run entries; zero state, age, previous-state, control,
  loop-score, or movement-prediction mismatch. No outcome, direction, or
  signed-return column was read.
- Empty-ledger embargo test with a deliberately invalid provider path: exited
  with the support-embargo status before provider access.
- Durable initial status: zero batches, zero anchors, no outcome data read, no
  performance metric calculated.

PyArrow emitted harmless sandbox CPU-cache `sysctlbyname` warnings during
tests; every command completed successfully.

## Safety

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`. No API key is stored. Provider volume is labelled
`historical_volume`; it is not exchange-wide volume or order flow. Historical
volume remains an input to the frozen state detector, while the movement
regressions do not use volume directly. No deployment or strategy promotion
occurred.
