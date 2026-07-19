# Observable Event Cross-Sectional Ranking V1

## Safety and scope

Research and backtesting only. Execution is disabled, no live or paper order submission
is permitted, no account or position data was requested, and production runtime behavior
was not modified. No orders were sent.

## Descriptive result

- Safe pre-cutoff EODHD raw-file symbols inventoried: 43.
- Safe raw-file period: 2024-01-01 through 2025-08-23; 43 files were SHA-256 hashed
  without parsing market rows.
- Protected raw-file period: 2025-08-23 through 2026-06-30; all 43 files remained
  unopened, as did protected processed Parquet data.
- Existing source-audit metadata: 82 five-minute audits and 42 vendor-QA reports.
- Protected source files opened: 0.
- Event rows: 0.
- Supported slates: 0.
- Point-in-time sector membership: unavailable.

No event frequency or stock, sector, or clock distribution can be estimated because the
required effective-dated sector ledger is absent. The available current screener sector
strings are not projected backward.

## Structural result

Targets and models were not permitted to run. There is no candidate-versus-baseline IC,
top-two-minus-median result, uncertainty interval, or stability result. A positive IC, if
later observed, would establish only structural rank information and not an executable edge.

## Directional interpretation

No future-return rank was constructed. Accordingly there is no directional prediction
evidence in this run.

## Economic interpretation

No gross payoff or transaction-cost result exists. Provider bar prices are structural
references, not achieved IBKR fills, and EODHD volume is only provider-reported activity
proxy data.

## Executability and IBKR

The read-only protocol, fake client, append-only quote ledger, schemas, classifications,
throttling configuration, contract ledger, and observation plan are implemented. The
official IBKR API transport is blocked because IBKR distributes it through its official
ZIP/MSI rather than a supported Python package index dependency. No TWS/IB Gateway
connection or subscription check was attempted. Delayed and frozen observations are
explicitly non-executable; a recorded bid and ask would bound a reference quote, never
prove a fill.

## Scientific decision

- Decision: `blocked_missing_point_in_time_sector_membership`.
- Support gate passed: `false`.
- Targets permitted: `false`.
- Models permitted: `false`.
- Exact rerun: `true`.
- Independent audit: `true`.

### Gate values

- `bar_label_convention_proven`: `False`
- `corporate_action_handling_resolved`: `False`
- `point_in_time_sector_membership`: `False`
- `safe_source_symbol_count`: `43`
- `static_provenance_audit`: `True`
- `valid_slate_minimum_universe`: `50`

This is a successful fail-closed implementation outcome. A genuinely new run would need a
larger pre-cutoff source universe, trusted point-in-time sector membership, proven bar-time
semantics, and resolved corporate-action handling under the unchanged contract.
