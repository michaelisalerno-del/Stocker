# Observable Extreme-Tail Cross-Stock Replication V1

## Decision

`blocked_no_clean_cross_stock_holdout_remaining`

The mandatory stock outcome-exposure gate stopped the experiment before model
reconstruction and before any assessment market outcome was opened. The actual safe
EODHD stock manifest contains 43 symbols;
41 are outcome-exposed, 1 are
unknown-assume-exposed, and only 1 is
machine-evidenced as outcome-unexposed. The contract requires at least 15.

Outcome-unexposed symbols: CRCL.

QA-eligible cross-stock assessment symbols: none. The clean symbol is excluded
when its existing vendor QA or bar audit has not passed.

## Boundary

No raw market row was parsed or materialised. Assessment and development model rows
are both zero, the minimum/maximum market timestamps read are null, protected files
touched are zero, and `protected_rows_materialised=0`.

## Scientific scope

This is a retrospective, research-only blocker result. No candidate prediction,
admission threshold, slate, delayed return, baseline, bootstrap, permutation null,
execution result, or deployable edge was calculated. The predecessor was used only
to establish model/outcome exposure of its development stocks; reconstruction was
not reached because Phase A failed first.
