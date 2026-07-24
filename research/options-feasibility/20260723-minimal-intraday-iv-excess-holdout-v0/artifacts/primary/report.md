# Minimal Intraday Stock → IV-Excess Holdout Validation V0

Overall decision: `blocked_quick_resource_limit`.

The frozen historical reconstruction passed: M0 reproduced predecessor G0 exactly, M1 remained exactly Group O + Group I, and the reconstructed H0 surface matched the predecessor rows, features, and weights to 1e-12.

The non-compact exact-date acquisition reached the 350,000 newly downloaded
record ceiling after 1,450 of 1,700 complete stock-session requests. The full
accounting is 349,802 records in complete receipts plus 198 records in the
incomplete page, totalling 317,272,704 bytes. It returned 288,861 exact-date
records, retained 288,698 from complete receipts, excluded the remaining 163
with the incomplete request, and rejected 61,139 extra-date records. Of those
extra-date records, 1,833 protected-date records were rejected and zero were
materialized.

The remaining 250 requests had no pre-existing complete receipts. All 80
stock-month request cells are reported, with pair coverage explicitly blocked
before selection. No partial 17-stock subgroup was modeled, no binding holdout
outcome was materialized, and no bootstrap or H0-null result was produced.

This is a retrospective research blocker, not evidence about option profitability, economic edge, direction, prospective performance, or trading utility.
