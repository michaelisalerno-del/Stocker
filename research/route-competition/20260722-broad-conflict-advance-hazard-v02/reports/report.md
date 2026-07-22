# Broad-Conflict Advance-Hazard Dense-Checkpoint Quick Screen V0.2

Primary decision: `broad_route_conflict_adds_clean_advance_warning`.

This retrospective structural screen used only frozen observable five-minute-bar, regime, behavioural, registered-loop and prefix ledgers. It did not open returns, direction, options, entries, exits, trading, execution, accounts or broker data.

## Support

- Raw assessment rows: 47,847 of 48,000 (99.6813%).
- Clean advance rows: 34,577; positives: 451; weighted base rate: 1.8365%.

## Models

- A0: log loss 0.11439166; Brier 0.02009030; AUC 0.71152054; average precision 0.04104172; top-decile precision 4.7184%.
- A1: log loss 0.11323946; Brier 0.01999503; AUC 0.75801169; average precision 0.05484722; top-decile precision 6.7432%.

A1-minus-A0 improvements: log loss 0.00115220; Brier 0.00009527; AUC 0.04649115; average precision 0.01380550.

## Frozen route states

- BROAD_CONFLICT: 6,261 rows; 221 positives; weighted completion rate 4.5187%.
- NARROWING: 743 rows; 7 positives; weighted completion rate 1.1023%.
- DOMINANT_ROUTE: 0 rows; 0 positives; weighted completion rate unsupported.
- LOW_ROUTE_SUPPORT: 10,496 rows; 40 positives; weighted completion rate 0.6457%.
- OTHER: 17,077 rows; 183 positives; weighted completion rate 1.5966%.

Independent lightweight audit passed: True.
These findings are not prospective validation and provide no evidence of economic value, directional edge, options edge, trading utility or deployability.
