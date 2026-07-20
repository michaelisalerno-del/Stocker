# Immediate Regime-Pair Closure History Diagnostic V1

Safety boundary: `research_only=True`, `execution_enabled=False`, `order_placement=disabled`, `live_ordering_enabled=False`, `strategy_promotion=False`.

## Scope

Fixed-model structural sensitivity only. The experiment asks whether a five-state context (the current run plus four preceding runs) adds information beyond the immediate A→B pair for predicting the next-state A→B→A closure. It uses no future price, return, payoff, spread, broker, order, position, or 2026 row.

## Scientific limitation

The repaired regime representation remains semantically unstable across refits. Numeric pair identities in this report are valid only inside the hash-bound fitted model and are non-promotable. Blocked Part B was not reopened.

## Population

- Development 2024 pair decisions: 177,096 across 22 stocks.
- Unchanged assessment 2025 pair decisions: 180,313 across 22 stocks.
- Primary representation: causal hysteretic semantic labels; causal hard labels are a required sensitivity.
- Volume provenance: provider-reported EODHD historical activity only; volume was not a model input to this diagnostic.

## Primary result

- M5-minus-M2 assessment log-loss improvement: 0.00555976.
- Paired session-block 95% interval: [0.00460326, 0.00657491].
- M5-minus-M2 Brier improvement: 0.00248116.
- Primary development-selected supported pair orientations: 36.
- Primary same-direction, BH-significant 2025 replications: 32.

## Decision

`fixed_model_history_increment_observed_nonpromotable`

This is evidence about fixed-model structural predictability only. It is not directional price evidence, economic payoff evidence, executable edge evidence, or authorization for another loop dictionary, strategy, or economic test.

## Reproducibility

- Exact rerun byte-identical: `True`.
- Independent audit passed: `True`.
- No orders, broker connections, positions, accounts, or execution interfaces were used.
