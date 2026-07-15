# SLRNO V2 research handoff

This directory is the complete research-only SLRNO V2 investigation handoff as of 2026-07-14.

## Start here

Read these reports in order:

1. [`work/reports/20260714-dynamic-loop-edge-state-v2.md`](work/reports/20260714-dynamic-loop-edge-state-v2.md)
2. [`work/reports/20260713-dynamic-loop-context-edge-v1.md`](work/reports/20260713-dynamic-loop-context-edge-v1.md)
3. [`work/reports/20260714-selective-payoff-equations-v1.md`](work/reports/20260714-selective-payoff-equations-v1.md)
4. [`work/reports/20260714-long-short-neutral-detector-v1.md`](work/reports/20260714-long-short-neutral-detector-v1.md)

The latest conclusion is that temporary payoff states are visible descriptively, but the registered full hierarchical breadth/coherence model is rejected: it was less calibrated, slower, and materially worse after costs than both V1 and the payoff-only change-point model. The payoff-only model's positive retrospective P&L is not prospective validation and was not robust enough to approve. The exact rerun and 48-check independent audit are stored at:

- [`work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun/`](work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun/)
- [`work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun/independent_audit.json`](work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun/independent_audit.json)

The highest-value next experiment is a sealed prospective, no-execution comparison of the full model against the identical hierarchical model with leading features disabled. Do not retune on the opened 2023/2025 surfaces.

## Included material

- frozen contracts and prospective logging specifications;
- Markdown reports and discovery handoffs;
- research runners, independent auditors, and focused tests;
- outcome-free frozen ledgers;
- primary and exact-rerun artifacts;
- model, bootstrap, path, cost, stability, and audit outputs.

## Scientific and safety boundary

- Research only.
- The source periods have already been opened; retrospective results are development evidence, not validation.
- No strategy in this directory is approved for promotion.
- Live, paper, demo, broker, deployment, position, and order functionality remain disabled and out of scope.
- Historical provider volume is an activity proxy, not quote flow or order-book volume.

Independent audits and exact reruns are stored beside their corresponding primary artifacts. Consult each report for the applicable hashes, qualification gates, failures, and retired paths.
