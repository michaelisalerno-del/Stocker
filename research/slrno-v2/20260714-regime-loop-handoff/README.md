# SLRNO V2 research handoff

This directory is the complete research-only SLRNO V2 investigation handoff as of 2026-07-14.

## Start here

Read these reports in order:

1. [`work/reports/20260713-dynamic-loop-context-edge-v1.md`](work/reports/20260713-dynamic-loop-context-edge-v1.md)
2. [`work/reports/20260714-selective-payoff-equations-v1.md`](work/reports/20260714-selective-payoff-equations-v1.md)
3. [`work/reports/20260714-long-short-neutral-detector-v1.md`](work/reports/20260714-long-short-neutral-detector-v1.md)

The latest conclusion is that long, short, and neutral are useful first-class research states, but the tested one-shot static detector did not identify direction profitably. The next prospective-only specification is:

- [`work/contracts/20260714-dynamic-long-short-neutral-prospective-log-v1.json`](work/contracts/20260714-dynamic-long-short-neutral-prospective-log-v1.json)

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
