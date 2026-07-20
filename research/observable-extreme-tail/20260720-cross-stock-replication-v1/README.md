# Observable Extreme-Tail Cross-Stock Replication V1

This is a bounded, retrospective, research-only cross-stock and forward-time
replication of the frozen observable-only candidate from Movement-Conditioned
Regime-Path Probability Chain V0. It cannot place orders, modify production runtime,
or support a claim of achievable execution or deployable net edge.

The chronology is intentionally split. Phase A freezes the stock outcome-exposure
ledger before any assessment outcome is opened:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/observable-extreme-tail/20260720-cross-stock-replication-v1/prepare_freeze.py
```

The assessment command consumes that freeze and fails closed on any preregistered
blocker:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/observable-extreme-tail/20260720-cross-stock-replication-v1/run_replication.py \
  --audit --exact-rerun
```

Both commands accept `--output`. The assessment command also accepts `--audit` and
`--exact-rerun`. `--max-symbols` and `--max-sessions` are only for non-scientific
smoke checks; using either stamps every generated decision artifact with
`non_scientific_smoke_test=true`.

The scientific run stopped at the mandatory Phase A gate because fewer than 15
machine-evidenced, genuinely outcome-unexposed stocks remained. Consequently no 2025
assessment market row was materialised and no model, score, admission threshold,
selection, return, baseline, bootstrap, or permutation outcome was calculated.
