# Dynamic loop temporary payoff edge state V2

## Baseline
- Branch: `agent/slrno-research-handoff`
- Commit: `8baf974f2d13751064dbc4d2c7cf65d02e3a8912`
- Evaluator: frozen V1 exact runner; no `tools/arbor_template_discovery_eval.py` or `backend/tests/v2` exists in this repository
- Baseline result: exact V1 summary reproduced byte-for-byte. At 24 bars V1 mean net payoff was -0.01 bps in 2025 and +1.25 bps in backward-2023; the overall V1 hypothesis was rejected.

## Hypotheses
1. Payoff-history BOCPD can reduce activation/termination lag versus V1's 60-session selector. Targets: `online_state.py`, V2 runner. Benefit: useful causal state transitions. Safety risk: hindsight episode labels entering the filter. Validation: synthetic shift tests plus prequential delay metrics. Stop: reject if lag ratio or calibration does not improve without hindsight inputs.
2. A shared online payoff environment can improve sparse loop/orientation uncertainty only when population evidence is directionally relevant. Targets: `online_state.py`. Benefit: practical partial pooling without fixed pseudocount dominance. Safety risk: unrelated loops contaminating a cell. Validation: positive/negative shared-environment unit test and leave-one-stock-out stress. Stop: reject pooling if sparse forecasts move the same way under opposing shared evidence or concentration worsens.
3. Compact lagged breadth/coherence features can improve economic-state calibration over payoff-only BOCPD. Targets: V2 feature adapter and walk-forward runner. Benefit: onset/termination lead information. Safety risk: current-session or outcome leakage and duplicated raw-history/loop-score inputs. Validation: timestamp tests, appended-future invariance, Brier/log-loss/delay comparison. Stop: reject the feature increment if full hierarchy does not improve the frozen metrics.
4. Equal-stock robust session aggregation can prevent correlated fill clusters and single outliers from creating false support or activation. Targets: `session_payoff.py`. Benefit: correct statistical unit and stable Student-t updates. Safety risk: hiding true tail losses. Validation: fill-cluster, independent-stock, full-cost, outlier, and median sensitivity tests. Stop: reject if outlier protection creates persistent activation or alternative aggregation reverses the conclusion.

## Changes
- Frozen V2 contract added before final scoring.
- Session aggregation, Student-t BOCPD, empirical-Bayes hierarchy, state classification, and causal walk-forward seams implemented test-first.
- A hash-verified recovery adapter reconstructs the 250-session causal anchor context from sealed derived artifacts because the original ephemeral V1 anchor/provider inputs expired. It proved exact top-loop/probability/state/history-token equality on all 10,382 V1 scored h24 rows before V2 scoring.
- Four frozen selectors, calibration/change/episode diagnostics, admission decisions, concentration slices, bounded stresses, plots, metadata, manifests, and the concise report were generated.

## Validation
- Frozen V1 exact runner: passed; summary SHA-256 matched archived exact rerun.
- Focused V2 suite: `28 passed`.
- Full repository suite: `397 passed`; four pre-existing NumPy `Mean of empty slice` warnings in `test_behavioral_state_similarity.py`.
- Scoped Ruff lint/format: passed for all 10 V2 source/test files.
- Strict mypy: passed for all 5 reusable V2 module files.
- Real-run causal audit: zero training-availability, feature-availability, prediction-freeze, or admission-join violations.
- Final artifact manifest: 18 files verified with zero SHA-256 mismatches.
- Repository-wide Ruff remains red on 1,153 pre-existing errors outside the V2 change; no unrelated cleanup was attempted.
- `python3 tools/arbor_template_discovery_eval.py`: unavailable in this repository.
- `python3 -m pytest backend/tests`: not applicable; this repository uses root `tests/`.
- `git diff --check`: passed.

## Decision
- Kept: the modular causal implementation, explicit abstention, robust statistical unit, payoff-only comparator, and immutable h24 evaluation artifacts.
- Rejected: the registered full breadth/coherence hierarchy as an economic gate. It lost 6,056.59 bps after costs, had a 0.4899 detection-lag ratio, failed twice-cost and leave-one-stock-out stresses, and underperformed payoff-only.
- Safety result: research-only boundary held; no broker, paper/demo, deployment, application position, or frozen-exit code changed.
- Remaining work: sealed prospective logging only. Do not tune this opened 2023/2025 surface. The next experiment is an immutable prospective payoff-only versus breadth/coherence comparison with better independent support.
