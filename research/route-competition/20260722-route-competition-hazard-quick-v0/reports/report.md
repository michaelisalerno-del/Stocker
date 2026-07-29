# Route-Competition Completion-Hazard Quick Screen V0

Retrospective, observable, structural, research-only quick screen. No economic, directional, execution, broker, or strategy-promotion outcome was opened.

Primary decision: `blocked_insufficient_support`.

## Support

Assessment rows: 25518; sessions: 160; stocks: 20; months: 8; positive outcomes: 1139.
Feature retention: 0.966178. The preregistered 30,000-row assessment gate is not met.
Maximum weighted target-class share: 0.955335; maximum weighted stock share: 0.050181.

## Pooled models

- H0: log loss 0.165773; Brier 0.040860; AUC 0.753346; average precision 0.135933; top-decile precision 0.157199; top-quintile precision 0.126570.
- H1: log loss 0.142655; Brier 0.037392; AUC 0.862589; average precision 0.251224; top-decile precision 0.259194; top-quintile precision 0.176264.

H1-minus-H0 increments (positive means improvement): auc_improvement=0.10924314, average_precision_improvement=0.11529093, brier_improvement=0.00346792, log_loss_improvement=0.02311779, top_decile_precision_improvement=0.10199472.

## Route-resolution states

- BROAD_CONFLICT: rows 3712, three-bar completion rate 0.050135, supported=True.
- NARROWING: rows 684, three-bar completion rate 0.095005, supported=True.
- DOMINANT_ROUTE: rows 908, three-bar completion rate 0.104708, supported=True.
- LOW_ROUTE_SUPPORT: rows 5598, three-bar completion rate 0.003385, supported=True.
- OTHER: rows 14616, three-bar completion rate 0.052990, supported=True.
Positive log-loss months: 8; materially adverse checkpoints: 0.
Candidate-count change alone: decreasing 0.037152, unchanged 0.028783, increasing 0.093499.

## Route coefficients

active_prefix_count=0.105087, active_prefix_family_count=0.099986, top_prefix_depth_fraction=0.076845, second_prefix_depth_fraction=0.074097, top_minus_second_prefix_depth=0.042689, prefix_family_entropy=0.323426, orientation_disagreement_fraction=0.396702, new_prefixes_last_1_bar=-0.033163, invalidated_prefixes_last_1_bar=0.084601, active_prefix_count_change_last_1_bar=-0.104257, active_prefix_count_change_last_3_bars=-0.039777, top_prefix_depth_change_last_1_bar=0.033898, top_prefix_depth_change_last_3_bars=0.054721, matching_recent_loop_prefix_count=-0.091584, recent_loop_memory_weighted_top_depth=0.208128

## Resampling

- Bootstrap log_loss_improvement 80%: [0.02199069, 0.02427180].
- Bootstrap log_loss_improvement 90%: [0.02164639, 0.02438131].
- Bootstrap log_loss_improvement 95%: [0.02135892, 0.02446102].
- Bootstrap brier_improvement 80%: [0.00307036, 0.00364421].
- Bootstrap brier_improvement 90%: [0.00299102, 0.00369770].
- Bootstrap brier_improvement 95%: [0.00292194, 0.00370938].
- Bootstrap auc_improvement 80%: [0.10222525, 0.11759281].
- Bootstrap auc_improvement 90%: [0.10191582, 0.12119095].
- Bootstrap auc_improvement 95%: [0.10178430, 0.12408834].
- Bootstrap average_precision_improvement 80%: [0.10128080, 0.12031132].
- Bootstrap average_precision_improvement 90%: [0.09553978, 0.12154368].
- Bootstrap average_precision_improvement 95%: [0.09399224, 0.12204530].
- Bootstrap top_decile_precision_improvement 80%: [0.08327584, 0.10375061].
- Bootstrap top_decile_precision_improvement 90%: [0.07877100, 0.10498765].
- Bootstrap top_decile_precision_improvement 95%: [0.07833882, 0.10570764].
- Real log_loss_improvement exceeded 3 of 3 route-bundle null increments.
- Real brier_improvement exceeded 3 of 3 route-bundle null increments.
- Real auc_improvement exceeded 3 of 3 route-bundle null increments.

Fast determinism passed=True; maximum probability difference 0; row mismatches 0; feature hash match=True.
Independent artifact audit passed=True; maximum route-feature difference 0; maximum bootstrap difference 6.11e-16.

This is not prospective validation and provides no evidence of economic value, directional edge, trading utility, or deployability.
