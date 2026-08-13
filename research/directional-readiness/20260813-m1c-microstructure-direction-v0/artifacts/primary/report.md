# M1C Microstructure Direction V0

RESEARCH ONLY — MICROSTRUCTURE DIRECTION V0 — NOT VALIDATED — NO RECOMMENDATION

## Result

`D_INSUFFICIENT_PROSPECTIVE_MICROSTRUCTURE_DATA`

The read-only census found **0** recorded M1C episodes and
**0** episodes with linked microstructure summaries. No
development/assessment split or directional performance claim is possible.

All M01–M06 definitions, nine causal timing windows, and +5/+10/+15 minute
underlying horizons are emitted in the CSV schemas, but metric cells remain
unestimated rather than being imputed.

The runtime capacity manifest reports 2
available tick-by-tick subscriptions, supporting
1 paired BidAsk + Last
high-resolution underlying. No capacity or Level II configuration change is recommended.

## Reproduce this zero-data assessment

```bash
PYTHONPATH=packages/stocker_prospective/src python3 -m stocker_prospective.microstructure_direction_v0_research --census-json research/directional-readiness/20260813-m1c-microstructure-direction-v0/source_census.json --output research/directional-readiness/20260813-m1c-microstructure-direction-v0/artifacts/primary
```
