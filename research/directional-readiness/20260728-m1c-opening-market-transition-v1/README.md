# M1C Opening Market Transition V1

This is one preregistered, research-only retrospective experiment for the
checkpoint-6 fresh frozen-M1C `FIRST_ENTRY` population. It uses the canonical
five-minute EODHD VTI series and one fixed same-session window from the NYSE
regular-session open through the final complete bar before the 10:00 ET entry.

The three mechanisms remain separate:

1. follow the severe signed VTI opening transition;
2. continue when the stock amplified that transition;
3. reverse when the stock resisted that transition.

The runner reads only sessions before 2026, freezes thresholds from 2024
predictors, applies them unchanged to the already-opened 2025 assessment and
stress periods, and writes all results beneath `artifacts/primary` and
`reports`. It never accesses a broker or an order path.

Run:

```bash
rtk uv run python research/directional-readiness/20260728-m1c-opening-market-transition-v1/run_experiment.py
```
