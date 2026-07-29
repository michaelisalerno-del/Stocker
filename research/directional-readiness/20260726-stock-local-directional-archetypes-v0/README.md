# Stock-Local Directional Archetype Screen V0

This experiment retrospectively tests three directional mechanisms after a
causal M1-qualified underlying-stock movement signal:

1. continuation and level acceptance (`C1`);
2. bar-derived attempted-movement failure and reversal (`A1`);
3. stock-specific beta-adjusted relative strength (`R1`).

The three feature bundles and models remain separate for primary inference.
Every direction feature ends at the close of completed bar `T-1`; trigger bar
`T` is excluded. Normalisation is fitted from each stock's 2024 history, with
no contemporaneous peer-stock slate. Archived signed pressure and every
future-filtered or otherwise peer-normalised descendant are excluded.

Because archived M1 used contaminated features, Phase 0 fits `M1C` from
unchanged Group O plus only causally valid Group I inputs. Its preprocessing,
coefficients, and weighted 2024 95th-percentile threshold are frozen before
2025 scoring.

Run the screen and its separate auditor with the repository environment:

```bash
uv run python research/directional-readiness/20260726-stock-local-directional-archetypes-v0/run_screen_v0.py --run
uv run python research/directional-readiness/20260726-stock-local-directional-archetypes-v0/audit_screen_v0.py --audit
```

The runner defaults to the frozen local predecessor artifacts used for this
research snapshot. On another checkout, point it at byte-identical inputs with:

- `STOCKER_ARCHETYPE_DENSE_CAUSAL_PATH`
- `STOCKER_ARCHETYPE_DENSE_MODEL_CONFIG_PATH`
- `STOCKER_ARCHETYPE_HISTORICAL_OPTIONS_PATH`
- `STOCKER_ARCHETYPE_STATE_PATH`
- `STOCKER_ARCHETYPE_STRESS_OPTIONS_PATH`
- `STOCKER_ARCHETYPE_ARCHIVED_EPISODES_PATH`

The independent auditor uses the dense-causal, historical-options, and state
overrides. Input hashes and roles are frozen in `source_manifest.json`; a hash
mismatch fails the audit.

Primary artifacts and the report are under `artifacts/primary/`. This is
retrospective directional candidate research on underlying-stock returns. It
is not direct order-flow measurement, institutional-activity observation,
option P&L, prospective validation, paper/live readiness, or a trading
strategy.
