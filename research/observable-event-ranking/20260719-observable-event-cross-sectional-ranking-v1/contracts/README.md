# Contract sources

The canonical frozen experiment contract is code-defined in
`stocker_research.observable_event_ranking_v1.contract` so tests and every stage consume
the same values. A run writes its canonical JSON representation to
`work/artifacts/<run-kind>/frozen_experiment_contract.json` and binds every scientific
artifact to that contract hash.

The contract freezes the safety flags, development cutoff, universe rules, event family,
decision grid, feature surface, target timing, baselines, M1 configuration, bootstrap,
support/development/prospective gates, retired-input vocabulary, and source semantics.
Changing it is a new experiment, not a rerun of V1.
