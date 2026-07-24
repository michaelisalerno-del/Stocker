# Stocker Server

The original `run_executor.py` remains a legacy dry-run scaffold. The first
deployable prospective slice now lives in `stocker_prospective` and is strictly
record-only/shadow:

```bash
uv sync --locked --no-default-groups --group server
uv run stocker-prospective replay run \
  --config configs/prospective/replay.example.yaml
uv run stocker-prospective web run \
  --config configs/prospective/replay.example.yaml
```

The recorder and web process are separate. Neither exposes a paper/live order
path. Real frozen M1 scoring remains blocked until an approved serialized
bundle and feature-parity gate exist.

See:

- `docs/architecture/prospective-evidence-recorder.md`
- `docs/operations/prospective-server-runbook.md`
- `docs/operations/ibkr-official-api-review.md`
