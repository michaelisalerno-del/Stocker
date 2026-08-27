# Stocker Server

The deployable Stocker V2 runtime is strictly prospective-record/shadow:

```bash
uv sync --locked --no-editable --no-default-groups --group server
uv run --no-sync stocker-runtime recorder run \
  --config configs/runtime/recorder.example.json \
  --inputs configs/runtime/market-data.example.json
uv run --no-sync stocker-runtime web run \
  --config configs/runtime/web.example.json
```

The recorder and web processes are separate. Neither exposes an order path. The web
database connection is query-only, and only one recorder owns the operational writer.

See:

- `docs/architecture/stocker-platform-purpose.md`
- `docs/operations/stocker-v2-cutover.md`
- `docs/operations/ibkr-official-api-review.md`
