# Stocker Desktop

The desktop workspace is for heavy research on macOS: data audits, notebooks,
baseline tests, statistical hypothesis work, backtests, plots, and reports.

It should not contain broker credentials or live order placement code.

Run:

```bash
uv sync --all-groups
uv run python apps/desktop/scripts/run_lab.py
uv run jupyter lab apps/desktop/notebooks
```
