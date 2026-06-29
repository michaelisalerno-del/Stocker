# Stocker MCP Server

Stocker includes a local read-only MCP server for AI clients that need narrow access to
Stocker code, StockerLocal research reports, and local Stocker database summaries.
It is infrastructure only. It does not place orders, connect to brokers, fetch vendor
data, run research scans, or expose arbitrary filesystem access.

## Roots

The server resolves two allowed roots:

- Repo root: the current Stocker project directory containing `pyproject.toml` and
  `packages/`.
- Stocker workspace: `STOCKER_HOME`, or `~/StockerLocal` when the variable is unset.

Report tools search these roots, in order:

- `$STOCKER_HOME/data/reports/research`
- `data/reports/research` inside the repo, when present

Database tools are scoped to:

- `$STOCKER_HOME/db`

## Launch

```bash
export STOCKER_HOME="$HOME/StockerLocal"
uv run stocker-mcp --help
uv run stocker-mcp doctor
uv run stocker-mcp
```

`uv run stocker-mcp` starts the MCP server over stdio. No public HTTP listener is
enabled.

## Codex Registration

This machine uses Codex TOML MCP config at:

```text
~/.codex/config.toml
```

Use this local stdio entry:

```toml
[mcp_servers.stocker]
command = "uv"
args = ["run", "stocker-mcp"]
cwd = "/Users/michaelsalerno/Documents/Codex/2026-06-29-we-are-working-in-my-stocker"
startup_timeout_sec = 120

[mcp_servers.stocker.env]
STOCKER_HOME = "/Users/michaelsalerno/StockerLocal"
```

Do not add API keys or broker settings to the MCP server environment.

## Tools

Code and git tools:

- `get_repo_info`
- `git_status`
- `git_log`
- `git_current_commit`
- `list_files`
- `search_code`
- `read_code_file`
- `git_diff`

Workspace and report tools:

- `workspace_doctor`
- `list_recent_research_runs`
- `get_latest_universe_run`
- `read_universe_run`
- `summarise_universe_run`
- `list_symbol_reports`
- `read_symbol_report`
- `compare_universe_runs`
- `find_candidate_symbols`
- `find_interesting_symbols`
- `filter_symbol_results`

Database tools:

- `db_list_databases`
- `db_list_tables`
- `db_describe_table`
- `db_preview_table`
- `db_get_symbol_bars`
- `db_get_latest_catalysts`
- `db_get_trade_attribution`
- `db_select`

Diagnostics:

- `export_diagnostics_zip`

Resources:

- `stocker://workspace/doctor`
- `stocker://reports/latest`
- `stocker://repo/status`

Prompts:

- `summarise_latest_stocker_scan`
- `compare_two_stocker_universe_runs`
- `investigate_symbol_gate_failure`

## Security

The server is read-only except for a local redacted audit log and diagnostics zip
exports under `$STOCKER_HOME/exports`.

It blocks:

- Paths outside the allowed roots
- Path traversal
- Symlink escapes
- `.env`, `.env.*`, private keys, SSH keys, credential, token, and API-key-like files
- `.git`, virtualenvs, caches, and local tool runtime directories
- Arbitrary shell commands
- File write, patch, delete, rename, and move operations
- Broker, IBKR, order, live execution, paper execution, and vendor-fetch tools

Text output is bounded and redacted for secret-looking values such as API keys,
tokens, bearer headers, passwords, and credential-like database columns.

`db_select` is optional and restricted:

- SELECT only
- No semicolons
- No `ATTACH`, `DETACH`, `COPY`, `EXPORT`, `INSTALL`, `LOAD`, `PRAGMA`, `CREATE`,
  `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, or similar statements
- No file-reading functions such as `read_csv` or `read_parquet`
- Forced row limit
- Read-only SQLite and DuckDB connections where supported

Disable DB querying with:

```bash
export STOCKER_MCP_DISABLE_DB=1
uv run stocker-mcp
```

## Diagnostics Export

Create a redacted zip suitable for uploading to ChatGPT:

```bash
export STOCKER_HOME="$HOME/StockerLocal"
uv run stocker-mcp doctor
```

From an MCP client, call:

```text
export_diagnostics_zip()
```

The zip includes workspace diagnostics, git status, recent git log, latest universe
run summaries, selected report JSON/Markdown files, DB schema summaries, and safe
config examples. It excludes `.env`, API keys, database files, parquet/raw vendor
payloads, broker state, and huge data files.

## Future Tools

Add future MCP tools only through the central security context. New tools should have
focused tests first, avoid broker/order/vendor-fetch behavior unless explicitly
approved, and return summaries or bounded slices instead of raw dumps.
