"use strict";

const main = document.querySelector("#main");
const fmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 2 });
let timer;
let lastOutcome = null;
let runStartStatus = { status: "IDLE" };
let pageHasRendered = false;
const INTERACTIVE_ROUTES = new Set(["universes", "candidates", "trades", "settings"]);

const RUN_COLUMNS = [
  { key: "display_name", label: "Run" },
  { label: "Status", render: (row) => badge(row.status) },
  { key: "candidate_count", label: "Watch", numeric: true },
  { key: "signals_today", label: "Signals", numeric: true },
  { key: "open_positions", label: "Positions", numeric: true },
  { label: "Today realised", numeric: true, render: (row) => money(row.today_realised_pnl, row.currency) },
  { label: "Unrealised", numeric: true, render: (row) => row.unrealised_status === "AVAILABLE" ? money(row.unrealised_pnl, row.currency) : esc(row.unrealised_status) },
  { label: "20-session R", numeric: true, render: (row) => row.total_r == null ? "—" : `${number(row.total_r)}R` },
];

function esc(value) {
  return String(value ?? "—").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
function number(value) { return value == null ? "—" : fmt.format(value); }
function money(value, currency = "USD") {
  if (value == null) return "—";
  currency ||= "USD";
  try { return new Intl.NumberFormat("en-GB", { style: "currency", currency, maximumFractionDigits: 2 }).format(value); }
  catch (_) { return `${value < 0 ? "−" : ""}${esc(currency)} ${fmt.format(Math.abs(value))}`; }
}
function badge(value, kind = "") { return `<span class="badge ${esc((kind || value || "").toLowerCase())}">${esc(value)}</span>`; }
function environment(value) { return badge(value, value === "LIVE" ? "live" : "paper"); }
function head(title, note) { return `<header class="page-head"><div><span class="kicker">STOCKER / OPERATIONS</span><h1>${esc(title)}</h1></div><p>${esc(note)}</p></header>`; }
function table(columns, rows, action) {
  if (!rows.length) return `<div class="empty">No records in this view.</div>`;
  const headers = columns.map((column) => `<th class="${column.numeric ? "num" : ""}">${esc(column.label)}</th>`).join("");
  const body = rows.map((row) => `<tr ${action ? `data-action="${esc(action(row))}"` : ""}>${columns.map((column) => `<td class="${column.numeric ? "num" : ""}">${column.render ? column.render(row) : esc(row[column.key])}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table></div>`;
}
async function api(path, options) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.json();
}
function route() {
  const name = location.pathname.split("/").filter(Boolean)[0] || "overview";
  return ["overview", "runs", "universes", "candidates", "orders", "positions", "trades", "system", "settings"].includes(name) ? name : "overview";
}
function activate(name) {
  document.querySelectorAll("[data-nav]").forEach((link) => link.classList.toggle("active", link.dataset.nav === name));
}
async function refreshHeader() {
  const data = await api("/api/overview");
  const status = (name) => data.environments[name];
  for (const name of ["PAPER", "LIVE"]) {
    const node = document.querySelector(`#${name.toLowerCase()}-status`);
    const item = status(name);
    node.textContent = item ? (item.connected ? "CONNECTED" : "DISCONNECTED") : "NOT CONFIGURED";
  }
  document.querySelector("#system-status").textContent = data.system;
  document.querySelector("#active-runs").textContent = data.active_runs;
  document.querySelector("#open-positions").textContent = data.open_positions;
  document.querySelector("#last-refresh").textContent = new Date(data.as_of).toLocaleTimeString();
  return data;
}
function showRefreshWarning() {
  let warning = main.querySelector("#refresh-warning");
  if (!warning) {
    warning = document.createElement("p");
    warning.id = "refresh-warning";
    warning.className = "notice";
    warning.setAttribute("role", "status");
    main.prepend(warning);
  }
  warning.textContent = "Dashboard update delayed. Showing last known data; retrying automatically.";
  document.querySelector("#last-refresh").textContent = "RETRYING";
}
function clearRefreshWarning() {
  main.querySelector("#refresh-warning")?.remove();
}

async function overview(cached) {
  const data = cached || await api("/api/overview");
  const envCard = (name) => {
    const item = data.environments[name];
    return `<article class="status-card"><span class="eyebrow">IBKR ${name}</span><strong>${item ? (item.connected ? "CONNECTED" : "DISCONNECTED") : "NOT CONFIGURED"}</strong><small>${item ? `${esc(item.account)} · ${item.reconciled ? "RECONCILED" : "NOT RECONCILED"}<br>Net liquidation ${money(item.equity)} · Buying power ${money(item.buying_power)}` : "No runtime destination"}</small></article>`;
  };
  const positionColumns = [
    { key: "symbol", label: "Symbol" }, { key: "run_id", label: "Run" }, { label: "Env", render: (row) => environment(row.environment) },
    { key: "side", label: "Side" }, { key: "quantity", label: "Qty", numeric: true }, { label: "Entry", numeric: true, render: (row) => number(row.average_entry) },
    { label: "P/L", numeric: true, render: (row) => money(row.unrealised_pnl) },
  ];
  const metrics = Object.entries(data.today).map(([key, value]) => `<div class="metric"><span>${esc(key.replaceAll("_", " "))}</span><strong>${number(value)}</strong></div>`).join("");
  const attention = data.attention.length ? data.attention.map((item) => `<div class="attention-item"><b>${esc(item.scope)}</b><span>${esc(item.message)}</span></div>`).join("") : `<div class="empty">No current operational issues.</div>`;
  main.innerHTML = `${head("Overview", "Current PAPER, LIVE, run, and broker-authoritative exposure at a glance.")}
    <section class="status-grid">${envCard("PAPER")}${envCard("LIVE")}<article class="status-card"><span class="eyebrow">SYSTEM</span><strong>${esc(data.system)}</strong><small>${data.active_runs} active runs · ${data.open_positions} open positions</small></article></section>
    <section class="environment-section paper-zone"><div class="section-head"><h2>PAPER RUNS</h2><span class="muted">Independent simulated execution</span></div>${table(RUN_COLUMNS, data.runs.filter((row) => row.environment === "PAPER"), (row) => `run:${row.run_id}`)}</section>
    <section class="environment-section live-zone"><div class="section-head"><h2>LIVE RUNS</h2><span class="muted">Deliberate broker execution</span></div>${table(RUN_COLUMNS, data.runs.filter((row) => row.environment === "LIVE"), (row) => `run:${row.run_id}`)}</section>
    <section class="section"><div class="section-head"><h2>Open positions</h2></div>${table(positionColumns, data.positions)}</section>
    <section class="section"><div class="section-head"><h2>Today</h2></div><div class="metric-strip">${metrics}</div></section>
    <section class="section"><div class="section-head"><h2>Attention</h2></div><div class="attention">${attention}</div></section>`;
}

async function runsPage() {
  const params = new URLSearchParams(location.search);
  const runId = params.get("run");
  if (runId) return runDetail(runId);
  const rows = await api("/api/runs");
  main.innerHTML = `${head("Runs", "Run-level operations and economics, separated by execution environment.")}
    <section class="environment-section paper-zone"><div class="section-head"><h2>PAPER RUNS</h2>${environment("PAPER")}</div>${table(RUN_COLUMNS, rows.filter((row) => row.environment === "PAPER"), (row) => `run:${row.run_id}`)}</section>
    <section class="environment-section live-zone"><div class="section-head"><h2>LIVE RUNS</h2>${environment("LIVE")}</div>${table(RUN_COLUMNS, rows.filter((row) => row.environment === "LIVE"), (row) => `run:${row.run_id}`)}</section>`;
}

async function universesPage() {
  const [options, grouped, starting] = await Promise.all([
    api("/api/universe-builder/options"),
    api("/api/universe-runs"),
    api("/api/universe-runs/start-status"),
  ]);
  runStartStatus = starting;
  const marketOptions = options.markets.map((item) => `<option value="${esc(item.market_id)}" ${item.market_id === starting.market ? "selected" : ""}>${esc(item.label)}${item.experimental ? " · PAPER test" : ""}</option>`).join("");
  const strategyOptions = options.strategies.map((item) => `<option value="${esc(item.strategy_id)}" data-version="${esc(item.strategy_version)}" data-environments="${esc(item.environments.join(","))}">${esc(item.label)}</option>`).join("");
  const runCard = (row) => `<article class="run-card ${row.enabled ? "" : "disabled"}">
    <div class="run-card-title"><div><span class="eyebrow">${esc(row.market_id || row.universe)} · ${esc(row.currency || "NATIVE")}</span><h3>${esc(row.display_name)}</h3></div>${badge(row.enabled ? row.status : "DISABLED", row.enabled ? row.status : "warn")}</div>
    ${row.search_status ? `<p class="muted">Search: ${esc(row.search_status)}</p>` : ""}${row.reason ? `<p class="notice">${esc(row.reason)}</p>` : ""}
    <div class="run-stats"><div><span>Watch</span><strong>${number(row.candidate_count)}</strong></div><div><span>Signals today</span><strong>${number(row.signals_today)}</strong></div><div><span>Positions</span><strong>${number(row.open_positions)}</strong></div><div><span>Today realised</span><strong>${money(row.today_realised_pnl, row.currency)}</strong></div><div><span>Unrealised</span><strong>${row.unrealised_status === "AVAILABLE" ? money(row.unrealised_pnl, row.currency) : esc(row.unrealised_status)}</strong></div><div><span>20-session R</span><strong>${row.total_r == null ? "—" : `${number(row.total_r)}R`}</strong></div></div>
    <div class="control-rail"><button class="secondary" data-action="run:${esc(row.run_id)}">Open run</button>${row.historical_only ? '<span class="muted">Historical only</span>' : `<button data-universe-control="${row.enabled ? "disable" : "enable"}" data-run-id="${esc(row.run_id)}" data-environment="${esc(row.environment)}">${row.enabled ? `Remove from ${esc(row.environment)}` : `Re-enable ${esc(row.environment)}`}</button>`}</div>
  </article>`;
  const cards = (rows) => rows.length ? `<div class="run-card-grid">${rows.map(runCard).join("")}</div>` : '<div class="empty">No runs in this environment.</div>';
  main.innerHTML = `${head("Market → Method → Run", "Choose a market and method, then start a run. The method owns stock search, qualification, vetoes, entry and exits.")}
    <p id="run-start-status" class="notice" role="status" aria-live="polite" hidden></p>
    ${lastOutcome ? `<p class="notice" role="status"><b>${esc(lastOutcome.apply_mode)}</b> · ${esc(lastOutcome.detail)}</p>` : ""}
    <section class="builder-panel"><div class="builder-stripe"><span>CREATE / SELECT</span><strong>MARKET RUN</strong></div><form id="universe-builder"><div class="builder-grid"><label>Market<select name="market_id">${marketOptions}</select></label><label>Method<select name="strategy_id">${strategyOptions}</select></label></div><details><summary>Account risk and capacity</summary><div class="builder-grid"><label>Risk per trade<input name="risk_per_trade" type="number" min="0.000001" max="1" step="any" value="0.001" required></label><label>Maximum positions<input name="max_concurrent_positions" type="number" min="1" step="1" value="1"></label></div></details><div class="builder-context"><div><span>Candidate selection</span><strong>${esc(options.candidate_screen.label)}</strong></div><div><span>Suitability</span><strong>Required data; no validated cap filter</strong></div><div><span>Market session</span><strong id="builder-session">—</strong></div><div><span>Search policy</span><strong id="builder-readiness">—</strong></div></div><div class="builder-actions"><button type="submit" data-add-environment="PAPER">Start PAPER run</button><button type="button" class="danger" data-add-environment="LIVE">Add to LIVE</button><span id="live-prerequisite" class="muted">Matching PAPER run required.</span></div></form></section>
    <section class="environment-section paper-zone"><div class="section-head"><h2>PAPER RUNS</h2>${environment("PAPER")}</div>${cards(grouped.PAPER)}</section>
    <section class="environment-section live-zone"><div class="section-head"><h2>LIVE RUNS</h2>${environment("LIVE")}</div>${cards(grouped.LIVE)}</section>`;

  const form = document.querySelector("#universe-builder");
  const marketSelect = form.elements.market_id;
  const strategySelect = form.elements.strategy_id;
  const selectedStrategy = () => strategySelect.selectedOptions[0];
  const selectedDefinition = () => options.markets.find((item) => item.market_id === marketSelect.value);
  const matchingPaper = () => grouped.PAPER.some((item) => item.market_id === marketSelect.value && item.strategy_id === strategySelect.value);
  const refreshBuilderContext = () => {
    const market = selectedDefinition();
    document.querySelector("#builder-session").textContent = market ? `${market.session} ${market.timezone}` : "—";
    document.querySelector("#builder-readiness").textContent = market ? `${market.search_policy} · ${market.validation}` : "—";
    const liveButton = form.querySelector('[data-add-environment="LIVE"]');
    const supportsLive = selectedStrategy().dataset.environments.split(",").includes("LIVE");
    liveButton.disabled = runStartStatus.status === "STARTING" || !supportsLive || !matchingPaper();
    document.querySelector("#live-prerequisite").textContent = !supportsLive ? `${selectedStrategy().textContent} is PAPER-only.` : (matchingPaper() ? "PAPER prerequisite satisfied." : "Matching PAPER run required.");
  };
  refreshBuilderContext();
  showRunStartStatus();
  form.addEventListener("change", refreshBuilderContext);
  const submitRun = async (target) => {
    if (runStartStatus.status === "STARTING") return;
    const selectedStrategyOption = selectedStrategy();
    const body = {
      market_id: marketSelect.value,
      strategy_id: strategySelect.value,
      strategy_version: selectedStrategyOption.dataset.version,
      risk_per_trade: Number(form.elements.risk_per_trade.value),
      max_concurrent_positions: form.elements.max_concurrent_positions.value ? Number(form.elements.max_concurrent_positions.value) : null,
    };
    if (target === "LIVE") {
      const settings = await api("/api/settings");
      const broker = settings.broker_configuration.find((item) => item.environment === "LIVE");
      if (!broker?.expected_account) throw new Error("LIVE expected account is not configured");
      if (!window.confirm(`Create a separate LIVE run for ${selectedDefinition().label} / ${selectedStrategyOption.textContent} on ${broker.expected_account}?`)) return;
      body.confirmed = true;
      body.target_account = broker.expected_account;
    }
    runStartStatus = { status: "STARTING", detail: "Sending start request…" };
    showRunStartStatus();
    try {
      runStartStatus = await api(`/api/universe-runs/${target.toLowerCase()}?background=true`, { method: "POST", body: JSON.stringify(body) });
      showRunStartStatus();
    } catch (error) {
      runStartStatus = { status: "FAILED", detail: error.message };
      showRunStartStatus();
      refreshBuilderContext();
    }
  };
  form.addEventListener("submit", async (event) => { event.preventDefault(); try { await submitRun("PAPER"); } catch (error) { alert(error.message); } });
  form.querySelector('[data-add-environment="LIVE"]').addEventListener("click", async () => { try { await submitRun("LIVE"); } catch (error) { alert(error.message); } });
  main.querySelectorAll("[data-universe-control]").forEach((button) => button.addEventListener("click", async () => {
    try {
      let body = {};
      if (button.dataset.environment === "LIVE" && button.dataset.universeControl === "enable") body = await confirmLive(button.dataset.runId);
      lastOutcome = await api(`/api/universe-runs/${encodeURIComponent(button.dataset.runId)}/${button.dataset.universeControl}`, { method: "POST", body: JSON.stringify(body) });
      await render();
    } catch (error) { if (error.message !== "cancelled") alert(error.message); }
  }));
}

function showRunStartStatus() {
  const notice = document.querySelector("#run-start-status");
  if (!notice) return;
  const starting = runStartStatus.status === "STARTING";
  notice.hidden = runStartStatus.status === "IDLE";
  notice.textContent = `${runStartStatus.status}: ${runStartStatus.detail || ""}`;
  notice.classList.toggle("error", runStartStatus.status === "FAILED");
  const form = document.querySelector("#universe-builder");
  form.setAttribute("aria-busy", String(starting));
  form.querySelectorAll('input, select, [data-add-environment="PAPER"]').forEach((control) => { control.disabled = starting; });
  const paper = form.querySelector('[data-add-environment="PAPER"]');
  paper.textContent = starting ? "Starting…" : "Start PAPER run";
  if (starting) form.querySelector('[data-add-environment="LIVE"]').disabled = true;
}

async function runDetail(runId) {
  const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const selectedPeriod = new URLSearchParams(location.search).get("period") || "TODAY";
  const performance = await api(`/api/runs/${encodeURIComponent(runId)}/performance?period=${encodeURIComponent(selectedPeriod)}`);
  const details = ["market", "environment", "account", "currency", "market_state", "session", "screen_state", "watchlist_size", "last_checkpoint", "next_checkpoint", "risk_per_trade", "max_concurrent_positions"];
  const grid = details.map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${key === "environment" ? environment(run[key]) : esc(run[key])}</strong></div>`).join("");
  const funnel = run.funnel.map((step, index) => `${index ? '<div class="funnel-arrow"></div>' : ""}<div class="funnel-step"><span>${esc(step.stage)}</span><strong>${number(step.count)}</strong></div>`).join("");
  const metricValues = [["Closed trades", performance.closed_trades], ["Wins", performance.wins], ["Losses", performance.losses], ["Win %", performance.win_percent], ["Realised P/L", money(performance.realised_pnl, performance.currency)], ["Unrealised P/L", performance.unrealised_status === "AVAILABLE" ? money(performance.unrealised_pnl, performance.currency) : performance.unrealised_status], ["Total R", performance.total_r == null ? "—" : `${number(performance.total_r)}R`], ["Mean R", performance.mean_r == null ? "—" : `${number(performance.mean_r)}R`], ["Max drawdown", money(performance.max_realised_drawdown, performance.currency)]];
  const metrics = metricValues.map(([label, value]) => `<div class="metric"><span>${esc(label)}</span><strong>${typeof value === "number" ? number(value) : esc(value)}</strong></div>`).join("");
  const historyColumns = [{ key: "date", label: "Date" }, { key: "trades", label: "Trades", numeric: true }, { label: "P/L", numeric: true, render: (row) => money(row.pnl, performance.currency) }, { label: "R", numeric: true, render: (row) => row.r == null ? "—" : `${number(row.r)}R` }];
  const periods = [["TODAY", "Today"], ["5_SESSIONS", "5 Sessions"], ["20_SESSIONS", "20 Sessions"], ["ALL", "All"]].map(([value, label]) => `<button class="${value === selectedPeriod ? "active" : "secondary"}" data-performance-period="${value}">${label}</button>`).join("");
  main.innerHTML = `${head(run.display_name, `${run.market || run.universe} / ${run.environment}`)}
    <section class="section"><div class="control-rail"><button ${run.historical_only ? "disabled" : ""} data-control="${run.enabled ? "disable" : "enable"}">${run.enabled ? "Disable run" : "Enable run"}</button><button class="secondary" data-control="edit">Edit risk & capacity</button></div><p class="notice">Market, method specification, universe snapshot and environment belong to the saved run. Create another run to change them. Disabling stops future entries and never flattens exposure.</p>${lastOutcome ? `<p class="notice" role="status"><b>${esc(lastOutcome.apply_mode)}</b> · ${esc(lastOutcome.detail)}</p>` : ""}</section>
    <section class="section"><div class="section-head"><h2>Run configuration</h2></div><div class="detail-grid">${grid}</div><details><summary>Method specification and run provenance</summary><pre>${esc(JSON.stringify({ method: run.strategy_id, version: run.strategy_version, spec_hash: run.method_spec_hash, specification: run.method_spec, provenance: run.provenance }, null, 2))}</pre></details></section>
    <section class="section"><div class="section-head"><h2>Pipeline funnel</h2><span class="muted">Persisted counters only</span></div><div class="funnel">${funnel}</div></section>
    <section class="section"><div class="section-head"><h2>Performance</h2><div class="tabs">${periods}</div></div><div class="metric-strip performance-strip">${metrics}</div>${table(historyColumns, performance.history)}</section>`;
  main.querySelectorAll("[data-control]").forEach((button) => button.addEventListener("click", () => runControl(run, button.dataset.control)));
  main.querySelectorAll("[data-performance-period]").forEach((button) => button.addEventListener("click", () => { location.href = `/runs?run=${encodeURIComponent(runId)}&period=${button.dataset.performancePeriod}`; }));
}
async function runControl(run, control) {
  try {
    let result = null;
    if (control === "disable") result = await api(`/api/runs/${encodeURIComponent(run.run_id)}/disable`, { method: "POST" });
    if (control === "enable") {
      let body = {};
      if (run.environment === "LIVE") body = await confirmLive(run.run_id);
      result = await api(`/api/runs/${encodeURIComponent(run.run_id)}/enable`, { method: "POST", body: JSON.stringify(body) });
    }
    if (control === "edit") {
      const risk = prompt("Risk per trade", run.risk_per_trade);
      const slots = prompt("Maximum concurrent positions", run.max_concurrent_positions ?? "");
      if (risk === null || slots === null) return;
      let body = { universe: run.universe, strategy: run.strategy, risk_per_trade: Number(risk), max_concurrent_positions: slots === "" ? null : Number(slots) };
      if (run.environment === "LIVE") {
        body = { ...body, ...(await confirmLive(run.run_id, { risk_per_trade: body.risk_per_trade, max_concurrent_positions: body.max_concurrent_positions })) };
      }
      result = await api(`/api/runs/${encodeURIComponent(run.run_id)}`, { method: "PUT", body: JSON.stringify(body) });
    }
    if (result) lastOutcome = result;
    await render();
  } catch (error) { if (error.message !== "cancelled") alert(error.message); }
}
async function confirmLive(runId, proposedRisk = null) {
  const context = await api(`/api/runs/${encodeURIComponent(runId)}/live-confirmation`);
  const dialog = document.querySelector("#live-dialog");
  const check = document.querySelector("#live-check");
  check.checked = false;
  const risk = proposedRisk || context.risk;
  document.querySelector("#live-context").innerHTML = `<div class="detail-grid">${["run_id", "universe", "strategy", "target_environment", "target_account"].map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${esc(context[key])}</strong></div>`).join("")}<div class="detail-cell"><span>risk configuration</span><strong>${esc(JSON.stringify(risk))}</strong></div></div>`;
  dialog.showModal();
  return new Promise((resolve, reject) => dialog.addEventListener("close", () => {
    if (dialog.returnValue === "confirm" && check.checked) resolve({ confirmed: true, target_account: context.target_account });
    else reject(new Error("cancelled"));
  }, { once: true }));
}

async function candidatesPage() {
  const params = new URLSearchParams(location.search);
  const signal = params.get("signal");
  if (signal) return candidateDetail(signal);
  const runs = (await api("/api/runs")).filter((run) => run.enabled);
  const requestedRun = params.get("run");
  const selected = runs.some((run) => run.run_id === requestedRun) ? requestedRun : runs[0]?.run_id || "";
  const session = params.get("session") || new Date().toISOString().slice(0, 10);
  const checkpoint = params.get("checkpoint") || "";
  const status = params.get("status") || "";
  const offset = Number(params.get("offset") || 0);
  const query = new URLSearchParams({ run_id: selected, session, limit: "100", offset: String(offset) });
  if (checkpoint) query.set("checkpoint", new Date(checkpoint).toISOString());
  if (status) query.set("status", status);
  const data = selected ? await api(`/api/candidates?${query}`) : { items: [], total: 0, limit: 100, offset: 0 };
  const columns = [
    { key: "symbol", label: "Symbol" },
    { label: "Method score", numeric: true, render: (row) => number(row.session_hard_score) },
    { label: "PRE_MOVE", numeric: true, render: (row) => number(row.pre_move_m) },
    { label: "Whipsaw risk", numeric: true, render: (row) => number(row.whipsaw_risk_score) },
    { label: "Q1", render: (row) => row.q1_eligible == null ? "—" : (row.q1_eligible ? "Admitted" : "Vetoed") },
    { key: "side", label: "Direction" },
    { label: "Entry", numeric: true, render: (row) => number(row.entry) },
    { label: "Status", render: (row) => badge(row.status) },
    { key: "reason", label: "Reason" },
  ];
  const options = runs.map((run) => `<option ${run.run_id === selected ? "selected" : ""}>${esc(run.run_id)}</option>`).join("");
  const statuses = ["", "PRE_CONTEXT_NOT_READY", "PRE_MOVE_NOT_READY", "INELIGIBLE", "WAITING_FOR_ENTRY", "ENTRY_TRIGGERED", "NOT_QUALIFIED", "EXPIRED"];
  const statusOptions = statuses.map((value) => `<option value="${esc(value)}" ${value === status ? "selected" : ""}>${esc(value || "ALL")}</option>`).join("");
  const pageStart = data.total ? offset + 1 : 0;
  const pageEnd = Math.min(offset + data.items.length, data.total);
  const pager = `<div class="toolbar"><button id="candidate-prev" class="secondary" ${offset === 0 ? "disabled" : ""}>Previous</button><span>${pageStart}–${pageEnd} of ${data.total}</span><button id="candidate-next" class="secondary" ${offset + data.items.length >= data.total ? "disabled" : ""}>Next</button></div>`;
  main.innerHTML = `${head("Candidates", "Method qualification, Q1 admission and causal entry state. Open a candidate for detailed provenance.")}<div class="toolbar"><label>Run<select id="candidate-run">${options}</select></label><label>Session<input id="candidate-session" type="date" value="${esc(session)}"></label><label>Checkpoint<input id="candidate-checkpoint" type="datetime-local" value="${esc(checkpoint)}"></label><label>Status<select id="candidate-status">${statusOptions}</select></label></div><section>${table(columns, data.items, (row) => row.signal_id ? `candidate:${row.signal_id}` : "")}</section>${pager}`;
  const updateCandidateFilters = () => {
    const next = new URLSearchParams();
    next.set("run", document.querySelector("#candidate-run").value);
    next.set("session", document.querySelector("#candidate-session").value);
    const nextCheckpoint = document.querySelector("#candidate-checkpoint").value;
    const nextStatus = document.querySelector("#candidate-status").value;
    if (nextCheckpoint) next.set("checkpoint", nextCheckpoint);
    if (nextStatus) next.set("status", nextStatus);
    location.href = `/candidates?${next}`;
  };
  for (const id of ["candidate-run", "candidate-session", "candidate-checkpoint", "candidate-status"]) {
    document.querySelector(`#${id}`)?.addEventListener("change", updateCandidateFilters);
  }
  document.querySelector("#candidate-prev")?.addEventListener("click", () => { params.set("offset", String(Math.max(0, offset - 100))); location.href = `/candidates?${params}`; });
  document.querySelector("#candidate-next")?.addEventListener("click", () => { params.set("offset", String(offset + 100)); location.href = `/candidates?${params}`; });
}
async function candidateDetail(signalId) {
  const item = await api(`/api/candidates/${encodeURIComponent(signalId)}`);
  const fields = ["symbol", "con_id", "run_id", "universe", "strategy", "strategy_version", "environment", "account", "t0", "p0", "expected_move_source", "expected_move_observation_at", "expected_move_calculation_version", "raw_historical_volatility", "historical_volatility", "market_regular_minutes", "expected_absolute_return_15m", "m_price", "raw_pre_move", "pre_move_m", "cohort_percentile", "band", "session_hard_score", "reason", "side", "direction", "rank", "entry", "entry_reference", "stop", "target", "whipsaw_risk_score", "q1_eligible", "up_trigger", "down_trigger", "armed_at", "deadline", "method_spec_hash", "status", "signal_id", "order_plan_id"];
  main.innerHTML = `${head(item.symbol, "Candidate calculation lineage copied from authoritative Stage 5/6 outputs.")}<section class="section"><div class="detail-grid">${fields.map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${key === "environment" ? environment(item[key]) : esc(item[key])}</strong></div>`).join("")}</div></section>`;
}

async function ordersPage() {
  const params = new URLSearchParams(location.search);
  const plan = params.get("plan");
  if (plan) return orderDetail(plan);
  const scope = params.get("scope") || "open";
  const data = await api(`/api/orders?scope=${scope}`);
  const tabs = ["open", "today", "rejected", "all"].map((name) => `<button class="${name === scope ? "active" : ""}" data-scope="${name}">${name.toUpperCase()}</button>`).join("");
  const groups = data.items.map((item) => `<article class="order-group" data-action="order:${esc(item.order_plan_id)}"><div class="order-title"><strong>${esc(item.symbol)}</strong>${environment(item.environment)}<span>${esc(item.run_id)}</span><span class="muted">${esc(item.account)}</span></div>${item.orders.map((leg, index) => `<div class="order-leg ${index ? "child" : ""}"><b>${index ? "└─ " : ""}${esc(leg.role)}</b>${badge(leg.status)}<span>${leg.role === "ENTRY" ? `${number(item.filled)} @ ${number(item.average_fill || item.entry)}` : (leg.role === "STOP" ? number(item.stop) : number(item.target))}</span><span>#${esc(leg.ibkr_order_id)}</span></div>`).join("")}<div class="muted">Signal ${esc(item.signal_id)} · Plan ${esc(item.order_plan_id)}${item.rejection_reason ? ` · ${esc(item.rejection_reason)}` : ""}</div></article>`).join("");
  main.innerHTML = `${head("Orders", "Protected orders grouped by plan; broker identifiers remain visible.")}<div class="toolbar tabs">${tabs}</div>${groups || '<div class="empty">No orders in this view.</div>'}`;
  main.querySelectorAll("[data-scope]").forEach((button) => button.addEventListener("click", () => { location.href = `/orders?scope=${button.dataset.scope}`; }));
}
async function orderDetail(orderPlanId) {
  const item = await api(`/api/orders/${encodeURIComponent(orderPlanId)}`);
  const fields = ["order_plan_id", "signal_id", "run_id", "strategy", "strategy_version", "environment", "account", "con_id", "symbol", "side", "quantity", "order_type", "entry", "stop", "target", "status", "filled", "average_fill", "time", "rejection_reason"];
  const legs = item.orders.map((leg) => `<div class="order-leg"><b>${esc(leg.role)}</b>${badge(leg.status)}<span>IBKR #${esc(leg.ibkr_order_id)}</span></div>`).join("");
  main.innerHTML = `${head(item.symbol, "Broker order state and protected-order lineage.")}<section class="section"><div class="detail-grid">${fields.map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${key === "environment" ? environment(item[key]) : esc(item[key])}</strong></div>`).join("")}</div></section><section class="section"><div class="section-head"><h2>IBKR order legs</h2></div>${legs}</section>`;
}
async function positionsPage() {
  const params = new URLSearchParams(location.search);
  if (params.has("environment") && params.has("account") && params.has("con_id")) {
    return positionDetail(params.get("environment"), params.get("account"), params.get("con_id"));
  }
  const rows = await api("/api/positions");
  const columns = [
    { key: "symbol", label: "Symbol" }, { key: "run_id", label: "Run" }, { label: "Env", render: (row) => environment(row.environment) }, { key: "account", label: "Account" }, { key: "side", label: "Side" },
    { key: "quantity", label: "Qty", numeric: true }, { label: "Average entry", numeric: true, render: (row) => number(row.average_entry) }, { label: "Current", numeric: true, render: (row) => number(row.current_price) },
    { label: "Stop", numeric: true, render: (row) => number(row.stop) }, { label: "Target", numeric: true, render: (row) => number(row.target) }, { label: "Unrealised P/L", numeric: true, render: (row) => money(row.unrealised_pnl) }, { label: "Source", render: (row) => badge(row.source, row.source === "IBKR" ? "good" : "warn") },
  ];
  main.innerHTML = `${head("Positions", "Latest normalized IBKR snapshot; unknown broker exposure remains visible.")}<section class="section">${table(columns, rows, (row) => `position:${row.environment}/${encodeURIComponent(row.account)}/${row.con_id}`)}</section>`;
}
async function positionDetail(environmentName, account, conId) {
  const item = await api(`/api/positions/${encodeURIComponent(environmentName)}/${encodeURIComponent(account)}/${encodeURIComponent(conId)}`);
  const fields = ["symbol", "con_id", "run_id", "strategy", "strategy_version", "signal_id", "order_plan_id", "environment", "account", "side", "quantity", "average_entry", "current_price", "stop", "target", "unrealised_pnl", "opened_at", "observed_at", "source"];
  const legs = item.orders.map((leg) => `<div class="order-leg"><b>${esc(leg.role)}</b>${badge(leg.status)}<span>IBKR #${esc(leg.ibkr_order_id)}</span></div>`).join("") || '<div class="empty">No Stocker order lineage for this broker position.</div>';
  main.innerHTML = `${head(`${item.symbol} — ${item.environment}`, "Broker-authoritative position with Stocker lineage when known.")}<section class="section"><div class="detail-grid">${fields.map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${key === "environment" ? environment(item[key]) : esc(item[key])}</strong></div>`).join("")}</div></section><section class="section"><div class="section-head"><h2>Protection</h2></div>${legs}</section>`;
}
async function tradesPage() {
  const params = new URLSearchParams(location.search);
  const settings = await api("/api/settings");
  const period = params.get("period") || "today";
  const selectedEnvironment = params.get("environment") || "";
  const selectedRun = params.get("run") || "";
  const selectedStrategy = params.get("strategy") || "";
  const selectedUniverse = params.get("universe") || "";
  const selectedSymbol = params.get("symbol") || "";
  const offset = Number(params.get("offset") || 0);
  const now = new Date();
  let startDate = params.get("start") || "";
  let endDate = params.get("end") || "";
  if (period !== "custom") {
    const start = new Date(now);
    start.setHours(0, 0, 0, 0);
    if (period === "week") start.setDate(start.getDate() - 6);
    if (period === "month") start.setDate(1);
    startDate = start.toISOString();
    endDate = now.toISOString();
  }
  const query = new URLSearchParams({ limit: "100", offset: String(offset) });
  for (const [key, value] of Object.entries({ environment: selectedEnvironment, run_id: selectedRun, strategy: selectedStrategy, universe: selectedUniverse, symbol: selectedSymbol, start: startDate, end: endDate })) {
    if (value) query.set(key, value);
  }
  const data = await api(`/api/trades?${query}`);
  const summary = data.summary;
  const totalPnl = summary.pnl_status === "MULTIPLE_CURRENCIES" ? "Multiple currencies" : money(summary.total_pnl, summary.currency || "USD");
  const metrics = [["Trades", summary.trades], ["Wins", summary.wins], ["Losses", summary.losses], ["Win %", summary.win_percent], ["Total P/L", totalPnl], ["Total R", summary.total_r], ["Mean R", summary.mean_r]].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${typeof value === "number" ? number(value) : esc(value)}</strong></div>`).join("");
  const columns = [{ key: "date_time", label: "Date/time" }, { key: "run_id", label: "Run" }, { key: "strategy", label: "Strategy" }, { key: "universe", label: "Universe" }, { label: "Env", render: (row) => environment(row.environment) }, { key: "account", label: "Account" }, { key: "symbol", label: "Symbol" }, { key: "side", label: "Side" }, { key: "entry", label: "Entry", numeric: true }, { key: "exit", label: "Exit", numeric: true }, { key: "quantity", label: "Qty", numeric: true }, { label: "P/L", numeric: true, render: (row) => money(row.pnl, row.currency || "USD") }, { key: "r", label: "R", numeric: true }, { key: "exit_reason", label: "Exit reason" }];
  const options = (values, selected, allLabel) => `<option value="">${allLabel}</option>${values.map((value) => `<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(value)}</option>`).join("")}`;
  const runValues = settings.runs.map((item) => item.run_id);
  const strategyValues = [...new Set(settings.runs.map((item) => item.strategy))];
  const universeValues = settings.universes.map((item) => item.universe_id);
  const periodButtons = ["today", "week", "month", "custom"].map((value) => `<button data-period="${value}" class="${value === period ? "active" : "secondary"}">${value.toUpperCase()}</button>`).join("");
  main.innerHTML = `${head("Trades", "Completed execution ledger. PAPER and LIVE remain explicitly labelled.")}<div class="toolbar tabs">${periodButtons}</div><div class="toolbar"><label>Environment<select id="trade-environment">${options(["PAPER", "LIVE"], selectedEnvironment, "ALL")}</select></label><label>Run<select id="trade-run">${options(runValues, selectedRun, "ALL")}</select></label><label>Strategy<select id="trade-strategy">${options(strategyValues, selectedStrategy, "ALL")}</select></label><label>Universe<select id="trade-universe">${options(universeValues, selectedUniverse, "ALL")}</select></label><label>Symbol<input id="trade-symbol" value="${esc(selectedSymbol)}"></label><label>Start<input id="trade-start" type="datetime-local" value="${esc(period === "custom" ? startDate : "")}"></label><label>End<input id="trade-end" type="datetime-local" value="${esc(period === "custom" ? endDate : "")}"></label></div><div class="metric-strip">${metrics}</div><section class="section">${table(columns, data.items)}</section>`;
  const applyTradeFilters = () => {
    const next = new URLSearchParams({ period });
    for (const [key, id] of Object.entries({ environment: "trade-environment", run: "trade-run", strategy: "trade-strategy", universe: "trade-universe", symbol: "trade-symbol" })) {
      const value = document.querySelector(`#${id}`).value.trim();
      if (value) next.set(key, value);
    }
    if (period === "custom") {
      const customStart = document.querySelector("#trade-start").value;
      const customEnd = document.querySelector("#trade-end").value;
      if (customStart) next.set("start", new Date(customStart).toISOString());
      if (customEnd) next.set("end", new Date(customEnd).toISOString());
    }
    location.href = `/trades?${next}`;
  };
  for (const id of ["trade-environment", "trade-run", "trade-strategy", "trade-universe", "trade-start", "trade-end"]) document.querySelector(`#${id}`)?.addEventListener("change", applyTradeFilters);
  document.querySelector("#trade-symbol")?.addEventListener("change", applyTradeFilters);
  main.querySelectorAll("[data-period]").forEach((button) => button.addEventListener("click", () => { params.set("period", button.dataset.period); params.delete("offset"); location.href = `/trades?${params}`; }));
}
async function systemPage() {
  const data = await api("/api/system");
  const cards = data.environments.map((item) => `<article class="status-card"><span class="eyebrow">IBKR ${esc(item.environment)}</span><strong>${item.connected ? "CONNECTED" : "DISCONNECTED"}</strong><small>${esc(item.account)} · ${item.reconciled ? "RECONCILED" : "NOT RECONCILED"}<br>Net liquidation ${money(item.equity)} · Buying power ${money(item.buying_power)}<br>${item.open_orders} open orders · ${item.positions} positions</small></article>`).join("");
  const resources = data.ibkr_api_resources;
  const resourceView = resources ? `<section class="section"><div class="section-head"><h2>IBKR API RESOURCES</h2><span class="muted">Stocker-observed connection state; not the broker-authoritative account allowance</span></div><div class="detail-grid"><div class="detail-cell"><span>Stocker market-data budget</span><strong>${resources.market_data_line_budget}</strong></div><div class="detail-cell"><span>Active streaming lines</span><strong>${resources.active_market_data_lines}</strong><small>${resources.active_underlying_lines} underlying</small></div><div class="detail-cell"><span>Active scanners</span><strong>${resources.active_scanners}</strong></div><div class="detail-cell"><span>Pending historical work</span><strong>${resources.pending_historical_work}</strong><small>Concurrency bound ${resources.historical_concurrency_limit}</small></div><div class="detail-cell"><span>Requests today</span><strong>${resources.market_data_requests_today} market data</strong><small>${resources.historical_requests_today} historical · ${resources.scanner_requests_today} scanner</small></div><div class="detail-cell"><span>Pacing state</span><strong>${esc(resources.pacing_state)}</strong><small>${resources.pacing_violations_today} violations · ${resources.capacity_rejects_today} capacity rejects</small></div></div></section>` : "";
  const problems = data.problems.length ? data.problems.map((message) => `<div class="attention-item"><b>PROBLEM</b><span>${esc(message)}</span></div>`).join("") : '<div class="empty">No execution-system problems.</div>';
  const events = data.events.length ? data.events.map((item) => `<div class="attention-item"><b>${new Date(item.timestamp).toLocaleTimeString()}</b><span>${esc(item.message)}</span></div>`).join("") : '<div class="empty">No recent operational events.</div>';
  main.innerHTML = `${head("System", "IBKR connectivity, reconciliation, runtime health, and concise operational events.")}<section class="status-grid">${cards}<article class="status-card"><span class="eyebrow">RUNTIME</span><strong>${esc(data.application)}</strong><small>${data.runtime.active_runs} active runs</small></article></section>${resourceView}<section class="section"><div class="section-head"><h2>System problems</h2></div><div class="attention">${problems}</div></section><section class="section"><div class="section-head"><h2>Recent operational events</h2></div>${events}</section>`;
}
async function settingsPage() {
  const data = await api("/api/settings");
  const runtimeBroker = Object.fromEntries(data.broker.map((item) => [item.environment, item]));
  const broker = data.broker_configuration.map((item) => {
    const runtime = runtimeBroker[item.environment] || {};
    return `<form class="config-form" data-broker-form="${esc(item.environment)}"><div class="section-head"><h3>IBKR ${esc(item.environment)}</h3>${environment(item.environment)}</div><div class="detail-grid"><label class="detail-cell"><span>Host</span><input name="host" required value="${esc(item.host)}"></label><label class="detail-cell"><span>Port</span><input name="port" type="number" min="1" max="65535" required value="${esc(item.port)}"></label><label class="detail-cell"><span>Client ID</span><input name="client_id" type="number" min="1" required value="${esc(item.client_id)}"></label><label class="detail-cell"><span>Expected account</span><input name="expected_account" required value="${esc(item.expected_account || "")}"></label><label class="detail-cell"><span>Connect timeout (seconds)</span><input name="connect_timeout_seconds" type="number" min="0.1" step="0.1" required value="${esc(item.connect_timeout_seconds)}"></label><label class="detail-cell"><span>Request timeout (seconds)</span><input name="request_timeout_seconds" type="number" min="0.1" step="0.1" required value="${esc(item.request_timeout_seconds)}"></label><label class="detail-cell"><span>Stocker market-data budget</span><input name="market_data_line_budget" type="number" min="1" step="1" required value="${esc(item.market_data_line_budget)}"><small>Self-imposed per connection; not the broker-authoritative account allowance</small></label><div class="detail-cell"><span>Runtime</span><strong>${runtime.connected ? "CONNECTED" : "DISCONNECTED"} · ${runtime.account ? esc(runtime.account) : "—"}</strong></div></div><div class="control-rail"><button type="submit">Save and reconnect ${esc(item.environment)}</button></div></form>`;
  }).join("");
  const strategyColumns = [{ key: "strategy", label: "Strategy" }, { key: "identity", label: "Identity" }, { key: "version", label: "Version" }, { key: "description", label: "Definition" }];
  main.innerHTML = `${head("Settings", "Broker and application controls. Market-run construction now belongs to Universes; strategy identity remains read-only.")}${lastOutcome ? `<p class="notice" role="status"><b>${esc(lastOutcome.apply_mode)}</b> · ${esc(lastOutcome.detail)}</p>` : ""}<section class="section"><div class="section-head"><h2>Broker</h2><span class="muted">Reconnect and reconcile only the edited environment</span></div>${broker}</section><section class="section"><div class="section-head"><h2>Application / runtime</h2></div><p class="notice">Use Start run to select the market and method. Existing CUSTOM universe support remains available through backend configuration and CLI workflows.</p></section><section class="section"><div class="section-head"><h2>Methods</h2><span class="muted">Installed method identity is read-only</span></div>${table(strategyColumns, data.strategies)}</section>`;
  main.querySelectorAll("[data-broker-form]").forEach((form) => form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    body.port = Number(body.port);
    body.client_id = Number(body.client_id);
    body.connect_timeout_seconds = Number(body.connect_timeout_seconds);
    body.request_timeout_seconds = Number(body.request_timeout_seconds);
    body.market_data_line_budget = Number(body.market_data_line_budget);
    body.expected_account = body.expected_account || null;
    try {
      lastOutcome = await api(`/api/settings/broker/${encodeURIComponent(form.dataset.brokerForm)}`, { method: "PUT", body: JSON.stringify(body) });
      await render();
    } catch (error) { alert(error.message); }
  }));
}

async function render() {
  clearTimeout(timer);
  const name = route();
  activate(name);
  try {
    const cached = await refreshHeader();
    await ({ overview, runs: runsPage, universes: universesPage, candidates: candidatesPage, orders: ordersPage, positions: positionsPage, trades: tradesPage, system: systemPage, settings: settingsPage })[name](cached);
    pageHasRendered = true;
    clearRefreshWarning();
  } catch (error) {
    if (pageHasRendered) showRefreshWarning();
    else main.innerHTML = `${head("Read-only unavailable", "The trading runtime is isolated from this dashboard failure.")}<div class="section error">${esc(error.message)}</div>`;
  }
  scheduleRefresh();
}
function scheduleRefresh() {
  clearTimeout(timer);
  timer = setTimeout(refreshCurrentPage, route() === "orders" || route() === "positions" ? 5000 : 10000);
}
async function refreshCurrentPage() {
  if (INTERACTIVE_ROUTES.has(route())) {
    try {
      await refreshHeader();
      if (route() === "universes") {
        const wasStarting = runStartStatus.status === "STARTING";
        runStartStatus = await api("/api/universe-runs/start-status");
        if (wasStarting && runStartStatus.status !== "STARTING") {
          await render();
          return;
        }
        showRunStartStatus();
      }
      clearRefreshWarning();
    } catch (_) { showRefreshWarning(); }
    scheduleRefresh();
    return;
  }
  await render();
}
document.addEventListener("click", (event) => {
  const row = event.target.closest("[data-action]");
  if (!row || !row.dataset.action) return;
  const [type, id] = row.dataset.action.split(":");
  if (type === "run") location.href = `/runs?run=${encodeURIComponent(id)}`;
  if (type === "candidate") location.href = `/candidates?signal=${encodeURIComponent(id)}`;
  if (type === "order") location.href = `/orders?plan=${encodeURIComponent(id)}`;
  if (type === "position") {
    const [environmentName, account, conId] = id.split("/");
    location.href = `/positions?environment=${encodeURIComponent(environmentName)}&account=${encodeURIComponent(decodeURIComponent(account))}&con_id=${encodeURIComponent(conId)}`;
  }
});
document.querySelector("#menu-button").addEventListener("click", (event) => {
  const open = document.querySelector("#sidebar").classList.toggle("open");
  event.currentTarget.setAttribute("aria-expanded", String(open));
});
render();
