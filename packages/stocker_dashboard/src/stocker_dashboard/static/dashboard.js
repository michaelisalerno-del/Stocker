"use strict";

const main = document.querySelector("#main");
const fmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 2 });
let timer;

function esc(value) {
  return String(value ?? "—").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
function number(value) { return value == null ? "—" : fmt.format(value); }
function money(value) { return value == null ? "—" : `${value < 0 ? "−" : ""}$${fmt.format(Math.abs(value))}`; }
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
  return ["overview", "runs", "candidates", "orders", "positions", "trades", "system", "settings"].includes(name) ? name : "overview";
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

async function overview(cached) {
  const data = cached || await api("/api/overview");
  const envCard = (name) => {
    const item = data.environments[name];
    return `<article class="status-card"><span class="eyebrow">IBKR ${name}</span><strong>${item ? (item.connected ? "CONNECTED" : "DISCONNECTED") : "NOT CONFIGURED"}</strong><small>${item ? `${esc(item.account)} · ${item.reconciled ? "RECONCILED" : "NOT RECONCILED"}` : "No runtime destination"}</small></article>`;
  };
  const runColumns = [
    { key: "run_id", label: "Run" }, { key: "universe", label: "Universe" }, { key: "strategy", label: "Strategy" },
    { label: "Env", render: (row) => environment(row.environment) }, { label: "Status", render: (row) => badge(row.status) },
    { key: "candidate_count", label: "Candidates", numeric: true }, { key: "signals_today", label: "Signals", numeric: true }, { key: "open_positions", label: "Positions", numeric: true },
  ];
  const positionColumns = [
    { key: "symbol", label: "Symbol" }, { key: "run_id", label: "Run" }, { label: "Env", render: (row) => environment(row.environment) },
    { key: "side", label: "Side" }, { key: "quantity", label: "Qty", numeric: true }, { label: "Entry", numeric: true, render: (row) => number(row.average_entry) },
    { label: "P/L", numeric: true, render: (row) => money(row.unrealised_pnl) },
  ];
  const metrics = Object.entries(data.today).map(([key, value]) => `<div class="metric"><span>${esc(key.replaceAll("_", " "))}</span><strong>${number(value)}</strong></div>`).join("");
  const attention = data.attention.length ? data.attention.map((item) => `<div class="attention-item"><b>${esc(item.scope)}</b><span>${esc(item.message)}</span></div>`).join("") : `<div class="empty">No current operational issues.</div>`;
  main.innerHTML = `${head("Overview", "Current PAPER, LIVE, run, and broker-authoritative exposure at a glance.")}
    <section class="status-grid">${envCard("PAPER")}${envCard("LIVE")}<article class="status-card"><span class="eyebrow">SYSTEM</span><strong>${esc(data.system)}</strong><small>${data.active_runs} active runs · ${data.open_positions} open positions</small></article></section>
    <section class="section"><div class="section-head"><h2>Active runs</h2><span class="muted">Backend status is authoritative</span></div>${table(runColumns, data.runs, (row) => `run:${row.run_id}`)}</section>
    <section class="section"><div class="section-head"><h2>Open positions</h2></div>${table(positionColumns, data.positions)}</section>
    <section class="section"><div class="section-head"><h2>Today</h2></div><div class="metric-strip">${metrics}</div></section>
    <section class="section"><div class="section-head"><h2>Attention</h2></div><div class="attention">${attention}</div></section>`;
}

async function runsPage() {
  const params = new URLSearchParams(location.search);
  const runId = params.get("run");
  if (runId) return runDetail(runId);
  const rows = await api("/api/runs");
  const columns = [
    { key: "run_id", label: "Run" }, { key: "universe", label: "Universe" }, { key: "strategy", label: "Strategy" },
    { label: "Env", render: (row) => environment(row.environment) }, { label: "Status", render: (row) => badge(row.status) },
    { key: "candidate_count", label: "Candidates", numeric: true }, { key: "signals_today", label: "Signals", numeric: true }, { key: "open_positions", label: "Positions", numeric: true },
  ];
  main.innerHTML = `${head("Runs", "Independent universe, strategy, execution environment, and risk configurations.")}<section class="section">${table(columns, rows, (row) => `run:${row.run_id}`)}</section>`;
}
async function runDetail(runId) {
  const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const details = ["run_id", "universe", "strategy", "strategy_version", "environment", "account", "status", "risk_per_trade", "max_concurrent_positions", "last_checkpoint", "next_checkpoint"];
  const grid = details.map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${key === "environment" ? environment(run[key]) : esc(run[key])}</strong></div>`).join("");
  const funnel = run.funnel.map((step, index) => `${index ? '<div class="funnel-arrow"></div>' : ""}<div class="funnel-step"><span>${esc(step.stage)}</span><strong>${number(step.count)}</strong></div>`).join("");
  main.innerHTML = `${head(run.run_id, `${run.universe} / ${run.strategy} / ${run.environment}`)}
    <section class="section"><div class="control-rail"><button data-control="${run.enabled ? "disable" : "enable"}">${run.enabled ? "Disable run" : "Enable run"}</button><button class="secondary" data-control="edit">Edit run config</button><button class="${run.environment === "LIVE" ? "secondary" : "danger"}" data-control="environment">Move to ${run.environment === "LIVE" ? "PAPER" : "LIVE"}</button></div><p class="notice">Configuration changes use the backend command path and require a runtime reload. Existing positions retain their original environment and account.</p></section>
    <section class="section"><div class="section-head"><h2>Run configuration</h2></div><div class="detail-grid">${grid}</div></section>
    <section class="section"><div class="section-head"><h2>Pipeline funnel</h2><span class="muted">Persisted counters only</span></div><div class="funnel">${funnel}</div></section>`;
  main.querySelectorAll("[data-control]").forEach((button) => button.addEventListener("click", () => runControl(run, button.dataset.control)));
}
async function runControl(run, control) {
  try {
    if (control === "disable") await api(`/api/runs/${encodeURIComponent(run.run_id)}/disable`, { method: "POST" });
    if (control === "enable") {
      let body = {};
      if (run.environment === "LIVE") body = await confirmLive(run.run_id);
      await api(`/api/runs/${encodeURIComponent(run.run_id)}/enable`, { method: "POST", body: JSON.stringify(body) });
    }
    if (control === "environment") {
      const target = run.environment === "LIVE" ? "PAPER" : "LIVE";
      let body = { environment: target };
      if (target === "LIVE") body = { ...body, ...(await confirmLive(run.run_id)) };
      await api(`/api/runs/${encodeURIComponent(run.run_id)}/environment`, { method: "POST", body: JSON.stringify(body) });
    }
    if (control === "edit") {
      const risk = prompt("Risk per trade", run.risk_per_trade);
      const slots = prompt("Maximum concurrent positions", run.max_concurrent_positions ?? "");
      if (risk === null || slots === null) return;
      let body = { universe: run.universe, strategy: run.strategy, risk_per_trade: Number(risk), max_concurrent_positions: slots === "" ? null : Number(slots) };
      if (run.environment === "LIVE") body = { ...body, ...(await confirmLive(run.run_id)) };
      await api(`/api/runs/${encodeURIComponent(run.run_id)}`, { method: "PUT", body: JSON.stringify(body) });
    }
    await render();
  } catch (error) { if (error.message !== "cancelled") alert(error.message); }
}
async function confirmLive(runId) {
  const context = await api(`/api/runs/${encodeURIComponent(runId)}/live-confirmation`);
  const dialog = document.querySelector("#live-dialog");
  const check = document.querySelector("#live-check");
  check.checked = false;
  document.querySelector("#live-context").innerHTML = `<div class="detail-grid">${["run_id", "universe", "strategy", "target_environment", "target_account"].map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${esc(context[key])}</strong></div>`).join("")}<div class="detail-cell"><span>risk configuration</span><strong>${esc(JSON.stringify(context.risk))}</strong></div></div>`;
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
  const runs = await api("/api/runs");
  const selected = params.get("run") || runs[0]?.run_id || "";
  const data = await api(`/api/candidates?run_id=${encodeURIComponent(selected)}&limit=100`);
  const columns = [
    { key: "rank", label: "Rank", numeric: true }, { key: "symbol", label: "Symbol" }, { label: "PRE_MOVE_M", numeric: true, render: (row) => number(row.pre_move_m) },
    { label: "Percentile", numeric: true, render: (row) => number(row.cohort_percentile) }, { key: "band", label: "Band" },
    { label: "HARD", render: (row) => row.session_hard == null ? "—" : (row.session_hard ? "✓" : "—") }, { key: "structure", label: "Structure" },
    { key: "direction", label: "Direction" }, { label: "Entry", numeric: true, render: (row) => number(row.entry) }, { label: "Status", render: (row) => badge(row.status) },
  ];
  const options = runs.map((run) => `<option ${run.run_id === selected ? "selected" : ""}>${esc(run.run_id)}</option>`).join("");
  main.innerHTML = `${head("Candidates", "Authoritative Stage 5 feature and Stage 6 strategy evaluation snapshots.")}<div class="toolbar"><label>Run<select id="candidate-run">${options}</select></label><label>Session<input type="date" value="${new Date().toISOString().slice(0, 10)}"></label><label>Status<select><option>All</option><option>WAITING_FOR_ENTRY</option><option>ENTRY_TRIGGERED</option><option>NOT_QUALIFIED</option></select></label></div><section>${table(columns, data.items, (row) => row.signal_id ? `candidate:${row.signal_id}` : "")}</section>`;
  document.querySelector("#candidate-run")?.addEventListener("change", (event) => { location.href = `/candidates?run=${encodeURIComponent(event.target.value)}`; });
}
async function candidateDetail(signalId) {
  const item = await api(`/api/candidates/${encodeURIComponent(signalId)}`);
  const fields = ["symbol", "con_id", "run_id", "universe", "strategy", "strategy_version", "environment", "account", "t0", "p0", "expected_absolute_return_15m", "m_price", "raw_pre_move", "pre_move_m", "cohort_percentile", "band", "session_hard_score", "structure", "direction", "rank", "entry", "entry_reference", "status", "signal_id", "order_plan_id"];
  main.innerHTML = `${head(item.symbol, "Candidate calculation lineage copied from authoritative Stage 5/6 outputs.")}<section class="section"><div class="detail-grid">${fields.map((key) => `<div class="detail-cell"><span>${esc(key.replaceAll("_", " "))}</span><strong>${key === "environment" ? environment(item[key]) : esc(item[key])}</strong></div>`).join("")}</div></section>`;
}

async function ordersPage() {
  const scope = new URLSearchParams(location.search).get("scope") || "open";
  const data = await api(`/api/orders?scope=${scope}`);
  const tabs = ["open", "today", "rejected", "all"].map((name) => `<button class="${name === scope ? "active" : ""}" data-scope="${name}">${name.toUpperCase()}</button>`).join("");
  const groups = data.items.map((item) => `<article class="order-group"><div class="order-title"><strong>${esc(item.symbol)}</strong>${environment(item.environment)}<span>${esc(item.run_id)}</span><span class="muted">${esc(item.account)}</span></div>${item.orders.map((leg, index) => `<div class="order-leg ${index ? "child" : ""}"><b>${index ? "└─ " : ""}${esc(leg.role)}</b>${badge(leg.status)}<span>${leg.role === "ENTRY" ? `${number(item.filled)} @ ${number(item.average_fill || item.entry)}` : (leg.role === "STOP" ? number(item.stop) : number(item.target))}</span><span>#${esc(leg.ibkr_order_id)}</span></div>`).join("")}<div class="muted">Signal ${esc(item.signal_id)} · Plan ${esc(item.order_plan_id)}${item.rejection_reason ? ` · ${esc(item.rejection_reason)}` : ""}</div></article>`).join("");
  main.innerHTML = `${head("Orders", "Protected orders grouped by plan; broker identifiers remain visible.")}<div class="toolbar tabs">${tabs}</div>${groups || '<div class="empty">No orders in this view.</div>'}`;
  main.querySelectorAll("[data-scope]").forEach((button) => button.addEventListener("click", () => { location.href = `/orders?scope=${button.dataset.scope}`; }));
}
async function positionsPage() {
  const rows = await api("/api/positions");
  const columns = [
    { key: "symbol", label: "Symbol" }, { key: "run_id", label: "Run" }, { label: "Env", render: (row) => environment(row.environment) }, { key: "account", label: "Account" }, { key: "side", label: "Side" },
    { key: "quantity", label: "Qty", numeric: true }, { label: "Average entry", numeric: true, render: (row) => number(row.average_entry) }, { label: "Current", numeric: true, render: (row) => number(row.current_price) },
    { label: "Stop", numeric: true, render: (row) => number(row.stop) }, { label: "Target", numeric: true, render: (row) => number(row.target) }, { label: "Unrealised P/L", numeric: true, render: (row) => money(row.unrealised_pnl) }, { label: "Source", render: (row) => badge(row.source, row.source === "IBKR_RECONCILED" ? "good" : "warn") },
  ];
  main.innerHTML = `${head("Positions", "Open exposure is shown only from reconciled Stage 7 broker-normalized state.")}<section class="section">${table(columns, rows)}</section>`;
}
async function tradesPage() {
  const data = await api("/api/trades?limit=100");
  const summary = data.summary;
  const metrics = [["Trades", summary.trades], ["Wins", summary.wins], ["Losses", summary.losses], ["Win %", summary.win_percent], ["Total P/L", money(summary.total_pnl)], ["Total R", summary.total_r], ["Mean R", summary.mean_r]].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${typeof value === "number" ? number(value) : esc(value)}</strong></div>`).join("");
  const columns = [{ key: "date_time", label: "Date/time" }, { key: "run_id", label: "Run" }, { label: "Env", render: (row) => environment(row.environment) }, { key: "symbol", label: "Symbol" }, { key: "side", label: "Side" }, { key: "entry", label: "Entry", numeric: true }, { key: "exit", label: "Exit", numeric: true }, { key: "quantity", label: "Qty", numeric: true }, { label: "P/L", numeric: true, render: (row) => money(row.pnl) }, { key: "r", label: "R", numeric: true }, { key: "exit_reason", label: "Exit reason" }];
  main.innerHTML = `${head("Trades", "Completed execution ledger. PAPER and LIVE remain explicitly labelled.")}<div class="toolbar"><button>Today</button><button class="secondary">Week</button><button class="secondary">Month</button><select aria-label="Environment"><option>ALL</option><option>PAPER</option><option>LIVE</option></select></div><div class="metric-strip">${metrics}</div><section class="section">${table(columns, data.items)}</section>`;
}
async function systemPage() {
  const data = await api("/api/system");
  const cards = data.environments.map((item) => `<article class="status-card"><span class="eyebrow">IBKR ${esc(item.environment)}</span><strong>${item.connected ? "CONNECTED" : "DISCONNECTED"}</strong><small>${esc(item.account)} · ${item.reconciled ? "RECONCILED" : "NOT RECONCILED"}<br>${item.open_orders} open orders · ${item.positions} positions</small></article>`).join("");
  const problems = data.problems.length ? data.problems.map((message) => `<div class="attention-item"><b>PROBLEM</b><span>${esc(message)}</span></div>`).join("") : '<div class="empty">No execution-system problems.</div>';
  const events = data.events.length ? data.events.map((item) => `<div class="attention-item"><b>${new Date(item.timestamp).toLocaleTimeString()}</b><span>${esc(item.message)}</span></div>`).join("") : '<div class="empty">No recent operational events.</div>';
  main.innerHTML = `${head("System", "IBKR connectivity, reconciliation, runtime health, and concise operational events.")}<section class="status-grid">${cards}<article class="status-card"><span class="eyebrow">RUNTIME</span><strong>${esc(data.application)}</strong><small>${data.runtime.active_runs} active runs</small></article></section><section class="section"><div class="section-head"><h2>System problems</h2></div><div class="attention">${problems}</div></section><section class="section"><div class="section-head"><h2>Recent operational events</h2></div>${events}</section>`;
}
async function settingsPage() {
  const data = await api("/api/settings");
  const broker = data.broker.map((item) => `<div class="detail-cell"><span>IBKR ${esc(item.environment)}</span><strong>${esc(item.account)} · ${item.connected ? "CONNECTED" : "DISCONNECTED"}</strong></div>`).join("");
  const universeColumns = [{ key: "universe_id", label: "Universe" }, { key: "name", label: "Name" }, { key: "members", label: "Members", numeric: true }];
  const strategyColumns = [{ key: "strategy", label: "Strategy" }, { key: "identity", label: "Identity" }, { key: "version", label: "Version" }, { key: "description", label: "Definition" }];
  main.innerHTML = `${head("Settings", "Small backend-owned configuration surface; frozen strategy constants are read-only.")}<section class="section"><div class="section-head"><h2>Broker</h2></div><div class="detail-grid">${broker}</div></section><section class="section"><div class="section-head"><h2>Universes</h2></div>${table(universeColumns, data.universes)}</section><section class="section"><div class="section-head"><h2>Runs</h2><span class="muted">Edit from Run Detail</span></div>${table([{ key: "run_id", label: "Run" }, { key: "universe", label: "Universe" }, { key: "strategy", label: "Strategy" }, { label: "Env", render: (row) => environment(row.environment) }, { label: "Enabled", render: (row) => badge(row.enabled ? "YES" : "NO", row.enabled ? "good" : "warn") }], data.runs, (row) => `run:${row.run_id}`)}</section><section class="section"><div class="section-head"><h2>Strategies</h2></div>${table(strategyColumns, data.strategies)}</section>`;
}

async function render() {
  clearTimeout(timer);
  const name = route();
  activate(name);
  try {
    const cached = await refreshHeader();
    await ({ overview, runs: runsPage, candidates: candidatesPage, orders: ordersPage, positions: positionsPage, trades: tradesPage, system: systemPage, settings: settingsPage })[name](cached);
  } catch (error) {
    main.innerHTML = `${head("Read-only unavailable", "The trading runtime is isolated from this dashboard failure.")}<div class="section error">${esc(error.message)}</div>`;
  }
  timer = setTimeout(render, route() === "orders" || route() === "positions" ? 5000 : 10000);
}
document.addEventListener("click", (event) => {
  const row = event.target.closest("[data-action]");
  if (!row || !row.dataset.action) return;
  const [type, id] = row.dataset.action.split(":");
  if (type === "run") location.href = `/runs?run=${encodeURIComponent(id)}`;
  if (type === "candidate") location.href = `/candidates?signal=${encodeURIComponent(id)}`;
});
document.querySelector("#menu-button").addEventListener("click", (event) => {
  const open = document.querySelector("#sidebar").classList.toggle("open");
  event.currentTarget.setAttribute("aria-expanded", String(open));
});
render();
