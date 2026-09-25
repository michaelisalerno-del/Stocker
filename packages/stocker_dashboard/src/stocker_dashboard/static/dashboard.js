"use strict";
const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (value) => typeof value === "number" && Number.isFinite(value);
const money = (value) =>
  number(value)
    ? new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(value)
    : "Unavailable";
const percent = (value) =>
  number(value) ? `${value.toFixed(2)}%` : "Unavailable";
const stamp = (value) =>
  value && !Number.isNaN(Date.parse(value))
    ? new Date(value).toLocaleString("en-GB", {
        timeZone: "America/New_York",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }) + " ET"
    : "—";
function text(element, value) {
  const next = String(value ?? "—");
  if (element.textContent !== next) element.textContent = next;
}
function show(element, visible) {
  if (element.hidden === visible) element.hidden = !visible;
}
function field(root, key, value) {
  text(root.querySelector(`[data-field="${key}"]`), value);
}
function message(id, value) {
  text($(id), value);
  show($(id), Boolean(value));
}
const menu = $("#menu-button");
const sidebar = $("#sidebar");
function setMenuOpen(open) {
  sidebar.classList.toggle("open", open);
  menu.setAttribute("aria-expanded", String(open));
}
menu.onclick = () => setMenuOpen(!sidebar.classList.contains("open"));
sidebar.onclick = (event) => {
  if (event.target.closest("a")) setMenuOpen(false);
};
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sidebar.classList.contains("open")) {
    setMenuOpen(false);
    menu.focus();
  }
});
const route = location.pathname;
const title =
  {
    "/": "Overview",
    "/opportunities": "Opportunities",
    "/execution": "Execution",
    "/system": "System",
  }[route] || "Overview";
const params = new URLSearchParams(location.search);
const historical = Boolean(params.get("session"));
const offset = Math.max(0, Number(params.get("offset")) || 0);
const decision = ["selected", "rejected"].includes(params.get("decision"))
  ? params.get("decision")
  : "all";
const interval =
  route === "/opportunities"
    ? 15000
    : route === "/system" || historical
      ? 30000
      : 5000;
let revision = 0;
let refreshing = false;
let controlling = false;
let uncertain = false;
let paused = false;
let timer;
let activeRead;
let visibilityRefresh = false;
let lastGood;
let lastData;
let selectedDetail;
let detailVersion = 0;
let detailController;
let detailOffset = 0;
let detailMore = false;
let initial = true;
for (const link of sidebar.querySelectorAll("nav a")) {
  if (link.getAttribute("href") === route)
    link.setAttribute("aria-current", "page");
}
document.title = `SLRNO — ${title} / FIRST4 PAPER`;
function pair(label, key) {
  return `<div><dt>${label}</dt><dd data-field="${key}">—</dd></div>`;
}
function tableShell(id, labels) {
  return `<div class="table-wrap" tabindex="0" aria-label="${esc(id)} evidence table"><table id="${id}"><thead><tr>${labels.map((label) => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody></tbody></table></div>`;
}
function cardShell(slot) {
  return `<article class="slot-card waiting" data-slot="${slot}"><span class="label">Slot ${slot}</span><h3 data-field="symbol">Waiting</h3>
    <div class="state" data-field="state">NOT ALLOCATED</div><div class="hero-value" data-field="metric">—</div><span class="label" data-field="metric-label">Permanent allocation</span>
    <dl><div><dt data-field="seen-label">Seen</dt><dd data-field="seen">—</dd></div><div><dt data-field="entry-label">Scheduled entry</dt><dd data-field="entry">—</dd></div><div><dt data-field="trigger-label">PRIOR15 / rank</dt><dd data-field="trigger">—</dd></div></dl>
    <p class="next-step" data-field="next">Awaiting a qualifying first appearance.</p>
    <details><summary>Timing &amp; held legs</summary><dl>${pair("Anchor", "anchor")}${pair("Actual fill", "fill")}${pair("Exit", "exit")}${pair("Elapsed / until exit", "elapsed")}${pair("Actual premium paid", "premium")}${pair("Fees", "fees")}${pair("Remaining legs", "legs")}${pair("Return on actual premium paid", "return")}</dl><small data-field="caveat">—</small></details>
    ${First4Flow.cardShell()}
    <button class="secondary allocation-detail" disabled>Allocation evidence</button></article>`;
}
function evidenceShell() {
  return `<section id="evidence" class="section evidence" hidden><div class="section-head"><h2 id="evidence-title">Allocation evidence</h2><button id="close-detail" class="secondary">Close detail</button></div>
    <p id="detail-receipt" class="muted"></p><p id="detail-error" class="warning" hidden></p><button id="reload-detail" class="secondary">Refresh detail</button>
    ${First4Flow.detailShell()}
    <h3>Order reservations &amp; broker status</h3>${tableShell("detail-orders", ["Reference", "Role", "Broker order ID", "Status", "Completion verified"])}
    <h3>Quoted comparisons — not broker P&amp;L</h3>${tableShell("detail-quotes", ["Order", "Entry ask for exit legs", "Exit bid", "Quoted gross difference"])}
    <h3>Actual PAPER leg executions</h3>${tableShell("detail-fills", ["Execution", "Contract ID", "Side", "Quantity", "Price", "Multiplier", "Commission", "Time", "Superseded"])}
    <div class="pager"><button id="detail-prev" class="secondary">Previous evidence</button><span id="detail-page"></span><button id="detail-next" class="secondary">Next evidence</button></div>
    <p>Corrections remain visible. Superseded executions are excluded from accounting. Evidence pages never limit financial totals.</p>
    <details><summary>Technical evidence / quoted comparisons</summary><p>Quotes are comparisons, not realised broker P&amp;L or current valuation.</p><pre id="detail-json"></pre></details></section>`;
}
const subtitle = {
  "/": "Four permanent allocations. Selection is distinct from execution.",
  "/opportunities":
    "Frozen first appearances and final admission decisions. No reconsideration.",
  "/execution":
    "Selection, actual leg fills, remaining exposure and results — by allocation.",
  "/system": "Connectivity, entry authority and operational evidence.",
}[route];
let shell = `<div class="page-head"><div><span class="eyebrow">FIRST4 / US PAPER</span><h1>${title}</h1><p class="muted">${subtitle}</p></div><span id="view-session" class="label"></span></div><p id="warning" class="warning" hidden></p>`;
if (["/opportunities", "/execution"].includes(route)) {
  shell += `<form class="toolbar" method="get"><label>Session (blank = current)<input type="date" name="session" value="${esc(params.get("session") || "")}"></label>`;
  if (route === "/opportunities")
    shell += `<label>Decisions<select name="decision"><option value="all">All</option><option value="selected">Selected</option><option value="rejected">Rejected</option></select></label>`;
  shell += `<button>Load session</button><a href="${route}">Current session</a></form><p class="muted">${historical ? "Historical snapshot. Reload explicitly for updated evidence; global status still refreshes." : "Current session updates automatically."}</p>`;
}
if (route === "/" || route === "/execution") {
  shell += `<section class="section"><div class="section-head"><h2>Permanent FIRST4 allocations</h2><span id="slot-count" class="label">0 / 4 consumed</span></div><div class="slot-grid">${[1, 2, 3, 4].map(cardShell).join("")}</div></section>`;
  if (route === "/")
    shell += `<section class="section"><div class="section-head"><h2>Top opportunity snapshots</h2><a href="/opportunities">All decisions →</a></div><p class="muted">Nearest to Q5 at first observation. These are final decisions, not a live watchlist. PRIOR15 can decrease; proximity is not a probability.</p><p id="q5-rule" class="label"></p><div id="snapshots" class="snapshot-grid"></div><p id="no-snapshots" hidden>No suitable candidate snapshots recorded.</p></section>`;
  shell += `<section class="section"><div class="section-head"><h2>Session economics</h2><span class="label">Actual PAPER fills / USD</span></div><div class="metrics">${[
    ["realised", "Realised / known fees"],
    ["cash", "Net cash flow"],
    ["fees", "Reported fees"],
    ["exposed", "Allocations with exposure"],
  ]
    .map(
      ([key, label]) =>
        `<div class="metric"><span class="label">${label}</span><strong id="economics-${key}">—</strong></div>`,
    )
    .join("")}</div><p id="economics-note" class="muted"></p></section>`;
}
if (route === "/opportunities")
  shell += `<section class="section"><h2>First-appearance decisions</h2><p class="muted">Displayed newest first; ties use scanner rank and symbol. This ordering never changes allocation.</p>${tableShell("decisions", ["Symbol / Slot", "Seen", "Scanner rank", "PRIOR15", "Decision", "Outcome", "Scheduled entry", "Evidence"])}<div class="pager"><a id="previous">← Newer page</a><span id="page-count"></span><a id="next">Older page →</a></div></section>`;
if (route === "/system")
  shell += `<div class="section system-grid"><section><h2>Entry authority</h2><dl>${pair("Entry blocker", "entry_block_reason")}${pair("Configured armed", "configured_armed")}${pair("Worker / manager", "workers")}${pair("Outstanding obligations", "outstanding_obligations")}${pair("Prerequisites", "missing")}</dl></section><section><h2>Underlying data freshness</h2><dl>${pair("Last scanner observation", "last_scanner_observation")}${pair("Option quote state", "option_quote_state")}${pair("Last option quote check", "last_option_quote_check")}${pair("Upstream", "upstream_status")}${pair("Data problem", "data_problem")}</dl><p class="muted">Dashboard receipt does not prove scanner, broker or quote freshness.</p></section></div><section class="section"><h2>Opening verification</h2><pre id="opening"></pre><details><summary>Read-only configuration</summary><pre id="configuration"></pre></details><details><summary>Technical diagnostics</summary><pre id="diagnostics"></pre><p>Obligations and broker positions: up to 50 each, explicitly paginated. Reconciliation always uses the complete ledger.</p><div class="pager"><a id="diagnostic-prev">Previous diagnostics</a><span id="diagnostic-offset"></span><a id="diagnostic-next">Next diagnostics</a></div></details></section>`;
if (route !== "/system") shell += evidenceShell();
$("#main").innerHTML = shell; // Once per navigation. Automatic refresh never replaces this shell.
if ($("select[name=decision]")) $("select[name=decision]").value = decision;

// Stable record rows and stable cells; untouched text keeps focus and selection.
function updateTable(id, rows, identity, values) {
  const body = $(`#${id} tbody`);
  const existing = new Map(
    [...body.children].map((row) => [row.dataset.key, row]),
  );
  rows.forEach((record, index) => {
    const key = String(identity(record));
    let row = existing.get(key);
    if (!row) {
      row = document.createElement("tr");
      row.dataset.key = key;
    }
    existing.delete(key);
    const cells = values(record);
    cells.forEach((value, column) => {
      let cell = row.children[column];
      if (!cell) {
        cell = document.createElement("td");
        row.append(cell);
      }
      if (value && typeof value === "object" && value.symbol) {
        let button = cell.querySelector("button");
        if (!button) {
          button = document.createElement("button");
          button.className = "secondary";
          text(button, "Inspect");
          cell.append(button);
        }
        button.onclick = () => openDetail(value.session, value.symbol, button);
      } else text(cell, value);
      const numeric = number(value);
      cell.classList.toggle("num", numeric);
    });
    row.classList.toggle("selected", Boolean(record.slot));
    if (body.children[index] !== row)
      body.insertBefore(row, body.children[index] || null);
  });
  for (const row of existing.values()) row.remove();
}
function duration(item) {
  if (!item.has_exposure || !item.actual_entry_at) return "—";
  const elapsed = Math.max(
    0,
    Math.floor((Date.now() - Date.parse(item.actual_entry_at)) / 60000),
  );
  const until = item.intended_exit_at
    ? Math.ceil((Date.parse(item.intended_exit_at) - Date.now()) / 60000)
    : null;
  return `${elapsed}m / ${until === null ? "unknown" : until < 0 ? `${-until}m overdue` : `${until}m left`}`;
}
function renderSlots(items) {
  let used = 0;
  for (const card of document.querySelectorAll(".slot-card")) {
    const item = items.find((row) => row.slot === Number(card.dataset.slot));
    const allocated = Boolean(item?.symbol);
    used += allocated ? 1 : 0;
    card.classList.toggle("waiting", !allocated);
    field(card, "symbol", item?.symbol || "Waiting");
    field(card, "state", allocated ? item.state : "NOT ALLOCATED");
    const result = item?.realised ?? item?.provisional_result;
    const metric = item?.closed
      ? money(result)
      : item?.has_exposure
        ? money(item.actual_premium_paid)
        : allocated
          ? percent(item.prior15)
          : "—";
    field(card, "metric", metric);
    const metricEl = card.querySelector('[data-field="metric"]');
    metricEl.classList.toggle("positive", Boolean(item?.closed && result >= 0));
    metricEl.classList.toggle("negative", Boolean(item?.closed && result < 0));
    field(
      card,
      "metric-label",
      item?.closed
        ? item.fees_complete
          ? "Realised result"
          : "Provisional result / fees pending"
        : item?.has_exposure
          ? "Actual premium paid"
          : allocated
            ? "Trigger PRIOR15"
            : "Permanent allocation",
    );
    const held =
      item?.legs
        ?.filter((leg) => leg.remaining)
        .map((leg) => {
          const contract = item.contracts?.find((c) => c.conId === leg.con_id);
          return `${contract?.right || leg.con_id}${contract?.strike ? " " + contract.strike : ""} × ${leg.remaining}`;
        })
        .join(" / ") || "None";
    field(
      card,
      "seen-label",
      item?.closed ? "Actual entry" : item?.has_exposure ? "Held legs" : "Seen",
    );
    field(
      card,
      "seen",
      item?.closed
        ? stamp(item.actual_entry_at)
        : item?.has_exposure
          ? held
          : stamp(item?.information_at),
    );
    field(
      card,
      "entry-label",
      item?.closed
        ? "Actual exit"
        : item?.has_exposure
          ? "Elapsed / exit"
          : "Scheduled entry",
    );
    field(
      card,
      "entry",
      item?.closed
        ? stamp(item.actual_exit_at)
        : item?.has_exposure
          ? duration(item)
          : stamp(item?.entry_at),
    );
    field(
      card,
      "trigger-label",
      item?.closed
        ? "Return / premium"
        : item?.has_exposure
          ? "Reported fees"
          : "PRIOR15 / rank",
    );
    field(
      card,
      "trigger",
      item?.closed
        ? `${percent(item.return_pct)}${item.pending_fees ? " provisional" : ""}`
        : item?.has_exposure
          ? `${money(item.fees)}${item.pending_fees ? " (pending)" : ""}`
          : allocated
            ? `${percent(item.prior15)} / #${item.rank ?? "—"}`
            : "—",
    );
    field(
      card,
      "next",
      allocated ? [item.position_explanation, item.next_step].filter(Boolean).join(" · ") : "Awaiting a qualifying first appearance.",
    );
    field(
      card,
      "anchor",
      number(item?.anchor)
        ? `${item.anchor} · ${stamp(item.anchor_observed_at)}`
        : "Unavailable",
    );
    field(card, "fill", stamp(item?.actual_entry_at));
    field(
      card,
      "exit",
      stamp(item?.closed ? item.actual_exit_at : item?.intended_exit_at),
    );
    field(card, "elapsed", allocated ? duration(item) : "—");
    field(card, "premium", allocated ? money(item.actual_premium_paid) : "—");
    field(
      card,
      "fees",
      allocated
        ? `${money(item.fees)}${item.pending_fees ? " · pending commissions" : ""}`
        : "—",
    );
    field(
      card,
      "legs",
      item?.legs
        ?.filter((leg) => leg.remaining)
        .map((leg) => `${leg.con_id}: ${leg.remaining}`)
        .join(" / ") || "None",
    );
    field(card, "return", percent(item?.return_pct));
    field(
      card,
      "caveat",
      item?.has_exposure
        ? "UNREALISED P&L UNAVAILABLE. No current valuation."
        : item?.closed
          ? `Actual PAPER fills. ${item.fees_complete ? "All commissions reported." : "Return is provisional until commissions arrive."}`
          : "Selection does not imply a fill or an open position.",
    );
    First4Flow.renderCard(card, item?.order_flow);
    const button = card.querySelector("button");
    button.disabled = !allocated;
    button.onclick = () => openDetail(item.session, item.symbol, button);
  }
  text($("#slot-count"), `${used} / 4 consumed`);
}
function renderSnapshots(items, q5) {
  text(
    $("#q5-rule"),
    `Q5 ${q5}% · actual rule PRIOR15 > Q5 (strictly greater)`,
  );
  const container = $("#snapshots");
  items.forEach((item, index) => {
    let card = container.children[index];
    if (!card) {
      card = document.createElement("article");
      card.className = "snapshot";
      card.innerHTML = `<div class="section-head"><h3 data-field="symbol"></h3><span class="label" data-field="rank"></span></div><span class="label">Rejected first appearance</span><p class="value" data-field="prior"></p><p data-field="distance"></p><small data-field="decision"></small><p class="muted" data-field="seen"></p>`;
      container.append(card);
    }
    const valid = number(item.prior15) && number(q5) && q5 > 0;
    field(card, "symbol", item.symbol);
    field(card, "rank", `RANK ${item.rank ?? "—"}`);
    field(card, "prior", `${percent(item.prior15)} / ${percent(q5)}`);
    const distance = !valid
      ? "PRIOR15 unavailable"
      : item.prior15 === q5
        ? "Equal to Q5 — does not pass strict threshold"
        : `${Math.abs(item.prior15 - q5).toFixed(3)} percentage points ${item.prior15 > q5 ? "above" : "below"} Q5`;
    field(card, "distance", distance);
    field(
      card,
      "decision",
      `${item.decision} · ${item.outcome} · Rejected for this session — not reconsidered`,
    );
    field(
      card,
      "seen",
      `Snapshot ${stamp(item.information_at)} · not re-evaluated`,
    );
  });
  while (container.children.length > items.length) container.lastChild.remove();
  show($("#no-snapshots"), !items.length);
}
function renderEconomics(pnl) {
  text($("#economics-realised"), money(pnl.realised));
  text($("#economics-cash"), money(pnl.net_cash_flow));
  text($("#economics-fees"), money(pnl.actual_fees_usd));
  text(
    $("#economics-exposed"),
    `${pnl.exposed_allocations} / ${pnl.allocations_used}`,
  );
  text(
    $("#economics-note"),
    `Session only. Cash flow is not profit. ${pnl.pending_fee_executions || 0} commissions pending. Unrealised P&L unavailable. Reserved fee allowance ${money(pnl.reserved_fee_allowance_usd)}; reserved allocation ${money(pnl.session_allocation_usd)}.`,
  );
}
function renderStatus(s) {
  paused = s.paused === true;
  text($("#paper-status"), s.connected ? "CONNECTED" : "DISCONNECTED");
  text($("#reconciliation"), s.reconciled ? "RECONCILED" : "REQUIRED");
  text($("#system-status"), paused ? "PAUSED" : s.armed ? "ARMED" : "UNARMED");
  text($("#current-session"), s.session || "No active session");
  const warnings = [];
  if (!s.connected) warnings.push("PAPER disconnected");
  if (!s.reconciled) warnings.push("Reconciliation required");
  if (s.ledger_available === false) warnings.push("Ledger unavailable");
  if (s.problem) warnings.push(s.problem);
  if (s.worker_health === "FAILED" || s.manager_health === "FAILED")
    warnings.push("Execution worker/manager failure — check System");
  message("#warning", warnings.join(" · "));
  $("#pause").disabled = controlling || uncertain || paused;
}
function render(data) {
  renderStatus(data.system || data);
  if (route === "/system") {
    for (const key of [
      "entry_block_reason",
      "configured_armed",
      "outstanding_obligations",
      "option_quote_state",
      "upstream_status",
      "data_problem",
    ])
      field(
        $("#main"),
        key,
        typeof data[key] === "boolean" ? (data[key] ? "Yes" : "No") : data[key],
      );
    field(
      $("#main"),
      "workers",
      `${data.worker_health} / ${data.manager_health}`,
    );
    field($("#main"), "missing", data.missing_settings?.join(", ") || "None");
    for (const key of ["last_scanner_observation", "last_option_quote_check"])
      field($("#main"), key, stamp(data[key]));
    const check = data.opening_check || {};
    const verification = data.opening_check_active
      ? `Verification active · ${check.attempt || 0} attempts · ${Math.ceil(data.opening_remaining_seconds || 0)}s remaining`
      : `Historical opening result: ${check.status || "None"} · ${check.attempt || 0} attempts (not current authorisation)`;
    text($("#opening"), `Current readiness: ${data.armed ? "Armed" : "Unarmed"}\n${verification}\n\n${JSON.stringify(check, null, 2)}`);
    text($("#configuration"), JSON.stringify(data.settings || {}, null, 2));
    const { settings, opening_check, ...diagnostics } = data;
    text($("#diagnostics"), JSON.stringify(diagnostics, null, 2));
    text($("#diagnostic-offset"), `Offset ${offset}`);
    $("#diagnostic-prev").href = `/system?offset=${Math.max(0, offset - 50)}`;
    $("#diagnostic-next").href = `/system?offset=${offset + 50}`;
    show($("#diagnostic-prev"), offset > 0);
    show(
      $("#diagnostic-next"),
      Math.max(
        data.obligations?.length || 0,
        data.broker_positions?.length || 0,
      ) === 50,
    );
  } else {
    text(
      $("#view-session"),
      `${historical ? "Historical" : "Viewing"} session ${data.session || "—"}`,
    );
    if (route === "/opportunities") {
      updateTable(
        "decisions",
        data.rows,
        (r) => `${r.session}/${r.symbol}`,
        (r) => [
          `${r.symbol}${r.slot ? ` / SLOT ${r.slot}` : ""}`,
          stamp(r.information_at),
          r.rank,
          r.prior15_explanation || percent(r.prior15),
          [r.decision, r.decision_explanation].filter(Boolean).join(" · "),
          r.outcome,
          stamp(r.entry_at),
          { session: r.session, symbol: r.symbol },
        ],
      );
      text(
        $("#page-count"),
        `Records ${offset + 1}–${offset + data.rows.length}`,
      );
      for (const [id, nextOffset, visible] of [
        ["previous", Math.max(0, offset - 50), offset > 0],
        ["next", offset + 50, data.has_more],
      ]) {
        const query = new URLSearchParams(params);
        query.set("offset", nextOffset);
        $(`#${id}`).href = `${route}?${query}`;
        show($(`#${id}`), visible);
      }
    } else {
      renderSlots(data.slots || data.allocations);
      renderEconomics(data.pnl);
      if (route === "/") renderSnapshots(data.candidates, data.q5);
    }
  }
}
async function request(url, options = {}, controller = new AbortController()) {
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(url, {
      ...options,
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}
function schedule(delay = interval) {
  clearTimeout(timer);
  if (!document.hidden && !controlling) timer = setTimeout(refresh, delay);
}
async function refresh() {
  clearTimeout(timer);
  if (refreshing || controlling || document.hidden) return;
  refreshing = true;
  const version = revision;
  activeRead = new AbortController();
  try {
    const statusOnly = historical && !initial;
    const endpoint = statusOnly
      ? "/api/status"
      : {
          "/": "/api/overview",
          "/opportunities": "/api/opportunities",
          "/execution": "/api/execution",
          "/system": "/api/system",
        }[route];
    const query = new URLSearchParams(params);
    query.set("limit", "50");
    const data = await request(`${endpoint}?${query}`, {}, activeRead);
    if (version !== revision || document.hidden) return;
    if (uncertain) {
      uncertain = false;
      message(
        "#control-result",
        `Previous pause outcome was uncertain. Authoritative read: ${data.paused || data.system?.paused ? "PAUSED" : "NOT PAUSED"}. No automatic retry was made.`,
      );
    }
    if (statusOnly) renderStatus(data);
    else {
      render(data);
      lastData = data;
    }
    initial = false;
    lastGood = new Date();
    text(
      $("#last-refresh"),
      `Dashboard received ${lastGood.toLocaleTimeString()} · CURRENT`,
    );
    message("#transport-error", "");
  } catch (error) {
    if (version === revision && !document.hidden) {
      const receipt = lastGood ? lastGood.toLocaleTimeString() : "never";
      text($("#last-refresh"), `STALE · last success ${receipt}`);
      message(
        "#transport-error",
        `Dashboard update failed (${error.name === "AbortError" ? "read timed out" : error.message}). Last success ${receipt}. Keeping last good data; underlying data may also be stale.`,
      );
    }
  } finally {
    refreshing = false;
    activeRead = null;
    schedule(visibilityRefresh ? 0 : interval);
    visibilityRefresh = false;
  }
}
document.addEventListener("visibilitychange", () => {
  clearTimeout(timer);
  if (document.hidden) {
    revision++;
    activeRead?.abort();
    text(
      $("#last-refresh"),
      `Polling suspended · last success ${lastGood?.toLocaleTimeString() || "never"}`,
    );
  } else if (!refreshing) schedule(0);
  else visibilityRefresh = true;
  // If an abort is settling, its finally schedules the only next cycle.
});
$("#pause").onclick = () => {
  if (!controlling && !uncertain && !paused) $("#pause-dialog").showModal();
};
$("#cancel-pause").onclick = () => $("#pause-dialog").close();
$("#confirm-pause").onclick = pauseEntries;
async function pauseEntries() {
  if (controlling || uncertain || paused || !$("#pause-dialog").open) return;
  controlling = true;
  revision++;
  clearTimeout(timer);
  activeRead?.abort();
  $("#pause").disabled = true;
  $("#confirm-pause").disabled = true;
  $("#cancel-pause").disabled = true;
  try {
    const state = await request("/api/first4/pause", { method: "POST" });
    if (!state.paused || state.armed || !state.ledger_available)
      throw new Error("Pause not confirmed by authoritative state");
    renderStatus(state);
    message(
      "#control-result",
      "Entries PAUSED. Existing position and exit management continues.",
    );
  } catch (error) {
    uncertain = true;
    message(
      "#control-result",
      `Pause outcome uncertain (${error.message}). Checking authoritative state; no automatic retry.`,
    );
    try {
      const state = await request("/api/status");
      uncertain = false;
      renderStatus(state);
      message(
        "#control-result",
        `Pause request failed or was uncertain (${error.message}). Authoritative state: ${state.paused ? "PAUSED" : "NOT PAUSED"}. No automatic retry.`,
      );
    } catch {
      message(
        "#control-result",
        `Pause outcome uncertain (${error.message}); state check also failed. Retry disabled until an authoritative read succeeds.`,
      );
    }
  } finally {
    controlling = false;
    $("#confirm-pause").disabled = false;
    $("#cancel-pause").disabled = false;
    $("#pause-dialog").close();
    $("#pause").disabled = uncertain || paused;
    schedule(0);
  }
}
let detailTrigger;
async function openDetail(session, symbol, trigger, offset = 0) {
  const version = ++detailVersion;
  detailController?.abort();
  detailController = new AbortController();
  const changed = selectedDetail !== `${session}/${symbol}`;
  selectedDetail = `${session}/${symbol}`;
  detailTrigger = trigger;
  detailOffset = offset;
  detailMore = false;
  if (changed) {
    updateTable(
      "detail-quotes",
      [],
      (r) => r.reference,
      () => [],
    );
    updateTable(
      "detail-orders",
      [],
      (r) => r.reference,
      () => [],
    );
    updateTable(
      "detail-fills",
      [],
      (r) => r.exec_id,
      () => [],
    );
    text($("#detail-json"), "");
    text($("#detail-receipt"), "Loading evidence…");
  }
  show($("#evidence"), true);
  text($("#evidence-title"), `${symbol} / ${session} — evidence`);
  for (const card of document.querySelectorAll(".slot-card"))
    card.classList.toggle("chosen", card.querySelector("button") === trigger);
  $("#reload-detail").disabled = true;
  $("#detail-prev").disabled = true;
  $("#detail-next").disabled = true;
  message("#detail-error", "");
  try {
    const query = new URLSearchParams({ session, symbol, offset, limit: 50 });
    const data = await request(`/api/detail?${query}`, {}, detailController);
    if (version !== detailVersion) return;
    updateTable(
      "detail-orders",
      data.orders,
      (r) => r.reference,
      (r) => [
        r.reference,
        r.role,
        r.order_id,
        r.status,
        r.obligation_done ? "Yes" : "No",
      ],
    );
    updateTable(
      "detail-quotes",
      data.orders.filter((order) => order.role === "EXIT"),
      (r) => r.reference,
      (r) => [
        r.reference,
        money(r.payload?.quoted_entry_ask_for_exit_legs_usd),
        money(r.payload?.quoted_exit_bid_usd),
        money(r.payload?.quoted_ask_to_bid_gross_usd),
      ],
    );
    updateTable(
      "detail-fills",
      data.fills,
      (r) => r.exec_id,
      (r) => [
        r.exec_id,
        r.con_id,
        r.side,
        r.quantity,
        money(r.price),
        r.multiplier,
        number(r.commission) ? money(r.commission) : "Pending",
        stamp(r.time),
        r.superseded ? "Yes" : "No",
      ],
    );
    First4Flow.renderDetail(data.order_flow, changed);
    text($("#detail-json"), JSON.stringify(data, null, 2));
    text(
      $("#detail-receipt"),
      `Evidence snapshot received ${new Date().toLocaleTimeString()}. Refresh explicitly for new executions.`,
    );
    text($("#detail-page"), `Evidence offset ${offset}`);
    detailMore = data.has_more;
  } catch (error) {
    if (version === detailVersion)
      message(
        "#detail-error",
        `Evidence unavailable: ${error.message}. Any retained evidence is stale.`,
      );
  } finally {
    if (version === detailVersion) {
      $("#reload-detail").disabled = false;
      $("#detail-prev").disabled = offset === 0;
      $("#detail-next").disabled = !detailMore;
    }
  }
}
if ($("#evidence")) {
  $("#close-detail").onclick = () => {
    detailVersion++;
    detailController?.abort();
    selectedDetail = null;
    show($("#evidence"), false);
    detailTrigger?.focus();
  };
  function reloadDetail(offset) {
    if (!selectedDetail) return;
    const slash = selectedDetail.indexOf("/");
    openDetail(
      selectedDetail.slice(0, slash),
      selectedDetail.slice(slash + 1),
      detailTrigger,
      offset,
    );
  }
  $("#reload-detail").onclick = () => reloadDetail(detailOffset);
  $("#detail-prev").onclick = () =>
    reloadDetail(Math.max(0, detailOffset - 50));
  $("#detail-next").onclick = () => reloadDetail(detailOffset + 50);
}
refresh();
