"use strict";
const menuButton = document.getElementById("menu-button");
const sidebar = document.getElementById("sidebar");
function setMenuOpen(open) {
 sidebar.classList.toggle("open", open);
 menuButton.setAttribute("aria-expanded", String(open));
}
menuButton.addEventListener("click", () => setMenuOpen(!sidebar.classList.contains("open")));
sidebar.addEventListener("click", event => {
 if (event.target.closest("a")) setMenuOpen(false);
});
document.addEventListener("keydown", event => {
 if (event.key === "Escape" && sidebar.classList.contains("open")) {
  setMenuOpen(false);
  menuButton.focus();
 }
});
const esc = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function table(title, rows, keys) {
 return `<section class="panel"><h2>${esc(title)}</h2><div class="table-wrap"><table><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${keys.map(k=>`<td>${esc(typeof r[k] === "object" ? JSON.stringify(r[k]) : r[k])}</td>`).join("")}</tr>`).join("") || `<tr><td colspan="${keys.length}">No records</td></tr>`}</tbody></table></div></section>`;
}
let refreshing = false;
let controlling = false;
let revision = 0;
async function request(url, options = {}) {
 const controller = new AbortController();
 const timer = setTimeout(() => controller.abort(), 8000);
 try {
  const response = await fetch(url, {...options, cache:"no-store", signal:controller.signal});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json();
 } finally { clearTimeout(timer); }
}
async function pauseEntries() {
 if (controlling) return;
 controlling = true;
 revision++;
 const button = document.getElementById("pause");
 button.disabled = true;
 try {
  const result = await request("/api/first4/pause", {method:"POST"});
  if (result.armed || !result.ledger_available) throw new Error("Pause was not confirmed");
  document.getElementById("system-status").textContent = "PAUSED";
 } catch(error) {
  document.getElementById("system-status").textContent = `PAUSE FAILED: ${error.message}`;
 } finally {
  controlling = false;
  button.disabled = false;
 }
}
async function refresh() {
 if (refreshing || controlling) return;
 refreshing = true;
 const version = revision;
 try {
  const params = new URLSearchParams(location.search);
  params.set("view", {"/system":"system","/settings":"settings","/candidates":"candidates","/orders":"orders","/positions":"positions","/trades":"trades"}[location.pathname] || "all");
  const data = await request(`/api/overview?${params}`), s = data.system;
  if (version !== revision) return;
  document.getElementById("paper-status").textContent = `${s.account} / ${s.connected ? "CONNECTED" : "DISCONNECTED"}`;
  document.getElementById("system-status").textContent = s.market_data_block?.code === 10197 ? "DATA BLOCKED (10197)" : s.armed ? "ARMED" : "UNARMED";
  document.getElementById("active-runs").textContent = "FIRST4";
  document.getElementById("open-positions").textContent = data.positions.filter(p=>p.quantity).length;
  const path = location.pathname;
  let html = `<div class="page-header"><div><p class="eyebrow">US / IBKR PAPER</p><h1>Frozen FIRST4</h1><p>PRIOR15 &gt; 4.459368321659181% · first four opportunities · buy 98% put + 102% call</p></div><button id="pause">Pause entries</button></div><form method="get"><label>Session <input type="date" name="session" value="${esc(data.session)}"></label><button>View session</button></form><p>Showing up to ${esc(data.limit)} records per table. <a href="?session=${encodeURIComponent(data.session || "")}&offset=${(data.offset || 0) + (data.limit || 150)}">Older records</a> · <a href="${esc(location.pathname)}">Current session</a></p>`;
  if (s.problem || s.missing_settings.length) html += `<section class="panel"><h2>Execution prerequisites</h2><p>${esc(s.problem)}</p><p>${esc(s.missing_settings.join(", "))}</p></section>`;
  if (s.market_data_block?.code === 10197) html += table("Market data blocked — competing session (10197)",[s.market_data_block],["time","request_id","message","contract"]);
  if (s.opening_check && Object.keys(s.opening_check).length) html += table("Opening PAPER verification",[s.opening_check],["session","status","started_at","deadline","armed_at","error"]);
  if (["/","/system","/settings"].includes(path)) html += table("PAPER status",[s],["account","worker_health","manager_health","web_health","connected","upstream_available","upstream_status","data_problem","option_quote_state","last_option_quote_check","reconciled","configured_armed","armed","last_scanner_observation","entry_block_reason","outstanding_obligations","operator_exceptions"])+table("IBKR PAPER simulated fills / P&L",[data.pnl],["basis","currency","net_cash_flow","net_cash_flow_basis","completed_gross_usd","realised","realised_basis","partial_close_gross_usd","open_owned_legs","actual_fees_usd","fees_complete","pending_fee_executions","completed_pending_fee_allocations","reserved_fee_allowance_usd","session_allocation_usd"]);
  if (["/","/positions","/system"].includes(path)) html += table("Outstanding exit obligations",data.obligations || [],["reference","session","symbol","status"]);
  if (["/","/trades"].includes(path)) html += table("Quoted ask-to-bid comparison (not fills)",data.quote_comparisons || [],["reference","quoted_entry_ask_for_exit_legs_usd","quoted_exit_bid_usd","quoted_ask_to_bid_gross_usd","quotes","exit_quotes"]);
  if (["/","/candidates"].includes(path)) html += table("Candidate decisions and permanent slots",data.candidates,["session","symbol","information_at","rank","prior15","decision","slot","entry_at","outcome","detail"]);
  if (["/","/orders"].includes(path)) html += table("Broker orders",data.orders,["reference","role","order_id","perm_id","status","payload"]);
  if (["/","/positions"].includes(path)) html += table("Actual option legs",data.positions,["con_id","quantity","payload"]);
  if (["/","/trades"].includes(path)) html += table("Actual leg executions",data.fills,["exec_id","reference","con_id","quantity","price","side","commission","time"]);
  if (["/","/system"].includes(path)) html += table("Actionable status",data.errors,["key","value"]);
  if (path==="/settings") html += table("Execution settings",Object.entries(s.settings).map(([key,value])=>({key,value})),["key","value"]);
  document.getElementById("main").innerHTML = html;
  document.getElementById("pause").onclick=pauseEntries;
  document.getElementById("last-refresh").textContent = new Date().toLocaleTimeString();
 } catch (error) {
  if (version === revision) {
   document.getElementById("system-status").textContent = `UNAVAILABLE: ${error.message}`;
   document.getElementById("last-refresh").textContent = "STALE — refresh failed";
   try {
    const health = await request("/api/system");
    document.getElementById("paper-status").textContent = `Worker ${health.worker_health} / manager ${health.manager_health} / ${health.entry_block_reason}`;
   } catch { document.getElementById("paper-status").textContent = "Health unavailable"; }
  }
 } finally { refreshing = false; }
}
refresh();setInterval(refresh,3000);
