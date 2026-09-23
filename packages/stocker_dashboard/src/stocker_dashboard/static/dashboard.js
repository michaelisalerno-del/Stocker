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
async function refresh() {
 try {
  const response = await fetch("/api/overview",{cache:"no-store"});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json(), s = data.system;
  document.getElementById("paper-status").textContent = `${s.account} / ${s.connected ? "CONNECTED" : "DISCONNECTED"}`;
  document.getElementById("system-status").textContent = s.armed ? "ARMED" : "UNARMED";
  document.getElementById("active-runs").textContent = "FIRST4";
  document.getElementById("open-positions").textContent = data.positions.filter(p=>p.quantity).length;
  const path = location.pathname;
  let html = `<div class="page-header"><div><p class="eyebrow">US / IBKR PAPER</p><h1>Frozen FIRST4</h1><p>PRIOR15 &gt; 4.459368321659181% · first four opportunities · buy 98% put + 102% call</p></div><button id="pause">Pause entries</button></div>`;
  if (s.problem || s.missing_settings.length) html += `<section class="panel"><h2>Execution prerequisites</h2><p>${esc(s.problem)}</p><p>${esc(s.missing_settings.join(", "))}</p></section>`;
  if (s.opening_check && Object.keys(s.opening_check).length) html += table("Opening PAPER verification",[s.opening_check],["session","status","started_at","deadline","armed_at","error"]);
  if (["/","/system","/settings"].includes(path)) html += table("PAPER status",[s],["account","connected","reconciled","armed","session","problem"])+table("IBKR PAPER simulated fills / P&L",[data.pnl],["basis","currency","net_cash_flow","realised","actual_fees_usd","fees_complete","reserved_fee_allowance_usd","session_allocation_usd","status"]);
  if (["/","/trades"].includes(path)) html += table("Quoted ask-to-bid comparison (not fills)",data.quote_comparisons || [],["reference","quoted_entry_ask_for_exit_legs_usd","quoted_exit_bid_usd","quoted_ask_to_bid_gross_usd","quotes","exit_quotes"]);
  if (["/","/candidates"].includes(path)) html += table("Candidate decisions and permanent slots",data.candidates.slice(-150),["session","symbol","information_at","rank","prior15","decision","slot","entry_at","outcome","detail"]);
  if (["/","/orders"].includes(path)) html += table("Broker orders",data.orders.slice(-100),["reference","role","order_id","perm_id","status","payload"]);
  if (["/","/positions"].includes(path)) html += table("Actual option legs",data.positions,["con_id","quantity","payload"]);
  if (["/","/trades"].includes(path)) html += table("Actual leg executions",data.fills.slice(-100),["exec_id","reference","con_id","quantity","price","side","commission","time"]);
  if (["/","/system"].includes(path)) html += table("Actionable status",data.errors,["key","value"]);
  if (path==="/settings") html += table("Execution settings",Object.entries(s.settings).map(([key,value])=>({key,value})),["key","value"]);
  document.getElementById("main").innerHTML = html;
  document.getElementById("pause").onclick=async()=>{await fetch("/api/first4/pause",{method:"POST"});await refresh();};
  document.getElementById("last-refresh").textContent = new Date().toLocaleTimeString();
 } catch (error) { document.getElementById("system-status").textContent = `UNAVAILABLE: ${error.message}`; }
}
refresh();setInterval(refresh,3000);
