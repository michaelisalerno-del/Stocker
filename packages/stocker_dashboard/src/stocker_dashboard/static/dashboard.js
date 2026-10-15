"use strict";
// SLRNO paper record. Display only: the server never places an order.
const $ = (id) => document.getElementById(id);
const route = location.pathname.replace(/^\/+|\/+$/g, "") || "today";
const page = ["today", "record", "system"].includes(route) ? route : "today";
const REFRESH_SECONDS = page === "today" ? 5 : 30;
const EVENTS = {
  FIRST_CANDLE: "First candle · CFD trades",
  FIRST_CANDLE_QQQ_ENTRY: "First candle · QQQ option bought",
  FIRST_CANDLE_QQQ_EXIT: "First candle · QQQ option sold",
  CONNORS_QQQ: "Connors check · QQQ call",
  FIRST_CANDLE_QQQ_LATE: "QQQ option 15:55 mark (reported)",
  CONNORS_QQQ_EXIT: "Connors · QQQ call sold",
  CFD_EXIT: "CFD trades closed",
};
const MARKETS = {NAS100: "Nasdaq CFD", US500: "S&P CFD"};
const METHODS = {"CFD NAS100": "Nasdaq CFD · first candle", "CFD US500": "S&P CFD · first candle", FIRST_CANDLE_QQQ: "QQQ option · first candle", CONNORS_QQQ: "QQQ call · Connors"};
const USD = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"});
let offset = 0, pending = null, lastRefresh = "";

const text = (id, value) => { const el = $(id); if (el && el.textContent !== String(value)) el.textContent = String(value); };
const show = (id, on) => { const el = $(id); if (el) el.hidden = !on; };
const display = (code) => code ? String(code).replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase()) : "";
const clockAt = (iso, zone) => iso ? new Date(iso).toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit", timeZone: zone}) : "—";
const ny = (iso) => clockAt(iso, "America/New_York");
const uk = (iso) => clockAt(iso, "Europe/London");
const money = (v) => v == null ? "—" : (v > 0 ? "+" : "") + USD.format(v);
const tone = (v) => v == null ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "";
const side = (s) => s > 0 ? "Long" : s < 0 ? "Short" : "None";
const price = (v) => v == null ? "—" : Number(v).toFixed(2);

function rows(id, items, columns, classes = {}) {
  const body = $(id);
  body.replaceChildren(...items.map((item) => {
    const tr = document.createElement("tr");
    const labels = [...body.closest("table").querySelectorAll("thead th")].map((th) => th.textContent);
    columns.forEach((f, i) => {
      const td = document.createElement("td");
      td.dataset.label = labels[i] || "";
      td.textContent = String(f(item) ?? "—");
      const cls = classes[i]?.(item);
      if (cls) td.className = cls;
      tr.append(td);
    });
    return tr;
  }));
  show(`${id}-empty`, items.length === 0);
  body.closest(".table-scroll").hidden = items.length === 0;
}

async function request(path, {method = "GET", signal, timeout = 8000} = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, {once: true});
  const timer = setTimeout(() => controller.abort("timeout"), timeout);
  try {
    const response = await fetch(path, {method, cache: "no-store", credentials: "same-origin", referrerPolicy: "no-referrer", headers: {Accept: "application/json"}, signal: controller.signal});
    const result = await response.json().catch(() => null);
    if (!response.ok) throw Error((typeof result?.detail === "string" && result.detail) || result?.error || `Request failed (${response.status})`);
    if (result == null) throw Error("The server returned an unreadable response");
    return result;
  } catch (error) {
    if (controller.signal.reason === "timeout") throw Error("Request timed out; retrying on the next refresh");
    if (controller.signal.aborted) throw new DOMException("Request cancelled", "AbortError");
    throw error instanceof TypeError ? Error("Connection unavailable; checking again shortly") : error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

function renderStatus(s) {
  const login = s.oauth === "AUTHENTICATED";
  text("login-chip", login ? `Logged in · ${s.login_minutes_left ?? "?"} min left` : `Login: ${display(s.oauth)}`);
  $("login-chip").className = login ? "chip" : "chip bad";
  text("day-chip", s.session ? `US session ${ny(s.session.open)}–${ny(s.session.close)} NY${s.session.regular ? "" : " · early close, skipped"}` : "No US session today");
  text("delay-chip", s.qqq_delay_minutes == null ? "QQQ delay not measured yet" : s.qqq_delay_minutes ? `QQQ prices ${s.qqq_delay_minutes} min late` : "QQQ prices live");
  const problems = Object.entries(s.problems || {});
  show("problem-chip", problems.length > 0);
  text("problem-chip", problems.map(([k, v]) => `${display(k)}: ${display(v)}`).join(" · "));
  $("problem-chip").className = "chip bad";
  $("status").dataset.tone = !login || s.worker_health !== "RUNNING" ? "bad" : problems.length ? "warn" : "ok";
}

function signalText(s) {
  if (!s) return "Not yet";
  const d = s.detail || {};
  if (s.method === "CONNORS") {
    if (s.reason) return display(s.reason);
    const sum = d.rsi2 != null && d.rsi2_prev != null ? (d.rsi2 + d.rsi2_prev).toFixed(1) : "?";
    return `${s.side ? "Signal: buy the call" : "No signal"} · RSI2 sum ${sum} · price ${price(d.price_15_50)} vs 200-day ${price(d.sma200)}`;
  }
  if (!s.side) return `No trade · ${display(s.reason) || "close equals EMA"}`;
  return `${side(s.side)} · candle ${price(d.close)} vs EMA12 ${price(d.ema)}`;
}

function renderToday(d) {
  renderStatus(d.system);
  text("today-day", `${new Date(d.day + "T12:00:00Z").toLocaleDateString("en-GB", {weekday: "long", day: "numeric", month: "long", year: "numeric"})}`.toUpperCase());
  const signal = (method, market) => d.signals.find((s) => s.method === method && s.market === market);
  text("sig-NAS100", signalText(signal("FIRST_CANDLE", "NAS100")));
  text("sig-US500", signalText(signal("FIRST_CANDLE", "US500")));
  text("sig-CONNORS", signalText(signal("CONNORS", "QQQ")));
  const status = new Map(d.events.map((e) => [e.name, e]));
  rows("plan", d.plan, [
    (p) => EVENTS[p.event] || p.event, (p) => p.new_york, (p) => p.uk,
    (p) => { const e = status.get(p.event); return e ? `${display(e.status)}${e.note ? ` · ${display(e.note)}` : ""}` : p.qqq_delayed ? "Waiting · read after the QQQ delay" : "Waiting"; },
  ], {3: (p) => ({FAILED: "neg", MISSED: "neg", DONE: "pos"})[status.get(p.event)?.status] || ""});
  rows("cfd", d.cfd, [
    (t) => MARKETS[t.market] || t.market, (t) => side(t.side), (t) => t.amount,
    (t) => `${price(t.entry_price)} · ${ny(t.entry_at)} NY`, (t) => price(t.stop),
    (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${ny(t.exit_at)} NY` : "Open",
    (t) => t.exit_at ? money(t.pnl_usd) : t.open_pnl_usd != null ? `${money(t.open_pnl_usd)} open` : "Open",
  ], {6: (t) => tone(t.exit_at ? t.pnl_usd : t.open_pnl_usd)});
  rows("options", d.options, [
    (o) => METHODS[o.method] || o.method,
    (o) => `${o.put_call} ${o.strike} · expires ${o.expiry}`,
    (o) => o.entry_delta == null ? "—" : Number(o.entry_delta).toFixed(2),
    (o) => `${price(o.entry_ask)} · ${ny(o.entry_for)} NY`,
    (o) => o.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : o.exit_for ? `${price(o.exit_bid)} · ${ny(o.exit_for)} NY` : `Open · sells ${o.method === "CONNORS_QQQ" ? o.exit_day + " 15:55" : "13:00"} NY`,
    (o) => o.pnl_usd == null ? (o.exit_reason ? display(o.exit_reason) : "Open") : money(o.pnl_usd),
  ], {5: (o) => tone(o.pnl_usd)});
}

function renderRecord(d) {
  renderStatus(d.system);
  rows("summary", d.summary, [
    (m) => METHODS[m.method] || m.method, (m) => m.trades, (m) => m.closed, (m) => m.wins ?? 0,
    (m) => money(m.total_usd), (m) => money(m.mean_usd), (m) => m.first_day,
  ], {4: (m) => tone(m.total_usd)});
  rows("trades", d.trades, [
    (t) => t.day, (t) => METHODS[t.method] || t.method, (t) => side(t.side),
    (t) => t.opened ? `${ny(t.opened)} NY` : "—", (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit" : t.closed ? `${ny(t.closed)} NY` : "Open",
    (t) => display(t.exit_reason) || "—", (t) => money(t.pnl_usd),
  ], {6: (t) => tone(t.pnl_usd)});
  $("prev").disabled = offset === 0;
  $("next").disabled = !d.has_more;
}

function renderSystem(s) {
  renderStatus(s);
  text("auth-state", `Saxo login: ${s.oauth}${s.oauth_failure ? ` · ${display(s.oauth_failure)}` : ""}`);
  const facts = [
    ["Environment", s.data_environment], ["Build", s.build], ["Worker", display(s.worker_health)],
    ["Saxo session level", s.trade_level ? `${s.trade_level} (the app never changes it)` : "Not read yet"],
    ["Login renewal", s.login_minutes_left == null ? "No login" : `${s.login_minutes_left} min before the next renewal is due`],
    ["QQQ delay", s.qqq_delay_minutes == null ? "Not measured yet" : `${s.qqq_delay_minutes} min`],
    ["Calendar", s.calendar_loaded ? `Loaded for ${s.calendar_loaded}` : "Not loaded"],
    ["Saxo requests since start", s.saxo_calls],
  ];
  $("facts").replaceChildren(...facts.map(([k, v]) => {
    const div = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = String(v ?? "—"); div.append(dt, dd); return div;
  }));
  if ($("system-detail").open) text("system-json", JSON.stringify(s, null, 2));
}

async function refresh(force = false) {
  if (document.hidden || (pending && !force)) return;
  if (force) pending?.abort();
  const controller = new AbortController(); pending = controller;
  try {
    const path = page === "record" ? `/api/record?offset=${offset}` : page === "system" ? "/api/system" : "/api/today";
    const d = await request(path, {signal: controller.signal});
    if (controller.signal.aborted) return;
    if (page === "record") renderRecord(d);
    else if (page === "system") renderSystem(d);
    else renderToday(d);
    lastRefresh = new Date().toISOString();
    text("notice", "");
    text("last-refresh", `Refreshed ${uk(lastRefresh)} UK`);
  } catch (e) {
    if (e.name !== "AbortError") text("notice", `${e.message} · displayed values may be stale`);
  } finally {
    if (pending === controller) pending = null;
  }
}

show(page, true);
for (const a of document.querySelectorAll("nav a")) if ((a.getAttribute("href").replace(/^\//, "") || "today") === page) a.setAttribute("aria-current", "page");
$("saxo-connect").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button"); button.disabled = true;
  text("oauth-notice", "");
  try { const result = await request("/oauth/saxo/start", {method: "POST", timeout: 15000}); window.location.assign(result.authorization_url); }
  catch (e) { text("oauth-notice", e.message); button.disabled = false; }
};
$("prev").onclick = () => { offset = Math.max(0, offset - 100); refresh(true); };
$("next").onclick = () => { offset += 100; refresh(true); };
$("system-detail").addEventListener("toggle", () => refresh(true));
document.addEventListener("visibilitychange", () => { if (document.hidden) pending?.abort(); else refresh(true); });
setInterval(() => {
  const now = new Date().toISOString();
  text("clock", `${ny(now)} NY · ${uk(now)} UK`);
  if (lastRefresh && Date.now() - Date.parse(lastRefresh) > (REFRESH_SECONDS * 3 + 5) * 1000) text("last-refresh", `Stale · last refresh ${uk(lastRefresh)} UK`);
}, 1000);
setInterval(() => refresh(), REFRESH_SECONDS * 1000);
refresh(true);
