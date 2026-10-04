/* Real dashboard browser checks with explicitly labelled offline API fixtures. No broker. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const http = require("node:http");
const { chromium } = require("playwright");

const at = "2026-09-28T14:20:00Z";
// Fixture roles go by position (index 2, once gas, is now E-mini S&P); the cards render in cardOrder.
const markets = ["CL", "GC", "ES", "NQ"];
const cardOrder = ["CL", "ES", "GC", "NQ"];
const staticRoot = path.resolve(
  "packages/stocker_dashboard/src/stocker_dashboard/static",
);
// Screenshots go to a temporary directory unless explicitly requested, e.g.
// `node tests/dashboard_futures.cjs --screenshots docs/cleanup-screenshots`.
const screenshotFlag = process.argv.indexOf("--screenshots");
const output =
  screenshotFlag > 0
    ? path.resolve(process.argv[screenshotFlag + 1])
    : fs.mkdtempSync(path.join(os.tmpdir(), "slrno-dashboard-"));
const state = {
  system: {
    account: "OFFLINE FIXTURE",
    data_environment: "SAXO_SIM", execution_mode: "SAXO_SIM",
    oauth: "OFFLINE FIXTURE",
    session: {TradeLevel: "FullTradingAndChat"},
    connected: true,
    reconciled: true,
    armed: false,
    paused: false,
    reserved_open_trades: 2,
    allocation_pennies: 10000,
    limits: {per_trade_gbp:50,allocation_gbp:200,slots:4,quote_max_age_seconds:5},
    entry_block_reason:"PAPER_DISARMED",
    server_time: Date.parse(at)/1000,
    next_clock: "2026-09-28T15:00:00Z",
    alerts: {enabled:false, problem:"", active:[], sent:0, last_error:""},
    setup: [
      {key:"oauth",label:"Saxo login",done:true,detail:"",optional:false},
      {key:"contracts",label:"Futures contracts verified 4/4",done:true,detail:"",optional:false},
      {key:"approvals",label:"Option approvals 1/4",done:false,detail:"Approve product, delta tolerance, fees and expiry times (GC optional)",optional:false},
      {key:"mode",label:"Paper execution mode",done:true,detail:"",optional:false},
      {key:"armed",label:"Armed after preflight",done:false,detail:"Run preflight, then arm explicitly",optional:false},
      {key:"alerts",label:"Alerts",done:false,detail:"Optional: configure an alert URL file",optional:true},
    ],
    market_data: {
      owned_lines: 24,
      app_budget: 32,
      errors: [],
    },
    l2_recording: {
      problem: "MINIMUM_DISK_RESERVE_REACHED",
      disk_bytes: 12582912,
      disk_limit: 268435456,
      memory_bytes: 2097152,
      memory_limit: 33554432,
      recording_gaps: 1,
      writer_queue: 0,
    },
  },
  account: {environment:"SIM",label:"SIM · simulated funds",account:"••••1234",currency:"GBP",status:"Current",last_success_at:Date.parse(at)/1000,valid_until:Date.parse(at)/1000+30,total_value:12345.67,cash_balance:10000,cash_available_for_trading:null,connection_note:"Real-money balances are not connected",details:{CalculationReliability:"Ok"}},
  pnl: {
    broker_reported: {count:1, closed_profit_loss_base:12.5, costs_base:1.5, currency:"EUR"},
    realised_net_gbp: null,
    provisional_closed: 1,
    opportunities: 12,
    eligible_trades: 3,
    skip_reasons: { MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET: 5, SKIP_CAPACITY_FULL: 1 },
    fills: 3,
    win_rate: null,
    wins: 0,
    closed_with_complete_costs: 0,
  },
  markets: markets.map((market, i) => ({
    market,
    contract: `${market} · fixture contract`,
    market_status: "OPEN",
    data_status: "CURRENT",
    updated_at: at,
    last_receipt:Date.parse(at)/1000,
    l1: { status: "CURRENT", last_receipt: new Date(Date.parse(at) - 60000).toISOString(), standing_receipt: at, quote: {Bid: 70 + i, Ask: 70.01 + i}, sizes: {bid: 2, ask: 3}, spread: .01, delay_minutes: 0 },
    recorder: {state: i === 4 ? "STORAGE_LIMIT" : "BUFFERING", prehistory_seconds: [900,42,15,0,0][i], reason: i === 4 ? "STORAGE_LIMIT_REACHED" : ""},
    option_context: {candidate_uic: 1001+i, candidate_changes: [], latest_event: {id: "fixture-event", context: {identity: {uic: 1001+i}, pre_trigger_seconds: 42}}, contracts: [{
      identity: {uic: 1001+i, underlying_uic: 100+i, underlying_symbol: `${market}Z6`, symbol: `${market} fixture option`, right: "Put", strike: 70, expiry: "2026-09-28", last_trade_at: "2026-09-28T20:00:00Z"},
      quote: {Bid: .01, Ask: .02, PriceTypeBid: "Tradable", PriceTypeAsk: "Tradable", DelayedByMinutes: 0}, quote_status: "OBSERVED",
      sizes: {Bid: 2, Ask: 3}, size_status: {Bid: "STALE_OR_MISSING", Ask: "OBSERVED"}, coverage_seconds: 55, subscription_started_at: Date.parse(at)/1000-55,
      analytics: {"Greeks.Delta": {value: -.1, status: "OBSERVED_UNVERIFIED", age_seconds: 1}, "Greeks.MidVol": {value: .31, status: "OBSERVED_UNVERIFIED", age_seconds: 1}, "InstrumentPriceDetails.OpenInterest": {value: 200, status: "AS_OF_EFFECTIVE_TIME_UNKNOWN", age_seconds: 120}},
      costs: {minimum_purchase_cost_gbp: 20, entry_costs_gbp: .1, estimated_exit_costs_gbp: .1, total_gbp: 20.1, remaining_budget_gbp: -10.1, budget_result: "MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET"},
    }]},
    l2: {
      status: ["L2_AVAILABLE", "L2_AVAILABLE", "L1_ONLY", "L2_UNAVAILABLE", "L2_UNAVAILABLE"][i],
      target_pre_seconds: 900,
      pre_seconds: [900, 42, 15, 0, 0][i],
      fresh: i === 0 || i === 1,
      last_receipt: at,
      valid_until: Date.parse(at)/1000 + 5,
      asks: Array.from({ length: 5 }, (_, n) => ({
        price: 2651 + n,
        size: 10 + n,
      })),
      bids: Array.from({ length: 5 }, (_, n) => ({
        price: 2650 - n,
        size: 12 + n,
      })),
      reason: i === 4 ? "SAXO_DEPTH_PERMISSION_UNVERIFIED" : "",
    },
    entry_enabled: false,
    strategy_state: [
      "BLOCKED",
      "BLOCKED",
      "OPEN",
      "MONITORING",
      "RESERVED",
    ][i],
    direction: i === 0 ? "BUY CALL" : "BUY PUT",
    block_reason: [
      "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED",
      "GC_LISTED_EXECUTION_RULE_UNAPPROVED",
      "ENTRIES_PAUSED · existing position managed",
      "EXECUTION_UNARMED",
      "ORDER_STATUS_UNCERTAIN",
    ][i],
    conditions: { rv15: 0.0008 + i * 0.0002 },
    next_time: "2026-09-28T15:00:00Z",
    next_market_time: "2026-09-28T21:00:00Z",
    exchange_trade_date: null,
    diagnostic: market === "GC" ? "L2 observation only" : "",
    details: {
      fixture: true,
      mapping: null,
      entry_timezone: "America/New_York",
      entry_clocks: "Hourly through the CME session: 18:00-16:00 NY",
      volume: "Observation only",
      exit_anchor: "Original opportunity + 60 minutes",
    },
    chart: Array.from({ length: 90 }, (_, n) => ({
      at: new Date(Date.parse(at) - (90 - n) * 60000).toISOString(),
      close:
        [72, 2650, 3, 23000, 32][i] *
        (1 + n * 0.00001 + Math.sin(n * 0.25 + i) * 0.0008),
    })),
    signals: [
      {
        id: `${market}-signal`,
        signal_at: "2026-09-28T14:00:00Z",
        decision: "SIGNAL_OBSERVED",
        reason: "",
      },
    ],
    trades:
      market === "ES"
        ? [
            {
              id: "es-open",
              state: "OPEN",
              quantity: 1,
              entry_at: "2026-09-28T14:00:04Z",
              exit_at: "2026-09-28T15:00:00Z",
              valuation: {
                value_gbp: -0.8,
                fresh: true,
                quote_at: at,
                method: "CONSERVATIVE_BID_ESTIMATE",
              },
            },
          ]
        : [],
  })),
};
const gateLabels = [["saxo","Saxo"],["contract","Contract"],["quote","Quote"],["history","History"],["approval","Option approval"],["strike","Strike"],["cost","Cost ≤ £1,000"],["execution","Execution"]];
for (const m of state.markets) {
  m.gates = gateLabels.map(([key,label],i)=>({key,label,ok:i<4 || (m.market==="NQ" && i<6),detail:i<4?"":i===4?"LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED":i===6?"MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET":"PAPER_DISARMED"}));
  m.events = m.market==="CL" ? [{name:"EIA Weekly Petroleum Status",at:"2026-09-28T14:30:00Z",relation:"DURING_HOLDING_WINDOW"}] : [];
  m.chart_context = {clocks:[13,14,15].map(h=>({at:`2026-09-28T${h}:00:00Z`,session:"AutomatedTrading"})), rv_window:["2026-09-28T14:05:00Z","2026-09-28T14:20:00Z"]};
  m.candidate_deltas = {[String(1001+markets.indexOf(m.market))]: 0.1032};
  m.target_delta = 0.1;
  m.sessions_today = [{start:"2026-09-27T22:00:00Z",end:"2026-09-28T21:00:00Z",state:"AutomatedTrading"}];
  m.events_today = m.market==="GC" ? [{at:"2026-09-28T12:30:00Z",name:"US CPI (fixture)"}] : [];
  m.smile = {expiry:"2026-09-28T00:00:00Z", mid_strike_price:70.2, scaling:"PROVIDER_NATIVE_UNVERIFIED", executable:false,
    strikes:[66,67,68,69,70,71,72].map((strike,k)=>({strike, mid_volatility_pct:.3, call:null, put:{uic:900+k, mid_volatility:.28+Math.abs(69-strike)*.012, open_interest:[40,120,300,800,500,90,20][k]}}))};
  m.model_sigma = 0.27;
  m.quote = {Bid: 2638, Ask: 2638.5};
  m.forecast = {status:"OBSERVED", level:0.82, next_hour_move:0.0031, option:{uic:1001+markets.indexOf(m.market), implied_over_forecast:1.29, implied_move_to_expiry:0.0062, forecast_move_to_expiry:0.0048}};
  m.price_context = {open:2641,high:2660,low:2630,last_close:2638,net_change:12,percent_change:0.45,open_interest:512000,market_state:"Open",daily_range:31.5};
  m.identity = {environment:"SAXO_SIM",uic:100+markets.indexOf(m.market)};
  m.book_flow = {
    version:"SAXO_SAMPLED_BOOK_FLOW_V1", at:Date.parse(at)/1000, valid_until:Date.parse(at)/1000+30,
    status:"CURRENT", quality_flags:[], delay_minutes:0, spread_ticks:1,
    available_levels:{bid:5,ask:5}, weighted_midpoint:70.0075, weighted_displacement_ticks:.25,
    latest_trade:{price:70,size:2}, volume:{value:100,change:null,status:"SEMANTICS_UNVERIFIED"},
    feed:{granted_refresh_ms:1500,observed_receipt_ms:{samples:20,mean:1800,minimum:1000,maximum:3000},last_receipt:Date.parse(at)/1000,last_field_change:Date.parse(at)/1000-50,last_contact:Date.parse(at)/1000},
    depth:Object.fromEntries([1,3,5,10].map(n=>[n,n<=5?{bid:3*n,ask:n,imbalance:.5,order_imbalance:null,label:"BID_HEAVY"}:{}])),
    lookbacks:Object.fromEntries([5,30,60].map(s=>[s,{5:{status:"AVAILABLE",bid_change:2,ask_change:-1,bid_heavy_fraction:.5,ask_heavy_fraction:.25,balanced_fraction:.25}}]))
  };
}
const bookSeries = {status:"AVAILABLE",tick_size:0.1,bucket_seconds:5,semantics:"Sampled Saxo depth, last observation per bucket; not an execution tape",
  series:Array.from({length:120},(_,i)=>({at:Date.parse(at)/1000-600+i*5,status:"CURRENT",spread_ticks:1+(i%7===0),imbalance1:Math.sin(i/9),imbalance5:Math.sin(i/15)*.6,
    bid:Array.from({length:10},(_,n)=>[26500-n-Math.round(Math.sin(i/20)*3),5+n*2+(n===4?40:0)]),
    ask:Array.from({length:10},(_,n)=>[26501+n-Math.round(Math.sin(i/20)*3),4+n*2+(n===6?35:0)])}))};
state.today = {day:"2026-09-28", closed:1, wins:0, net_gbp:-27.77, trades:[
  {id:"t-nq", market:"NQ", signal_at:"2026-09-28T13:00:00Z", exit_at:"2026-09-28T14:00:00Z", state:"CLOSED", option:"Put 30650", bought:46.25, sold:45.25, paid_gbp:705.39, net_gbp:-27.77},
  {id:"es-open", market:"ES", signal_at:"2026-09-28T14:00:00Z", exit_at:"2026-09-28T15:00:00Z", state:"OPEN", option:"Put 3", bought:0.05, sold:null, paid_gbp:22.4, net_gbp:null}]};
const recentTrades = state.today.trades.map((t) => ({...t}));
const timeline = [
  {at:"2026-09-28T14:00:00Z",kind:"CLOCK",ok:true,title:"Frozen clock · ES",detail:"rule CLOCK60_23H_ES_20261004 · exit anchor 2026-09-28T15:00:00Z"},
  {at:"2026-09-28T14:00:00Z",kind:"CHECKS",ok:true,title:"Data and eligibility checks",detail:"rv15 0.0012 · futures_price 3.1"},
  {at:"2026-09-28T14:00:00Z",kind:"OPTION",ok:true,title:"Option selection",detail:"Put 3 · UIC 1003"},
  {at:"2026-09-28T14:00:02Z",kind:"ADMISSION",ok:true,title:"Admission",detail:"reserved · cost £22.4"},
  {at:null,kind:"OUTCOME",ok:null,title:"Current state",detail:"BROKER_PAPER_FILL · 1 fill(s)"},
];
const rows = Array.from({ length: 80 }, (_, i) => ({
  id: `fixture-${i}`,
  market: markets[i % markets.length],
  signal_at: new Date(Date.parse(at) - i * 60000).toISOString(),
  rule_version: "CLOCK60_NG13_20260927",
  decision: i % 3 ? "SKIPPED" : "BROKER_PAPER_FILL",
  reason: i % 3 ? "MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET" : "",
  state: null,
  trade: i % 3 ? null : {state:"CLOSED", option:"Put 3", bought:0.05, sold:0.08, paid_gbp:22.4, net_gbp:i ? 6.5 : -4.25},
}));
let primaryRequests = 0, lastHistoryQuery = "";
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let data;
  if (url.pathname === "/api/overview") data = state;
  else if (url.pathname.endsWith("/book")) data = bookSeries;
  else if (url.pathname.startsWith("/api/market/")) data={system:state.system,markets:state.markets.filter(m=>m.market===url.pathname.split("/").at(-1))};
  else if (url.pathname === "/api/execution") data={system:state.system,trades:[],orders:[],fills:[{exec_id:"f1",at:"2026-09-28T13:00:01Z",con_id:1003,market:"ES",option_right:"Put",option_strike:3,role:"ENTRY",side:"BOT",quantity:1,price:0.05,commission:2,commission_currency:"USD"}],positions:[],recent_trades:recentTrades};
  else if (url.pathname === "/api/recordings") data = {active: [], completed: []};
  else if (url.pathname === "/api/history" && (lastHistoryQuery = url.search) !== null)
    data = {
      system:state.system,
      rows: [...rows].sort((a,b)=>url.searchParams.get("sort")==="asc" ? a.signal_at.localeCompare(b.signal_at) : b.signal_at.localeCompare(a.signal_at)).filter(
        (r) =>
          !url.searchParams.get("market") ||
          r.market === url.searchParams.get("market"),
      ),
      has_more: false,
    };
  else if (url.pathname === "/api/system")
    data = {
      fixture: true,
      ...state.system,
      // Production shape: per-market problem and capability view (saxo_data.capability_view).
      markets:state.markets.map(m=>({market:m.market,problem:m.block_reason,reference_sessions:5,capabilities:{l2:m.l2},candidates:null})),
      live_available: false,
      mappings: {},
    };
  else if (url.pathname === "/api/detail")
    data = {
      fixture: true,
      timeline,
      orders: [],
      fills: [],
      note: "Offline browser fixture",
    };
  else if (url.pathname.startsWith("/api/entries/")) {
    state.system.paused = url.pathname.endsWith("pause");
    data = state.system;
  } else if (url.pathname === "/api/session/primary" && req.method === "POST") {
    primaryRequests += 1;
    data = state.system;
  }
  if (data) {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(data));
    return;
  }
  const file = url.pathname.startsWith("/static/")
    ? path.join(staticRoot, path.basename(url.pathname))
    : path.join(staticRoot, "index.html");
  res.setHeader(
    "Content-Type",
    file.endsWith(".css")
      ? "text/css"
      : file.endsWith(".js")
        ? "text/javascript"
        : "text/html",
  );
  res.end(fs.readFileSync(file));
});
async function label(page) {
  await page.evaluate(() => {
    const el = document.createElement("p");
    el.id = "fixture-label";
    el.textContent =
      "OFFLINE TEST FIXTURE · illustrative prices and states · no broker orders";
    el.style.cssText =
      "margin:0;padding:9px 3.2vw;background:#efc184;color:#10191e;font:12px monospace";
    document.body.prepend(el);
  });
}
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ headless: true });
  try {
    fs.mkdirSync(output, { recursive: true });
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1080 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.clock.install({ time: new Date(at) });
    const base = `http://127.0.0.1:${server.address().port}`;
    await page.goto(base);
    await page.waitForFunction(
      () =>
        document.querySelector("#account").textContent.includes("SIM · simulated funds"),
    );
    assert.deepEqual(
      await page
        .locator(".market")
        .evaluateAll((nodes) => nodes.map((n) => n.id)),
      cardOrder.map((m) => `card-${m}`),
    );
    assert.match(await page.locator("#state-GC").textContent(), /BLOCKED/);
    assert.match(await page.locator("#allocation").textContent(), /100.00.*200.00/);
    assert.equal(await page.locator("#account-available").textContent(), "Unavailable");
    assert.match(await page.locator("#account-note").textContent(), /Real-money balances are not connected/);
    assert.match(await page.locator("#pnl-note").textContent(), /provisional.*Saxo reports 12\.5 EUR closed P&L \(1 position\)/);
    assert.equal(await page.locator("#entries").textContent(), "Unarmed");
    assert.equal(await page.locator("#overview-slots > div").count(), 4);
    assert.match(await page.locator("#live-badge").textContent(), /LIVE ORDERS DISABLED/);
    assert.match(await page.locator("#exec-chip").textContent(), /Execution SAXO_SIM/);
    // Primary session without a verified real-time feed must not read as fully on.
    assert.equal(await page.locator("#realtime-chip").textContent(), "Real-time ON · Saxo still delayed");
    await page.clock.fastForward(1100);
    assert.match(await page.locator("#countdown").textContent(), /Next clock 11:00 NY \(16:00 London\) · in 39:5\d/);
    assert.match(await page.locator("#setup-progress").textContent(), /3 of 5 required steps done/);
    assert.equal(await page.locator("#gates-NQ li.ok").count(), 6);
    assert.match(await page.locator("#gates-GC li.fail").first().getAttribute("title"), /Option product and selection approval required/);
    assert.match(await page.locator("#block-GC").textContent(), /Option approval: Option product and selection approval required/);
    assert.match(await page.locator("#events-CL").textContent(), /EIA Weekly Petroleum Status 10:30 NY/);
    assert(await page.locator("#events-GC").isHidden());
    assert(await page.locator("#position-ES").isVisible() && await page.locator("#position-CL").isHidden());
    assert.equal(await page.locator("#price-GC").textContent(), "2638.25");
    assert.match(await page.locator("#change-GC").textContent(), /^\+12 \(\+0\.45%\)$/);
    assert.equal(await page.locator("#today-pnl").textContent(), "−£27.77");
    assert.match(await page.locator("#today-note").textContent(), /2 trades · 0 of 1 closed in profit · 1 open/);
    assert.deepEqual(await page.locator("#today-list li").first().locator("span").allTextContents(), ["15:00", "ES Put 3", "0.05 → …", "open · −£0.80"]); // the card's live bid estimate
    assert.equal(await page.locator("#today-list li").nth(1).locator("span").last().getAttribute("class"), "neg");
    assert.match(await page.locator("#today-NQ").textContent(), /Today: 1 trade · −£27\.77/);
    assert.match(await page.locator("#checks-NQ").textContent(), /6 of \d+ checks pass/);
    assert(await page.locator("#status-toggle").isHidden()); // desktop shows the full status row
    await label(page);
    await page.screenshot({
      path: path.join(output, "overview-desktop-fixture.png"),
      fullPage: true,
    });
    state.markets[1].l2.fresh = true;
    state.markets[1].l2.valid_until = Date.parse(at)/1000+30;
    await page.goto(`${base}/markets`);
    await page.locator("#tab-GC").click();
    await page.evaluate(() => refresh(true));
    assert.equal(await page.locator("#card-GC").isVisible(), true);
    assert.equal(await page.locator("#card-CL").isVisible(), false);
    assert.match(await page.locator("#book-flow-GC").textContent(), /Sampled order-book observations; not a complete execution tape/);
    assert.match(await page.locator("#flow-depth-GC-10").textContent(), /UNAVAILABLE/);
    assert.deepEqual(await page.locator("#flow-depth-GC-5 td").allTextContents(), ["15", "5", "0.5", "BID_HEAVY"]);
    assert.match(await page.locator("#book-flow-GC").textContent(), /Saxo sends no order counts/);
    assert.match(await page.locator("#flow-volume-GC").textContent(), /change UNAVAILABLE/);
    assert.match(await page.locator("#flow-feed-GC").textContent(), /granted 1500 ms.*mean 1800 ms/);
    assert.match(await page.locator("#option-identity-GC").textContent(), /UIC 1002.*underlying GCZ6 \/ 101/);
    assert.match(await page.locator("#option-quote-GC").textContent(), /bid size STALE_OR_MISSING/);
    assert.match(await page.locator("#option-volume-GC").textContent(), /AS_OF_EFFECTIVE_TIME_UNKNOWN.*effective unknown/);
    assert.match(await page.locator("#option-coverage-GC").textContent(), /actual pre-trigger 42 \/ 900s/);
    assert.match(await page.locator("#option-cost-GC").textContent(), /MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET/);
    assert.match(await page.locator("#ticket-contract-GC").textContent(), /Candidate · GC fixture option · Put 70/);
    assert.match(await page.locator("#ticket-spread-GC").textContent(), /66\.7% of mid/);
    assert.match(await page.locator("#ticket-delta-GC").textContent(), /model \|Δ\| 0\.1032 vs target 0\.1/);
    assert.match(await page.locator("#ticket-iv-GC").textContent(), /Implied 31\.0% vs frozen model σ 27\.0% · \+4\.0 pts/);
    assert.match(await page.locator("#ticket-forecast-GC").textContent(), /0\.82× normal .* ±0\.31% · priced for 1\.29× the forecast move to expiry \(0\.62% vs 0\.48%\)/);
    assert.match(await page.locator("#ticket-cost-GC").textContent(), /£20\.10 all-in of £50\.00/);
    assert.match(await page.locator("#sessions-GC").textContent(), /18:00–17:00 AutomatedTrading \(NY\)/);
    assert.match(await page.locator("#events-today-GC").textContent(), /08:30 US CPI/);
    assert.match(await page.locator("#price-context-GC").textContent(), /high 2660.*OI 512000/);
    assert.equal(await page.locator("#smile-GC rect.oi").count(), 7);
    assert.equal(await page.locator("#smile-GC circle.selected").count(), 1);
    assert.match(await page.locator("#smile-note-GC").textContent(), /put IV by strike.*frozen model σ 0\.27/);
    assert.equal(await page.locator("#clocks-GC line").count(), 2); // 13:00 and 14:00 UTC inside the 90-minute chart
    assert(Number(await page.locator("#rv-GC").getAttribute("width")) > 0);
    assert.match(await page.locator("#imbalance-GC").textContent(), /\+0\.50 · bid heavy/);
    assert(Number(await page.locator("#depth-GC-0 .bar.bid i").evaluate(n=>parseFloat(n.style.width))) > 0);
    // Sellers on the left, buyers on the right, the same way round as the imbalance bar (2026-10-01).
    assert.match(await page.locator("#ladder-GC .ladder-head").textContent(), /^Ask size\s*Ask\s*Bid\s*Bid size$/);
    assert.equal(await page.locator("#depth-GC-0 span").first().textContent(), "10");
    assert.equal(await page.locator("#depth-GC-0 .price.ask").textContent(), "2651");
    assert.equal(await page.locator("#depth-GC-0 .price.bid").textContent(), "2650");
    assert.equal(await page.locator("#depth-GC-0 span").last().textContent(), "12");
    await page.locator("#book-history-GC summary").click();
    await page.waitForFunction(() => document.querySelector("#heatmap-GC").dataset.columns === "120");
    assert.match(await page.locator("#heatmap-note-GC").textContent(), /120 × 5s buckets/);
    assert(await page.locator("#spark-imbalance-GC polyline").count() >= 1);
    await label(page);
    await page.screenshot({path:path.join(output,"book-history-desktop-fixture.png"),fullPage:true});
    await page.locator("#book-history-GC summary").click();
    await page.locator("#flow-detail-GC summary").click();
    await page.locator("#flow-detail-GC summary").focus();
    await page.evaluate(() => { window.flowRow=document.querySelector("#flow-depth-GC-5"); window.flowFocus=document.activeElement; window.scrollTo(0,350); });
    const flowY = await page.evaluate(() => scrollY);
    await page.evaluate(() => refresh());
    assert(await page.evaluate(() => flowRow===document.querySelector("#flow-depth-GC-5") && flowFocus===document.activeElement));
    assert(await page.locator("#flow-detail-GC").evaluate(n=>n.open));
    assert.equal(await page.evaluate(() => scrollY), flowY);
    await page.locator("#flow-detail-GC summary").click();
    await label(page);
    await page.screenshot({path:path.join(output,"book-flow-desktop-fixture.png"),fullPage:true});
    await page.clock.fastForward(6000);
    assert(await page.locator("#depth-GC-0").isVisible()); // unchanged, healthy book
    // An unchanged quote on a live feed stays current (the server's standing time, a minute after
    // its last change); the page degrades it only once that is older than the limit plus a refresh.
    assert.equal(await page.locator("#l1-GC").textContent(), "L1 CURRENT");
    state.markets[1].l1.standing_receipt = new Date(Date.parse(at) - 60000).toISOString();
    await page.evaluate(() => refresh(true));
    await page.clock.fastForward(1100);
    assert.equal(await page.locator("#l1-GC").textContent(), "L1 STALE OR MISSING");
    state.markets[1].l1.standing_receipt = at;
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path:path.join(output,"book-flow-mobile-fixture.png"),fullPage:true});
    await page.setViewportSize({width:1440,height:1080});
    await page.goto(`${base}/markets`);
    assert.equal(await page.locator("#tab-GC").getAttribute("aria-selected"), "true");
    await page.goto(`${base}/opportunities`);
    await page.waitForFunction(
      () => document.querySelectorAll("#history tr").length === 80,
    );
    await label(page);
    await page.locator("#sort-filter").selectOption("asc");
    await page.locator("#version-filter").fill("CLOCK60_NG13_20260927");
    await page.locator("#version-filter").focus();
    await page.evaluate(() => {
      window.savedRow = document.querySelector("#history tr");
      window.savedFocus = document.activeElement;
      document.querySelector("#trades .table-scroll").scrollTop = 240;
    });
    await page.evaluate(() => refresh());
    assert(
      await page.evaluate(
        () =>
          savedRow === document.querySelector("#history tr") &&
          savedFocus === document.activeElement,
      ),
    );
    assert.equal(await page.locator("#sort-filter").inputValue(), "asc");
    assert.equal(
      await page.locator("#version-filter").inputValue(),
      "CLOCK60_NG13_20260927",
    );
    assert.equal(
      await page.locator("#trades .table-scroll").evaluate((n) => n.scrollTop),
      240,
    );
    const traded = page.locator('#history tr[data-key="fixture-0"] td');
    assert.equal(await traded.nth(5).textContent(), "−£4.25");
    assert.equal(await traded.nth(5).getAttribute("class"), "neg");
    assert.equal(await traded.nth(4).textContent(), "0.05 → 0.08");
    await page.locator("#trades-filter").check();
    for (let i = 0; i < 50 && !lastHistoryQuery.includes("trades=true"); i++) await new Promise((r) => setTimeout(r, 50));
    assert.match(lastHistoryQuery, /trades=true/);
    await page.locator("#trades-filter").uncheck();
    await page.locator("#history tr button").first().click();
    await page.waitForFunction(() => document.querySelectorAll("#trade-timeline li").length === 5);
    assert.equal(await page.locator("#trade-timeline li.ok").count(), 4);
    assert.match(await page.locator("#trade-timeline li").nth(3).textContent(), /Admission.*reserved · cost £22\.4/);
    await page.screenshot({
      path: path.join(output, "history-desktop-fixture.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    // Phones read each opportunity as a card: no sideways scrolling, every value labelled.
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.equal(await page.locator("#history tr").first().locator("td").nth(1).getAttribute("data-label"), "Market");
    await page.evaluate(() => window.scrollTo(0, 600));
    await page.evaluate(() => refresh());
    assert.equal(await page.evaluate(() => scrollY), 600);
    await page.goto(base);
    await page.waitForFunction(
      () =>
        document.querySelector("#account").textContent.includes("SIM · simulated funds"),
    );
    await label(page);
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    // Phones show one status line; a tap opens the chips and the Pause control.
    assert(await page.locator("#status-chips").isHidden());
    assert.match(await page.locator("#status-text").textContent(), /Saxo still delayed · Not armed/);
    await page.locator("#status-toggle").click();
    assert(await page.locator("#pause").isVisible());
    const boxes = await page
      .locator(".market")
      .evaluateAll((nodes) => nodes.map((n) => n.getBoundingClientRect().x));
    assert(boxes.every((x) => x === boxes[0]));
    await page.screenshot({
      path: path.join(output, "overview-mobile-fixture.png"),
      fullPage: true,
    });
    await page.goto(`${base}/system`);
    // Real-time is confirmed on the page (no window.confirm): the first tap only asks.
    await page.locator("#primary-session").click();
    assert.match(await page.locator("#primary-session").textContent(), /Tap again to confirm/);
    assert.equal(primaryRequests, 0);
    await page.locator("#primary-session").click();
    await page.waitForFunction(() => document.querySelector("#notice").textContent.includes("real-time requested"));
    assert.equal(primaryRequests, 1);
    assert.equal(await page.locator("#primary-session").textContent(), "Use real-time in SLRNO");
    await page.locator("#system-detail summary").click();
    await page.waitForFunction(() =>
      document
        .querySelector("#system-json")
        .textContent.includes('"live_available": false'),
    );
    await page.setViewportSize({ width: 1440, height: 1080 });
    await label(page);
    assert.match(await page.locator("#api-lines").textContent(), /24 \/ 32/);
    assert.match(
      await page.locator("#api-depth").textContent(),
      /2 \/ 4 markets with received depth/,
    );
    assert.match(await page.locator("#api-external").textContent(), /explicit/);
    await page.screenshot({
      path: path.join(output, "system-api-fixture.png"),
      fullPage: true,
    });
    // Execution: one row per trade with its result; fills name the option, not a UIC.
    await page.goto(`${base}/execution`);
    await page.waitForFunction(() => document.querySelectorAll("#execution-results tr").length === 2);
    assert.deepEqual(await page.locator('#execution-results tr[data-key="t-nq"] td').allTextContents(), ["28 Sept, 14:00:00", "NQ", "Put 30650", "46.25", "45.25", "£705.39", "−£27.77"]);
    assert.deepEqual((await page.locator("#execution-fills tr td").allTextContents()).slice(1, 3), ["ES Put 3", "Entry"]);
    assert(await page.locator("#execution-orders-empty").isVisible());
    assert.deepEqual(errors, []);
    console.log(
      "PASS: four fixed cards, signal/fill markers, provisional P&L, refresh identity/focus/scroll/filter retention, mobile layout and System",
    );
  } finally {
    await browser.close();
    server.close();
  }
})().catch((error) => {
  console.error(error);
  server.close();
  process.exitCode = 1;
});
