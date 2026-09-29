/* Real dashboard browser checks with explicitly labelled offline API fixtures. No broker. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { chromium } = require("playwright");

const at = "2026-09-28T14:20:00Z";
const markets = ["CL", "GC", "NG", "NQ", "SI"];
const staticRoot = path.resolve(
  "packages/stocker_dashboard/src/stocker_dashboard/static",
);
const output = path.resolve("docs/cleanup-screenshots");
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
    limits: {per_trade_gbp:50,allocation_gbp:200,slots:4},
    entry_block_reason:"PAPER_DISARMED",
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
    l1: { status: "CURRENT", last_receipt: at, quote: {Bid: 70 + i, Ask: 70.01 + i}, sizes: {bid: 2, ask: 3}, spread: .01, delay_minutes: 0 },
    recorder: {state: i === 4 ? "STORAGE_LIMIT" : "BUFFERING", prehistory_seconds: [900,42,15,0,0][i], reason: i === 4 ? "STORAGE_LIMIT_REACHED" : ""},
    option_context: {candidate_uic: 1001+i, candidate_changes: [], latest_event: {id: "fixture-event", context: {identity: {uic: 1001+i}, pre_trigger_seconds: 42}}, contracts: [{
      identity: {uic: 1001+i, underlying_uic: 100+i, underlying_symbol: `${market}Z6`, symbol: `${market} fixture option`, right: "Put", strike: 70, expiry: "2026-09-28", last_trade_at: "2026-09-28T20:00:00Z"},
      quote: {Bid: .01, Ask: .02, PriceTypeBid: "Tradable", PriceTypeAsk: "Tradable", DelayedByMinutes: 0}, quote_status: "OBSERVED",
      sizes: {Bid: 2, Ask: 3}, size_status: {Bid: "STALE_OR_MISSING", Ask: "OBSERVED"}, coverage_seconds: 55, subscription_started_at: Date.parse(at)/1000-55,
      analytics: {"Greeks.Delta": {value: -.1, status: "OBSERVED_UNVERIFIED", age_seconds: 1}, "InstrumentPriceDetails.OpenInterest": {value: 200, status: "AS_OF_EFFECTIVE_TIME_UNKNOWN", age_seconds: 120}},
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
      "MONITOR_ONLY",
      "POSITION_OPEN",
      "MONITORING",
      "ORDER_PENDING",
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
      entry_clocks: "09:00–16:00 weekdays; NG 13:00 veto",
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
      market === "NG"
        ? [
            {
              id: "ng-open",
              state: "POSITION_OPEN",
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
        : market === "SI"
          ? [
              {
                id: "si-pending",
                state: "EXPOSURE_REQUIRES_RECONCILIATION",
                quantity: 0,
                exit_at: "2026-09-28T15:00:00Z",
                valuation: { value_gbp: null, fresh: false },
              },
            ]
          : [],
  })),
};
for (const m of state.markets) {
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
const rows = Array.from({ length: 80 }, (_, i) => ({
  id: `fixture-${i}`,
  market: markets[i % 5],
  signal_at: new Date(Date.parse(at) - i * 60000).toISOString(),
  rule_version: "CLOCK60_NG13_20260927",
  decision: i % 3 ? "SKIPPED" : "BROKER_PAPER_FILL",
  reason: i % 3 ? "MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET" : "",
  state: null,
}));
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let data;
  if (url.pathname === "/api/overview") data = state;
  else if (url.pathname.startsWith("/api/market/")) data={system:state.system,markets:state.markets.filter(m=>m.market===url.pathname.split("/").at(-1))};
  else if (url.pathname === "/api/execution") data={system:state.system,trades:[],orders:[],fills:[],positions:[]};
  else if (url.pathname === "/api/recordings") data = {active: [], completed: []};
  else if (url.pathname === "/api/history")
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
      orders: [],
      fills: [],
      note: "Offline browser fixture",
    };
  else if (url.pathname.startsWith("/api/entries/")) {
    state.system.paused = url.pathname.endsWith("pause");
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
      markets.map((m) => `card-${m}`),
    );
    assert.match(await page.locator("#state-GC").textContent(), /MONITOR ONLY/);
    assert.match(await page.locator("#allocation").textContent(), /100.00.*200.00/);
    assert.equal(await page.locator("#account-available").textContent(), "Unavailable");
    assert.match(await page.locator("#account-note").textContent(), /Real-money balances are not connected/);
    assert.match(await page.locator("#pnl-note").textContent(), /provisional/);
    assert.equal(await page.locator("#entries").textContent(), "Unarmed");
    assert.equal(await page.locator("#overview-slots > div").count(), 4);
    assert.match(await page.locator("#mode-banner").textContent(), /LIVE ORDERS DISABLED/);
    await label(page);
    await page.screenshot({
      path: path.join(output, "overview-desktop-fixture.png"),
      fullPage: true,
    });
    state.markets[1].l2.fresh = true;
    state.markets[1].l2.valid_until = Date.parse(at)/1000+30;
    await page.goto(`${base}/markets`);
    await page.locator("#selected-market").selectOption("GC");
    await page.evaluate(() => refresh(true));
    assert.equal(await page.locator("#card-GC").isVisible(), true);
    assert.equal(await page.locator("#card-CL").isVisible(), false);
    assert.match(await page.locator("#book-flow-GC").textContent(), /Sampled order-book observations; not a complete execution tape/);
    assert.match(await page.locator("#flow-depth-GC-10").textContent(), /UNAVAILABLE/);
    assert.match(await page.locator("#flow-volume-GC").textContent(), /change UNAVAILABLE/);
    assert.match(await page.locator("#flow-feed-GC").textContent(), /granted 1500 ms.*mean 1800 ms/);
    assert.match(await page.locator("#option-identity-GC").textContent(), /UIC 1002.*underlying GCZ6 \/ 101/);
    assert.match(await page.locator("#option-quote-GC").textContent(), /bid size STALE_OR_MISSING/);
    assert.match(await page.locator("#option-volume-GC").textContent(), /AS_OF_EFFECTIVE_TIME_UNKNOWN.*effective unknown/);
    assert.match(await page.locator("#option-coverage-GC").textContent(), /actual pre-trigger 42 \/ 900s/);
    assert.match(await page.locator("#option-cost-GC").textContent(), /MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET/);
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
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path:path.join(output,"book-flow-mobile-fixture.png"),fullPage:true});
    await page.setViewportSize({width:1440,height:1080});
    await page.goto(`${base}/markets`);
    assert.equal(await page.locator("#selected-market").inputValue(), "GC");
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
    await page.screenshot({
      path: path.join(output, "history-desktop-fixture.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#trades .table-scroll").evaluate((n) => {
      n.scrollLeft = 220;
      n.scrollTop = 140;
    });
    await page.evaluate(() => refresh());
    assert.equal(
      await page.locator("#trades .table-scroll").evaluate((n) => n.scrollLeft),
      220,
    );
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
    const boxes = await page
      .locator(".market")
      .evaluateAll((nodes) => nodes.map((n) => n.getBoundingClientRect().x));
    assert(boxes.every((x) => x === boxes[0]));
    await page.screenshot({
      path: path.join(output, "overview-mobile-fixture.png"),
      fullPage: true,
    });
    await page.goto(`${base}/system`);
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
      /2 \/ 5 markets with received depth/,
    );
    assert.match(await page.locator("#api-external").textContent(), /explicit/);
    await page.screenshot({
      path: path.join(output, "system-api-fixture.png"),
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    console.log(
      "PASS: five fixed cards, signal/fill markers, provisional P&L, refresh identity/focus/scroll/filter retention, mobile layout and System",
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
