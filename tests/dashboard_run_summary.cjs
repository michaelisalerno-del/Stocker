// All API traffic is mocked; no broker connection or order submission.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let requests = 123;
    let auditDownloads = 0;
    let evaluation = "NO_MORE_CHECKPOINTS_TODAY";
    let activity = null;
    let discovery = null;
    let selection = null;
    let acquisition = null;
    const staticPath = path.join(__dirname, "../packages/stocker_dashboard/src/stocker_dashboard/static");
    await page.context().route("http://stocker.test/**", async route => {
      const url = new URL(route.request().url());
      assert.equal(route.request().method(), "GET");
      const json = value => route.fulfill({ json: value });
      if (url.pathname === "/api/overview") return json({ environments: {}, system: "READY", active_runs: 1, open_positions: 0, as_of: "2026-09-08T19:00:00Z" });
      if (url.pathname === "/api/runs/test") return json({
        run_id: "test", display_name: "US · Session HARD", environment: "PAPER", enabled: true,
        evaluation_status: evaluation, last_checkpoint: "2026-09-08T16:00:00Z",
        evaluation_progress: { completed: 40, total: 6570, preparing_history: true, trade_stream_unavailable: 6565 },
        downloads: { scope: "ALL_RUNS", requests, pending: 4 },
        method_spec: { entry: { trigger_M: 0.2 } },
        activity_screen: activity,
        discovery,
        candidate_selection: selection,
        acquisition, session: "2026-09-08",
        funnel: [{ stage: "Stock eligibility", count: 6570 }, { stage: "Required data ready", count: 153 }],
      });
      if (url.pathname === "/api/runs/test/performance") return json({ history: [], currency: "USD", closed_trades: 0 });
      if (url.pathname === "/api/runs/test/provenance") {
        auditDownloads++;
        return route.fulfill({ json: { historical: "FULL_UNIVERSE_DATA" }, headers: { "Content-Disposition": 'attachment; filename="run-provenance.json"' } });
      }
      const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
      return route.fulfill({ body: await fs.readFile(path.join(staticPath, file)), contentType: file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html" });
    });
    await page.goto("http://stocker.test/runs?run=test");
    await page.getByText("No further method checkpoints today.", { exact: false }).waitFor();
    assert.match(await page.locator("main").innerText(), /123 history requests/);
    assert.equal(auditDownloads, 0);
    evaluation = "EVALUATING";
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /40 of 6,570 stocks processed/);
    assert.match(await page.locator("main").innerText(), /6,565 stocks lacked a required trade feed/);
    evaluation = "INCOMPLETE";
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /Checkpoint incomplete/);
    assert.equal(await page.locator("main").getByRole("row").count(), 0);
    assert(await page.getByRole("link", { name: "Browse data-ready stocks" }).getAttribute("href").then(href => href.includes("status=READY")));
    requests = 127;
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /127 history requests/);
    assert.equal(auditDownloads, 0);
    await page.getByText("Method specification", { exact: true }).click();
    assert.equal(await page.getByRole("link", { name: "Download full run audit record" }).getAttribute("download"), "run-provenance.json");
    // Chromium cancels native downloads for intercepted fictitious hosts. Exercise the same
    // explicit GET through fetch; the API suite verifies the attachment response header.
    const audit = await page.evaluate(async () => {
      const response = await fetch(document.querySelector("a[download]").href);
      return response.json();
    });
    assert.equal(audit.historical, "FULL_UNIVERSE_DATA");
    assert.equal(auditDownloads, 1);
    activity = { reason: "", candidates: [
      { symbol: "AAPL", currency: "USD", scan_hit_count: 3, aggregate_screen_score: 2.9, best_component_rank: 1, selected: true },
      { symbol: "MSFT", currency: "USD", scan_hit_count: 2, aggregate_screen_score: 1.8, best_component_rank: 3, selected: false },
    ] };
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /2 activity candidates · 1 selected/);
    await page.getByText("Inspect stock selection", { exact: true }).click();
    assert.match(await page.locator("main").innerText(), /AAPL/);
    assert.match(await page.locator("main").innerText(), /Outside capacity limit/);
    activity = null;
    discovery = {
      status: "READY", reason: "", raw_candidates: 214, unique_candidates: 187,
      watch_pool_size: 150, session_hard_qualified: 4,
      last_successful_discovery: "2026-09-08T13:46:00Z",
      warnings: ["IBKR scanner precision warning (492): LSE permissions"],
    };
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /Dynamic universe: IBKR/i);
    assert.match(await page.locator("main").innerText(), /precision warning \(492\): LSE permissions/);
    assert.match(await page.locator("main").innerText(), /Raw: 214 · Unique: 187 · Watch pool: 150 · Session HARD qualified: 4/);
    assert.equal(await page.getByRole("button", { name: "Rebuild on next enable" }).isDisabled(), true);
    await page.getByText("Discovery diagnostics", { exact: true }).click();
    assert.equal(await page.getByRole("link", { name: "Inspect discovery runs, filters and candidate provenance" }).getAttribute("href"), "/api/runs/test/discovery");
    assert(!await page.locator("main").innerText().then(text => text.includes("FULL_UNIVERSE_DATA")));
    discovery = null;
    selection = {
      state: "SESSION_HARD_ACTIVE", reason: "", session: "2026-09-08",
      broad_eligible: 6570, source_population_count: 6570, source: "AUTHORITATIVE_LISTINGS",
      recipe: { cross_market_evidence: "US_DEVELOPMENT_POPULATION", evidence_status: "VALIDATED_EXISTING_RESEARCH" },
      stages: [{minutes:5,stage_id:"RANGE250",selected:250}, {minutes:10,stage_id:"RV50",selected:50}, {minutes:15,stage_id:"RV30",selected:30}],
      session_hard_qualified: 4,
    };
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /Broad eligible/);
    assert.match(await page.locator("main").innerText(), /Range5 HIGH250 → RV10 HIGH50 → RV15 HIGH30/);
    assert.equal(await page.locator("main").getByRole("row").count(), 0);
    selection.recipe.cross_market_evidence = "UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER";
    selection.recipe.evidence_status = "UNVALIDATED_TRANSFER";
    selection.state = "DEGRADED";
    selection.reason = "BROAD_OPENING_DATA_CAPACITY_UNRESOLVED";
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER/);
    assert.match(await page.locator("main").innerText(), /BROAD_OPENING_DATA_CAPACITY_UNRESOLVED/);
    assert.equal(auditDownloads, 1);
    acquisition = {
      state: "SCANNER_ACQUISITION_READY", upstream_evidence: "PROSPECTIVE_IBKR_TEST",
      broad_membership: 6570, raw_hits: 1240, acquisition_count: 486,
      range5_prefixes_ready: 472, oracle_state: "COMPLETE", oracle_completed: 6570,
      oracle_metrics: { recipes: { ACTIVE: { RANGE250: {recall: .988}, RV50: {recall: 1}, RV30: {recall: 1} } } },
    };
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /Scanner-assisted acquisition/i);
    assert.match(await page.locator("main").innerText(), /PROSPECTIVE_IBKR_TEST/);
    assert.match(await page.locator("main").innerText(), /RANGE250 recall: 98.8%/);
    assert.equal(await page.locator("main").getByRole("row").count(), 0);
    assert.deepEqual(errors, []);
    console.log("PASS: compact summary, live download refresh, checkpoint status, on-demand audit, GET-only");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
