const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const {chromium} = require("playwright");
const root=path.resolve("packages/stocker_dashboard/src/stocker_dashboard/static");
const calls=[];
const system={data_environment:"SAXO_SIM",execution_mode:"DISABLED",connected:true,reconciled:true,armed:false,paused:false,reserved_open_trades:0,allocation_pennies:0,limits:{per_trade_gbp:50,allocation_gbp:200,slots:4},entry_block_reason:"EXECUTION_DISABLED"};
const account={label:"SIM · simulated funds",environment:"SIM",account:"••••1234",currency:"USD",status:"Current",last_success_at:Date.now()/1000,valid_until:Date.now()/1000+60,total_value:100,cash_balance:null,cash_available_for_trading:0,connection_note:"Real-money balances are not connected",details:{}};
let stalled=false, fail=false, failControl=false;
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,"http://localhost");
 if(url.pathname.startsWith("/api/")) {
   calls.push(req.url);
   if(stalled && url.pathname==="/api/overview") return;
   if(fail || (failControl && url.pathname.startsWith("/api/entries/"))) {res.writeHead(503,{"Content-Type":"application/json"}); return res.end(JSON.stringify({detail:"Fixture service unavailable"}));}
   let data;
   if(url.pathname==="/api/overview") data={system,account,markets:[],pnl:{realised_net_gbp:12,skip_reasons:{}}};
   else if(url.pathname.startsWith("/api/entries/")) {system.paused=url.pathname.endsWith("pause");data=system;}
   else if(url.pathname.startsWith("/api/market/")) {
     const market=url.pathname.split("/").at(-1);
     data={system,markets:[{market,contract:market,conditions:{},trades:[],l1:{},l2:{},chart:[],strategy_state:"BLOCKED",block_reason:"EXECUTION_DISABLED"}]};
   } else if(url.pathname==="/api/history") data={system,rows:[{id:url.searchParams.get("market") || "all",market:url.searchParams.get("market") || "CL",signal_at:new Date().toISOString(),decision:"SKIPPED",rule_version:"fixture"}],has_more:false};
   else if(url.pathname==="/api/execution") data={system,trades:[],orders:[],fills:[],positions:[]};
   else if(url.pathname==="/api/detail") data={identity:url.searchParams.get("identity")};
   else if(url.pathname==="/api/system") data={...system,markets:[]};
   else data={active:[],completed:[]};
   const send=()=>{res.setHeader("Content-Type","application/json");res.end(JSON.stringify(data));};
   if(url.pathname.endsWith("/CL") || url.searchParams.get("market")==="CL") setTimeout(send,200);
   else send();
   return;
 }
 const file=url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
 res.setHeader("Content-Type",file.endsWith(".js")?"text/javascript":file.endsWith(".css")?"text/css":"text/html");res.end(fs.readFileSync(path.join(root,file)));
});
(async()=>{
 await new Promise(r=>server.listen(0,"127.0.0.1",r));
 const browser=await chromium.launch({headless:true});
 try {
  const base=`http://127.0.0.1:${server.address().port}`;
  const page=await browser.newPage(); const errors=[];
  page.on("pageerror",e=>errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(()=>document.querySelector("#account-value").textContent.includes("100.00 USD"));
  assert.equal(await page.locator("#account-cash").textContent(),"Unavailable");
  assert.equal(await page.locator("#account-available").textContent(),"0.00 USD");
  account.status="Stale";
  await page.evaluate(()=>refresh(true));
  assert.match(await page.locator("#account-freshness").textContent(),/Stale.*last successful/);
  account.environment="LIVE";account.label="LIVE · real-money account";account.connection_note="Selected authenticated LIVE account · ordering disabled";
  await page.evaluate(()=>refresh(true));
  assert.match(await page.locator("#account").textContent(),/LIVE · real-money/);
  failControl=true;
  await page.locator("#pause").click();
  await page.waitForFunction(()=>!document.querySelector("#pause").disabled);
  assert.equal(await page.locator("#pause").textContent(),"Pause entries");
  assert.match(await page.locator("#notice").textContent(),/Fixture service unavailable/);
  failControl=false;
  await page.locator("#pause").click();
  await page.waitForFunction(()=>document.querySelector("#pause").textContent==="Resume entries");
  assert.match(await page.locator("#notice").textContent(),/Server confirmed/);
  assert.equal(await page.locator("#realised").textContent(),"£12.00");
  fail=true;await page.evaluate(()=>refresh(true));assert.equal(await page.evaluate(()=>pending),null);
  fail=false;await page.evaluate(()=>refresh(true));assert.equal(await page.locator("#notice").textContent(),"");
  // A hung response must time out and release the actual refresh guard.
  await page.clock.install();stalled=true;
  await page.evaluate(()=>{window.waitingRefresh=refresh(true);});
  await page.clock.fastForward(8001);
  await page.evaluate(()=>window.waitingRefresh);
  assert.equal(await page.evaluate(()=>pending),null);
  assert.match(await page.locator("#notice").textContent(),/timed out/);
  stalled=false;await page.evaluate(()=>refresh(true));
  assert.equal(await page.locator("#notice").textContent(),"");
  await page.goto(base+"/markets");
  await page.locator("#selected-market").selectOption("NG");
  await page.waitForFunction(()=>document.querySelector("#contract-NG").textContent==="NG");
  await new Promise(r=>setTimeout(r,250));
  assert.equal(await page.locator("#contract-CL").textContent(),"");
  await page.goto(base+"/opportunities");
  await page.locator("#market-filter").selectOption("CL");
  await page.locator("#market-filter").selectOption("SI");
  await page.waitForFunction(()=>document.querySelector("#history tr")?.dataset.key==="SI");
  await new Promise(r=>setTimeout(r,250));
  assert.equal(await page.locator("#history tr").getAttribute("data-key"),"SI");
  const before=calls.length;
  await page.evaluate(()=>{Object.defineProperty(document,"hidden",{configurable:true,value:true});document.dispatchEvent(new Event("visibilitychange"));});
  await page.evaluate(()=>refresh(true)); assert.equal(calls.length,before);
  // Multiple browser consumers request only their own page's state.
  const pages=await Promise.all(["/markets","/opportunities","/execution","/system"].map(async route=>{const p=await browser.newPage();await p.goto(base+route);return p;}));
  const start=calls.length;
  await Promise.all(pages.map(p=>p.evaluate(()=>refresh(true))));
  assert(!calls.slice(start).some(p=>p.startsWith("/api/overview")));
  assert(!calls.slice(start).some(p=>p.startsWith("/api/recordings")));
  assert.deepEqual(errors,[]);
  console.log("PASS: timeout/error/cancellation recovery; obsolete selection/filter responses; confirmed controls; native SIM/LIVE/unavailable/stale balances; four browser clients; hidden-tab suppression");
 } finally {await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);server.closeAllConnections();server.close();process.exitCode=1;});
