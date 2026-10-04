/* Production Connect form + privacy header, local fake OAuth only. No Saxo calls. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { chromium } = require("playwright");
const root = path.resolve("packages/stocker_dashboard/src/stocker_dashboard/static");
const starts = [];
let base, signedIn = false;
const server = http.createServer((req, res) => {
  const url = new URL(req.url, base);
  res.setHeader("Referrer-Policy", "no-referrer");
  if (url.pathname === "/oauth/saxo/start") {
    starts.push({origin:req.headers.origin, accept:req.headers.accept, method:req.method});
    if (req.headers.origin !== base) {
      res.writeHead(403, {"Content-Type":"application/json"});
      return res.end(JSON.stringify({detail:"Cross-origin access rejected"}));
    }
    res.writeHead(200, {"Content-Type":"application/json"});
    return res.end(JSON.stringify({authorization_url:`http://localhost:${idp.address().port}/authorize`}));
  }
  if (url.pathname === "/oauth/saxo/callback") {
    signedIn = true;
    res.writeHead(303, {Location:"/system"});
    return res.end();
  }
  if (url.pathname.startsWith("/api/")) {
    res.setHeader("Content-Type","application/json");
    return res.end(JSON.stringify(url.pathname === "/api/system"
      ? {oauth:signedIn ? "AUTHENTICATED_FIXTURE" : "RECONNECT_REQUIRED",markets:[]}
      : {active:[],completed:[]}));
  }
  const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
  res.setHeader("Content-Type",file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
  res.end(fs.readFileSync(path.join(root,file)));
});
const idp = http.createServer((req,res) => {
  res.writeHead(303,{Location:base+"/oauth/saxo/callback?state=fixture&code=fixture"});
  res.end();
});
(async () => {
  let browser;
  try {
    await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
    await new Promise(resolve=>idp.listen(0,"127.0.0.1",resolve));
    base=`http://127.0.0.1:${server.address().port}`;
    browser=await chromium.launch({headless:true});
    const page=await browser.newPage();
    await page.goto(base+"/system");
    await Promise.all([
      page.waitForResponse(response=>response.url()===base+"/oauth/saxo/start"),
      page.getByRole("button",{name:"Connect configured Saxo environment"}).click(),
    ]);
    assert.equal(starts.length,1);
    assert.equal(starts[0].origin,base,`Connect request origin: ${starts[0].origin}; body: ${await page.locator("body").innerText()}`);
    assert.equal(starts[0].method,"POST");
    assert.match(starts[0].accept,/application\/json/);
    await page.waitForFunction(()=>document.querySelector("#auth-state")?.textContent.includes("AUTHENTICATED_FIXTURE"));
    assert.equal(page.url(),base+"/system");
    console.log("PASS: Connect uses same-origin fetch with no-referrer; local OAuth redirect returns to System");
  } finally {
    if (browser) await browser.close();
    await Promise.all([new Promise(resolve=>server.close(resolve)),new Promise(resolve=>idp.close(resolve))]);
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
