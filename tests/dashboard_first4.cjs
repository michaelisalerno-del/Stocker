const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
(async()=>{
 const browser = await chromium.launch({headless:true});
 const page = await browser.newPage({viewport:{width:1280,height:850}});
 const root=path.join(__dirname,'../packages/stocker_dashboard/src/stocker_dashboard/static');
 await page.route('http://stocker.test/**', async route=>{
  const url=new URL(route.request().url());
  if(url.pathname==='/api/overview') return route.fulfill({json:{system:{account:'DUP655399',connected:true,reconciled:true,armed:false,missing_settings:['premium_budget_usd'],settings:{},problem:'',session:'2026-09-23'},candidates:[{symbol:'<img src=x onerror=alert(1)>',decision:'SELECTED',slot:1,outcome:'UNARMED'}],orders:[],positions:[],fills:[],errors:[],pnl:{currency:'USD',realised:0}}});
  const file=url.pathname.startsWith('/static/')?url.pathname.slice(8):'index.html';
  return route.fulfill({body:fs.readFileSync(path.join(root,file)),contentType:file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html'});
 });
 await page.goto('http://stocker.test/');
 await page.getByText('Execution prerequisites',{exact:true}).waitFor();
 assert.equal(await page.locator('#system-status').textContent(),'UNARMED');
 assert.equal(await page.locator('#main img').count(),0);
 assert.equal(await page.getByText('Start run',{exact:true}).count(),0);
 assert.equal(await page.getByText('Session HARD',{exact:false}).count(),0);
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.getByRole('button',{name:'Pause entries'}).count(),1);
 await browser.close();console.log('PASS FIRST4 dashboard, escaped broker data, removed legacy controls');
})().catch(error=>{console.error(error);process.exit(1);});
