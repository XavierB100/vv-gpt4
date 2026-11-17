// Layout and navigation checks against the running app; reads existing runs only.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'@playwright/test');
const assert=require('node:assert/strict'),path=require('node:path');
const url=process.env.STUDIO_URL||'http://127.0.0.1:5054';
(async()=>{const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH});try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto(url);await page.waitForFunction(()=>document.querySelector('#run-name').textContent!=='Start your first experiment');await page.evaluate(()=>document.fonts.ready);
 await page.screenshot({path:path.resolve('docs/navigation-desktop.png'),fullPage:true});
 await page.locator('#loss-chart').hover();await page.locator('#chart-tooltip').waitFor({state:'visible'});
 await page.locator('#loss-chart').focus();await page.keyboard.press('End');assert.match(await page.locator('#chart-tooltip').innerText(),/Step/);
 await page.locator('#use-run').click();assert.equal(await page.locator('dialog[open]').count(),0);assert(await page.locator('#sample-prompt').evaluate(el=>el===document.activeElement));assert(await page.locator('[data-open=play]').getAttribute('aria-current')==='page');
 await page.locator('[data-open=models]').click();await page.locator('#model-search').fill('Shakespeare / continued learning');assert.equal(await page.locator('.model-card:visible').count(),1);await page.locator('#model-search').fill('no-such-model-xyz');assert(await page.locator('#model-empty').isVisible());await page.locator('#model-search').fill('');
 await page.screenshot({path:path.resolve('docs/navigation-models.png'),fullPage:true});
 await page.evaluate(()=>scrollTo(0,1200));
 await page.locator('[data-open=data]').click();assert.equal(await page.evaluate(()=>scrollY),0);assert.equal(await page.locator('.workspace-view:visible').count(),1);await page.goBack();assert(await page.locator('#drawer-models').isVisible());
 await page.locator('[data-view=train]').click();await page.locator('[data-open=history]').click();await page.locator('#run-search').fill('Shakespeare / continued learning');await page.locator('#runs button').click();await page.waitForFunction(()=>document.querySelector('#run-name').textContent==='Shakespeare / continued learning');
 await page.reload();await page.waitForFunction(()=>document.querySelector('#run-name').textContent==='Shakespeare / continued learning');
 await page.locator('[data-open=measurements]').click();assert(await page.locator('#metric-rows').isVisible());await page.keyboard.press('Escape');assert(await page.locator('[data-open=measurements]').evaluate(el=>el===document.activeElement));
 await page.locator('#new-run').click();await page.locator('#initial-model').selectOption('legacy:shakespeare-small-v1');assert(await page.locator('#architecture').isDisabled());await page.keyboard.press('Escape');
 await page.locator('[data-open=history]').click();await page.locator('#run-search').fill('Simple stories / further learning');await page.locator('#runs button').click();await page.waitForFunction(()=>document.querySelector('#run-name').textContent==='Simple stories / further learning');
 for(const width of [1024,768,390,320]){
  await page.setViewportSize({width,height:844});
  for(const view of ['train','play','models','data']){
   await page.locator(view==='train'?'[data-view=train]':`[data-open=${view}]`).click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,view+' overflow '+width);assert.equal(await page.locator('.workspace-view:visible').count(),1);
  }
  await page.locator('[data-view=train]').click();await page.locator('#new-run').click();assert.equal(await page.evaluate(()=>document.querySelector('dialog[open]').scrollWidth>document.querySelector('dialog[open]').clientWidth),false,'setup overflow');await page.keyboard.press('Escape');
  if(width===390)await page.screenshot({path:path.resolve('docs/navigation-mobile.png'),fullPage:true});
 }
 assert.deepEqual(errors,[]);console.log('Persistent navigation, browser back, run persistence, search, contextual generation, keyboard chart, sheet focus and 320–1440px layouts passed.');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1});
