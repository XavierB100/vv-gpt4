const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'@playwright/test');
const path=require('node:path'),assert=require('node:assert/strict');
(async()=>{const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH});try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto(process.env.STUDIO_URL||'http://127.0.0.1:5054');
 await page.waitForFunction(()=>document.querySelector('#run-step').textContent!=='—');
 await page.locator('[data-open=history]').click();
 await page.locator('#runs button').filter({hasText:'Shakespeare / continued learning'}).click();
 await page.waitForFunction(()=>document.querySelector('#run-name').textContent==='Shakespeare / continued learning');
 assert.equal(await page.locator('dialog[open]').count(),0);
 assert.equal(await page.locator('#training-form').isVisible(),false);
 await page.locator('#loss-chart').hover();await page.locator('#chart-tooltip').waitFor({state:'visible'});
 await page.mouse.move(0,0);
 await page.screenshot({path:path.resolve('docs/research-desktop.png'),fullPage:true});
 const count=await page.locator('#compare-run option').count();assert(count>=2);
 await page.locator('#compare-run').selectOption({index:1});
 await page.waitForFunction(()=>!document.querySelector('.comparison-key').hidden);
 await page.locator('#new-run').click();
 await page.locator('#initial-model').selectOption('legacy:shakespeare-small-v1');
 assert(await page.locator('#architecture').isDisabled());
 assert(await page.locator('[name=context]').isDisabled());
 await page.keyboard.press('Escape');
 assert.equal(await page.locator('dialog[open]').count(),0);
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:path.resolve('docs/research-mobile.png'),fullPage:true});
 for(const name of ['history','models','data','play']){
  if(name==='history') await page.locator('[data-view=train]').click();
  await page.locator(`[data-open=${name}]`).click();
  assert.equal(await page.evaluate(()=>[...document.querySelectorAll('dialog[open]')].some(d=>d.scrollWidth>d.clientWidth)),false,name+' overflow');
  await page.keyboard.press('Escape');
 }
 await page.locator('[data-view=train]').click();
 await page.locator('#new-run').click();
 await page.screenshot({path:path.resolve('docs/research-drawer-mobile.png'),fullPage:false});
 assert.equal(await page.evaluate(()=>[...document.querySelectorAll('dialog[open]')].some(d=>d.scrollWidth>d.clientWidth)),false,'setup overflow');
 assert.deepEqual(errors,[]);console.log('Research layout, chart comparison, drawers, inherited training settings and mobile overflow passed.');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1});
