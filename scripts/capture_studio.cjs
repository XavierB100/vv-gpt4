// Capture the actual local studio after verification, without changing training runs.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'@playwright/test');
const path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
(async()=>{
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:5054');
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent==='completed');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(root,'docs/studio-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  for(const name of ['train','play','data','models']){
   await page.locator(`[data-page=${name}]`).click();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${name} mobile overflow`);
  }
  await page.locator('[data-page=train]').click();
  await page.screenshot({path:path.join(root,'docs/studio-mobile.png'),fullPage:true});
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('[data-page=models]').click();
  await page.screenshot({path:path.join(root,'docs/models-desktop.png'),fullPage:true});
  if(process.argv.includes('--generate')){
   await page.locator('[data-page=play]').click();
   await page.locator('#sample-model').selectOption('legacy:shakespeare-small-v1');
   await page.locator('#sample-prompt').fill('ROMEO:\nThe night is quiet, and');
   await page.locator('#sample-tokens').fill('120');
   await page.locator('#sample-submit').click();
   await page.waitForFunction(()=>!document.querySelector('#sample-submit').disabled&&document.querySelector('#enhanced-meta').textContent.includes('Seed 42'),{},{timeout:90000});
   await page.screenshot({path:path.join(root,'docs/playground-desktop.png'),fullPage:true});
  }
  await page.goto('http://127.0.0.1:5054/report');
  assert.match(await page.title(),/evolution/);
  await page.screenshot({path:path.join(root,'docs/report-preview.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'report mobile overflow');
  assert.deepEqual(errors,[]);
  console.log('Captured actual pilot curves, models, mobile studio and evidence report. No page errors or horizontal overflow.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
