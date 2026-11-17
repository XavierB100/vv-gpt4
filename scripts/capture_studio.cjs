// Capture the actual studio without modifying training runs or model weights.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'@playwright/test');
const path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),url=process.env.STUDIO_URL||'http://127.0.0.1:5054';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(url);
  await page.waitForFunction(()=>document.querySelector('#run-name').textContent!=='No experiment selected');
  await page.locator('[data-open=history]').click();
  await page.locator('#runs button').filter({hasText:'Simple stories / further learning'}).click();
  await page.waitForFunction(()=>document.querySelector('#run-name').textContent==='Simple stories / further learning');
  await page.mouse.move(0,0);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(root,'docs/studio-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(root,'docs/studio-mobile.png'),fullPage:true});
  for(const name of ['history','play','data','models']){
   await page.locator(`[data-open=${name}]`).click();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth||[...document.querySelectorAll('dialog[open]')].some(d=>d.scrollWidth>d.clientWidth)),false,`${name} mobile overflow`);
   await page.keyboard.press('Escape');
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('[data-open=models]').click();
  await page.screenshot({path:path.join(root,'docs/models-desktop.png'),fullPage:true});
  await page.keyboard.press('Escape');
  if(process.argv.includes('--generate')){
   await page.locator('[data-open=play]').click();
   await page.locator('#sample-model').selectOption({label:'Simple stories / further learning'});
   await page.locator('#sample-prompt').fill('It was raining outside. Mia wanted to stay dry, so she took her');
   await page.locator('#sample-form summary').click();
   await page.locator('#sample-tokens').fill('160');
   await page.locator('#sample-seed').fill('42');
   await page.locator('#sample-submit').click();
   await page.waitForFunction(()=>!document.querySelector('#sample-submit').disabled&&document.querySelector('#enhanced-meta').textContent.includes('seed 42'),{},{timeout:90000});
   await page.locator('#sample-form summary').click();
   await page.locator('#drawer-play').evaluate(dialog=>dialog.scrollTop=0);
   await page.screenshot({path:path.join(root,'docs/playground-desktop.png'),fullPage:true});
  }
  await page.goto(url+'/report');
  assert.match(await page.title(),/Coherence research/);
  await page.screenshot({path:path.join(root,'docs/report-preview.png'),fullPage:false});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'report mobile overflow');
  assert.deepEqual(errors,[]);
  console.log('Captured real learning curves, models, mobile drawers and research report. No browser errors or horizontal overflow.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
