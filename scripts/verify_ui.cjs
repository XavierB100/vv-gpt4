// Real custom-model UI integration in isolated storage. Requires Playwright/Chromium.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'@playwright/test');
const {spawn}=require('node:child_process');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),temporary=fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(),'vv-gpt4-ui-')));
const port=5064,url=`http://127.0.0.1:${port}`;
const server=spawn(process.env.VVGPT_PYTHON||path.join(root,'.venv/bin/python'),['app.py'],{cwd:root,env:{...process.env,VVGPT_PORT:String(port),VVGPT_DATA:temporary,VVGPT_LEGACY_MODELS:path.join(temporary,'no-legacy')},stdio:['ignore','ignore','pipe']});
let browser,serverErrors='';server.stderr.on('data',x=>serverErrors+=x);
(async()=>{
  let ready=false;
  for(let i=0;i<100;i++){try{const r=await fetch(url+'/health');const h=await r.json();if(r.ok&&h.data===temporary&&h.mode==='custom-models'){ready=true;break;}}catch{}await new Promise(r=>setTimeout(r,100));}
  assert(ready,'Test server did not start: '+serverErrors);
  browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH});
  const page=await browser.newPage({viewport:{width:1440,height:1024}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(url);
  await page.waitForFunction(()=>document.querySelector('#sample-model').options.length===1);
  assert(await page.locator('#sample-submit').isDisabled());
  assert.equal(await page.locator('#tokenizer').inputValue(),'gpt2');
  await page.locator('#new-run').click();
  await page.locator('#setup-example').click();
  await page.waitForFunction(()=>document.querySelector('#dataset option').value!=='');
  assert.equal(await page.locator('#setup-example').isVisible(),false);
  await page.locator('#dataset-file').setInputFiles({name:'ui-smoke.txt',mimeType:'text/plain',buffer:Buffer.from('A river flows through the valley. The village sleeps beneath the stars.\n'.repeat(400))});
  await page.waitForFunction(()=>document.querySelector('#dataset option').value!=='');
  await page.locator('[name=name]').fill('UI smoke only');
  await page.locator('#training-form summary').click();
  await page.locator('#tokenizer').selectOption('compact');
  await page.locator('[name=steps]').fill('6');
  await page.locator('[name=context]').fill('32');
  await page.locator('[name=batch]').fill('2');
  await page.locator('[name=eval_every]').fill('2');
  await page.getByRole('button',{name:'Start training',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent==='completed',{},{timeout:90000});
  assert.equal(await page.locator('#run-step').innerText(),'6 / 6');
  assert.match(await page.locator('#run-insight').innerText(),/bits\/byte/);
  await page.locator('#loss-chart').hover();
  await page.locator('#chart-tooltip').waitFor({state:'visible'});
  await page.locator('#chart-mode').selectOption('speed');
  assert.match(await page.locator('#chart-caption').innerText(),/Steps\/s/);
  await page.locator('[data-open=models]').click();
  await page.locator('#model-library').getByText('UI smoke only',{exact:true}).waitFor();
  await page.keyboard.press('Escape');
  await page.locator('[data-open=play]').click();
  const trained=await (await fetch(url+'/api/state')).json();
  await page.locator('#sample-model').selectOption('run:'+trained.runs[0].id);
  await page.locator('#sample-prompt').fill('ROMEO:\nThe night is quiet, and');
  await page.locator('#sample-form summary').click();
  await page.locator('#sample-tokens').fill('40');
  await page.locator('#compare').check();
  await page.locator('#sample-seed').fill('123');
  await page.locator('#sample-submit').click();
  await page.waitForFunction(()=>!document.querySelector('#sample-submit').disabled&&document.querySelector('#enhanced-meta').textContent.includes('seed 123'),{},{timeout:90000});
  const output=await page.locator('#enhanced-output').innerText();
  assert(output.length>5);
  await page.reload();
  await page.keyboard.press('Escape');
  await page.locator('[data-open=play]').click();
  await page.locator('.saved-section summary').click();
  await page.locator('#samples button').first().click();
  await page.waitForFunction(value=>document.querySelector('#enhanced-output').innerText===value,output);
  assert.equal(await page.locator('#enhanced-output').innerText(),output);
  assert.equal(await page.locator('#sample-seed').inputValue(),'123');
  assert.equal(await page.locator('dialog[open]').count(),0);
  assert.equal(await page.locator('#drawer-play').isVisible(),true);
  assert.equal(await page.locator('#sample-tokens').inputValue(),'40');
  // Train a child through the drawer; the parent's architecture/context/vocabulary must inherit.
  await page.keyboard.press('Escape');
  await page.locator('[data-open=models]').click();
  const parentCard=page.locator('.model-card').filter({has:page.getByRole('heading',{name:'UI smoke only',exact:true})});
  await parentCard.getByRole('button',{name:'Continue training'}).click();
  await page.locator('[name=name]').fill('UI continued child');
  await page.locator('[name=steps]').fill('4');
  assert(await page.locator('#architecture').isDisabled());
  assert(await page.locator('[name=context]').isDisabled());
  await page.locator('#training-form summary').click();
  await page.locator('[name=batch]').fill('2');
  await page.locator('[name=eval_every]').fill('2');
  await page.getByRole('button',{name:'Start training',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent==='completed'&&document.querySelector('#run-name').textContent==='UI continued child',{},{timeout:90000});
  assert.match(await page.locator('#run-detail').innerText(),/continued from your checkpoint/);
  const state=await (await fetch(url+'/api/state')).json();
  const parent='run:'+state.runs.find(r=>r.name==='UI smoke only').id;
  const child='run:'+state.runs.find(r=>r.name==='UI continued child').id;
  await page.locator('[data-open=play]').click();
  await page.locator('#sample-model').selectOption(child);
  await page.locator('#baseline-model').selectOption(parent);
  await page.locator('#sample-submit').click();
  await page.waitForFunction(()=>!document.querySelector('#sample-submit').disabled&&document.querySelector('#baseline-label').textContent==='UI smoke only',{},{timeout:90000});
  assert.match(await page.locator('#baseline-meta').innerText(),/seed 123/);
  await page.setViewportSize({width:390,height:844});
  await page.keyboard.press('Escape');
  for(const name of ['history','play','data','models']){
    if(name==='history') await page.locator('[data-view=train]').click();
    await page.locator(`[data-open=${name}]`).click();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth || [...document.querySelectorAll('dialog[open]')].some(d=>d.scrollWidth>d.clientWidth)),false,`${name} overflows on mobile`);
    await page.keyboard.press('Escape');
  }
  await page.locator('[data-view=train]').click();
  await page.locator('#new-run').click();
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('dialog[open]').count(),0);
  for (const width of [320,768,1024,1440]) {
    await page.setViewportSize({width,height:900});
    for (const view of ['train','play','models','data']) {
      await page.goto(url+'/#'+view);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${view} overflow at ${width}`);
    }
  }
  assert.deepEqual(errors,[]);
  const report={date:'2026-11-17',passed:true,checks:['original vocabulary default','bundled example without legacy files','dataset upload','real compact training','held-out score','chart hover and pace','own trained model generation comparison','sample and settings persistence','continued training with inherited settings','model-to-model comparison','four persistent destinations without mobile overflow','320-1440px responsive destinations','no browser errors']};
  fs.writeFileSync(path.join(root,'docs/ui-verification.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report));
})().catch(error=>{console.error(error);console.error(serverErrors.slice(-2000));process.exitCode=1;}).finally(async()=>{
  if(browser)await browser.close();server.kill('SIGTERM');
  if(!process.exitCode)fs.rmSync(temporary,{recursive:true,force:true});
});
