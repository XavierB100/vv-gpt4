'use strict';
const $ = selector => document.querySelector(selector);
const token = $('meta[name="vv-token"]').content;
let state, selectedRun = null, metrics = [], comparisonMetrics = [], comparisonName = '', hover = -1;
let lastSample = null, busy = false, refreshing = false, noticeTimer;
let previousInitial = '', scratchSettings = null, modelSelectionReady = false;
let currentView = 'train';
const storageKey = 'vvgpt.selectedRun';
try { selectedRun = localStorage.getItem(storageKey); } catch {}
const number = (value, digits = 2) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const duration = value => value == null ? '—' : value < 60 ? `${Math.round(value)}s` : `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`;
function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function notice(text, error = false) {
  clearTimeout(noticeTimer);
  $('#notice').textContent = text; $('#notice').className = error ? 'error' : ''; $('#notice').hidden = false;
  const dialog = document.querySelector('dialog[open]');
  (dialog || document.body).append($('#notice'));
  if (!error) noticeTimer = setTimeout(() => $('#notice').hidden = true, 6000);
}
const handle = fn => async event => { try { await fn(event); } catch (error) { notice(error.message, true); } };
async function api(url, options = {}) {
  const headers = {'X-VV-Token': token};
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(options.body);
  }
  const response = await fetch(url, {...options, headers}), data = await response.json();
  if (!response.ok) throw Error(data.error || 'Request failed.');
  return data;
}
function closeDrawers() { document.querySelectorAll('dialog[open]').forEach(el => el.close()); }
function navigate(name, updateHash = true) {
  if (!['train','play','models','data'].includes(name)) name = 'train';
  const changed = name !== currentView;
  closeDrawers(); currentView = name;
  for (const view of document.querySelectorAll('.workspace-view')) view.hidden = view.id !== (name === 'train' ? 'view-train' : 'drawer-' + name);
  for (const button of document.querySelectorAll('.topbar nav button')) {
    const active = (button.dataset.view || button.dataset.open) === name;
    if (active) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current');
  }
  if (updateHash && location.hash !== '#' + name) location.hash = name;
  document.title = `VV-GPT / ${{train:'Train',play:'Generate',models:'Models',data:'Data'}[name]}`;
  $('#notice').hidden = true;
  if (changed) window.scrollTo(0,0);
  if (name === 'train') requestAnimationFrame(draw);
}
function openDrawer(name) {
  if (['play','models','data'].includes(name)) { navigate(name); return; }
  closeDrawers(); $('#notice').hidden = true;
  $('#drawer-' + name).showModal();
}
function options(select, items, empty) {
  const previous = select.value; select.replaceChildren();
  if (!items.length) items = [{value:'',label:empty}];
  for (const item of items) { const option = node('option', '', item.label); option.value = item.value; select.append(option); }
  if (items.some(item => item.value === previous)) select.value = previous;
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    state = await api('/api/state');
    if (selectedRun && !state.runs.some(run => run.id === selectedRun)) selectedRun = null;
    if (!selectedRun && state.runs.length) selectedRun = state.runs[0].id;
    renderRuns(); renderModels(); renderData(); renderSamples();
    options($('#dataset'), state.datasets.map(d => ({value:d.id,label:d.name})), 'Import a dataset');
    datasetDetail(); initialDetail();
    $('#setup-example').hidden = !!state.datasets.length;
    if (selectedRun) await loadRun(selectedRun);
    $('#workspace-status').textContent = state.runs.some(r => ['running','queued','evaluating'].includes(r.status)) ? 'Training locally' : 'Local workspace';
  } finally { refreshing = false; }
}
function renderRuns() {
  $('#run-count').textContent = state.runs.length;
  const target = $('#runs'); target.replaceChildren();
  if (!state.runs.length) target.append(node('p','empty','No experiments yet. Create a run to start training.'));
  const query = $('#run-search').value.trim().toLowerCase();
  for (const run of state.runs.filter(run => run.name.toLowerCase().includes(query))) {
    const button = node('button', run.id === selectedRun ? 'selected' : '');
    button.append(node('span','',run.name), node('small','',`${run.status.replaceAll('_',' ')} / ${run.step.toLocaleString()} steps`));
    button.onclick = handle(async () => { selectedRun = run.id; $('#compare-run').value = ''; closeDrawers(); navigate('train'); await loadRun(run.id); });
    target.append(button);
  }
  if (state.runs.length && !target.children.length) target.append(node('p','empty','No experiments match your search.'));
}
async function loadRun(id) {
  const data = await api('/api/runs/' + id);
  if (selectedRun !== id) return;
  const run = data.run, summary = data.summary, config = JSON.parse(run.config);
  try { localStorage.setItem(storageKey, id); } catch {}
  metrics = data.metrics;
  $('#run-name').textContent = run.name; $('#run-status').textContent = run.status.replaceAll('_',' ');
  $('#run-status').dataset.status = run.status;
  $('#run-step').textContent = `${run.step.toLocaleString()} / ${config.steps.toLocaleString()}`;
  $('#run-best').textContent = number(run.best, 3);
  $('#run-speed').textContent = summary.steady_ms_per_step ? `${number(summary.steady_ms_per_step,0)} ms` : '—';
  $('#run-time').textContent = duration(summary.elapsed); $('#run-progress').value = run.step / config.steps * 100;
  $('#run-error').textContent = run.error || '';
  $('#run-detail').textContent = summary.parameters ? `${(summary.parameters/1e6).toFixed(2)}M parameters, ${summary.architecture || 'classic'} Transformer${summary.initial_model ? ', continued from your checkpoint' : ''}` : 'Preparing dataset…';
  $('#run-context').textContent = `${summary.release_demo ? 'Recorded demo · ' : ''}${config.context} token context on ${summary.device?.toUpperCase() || 'your device'}`;
  $('#use-run').disabled = !state.models.some(model => model.id === 'run:' + id);
  const startLoss = metrics[0]?.val;
  $('#loss-change').textContent = startLoss > 0 && run.best != null && run.best < startLoss ? `${((1-run.best/startLoss)*100).toFixed(1)}% below first evaluation` : '';
  document.querySelectorAll('[data-control]').forEach(button => {
    button.disabled = button.dataset.control === 'resume' ? !['paused','failed','stopped','interrupted'].includes(run.status) : !['running','queued','paused'].includes(run.status);
  });
  $('.actions').hidden = ![...document.querySelectorAll('[data-control]')].some(button => !button.disabled);
  $('#run-insight').textContent = summary.test ? `Test sample: ${number(summary.test.bits_per_byte,3)} bits/byte across ${summary.test.chars.toLocaleString()} characters. Best weights saved at step ${summary.best_step.toLocaleString()}.` : 'The best weights are saved automatically using validation loss.';
  $('#run-insight').title = `${summary.split || '80/10/10'}; test covers up to 8,000 characters. Compare scores only on matching held-out text. Historical research splits have prior-selection caveats in the research notes.`;
  $('#metric-rows').replaceChildren();
  metrics.forEach(m => {
    const row = node('tr'); [m.step, number(m.train,4), number(m.val,4), m.lr.toExponential(2), duration(m.elapsed)].forEach(value => row.append(node('td','',value)));
    $('#metric-rows').append(row);
  });
  const comparable = state.runs.filter(other => {
    if (other.id === id) return false;
    const candidate = JSON.parse(other.config);
    return candidate.dataset === config.dataset && candidate.context === config.context && candidate.codec_sha256 === config.codec_sha256 && candidate.split_sha256 === config.split_sha256;
  });
  options($('#compare-run'), [{value:'',label:'Compare run'}, ...comparable.map(r => ({value:r.id,label:r.name}))]);
  comparisonMetrics = []; comparisonName = '';
  if ($('#compare-run').value) {
    const comparison = await api('/api/runs/' + $('#compare-run').value);
    comparisonMetrics = comparison.metrics; comparisonName = comparison.run.name;
  }
  renderRuns(); draw();
}
function draw() {
  const canvas = $('#loss-chart'), context = canvas.getContext('2d'), width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = devicePixelRatio || 1; canvas.width = width*ratio; canvas.height = height*ratio; context.scale(ratio,ratio);
  const pad = {left:width<500?37:45,right:width<500?14:70,top:24,bottom:35}, pace = $('#chart-mode').value === 'speed';
  const series = pace ? [{name:'Steps/s',color:'#a7b4fa',points:metrics.map((m,i) => ({step:m.step,value:i ? (m.step-metrics[i-1].step)/Math.max(.001,m.elapsed-metrics[i-1].elapsed) : m.step/Math.max(.001,m.elapsed)}))}] : [
    {name:'Training',color:'#a7b4fa',points:metrics.map(m => ({step:m.step,value:m.train}))},
    {name:'Validation',color:'#7bc6b5',points:metrics.map(m => ({step:m.step,value:m.val}))}
  ];
  if (!pace && comparisonMetrics.length) series.push({name:comparisonName,color:'#d9ad83',dashed:true,points:comparisonMetrics.map(m => ({step:m.step,value:m.val}))});
  const values = series.flatMap(s => s.points.map(p => p.value)).filter(Number.isFinite);
  const low = pace || !values.length ? 0 : Math.max(0,Math.min(...values)-.15);
  const high = Math.max(low+.1, values.length ? Math.max(...values)*1.04 : 1);
  const maxStep = Math.max(1,...series.flatMap(s => s.points.map(p => p.step)));
  const x = step => pad.left + step/maxStep*(width-pad.left-pad.right);
  const y = value => pad.top + (high-value)/(high-low)*(height-pad.top-pad.bottom);
  context.font = `${width<500?10:11}px Manrope, sans-serif`; context.textAlign = 'right';
  for (let i=0;i<=4;i++) {
    const value = low+(high-low)*i/4, yy = y(value);
    context.strokeStyle = '#ffffff0b'; context.lineWidth = 1; context.setLineDash([3,5]);context.beginPath(); context.moveTo(pad.left,yy); context.lineTo(width-pad.right,yy); context.stroke();context.setLineDash([]);
    context.fillStyle = '#a1a9b5'; if (values.length) context.fillText(value.toFixed(1),pad.left-10,yy+4);
  }
  context.textAlign = 'center';
  for (let i=0;i<=4;i++) { const step=Math.round(maxStep*i/4); if (metrics.length) context.fillText(step.toLocaleString(),x(step),height-11); }
  $('#chart-empty').hidden = !!metrics.length;
  $('#chart-caption').textContent = pace ? 'Steps/s includes evaluation and saves' : 'Loss over training steps';
  $('.legend').hidden = pace;
  $('.comparison-key').hidden = !comparisonMetrics.length;
  $('.comparison-key').title = comparisonName;
  for (const s of series) {
    if (!s.points.length) continue;
    if (s.name === 'Validation' && metrics.length>1) {
      const fill = context.createLinearGradient(0,pad.top,0,height-pad.bottom);
      fill.addColorStop(0,'#7bc6b510');fill.addColorStop(1,'#7bc6b500');
      context.beginPath();context.moveTo(x(s.points[0].step),height-pad.bottom);
      for (const p of s.points) context.lineTo(x(p.step),y(p.value));
      context.lineTo(x(s.points.at(-1).step),height-pad.bottom);context.closePath();context.fillStyle=fill;context.fill();
    }
    context.beginPath(); context.setLineDash(s.dashed ? [5,5] : []);
    s.points.forEach((p,i) => i ? context.lineTo(x(p.step),y(p.value)) : context.moveTo(x(p.step),y(p.value)));
    context.strokeStyle = s.color; context.lineWidth = 2.3; context.lineJoin='round';context.stroke(); context.setLineDash([]);
    const last=s.points.at(-1);context.beginPath();context.arc(x(last.step),y(last.value),3.5,0,Math.PI*2);context.fillStyle=s.color;context.fill();
    if (width>=500 && !s.dashed) {context.textAlign='left';context.fillText(last.value.toFixed(3),x(last.step)+12,y(last.value)+4);}
  }
  const tooltip = $('#chart-tooltip'); tooltip.hidden = true;
  if (hover >= 0 && metrics[hover]) {
    const step=metrics[hover].step, xx=x(step); context.strokeStyle='#667180'; context.setLineDash([3,5]);
    context.beginPath(); context.moveTo(xx,pad.top); context.lineTo(xx,height-pad.bottom); context.stroke(); context.setLineDash([]);
    tooltip.hidden=false; tooltip.style.left=`${Math.min(width-180,Math.max(0,xx+15))}px`;
    tooltip.textContent=`Step ${step}\n`+series.map(s => {const p=s.points.find(p=>p.step===step);return p ? `${s.name}: ${p.value.toFixed(3)}` : null;}).filter(Boolean).join('\n');
  }
  canvas._x=x;
  if (metrics.length) canvas.setAttribute('aria-label',pace ? 'Training steps per second. Measured intervals include evaluation and saving.' : `Latest training loss ${number(metrics.at(-1).train,3)}, validation ${number(metrics.at(-1).val,3)}. Every point is available in Recorded measurements.`);
}
function datasetDetail() {
  const data = state.datasets.find(d => d.id === $('#dataset').value);
  $('#dataset-detail').textContent = data ? `${data.chars.toLocaleString()} prepared characters. Original retained.` : 'Import training text to begin.';
  $('#split-note').textContent = data?.analysis.split === 'explicit' ? 'Fixed research partitions. See the dataset provenance in research notes.' : '80% training, 10% validation, 10% final test.';
}
function initialDetail() {
  const chosen = $('#initial-model').value;
  const inherited = state.models.find(model => model.id === chosen);
  if (chosen !== previousInitial) {
    const fields = ['#architecture','#tokenizer','[name=context]'];
    if (chosen && !previousInitial) scratchSettings = Object.fromEntries(fields.map(selector => [selector,$(selector).value]));
    if (inherited) {
      $('#architecture').value = inherited.summary?.architecture || 'classic';
      $('#tokenizer').value = inherited.tokenizer;
      $('[name=context]').value = inherited.context;
    } else if (scratchSettings) {
      for (const [selector,value] of Object.entries(scratchSettings)) $(selector).value = value;
    }
    previousInitial = chosen;
  }
  $('#model-size').closest('label').hidden = !!chosen;
  for (const selector of ['#architecture','#model-size','#tokenizer','[name=context]']) $(selector).disabled = !!chosen;
  $('#initial-help').textContent = chosen ? 'Continue a copy of your best weights. Architecture, context and vocabulary are inherited. Use a fresh evaluation split if the model has already seen this data.' : 'New weights. Your existing checkpoints stay intact.';
}
function renderData() {
  const target=$('#datasets'); target.replaceChildren();
  if (!state.datasets.length) target.append(node('p','empty','Import a text file or use the public Shakespeare example.'));
  for (const data of state.datasets) {
    const row=node('article','dataset-row'), info=node('div'), size=node('div');
    info.append(node('h3','',data.name), node('p','',data.analysis.format === 'whatsapp' ? `${data.analysis.messages} messages. Speaker labels retained.` : 'Text / Unicode preserved'));
    (data.analysis.warnings||[]).forEach(w=>info.append(node('p','warning',w)));
    size.append(node('span','dataset-number',data.chars.toLocaleString()),node('p','','Prepared characters'));
    row.append(info,size); target.append(row);
  }
  filterLibrary('data');
}
function renderModels() {
  const target=$('#model-library'); target.replaceChildren();
  for (const model of state.models) {
    const s=model.summary||{}, card=node('article','model-card');
    const info=node('div','model-info');
    info.append(node('span','model-origin',model.readonly ? 'Original VV-GPT3' : s.initial_model ? 'Continued from your weights' : `${s.architecture||'Classic'} Transformer`),node('h2','',model.name));card.append(info);
    const dl=node('dl'), params=model.params||s.parameters;
    [['Parameters',params ? `${(params/1e6).toFixed(2)}M` : '—'],['Context',`${model.context} tokens`],['Validation',number(model.best,3)],['Test bits/byte',number(s.test?.bits_per_byte,3)]].forEach(([k,v])=>{const stat=node('div','model-stat');stat.append(node('dt','',k),node('dd','',v));dl.append(stat);});
    card.append(dl); const actions=node('div','model-actions');
    const generate=node('button','','Generate'); generate.onclick=()=>{$('#sample-model').value=model.id;openDrawer('play');};
    const train=node('button','','Continue training'); train.onclick=()=>{$('#initial-model').value=model.id;initialDetail();$('#training-form [name=lr]').value='0.0001';openDrawer('new');};
    actions.append(generate,train);card.append(actions);target.append(card);
  }
  if (!state.models.length) target.append(node('p','empty','Your trained models will appear here.'));
  const list=state.models.map(m=>({value:m.id,label:m.name}));
  options($('#sample-model'),list,'Train a model first');
  $('#sample-submit').disabled = busy || !list.length;
  $('#generation-help').hidden = !!list.length;
  if (!modelSelectionReady && list.length) {
    const current = 'run:' + selectedRun;
    if (list.some(item => item.value === current)) $('#sample-model').value = current;
    modelSelectionReady = true;
  }
  options($('#initial-model'),[{value:'',label:'Train from scratch'},...list]);
  options($('#baseline-model'),[{value:'',label:'Same model / no sampling penalties'},...list]);
  filterLibrary('model');
}
function filterLibrary(kind) {
  const query = $('#' + kind + '-search').value.trim().toLowerCase();
  const rows = document.querySelectorAll(kind === 'model' ? '.model-card' : '.dataset-row');
  let visible=0;
  rows.forEach(row => {row.hidden = !row.querySelector('h2,h3').textContent.toLowerCase().includes(query);if(!row.hidden)visible++;});
  $('#' + kind + '-empty').hidden = !query || visible>0;
}
function resultView(result) {
  lastSample=result; const baseline=result.baseline;
  $('#sample-columns').classList.toggle('single',!baseline); $('.sample-card.baseline').hidden=!baseline;
  $('#baseline-label').textContent = result.comparison_kind === 'weights' ? state.models.find(m=>m.id===result.settings?.baseline_model)?.name || 'Comparison model' : 'Without sampling penalties';
  for (const key of ['baseline','enhanced']) {
    const sample=result[key]; if (!sample) continue;
    $(`#${key}-output`).textContent=sample.text || '[End of text]';
    $(`#${key}-meta`).textContent=`${sample.tokens} tokens / ${sample.seconds}s / ${(sample.repeat_fraction*100).toFixed(1)}% repeated 4-token phrases\nCheckpoint step ${sample.step} / seed ${sample.seed}`+(sample.prompt_tokens>sample.used_prompt_tokens?`\nPrompt trimmed to its last ${sample.used_prompt_tokens} tokens.`:'');
  }
}
function renderSamples() {
  const target=$('#samples'); target.replaceChildren();
  for (const sample of state.samples) {
    const button=node('button','',sample.prompt.slice(0,100));
    button.onclick=handle(async()=>{
      const data=await api('/api/samples/'+sample.id), settings=data.settings;
      $('#sample-prompt').value=data.prompt;$('#sample-model').value=data.model;
      for (const [id,key] of [['sample-tokens','tokens'],['temperature','temperature'],['penalty','repetition_penalty'],['ngram','no_repeat_ngram'],['sample-seed','seed'],['baseline-model','baseline_model']]) if (settings[key]!==undefined) $('#'+id).value=settings[key];
      $('#compare').checked=!!settings.compare; $('#comparison-settings').hidden=!settings.compare;
      resultView({...data.result,model:data.model,prompt:data.prompt,settings});
    });target.append(button);
  }
}
async function upload(input) {
  if (!input.files.length) return;
  const form=new FormData(); form.append('file',input.files[0]);input.disabled=true;
  try {const result=await api('/api/datasets',{method:'POST',body:form});await refresh();$('#dataset').value=result.id;datasetDetail();notice('Dataset imported. Original text preserved.');}
  finally {input.disabled=false;input.value='';}
}
document.querySelectorAll('[data-open]').forEach(button=>button.onclick=()=>openDrawer(button.dataset.open));
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>navigate(button.dataset.view));
window.addEventListener('hashchange',()=>navigate(location.hash.slice(1),false));
document.querySelectorAll('[data-close]').forEach(button=>button.onclick=()=>button.closest('dialog').close());
document.querySelectorAll('dialog').forEach(dialog=>{
  dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
  dialog.addEventListener('close',()=>{document.body.append($('#notice'));$('#notice').hidden=true;});
});
$('.brand').onclick=event=>{event.preventDefault();navigate('train');};
$('.skip-link').onclick=event=>{event.preventDefault();$('#workspace').focus();};
$('#run-search').oninput=renderRuns;
$('#model-search').oninput=()=>filterLibrary('model');$('#data-search').oninput=()=>filterLibrary('data');
$('#use-run').onclick=()=>{$('#sample-model').value='run:'+selectedRun;navigate('play');$('#sample-prompt').focus();};
$('#new-run').onclick=()=>openDrawer('new');$('#refresh').onclick=handle(refresh);
$('#dataset').onchange=datasetDetail;$('#initial-model').onchange=()=>{initialDetail();$('#training-form [name=lr]').value=$('#initial-model').value?'0.0001':'0.0003';};
$('#compare').onchange=()=>$('#comparison-settings').hidden=!$('#compare').checked;
$('#dataset-file').onchange=handle(event=>upload(event.target));$('#data-file').onchange=handle(event=>upload(event.target));
$('#setup-example').onclick=()=>$('#example').click();
$('#example').onclick=handle(async()=>{await api('/api/datasets/example',{method:'POST'});await refresh();openDrawer('new');});
$('#compare-run').onchange=handle(()=>loadRun(selectedRun));$('#chart-mode').onchange=()=>{hover=-1;draw();};
$('#loss-chart').addEventListener('pointermove',event=>{
  if(!metrics.length)return;const xx=event.clientX-event.target.getBoundingClientRect().left;
  hover=metrics.reduce((best,m,i)=>Math.abs(event.target._x(m.step)-xx)<Math.abs(event.target._x(metrics[best].step)-xx)?i:best,0);draw();
});
$('#loss-chart').addEventListener('pointerleave',()=>{hover=-1;draw();});window.addEventListener('resize',draw);
$('#loss-chart').addEventListener('keydown',event=>{
  if (!metrics.length || !['ArrowLeft','ArrowRight','Home','End','Escape'].includes(event.key)) return;
  event.preventDefault();
  hover=event.key==='Escape'?-1:event.key==='Home'?0:event.key==='End'?metrics.length-1:Math.min(metrics.length-1,Math.max(0,hover+(event.key==='ArrowLeft'?-1:1)));draw();
});
$('#loss-chart').addEventListener('blur',()=>{hover=-1;draw();});
$('#training-form').onsubmit=handle(async event=>{
  event.preventDefault();const button=event.target.querySelector('[type=submit]');button.disabled=true;
  try{const data=Object.fromEntries(new FormData(event.target));const result=await api('/api/runs',{method:'POST',body:data});selectedRun=result.id;closeDrawers();navigate('train');await refresh();notice('Training started.');}finally{button.disabled=false;}
});
document.querySelectorAll('[data-control]').forEach(button=>button.onclick=handle(async()=>{await api(`/api/runs/${selectedRun}/${button.dataset.control}`,{method:'POST'});await loadRun(selectedRun);}));
$('#sample-form').onsubmit=handle(async event=>{
  event.preventDefault();if(busy || !state.models.length)return;busy=true;$('#sample-submit').disabled=true;$('#notice').hidden=true;$('#enhanced-output').textContent='Generating…';
  try{
    const settings={model:$('#sample-model').value,prompt:$('#sample-prompt').value,tokens:Number($('#sample-tokens').value),temperature:Number($('#temperature').value),repetition_penalty:Number($('#penalty').value),no_repeat_ngram:Number($('#ngram').value),seed:Number($('#sample-seed').value),compare:$('#compare').checked,baseline_model:$('#baseline-model').value};
    const result=await api('/api/sample',{method:'POST',body:settings});resultView({...result,model:settings.model,prompt:settings.prompt,settings});await refresh();
  }catch(error){$('#enhanced-output').textContent='Generation failed.';throw error;}finally{busy=false;$('#sample-submit').disabled=false;}
});
$('#export-sample').onclick=()=>{
  if(!lastSample){notice('Generate a sample first.');return;}
  const url=URL.createObjectURL(new Blob([JSON.stringify(lastSample,null,2)],{type:'application/json'})),link=node('a');link.href=url;link.download='vv-gpt-sample.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
setInterval(()=>{if(state?.runs.some(run=>['queued','running','paused','evaluating'].includes(run.status)))refresh().catch(error=>notice(error.message,true));},2500);
navigate(location.hash.slice(1),false);
document.fonts.ready.then(draw);
refresh().catch(error=>notice(error.message,true));
