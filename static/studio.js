'use strict';
const $ = s => document.querySelector(s);
const token = $('meta[name="vv-token"]').content;
let state, conversation = null, selectedRun = null, generating = false, currentPage = 'assistant';
let lastMetrics = [], activeController = null;
const titles = {assistant:'Assistant', knowledge:'Knowledge', lab:'Training lab', models:'My models', preferences:'Preferences'};
function el(tag, cls, text) { const node=document.createElement(tag); if(cls)node.className=cls; if(text!==undefined)node.textContent=text; return node; }
function notify(message, error=false) { const n=$('#notice'); n.textContent=message; n.className=error?'error':''; n.hidden=false; }
function clearNotice() { $('#notice').hidden=true; }
async function api(path, options={}) {
  const headers={'X-VV-Token':token,...options.headers};
  if(options.body && !(options.body instanceof FormData)){headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
  const response=await fetch(path,{...options,headers});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'The request failed.');
  return data;
}
function handle(fn) { return async event => {try{await fn(event);}catch(error){notify(error.message,true);}}; }
function page(name) {
  currentPage=name;
  document.querySelectorAll('.page').forEach(node=>node.hidden=node.id!==`page-${name}`);
  document.querySelectorAll('nav button').forEach(node=>node.classList.toggle('active',node.dataset.page===name));
  $('#page-title').textContent=titles[name];
  if(name==='lab' && selectedRun) loadRun(selectedRun).catch(e=>notify(e.message,true));
}
document.querySelectorAll('[data-page]').forEach(button=>button.addEventListener('click',()=>page(button.dataset.page)));
$('.brand').addEventListener('click',e=>{e.preventDefault();page('assistant');});
function options(select, items, placeholder) {
  const old=select.value; select.replaceChildren();
  if(!items.length){const o=el('option','',placeholder);o.value='';select.append(o);return;}
  items.forEach(item=>{const o=el('option','',item.label);o.value=item.value;select.append(o);});
  if(items.some(i=>i.value===old))select.value=old;
}
async function refresh() {
  const first=!state;
  state=await api('/api/state');
  const models=state.provider.models.map(m=>({value:m.name,label:m.name}));
  options($('#assistant-model'),models,state.provider.online?'No local models installed':'Open Ollama to connect');
  if(first && models.some(m=>m.value==='qwen3:8b'))$('#assistant-model').value='qwen3:8b';
  $('#engine-state').textContent=state.provider.online?`${models.length} local models available`:'Ollama is offline';
  $('#engine-dot').classList.toggle('offline',!state.provider.online);
  $('#send').disabled=generating||!models.length;
  if(first){$('#memory').value=state.memory; if(!state.provider.online)notify(state.provider.message);}
  renderHistory();renderDocuments();renderModels();renderRuns();
  options($('#dataset'),state.datasets.map(d=>({value:d.id,label:d.name})), 'Import a dataset to begin');
  datasetDetail();
}
function renderHistory(){
  const target=$('#history');target.replaceChildren();
  const picker=$('#mobile-history');picker.replaceChildren();const placeholder=el('option','','Conversations');placeholder.value='';picker.append(placeholder);
  state.conversations.forEach(c=>{const option=el('option','',c.title);option.value=c.id;picker.append(option);});picker.value=conversation||'';
  state.conversations.forEach(c=>{const button=el('button',c.id===conversation?'selected':'',c.title);button.title=c.title;
    button.addEventListener('click',handle(()=>openConversation(c.id)));target.append(button);});
  if(!state.conversations.length)target.append(el('div','empty','Your conversations will appear here.'));
}
$('#mobile-history').addEventListener('change',handle(e=>e.target.value?openConversation(e.target.value):Promise.resolve()));
function sourcesView(sources){
  const wrapper=el('div','sources');
  sources.forEach(source=>{const details=el('details');details.append(el('summary','',`[${source.citation}] ${source.name}`),el('p','',source.content));wrapper.append(details);});
  return wrapper;
}
function message(role,text,model='',status='complete',sources=[]){
  const node=el('article',`message ${role}`);const byline=el('div','byline');
  byline.append(el('strong','',role==='user'?'You':'VV-GPT'));
  if(model)byline.append(el('span','',model));
  const body=el('div','body',text|| (role==='assistant'?(status==='pending'?'Thinking…':'No answer was saved.'):''));
  node.append(byline,body);
  if(sources.length)node.append(sourcesView(sources));
  if(status!=='complete'&&status!=='pending')node.append(el('div','result-note',status==='truncated'?'Response reached its length limit.':`Response ${status}.`));
  $('#messages').append(node);return node;
}
async function openConversation(cid){
  if(generating){notify('Stop the current response before changing conversations.');return;}
  conversation=cid;page('assistant');$('#messages').replaceChildren();
  const data=await api(`/api/conversations/${cid}`);
  data.messages.forEach(m=>message(m.role,m.content,m.model,m.status,JSON.parse(m.sources||'[]')));
  $('#welcome').hidden=!!data.messages.length;renderHistory();
}
$('#new-chat').addEventListener('click',handle(async()=>{
  if(generating){notify('Stop the current response before starting a new conversation.');return;}
  conversation=null;$('#messages').replaceChildren();$('#welcome').hidden=false;$('#prompt').value='';renderHistory();page('assistant');$('#prompt').focus();
}));
document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>{$('#prompt').value=button.dataset.prompt;$('#prompt').focus();}));
$('#knowledge-shortcut').addEventListener('click',()=>page('knowledge'));
$('#refresh').addEventListener('click',handle(async()=>{await refresh();notify('Workspace refreshed.');}));
$('#prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!generating && !$('#send').disabled)$('#composer').requestSubmit();}});
$('#composer').addEventListener('submit',handle(async event=>{
  event.preventDefault();if(generating)return;
  const question=$('#prompt').value.trim(),model=$('#assistant-model').value;if(!question||!model)return;
  clearNotice();
  generating=true;$('#send').disabled=true;
  try{if(!conversation){conversation=(await api('/api/conversations',{method:'POST'})).id;}}
  catch(error){generating=false;$('#send').disabled=false;throw error;}
  const cid=conversation;
  generating=true;$('#send').disabled=true;$('#stop-chat').hidden=false;$('#prompt').value='';$('#welcome').hidden=true;
  message('user',question);const answer=message('assistant','',model,'pending'),body=answer.querySelector('.body');
  let output='',buffer='',gotDone=false;
  activeController=new AbortController();
  try{
    const response=await fetch(`/api/conversations/${cid}/chat`,{method:'POST',headers:{'Content-Type':'application/json','X-VV-Token':token},body:JSON.stringify({message:question,model,knowledge:$('#use-knowledge').checked}),signal:activeController.signal});
    if(!response.ok)throw new Error((await response.json()).error);
    const reader=response.body.getReader(),decoder=new TextDecoder();
    while(true){
      const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
      const lines=buffer.split('\n');buffer=lines.pop();
      for(const line of lines){
        if(!line.trim())continue;
        const event=JSON.parse(line);
        if(event.type==='sources'&&event.sources.length)answer.append(sourcesView(event.sources));
        if(event.type==='token'){output+=event.text;body.textContent=output;}
        if(event.type==='metrics')answer.append(el('div','result-note',`${event.tokens} tokens · ${event.tokens_per_second??'—'} tokens/s`));
        if(event.type==='error')throw new Error(event.error);
        if(event.type==='done'){gotDone=true;if(event.status!=='complete')answer.append(el('div','result-note',`Response ${event.status}.`));}
      }
      if(done)break;
    }
    if(!output)body.textContent='No answer was generated. Try again or choose a different model.';
    if(!gotDone)notify('The connection ended before the response completed.',true);
  }catch(error){
    if(error.name!=='AbortError')notify(error.message,true);
    body.textContent=output||'Generation stopped without an answer.';
    answer.append(el('div','result-note','Incomplete response.'));
  }finally{
    generating=false;activeController=null;$('#stop-chat').hidden=true;await refresh();$('#prompt').focus();
  }
}));
$('#stop-chat').addEventListener('click',handle(async()=>{
  await api(`/api/conversations/${conversation}/stop`,{method:'POST'});
  $('#composer-hint').textContent='Stopping at the next output chunk…';
  // Closing the stream also disconnects the local inference request when output resumes.
  if(activeController)activeController.abort();
  $('#composer-hint').textContent='Conversation context stays on this Mac';
}));
$('#delete-chat').addEventListener('click',handle(async()=>{
  if(!conversation||generating)return;
  if(!confirm('Delete this conversation from your Mac?'))return;
  await api(`/api/conversations/${conversation}`,{method:'DELETE'});conversation=null;$('#messages').replaceChildren();$('#welcome').hidden=false;await refresh();
}));
$('#export-chat').addEventListener('click',handle(async()=>{
  if(!conversation){notify('Start a conversation before exporting.');return;}
  const data=await api(`/api/conversations/${conversation}`);
  const text=data.messages.map(m=>`${m.role==='user'?'You':'VV-GPT'}${m.model?' ('+m.model+')':''}\n\n${m.content}\n${m.status!=='complete'?'['+m.status+']':''}`).join('\n\n---\n\n');
  const url=URL.createObjectURL(new Blob([text],{type:'text/plain'}));const a=el('a');a.href=url;a.download='vv-gpt-conversation.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}));
function renderDocuments(){
  const target=$('#documents');target.replaceChildren();
  if(!state.documents.length)target.append(el('p','empty','No documents yet. Add a file, then ask a question about it in Assistant.'));
  state.documents.forEach(doc=>{const row=el('div','item'),info=el('div');info.append(el('strong','',doc.name),el('small','',`${doc.chars.toLocaleString()} characters · Available for local search`));const remove=el('button','quiet','Remove');remove.addEventListener('click',handle(async()=>{if(!confirm(`Remove ${doc.name} from knowledge? Existing conversation excerpts will remain.`))return;await api('/api/documents/'+doc.id,{method:'DELETE'});await refresh();}));row.append(info,remove);target.append(row);});
}
async function upload(input,path){
  for(const file of input.files){const data=new FormData();data.append('file',file);await api(path,{method:'POST',body:data});}
  input.value='';await refresh();notify('Import complete.');
}
$('#knowledge-file').addEventListener('change',handle(e=>upload(e.target,'/api/documents')));
$('#dataset-file').addEventListener('change',handle(e=>upload(e.target,'/api/datasets')));
function datasetDetail(){const data=state.datasets.find(d=>d.id===$('#dataset').value);$('#dataset-detail').textContent=data?`${data.tokens.toLocaleString()} tokens. Last 10% reserved for validation. Repeated text can still make this score optimistic.`:'';}
$('#dataset').addEventListener('change',datasetDetail);
$('#memory-form').addEventListener('submit',handle(async e=>{e.preventDefault();await api('/api/memory',{method:'PUT',body:{memory:$('#memory').value}});notify('Preferences saved for future messages.');}));
$('#training-form').addEventListener('submit',handle(async e=>{
  e.preventDefault();const button=e.target.querySelector('button[type=submit]');button.disabled=true;
  try{const data=Object.fromEntries(new FormData(e.target));const result=await api('/api/runs',{method:'POST',body:data});selectedRun=result.id;await refresh();await loadRun(selectedRun);notify('Training started in its own worker. You can leave this page and return.');}finally{button.disabled=false;}
}));
function renderRuns(){const target=$('#runs');target.replaceChildren();
  if(!state.runs.length)target.append(el('p','empty','No experiments yet. Import text and start a small run.'));
  state.runs.forEach(run=>{const button=el('button',run.id===selectedRun?'selected':'');button.append(el('span','',run.name),el('small','',run.status.replaceAll('_',' ')));button.addEventListener('click',handle(()=>loadRun(run.id)));target.append(button);});
}
async function loadRun(rid){
  const data=await api(`/api/runs/${rid}`);selectedRun=rid;$('#run-detail').hidden=false;$('#run-name').textContent=data.run.name;
  $('#run-status').textContent=data.run.status.replaceAll('_',' ');$('#run-step').textContent=`${data.run.step} / ${JSON.parse(data.run.config).steps}`;
  $('#run-best').textContent=data.run.best?.toFixed(3)??'—';$('#run-error').textContent=data.run.error||'';
  document.querySelectorAll('[data-run-action]').forEach(b=>{const s=data.run.status;b.disabled=b.dataset.runAction==='pause'?!['running','queued'].includes(s):b.dataset.runAction==='stop'?!['running','paused','queued'].includes(s):!['paused','failed','stopped','interrupted'].includes(s);});
  lastMetrics=data.metrics;drawChart();renderRuns();
}
function drawChart(){
  const canvas=$('#loss-chart'),ctx=canvas.getContext('2d'),ratio=window.devicePixelRatio||1;
  const width=canvas.clientWidth;if(!width)return;const height=240;canvas.width=width*ratio;canvas.height=height*ratio;ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
  ctx.font='10px -apple-system, sans-serif';ctx.fillStyle='#8490a3';
  if(!lastMetrics.length){ctx.fillText('Loss curves will appear after the first evaluation.',20,110);return;}
  const left=35,right=12,top=16,bottom=25;
  const values=lastMetrics.flatMap(m=>[m.train,m.val]).filter(Number.isFinite);const low=Math.max(0,Math.min(...values)-.2),high=Math.max(...values)+.2,steps=Math.max(1,lastMetrics.at(-1).step);
  const x=s=>left+s/steps*(width-left-right),y=v=>top+(high-v)/(high-low)*(height-top-bottom);
  for(let i=0;i<=4;i++){const value=low+(high-low)*i/4,pos=y(value);ctx.strokeStyle='#e2e7f0';ctx.beginPath();ctx.moveTo(left,pos);ctx.lineTo(width-right,pos);ctx.stroke();ctx.fillText(value.toFixed(1),2,pos+3);}
  ['train','val'].forEach((key,i)=>{ctx.strokeStyle=i?'#ae839f':'#5874b4';ctx.lineWidth=2;ctx.beginPath();lastMetrics.forEach((m,index)=>{if(index===0)ctx.moveTo(x(m.step),y(m[key]));else ctx.lineTo(x(m.step),y(m[key]));});ctx.stroke();lastMetrics.forEach(m=>{ctx.fillStyle=i?'#ae839f':'#5874b4';ctx.beginPath();ctx.arc(x(m.step),y(m[key]),2,0,Math.PI*2);ctx.fill();});});
  ctx.fillStyle='#8490a3';ctx.fillText('0',left,height-6);ctx.fillText(`Step ${steps}`,width-70,height-6);
  canvas.setAttribute('aria-label',`Loss at step ${steps}: training ${lastMetrics.at(-1).train.toFixed(3)}, validation ${lastMetrics.at(-1).val.toFixed(3)}.`);
}
window.addEventListener('resize',drawChart);
document.querySelectorAll('[data-run-action]').forEach(button=>button.addEventListener('click',handle(async()=>{await api(`/api/runs/${selectedRun}/${button.dataset.runAction}`,{method:'POST'});await loadRun(selectedRun);})));
function renderModels(){
  const target=$('#model-library');target.replaceChildren();
  if(!state.models.length)target.append(el('p','empty','Your trained models will appear here after their first checkpoint.'));
  state.models.forEach(model=>{const node=el('article','model-card');node.append(el('span','origin',model.origin+(model.readonly?' · Preserved original':'')),el('h2','',model.name));const dl=el('dl');[['Parameters',model.params?`${(model.params/1e6).toFixed(1)}M`:'Custom'],['Best loss',model.best?.toFixed(3)??'—'],['Context',`${model.context} tokens`],['Checkpoint','Best']].forEach(([k,v])=>dl.append(el('dt','',k),el('dd','',v)));node.append(dl,el('div','caption',model.readonly?'Original files are read-only':'Saved independently in VV-GPT4'));target.append(node);});
  options($('#sample-model'),state.models.map(m=>({value:m.id,label:`${m.name} · ${m.origin}`})),'Train a model to begin');
}
$('#sample-form').addEventListener('submit',handle(async e=>{e.preventDefault();const button=$('#sample-submit');button.disabled=true;$('#sample-output').textContent='Loading the best checkpoint and generating…';try{const result=await api('/api/sample',{method:'POST',body:{model:$('#sample-model').value,prompt:$('#sample-prompt').value,tokens:Number($('#sample-tokens').value)}});$('#sample-output').textContent=`${result.text}\n\n— ${result.tokens} new tokens · ${result.seconds}s · best checkpoint at step ${result.step}`;}catch(error){$('#sample-output').textContent='Generation failed. See the message above.';throw error;}finally{button.disabled=false;}}));
setInterval(async()=>{if(currentPage!=='lab'||!state)return;try{state.runs=(await api('/api/state')).runs;renderRuns();if(selectedRun)await loadRun(selectedRun);}catch(error){notify(error.message,true);}},3000);
refresh().catch(error=>notify(error.message,true));
