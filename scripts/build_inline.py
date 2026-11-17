"""Compact inline review of the actual interface and measured model results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'docs/coherence-results.json').read_text())['conditions']
s=json.loads((ROOT/'docs/stories-continuation-results.json').read_text())['conditions']
chat=json.loads((ROOT/'docs/chat-research-results.json').read_text())['results']
cases=[]
for seed in [42,123]:
 cases.append({'name':f'Transformer / seed {seed}','labels':['Classic','Revised'],
 'values':[c[f'{kind}_{seed}']['scores']['scratch_test']['bits_per_byte'] for kind in ['classic','revised']],
 'note':'Same corpus, vocabulary, context and 1,500-step budget. About 2.6M parameters each. Two seeds support a prediction gain; generated dialogue still loses coherence.'})
cases.append({'name':'Your Shakespeare weights / external Hamlet','labels':['Original','Continued'],
 'values':[c[k]['scores']['external_hamlet']['bits_per_byte'] for k in ['original','continued_ce']],
 'note':'Your original trained checkpoint plus 1,200 updates. This 30,000-byte Hamlet excerpt was not used for training or selection. Grammar and dialogue continuity still have errors.'})
cases.append({'name':'Story model / further training','labels':['First budget','Continued'],
 'values':[s[k]['test']['bits_per_byte'] for k in ['first_budget','continued']],
 'note':'Same reserved 100,000-character slice for both weights. Extra training and lower dropout were chosen after the first samples; this is an exploratory result, not an independent coherence benchmark.'})
cases.append({'name':'Your chat weights / historical text','labels':['Original','Continued / CE'],
 'values':[chat[k]['historical_test']['bits_per_byte'] for k in ['original','chat_ce']],
 'note':'Only a marginal change. The historical tail was previously used to select the parent. No meaningful improvement in chat coherence is established.'})
html='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;color:var(--foreground);font-family:var(--font-sans)}button,select{font:inherit}nav{display:flex;gap:8px;margin-bottom:16px}button{border:1px solid var(--border);border-radius:var(--radius);background:transparent;color:var(--muted-foreground);padding:9px 14px;font-size:12px;cursor:pointer}button[aria-selected=true]{background:var(--secondary);color:var(--foreground)}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid var(--ring);outline-offset:3px}[hidden]{display:none!important}figure{margin:0}img{display:block;width:100%;border:1px solid var(--border);border-radius:var(--radius)}figcaption{display:flex;justify-content:space-between;gap:20px;font-size:11px;line-height:1.8;color:var(--muted-foreground);margin-top:12px}a{color:var(--primary);text-decoration:none}.evidence{padding:16px;background:var(--card);border-radius:var(--radius)}label{font-size:11px;color:var(--muted-foreground)}select{display:block;margin-top:8px;width:100%;border:1px solid var(--border);border-radius:var(--radius);padding:11px;background:var(--background);color:var(--foreground);font-size:12px}svg{display:block;width:100%;height:205px;margin-top:10px}svg text{fill:var(--foreground);font:12px var(--font-sans)}.measure{font-size:12px;color:var(--foreground);margin:0}.note{font-size:12px;color:var(--muted-foreground);line-height:1.85;margin:14px 0 0}.legend{font-size:10px;color:var(--muted-foreground);margin:0 0 15px}.gain{font-variant-numeric:tabular-nums;font-weight:550;color:var(--foreground)}@media(max-width:480px){svg text{font-size:21px}}
</style><nav role="tablist" aria-label="Project review"><button role="tab" id="interface-tab" aria-controls="interface" aria-selected="true">Interface</button><button role="tab" id="results-tab" aria-controls="results" aria-selected="false">Model evidence</button></nav>
<section id="interface" role="tabpanel" aria-labelledby="interface-tab"><figure><img src="'''+'studio-desktop.png'+'''" alt="Actual VV-GPT studio: dark background, one large learning chart, settings in drawers"><figcaption><span>Actual local studio. Training, models and generation open in drawers.</span><a href="http://127.0.0.1:5054" target="_blank">Open studio ↗</a></figcaption></figure></section>
<section id="results" role="tabpanel" aria-labelledby="results-tab" hidden><div class="evidence"><label>Comparison<select id="case"></select></label><svg id="plot" viewBox="0 0 620 205" role="img"></svg><p class="legend">Held-out bits per byte ↓ · bars start at zero</p><p class="measure"><span id="gain" class="gain"></span> lower prediction loss on this text.</p><p id="note" class="note"></p></div><p class="note">Prediction, repetition and coherent reasoning are different measures. Full samples, limitations and sources are in <a href="http://127.0.0.1:5054/report" target="_blank">Research notes ↗</a>.</p></section>
<script>const cases='''+json.dumps(cases).replace('</','<\/')+''';
const $=s=>document.querySelector(s);for(const [i,c] of cases.entries()){const o=document.createElement('option');o.value=i;o.textContent=c.name;$('#case').append(o)}
function show(){const c=cases[$('#case').value],max=Math.max(...c.values)*1.15;$('#plot').setAttribute('aria-label',c.labels.map((n,i)=>n+': '+c.values[i].toFixed(3)+' bits per byte').join('; '));$('#plot').innerHTML=c.labels.map((n,i)=>{let y=38+i*85,w=c.values[i]/max*460;return `<text x="0" y="${y}">${n}</text><rect x="0" y="${y+12}" width="${w}" height="17" rx="3" fill="var(--chart-${i+1})"/><text x="${w+13}" y="${y+26}">${c.values[i].toFixed(3)}</text>`}).join('');$('#gain').textContent=((1-c.values[1]/c.values[0])*100).toFixed(2)+'%';$('#note').textContent=c.note}
$('#case').onchange=show;show();const tabs=[...document.querySelectorAll('[role=tab]')];for(const tab of tabs){tab.onclick=()=>{for(const t of tabs){const on=t===tab;t.setAttribute('aria-selected',on);t.tabIndex=on?0:-1;$('#'+t.getAttribute('aria-controls')).hidden=!on}};tab.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();const next=tabs[1-tabs.indexOf(tab)];next.click();next.focus()}}}
</script></html>'''
(ROOT/'docs/studio-inline.html').write_text(html)
print('Built inline review from screenshots and retained metrics.')
