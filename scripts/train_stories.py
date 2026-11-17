"""Train the declared clean-language candidate, then retain every planned sample."""
import json
import sys
import re
import statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.research_coherence as driver
from vvgpt.lab import dataset_splits,dataset_codec,generate_sample,ModelCache,model_path
from vvgpt.data import fingerprint
from vvgpt.model import repeat_fraction
from vvgpt.evaluation import score_text

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'runtime/research/stories/manifest.json';driver.MANIFEST=path
report=json.loads(path.read_text());store=driver.STORE
codec,_=dataset_codec(store,report['dataset'],'compact')
assert codec.encode('<|endoftext|>')==[codec.eos], 'Story boundaries must encode as EOS.'
driver.run_condition(report,'stories_revised',{'name':'Simple stories / learned language','dataset':report['dataset'],
 'size':'mini','architecture':'revised','tokenizer':'compact','steps':5000,'batch':16,'context':256,'seed':42,
 'lr':5e-4,'dropout':.1,'eval_every':250,'patience':12})
cache=ModelCache();model_id='run:'+report['runs'][0]['run']
model,codec,device,step=cache.get(model_path(store,model_id))
text=dataset_splits(store,report['dataset'])['test'][:100000]
score=score_text(model,codec,text,device,max_chars=len(text));score['sha256']=fingerprint(text)
results={}
for name,mid in [('own_original','legacy:shakespeare-small-v1'),('stories_revised',model_id)]:
 samples=[]
 for prompt in report['prompts']:
  for seed in report['seeds']:
   result=generate_sample(store,mid,prompt,160,{'seed':seed,'temperature':.8,'repetition_penalty':1.,'no_repeat_ngram':0},cache)
   result.update(prompt=prompt,word_repeat_fraction=repeat_fraction(re.findall(r"[A-Za-z']+",result['text'].lower()),4))
   samples.append(result)
 results[name]={'model':mid,'samples':samples,'mean_word_repeat_fraction':statistics.mean(s['word_repeat_fraction'] for s in samples)}
output={'provenance':report['provenance'],'runs':report['runs'],'test':score,'generation':results,
 'limitations':'Exploratory clean-language data experiment; one run/seed, restricted synthetic story domain. Comparing task capability with a Shakespeare-trained parent is not a controlled architecture comparison. No general assistant or factuality claim. EOS separators count toward the loss denominator.'}
(ROOT/'docs/stories-results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'test':score,'mean_word_repetition':{k:v['mean_word_repeat_fraction'] for k,v in results.items()}}),flush=True)
