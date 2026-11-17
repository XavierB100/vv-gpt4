"""One declared exploratory continuation; retain first-budget results separately."""
import json
import sys
import re
import statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.research_coherence as driver
from vvgpt.lab import dataset_splits,generate_sample,ModelCache,model_path
from vvgpt.data import fingerprint
from vvgpt.model import repeat_fraction
from vvgpt.evaluation import score_text
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'runtime/research/stories/continued-manifest.json';driver.MANIFEST=path
original=json.loads((ROOT/'runtime/research/stories/manifest.json').read_text())
source='run:'+original['runs'][0]['run']
report=json.loads(path.read_text()) if path.exists() else {'dataset':original['dataset'],'source_model':source,'runs':[]}
driver.run_condition(report,'stories_continued',{'name':'Simple stories / further learning','dataset':report['dataset'],
 'initial_model':source,'steps':5000,'batch':16,'seed':42,'lr':3e-4,'dropout':0,'eval_every':250,'patience':12})
cache=ModelCache();candidate='run:'+report['runs'][0]['run']
text=dataset_splits(driver.STORE,report['dataset'])['test'][100000:200000]
results={}
for name,mid in [('first_budget',source),('continued',candidate)]:
 model,codec,device,step=cache.get(model_path(driver.STORE,mid))
 score=score_text(model,codec,text,device,max_chars=len(text));score.update(sha256=fingerprint(text),slice_start=100000,slice_end=200000)
 samples=[]
 if name=='first_budget':
  samples=json.loads((ROOT/'docs/stories-results.json').read_text())['generation']['stories_revised']['samples']
 else:
  for prompt in original['prompts']:
   for seed in original['seeds']:
    result=generate_sample(driver.STORE,mid,prompt,160,{'seed':seed,'temperature':.8,'repetition_penalty':1.,'no_repeat_ngram':0},cache)
    result.update(prompt=prompt,word_repeat_fraction=repeat_fraction(re.findall(r"[A-Za-z']+",result['text'].lower()),4))
    samples.append(result)
 results[name]={'model':mid,'checkpoint_step':step,'test':score,'samples':samples,'mean_word_repeat_fraction':statistics.mean(s['word_repeat_fraction'] for s in samples)}
output={'runs':report['runs'],'conditions':results,'limitations':'Exploratory second budget and dropout change selected after first outputs. Test slice was not scored in the first round, but is from the same reserved partition. Prompts informed the continuation decision; improvement on them is not confirmation on unseen tasks. Fresh optimizer. No claim of general chat or factual reliability.'}
(ROOT/'docs/stories-continuation-results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:{'test':v['test'],'word_repetition':v['mean_word_repeat_fraction']} for k,v in results.items()}),flush=True)
