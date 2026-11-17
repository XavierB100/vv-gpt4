"""Post-selection evaluation. Never changes training choices or model weights."""
import argparse
import json
import re
import sys
import statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vvgpt.store import Store
from vvgpt.lab import ModelCache,model_path,generate_sample,dataset_splits
from vvgpt.data import fingerprint
from vvgpt.evaluation import score_text
from vvgpt.model import repeat_fraction

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'docs/coherence-results.json'
manifest=json.loads((ROOT/'runtime/research/coherence/manifest.json').read_text())
store=Store(ROOT/'runtime')
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['continued','architecture']);args=parser.parse_args()
report=json.loads(REPORT.read_text()) if REPORT.exists() else {'protocol':'docs/RESEARCH_PROTOCOL.md','conditions':{},'limitations':manifest['provenance']['limitation']}
report['runs']=manifest['runs']
REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
cache=ModelCache()
conditions=[('original',manifest['source_model'],manifest['continued_dataset'])] if args.stage=='continued' else []
for run in manifest['runs']:
    matches=run['kind'].startswith('continued') if args.stage=='continued' else run['kind'].startswith(('classic_','revised_'))
    if matches:
        status=store.rows('SELECT status FROM runs WHERE id=?',(run['run'],))[0]['status']
        if status not in ('completed','early_stopped'):raise RuntimeError('Wait for the full planned run before evaluation: '+run['kind'])
        conditions.append((run['kind'],'run:'+run['run'],run['config']['dataset']))
for name,model_id,did in conditions:
    if name in report['conditions']:continue
    model,codec,device,step=cache.get(model_path(store,model_id))
    texts={'historical_test':dataset_splits(store,did)['test']} if args.stage=='continued' else {'scratch_test':dataset_splits(store,did)['test']}
    if args.stage=='continued':texts['external_hamlet']=(ROOT/'runtime/research/coherence/hamlet-test.txt').read_text()
    scores={}
    for key,text in texts.items():
        scores[key]=score_text(model,codec,text,device,max_chars=len(text))
        scores[key]['sha256']=fingerprint(text)
    samples=[]
    for prompt in manifest['prompts']:
        for seed in manifest['generation_seeds']:
            result=generate_sample(store,model_id,prompt,160,{'seed':seed,'temperature':.8,'repetition_penalty':1.,'no_repeat_ngram':0},cache)
            words=re.findall(r"[A-Za-z']+",result['text'].lower())
            result.update(prompt=prompt,word_repeat_fraction=repeat_fraction(words,4),paragraph_markers=result['text'].count('<|paragraph|>'))
            samples.append(result)
    result={'model':model_id,'checkpoint_step':step,'scores':scores,'samples':samples,'summary':{
        'mean_repeated_token_4grams':statistics.mean(s['repeat_fraction'] for s in samples),
        'mean_repeated_word_4grams':statistics.mean(s['word_repeat_fraction'] for s in samples),
        'paragraph_markers':sum(s['paragraph_markers'] for s in samples),
        'samples':len(samples)}}
    report['conditions'][name]=result
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'condition':name,'scores':scores,'summary':result['summary']}),flush=True)
