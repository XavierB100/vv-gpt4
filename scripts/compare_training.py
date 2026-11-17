"""Two prespecified scratch-training pilots; same architecture, corpus, steps and seed."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vvgpt.store import Store
from vvgpt.lab import start_run,run_summary

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True,help='New result file; existing evidence is never overwritten.')
parser.add_argument('--equal-budget',action='store_true',help='Include the exploratory 800-step compact condition.')
args=parser.parse_args()
if args.output.exists():parser.error('Output already exists; choose a new file.')
store=Store(root/'runtime')
dataset=store.rows('SELECT id FROM datasets WHERE name=?',('Shakespeare · original corpus',))[0]['id']
results=[]
for kind,steps in [('gpt2',400),('compact',400)]+([('compact',800)] if args.equal_budget else []):
    config={'dataset':dataset,'tokenizer':kind,'size':'nano','steps':steps,'batch':8,'context':128,
            'eval_every':50,'patience':20,'seed':42,'name':f'Shakespeare pilot · {"original vocabulary" if kind=="gpt2" else "compact vocabulary"}'}
    rid=start_run(store,config)
    print(json.dumps({'started':kind,'id':rid}),flush=True)
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        run=store.rows('SELECT * FROM runs WHERE id=?',(rid,))[0]
        if run['status'] not in ('queued','running','paused','evaluating'):break
        time.sleep(1)
    if run['status'] not in ('completed','early_stopped'):
        raise RuntimeError(f'Pilot did not complete: {run["status"]}, {run["error"]}')
    summary=run_summary(store,rid)
    result={'kind':kind,'run':rid,'config':config,'summary':summary,
            'metrics':store.rows('SELECT step,train,val,lr,elapsed FROM metrics WHERE run=? ORDER BY step',(rid,))}
    results.append(result)
    print(json.dumps({'completed':kind,'summary':summary}),flush=True)
    args.output.write_text(json.dumps({'runs':results,'caveat':'One seed, 400 steps, one corpus. Different token counts per raw byte; token losses are not directly comparable. Test bits/byte uses the same raw text and each model has a 128-token context. Not a general coherence benchmark.'},indent=2))
