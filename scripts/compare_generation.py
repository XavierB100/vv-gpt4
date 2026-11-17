"""Public prompts only; read-only sampling from Xavier's existing best checkpoint."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vvgpt.lab import ModelCache,generate_sample
from vvgpt.store import Store

ROOT=Path(__file__).resolve().parents[1]
store=Store(ROOT/'runtime');cache=ModelCache()
prompts=['ROMEO:\nThe night is quiet, and','KING RICHARD III:\nMy lord,','JULIET:\nIf love be']
results=[]
for prompt,seed in zip(prompts,[42,123,7]):
    item={'prompt':prompt,'seed':seed}
    for mode,settings in [('baseline',{'repetition_penalty':1.,'no_repeat_ngram':0}),('controlled',{'repetition_penalty':1.12,'no_repeat_ngram':0}),('guarded',{'repetition_penalty':1.12,'no_repeat_ngram':4})]:
        item[mode]=generate_sample(store,'legacy:shakespeare-small-v1',prompt,200,dict(settings,seed=seed,temperature=.8),cache)
    results.append(item)
    print(json.dumps({'prompt':prompt,'baseline_repetition':item['baseline']['repeat_fraction'],'controlled_repetition':item['controlled']['repeat_fraction']}),flush=True)
report={'model':'legacy:shakespeare-small-v1','checkpoint':'best','tokens_requested':200,'samples':results,
        'limitations':'Three public prompts, one seed each. Phrase repetition is a token metric, not a coherence or factuality measure. Controls change sampling, not learned weights.'}
(ROOT/'docs/generation-comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
