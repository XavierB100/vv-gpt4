"""Continue the user's own chat checkpoint locally; never publish chat text/samples."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_coherence import run_condition,STORE
from vvgpt.lab import add_partitioned_dataset,ModelCache,model_path,dataset_splits
from vvgpt.tokenizer import Codec
from vvgpt.evaluation import score_text
from vvgpt.data import fingerprint

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runtime/research/chat'
FOLDER.mkdir(exist_ok=True)
path=FOLDER/'manifest.json'
if path.exists():report=json.loads(path.read_text())
else:
    original=Path('/Users/xavier/AI/hermes-workspace/vv-gpt3');sys.path.insert(0,str(original))
    from src.training.data_loader import DataProcessor
    processor=DataProcessor();text,metadata=processor.load_and_process_data(original/'uploads/chat_20260513_004648.txt','auto')
    codec=Codec();ids=codec.encode(text)
    assert len(text)==4044463 and len(ids)==1424331
    first=int(.9*len(ids));second=first+(len(ids)-first)//2
    splits={'train':codec.decode(ids[:first]),'validation':codec.decode(ids[first:second]),'test':codec.decode(ids[second:])}
    did=add_partitioned_dataset(STORE,'Chat / original training partitions',splits,{'source':'Original private chat corpus','original_cuts':[first,second],
        'limitation':'Original best selected using the complete historical validation tail; current test is not selection-naive for the parent.'})
    report={'dataset':did,'runs':[],'privacy':'Raw data and any outputs stay in local runtime. Public evidence contains aggregate scores only.'}
    path.write_text(json.dumps(report,indent=2))
# The generic runner persists through a module global; bind this study's own manifest.
import scripts.research_coherence as driver
driver.MANIFEST=path
for kind,alpha in [('chat_ce',0),('chat_ul',.1)]:
    run_condition(report,kind,{'name':'Chat / '+('continued learning' if not alpha else 'repetition objective'),'dataset':report['dataset'],
        'initial_model':'legacy:chat','steps':800,'batch':4,'context':512,'lr':3e-5,'dropout':.1,
        'eval_every':100,'patience':4,'seed':42,'unlikelihood':alpha})
cache=ModelCache();test=dataset_splits(STORE,report['dataset'])['test']
results={}
for name,model_id in [('original','legacy:chat')]+[(r['kind'],'run:'+r['run']) for r in report['runs']]:
    model,codec,device,step=cache.get(model_path(STORE,model_id))
    score=score_text(model,codec,test,device,max_chars=len(test));score['sha256']=fingerprint(test)
    results[name]={'model':model_id,'checkpoint_step':step,'historical_test':score}
    print(json.dumps({'condition':name,'test_bits_per_byte':score['bits_per_byte'],'step':step}),flush=True)
public={'results':results,'runs':report['runs'],'privacy':report['privacy'],
        'limitation':'Original checkpoint was previously selected on the full historical validation tail. No fresh external chat corpus or independent human coherence ratings were used.'}
(ROOT/'docs/chat-research-results.json').write_text(json.dumps(public,indent=2)+'\n')
print('Chat study complete. No private text exported.',flush=True)
