"""Prespecified learned-weight experiments. Resume this driver safely after interruption."""
import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vvgpt.store import Store
from vvgpt.lab import add_partitioned_dataset,start_run,run_summary,ModelCache,model_path,dataset_codec,reconcile,control_run
from vvgpt.data import fingerprint
from vvgpt.tokenizer import Codec
from vvgpt.evaluation import score_text

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runtime/research/coherence'
MANIFEST=FOLDER/'manifest.json'
STORE=Store(ROOT/'runtime')


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def prepare():
    if MANIFEST.exists():return json.loads(MANIFEST.read_text())
    # Reconstruct the exact original data treatment; never import or run the old trainer.
    original=Path('/Users/xavier/AI/hermes-workspace/vv-gpt3')
    sys.path.insert(0,str(original))
    from src.training.data_loader import DataProcessor
    processor=DataProcessor();text,_=processor.load_and_process_data(original/'data/input.txt','plain_text')
    codec=Codec();ids=codec.encode(text)
    assert len(text)==1216479 and len(ids)==381335
    first=int(.9*len(ids));second=first+(len(ids)-first)//2
    strip=lambda seq:re.sub(r'\n{3,}','\n\n',codec.decode(seq).replace('<|paragraph|>\n','')).strip()+'\n'
    splits={'train':strip(ids[:first]),'validation':strip(ids[first:second]),'test':strip(ids[second:])}
    provenance={'source':'Original public Shakespeare corpus','original_tokens':len(ids),'original_cuts':[first,second],
                'limitation':'Original best checkpoint was previously selected using the whole historical validation tail. Current partitions are disjoint but the test is not selection-naive for that original checkpoint.'}
    did=add_partitioned_dataset(STORE,'Shakespeare — continued learning',splits,provenance)
    class PlayParser(HTMLParser):
        def __init__(self):super().__init__();self.capture=None;self.lines=[]
        def handle_starttag(self,tag,attrs):
            if tag.lower()=='a':
                name=dict(attrs).get('name','')
                self.capture='line' if re.fullmatch(r'\d+\.\d+\.\d+',name) else ('speaker' if name.startswith('speech') else None)
        def handle_data(self,data):
            if self.capture and data.strip():self.lines.append(('\n'+data.strip()+':') if self.capture=='speaker' else data.strip())
        def handle_endtag(self,tag):
            if tag.lower()=='a':self.capture=None
    parser=PlayParser();parser.feed((FOLDER/'hamlet-source.html').read_text())
    hamlet='\n'.join(parser.lines).strip()[:30000]
    assert len(hamlet)==30000 and 'BERNARDO:' in hamlet
    (FOLDER/'hamlet-test.txt').write_text(hamlet)
    prompts=['ROMEO:\nThe night is quiet, and','JULIET:\nWhen morning comes, I shall','KING RICHARD III:\nMy lord, the people','GLOUCESTER:\nI have a secret','Nurse:\nCome, my dear child,','QUEEN ELIZABETH:\nWhere is my son?']
    result={'continued_dataset':did,'source_model':'legacy:shakespeare-small-v1','baseline_path':str(model_path(STORE,'legacy:shakespeare-small-v1')),
            'hamlet':{'url':'https://shakespeare.mit.edu/hamlet/full.html','sha256':fingerprint(hamlet),'chars':len(hamlet)},
            'prompts':prompts,'generation_seeds':[42,123],'runs':[],'provenance':provenance}
    write(MANIFEST,result);return result


def run_condition(report,kind,config):
    existing=next((x for x in report['runs'] if x['kind']==kind),None)
    if existing:
        rid=existing['run']
        reconcile(STORE)
        if STORE.rows('SELECT status FROM runs WHERE id=?',(rid,))[0]['status']=='interrupted':
            control_run(STORE,rid,'resume')
    else:
        rid=start_run(STORE,config);existing={'kind':kind,'run':rid,'config':config};report['runs'].append(existing);write(MANIFEST,report)
    last=None
    while True:
        row=STORE.rows('SELECT * FROM runs WHERE id=?',(rid,))[0]
        if row['step']//100!=last:
            print(json.dumps({'condition':kind,'status':row['status'],'step':row['step'],'best':row['best']}),flush=True);last=row['step']//100
        if row['status'] not in ('queued','running','paused','evaluating'):break
        time.sleep(2)
    if row['status'] not in ('completed','early_stopped'):raise RuntimeError(f'{kind}: {row["status"]}: {row["error"]}')
    existing['summary']=run_summary(STORE,rid)
    existing['metrics']=STORE.rows('SELECT step,train,val,lr,elapsed FROM metrics WHERE run=? ORDER BY step',(rid,))
    write(MANIFEST,report)


def main():
    args=argparse.ArgumentParser();args.add_argument('stage',choices=['prepare','continued','architecture']);stage=args.parse_args().stage
    report=prepare()
    if stage=='continued':
        for objective,alpha in [('continued_ce',0),('continued_ul',.1)]:
            run_condition(report,objective,{'name':'Shakespeare / '+('continued learning' if alpha==0 else 'repetition objective'),
                'dataset':report['continued_dataset'],'initial_model':report['source_model'],'steps':1200,'batch':8,'context':256,
                'lr':1e-4,'dropout':.1,'eval_every':100,'patience':6,'seed':42,'unlikelihood':alpha})
    elif stage=='architecture':
        did=STORE.rows("SELECT id FROM datasets WHERE name='Shakespeare · original corpus'")[0]['id']
        for seed in [42,123]:
            for architecture in ['classic','revised']:
                run_condition(report,f'{architecture}_{seed}',{'name':f'{architecture.title()} / seed {seed}','dataset':did,
                    'architecture':architecture,'tokenizer':'compact','size':'micro','steps':1500,'batch':16,'context':128,
                    'lr':3e-4,'dropout':.1,'eval_every':100,'patience':100,'seed':seed})
    print('Research stage complete: '+stage,flush=True)

if __name__=='__main__':main()
