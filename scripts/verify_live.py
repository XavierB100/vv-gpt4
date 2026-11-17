"""Opt-in real local-engine and MPS smoke checks. No model downloads or cloud calls."""
import io
import json
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from app import create_app
from vvgpt.lab import generate_sample, load_checkpoint

ROOT=Path(__file__).resolve().parents[1]
report={'date':'2026-11-17','scope':'Short local smoke checks; not a general capability benchmark'}
with tempfile.TemporaryDirectory(prefix='vv-gpt4-live-') as temporary:
    app=create_app(temporary)
    client=app.test_client();headers={'X-VV-Token':app.extensions['csrf']}
    store=app.extensions['store']
    client.post('/api/documents',headers=headers,data={'file':(io.BytesIO(b'Project Alder launches on 23 February 2027. Its approved budget is exactly 450 pounds.'),'alder-test.txt')})
    cid=client.post('/api/conversations',headers=headers).json['id']
    report['assistant']=[]
    for question, expected in [('When does Project Alder launch? Cite the source in one short sentence.','23 February 2027'),('What is its approved budget? Cite the source in one short sentence.','450')]:
        start=time.monotonic()
        response=client.post(f'/api/conversations/{cid}/chat',headers=headers,json={'model':'qwen3:8b','message':question})
        assert response.status_code==200
        events=[json.loads(line) for line in response.data.decode().splitlines()]
        text=''.join(e.get('text','') for e in events)
        assert expected in text and '[1]' in text, text
        report['assistant'].append({'question':question,'answer':text,'seconds':round(time.monotonic()-start,3),
                                    'metrics':next(e for e in events if e['type']=='metrics')})
    # Small integration run with the real Nano architecture and device selection.
    text=('A river runs through the valley. The morning sun lights the water. '
          'At night the village is quiet, and stars shine above the hills.\n')*400
    uploaded=client.post('/api/datasets',headers=headers,data={'file':(io.BytesIO(text.encode()),'synthetic-smoke.txt')})
    assert uploaded.status_code==200,uploaded.json
    response=client.post('/api/runs',headers=headers,json={'name':'Integration smoke only','dataset':uploaded.json['id'],
        'size':'nano','steps':6,'context':32,'batch':2,'eval_every':2,'patience':5})
    assert response.status_code==200,response.json
    rid=response.json['id'];deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        result=client.get('/api/runs/'+rid).json
        if result['run']['status'] not in ('queued','running'):break
        time.sleep(.3)
    assert result['run']['status']=='completed',result
    checkpoint=load_checkpoint(store.root/'runs'/rid/'best.pt')
    assert checkpoint['val']==min(m['val'] for m in result['metrics'])
    report['training']={'architecture':'nano','steps':6,'metrics':result['metrics'],
                         'best_checkpoint_step':checkpoint['step'],'best_checkpoint_val':checkpoint['val'],
                         'caveat':'Synthetic repeated text verifies the pipeline only, not model quality.'}
    report['legacy_sample']=generate_sample(store,'legacy:shakespeare-small-v1','ROMEO:',24)
    import torch
    report['device']='mps' if torch.backends.mps.is_available() else 'cpu'
(ROOT/'docs'/'live-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
