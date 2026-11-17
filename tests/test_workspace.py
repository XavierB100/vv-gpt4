import io
import pytest
from vvgpt import lab


def test_loopback_and_csrf(client, headers):
    assert client.get('/health',base_url='http://evil.example').status_code==403
    assert client.post('/api/datasets').status_code==403
    assert client.post('/api/datasets',headers={**headers,'Origin':'http://evil.example'}).status_code==403
    assert "frame-ancestors 'none'" in client.get('/').headers['Content-Security-Policy']
    assert client.get('/health').json['mode']=='custom-models'
    state=client.get('/api/state').json
    assert 'provider' not in state and 'models' in state
    assert client.get('/api/conversations').status_code==404


def test_import_preserves_raw_and_restart_preserves_data(app,client,headers):
    from app import create_app
    raw=('Alice: Good morning! 🌅 café\r\nBob: Hello, friend.\r\n'*100).encode()
    response=client.post('/api/datasets',headers=headers,data={'file':(io.BytesIO(raw),'chat.txt')})
    assert response.status_code==200
    did=response.json['id'];root=app.extensions['store'].root
    assert (root/'datasets'/f'{did}.raw.txt').read_bytes()==raw
    restarted=create_app(root)
    assert restarted.test_client().get('/api/state').json['datasets'][0]['id']==did
    assert restarted.extensions['csrf']!=app.extensions['csrf']


@pytest.mark.parametrize('content,name',[(b'%PDF','a.pdf'),(b'\xff','a.txt'),(b'  ','a.txt')])
def test_upload_validation(client,headers,content,name):
    assert client.post('/api/datasets',headers=headers,data={'file':(io.BytesIO(content),name)}).status_code==400


@pytest.mark.parametrize('key,value',[('tokens',2.5),('tokens',True),('seed',-1),('seed',4.5),('no_repeat_ngram',4.2),('temperature',float('nan')),('repetition_penalty',2),('prompt','')])
def test_bad_generation_settings_rejected_before_loading(client,headers,monkeypatch,key,value):
    monkeypatch.setattr(lab.ModelCache,'get',lambda *a:pytest.fail('Model should not load'))
    settings={'model':'absent','prompt':'Hello','tokens':20,key:value}
    assert client.post('/api/sample',headers=headers,json=settings).status_code==400


def test_generation_failure_releases_slot_and_success_persists(app,client,headers,monkeypatch):
    calls=[]
    def generate(store,model,prompt,length,settings,cache):
        calls.append(settings)
        if len(calls)==1:raise ValueError('Try again')
        return {'text':'Generated text','seed':settings['seed']}
    monkeypatch.setattr(lab,'generate_sample',generate)
    payload={'model':'own-model','prompt':'Hello','seed':12,'compare':True}
    assert client.post('/api/sample',headers=headers,json=payload).status_code==400
    response=client.post('/api/sample',headers=headers,json=payload)
    assert response.status_code==200
    saved=client.get('/api/samples/'+response.json['id']).json
    assert saved['prompt']=='Hello' and saved['settings']==payload
    assert saved['result']['baseline']['seed']==saved['result']['enhanced']['seed']==12
    assert calls[-1]['no_repeat_ngram']==0 and calls[-1]['repetition_penalty']==1
