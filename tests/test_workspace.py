import io
import json
import pytest
from vvgpt.providers import LocalProvider
from vvgpt.knowledge import retrieve


def test_loopback_and_csrf(client, headers):
    assert client.get('/health',base_url='http://evil.example').status_code==403
    assert client.post('/api/conversations').status_code==403
    assert client.post('/api/conversations',headers={**headers,'Origin':'http://evil.example'}).status_code==403
    assert client.post('/api/conversations',headers=headers).status_code==200
    assert "frame-ancestors 'none'" in client.get('/').headers['Content-Security-Policy']


def test_multiturn_memory_and_citations_persist(app, client, headers):
    client.put('/api/memory',headers=headers,json={'memory':'Use British spelling.'})
    doc=client.post('/api/documents',headers=headers,data={'file':(io.BytesIO(b'Project Alder launches on 23 February. Its budget is 450 pounds.'),'alder.txt')}).json
    cid=client.post('/api/conversations',headers=headers).json['id']
    first=client.post(f'/api/conversations/{cid}/chat',headers=headers,json={'message':'When does Alder launch?','model':'test:local'})
    assert first.status_code==200
    assert b'A local answer' in first.data
    second=client.post(f'/api/conversations/{cid}/chat',headers=headers,json={'message':'What is its budget?','model':'test:local'})
    assert second.status_code==200
    _=second.data
    messages=app.extensions['provider'].calls[-1]
    assert 'British spelling' in messages[0]['content']
    assert messages[1]['content']=='When does Alder launch?'
    assert messages[2]['content']=='A local answer [1].'
    assert '450 pounds' in messages[-1]['content']
    rows=client.get(f'/api/conversations/{cid}').json['messages']
    assert len(rows)==4 and rows[-1]['status']=='complete'
    assert json.loads(rows[-1]['sources'])[0]['name']=='alder.txt'
    assert client.delete('/api/documents/'+doc['id'],headers=headers).status_code==200
    assert retrieve(app.extensions['store'],'Alder')==[]
    assert client.delete(f'/api/conversations/{cid}',headers=headers).status_code==200
    assert app.extensions['store'].rows('SELECT * FROM messages')==[]


def test_no_knowledge_when_disabled(app,client,headers):
    client.post('/api/documents',headers=headers,data={'file':(io.BytesIO(b'Private reference about Alder.'),'notes.txt')})
    cid=client.post('/api/conversations',headers=headers).json['id']
    response=client.post(f'/api/conversations/{cid}/chat',headers=headers,json={'message':'Alder','model':'test:local','knowledge':False})
    assert b'"sources": []' in response.data
    assert 'Private reference' not in str(app.extensions['provider'].calls)


def test_errors_release_generation_slot(app,client,headers):
    def broken(*args):
        yield {'type':'token','text':'Partial'}
        raise OSError('Engine disappeared')
    app.extensions['provider'].stream=broken
    cid=client.post('/api/conversations',headers=headers).json['id']
    for _ in range(2):
        response=client.post(f'/api/conversations/{cid}/chat',headers=headers,json={'message':'Hi','model':'test:local'})
        assert response.status_code==200
        assert b'Engine disappeared' in response.data
    assert client.get(f'/api/conversations/{cid}').json['messages'][-1]['status']=='failed'


@pytest.mark.parametrize('data',[{'name':'qwen:cloud','size':5000000},{'name':'remote','size':0},
 {'name':'hidden-cloud','size':2000000},{'name':'remote','size':2000000,'remote_host':'cloud.example'}])
def test_cloud_models_excluded(data):
    assert not LocalProvider.local_candidate(data)


def test_upload_validation(client,headers):
    assert client.post('/api/documents',headers=headers,data={'file':(io.BytesIO(b'%PDF'),'a.pdf')}).status_code==400
    assert client.post('/api/documents',headers=headers,data={'file':(io.BytesIO(b'\xff'),'a.txt')}).status_code==400
    assert client.post('/api/documents',headers=headers,data={'file':(io.BytesIO(b'  '),'a.txt')}).status_code==400
    assert client.put('/api/memory',headers=headers,json={'memory':'a'*4001}).status_code==400


def test_restart_preserves_data_and_changes_session_token(app):
    from app import create_app
    store=app.extensions['store']
    store.execute('INSERT INTO settings VALUES(?,?)',('memory','Stay curious'))
    restarted=create_app(store.root,app.extensions['provider'])
    assert restarted.extensions['store'].setting('memory')=='Stay curious'
    assert restarted.extensions['csrf']!=app.extensions['csrf']
