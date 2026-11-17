import json
import time
from dataclasses import asdict
import pytest
import torch
from vvgpt.lab import add_dataset, training_config, load_checkpoint, library
from vvgpt.model import GPT, GPTConfig
from vvgpt.store import Store, uid
from vvgpt.worker import atomic_save, work


def tiny_config():
    return GPTConfig(vocab_size=64,block_size=16,n_layer=2,n_head=2,n_embd=16,dropout=0)


def test_cached_decoding_matches_full_context_and_is_causal():
    torch.manual_seed(3)
    model=GPT(tiny_config()).eval()
    x=torch.randint(0,64,(1,10))
    with torch.no_grad():
        full=model(x)[0]
        _,_,cache=model(x[:,:7],use_cache=True)
        cached=model(x[:,7:],use_cache=True,past_key_values=cache)[0]
        assert torch.allclose(full,cached,atol=1e-6)
        prefix=model(x[:,:8])[0]
        _,_,multi_cache=model(x[:,7:],use_cache=True,past_key_values=cache)
        # Each cached hidden state must match independent decoding of that prefix.
        _,_,one_cache=model(x[:,7:8],use_cache=True,past_key_values=cache)
        for multi,one in zip(multi_cache,one_cache):
            assert torch.allclose(multi[0][:,:,:8],one[0],atol=1e-6)


def test_checkpoint_is_immutable_and_weights_only_loadable(tmp_path):
    model=GPT(tiny_config())
    snapshot={key:value.clone() for key,value in model.state_dict().items()}
    atomic_save({'config':asdict(model.config),'model':model.state_dict(),'step':1,'val':2.0},tmp_path/'best.pt')
    with torch.no_grad():
        model.transformer.wte.weight.add_(1.)
    loaded=load_checkpoint(tmp_path/'best.pt')
    assert torch.equal(snapshot['transformer.wte.weight'],loaded['model']['transformer.wte.weight'])
    assert not torch.equal(model.transformer.wte.weight,loaded['model']['transformer.wte.weight'])
    assert not (tmp_path/'best.tmp').exists()


@pytest.mark.parametrize('field,value',[('steps',0),('batch',-2),('context',999999),('lr',float('nan')),('dropout',2),('size','huge'),('eval_every',0),('steps',2.5)])
def test_preflight_rejects_invalid_training_before_worker(tmp_path,field,value):
    store=Store(tmp_path)
    did=add_dataset(store,'sample.txt','The river flows through the city. '*1000)
    with pytest.raises(ValueError):
        training_config(store,{'dataset':did,field:value})


def test_real_training_stop_and_resume_preserves_optimizer_and_best(tmp_path,monkeypatch):
    store=Store(tmp_path)
    did=add_dataset(store,'sample.txt','The river flows through the city. '*200)
    config=training_config(store,{'dataset':did,'steps':4,'context':16,'batch':1,'eval_every':1})
    # Exercise the real training loop with a small architecture to keep regression tests quick.
    monkeypatch.setattr(GPTConfig,'get_preset',classmethod(lambda cls,size,vocab,block: GPTConfig(vocab_size=vocab,block_size=block,n_layer=1,n_head=2,n_embd=16)))
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    rid=uid();folder=store.root/'runs'/rid;folder.mkdir()
    store.execute('INSERT INTO runs(id,name,config,status,created,updated) VALUES(?,?,?,?,?,?)',(rid,'Test',json.dumps(config),'queued',time.time(),time.time()))
    original_save=atomic_save
    def stop_after_second(payload,path):
        original_save(payload,path)
        if path.name=='latest.pt' and payload['step']==2:
            store.execute("UPDATE runs SET control='stop' WHERE id=?",(rid,))
    monkeypatch.setattr('vvgpt.worker.atomic_save',stop_after_second)
    work(tmp_path,rid)
    assert store.rows('SELECT status FROM runs')[0]['status']=='stopped'
    checkpoint=load_checkpoint(folder/'latest.pt')
    assert checkpoint['step']==2 and checkpoint['optimizer']['state']
    store.execute("UPDATE runs SET control='' WHERE id=?",(rid,))
    monkeypatch.setattr('vvgpt.worker.atomic_save',original_save)
    work(tmp_path,rid)
    run=store.rows('SELECT * FROM runs')[0]
    assert run['status']=='completed' and run['step']==4
    metrics=store.rows('SELECT * FROM metrics ORDER BY step')
    assert [m['step'] for m in metrics]==[1,2,3,4]
    best=load_checkpoint(folder/'best.pt')
    assert best['val']==min(m['val'] for m in metrics)
    assert library(store)[-1]['id']==f'run:{rid}'

    # An interrupted CPU run should reach the same weights as an uninterrupted run.
    uninterrupted=uid();(store.root/'runs'/uninterrupted).mkdir()
    store.execute('INSERT INTO runs(id,name,config,status,created,updated) VALUES(?,?,?,?,?,?)',(uninterrupted,'Uninterrupted',json.dumps(config),'queued',time.time(),time.time()))
    work(tmp_path,uninterrupted)
    resumed_weights=load_checkpoint(folder/'latest.pt')['model']
    direct_weights=load_checkpoint(store.root/'runs'/uninterrupted/'latest.pt')['model']
    assert all(torch.equal(resumed_weights[k],direct_weights[k]) for k in resumed_weights)


def test_untrusted_pickle_not_loaded(tmp_path):
    class_payload={'not_allowed':set}
    # A global callable outside the config allowlist must fail rather than execute.
    import subprocess
    torch.save({'callable':subprocess.Popen},tmp_path/'untrusted.pt')
    with pytest.raises(Exception,match='Weights only load failed'):
        load_checkpoint(tmp_path/'untrusted.pt')
