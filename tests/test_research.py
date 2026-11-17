import json
import math
import hashlib
import torch
import pytest
from dataclasses import asdict
from vvgpt.model import GPT,GPTConfig
from vvgpt.objectives import token_unlikelihood
from vvgpt.lab import add_dataset,training_config,dataset_splits,add_partitioned_dataset
from vvgpt.store import Store
from vvgpt.tokenizer import fit_codec
from vvgpt.worker import work, atomic_save
from vvgpt.lab import start_run, load_checkpoint


@pytest.mark.parametrize('architecture',['classic','revised'])
def test_architecture_cached_prefix_causality_and_reload(architecture):
    torch.manual_seed(3)
    config=GPTConfig(vocab_size=32,block_size=24,n_layer=2,n_head=2,n_embd=16,architecture=architecture)
    model=GPT(config).eval();x=torch.randint(0,32,(1,12))
    with torch.no_grad():
        full=model(x)[0]
        _,_,cache=model(x[:,:7],use_cache=True)
        chunk,_,_=model(x[:,7:],past_key_values=cache,use_cache=True)
        assert torch.allclose(full,chunk,atol=1e-6)
        # Future suffix cannot change representations for earlier tokens.
        _,_,suffix=model(x[:,7:],past_key_values=cache,use_cache=True)
        _,_,single=model(x[:,7:8],past_key_values=cache,use_cache=True)
        for a,b in zip(suffix,single):
            assert torch.allclose(a[0][:,:,:8],b[0],atol=1e-6)
        copy=GPT(GPTConfig(**asdict(config))).eval();copy.load_state_dict(model.state_dict())
        assert torch.equal(full,copy(x)[0])
    if architecture=='revised':
        assert 'wpe' not in model.transformer
        assert not any(name.endswith('bias') for name,_ in model.named_parameters())


def test_unlikelihood_matches_unique_context_definition_and_backpropagates():
    torch.manual_seed(4)
    logits=torch.randn(2,5,8,requires_grad=True)
    inputs=torch.tensor([[1,1,2,3,1],[2,3,3,2,4]])
    targets=torch.tensor([[1,2,3,1,4],[3,3,2,4,-1]])
    actual=token_unlikelihood(logits,inputs,targets,window=3)
    expected=0
    for b in range(2):
        for t in range(5):
            if targets[b,t]<0:continue
            negatives=set(inputs[b,max(0,t-2):t+1].tolist())-{targets[b,t].item()}
            p=logits[b,t].softmax(-1)
            expected+=sum(-torch.log1p(-p[n]) for n in negatives)
    expected/=9
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.backward();assert logits.grad.isfinite().all() and logits.grad.abs().sum()>0


def test_explicit_partitions_remain_fixed_and_detect_tampering(tmp_path):
    store=Store(tmp_path);splits={'train':'A useful training sentence. '*100,'validation':'Validation text. '*50,'test':'Unseen test text. '*50}
    did=add_partitioned_dataset(store,'Test',splits,{'purpose':'unit test'})
    assert dataset_splits(store,did)==splits
    config=training_config(store,{'dataset':did,'context':16,'tokenizer':'compact'})
    assert config['split_label']=='Explicit research partitions'
    (store.root/'datasets'/f'{did}.txt').write_text('changed')
    with pytest.raises(ValueError,match='partitions changed'):dataset_splits(store,did)


def test_continue_uses_embedded_vocabulary_and_source_architecture(tmp_path,monkeypatch):
    store=Store(tmp_path);did=add_dataset(store,'text','A useful sentence to train. '*300)
    codec=fit_codec('Source words here. '*100)
    config=GPTConfig(vocab_size=codec.vocab_size,block_size=16,n_layer=1,n_head=2,n_embd=16,architecture='revised')
    model=GPT(config);source=tmp_path/'source.pt'
    torch.save({'config':asdict(config),'model':model.state_dict(),'codec':codec.spec},source)
    monkeypatch.setattr('vvgpt.lab.model_path',lambda *a:source)
    result=training_config(store,{'dataset':did,'initial_model':'source','tokenizer':'gpt2','context':128})
    assert result['architecture']=='revised' and result['context']==16 and result['tokenizer']=='compact'
    assert result['initial_sha256'] and result['codec_sha256']


def test_continuation_preserves_parent_and_best_when_learning_worsens(tmp_path,monkeypatch):
    store=Store(tmp_path)
    did=add_dataset(store,'text','The river flows past the house. '*300)
    codec=fit_codec('The river flows past the house. '*300)
    model=GPT(GPTConfig(vocab_size=codec.vocab_size,block_size=16,n_layer=1,n_head=2,n_embd=16,architecture='revised'))
    source=tmp_path/'parent.pt'
    torch.save({'config':asdict(model.config),'model':model.state_dict(),'codec':codec.spec},source)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr('vvgpt.lab.model_path',lambda *a:source)
    monkeypatch.setattr('vvgpt.lab.spawn',lambda *a:None)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    rid=start_run(store,{'name':'Continue','dataset':did,'initial_model':'parent','steps':4,'batch':1,'eval_every':1,'patience':8})
    folder=store.root/'runs'/rid
    # Controlled evaluation simulates a worse continuation while exercising real updates,
    # checkpoint copying, interruption, optimizer restoration and checkpoint selection.
    scores=iter([1.,1.]+[2.]*20)
    monkeypatch.setattr('vvgpt.worker.evaluate',lambda *a:next(scores))
    def interrupt_initial_save(payload,path):
        if path.name=='latest.pt':raise OSError('Simulated interruption after best save')
        atomic_save(payload,path)
    monkeypatch.setattr('vvgpt.worker.atomic_save',interrupt_initial_save)
    with pytest.raises(OSError,match='Simulated interruption'):work(tmp_path,rid)
    assert (folder/'best.pt').exists() and not (folder/'latest.pt').exists()
    scores=iter([1.,1.]+[2.]*20)
    def stop_after_second(payload,path):
        atomic_save(payload,path)
        if path.name=='latest.pt' and payload['step']==2:
            store.execute("UPDATE runs SET control='stop' WHERE id=?",(rid,))
    monkeypatch.setattr('vvgpt.worker.atomic_save',stop_after_second)
    work(tmp_path,rid)
    assert store.rows('SELECT status FROM runs')[0]['status']=='stopped'
    store.execute("UPDATE runs SET control='' WHERE id=?",(rid,))
    monkeypatch.setattr('vvgpt.worker.atomic_save',atomic_save)
    work(tmp_path,rid)
    best=load_checkpoint(folder/'best.pt');latest=load_checkpoint(folder/'latest.pt')
    assert best['step']==0 and best['val']==1. and latest['step']==4
    assert all(torch.equal(best['model'][k],v) for k,v in model.state_dict().items())
    assert not torch.equal(latest['model']['transformer.wte.weight'],best['model']['transformer.wte.weight'])
    assert json.loads((folder/'summary.json').read_text())['initial_val']==1.
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
