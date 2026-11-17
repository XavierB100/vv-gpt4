import math
from dataclasses import asdict
from types import SimpleNamespace
import torch
from vvgpt.data import prepare, split_text, fingerprint
from vvgpt.tokenizer import fit_codec, Codec
from vvgpt.lab import add_dataset, dataset_codec, ModelCache
from vvgpt.store import Store
from vvgpt.model import GPT, GPTConfig, banned_ngram_tokens, repeat_fraction
from vvgpt.worker import atomic_save
from vvgpt.evaluation import score_text


def test_chat_preparation_preserves_unicode_speakers_and_multiline():
    source='[12/11/2026, 9:31:02 PM] Alice: café 😀 + £5\nsecond line\n12/11/2026, 9:32 pm - Bob: Yes!\n12/11/2026, 9:33 pm - Messages are encrypted'
    prepared,report=prepare(source)
    assert prepared=='Alice: café 😀 + £5\nsecond line\nBob: Yes!'
    assert report['format']=='whatsapp' and report['messages']==2


def test_preparation_deduplicates_long_blocks_and_splits_without_overlap():
    block='A varied paragraph about a city and its inhabitants. '*4
    text,report=prepare(block+'\n\n'+block+'\n\n'+'A short ending.')
    assert report['removed_blocks']==1 and text.count(block.strip())==1
    raw=''.join(f'Line {i}: different words.\n' for i in range(1000))
    splits=split_text(raw)
    assert ''.join(splits.values())==raw
    assert not (set(splits['train'].splitlines()) & set(splits['test'].splitlines()))


def test_codec_fits_only_train_and_roundtrips_unseen_unicode(tmp_path,monkeypatch):
    import vvgpt.lab as lab
    store=Store(tmp_path)
    did=add_dataset(store,'source.txt',''.join(f'Item {i}: hello world.\n' for i in range(1000)))
    seen=[];real=lab.fit_codec
    def fit(text,kind):seen.append(text);return real(text,kind)
    monkeypatch.setattr(lab,'fit_codec',fit)
    codec,splits=dataset_codec(store,did,'compact')
    assert seen==[splits['train']]
    assert codec.decode(codec.encode('Unseen: 世界 🪐 café\n£5'))=='Unseen: 世界 🪐 café\n£5'
    assert dataset_codec(store,did,'compact')[0].spec==codec.spec
    assert len(seen)==1


def test_embedded_codec_checkpoint_and_seeded_phrase_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    codec=fit_codec('one two three\n'*100)
    model=GPT(GPTConfig(vocab_size=codec.vocab_size,block_size=16,n_layer=1,n_head=2,n_embd=16)).eval()
    path=tmp_path/'best.pt'
    atomic_save({'config':asdict(model.config),'model':model.state_dict(),'codec':codec.spec,'step':1},path)
    cache=ModelCache();net,loaded,device,step=cache.get(path)
    assert loaded.spec==codec.spec and loaded.decode(loaded.encode('🙂 é'))=='🙂 é'
    prompt=torch.tensor([loaded.encode('one')])
    args=dict(max_new_tokens=60,top_k=5,no_repeat_ngram_size=4)
    a=net.generate(prompt,generator=torch.Generator().manual_seed(5),**args)
    b=net.generate(prompt,generator=torch.Generator().manual_seed(5),**args)
    assert torch.equal(a,b) and repeat_fraction(a[0,prompt.size(1):].tolist())==0
    assert banned_ngram_tokens([1,2,3,1,2],3)=={3}


def test_evaluation_scores_every_target_once_with_context_and_restores_mode():
    class Bytes:
        eos=0
        def encode(self,text):return list(text.encode())
    class Recording(torch.nn.Module):
        config=SimpleNamespace(block_size=8)
        def __init__(self):super().__init__();self.targets=[]
        def forward(self,x,y):
            assert x.size(1)<=8
            self.targets.extend(y[y!=-1].tolist())
            return None,torch.tensor(math.log(256))
    model=Recording().train();text='ABCDEFGHIJKLMNOPQé'
    result=score_text(model,Bytes(),text,'cpu')
    assert model.targets==list(text.encode()) and model.training
    assert result['tokens']==result['bytes']==len(text.encode())
    assert math.isclose(result['bits_per_byte'],8,rel_tol=1e-6)
