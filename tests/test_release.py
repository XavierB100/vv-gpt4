import hashlib
import json
import zipfile
import pytest
import torch
from vvgpt import lab
from vvgpt.model import GPT, GPTConfig
from vvgpt.store import Store
from scripts.install_demo import install


def test_example_works_without_original_checkout(client, headers, monkeypatch, tmp_path):
    monkeypatch.setattr(lab, 'LEGACY', tmp_path/'absent')
    first = client.post('/api/datasets/example', headers=headers)
    assert first.status_code == 200
    assert client.post('/api/datasets/example', headers=headers).json == first.json
    assert len(client.get('/api/state').json['datasets']) == 1


def package(tmp_path):
    config = GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16)
    checkpoint = tmp_path/'best.pt'
    from dataclasses import asdict
    torch.save({'config':asdict(config),'model':GPT(config).state_dict()}, checkpoint)
    rid = '1'*32
    run = dict(id=rid, name='Recorded demo', config='{}', status='completed', step=2, best=2., created=1., updated=2.)
    archive = tmp_path/'demo.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.write(checkpoint,'best.pt')
        z.writestr('summary.json','{}')
        z.writestr('record.json',json.dumps({'run':run,'metrics':[dict(step=2,train=2.,val=2.,lr=.001,elapsed=1.)]}))
    manifest = dict(run_id=rid,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    return archive,manifest


def test_demo_install_is_verified_and_idempotent(tmp_path):
    archive,manifest = package(tmp_path)
    store = Store(tmp_path/'workspace')
    assert install(archive,store,manifest)
    assert not install(archive,store,manifest)
    assert len(store.rows('SELECT * FROM runs')) == len(store.rows('SELECT * FROM metrics')) == 1
    (store.root/'runs'/manifest['run_id']/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='different run'):
        install(archive,store,manifest)


def test_demo_bad_checksum_leaves_workspace_untouched(tmp_path):
    archive,manifest = package(tmp_path)
    manifest['sha256'] = '0'*64
    store = Store(tmp_path/'workspace')
    with pytest.raises(ValueError,match='checksum mismatch'):
        install(archive,store,manifest)
    assert not store.rows('SELECT * FROM runs')
    assert not list((store.root/'runs').iterdir())


def test_demo_rejects_unexpected_archive_paths(tmp_path):
    archive,manifest = package(tmp_path)
    with zipfile.ZipFile(archive,'a') as z:
        z.writestr('../unexpected','bad')
    manifest['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    store = Store(tmp_path/'workspace')
    with pytest.raises(ValueError,match='Unexpected demo contents'):
        install(archive,store,manifest)
    assert not list((store.root/'runs').iterdir())
    assert not (store.root/'unexpected').exists()
