"""One durable, isolated training job at a time. Original VV-GPT3 files are read-only."""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .store import uid
from .data import prepare, split_text, fingerprint
from .tokenizer import Codec, fit_codec
import threading

_CODEC_LOCK = threading.Lock()

LEGACY = Path('/Users/xavier/AI/hermes-workspace/vv-gpt3/models')
PRESETS = {'nano', 'micro', 'mini', 'tiny', 'small'}


def tokenize(text):
    import tiktoken
    return tiktoken.get_encoding('gpt2').encode_ordinary(text)


def add_dataset(store, name, text, deduplicate=True):
    prepared, report = prepare(text, deduplicate)
    encoded = tokenize(prepared)
    if len(encoded) < 200:
        raise ValueError('Add at least 200 tokens of varied text; useful training needs much more.')
    did = uid()
    (store.root/'datasets'/f'{did}.raw.txt').write_text(text,encoding='utf-8')
    (store.root/'datasets'/f'{did}.txt').write_text(prepared,encoding='utf-8')
    (store.root/'datasets'/f'{did}.json').write_text(json.dumps(report,indent=2))
    store.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',
                  (did,name,len(encoded),len(prepared),fingerprint(prepared),time.time()))
    return did


def dataset_info(store):
    result=store.rows('SELECT * FROM datasets ORDER BY created DESC')
    for item in result:
        path=store.root/'datasets'/f'{item["id"]}.json'
        item['analysis']=json.loads(path.read_text()) if path.exists() else {}
    return result


def dataset_splits(store, did):
    text = (store.root/'datasets'/f'{did}.txt').read_text(encoding='utf-8')
    manifest = store.root/'datasets'/f'{did}.splits.json'
    if not manifest.exists():
        return split_text(text)
    info = json.loads(manifest.read_text())
    if fingerprint(text) != info['sha256']:
        raise ValueError('Explicit dataset partitions changed.')
    first, second = info['cuts']
    if not 0 < first < second < len(text):
        raise ValueError('Invalid explicit dataset partitions.')
    return {'train': text[:first], 'validation': text[first:second], 'test': text[second:]}


def add_partitioned_dataset(store, name, splits, provenance):
    """Internal research import: preserve independently defined partitions verbatim."""
    did = uid()
    text = ''.join(splits[key] for key in ('train','validation','test'))
    prefix = store.root/'datasets'/did
    prefix.with_suffix('.txt').write_text(text, encoding='utf-8')
    prefix.with_suffix('.raw.txt').write_text(text, encoding='utf-8')
    manifest = {'sha256':fingerprint(text),'cuts':[len(splits['train']),len(splits['train'])+len(splits['validation'])],
                'provenance':provenance,'hashes':{k:fingerprint(v) for k,v in splits.items()}}
    prefix.with_suffix('.splits.json').write_text(json.dumps(manifest,indent=2))
    prefix.with_suffix('.json').write_text(json.dumps({'format':'text','split':'explicit','provenance':provenance,
        'warnings':['Research partitions are fixed in the source manifest.'],'prepared_sha256':fingerprint(text)}))
    store.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',(did,name,len(tokenize(text)),len(text),fingerprint(text),time.time()))
    return did


def dataset_codec(store, did, kind):
    if kind not in ('compact','gpt2'): raise ValueError('Unknown tokenizer.')
    splits=dataset_splits(store,did)
    target=store.root/'datasets'/f'{did}.{kind}.json'
    with _CODEC_LOCK:
        if target.exists():
            artifact=json.loads(target.read_text())
            if artifact['train_sha256']!=fingerprint(splits['train']):
                raise ValueError('Tokenizer source changed. Reimport the dataset.')
            codec=Codec(artifact['codec'])
        else:
            codec=fit_codec(splits['train'],kind)
            artifact={'train_sha256':fingerprint(splits['train']),'codec':codec.spec}
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(artifact));temporary.replace(target)
    return codec,splits


def training_config(store, payload):
    dataset = store.rows('SELECT * FROM datasets WHERE id=?', (payload.get('dataset'),))
    if not dataset:
        raise ValueError('Select a dataset first.')
    config = {'dataset': dataset[0]['id'], 'dataset_sha256': dataset[0]['sha256'],
              'size': payload.get('size', 'nano'), 'seed': 42, 'pipeline': 6,
              'tokenizer':payload.get('tokenizer','gpt2')}
    config['architecture'] = payload.get('architecture','revised')
    if config['architecture'] not in ('classic','revised'):
        raise ValueError('Unknown architecture.')
    config['unlikelihood'] = float(payload.get('unlikelihood',0))
    if not 0 <= config['unlikelihood'] <= 1:
        raise ValueError('Unlikelihood must be between 0 and 1.')
    if config['size'] not in PRESETS:
        raise ValueError('Unknown model size.')
    for key, default, low, high in [('steps', 500, 2, 100000), ('context', 128, 16, 1024),
                                   ('batch', 4, 1, 32), ('eval_every', 25, 1, 1000),
                                   ('patience', 8, 1, 100), ('accumulation', 1, 1, 16), ('seed',42,0,2147483647)]:
        raw = payload.get(key, default)
        if isinstance(raw, bool) or not math.isfinite(float(raw)) or float(raw) != int(raw):
            raise ValueError(f'{key} must be a whole number.')
        config[key] = int(raw)
        if not low <= config[key] <= high:
            raise ValueError(f'{key} must be between {low} and {high}.')
    config['lr'] = float(payload.get('lr', 0.0003))
    config['dropout'] = float(payload.get('dropout', 0.1))
    if not 0.000001 <= config['lr'] <= 0.01 or not 0 <= config['dropout'] <= 0.5:
        raise ValueError('Learning rate or dropout is out of range.')
    initial = payload.get('initial_model')
    if initial:
        path = model_path(store, initial)
        checkpoint = load_checkpoint(path)
        model_config = checkpoint.get('config') or asdict(checkpoint['model_args'])
        codec = Codec(checkpoint.get('codec'))
        config.update(initial_model=initial,initial_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      initial_config=model_config,context=model_config['block_size'],
                      architecture=model_config.get('architecture','classic'),tokenizer=codec.spec['kind'])
        splits = dataset_splits(store,config['dataset'])
    else:
        codec,splits=dataset_codec(store,config['dataset'],config['tokenizer'])
    lengths={key:len(codec.encode(value)) for key,value in splits.items()}
    if any(value<=config['context']+1 for value in lengths.values()):
        raise ValueError('A train/validation/test split is too short. Add more data or reduce context.')
    config['codec_sha256']=fingerprint(json.dumps(codec.spec,sort_keys=True))
    config['split_sha256']=fingerprint(json.dumps({k:fingerprint(v) for k,v in splits.items()},sort_keys=True))
    config['split_label']='Explicit research partitions' if (store.root/'datasets'/f'{config["dataset"]}.splits.json').exists() else '80/10/10 chronological'
    return config


def reconcile(store):
    for run in store.rows("SELECT id,pid,updated FROM runs WHERE status IN ('queued','running','paused','evaluating')"):
        if not run['pid'] and time.time() - run['updated'] < 30:
            continue
        if run['pid']:
            try:
                os.kill(run['pid'], 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
        store.execute("UPDATE runs SET status='interrupted',error='Worker exited. Resume from the last checkpoint.',updated=? WHERE id=?",
                      (time.time(), run['id']))


def spawn(store, rid):
    folder = store.root / 'runs' / rid
    folder.mkdir(exist_ok=True)
    with (folder / 'worker.log').open('ab') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'vvgpt.worker', str(store.root), rid],
                                cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log,
                                start_new_session=True)
    store.execute('UPDATE runs SET pid=? WHERE id=?', (proc.pid, rid))


def start_run(store, payload):
    name = str(payload.get('name', 'Untitled experiment')).strip()[:80]
    if not name:
        raise ValueError('Give the experiment a name.')
    config = training_config(store, payload)
    rid = uid()
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT id FROM runs WHERE status IN ('queued','running','paused','evaluating')").fetchone():
            raise ValueError('Stop the active run before starting another. A paused run still holds the training slot.')
        db.execute('INSERT INTO runs(id,name,config,status,created,updated) VALUES(?,?,?,?,?,?)',
                   (rid, name, json.dumps(config), 'queued', time.time(), time.time()))
    try:
        if config.get('initial_model'):
            from .worker import atomic_save
            folder=store.root/'runs'/rid
            folder.mkdir(exist_ok=True)
            path=model_path(store,config['initial_model'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=config['initial_sha256']:
                raise ValueError('Source checkpoint changed during setup. Try again.')
            checkpoint=load_checkpoint(path)
            atomic_save({'config':config['initial_config'],'model':checkpoint['model'],
                         'codec':checkpoint.get('codec',{'kind':'gpt2'})},folder/'initial.pt')
        spawn(store, rid)
    except Exception as exc:
        store.execute("UPDATE runs SET status='failed',error=? WHERE id=?", (str(exc), rid))
        raise
    return rid


def control_run(store, rid, action):
    rows = store.rows('SELECT * FROM runs WHERE id=?', (rid,))
    if not rows:
        raise ValueError('Experiment not found.')
    run = rows[0]
    if action in ('pause', 'stop'):
        if run['status'] not in ('running', 'paused', 'queued'):
            raise ValueError('This experiment is not running.')
        store.execute('UPDATE runs SET control=? WHERE id=?', (action, rid))
    elif action == 'resume':
        if run['status'] in ('paused', 'running'):
            store.execute("UPDATE runs SET control='' WHERE id=?", (rid,))
        elif run['status'] in ('interrupted', 'failed', 'stopped'):
            if not (store.root / 'runs' / rid / 'latest.pt').is_file():
                raise ValueError('No resumable checkpoint exists. Start a new experiment.')
            with store.db() as db:
                db.execute('BEGIN IMMEDIATE')
                if db.execute("SELECT id FROM runs WHERE status IN ('queued','running','paused','evaluating')").fetchone():
                    raise ValueError('Another experiment is active.')
                db.execute("UPDATE runs SET status='queued',control='',error=NULL WHERE id=?", (rid,))
            spawn(store, rid)
        else:
            raise ValueError('This experiment has finished.')
    else:
        raise ValueError('Unknown experiment action.')


def library(store):
    result = []
    for path in sorted(LEGACY.glob('*.metadata.json')):
        try:
            meta = json.loads(path.read_text())
            name = path.name.removesuffix('.metadata.json')
            checkpoint = LEGACY / f'{name}_best.pt'
            if not checkpoint.is_file() or not checkpoint.resolve().is_relative_to(LEGACY.resolve()):
                continue
            tm = meta.get('training_metadata', {})
            result.append({'id': 'legacy:' + name, 'name': name, 'origin': 'VV-GPT3', 'params': meta.get('params'),
                           'best': tm.get('best_val_loss'), 'step': tm.get('best_val_loss_step'),
                           'context': tm.get('block_size'), 'train_history': tm.get('train_loss_history', []),
                           'val_history': tm.get('val_loss_history', []), 'variant': 'best', 'readonly': True, 'tokenizer':'gpt2', 'vocab':meta.get('vocab_size',50257)})
        except (OSError, ValueError):
            continue
    for run in store.rows('SELECT * FROM runs ORDER BY created DESC'):
        if (store.root / 'runs' / run['id'] / 'best.pt').is_file():
            config = json.loads(run['config'])
            result.append({'id': 'run:' + run['id'], 'name': run['name'], 'origin': 'VV-GPT4',
                           'best': run['best'], 'step': run['step'], 'context': config['context'],
                           'variant': 'best', 'readonly': False, 'tokenizer':config.get('tokenizer','gpt2'),
                           'summary':run_summary(store,run['id'])})
    return result


def load_checkpoint(path):
    import torch
    from .model import GPTConfig
    # Only the inspected legacy configuration dataclass is allowed, never arbitrary pickle globals.
    with torch.serialization.safe_globals([(GPTConfig, 'src.models.enhanced_gpt.GPTConfig'),
                                            (GPTConfig, 'enhanced_gpt.GPTConfig')]):
        return torch.load(path, map_location='cpu', weights_only=True)


def run_summary(store,rid):
    path=store.root/'runs'/rid/'summary.json'
    return json.loads(path.read_text()) if path.exists() else {}


def model_path(store,model_id,variant='best'):
    if variant not in ('best','final','latest'): raise ValueError('Unknown checkpoint variant.')
    models={m['id']:m for m in library(store)}
    if model_id not in models: raise ValueError('Model not found.')
    model=models[model_id]
    if model_id.startswith('legacy:'):
        suffix={'best':'_best','latest':'_latest','final':''}[variant]
        return LEGACY/f'{model["name"]}{suffix}.pt'
    return store.root/'runs'/model_id[4:]/f'{"latest" if variant=="latest" else "best"}.pt'


class ModelCache:
    def __init__(self):
        self.key=None
        self.loaded=None

    def get(self,path):
        import torch
        from .model import GPT,GPTConfig
        key=(str(path.resolve()),path.stat().st_mtime_ns)
        if self.key!=key:
            self.loaded=None
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            checkpoint=load_checkpoint(path)
            config=checkpoint.get('config') or asdict(checkpoint['model_args'])
            net=GPT(GPTConfig(**config))
            net.load_state_dict(checkpoint['model'])
            device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            net.to(device).eval()
            codec=Codec(checkpoint.get('codec'))
            self.loaded=(net,codec,device,checkpoint.get('step',checkpoint.get('iter_num')))
            self.key=key
        return self.loaded


def generate_sample(store, model_id, prompt, length, settings=None, cache=None):
    import torch
    from .model import repeat_fraction
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>8000:
        raise ValueError('Enter a prompt of 1–8,000 characters.')
    def integer(value, label):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) != int(value):
            raise ValueError(f'{label} must be a whole number.')
        return int(value)
    length=integer(length, 'Tokens')
    if not 1<=length<=512: raise ValueError('Generate between 1 and 512 tokens.')
    settings=settings or {}
    seed=integer(settings.get('seed',42), 'Seed')
    temperature=float(settings.get('temperature',.8))
    penalty=float(settings.get('repetition_penalty',1.0))
    ngram=integer(settings.get('no_repeat_ngram',0), 'Phrase guard')
    if not 0<=seed<=2147483647 or not .1<=temperature<=2 or not 1<=penalty<=1.5 or ngram not in (0,3,4,5):
        raise ValueError('Generation settings are out of range.')
    net,codec,device,step=(cache or ModelCache()).get(model_path(store,model_id))
    token_ids=codec.encode(prompt)
    idx=torch.tensor([token_ids],device=device)
    start=time.monotonic()
    with torch.inference_mode():
        output=net.generate(idx,length,temperature=temperature,top_k=50,top_p=.9,
            repetition_penalty=penalty,no_repeat_ngram_size=ngram,
            generator=torch.Generator().manual_seed(seed),eos_token_id=codec.eos)
    generated=output[0,len(token_ids):].tolist()
    if generated and generated[-1]==codec.eos: generated=generated[:-1]
    text=codec.decode(generated)
    return {'text':text,'tokens':len(generated),'seconds':round(time.monotonic()-start,3),
            'checkpoint':'best','step':step,'repeat_fraction':repeat_fraction(generated,4),
            'seed':seed,'prompt_tokens':len(token_ids),'used_prompt_tokens':min(len(token_ids),net.config.block_size),
            'context':net.config.block_size,'settings':{'temperature':temperature,'repetition_penalty':penalty,'no_repeat_ngram':ngram}}
