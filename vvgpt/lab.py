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

LEGACY = Path('/Users/xavier/AI/hermes-workspace/vv-gpt3/models')
PRESETS = {'nano', 'micro', 'tiny', 'small'}


def tokenize(text):
    import tiktoken
    return tiktoken.get_encoding('gpt2').encode_ordinary(text)


def add_dataset(store, name, text):
    encoded = tokenize(text)
    if len(encoded) < 200:
        raise ValueError('Add at least 200 tokens of text. A useful model will need much more.')
    did = uid()
    (store.root / 'datasets' / f'{did}.txt').write_text(text, encoding='utf-8')
    store.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?)',
                  (did, name, len(encoded), len(text), hashlib.sha256(text.encode()).hexdigest(), time.time()))
    return did


def training_config(store, payload):
    dataset = store.rows('SELECT * FROM datasets WHERE id=?', (payload.get('dataset'),))
    if not dataset:
        raise ValueError('Select a dataset first.')
    config = {'dataset': dataset[0]['id'], 'dataset_sha256': dataset[0]['sha256'],
              'size': payload.get('size', 'nano'), 'seed': 42}
    if config['size'] not in PRESETS:
        raise ValueError('Unknown model size.')
    for key, default, low, high in [('steps', 500, 2, 100000), ('context', 128, 16, 1024),
                                   ('batch', 4, 1, 32), ('eval_every', 25, 1, 1000),
                                   ('patience', 6, 1, 100), ('accumulation', 1, 1, 16)]:
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
    if dataset[0]['tokens'] - int(dataset[0]['tokens'] * .9) <= config['context'] + 1:
        raise ValueError('The validation split is too short for this context. Add more data or reduce context.')
    return config


def reconcile(store):
    for run in store.rows("SELECT id,pid,updated FROM runs WHERE status IN ('queued','running','paused')"):
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
        if db.execute("SELECT id FROM runs WHERE status IN ('queued','running','paused')").fetchone():
            raise ValueError('Pause or stop the active run before starting another. Paused runs still hold the training slot.')
        db.execute('INSERT INTO runs(id,name,config,status,created,updated) VALUES(?,?,?,?,?,?)',
                   (rid, name, json.dumps(config), 'queued', time.time(), time.time()))
    try:
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
                if db.execute("SELECT id FROM runs WHERE status IN ('queued','running','paused')").fetchone():
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
                           'val_history': tm.get('val_loss_history', []), 'variant': 'best', 'readonly': True})
        except (OSError, ValueError):
            continue
    for run in store.rows('SELECT * FROM runs ORDER BY created DESC'):
        if (store.root / 'runs' / run['id'] / 'best.pt').is_file():
            config = json.loads(run['config'])
            result.append({'id': 'run:' + run['id'], 'name': run['name'], 'origin': 'VV-GPT4',
                           'best': run['best'], 'step': run['step'], 'context': config['context'],
                           'variant': 'best', 'readonly': False})
    return result


def load_checkpoint(path):
    import torch
    from .model import GPTConfig
    # Only the inspected legacy configuration dataclass is allowed, never arbitrary pickle globals.
    with torch.serialization.safe_globals([(GPTConfig, 'src.models.enhanced_gpt.GPTConfig'),
                                            (GPTConfig, 'enhanced_gpt.GPTConfig')]):
        return torch.load(path, map_location='cpu', weights_only=True)


def generate_sample(store, model_id, prompt, length):
    import torch
    import tiktoken
    from .model import GPT, GPTConfig
    models = {m['id']: m for m in library(store)}
    if model_id not in models:
        raise ValueError('Model not found.')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError('Enter a prompt of 1–8,000 characters.')
    length = int(length)
    if not 1 <= length <= 256:
        raise ValueError('Generate between 1 and 256 tokens.')
    model = models[model_id]
    path = LEGACY / f'{model["name"]}_best.pt' if model_id.startswith('legacy:') else store.root / 'runs' / model_id[4:] / 'best.pt'
    checkpoint = load_checkpoint(path)
    config = checkpoint.get('config') or asdict(checkpoint['model_args'])
    net = GPT(GPTConfig(**config))
    net.load_state_dict(checkpoint['model'])
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device).eval()
    encoder = tiktoken.get_encoding('gpt2')
    token_ids = encoder.encode_ordinary(prompt)
    idx = torch.tensor([token_ids], device=device)
    start = time.monotonic()
    with torch.inference_mode():
        output = net.generate(idx, length, temperature=.8, top_k=40, top_p=.9)
    generated = output[0, len(token_ids):].tolist()
    return {'text': encoder.decode(generated), 'tokens': len(generated),
            'seconds': round(time.monotonic()-start, 2), 'checkpoint': 'best',
            'step': checkpoint.get('step', checkpoint.get('iter_num'))}
