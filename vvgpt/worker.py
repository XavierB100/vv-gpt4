import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from .lab import tokenize
from .model import GPT, GPTConfig
from .store import Store


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def batch(data, size, context, generator, device):
    starts = torch.randint(len(data)-context, (size,), generator=generator)
    return (torch.stack([data[i:i+context] for i in starts]).to(device),
            torch.stack([data[i+1:i+context+1] for i in starts]).to(device))


@torch.no_grad()
def evaluate(model, data, config, device):
    generator = torch.Generator().manual_seed(12345)
    model.eval()
    losses = []
    for _ in range(4):
        x, y = batch(data, config['batch'], config['context'], generator, device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses)/len(losses)


def work(root, rid):
    store = Store(root)
    folder = store.root / 'runs' / rid
    run = store.rows('SELECT * FROM runs WHERE id=?', (rid,))[0]
    config = json.loads(run['config'])
    text = (store.root/'datasets'/f'{config["dataset"]}.txt').read_text()
    import hashlib
    if hashlib.sha256(text.encode()).hexdigest() != config['dataset_sha256']:
        raise ValueError('Dataset changed since this run was created.')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    encoded = torch.tensor(tokenize(text), dtype=torch.long)
    split = int(len(encoded)*.9)
    train, val = encoded[:split], encoded[split:]
    model_config = GPTConfig.get_preset(config['size'], 50257, config['context'])
    model_config.dropout = config['dropout']
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9,.95), weight_decay=.1)
    rng = torch.Generator().manual_seed(config['seed'])
    step, best, misses, previous_elapsed = 0, float('inf'), 0, 0
    if (folder/'latest.pt').is_file():
        checkpoint = torch.load(folder/'latest.pt', map_location='cpu', weights_only=True)
        if checkpoint['training_config'] != config:
            raise ValueError('Checkpoint configuration does not match this experiment.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, best, misses = checkpoint['step'], checkpoint['best_val'], checkpoint['misses']
        previous_elapsed = checkpoint['elapsed']
        rng.set_state(checkpoint['batch_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device == 'mps' and 'mps_rng' in checkpoint:
            torch.mps.set_rng_state(checkpoint['mps_rng'])
        if device == 'cuda' and 'cuda_rng' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    start = time.monotonic()
    store.execute("UPDATE runs SET status='running',error=NULL,updated=? WHERE id=?", (time.time(), rid))
    model.train()
    loss_value, val_loss, lr = None, None, config['lr']

    def save():
        payload = {'schema': 4, 'config': asdict(model_config), 'training_config': config,
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                   'best_val': best, 'val': val_loss, 'misses': misses,
                   'elapsed': previous_elapsed+time.monotonic()-start,
                   'batch_rng': rng.get_state(), 'torch_rng': torch.get_rng_state()}
        if device == 'mps':
            payload['mps_rng'] = torch.mps.get_rng_state()
        if device == 'cuda':
            payload['cuda_rng'] = torch.cuda.get_rng_state_all()
        atomic_save(payload, folder/'latest.pt')

    status = 'completed'
    while step < config['steps']:
        control = store.rows('SELECT control FROM runs WHERE id=?', (rid,))[0]['control']
        if control == 'pause':
            save()
            store.execute("UPDATE runs SET status='paused',step=?,updated=? WHERE id=?", (step,time.time(),rid))
            while control == 'pause':
                time.sleep(.3)
                control = store.rows('SELECT control FROM runs WHERE id=?', (rid,))[0]['control']
            store.execute("UPDATE runs SET status='running' WHERE id=?", (rid,))
        if control == 'stop':
            status = 'stopped'
            save()
            break
        warmup = max(1, min(50, config['steps']//10))
        progress = max(0, (step-warmup)/max(1,config['steps']-warmup))
        lr = config['lr'] * min((step+1)/warmup, .1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0
        for _ in range(config['accumulation']):
            x,y = batch(train, config['batch'], config['context'], rng, device)
            loss = model(x,y)[1] / config['accumulation']
            if not torch.isfinite(loss):
                raise ValueError('Loss became non-finite. Reduce the learning rate.')
            loss.backward()
            loss_value += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        step += 1
        if step == 1 or step % config['eval_every'] == 0 or step == config['steps']:
            val_loss = evaluate(model, val, config, device)
            improved = val_loss < best - .001
            if improved:
                best, misses = val_loss, 0
                # Serialize immediately. Never retain a live state_dict as a historical snapshot.
                atomic_save({'schema':4,'config':asdict(model_config),'model':model.state_dict(),
                             'step':step,'val':val_loss,'training_config':config}, folder/'best.pt')
            else:
                misses += 1
            elapsed = previous_elapsed+time.monotonic()-start
            store.execute('INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?)', (rid,step,loss_value,val_loss,lr,elapsed))
            store.execute('UPDATE runs SET step=?,best=?,updated=? WHERE id=?', (step,best,time.time(),rid))
            save()
            print(json.dumps({'step':step,'train':loss_value,'val':val_loss,'best':best}), flush=True)
            if misses >= config['patience']:
                status = 'early_stopped'
                break
    store.execute('UPDATE runs SET status=?,step=?,updated=? WHERE id=?', (status,step,time.time(),rid))


if __name__ == '__main__':
    root, rid = sys.argv[1:]
    try:
        work(root, rid)
    except Exception as exc:
        Store(root).execute("UPDATE runs SET status='failed',error=?,updated=? WHERE id=?", (str(exc),time.time(),rid))
        raise
