"""Training for Xavier's Transformer; durable jobs and vocabulary-aware checkpoints."""
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch
from .lab import tokenize, dataset_codec, dataset_splits
from .data import fingerprint
from .model import GPT, GPTConfig
from .store import Store
from .tokenizer import Codec
from .evaluation import score_text
from .objectives import token_unlikelihood


def atomic_save(payload,path):
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary,path)


def batch(data,size,context,generator,device):
    starts=torch.randint(len(data)-context,(size,),generator=generator)
    indices=starts[:,None]+torch.arange(context)[None,:]
    return data[indices].to(device),data[indices+1].to(device)


@torch.no_grad()
def evaluate(model,data,config,device):
    generator=torch.Generator().manual_seed(12345)
    was_training=model.training
    model.eval()
    losses=[]
    for _ in range(4):
        x,y=batch(data,config['batch'],config['context'],generator,device)
        losses.append(model(x,y)[1].item())
    model.train(was_training)
    return sum(losses)/len(losses)


def work(root,rid):
    store=Store(root)
    folder=store.root/'runs'/rid
    run=store.rows('SELECT * FROM runs WHERE id=?',(rid,))[0]
    config=json.loads(run['config'])
    text=(store.root/'datasets'/f'{config["dataset"]}.txt').read_text(encoding='utf-8')
    if fingerprint(text)!=config['dataset_sha256']:
        raise ValueError('Dataset changed since this run was created.')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    initial = torch.load(folder/'initial.pt',map_location='cpu',weights_only=True) if config.get('initial_model') else None
    if config.get('pipeline',0)>=5:
        if initial:
            codec,splits=Codec(initial['codec']),dataset_splits(store,config['dataset'])
        else:
            codec,splits=dataset_codec(store,config['dataset'],config['tokenizer'])
        if fingerprint(json.dumps(codec.spec,sort_keys=True))!=config['codec_sha256']:
            raise ValueError('Tokenizer changed. Import a new dataset instead of resuming this run.')
        if config.get('split_sha256') and fingerprint(json.dumps({k:fingerprint(v) for k,v in splits.items()},sort_keys=True))!=config['split_sha256']:
            raise ValueError('Dataset partitions changed since this run was created.')
        train=torch.tensor(codec.encode(splits['train']),dtype=torch.long)
        val=torch.tensor(codec.encode(splits['validation']),dtype=torch.long)
    else:
        # Preserve resume compatibility with alpha runs without changing their splits.
        codec=Codec();encoded=torch.tensor(tokenize(text),dtype=torch.long)
        cut=int(len(encoded)*.9);train,val=encoded[:cut],encoded[cut:]
        splits={'train':text,'validation':'','test':''}
    if initial:
        model_config=GPTConfig(**initial['config'])
    else:
        settings=asdict(GPTConfig.get_preset(config['size'],codec.vocab_size,config['context']))
        settings['architecture']=config.get('architecture','classic')
        model_config=GPTConfig(**settings)
    model_config.dropout=config['dropout']
    model=GPT(model_config).to(device)
    if initial:
        model.load_state_dict(initial['model'])
        del initial
    decay=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
    if config.get('pipeline',0)>=5:
        groups=[{'params':decay,'weight_decay':.1},{'params':no_decay,'weight_decay':0.}]
    else:
        groups=model.parameters()
    optimizer=torch.optim.AdamW(groups,lr=config['lr'],betas=(.9,.95),weight_decay=.1)
    rng=torch.Generator().manual_seed(config['seed'])
    step,best,misses,previous_elapsed=0,float('inf'),0,0.
    initial_val=None
    active_seconds,steady_seconds,steady_steps=0.,0.,0
    if (folder/'latest.pt').is_file():
        checkpoint=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if checkpoint['training_config']!=config:raise ValueError('Checkpoint configuration does not match this experiment.')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        step,best,misses=checkpoint['step'],checkpoint['best_val'],checkpoint['misses']
        previous_elapsed=checkpoint['elapsed']
        initial_val=checkpoint.get('initial_val')
        active_seconds=checkpoint.get('active_seconds',0.)
        steady_seconds=checkpoint.get('steady_seconds',0.);steady_steps=checkpoint.get('steady_steps',0)
        rng.set_state(checkpoint['batch_rng']);torch.set_rng_state(checkpoint['torch_rng'])
        if device=='mps' and 'mps_rng' in checkpoint:torch.mps.set_rng_state(checkpoint['mps_rng'])
        if device=='cuda' and 'cuda_rng' in checkpoint:torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if (folder/'best.pt').exists():
        # A crash may occur after saving improved weights but before latest.pt.
        saved_best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        if saved_best.get('training_config')!=config:
            raise ValueError('Best checkpoint configuration does not match this experiment.')
        best=min(best,saved_best['val'])
        del saved_best
    start=time.monotonic();paused_seconds=0.
    store.execute("UPDATE runs SET status='running',error=NULL,updated=? WHERE id=?",(time.time(),rid))
    model.train();val_loss=None
    summary={'parameters':sum(p.numel() for p in model.parameters()),'vocab_size':codec.vocab_size,
             'tokenizer':config.get('tokenizer','gpt2'),'train_tokens':len(train),'validation_tokens':len(val),
             'test_chars':len(splits['test']),'device':device,'dataset_sha256':config['dataset_sha256'],
             'split':config.get('split_label','80/10/10 chronological') if config.get('pipeline',0)>=5 else 'legacy 90/10',
             'architecture':model_config.architecture,'initial_model':config.get('initial_model'),
             'unlikelihood':config.get('unlikelihood',0)}
    if initial_val is not None:summary['initial_val']=initial_val

    def update_summary():
        elapsed=previous_elapsed+time.monotonic()-start-paused_seconds
        tokens=step*config['batch']*config['context']*config['accumulation']
        summary.update(step=step,elapsed=elapsed,training_seconds=active_seconds,steady_training_seconds=steady_seconds,
                       steady_steps=steady_steps,steady_ms_per_step=1000*steady_seconds/steady_steps if steady_steps else None,
                       tokens_per_second=tokens/max(.001,active_seconds),
                       best_loss=best if math.isfinite(best) else None)
        write_json(folder/'summary.json',summary)
        return elapsed

    def save():
        elapsed=update_summary()
        payload={'schema':5,'config':asdict(model_config),'training_config':config,'codec':codec.spec,
                 'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'best_val':best,
                 'last_evaluated_val':val_loss,'initial_val':summary.get('initial_val'),'misses':misses,'elapsed':elapsed,'active_seconds':active_seconds,
                 'steady_seconds':steady_seconds,'steady_steps':steady_steps,
                 'batch_rng':rng.get_state(),'torch_rng':torch.get_rng_state()}
        if device=='mps':payload['mps_rng']=torch.mps.get_rng_state()
        if device=='cuda':payload['cuda_rng']=torch.cuda.get_rng_state_all()
        atomic_save(payload,folder/'latest.pt')

    update_summary();status='completed'
    if config.get('initial_model') and step==0:
        # Continued training must beat the starting checkpoint on validation.
        train_loss=evaluate(model,train,config,device)
        val_loss=evaluate(model,val,config,device)
        summary['initial_val']=val_loss
        if val_loss<=best:
            best=val_loss
            atomic_save({'schema':5,'config':asdict(model_config),'model':model.state_dict(),
                         'codec':codec.spec,'step':0,'val':best,'training_config':config},folder/'best.pt')
        elapsed=update_summary()
        store.execute('INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?)',(rid,0,train_loss,val_loss,0.,elapsed))
        store.execute('UPDATE runs SET best=? WHERE id=?',(best,rid))
        save()
    while step<config['steps']:
        control=store.rows('SELECT control FROM runs WHERE id=?',(rid,))[0]['control']
        if control=='pause':
            save();pause_start=time.monotonic()
            store.execute("UPDATE runs SET status='paused',step=?,updated=? WHERE id=?",(step,time.time(),rid))
            while control=='pause':
                time.sleep(.25)
                control=store.rows('SELECT control FROM runs WHERE id=?',(rid,))[0]['control']
            paused_seconds+=time.monotonic()-pause_start
            store.execute("UPDATE runs SET status='running' WHERE id=?",(rid,))
        if control=='stop':status='stopped';save();break
        tick=time.monotonic()
        warmup=max(1,min(50,config['steps']//10))
        progress=max(0,(step-warmup)/max(1,config['steps']-warmup))
        lr=config['lr']*min((step+1)/warmup,.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config['accumulation']):
            x,y=batch(train,config['batch'],config['context'],rng,device)
            logits,loss=model(x,y)
            if config.get('unlikelihood',0):
                loss=loss+config['unlikelihood']*token_unlikelihood(logits,x,y)
            loss=loss/config['accumulation']
            if not torch.isfinite(loss):raise ValueError('Loss became non-finite. Reduce the learning rate.')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if device=='mps':torch.mps.synchronize()
        elif device=='cuda':torch.cuda.synchronize()
        seconds=time.monotonic()-tick
        active_seconds+=seconds
        step+=1
        if step>10:steady_seconds+=seconds;steady_steps+=1
        if step%5==0:
            store.execute('UPDATE runs SET step=?,updated=? WHERE id=?',(step,time.time(),rid))
            update_summary()
        if step==1 or step%config['eval_every']==0 or step==config['steps']:
            train_loss=evaluate(model,train,config,device)
            val_loss=evaluate(model,val,config,device)
            previous_best=best
            if val_loss<best:
                best=val_loss
                atomic_save({'schema':5,'config':asdict(model_config),'model':model.state_dict(),
                             'codec':codec.spec,'step':step,'val':val_loss,'training_config':config},folder/'best.pt')
            misses=0 if val_loss<previous_best-.001 else misses+1
            elapsed=update_summary()
            store.execute('INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?)',(rid,step,train_loss,val_loss,lr,elapsed))
            store.execute('UPDATE runs SET step=?,best=?,updated=? WHERE id=?',(step,best,time.time(),rid))
            save()
            print(json.dumps({'step':step,'train':train_loss,'val':val_loss,'best':best}),flush=True)
            if misses>=config['patience']:status='early_stopped';break
    if status in ('completed','early_stopped') and splits['test'] and (folder/'best.pt').exists():
        # The test score is produced once after selection, never used for early stopping.
        store.execute("UPDATE runs SET status='evaluating' WHERE id=?",(rid,))
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['model'])
        summary['best_step']=checkpoint['step']
        summary['test']=score_text(model,codec,splits['test'],device)
        summary['test']['text_sha256']=fingerprint(splits['test'][:8000])
    update_summary()
    store.execute('UPDATE runs SET status=?,step=?,updated=? WHERE id=?',(status,step,time.time(),rid))


if __name__=='__main__':
    root,rid=sys.argv[1:]
    try:work(root,rid)
    except Exception as exc:
        Store(root).execute("UPDATE runs SET status='failed',error=?,updated=? WHERE id=?",(str(exc),time.time(),rid))
        raise
