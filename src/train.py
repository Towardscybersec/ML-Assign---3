"""Development training. This module never constructs or reads a test dataset."""
import argparse, json, time, random
import numpy as np
import torch
from torch import nn
from common import *
from models import MLP,CNN,Autoencoder,count

@torch.no_grad()
def assess(model,x,y,img,condition,auto=False,noisy=False):
    model.eval(); total=0.; hits=0
    for start in range(0,len(x),256):
        raw=x[start:start+256]; clean=img.pixels(raw)
        inp=corrupt(clean,590001+start) if noisy else clean
        out=model(inp if auto else img(raw,condition))
        if auto: total+=nn.functional.mse_loss(out,clean,reduction='sum').item()
        else:
            yy=y[start:start+256]; total+=nn.functional.cross_entropy(out,yy,reduction='sum').item(); hits+=(out.argmax(1)==yy).sum().item()
    return total/(len(x)*3072 if auto else len(x)), (None if auto else hits/len(x))

def train_one(name,kind,condition='clean',seed=17,dim=None):
    path=ROOT/'artifacts/runs'/name; path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists(): print('Already complete',name,flush=True); return
    seed_all(seed); img=Images(); x,y,v,vy=load_development(); auto=kind in ['ae','dae']; mem=kind=='memorisation'
    config=CFG['autoencoder' if auto else 'memorisation' if mem else 'classifier'].copy()
    model=(Autoencoder(dim) if auto else MLP(512) if mem else MLP() if kind=='mlp' else CNN()).to(DEVICE)
    if kind=='mlp' and condition=='permuted':
        # x'[:,j]=x[:,p[j]] implies W'[:,j]=W[:,p[j]]. RGB block order preserved.
        p=(img.perm[None,:]+torch.arange(3,device=DEVICE)[:,None]*1024).flatten()
        with torch.no_grad(): model.layers[1].weight.copy_(model.layers[1].weight[:,p])
    mask=None
    if mem:
        z=np.load(ROOT/'artifacts/splits/memorisation.npz'); split=np.load(ROOT/'artifacts/splits/split.npz')['train']
        pos={int(a):i for i,a in enumerate(split)}; order=[pos[int(a)] for a in z['indices']]
        x=x[order]; y=torch.tensor(z['target'],device=DEVICE)
        original=torch.tensor(z['original'],device=DEVICE); mask=torch.tensor(z['mask'],device=DEVICE)
    meta={'name':name,'kind':kind,'condition':condition,'seed':seed,'dim':dim,'parameters':count(model),
          'config':config,'shuffle_seed_rule':'seed*100000+zero_based_epoch, dedicated generator','hardware':hardware(),'split_sha256':digest(ROOT/'artifacts/splits/split.npz'),
          'selection':'minimum validation MSE' if auto else 'minimum clean validation CE',
          'first_question':'Q1(b)' if mem else 'Q3(b)' if kind=='dae' else 'Q3(a)' if auto else 'Q1(a)/Q2(a)' if condition=='clean' else 'Q2(b)' if condition=='permuted' else 'Q2(c)',
          'source_hashes':{f:digest(ROOT/f) for f in ['src/train.py','src/models.py','src/common.py','configs/experiment.json']}}
    previous_meta=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else None
    if previous_meta is not None:
        for key in ['kind','condition','seed','dim','parameters','config','split_sha256']:
            assert previous_meta[key]==meta[key], f'Cannot resume changed configuration: {key}'
        if previous_meta['source_hashes']!=meta['source_hashes']:
            save_json(path/'resume_source_provenance.json',{'original':previous_meta['source_hashes'],'resumed':meta['source_hashes'],
                'reason':'AE logging-only correction; optimiser, data, models, seeds and losses unchanged'})
    save_json(path/'config.json',meta)
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['epochs'])
    start_epoch=0; best=float('inf'); history=[]; elapsed=0.
    if (path/'last.pt').exists():
        ck=load_ckpt(path/'last.pt'); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        start_epoch=ck['epoch']; best=ck['best']; history=ck['history']; elapsed=ck['elapsed']
        if auto:
            for row in history: row['train_accuracy']=None; row['val_accuracy']=None
        save_json(path/'resume_event.json',{'resumed_after_epoch':start_epoch,'checkpoint_sha256':digest(path/'last.pt'),'restored_optimizer_scheduler_rng':True})
        print(f'Resuming {name} after epoch {start_epoch}',flush=True)
        torch.set_rng_state(ck['rng_cpu'].cpu()); np.random.set_state(ck['rng_numpy']); random.setstate(ck['rng_python'])
        if DEVICE.type=='cuda': torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
    start=time.perf_counter()
    for epoch in range(start_epoch,config['epochs']):
        model.train()
        # Independent stream: architecture initialisation must not change batch order.
        shuffle=torch.Generator(device=DEVICE).manual_seed(seed*100000+epoch)
        order=torch.randperm(len(x),device=DEVICE,generator=shuffle)
        online_loss=0.; online_hit=0
        for batch in order.split(config['batch_size']):
            opt.zero_grad(set_to_none=True)
            if auto:
                clean=img.pixels(x[batch]); inp=corrupt(clean,seed*1000000+epoch*1000+int(batch[0])) if kind=='dae' else clean
                out=model(inp); loss=nn.functional.mse_loss(out,clean)
            else:
                out=model(img(x[batch],condition)); loss=nn.functional.cross_entropy(out,y[batch]); online_hit+=(out.argmax(1)==y[batch]).sum().item()
            loss.backward(); opt.step(); online_loss+=loss.item()*len(batch)
        sched.step()
        val_loss,val_acc=assess(model,v,vy,img,condition,auto,kind=='dae')
        train_loss,train_acc=assess(model,x,y,img,condition,auto,kind=='dae')
        row={'epoch':epoch+1,'train_loss':train_loss,'train_accuracy':train_acc,'online_loss':online_loss/len(x),
             'val_loss':val_loss,'val_accuracy':val_acc,'lr':opt.param_groups[0]['lr'],'elapsed_seconds':elapsed+time.perf_counter()-start}
        if mem:
            logits=predictions(model,x,img); pred=torch.tensor(logits.argmax(1),device=DEVICE)
            row.update({'uncorrupted_accuracy':float((pred[~mask]==original[~mask]).float().mean()),
                        'corrupt_target_accuracy':float((pred[mask]==y[mask]).float().mean()),
                        'corrupt_original_accuracy':float((pred[mask]==original[mask]).float().mean())})
        history.append(row)
        if val_loss<best:
            best=val_loss; atomic_torch(path/'best.pt',{'model':model.state_dict(),'epoch':epoch+1,'val_loss':best,'meta':meta})
        atomic_torch(path/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
            'epoch':epoch+1,'best':best,'history':history,'elapsed':row['elapsed_seconds'],
            'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state() if DEVICE.type=='cuda' else None,
            'rng_numpy':np.random.get_state(),'rng_python':random.getstate()})
        save_json(path/'history.json',history)
        acc_text='NA (reconstruction only)' if auto else f'{val_acc:.4f}'
        print(f'{name} {epoch+1}/{config["epochs"]} train={train_loss:.5f} val={val_loss:.5f} acc={acc_text} seconds={row["elapsed_seconds"]:.1f}',flush=True)
    save_json(path/'complete.json',{'epochs':config['epochs'],'seconds':history[-1]['elapsed_seconds'],'best_checkpoint_sha256':digest(path/'best.pt')})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['classifiers','memorisation','autoencoders','all'],default='all'); args=ap.parse_args()
    if args.stage in ['classifiers','all']:
        for seed in CFG['seeds']:
            for cond in ['clean','permuted']:
                for kind in ['mlp','cnn']: train_one(f'{kind}_{cond}_{seed}',kind,cond,seed)
        for kind in ['mlp','cnn']: train_one(f'{kind}_grayscale_17',kind,'grayscale',17)
    if args.stage in ['memorisation','all']: train_one('memorisation','memorisation',seed=43)
    if args.stage in ['autoencoders','all']:
        for dim in CFG['autoencoder']['bottlenecks']: train_one(f'ae_{dim}','ae',seed=59,dim=dim)
        choices=[(load_ckpt(ROOT/f'artifacts/runs/ae_{d}/best.pt')['val_loss'],d) for d in CFG['autoencoder']['bottlenecks']]
        val,dim=min(choices); save_json(ROOT/'artifacts/results/ae_selection.json',{'dim':dim,'validation_mse':val,'criterion':'minimum validation MSE'})
        train_one('dae','dae',seed=59,dim=dim)

if __name__=='__main__': main()
