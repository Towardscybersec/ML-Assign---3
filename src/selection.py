"""Train-only representations, validation-only probes/SVM/clustering; then freeze."""
import argparse, itertools, json, time, warnings
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from common import *
from models import *
from threadpoolctl import threadpool_limits
# Limit already-loaded BLAS/OpenMP libraries as well as PyTorch.
CPU_LIMITS = threadpool_limits(limits=4)

def get_model(name):
    ck=load_ckpt(ROOT/f'artifacts/runs/{name}/best.pt'); meta=ck['meta']
    m=(Autoencoder(meta['dim']) if meta['kind'] in ['ae','dae'] else MLP(512) if meta['kind']=='memorisation' else MLP() if meta['kind']=='mlp' else CNN()).to(DEVICE)
    m.load_state_dict(ck['model']); m.eval(); m.requires_grad_(False)
    assert all(not p.requires_grad for p in m.parameters())
    return m,meta

@torch.no_grad()
def ae_features(model,raw,img):
    return np.concatenate([model.features(img.pixels(a)).cpu().numpy() for a in raw.split(256)])

def extract():
    seed_all(59); x,y,v,vy=load_development(); img=Images()
    for name in ['cnn_clean_17']+[f'ae_{d}' for d in CFG['autoencoder']['bottlenecks']]+['dae']:
        path=ROOT/f'artifacts/features/{name}.npz'
        checkpoint=ROOT/f'artifacts/runs/{name}/best.pt'
        if path.exists():
            provenance=json.loads(path.with_suffix('.json').read_text()); assert provenance['checkpoint_sha256']==digest(checkpoint); continue
        m,meta=get_model(name); is_ae=meta['kind'] in ['ae','dae']
        f=lambda raw: ae_features(m,raw,img) if is_ae else predictions(m,raw,img,features=True)
        train,val=f(x),f(v)
        np.savez(path,train=train,val=val,train_y=y.cpu().numpy(),val_y=vy.cpu().numpy())
        save_json(path.with_suffix('.json'),{'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':digest(checkpoint),
          'split_sha256':digest(ROOT/'artifacts/splits/split.npz'),'dimension':train.shape[1],
          'extraction':'clean [0,1] AE linear bottleneck' if is_ae else 'training-normalised clean image; penultimate ReLU',
          'encoder_frozen':True,'feature_sha256':digest(path),'first_question':meta['first_question']})
        print('Extracted',name,train.shape,flush=True)

def probes():
    rows=[]; p=ROOT/'artifacts/results/probes'; p.mkdir(exist_ok=True)
    for name in [f'ae_{d}' for d in CFG['autoencoder']['bottlenecks']]+['dae']:
        z=np.load(ROOT/f'artifacts/features/{name}.npz'); scaler=StandardScaler().fit(z['train']); x=scaler.transform(z['train']); v=scaler.transform(z['val'])
        best=(-1,None,None); models={}
        for c in CFG['probe']['C']:
            path=p/f'{name}_C{c}.joblib'; start=time.perf_counter()
            if path.exists(): m=joblib.load(path)
            else:
                m=LogisticRegression(C=c,max_iter=CFG['probe']['max_iter'],solver='lbfgs',tol=1e-4,random_state=59).fit(x,z['train_y']); joblib.dump(m,path)
            acc=m.score(v,z['val_y']); models[c]=m
            row={'representation':name,'C':c,'train_accuracy':m.score(x,z['train_y']),'val_accuracy':acc,'iterations':int(m.n_iter_.max()),'seconds_this_invocation':time.perf_counter()-start}
            rows.append(row); print('Probe',row,flush=True)
            if acc>best[0]: best=(acc,c,m)
        joblib.dump({'model':best[2],'scaler':scaler,'representation':name,'C':best[1]},p/f'{name}_selected.joblib')
    pd.DataFrame(rows).to_csv(ROOT/'artifacts/results/probe_sweep.csv',index=False)

def svm():
    p=ROOT/'artifacts/results/svm'; p.mkdir(exist_ok=True)
    z=np.load(ROOT/'artifacts/features/cnn_clean_17.npz'); scaler=StandardScaler().fit(z['train']); full=scaler.transform(z['train']); v=scaler.transform(z['val'])
    ix=np.load(ROOT/'artifacts/splits/split.npz')['train']; pos={int(a):i for i,a in enumerate(ix)}
    subset=np.load(ROOT/'artifacts/splits/svm_subset.npy'); order=np.array([pos[int(a)] for a in subset]); x=full[order]; y=z['train_y'][order]
    joblib.dump(scaler,ROOT/'artifacts/results/representation_scaler.joblib')
    rows=[]; best=(-1,None,None,None)
    # C first ensures explicit tie preference: smaller C, then config kernel order.
    for c in CFG['svm']['C']:
        for kernel in CFG['svm']['kernels']:
            path=p/f'{kernel}_C{c}.joblib'; info=path.with_suffix('.json')
            if path.exists() and info.exists(): m=joblib.load(path); row=json.loads(info.read_text())
            else:
                start=time.perf_counter(); m=SVC(C=c,kernel=kernel,gamma='scale',degree=3,coef0=1,cache_size=256,decision_function_shape='ovo',tol=.001,max_iter=-1,random_state=71)
                m.fit(x,y); row={'kernel':kernel,'C':c,'train_accuracy':m.score(x,y),'val_accuracy':m.score(v,z['val_y']),
                  'sv_fraction':len(m.support_)/len(x),'support_vectors':len(m.support_),'gamma':float(m._gamma),'seconds':time.perf_counter()-start,'fit_status':int(m.fit_status_)}
                joblib.dump(m,path); save_json(info,row)
            rows.append(row); print('SVM',row,flush=True)
            if row['val_accuracy']>best[0]: best=(row['val_accuracy'],c,kernel,m)
    pd.DataFrame(rows).to_csv(ROOT/'artifacts/results/svm_sweep.csv',index=False)
    joblib.dump({'model':best[3],'scaler':scaler,'subset_global_indices':subset,'representation':'cnn_clean_17','dimension':128},p/'selected.joblib')
    save_json(ROOT/'artifacts/results/svm_selection.json',{'C':best[1],'kernel':best[2],'val_accuracy':best[0],'train_samples':len(subset),'feature_scaler_fit_samples':len(full),'representation':'cnn_clean_17','dimension':128})

def clustering():
    p=ROOT/'artifacts/results/clustering'; p.mkdir(exist_ok=True)
    z=np.load(ROOT/'artifacts/features/cnn_clean_17.npz'); scaler=joblib.load(ROOT/'artifacts/results/representation_scaler.joblib'); x=scaler.transform(z['train']); v=scaler.transform(z['val'])
    rng=np.random.default_rng(83); sample=rng.choice(len(v),CFG['clustering']['silhouette_sample'],replace=False); np.save(p/'validation_sample.npy',sample)
    rows=[]; seed_predictions={}; means=[]
    for k in CFG['clustering']['k']:
        labs=[]; scores=[]
        for seed in CFG['clustering']['seeds']:
            path=p/f'k{k}_seed{seed}.joblib'; info=path.with_suffix('.json')
            if path.exists() and info.exists(): m=joblib.load(path); row=json.loads(info.read_text())
            else:
                start=time.perf_counter(); m=KMeans(n_clusters=k,n_init=10,random_state=seed,max_iter=300,algorithm='lloyd').fit(x)
                pred=m.predict(v[sample]); score=silhouette_score(v[sample],pred)
                row={'k':k,'seed':seed,'val_silhouette':float(score),'train_inertia':float(m.inertia_),'iterations':int(m.n_iter_),'seconds':time.perf_counter()-start}
                joblib.dump(m,path); save_json(info,row)
            labs.append(m.predict(v[sample])); scores.append(row['val_silhouette']); rows.append(row); print('Cluster',row,flush=True)
        stability=[adjusted_rand_score(a,b) for a,b in itertools.combinations(labs,2)]
        means.append({'k':k,'mean_val_silhouette':float(np.mean(scores)),'std_val_silhouette':float(np.std(scores)),'mean_seed_ARI':float(np.mean(stability))})
    pd.DataFrame(rows).to_csv(ROOT/'artifacts/results/clustering_selection.csv',index=False)
    pd.DataFrame(means).to_csv(ROOT/'artifacts/results/clustering_summary.csv',index=False)
    chosen=sorted(means,key=lambda a:(-a['mean_val_silhouette'],a['k']))[0]
    m=joblib.load(p/f'k{chosen["k"]}_seed83.joblib'); joblib.dump(m,p/'selected.joblib')
    save_json(ROOT/'artifacts/results/clustering_choice.json',{**chosen,'final_seed':83,'train_samples':len(x),'method':'KMeans','labels_used_for_selection':False})

def freeze():
    path=ROOT/'artifacts/results/frozen.json'
    paths=[]
    for p in sorted((ROOT/'artifacts/runs').glob('*/best.pt')): paths.append(p)
    paths += list((ROOT/'artifacts/results/probes').glob('*_selected.joblib'))
    paths += [ROOT/'artifacts/results/svm/selected.joblib',ROOT/'artifacts/results/clustering/selected.joblib',ROOT/'artifacts/results/representation_scaler.joblib']
    paths += list((ROOT/'artifacts/splits').glob('*'))
    paths += [ROOT/'configs/experiment.json',ROOT/'EXPERIMENT_PLAN.md']
    # Seal experiment source as well as all model/preprocessing selections.
    paths += [ROOT/f for f in ['src/common.py','src/models.py','src/prepare.py','src/train.py','src/selection.py','src/evaluate.py']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    if path.exists(): assert json.loads(path.read_text())['sha256']==hashes,'Frozen configuration changed'; return
    from datetime import datetime, timezone
    save_json(path,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,'test_not_yet_evaluated':True,
                    'policy':'all choices fixed; post-test visual/error analysis allowed; no tuning'})
    print('Frozen',len(hashes),'artifacts before any test evaluation.',flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['extract','probes','svm','clustering','freeze','all'],default='all'); a=ap.parse_args()
    funcs={'extract':extract,'probes':probes,'svm':svm,'clustering':clustering,'freeze':freeze}
    for name,f in funcs.items():
        if a.stage in [name,'all']: f()
if __name__=='__main__': main()
