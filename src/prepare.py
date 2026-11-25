"""Download only the official CIFAR10 archive, save the sole split and controls."""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from common import ROOT, CFG, save_json, digest, hardware, seed_all

def main():
    seed_all(260012)
    p=ROOT/'artifacts/splits'; p.mkdir(parents=True,exist_ok=True)
    ds=CIFAR10(ROOT/'data',train=True,download=True)
    y=np.array(ds.targets)
    train,val=train_test_split(np.arange(50000),test_size=5000,stratify=y,random_state=260012)
    if (p/'split.npz').exists():
        old=np.load(p/'split.npz'); assert np.array_equal(train,old['train']) and np.array_equal(val,old['val'])
    else: np.savez(p/'split.npz',train=train,val=val)
    assert len(train)==45000 and len(val)==5000
    assert len(np.unique(np.r_[train,val]))==50000
    assert not np.intersect1d(train,val).size
    assert np.all(np.bincount(y[train])==4500) and np.all(np.bincount(y[val])==500)
    # Accumulate channel moments in small batches in float64; no validation access.
    total=np.zeros(3); sq=np.zeros(3); n=0
    for batch in np.array_split(train,90):
        x=ds.data[batch].astype(np.float64)/255
        total+=x.sum((0,1,2)); sq+=(x*x).sum((0,1,2)); n+=len(x)*1024
    mean=total/n; std=np.sqrt(sq/n-mean**2)
    save_json(p/'statistics.json',{'mean':mean.tolist(),'std':std.tolist(),'fitted_on':'train only','split_sha256':digest(p/'split.npz')})
    rng=np.random.default_rng(260013); perm=rng.permutation(1024)
    np.save(p/'permutation.npy',perm)
    assert np.array_equal(np.sort(perm),np.arange(1024))
    # Positions in these files are global official training dataset indices.
    subset,_=train_test_split(train,train_size=CFG['memorisation']['subset_size'],stratify=y[train],random_state=43)
    rng=np.random.default_rng(43); mask=np.zeros(len(subset),bool)
    mask[rng.choice(len(subset),int(.4*len(subset)),replace=False)]=True
    original=y[subset]; target=original.copy(); target[mask]=(original[mask]+rng.integers(1,10,mask.sum()))%10
    np.savez(p/'memorisation.npz',indices=subset,mask=mask,original=original,target=target)
    svm,_=train_test_split(train,train_size=CFG['svm']['subset_size'],stratify=y[train],random_state=71)
    np.save(p/'svm_subset.npy',svm)
    save_json(ROOT/'artifacts/results/setup.json',{'hardware':hardware(),'train':len(train),'val':len(val),
        'train_counts':np.bincount(y[train]).tolist(),'val_counts':np.bincount(y[val]).tolist(),
        'split_sha256':digest(p/'split.npz'),'official_archive_sha256':digest(ROOT/'data/cifar-10-python.tar.gz'),
        'permutation_seed':260013,'corruption_seed':43,'svm_subset_seed':71})
    print('Verified sole split, training statistics, permutation and fixed subsets.',flush=True)

if __name__=='__main__': main()
