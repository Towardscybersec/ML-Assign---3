"""Shared deterministic setup and artifact I/O. No test data are loaded here."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/experiment.json').read_text())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
    tmp.replace(path)

def atomic_torch(path, obj):
    path=Path(path); tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def load_ckpt(path):
    return torch.load(path, map_location=DEVICE, weights_only=False)

def hardware():
    import platform
    return {'python':platform.python_version(),'torch':torch.__version__, 'device':str(DEVICE),
            'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
            'cpu':platform.processor(), 'deterministic':True, 'tf32':False}

def load_development():
    from torchvision.datasets import CIFAR10
    ds=CIFAR10(ROOT/'data',train=True,download=False)
    ix=np.load(ROOT/'artifacts/splits/split.npz')
    raw=torch.from_numpy(ds.data).permute(0,3,1,2).to(DEVICE)
    y=torch.tensor(ds.targets,device=DEVICE)
    return raw[ix['train']],y[ix['train']],raw[ix['val']],y[ix['val']]

class Images:
    def __init__(self):
        stats=json.loads((ROOT/'artifacts/splits/statistics.json').read_text())
        self.mean=torch.tensor(stats['mean'],device=DEVICE)[None,:,None,None]
        self.std=torch.tensor(stats['std'],device=DEVICE)[None,:,None,None]
        self.perm=torch.tensor(np.load(ROOT/'artifacts/splits/permutation.npy'),device=DEVICE)
    def pixels(self,x,condition='clean'):
        x=x.float()/255 if x.dtype==torch.uint8 else x
        if condition=='permuted': x=x.flatten(2)[:,:,self.perm].reshape(-1,3,32,32)
        elif condition=='grayscale':
            x=(x*torch.tensor([.2989,.5870,.1140],device=x.device)[None,:,None,None]).sum(1,keepdim=True).repeat(1,3,1,1)
        elif condition!='clean': raise ValueError(condition)
        return x
    def normalise(self,x): return (x-self.mean)/self.std
    def __call__(self,x,condition='clean'): return self.normalise(self.pixels(x,condition))

def corrupt(x, seed, sigma=None):
    g=torch.Generator(device=x.device).manual_seed(seed)
    return (x+torch.randn(x.shape,device=x.device,generator=g)*(sigma or CFG['autoencoder']['noise_sigma'])).clamp(0,1)

@torch.no_grad()
def predictions(model, raw, image_transform, condition='clean', features=False, batch_size=256):
    model.eval(); out=[]
    for s in range(0,len(raw),batch_size):
        x=image_transform(raw[s:s+batch_size],condition)
        out.append((model.features(x) if features else model(x)).cpu().numpy())
    return np.concatenate(out)

def metrics(logits,y):
    from scipy.special import softmax, logsumexp
    from sklearn.metrics import confusion_matrix
    y=np.asarray(y); p=softmax(logits,axis=1); pred=p.argmax(1); conf=p.max(1); hit=pred==y
    ece=0.; bins=[]
    for i in range(15):
        mask=(conf>=i/15)&((conf<(i+1)/15) if i<14 else (conf<=1))
        n=int(mask.sum()); acc=float(hit[mask].mean()) if n else 0.; c=float(conf[mask].mean()) if n else 0.
        ece+=n/len(y)*abs(acc-c); bins.append({'n':n,'accuracy':acc,'confidence':c})
    return {'accuracy':float(hit.mean()),'nll':float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])),
            'brier':float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))), 'ece':float(ece),
            'incorrect_confidence':float(conf[~hit].mean()) if (~hit).any() else None,
            'confusion':confusion_matrix(y,pred,labels=np.arange(10)).tolist(),'reliability':bins}
