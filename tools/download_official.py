"""Optional ranged download of the very same official archive, MD5-verified.
Use only before prepare.py. Keeps partial bytes and resumes per-range files.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, urllib.request, shutil
p=Path('data/cifar-10-python.tar.gz'); size=170498071
if p.exists() and p.stat().st_size==size:
    assert hashlib.md5(p.read_bytes()).hexdigest()=='c58f30108f718f92721af3b95e74349a'; raise SystemExit('Already verified')
prefix=p.with_suffix('.prefix')
if not prefix.exists() and p.exists(): p.rename(prefix)
start=prefix.stat().st_size if prefix.exists() else 0
parts=[]; n=8; width=(size-start+n-1)//n
for i in range(n):
    lo=start+i*width; hi=min(size-1,lo+width-1)
    if lo<=hi: parts.append((i,lo,hi))
def fetch(spec):
    i,lo,hi=spec; dest=Path(f'data/official_part_{i}')
    have=dest.stat().st_size if dest.exists() else 0
    if have<hi-lo+1:
        request=urllib.request.Request('https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz',headers={'Range':f'bytes={lo+have}-{hi}'})
        with urllib.request.urlopen(request,timeout=120) as r, dest.open('ab') as f:
            assert r.status==206
            shutil.copyfileobj(r,f)
    assert dest.stat().st_size==hi-lo+1
    print('Verified length of range',i,flush=True)
    return dest
with ThreadPoolExecutor(max_workers=n) as pool: files=list(pool.map(fetch,parts))
with p.open('wb') as out:
    for source in ([prefix] if prefix.exists() else [])+files:
        with source.open('rb') as f: shutil.copyfileobj(f,out)
assert p.stat().st_size==size
assert hashlib.md5(p.read_bytes()).hexdigest()=='c58f30108f718f92721af3b95e74349a'
print('Official archive MD5 verified.',flush=True)
for source in ([prefix] if prefix.exists() else [])+files: source.unlink()
