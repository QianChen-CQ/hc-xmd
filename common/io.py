import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def verify_optimaler():
    data=json.loads((ROOT/'OPTIMALER_MANIFEST.json').read_text())
    for rel,entry in data['files'].items():
        path=ROOT/'vendor/optimaler'/rel
        if not path.is_file() or sha256(path)!=entry['sha256']:
            raise ValueError(f'Original author file hash mismatch: {rel}')
    return data['commit']
