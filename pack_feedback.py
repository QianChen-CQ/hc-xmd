#!/usr/bin/env python3
"""Pack one phase's evidence; exclude restart caches and deduplicate raw inputs."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from common.io import ROOT,sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);args=p.parse_args()
    root=args.outdir.resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    target=root.parent/(root.name+'_feedback.tar.gz');temp=target.with_name(target.name+'.tmp')
    include=[];excluded=[];manifest={};seen={};duplicates={}
    skip_names={'train_values.npy','resume_latest.pt','training_state.pt','checkpoint_last.pt','model_final.pt','FEEDBACK_MANIFEST.json','FEEDBACK_PACKAGE.json','DEDUPLICATION_MAP.json'}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        rel=str(path.relative_to(root))
        if path.name in skip_names or path.name.startswith('model_step') or '__pycache__' in path.parts:
            excluded.append(rel);continue
        digest=sha256(path);manifest[rel]={'bytes':path.stat().st_size,'sha256':digest}
        if digest in seen and path.stat().st_size>1024:
            duplicates[rel]=seen[digest]
        else:
            seen[digest]=rel;include.append(path)
    package_files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts
                   and ('vendor' not in p.relative_to(ROOT).parts) and p.suffix.lower() in {'.py','.sh','.txt','.md','.json','.csv'}]
    with tarfile.open(temp,'w:gz') as tar:
        for path in include:tar.add(path,arcname=root.name+'/'+str(path.relative_to(root)),recursive=False)
        for path in package_files:tar.add(path,arcname=root.name+'/evaluation_code/'+str(path.relative_to(ROOT)),recursive=False)
        meta={'files':manifest,'excluded_restart_or_duplicate_checkpoints':excluded,
              'note':'Identical large file bytes are stored once. DEDUPLICATION_MAP maps absent paths to retained paths; original results are unchanged.'}
        for name,data in [('FEEDBACK_MANIFEST.json',meta),('DEDUPLICATION_MAP.json',duplicates)]:
            blob=(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode();info=tarfile.TarInfo(root.name+'/'+name);info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
    temp.replace(target)
    write_json(root/'FEEDBACK_PACKAGE.json',{'archive':str(target),'bytes':target.stat().st_size,'sha256':sha256(target),
                'unique_result_files':len(include),'deduplicated_files':len(duplicates),'excluded_restart_files':excluded})
    print(f'UPLOAD THIS FILE: {target}',flush=True)

if __name__=='__main__':main()
