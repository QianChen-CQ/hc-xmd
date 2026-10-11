#!/usr/bin/env python3
"""Run explicit, resumable HC-XMD revision tasks without changing old projects."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from common.io import ROOT,write_json,sha256
from common.evaluate import evaluate

def command(arguments,logfile):
    print('RUN:', ' '.join(map(str,arguments)),flush=True)
    logfile=Path(logfile);logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a') as log:
        process=subprocess.Popen([str(x) for x in arguments],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait()
    if code:raise RuntimeError(f'Command failed with exit {code}; see {logfile}')

def completed(path,key='state'):
    path=Path(path)
    if not path.exists():return False
    data=json.loads(path.read_text())
    return data.get(key,data.get('status'))=='COMPLETED'

def phase1_train(args,method,setting,seed,base):
    from phase1.train import preset_config
    folder=base/'training'/f'{method}_{setting}'/f'seed{seed}'
    expected=preset_config(method,'smoke' if args.mode=='smoke' else 'paper',setting);expected['seed']=seed
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=expected:
        raise ValueError(f'Training configuration mismatch at {folder}; select a fresh output directory')
    checkpoint=folder/'checkpoint.pt'
    if completed(folder/'status.json') and checkpoint.exists():
        print(f'REUSE completed training: {folder}',flush=True)
    else:
        cmd=[sys.executable,'-m','phase1.train','--method',method,'--setting',setting,'--seed',seed,
             '--preset','smoke' if args.mode=='smoke' else 'paper','--output',folder,'--device',args.device]
        if folder.exists() and any(folder.iterdir()):cmd.append('--resume')
        command(cmd,base/'logs'/f'train_{method}_{setting}_seed{seed}.log')
    return checkpoint

def phase1_load(checkpoint,seed):
    from phase1.models import load_checkpoint
    model,metadata=load_checkpoint(checkpoint,'cpu')
    cfg=metadata['config'];method=metadata['method']
    metadata.update(method=method+'_PaperReimplementation',source_role='paper_reimplementation',family='static_additive',
                    n_bidders=cfg['model']['n'],n_items=cfg['model']['m'],setting=cfg['setting'],seed=seed,
                    seed_role='new_training_seed',checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                    menu_size=cfg['model'].get('menu_size',''))
    metadata['source_url']={'RegretNet':'https://proceedings.mlr.press/v97/duetting19a.html',
                            'AMenuNet':'https://arxiv.org/html/2305.12162v3'}[method]
    return model,metadata

def run_eval(args,model,metadata,folder,**extras):
    folder=Path(folder)
    if completed(folder/'status.json'):
        old=json.loads((folder/'provenance.json').read_text())
        for key in ['checkpoint_sha256','transform_sha256']:
            if old.get(key)!=metadata.get(key):
                raise ValueError(f'Existing evaluation has different {key}: {folder}')
        print(f'REUSE completed evaluation: {folder}',flush=True);return
    if folder.exists() and any(folder.iterdir()):
        # Keep diagnostic evidence, then rerun the deterministic evaluation cleanly.
        index=1
        while folder.with_name(folder.name+f'_failed_attempt{index}').exists():index+=1
        folder.rename(folder.with_name(folder.name+f'_failed_attempt{index}'))
    return evaluate(model,metadata,folder,samples=args.eval_samples,batch_size=args.eval_batch_size,
                    device=args.device,smoke=args.mode=='smoke',**extras)

def phase1(args):
    for method in ['RegretNet','AMenuNet']:
        for seed in args.seeds:
            checkpoint=phase1_train(args,method,'3x10',seed,args.outdir)
            model,metadata=phase1_load(checkpoint,seed)
            run_eval(args,model,metadata,args.outdir/'evaluation'/f'{method}_3x10'/f'seed{seed}')

def phase2(args):
    from common.official import load_official
    from gemnet.model import load_checkpoint as load_gemnet
    if not args.phase1_dir.is_dir():
        raise FileNotFoundError(f'First finish Phase I with the same mode. Expected {args.phase1_dir}')
    for method in ['RegretNet','AMenuNet']:
        for seed in args.seeds:
            checkpoint=args.phase1_dir/'training'/f'{method}_3x10'/f'seed{seed}'/'checkpoint.pt'
            if not checkpoint.is_file():raise FileNotFoundError(f'Missing completed Phase I checkpoint: {checkpoint}')
            model,metadata=phase1_load(checkpoint,seed)
            if bool(metadata.get('is_smoke'))!=(args.mode=='smoke'):
                raise ValueError('Cannot mix smoke Phase I weights with formal Phase II')
            run_eval(args,model,metadata,args.outdir/'evaluation'/f'{method}_3x10'/f'seed{seed}')
    for setting in ['3x10','2x2']:
        for method in ['RegretNet','RegretFormer']:
            for seed in args.seeds:
                model,metadata=load_official(method,seed,setting,args.device)
                run_eval(args,model,metadata,args.outdir/'evaluation'/f'{metadata["method"]}_{setting}'/f'seed{seed}')
    for seed in args.seeds:
        checkpoint=phase1_train(args,'AMenuNet','2x2',seed,args.outdir)
        model,metadata=phase1_load(checkpoint,seed)
        run_eval(args,model,metadata,args.outdir/'evaluation'/'AMenuNet_2x2'/f'seed{seed}')
    for seed in args.seeds:
        train=args.outdir/'training'/'GemNet_2x2'/f'seed{seed}'
        checkpoint=train/'checkpoint.pt'
        if not (completed(train/'train_status.json','status') and checkpoint.is_file()):
            cmd=[sys.executable,'-m','gemnet.run','train','--out',train,'--setting','2x2','--seed',seed,
                 '--preset','smoke' if args.mode=='smoke' else 'paper','--device',args.device]
            if train.exists() and any(train.iterdir()):cmd.append('--resume')
            command(cmd,args.outdir/'logs'/f'train_GemNet_2x2_seed{seed}.log')
        transform=args.outdir/'transforms'/'GemNet_2x2'/f'seed{seed}'
        transform_file=transform/'price_transform.npz'
        if not (completed(transform/'transform_status.json','status') and transform_file.is_file()):
            cmd=[sys.executable,'-m','gemnet.run','transform','--checkpoint',checkpoint,'--out',transform,
                     '--device',args.device,'--grid-points',3 if args.mode=='smoke' else 7,
                     '--diagnostic-samples',16 if args.mode=='smoke' else 2048]
            if transform.exists() and any(transform.iterdir()):cmd.append('--resume')
            command(cmd,args.outdir/'logs'/f'transform_GemNet_2x2_seed{seed}.log')
        obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=obj['model_config']
        diag=json.loads((transform/'transform_diagnostics.json').read_text())
        metadata={'method':'GemNet_PaperReimplementation','source_role':'paper_reimplementation_with_price_transform',
                  'source_url':'https://arxiv.org/html/2406.07428v3','family':'static_additive','setting':'2x2',
                  'seed':seed,'seed_role':'new_training_seed','n_bidders':2,'n_items':2,'menu_size':cfg['menu_size'],
                  'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
                  'transform_path':str(transform_file),'transform_sha256':sha256(transform_file),
                  'transform_diagnostics':diag,'global_continuous_certificate':False,
                  'training_config':obj.get('training_config',{}),'model_config':cfg,
                  'selected_checkpoint_step':obj.get('step'),'is_smoke':args.mode=='smoke',
                  'reimplementation_scope':'paper architecture, loss and MILP with documented screening; finite tests do not certify continuous feasibility'}
        model=load_gemnet(checkpoint,args.device,transform_file)
        run_eval(args,model,metadata,args.outdir/'evaluation'/'GemNet_2x2'/f'seed{seed}')

def phase3(args):
    from phase3_author.bridge import load_checkpoint,ContextBatchModel
    import numpy as np
    methods=['CITransNet','CITransNetNoContext','RegretNetPreferenceBaseline','PreferenceNet',
             'RegretNetProportionBaseline','ProportionNet']
    for method in methods:
        for seed in args.seeds:
            train=args.outdir/'training'/method/f'seed{seed}'
            checkpoint=train/'checkpoint.pt'
            if not (completed(train/'status.json') and checkpoint.is_file()):
                cmd=[sys.executable,'-m','phase3_author.train','--method',method,'--seed',seed,
                     '--output-dir',train,'--mode','smoke' if args.mode=='smoke' else 'full','--device',args.device,
                     '--eval-samples',args.eval_samples,'--n-agents',3 if method.startswith('CITransNet') else 2,
                     '--n-items',10 if method.startswith('CITransNet') else 2]
                if train.exists() and any(train.iterdir()):cmd.append('--resume')
                command(cmd,args.outdir/'logs'/f'train_{method}_seed{seed}.log')
            model,metadata=load_checkpoint(checkpoint,'cpu')
            metadata.update(seed=seed,seed_role='new_training_seed',checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
            # Detailed method-specific data/metric adapters are provided by the audited bridge.
            from phase3_author.bridge import evaluation_extras
            metadata,extras=evaluation_extras(model,metadata,train,args.device,args.eval_samples,args.mode=='smoke')
            run_eval(args,model,metadata,args.outdir/'evaluation'/method/f'seed{seed}',**extras)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['phase1','phase2','phase3'],required=True)
    p.add_argument('--mode',choices=['smoke','run'],default='run')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--phase1-dir',type=Path)
    p.add_argument('--device',default='auto')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--eval-samples',type=int,default=512)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args(argv)
    args.outdir=args.outdir.resolve();args.outdir.mkdir(parents=True,exist_ok=True)
    if args.phase1_dir is None:
        args.phase1_dir=Path('/root/hc_xmd_phase1_v5_'+('smoke' if args.mode=='smoke' else 'results'))
    args.phase1_dir=args.phase1_dir.resolve()
    args.seeds=list(dict.fromkeys(args.seeds))
    if any(seed not in [0,1,2] for seed in args.seeds):p.error('Bundled author checkpoint labels are 0/1/2; use those seeds for this suite')
    if args.eval_samples<2 or args.eval_batch_size<1:p.error('Need at least two test profiles and a positive batch size')
    args.device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(args.cpu_threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    plan={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    plan['source_sha256']={str(f.relative_to(ROOT)):sha256(f) for f in sorted(ROOT.rglob('*.py')) if 'vendor' not in f.parts and '__pycache__' not in f.parts}
    if (args.outdir/'RUN_PLAN.json').exists():
        old=json.loads((args.outdir/'RUN_PLAN.json').read_text())
        if old!=plan:raise ValueError('Existing run plan/code differs. Preserve old results and choose a new --outdir.')
    else:write_json(args.outdir/'RUN_PLAN.json',plan)
    state={'state':'RUNNING','phase':args.phase,'scope':'smoke_only' if args.mode=='smoke' else 'paper_reference_training_and_shared_evaluation'}
    write_json(args.outdir/'suite_status.json',state)
    began=time.time()
    try:
        {'phase1':phase1,'phase2':phase2,'phase3':phase3}[args.phase](args)
        if args.phase=='phase2':
            command([sys.executable,'-m','common.decision_slices','--root',args.outdir,'--device',args.device],args.outdir/'logs/slices.log')
        command([sys.executable,'-m','common.export_results','--root',args.outdir,'--phase',args.phase],args.outdir/'logs/export.log')
        expected={'phase1':2,'phase2':8,'phase3':6}[args.phase]*len(args.seeds)
        finished=[p for p in (args.outdir/'evaluation').glob('*/*/status.json') if completed(p)]
        if len(finished)!=expected:raise RuntimeError(f'Expected {expected} evaluations, found {len(finished)}')
        state.update(state='COMPLETED',wall_seconds=time.time()-began,completed_evaluations=len(finished),expected_evaluations=expected)
        write_json(args.outdir/'suite_status.json',state)
        print(f'SUITE COMPLETED: {args.outdir}',flush=True)
    except BaseException as exc:
        state.update(state='FAILED_OR_INTERRUPTED',error=str(exc),wall_seconds=time.time()-began)
        write_json(args.outdir/'suite_status.json',state)
        (args.outdir/'suite_error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
