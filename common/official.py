"""Strict adapters to released Optimaler weights, with source labels preserved."""
import copy
import importlib
import sys
from pathlib import Path
import torch
from .io import ROOT,sha256,verify_optimaler

def load_official(method,seed,setting,device):
    if method not in {'RegretNet','RegretFormer'} or setting not in {'3x10','2x2'}:
        raise ValueError((method,setting))
    commit=verify_optimaler()
    vendor=ROOT/'vendor/optimaler'
    sys.path.insert(0,str(vendor))
    cfg=copy.deepcopy(importlib.import_module(f'run_configs.{method.lower()}_{setting}').cfg)
    module='core.nets.additive_net' if method=='RegretNet' else 'core.nets.additive_net_attention'
    cls_name='AdditiveNet' if method=='RegretNet' else 'AdditiveNetAttention'
    mod=importlib.import_module(module)
    if vendor.resolve() not in Path(mod.__file__).resolve().parents:
        raise RuntimeError('Author module resolved outside pinned vendor tree')
    model=getattr(mod,cls_name)(cfg,torch.device(device)).to(device)
    p=vendor/'target_nets'/method/f'setting_additive_{setting}_uniform'/f'seed_{seed}'/'model_200000'
    model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
    label='RegretNet_OptimalerBudget' if method=='RegretNet' else 'RegretFormer_AuthorsCheckpoint'
    return model,{'method':label,'source_role':'released_author_checkpoint','source_url':'https://github.com/dimonenka/optimaler',
        'source_commit':commit,'checkpoint_path':str(p),'checkpoint_sha256':sha256(p),
        'seed_role':'author_checkpoint_directory_label','seed':seed,'setting':setting,
        'n_bidders':int(setting.split('x')[0]),'n_items':int(setting.split('x')[1]),
        'training_protocol':'Optimaler budget-based author training; not original 2019 RegretNet training',
        'author_model_config':dict(cfg)}
