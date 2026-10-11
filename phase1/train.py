#!/usr/bin/env python3
"""Train reconstructed Phase I models with explicit paper/smoke provenance.

Run from package root: python -m phase1.train --method RegretNet --preset paper
--seed 0 --output /root/hc_xmd_phase1_v5_results/RegretNet/seed0 --device auto
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch

from .models import create_model

FORMAT = 'HCXMD_PHASE1_V5'
SOURCES = {
    'RegretNet': {
        'paper': 'https://proceedings.mlr.press/v97/duetting19a.html',
        'repository': 'https://github.com/saisrivatsan/deep-opt-auctions',
        'commit': '466e154b123921c142ee811216d0e64415bf31a5',
        'implementation': 'New PyTorch port of original 2019 additive architecture and augmented-Lagrangian algorithm; not original TF code or weights',
    },
    'AMenuNet': {
        'paper': 'https://papers.nips.cc/paper_files/paper/2023/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html',
        'repository': 'https://github.com/Haoran0301/AMenuNet',
        'commit': 'a62ec44e9280ec7a87313d1b8036ccfa5402091e',
        'implementation': 'Independent NeurIPS 2023 paper implementation, architecture and AMA equations cross-checked against author code; not author weights',
    },
}


def preset_config(method, preset='paper', setting='3x10'):
    if setting not in ('3x10', '2x2'):
        raise ValueError(f'Unsupported setting: {setting}')
    n, m = map(int, setting.split('x'))
    common = {'distribution': 'iid Uniform[0,1], additive', 'preset': preset,
              'setting': setting, 'model': {'n': n, 'm': m}, 'seed': None}
    if method == 'RegretNet':
        common['model'].update(hidden=100, hidden_layers=5 if setting == '3x10' else 2)
        common.update(steps=400000, batch_size=128, train_profiles=640000,
                      learning_rate=0.001, inner_steps=25, inner_lr=0.1,
                      lambda_initial=5.0, rho_initial=1.0,
                      lambda_frequency=100, rho_frequency=10000, rho_increment=1.0 if setting == '3x10' else 5.0,
                      save_every=50000, log_every=1000)
        if preset == 'smoke':
            common.update(steps=4, batch_size=8, train_profiles=32,
                          inner_steps=3, lambda_frequency=2, rho_frequency=4,
                          save_every=2, log_every=1)
    elif method == 'AMenuNet':
        # Appendix B: embeddings16; author experiments.py defaults10. Explicitly
        # choose paper16 here, do not silently claim identical author defaults.
        common['model'].update(menu_size=512 if setting == '3x10' else 128, embed_dim=16, width=64, layers=3,
                               heads=4, allocation_temperature=10.0,
                               choice_temperature=500.0)
        common.update(steps=2000, batch_size=2048, profiles_per_round=32768,
                      learning_rate=0.0003, warmup_start=1e-8, warmup_rounds=100,
                      save_every=100, log_every=1)
        if setting == '2x2':
            common['new_setting_extension'] = True
            common['extension_note'] = '2x2 iidU01 not an AMenuNet Table3(C) column; transfer menu128/temp10 from2x5(C), use2000rounds, no exact original-row claim'
        if preset == 'smoke':
            # Keep paper architecture; only training budget changes.
            common.update(steps=2, batch_size=8, profiles_per_round=16,
                          save_every=1, log_every=1)
    else:
        raise ValueError(method)
    return common


def json_write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def torch_write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def unilateral_utilities(model, values, reports):
    """For every bidder, independently replace ONLY that bidder's report.

    values/reports: [batch,bidders,items]. Return each deviator's own utility,
    valued using the unchanged true values: [batch,bidders].
    """
    batch, bidders, items = values.shape
    diagonal = torch.eye(bidders, device=values.device).view(1, bidders, bidders, 1)
    submitted = values[:, None]*(1-diagonal)+reports[:, None]*diagonal
    allocation, payment = model(submitted.reshape(batch*bidders, bidders, items))
    allocation = allocation.reshape(batch, bidders, bidders, items)
    payment = payment.reshape(batch, bidders, bidders)
    idx = torch.arange(bidders, device=values.device)
    return (allocation[:, idx, idx]*values).sum(dim=-1)-payment[:, idx, idx]


def optimize_reports(model, values, initial_reports, steps, lr):
    """Projected Adam adversary, slots reset for each training minibatch."""
    reports = torch.nn.Parameter(initial_reports.detach().clone())
    adversary = torch.optim.Adam([reports], lr=lr)
    # Differentiate w.r.t. reports only; no network-gradient accumulation.
    for _ in range(steps):
        utility = unilateral_utilities(model, values, reports)
        gradient, = torch.autograd.grad(-utility.sum(), reports)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Non-finite misreport gradient')
        reports.grad = gradient
        adversary.step()
        with torch.no_grad():
            reports.clamp_(0.0, 1.0)
    return reports.detach()


def batch_objective(model, values, reports, lagrange, rho):
    allocation, payments = model(values)
    truth_utility = (allocation*values).sum(dim=-1)-payments
    deviation_utility = unilateral_utilities(model, values, reports)
    regret_by_bidder = (deviation_utility-truth_utility).clamp_min(0).mean(dim=0)
    revenue = payments.sum(dim=1).mean()
    penalty = 0.5*rho*regret_by_bidder.square().sum()
    lag_loss = (lagrange*regret_by_bidder).sum()
    return -revenue+penalty+lag_loss, revenue, regret_by_bidder


def rng_state():
    return {'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_model(out, model, config, method, step):
    payload = {'format': FORMAT, 'method': method, 'step': step, 'config': config,
               'source': SOURCES[method], 'model_state': model.state_dict(),
               'is_smoke': config['preset'] == 'smoke',
               'training_complete': step == config['steps']}
    name = f'model_step{step:07d}.pt'
    torch_write(out/name, payload)
    if step == config['steps']:
        torch_write(out/'model_final.pt', payload)
        torch_write(out/'checkpoint.pt', payload)
    return name


def append_log(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_regretnet(model, cfg, out, device, resume):
    count, n, m = cfg['train_profiles'], cfg['model']['n'], cfg['model']['m']
    # Local data RNG separates model initialization from synthetic data generation.
    data_rng = np.random.RandomState(cfg['seed'])
    values = data_rng.rand(count, n, m).astype('float32')
    initial_adv = data_rng.rand(count, n, m).astype('float32')
    values_tensor = torch.from_numpy(values)
    cache = torch.from_numpy(initial_adv)
    data_hash = hashlib.sha256(values.tobytes()).hexdigest()
    json_write(out/'training_data.json', {'generator': 'numpy.RandomState(seed).rand cast float32',
        'seed': cfg['seed'], 'shape': list(values.shape), 'sha256_raw_float32': data_hash,
        'distribution': cfg['distribution'], 'values_saved': 'train_values.npy'})
    if not (out/'train_values.npy').exists():
        np.save(out/'train_values.npy', values)
    else:
        saved_values = np.load(out/'train_values.npy', mmap_mode='r', allow_pickle=False)
        if saved_values.shape != values.shape or hashlib.sha256(saved_values.tobytes()).hexdigest() != data_hash:
            raise ValueError('Saved training data does not match its deterministic seed/configuration')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    lagrange = torch.full((n,), cfg['lambda_initial'], device=device)
    rho, start, cursor = cfg['rho_initial'], 0, count
    order = torch.arange(count)
    # The sequence of permutations is fully captured by this independent RNG.
    shuffle_rng = torch.Generator().manual_seed(cfg['seed']+101)
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer_state'])
        cache = resume['misreport_cache']
        lagrange = resume['lagrange'].to(device)
        rho, start, cursor = resume['rho'], resume['step'], resume['cursor']
        order = resume['order']
        shuffle_rng.set_state(resume['shuffle_rng'])
        restore_rng(resume['rng'])
    began = time.monotonic()
    for step in range(start+1, cfg['steps']+1):
        if cursor >= count:
            order = torch.randperm(count, generator=shuffle_rng)
            cursor = 0
        indices = order[cursor:cursor+cfg['batch_size']]
        cursor += indices.numel()
        x = values_tensor[indices].to(device)
        reports = optimize_reports(model, x, cache[indices].to(device),
                                   cfg['inner_steps'], cfg['inner_lr'])
        cache[indices] = reports.cpu()
        optimizer.zero_grad(set_to_none=True)
        loss, revenue, regrets = batch_objective(model, x, reports, lagrange, rho)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite RegretNet objective')
        constraint_quadratic = float((0.5*rho*regrets.square().sum()).detach())
        constraint_lagrangian = float((lagrange*regrets).sum().detach())
        rho_used = rho
        loss.backward()
        optimizer.step()
        if step % cfg['lambda_frequency'] == 0:
            with torch.no_grad():
                # Original source updates lambda after the network optimizer step.
                _, _, after_update_regrets = batch_objective(model, x, reports, lagrange, rho)
                lagrange += rho*after_update_regrets
        if step % cfg['rho_frequency'] == 0:
            rho += cfg['rho_increment']
        if step % cfg['log_every'] == 0 or step == cfg['steps']:
            row = {'step': step, 'optimizer_updates': step, 'sample_exposures': step*cfg['batch_size'],
                   'revenue_train_batch': float(revenue.detach()),
                   'regret_bidder_mean_train_batch': float(regrets.mean().detach()),
                   'loss': float(loss.detach()), 'constraint_quadratic': constraint_quadratic,
                   'constraint_lagrangian': constraint_lagrangian, 'rho_used': rho_used, 'lambda_mean': float(lagrange.mean()),
                   'rho': rho, 'elapsed_this_process_seconds': time.monotonic()-began}
            append_log(out/'train_log.csv', row)
            print(json.dumps(row), flush=True)
        if step % cfg['save_every'] == 0 or step == cfg['steps']:
            save_model(out, model, cfg, 'RegretNet', step)
            torch_write(out/'resume_latest.pt', {'step': step, 'format': FORMAT,
                'method': 'RegretNet', 'config': cfg, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'misreport_cache': cache,
                'lagrange': lagrange.cpu(), 'rho': rho, 'cursor': cursor, 'order': order,
                'shuffle_rng': shuffle_rng.get_state(), 'rng': rng_state()})
            json_write(out/'progress.json', {'state': 'TRAINING', 'method': 'RegretNet',
                'step': step, 'target': cfg['steps'], 'optimizer_updates': step})
    return cfg['steps']


def train_amenu(model, cfg, out, device, resume):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['warmup_start'])
    sample_rng = torch.Generator(device='cpu').manual_seed(cfg['seed']+202)
    start = 0
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer_state'])
        sample_rng.set_state(resume['sample_rng'])
        restore_rng(resume['rng'])
        start = resume['step']
    batches = cfg['profiles_per_round']//cfg['batch_size']
    began = time.monotonic()
    for step in range(start+1, cfg['steps']+1):
        # Match the intended 100-round linear warmup, correcting public code's
        # 101st increment. At round 1 use1e-8, round101 reaches target3e-4.
        ratio = min((step-1)/cfg['warmup_rounds'], 1.0)
        learning_rate = cfg['warmup_start']+ratio*(cfg['learning_rate']-cfg['warmup_start'])
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        x_all = torch.rand(cfg['profiles_per_round'], cfg['model']['n'], cfg['model']['m'], generator=sample_rng)
        total_loss, total_revenue = 0.0, 0.0
        for batch in x_all.split(cfg['batch_size']):
            x = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, payment = model.training_forward(x)
            loss = -payment.sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite AMenuNet objective')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_revenue += float(payment.detach().sum(dim=1).mean())
        if step % cfg['log_every'] == 0 or step == cfg['steps']:
            row = {'step': step, 'optimizer_updates': step*batches,
                   'sample_exposures': step*cfg['profiles_per_round'],
                   'surrogate_revenue_train_round': total_revenue/batches,
                   'loss': total_loss/batches, 'learning_rate': learning_rate,
                   'elapsed_this_process_seconds': time.monotonic()-began}
            append_log(out/'train_log.csv', row)
            print(json.dumps(row), flush=True)
        if step % cfg['save_every'] == 0 or step == cfg['steps']:
            save_model(out, model, cfg, 'AMenuNet', step)
            torch_write(out/'resume_latest.pt', {'step': step, 'format': FORMAT,
                'method': 'AMenuNet', 'config': cfg, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'sample_rng': sample_rng.get_state(),
                'rng': rng_state()})
            json_write(out/'progress.json', {'state': 'TRAINING', 'method': 'AMenuNet',
                'step': step, 'target': cfg['steps'], 'optimizer_updates': step*batches})
    return cfg['steps']*batches


def rollback_log(path, saved_step):
    """Remove rows for updates lost after the last durable checkpoint."""
    if not path.exists():
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = [row for row in reader if int(row['step']) <= saved_step]
    if not columns:
        return
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['RegretNet', 'AMenuNet'], required=True)
    parser.add_argument('--preset', choices=['paper', 'smoke'], default='paper')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--setting', choices=['3x10', '2x2'], default='3x10')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, help='Override updates (RegretNet) / outer rounds (AMenuNet); logged as custom budget')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = preset_config(args.method, args.preset, args.setting)
    cfg['seed'] = args.seed
    if args.steps is not None:
        if args.steps < 1:
            parser.error('--steps must be positive')
        cfg['steps'] = args.steps
        cfg['custom_budget'] = True
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        cfg['batch_size'] = args.batch_size
        cfg['custom_batch_size'] = True
    if args.method == 'AMenuNet' and cfg['profiles_per_round'] % cfg['batch_size']:
        parser.error('AMenuNet batch size must divide profiles_per_round')
    if args.method == 'RegretNet' and cfg['train_profiles'] % cfg['batch_size']:
        parser.error('RegretNet batch size must divide train_profiles')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'{out} is not empty; pass --resume to resume its own checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    resume = None
    if args.resume and (out/'resume_latest.pt').exists():
        resume = torch.load(out/'resume_latest.pt', weights_only=True, map_location='cpu')
        if resume.get('format') != FORMAT or resume.get('method') != args.method or resume.get('config') != cfg:
            raise ValueError('Resume method/configuration mismatch; reuse the same command or choose a new output')
    elif args.resume and any(out.iterdir()):
        # Failure before first checkpoint: permit restart only for matching cfg.
        existing = out/'config.json'
        if not existing.exists() or json.loads(existing.read_text()) != cfg:
            raise ValueError('Existing directory has no compatible configuration/checkpoint')
    if args.resume:
        rollback_log(out/'train_log.csv', 0 if resume is None else resume['step'])
        history_file = out/'resume_history.json'
        history = json.loads(history_file.read_text()) if history_file.exists() else []
        previous_provenance = json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else {}
        history.append({'resumed_step': 0 if resume is None else resume['step'],
                        'previous_source_sha256': previous_provenance.get('source_sha256', {}),
                        'source_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')}})
        json_write(history_file, history)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = create_model(args.method, cfg['model']).to(device).train()
    if resume is not None:
        model.load_state_dict(resume['model_state'], strict=True)
    json_write(out/'config.json', cfg)
    json_write(out/'provenance.json', {'format': FORMAT, 'method': args.method,
        'source': SOURCES[args.method], 'historical_recovery': False,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'device': device, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if device.startswith('cuda') else None},
        'source_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'notes': 'See REPRODUCTION_NOTES.md for source conflicts, deliberate runtime adaptations and evaluation scope.'})
    json_write(out/'status.json', {'state': 'TRAINING', 'method': args.method,
        'preset': args.preset, 'seed': args.seed, 'is_smoke': args.preset == 'smoke'})
    try:
        updates = (train_regretnet if args.method == 'RegretNet' else train_amenu)(model, cfg, out, device, resume)
        final = out/'model_final.pt'
        if not final.exists():
            raise RuntimeError('Training loop did not produce model_final.pt')
        json_write(out/'status.json', {'state': 'COMPLETED', 'method': args.method,
            'preset': args.preset, 'seed': args.seed, 'is_smoke': args.preset == 'smoke',
            'custom_budget': bool(cfg.get('custom_budget', False)), 'training_steps': cfg['steps'],
            'optimizer_updates': updates, 'checkpoint': str(final), 'checkpoint_sha256': sha256_file(final),
            'evaluation_state': 'NOT_RUN_USE_SHARED_EVALUATOR'})
        print(f'TRAINING_COMPLETED {final}', flush=True)
    except BaseException as exc:
        json_write(out/'status.json', {'state': 'FAILED_OR_INTERRUPTED', 'method': args.method,
            'preset': args.preset, 'seed': args.seed, 'error': str(exc),
            'resume_checkpoint_exists': (out/'resume_latest.pt').exists()})
        (out/'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
