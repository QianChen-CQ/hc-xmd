"""Train, transform and inspect the GemNet paper reimplementation."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .model import GemNet, load_checkpoint, PAPER_URL
from .transform import transform_model


def atomic_torch_save(data, path):
    path = Path(path)
    torch.save(data, path.with_suffix(path.suffix+'.tmp'))
    os.replace(path.with_suffix(path.suffix+'.tmp'), path)


def resolve_device(value):
    if value == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if value.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return value


def finite_metrics(values, allocations, payments, tolerance=1e-6):
    revenue = payments.sum(1)
    welfare = (values*allocations).sum((1, 2))
    utility = (values*allocations).sum(2)-payments
    excess = np.maximum(allocations.sum(1)-1, 0)
    return dict(n_profiles=len(values), revenue_mean=float(revenue.mean()),
                revenue_sample_std=float(revenue.std(ddof=1)) if len(revenue)>1 else 0.0,
                welfare_mean=float(welfare.mean()),
                incompatibility_rate=float((excess.max(1)>tolerance).mean()),
                allocation_excess_max=float(excess.max()),
                ir_violation_max=float(np.maximum(-utility, 0).max()),
                payment_negative_max=float(np.maximum(-payments, 0).max()),
                tolerance=tolerance,
                feasible_on_evaluated_profiles=bool(excess.max()<=tolerance),
                global_dsic_certified=False)


def predict(model, values, device, batch_size=256):
    allocations, payments = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            v = torch.as_tensor(values[start:start+batch_size], dtype=torch.float32, device=device)
            a, p = model(v)
            allocations.append(a.cpu().numpy()); payments.append(p.cpu().numpy())
    a, p = np.concatenate(allocations), np.concatenate(payments)
    if not (np.isfinite(a).all() and np.isfinite(p).all()):
        raise RuntimeError('Nonfinite model output')
    return a, p


def train(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Training output exists; choose a fresh directory to preserve old checkpoints')
    out.mkdir(parents=True, exist_ok=True)
    n, m = map(int, args.setting.lower().split('x'))
    preset = dict(hidden=1024, menu_size=1000 if m>5 else 300,
                  steps=20000, batch_size=8192, validation_samples=50000,
                  validate_every=100, lambda_update_every=1)
    if args.preset == 'smoke':
        preset.update(hidden=16, menu_size=5, steps=4, batch_size=16,
                      validation_samples=64, validate_every=2)
    for key in preset:
        value = getattr(args, key, None)
        if value is not None:
            preset[key] = value
    if min(preset['steps'], preset['batch_size'], preset['validation_samples'], preset['validate_every'])<1:
        raise ValueError('Training dimensions and steps must be positive')
    device = resolve_device(args.device)
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GemNet(n, m, preset['menu_size'], preset['hidden'], args.price_activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    validation = np.random.default_rng(seed+1000003).random((preset['validation_samples'], n, m)).astype(np.float32)
    config = dict(preset=preset, model_config=model.config, seed=seed, device=device,
                  setting=args.setting, distribution='independent Uniform[0,1], additive',
                  paper_url=PAPER_URL, origin='paper_based_reimplementation_not_author_code',
                  training_safety_margin=args.training_safety_margin,
                  violation_threshold=args.violation_threshold,
                  learning_rate=args.learning_rate, price_activation=args.price_activation,
                  torch_version=str(torch.__version__), numpy_version=np.__version__,
                  paper_ambiguities={'price_head': 'Paper diagram is linear, appendix assumes nonnegative prices; relu is the explicit default choice, linear is selectable.',
                                    'incompatibility_lambda_update_frequency': f'Paper does not state frequency; this replication uses every {preset["lambda_update_every"]} optimizer steps.',
                                    'training_safety_margin': 'Explicit configured value; transformation computes its own norm-derived margin.',
                                    'checkpoint_selection': 'Independent validation profiles; final test profiles are not used for selection.',
                                    'lr_decay': 'When adjacent validation revenue/rate deltas meet paper thresholds after two checks, multiply lr by 0.1 each 2000 subsequent steps.'},
                  inference='hard argmax; fixed zero-price null option; no cross-bidder normalization')
    status_path = out/'train_status.json'
    lamb = 0.1
    best_criterion = None
    best_meta = None
    val_history = []
    decay_start = None
    initial_step = 0
    elapsed_before_resume = 0.0
    if args.resume:
        if not (out/'checkpoint_last.pt').is_file():
            raise FileNotFoundError('--resume needs checkpoint_last.pt with optimizer and RNG state')
        saved = torch.load(out/'checkpoint_last.pt', map_location='cpu', weights_only=True)
        previous_config = saved['training_config']
        for key in ('model_config','seed','setting','training_safety_margin','violation_threshold','learning_rate','price_activation','device'):
            if previous_config[key] != config[key]:
                raise ValueError(f'Cannot resume with changed {key}')
        for key in preset:
            if key != 'steps' and previous_config['preset'][key] != preset[key]:
                raise ValueError(f'Cannot resume with changed preset {key}')
        required = ('best_criterion','best_meta','val_history','decay_start','cuda_rng_states','elapsed_seconds')
        if not all(key in saved for key in required):
            raise ValueError('This checkpoint predates complete resume-state support; it cannot be called an exact continuation')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.set_rng_state(saved['torch_rng_state'])
        if device.startswith('cuda'):
            if not saved['cuda_rng_states']:
                raise ValueError('No saved CUDA RNG states for exact CUDA resume')
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        lamb = saved['lambda_incompatibility']
        initial_step = saved['step']
        best_criterion = tuple(saved['best_criterion'])
        best_meta, val_history, decay_start = saved['best_meta'], saved['val_history'], saved['decay_start']
        elapsed_before_resume = saved['elapsed_seconds']
        validation = np.load(out/'validation_values.npy', allow_pickle=False)
        # A crash can leave non-checkpointed rows in text logs; retain them in a
        # separate file and continue the canonical log from the restored step.
        for file_name, kind in [('training.csv','csv'),('validation.jsonl','jsonl')]:
            path = out/file_name
            if not path.exists():
                continue
            original = path.read_text()
            if kind == 'csv':
                lines = original.splitlines()
                rows = list(csv.DictReader(lines))
                kept = [row for row in rows if int(row['step'])<=initial_step]
                if len(kept) != len(rows):
                    path.with_suffix(path.suffix+'.uncheckpointed').write_text(original)
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(kept)
            else:
                rows = [json.loads(line) for line in original.splitlines() if line.strip()]
                kept = [row for row in rows if row['step']<=initial_step]
                if len(kept) != len(rows):
                    path.with_suffix(path.suffix+'.uncheckpointed').write_text(original)
                    path.write_text(''.join(json.dumps(row)+'\n' for row in kept))
        with (out/'resume_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(restored_step=initial_step, requested_steps=preset['steps'], device=device))+'\n')
    else:
        np.save(out/'validation_values.npy', validation)
    (out/'train_config.json').write_text(json.dumps(config, indent=2))
    status_path.write_text(json.dumps({'status': 'RUNNING', 'restored_step': initial_step}))
    start_time = time.monotonic()
    log_file = out/'training.csv'
    with log_file.open('a' if args.resume else 'w', newline='') as stream:
        fields = ['step','loss','revenue_soft','incompatibility_loss','lambda_incompatibility',
                  'softmax_temperature','learning_rate','elapsed_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        for step in range(initial_step+1, preset['steps']+1):
            model.train()
            values = torch.rand(preset['batch_size'], n, m, device=device)
            temp = min(2000.0, 5.0*2**((step-1)//3000))
            allocation, payment = model(values, temperature=temp)
            revenue = payment.sum(1).mean()
            incompatible = (allocation.sum(1)-(1-args.training_safety_margin)).relu().sum(1).mean()
            loss = -revenue + lamb*incompatible
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            row = dict(step=step, loss=float(loss.detach()), revenue_soft=float(revenue.detach()),
                       incompatibility_loss=float(incompatible.detach()), lambda_incompatibility=lamb,
                       softmax_temperature=temp, learning_rate=optimizer.param_groups[0]['lr'],
                       elapsed_seconds=elapsed_before_resume+time.monotonic()-start_time)
            writer.writerow(row)
            if step % preset['lambda_update_every'] == 0:
                lamb += max(.01, float(incompatible.detach()))
            if decay_start is not None:
                factor = .1**((step-decay_start)//2000)
                for group in optimizer.param_groups:
                    group['lr'] = args.learning_rate*factor
            if step % preset['validate_every'] == 0 or step == preset['steps']:
                stream.flush()
                a, p = predict(model, validation, device, batch_size=args.eval_batch_size)
                report = finite_metrics(validation, a, p)
                report.update(step=step, validation_seed=seed+1000003)
                with (out/'validation.jsonl').open('a') as val_stream:
                    val_stream.write(json.dumps(report)+'\n')
                eligible = report['incompatibility_rate'] <= args.violation_threshold
                criterion = (1, report['revenue_mean']) if eligible else (0, -report['incompatibility_rate'], report['revenue_mean'])
                improved = best_criterion is None or criterion > best_criterion
                if improved:
                    best_criterion = criterion; best_meta = report.copy()
                    best_meta['qualified'] = eligible
                val_history.append(report)
                if len(val_history)>=3 and decay_start is None:
                    stable = all(abs(val_history[-j]['revenue_mean']-val_history[-j-1]['revenue_mean'])<=.03 and
                                 abs(val_history[-j]['incompatibility_rate']-val_history[-j-1]['incompatibility_rate'])<=.01
                                 for j in (1, 2))
                    if stable:
                        decay_start = step
                payload = dict(model_config=model.config, state_dict=model.state_dict(),
                               training_config=config, step=step, seed=seed,
                               validation=report, checkpoint_selection_qualified=eligible,
                               optimizer_state_dict=optimizer.state_dict(),
                               torch_rng_state=torch.get_rng_state(),
                               cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                               lambda_incompatibility=lamb, best_criterion=list(best_criterion),
                               best_meta=best_meta, val_history=val_history, decay_start=decay_start,
                               elapsed_seconds=elapsed_before_resume+time.monotonic()-start_time)
                atomic_torch_save(payload, out/'checkpoint_last.pt')
                if improved:
                    atomic_torch_save(payload, out/'checkpoint.pt')
                print(f'GEMNET_TRAIN step={step}/{preset["steps"]} revenue={report["revenue_mean"]:.6f} '
                      f'incompatible={report["incompatibility_rate"]:.6f} qualified={eligible}', flush=True)
                status_path.write_text(json.dumps(dict(status='RUNNING', step=step, selected=best_meta), indent=2))
    status_path.write_text(json.dumps(dict(status='COMPLETED', selected=best_meta,
                                          selected_checkpoint=str(out/'checkpoint.pt'),
                                          selected_checkpoint_qualified=best_meta['qualified'],
                                          elapsed_seconds=elapsed_before_resume+time.monotonic()-start_time,
                                          exact_paper_numerical_reproduction_claim=False), indent=2))
    return out/'checkpoint.pt'


def transform(args):
    device = resolve_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    transformed, metadata = transform_model(model, args.out, args.grid_points, args.grid_mode,
                                            args.context_count, args.own_count, args.seed,
                                            args.time_limit, args.margin_mode, args.max_contexts,
                                            args.transform_variant, args.resume)
    out = Path(args.out)
    # Independent off-grid data for diagnostics, never fed back into the MILP.
    test_values = np.random.default_rng(args.seed+104729).random((args.diagnostic_samples, model.n_bidders, model.n_items)).astype(np.float32)
    before_a, before_p = predict(model, test_values, device, args.eval_batch_size)
    after_a, after_p = predict(transformed, test_values, device, args.eval_batch_size)
    diagnostics = dict(before=finite_metrics(test_values, before_a, before_p),
                       after=finite_metrics(test_values, after_a, after_p),
                       test_seed=args.seed+104729, global_dsic_certified=False,
                       degenerate_allocation_rate=float((after_a.sum((1, 2))<1e-8).mean()),
                       before_null_choice_rate_by_bidder=(before_a.sum(2)<1e-8).mean(0).tolist(),
                       after_null_choice_rate_by_bidder=(after_a.sum(2)<1e-8).mean(0).tolist(),
                       revenue_retention_ratio=None if abs(before_p.sum(1).mean())<1e-12 else float(after_p.sum(1).mean()/before_p.sum(1).mean()))
    grid_count = args.grid_points ** (model.n_bidders*model.n_items)
    if args.grid_mode == 'cartesian' and grid_count <= args.max_grid_check:
        axis = np.linspace(0, 1, args.grid_points)
        iterator = itertools.product(axis, repeat=model.n_bidders*model.n_items)
        max_excess, count, violations = 0.0, 0, 0
        while True:
            chunk = list(itertools.islice(iterator, args.eval_batch_size))
            if not chunk:
                break
            v = np.asarray(chunk, dtype=np.float32).reshape(-1, model.n_bidders, model.n_items)
            a, _ = predict(transformed, v, device, args.eval_batch_size)
            e = np.maximum(a.sum(1)-1, 0).max(1)
            max_excess = max(max_excess, float(e.max()))
            count += len(v); violations += int((e>1e-6).sum())
        diagnostics['full_grid'] = dict(checked=True, n_profiles=count,
                                        allocation_excess_max=max_excess, violations=violations,
                                        finite_grid_feasible=violations==0)
    else:
        diagnostics['full_grid'] = dict(checked=False, n_profiles_possible=grid_count,
                                        reason='sampled grid or exceeds explicit max_grid_check limit')
    invalid = not diagnostics['after']['feasible_on_evaluated_profiles']
    if diagnostics['full_grid'].get('checked'):
        invalid |= not diagnostics['full_grid']['finite_grid_feasible']
    diagnostics['valid_for_empirical_revenue_comparison'] = not invalid
    diagnostics['status'] = 'INVALID_FEASIBILITY' if invalid else 'FINITE_CHECKS_PASSED_NOT_GLOBAL_CERTIFICATE'
    np.savez_compressed(out/'pre_post_diagnostics.npz', values=test_values,
                        before_allocation=before_a, before_payment=before_p,
                        after_allocation=after_a, after_payment=after_p)
    (out/'transform_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    (out/'transform_status.json').write_text(json.dumps(dict(status='COMPLETED',
                           validity=diagnostics['status'], checkpoint_sha256=checkpoint_hash,
                           price_transform=str(out/'price_transform.npz'),
                           exact_paper_numerical_reproduction_claim=False), indent=2))
    print('GEMNET_TRANSFORM ' + diagnostics['status'], flush=True)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--out', required=True); t.add_argument('--setting', default='2x2')
    t.add_argument('--seed', type=int, default=0); t.add_argument('--preset', choices=['paper','smoke'], default='paper')
    t.add_argument('--device', default='auto'); t.add_argument('--eval-batch-size', type=int, default=256)
    for flag in ['hidden','menu-size','steps','batch-size','validation-samples','validate-every','lambda-update-every']:
        t.add_argument('--'+flag, type=int)
    t.add_argument('--learning-rate', type=float, default=.005)
    t.add_argument('--training-safety-margin', type=float, default=1e-5)
    t.add_argument('--violation-threshold', type=float, default=.001)
    t.add_argument('--price-activation', choices=['relu','linear'], default='relu')
    t.add_argument('--resume', action='store_true')
    tr = sub.add_parser('transform')
    tr.add_argument('--checkpoint', required=True); tr.add_argument('--out', required=True)
    tr.add_argument('--device', default='auto'); tr.add_argument('--grid-points', type=int, default=7)
    tr.add_argument('--grid-mode', choices=['cartesian','sampled'], default='cartesian')
    tr.add_argument('--context-count', type=int, default=64); tr.add_argument('--own-count', type=int, default=128)
    tr.add_argument('--seed', type=int, default=271828); tr.add_argument('--time-limit', type=float, default=300)
    tr.add_argument('--margin-mode', choices=['paper','conservative'], default='paper')
    tr.add_argument('--transform-variant', choices=['paper-accelerated','strict'], default='paper-accelerated')
    tr.add_argument('--resume', action='store_true')
    tr.add_argument('--max-contexts', type=int, default=10000)
    tr.add_argument('--max-grid-check', type=int, default=1000000)
    tr.add_argument('--diagnostic-samples', type=int, default=2048)
    tr.add_argument('--eval-batch-size', type=int, default=256)
    return parser


def main():
    args = parser().parse_args()
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    if args.command == 'train':
        train(args)
    else:
        transform(args)


if __name__ == '__main__':
    main()
