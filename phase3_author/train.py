#!/usr/bin/env python3
"""Phase III: original author networks with explicit portable training adapters.

CITransNet follows the released Setting H procedure. PreferenceNet retains its
learned allocation classifier and iterative self-training. ProportionNet follows
the ORIGINAL paper's positive-part unfairness Eq.(4), not a share-matching proxy.
See REPRODUCTION_NOTES.md for differences from native upstream entrypoints.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import numpy as np
import torch
from torch import nn

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase3_author.bridge import (METHODS, CONTEXT_METHODS, PROPORTION_METHODS, build_model, build_preference_classifier,
                                 ContextBatchModel, direct_metrics, preference_score, sha256, method_provenance)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def make_data(count, n, m, seed, contextual):
    g = torch.Generator().manual_seed(seed)
    values = torch.rand(count, n, m, generator=g)
    result = {'values': values}
    if contextual:
        bc = torch.rand(count, n, 10, generator=g) * 2 - 1
        ic = torch.rand(count, m, 10, generator=g) * 2 - 1
        result['values'] = values * torch.sigmoid(bc @ ic.transpose(1, 2))
        result.update(bidder_context=bc, item_context=ic)
    return result


def deviations(model, truth, reports):
    """Evaluate n genuine unilateral deviations per profile, preserving contexts."""
    batch, n, m = truth.shape
    eye = torch.eye(n, device=truth.device).view(1, n, n, 1)
    tiled = truth[:, None] * (1 - eye) + reports[:, None] * eye
    alloc, pay = model(tiled.reshape(batch * n, n, m))
    alloc = alloc.reshape(batch, n, n, m)
    pay = pay.reshape(batch, n, n)
    idx = torch.arange(n, device=truth.device)
    return (alloc[:, idx, idx] * truth).sum(-1) - pay[:, idx, idx]


def optimize_reports(model, truth, initial, steps, lr, adam=False):
    report = initial.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([report], lr=lr) if adam else None
    # Network parameters need no inner-loop gradients; autograd.grad only reports.
    for _ in range(steps):
        util = deviations(model, truth, report)
        gradient, = torch.autograd.grad(util.sum(), report)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite misreport gradient')
        if adam:
            opt.zero_grad(set_to_none=True)
            report.grad = -gradient
            opt.step()
            with torch.no_grad():
                report.clamp_(0, 1)
        else:
            with torch.no_grad():
                report.add_(gradient, alpha=lr).clamp_(0, 1)
    return report.detach()


def random_allocations(count, n, m, generator):
    # Exact distribution from PreferenceNet's generate_random_allocations_payments.
    bids = torch.rand(count, n, m, generator=generator)
    logits = torch.rand(count, n + 1, m + 1, generator=generator)
    alloc = torch.softmax(logits, dim=1)[:, :-1, :-1]
    payment = torch.rand(count, n, generator=generator) * (bids * alloc).sum(-1)
    return bids, alloc, payment


def ranking_labels(scores, generator, pairs=5000):
    """Authors' finite random-pair ranking rule: preferred over >80% of pairs."""
    wins = torch.zeros_like(scores, dtype=torch.int32)
    for _ in range(pairs):
        wins += scores > scores[torch.randperm(len(scores), generator=generator)]
    return (wins > 0.8 * pairs).float()


def fit_classifier(classifier, data, labels, args, device, balanced, generator):
    # Warm optimizer restart, exactly as author routine, retaining network weights.
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3,
                                 betas=(0.5, 0.999), weight_decay=0.005)
    classifier.requires_grad_(True)
    classifier.train()
    size = len(labels)
    pos, neg = torch.where(labels == 1)[0], torch.where(labels == 0)[0]
    for _ in range(args.classifier_epochs):
        if balanced and len(pos) and len(neg):
            half = size // 2
            indices = torch.cat([pos[torch.randint(len(pos), (half,), generator=generator)],
                                 neg[torch.randint(len(neg), (size - half,), generator=generator)]])
            indices = indices[torch.randperm(size, generator=generator)]
        else:
            indices = torch.randperm(size, generator=generator)
        for idx in indices.split(args.batch_size):
            if len(idx) < 2:  # original BatchNorm cannot train on a singleton.
                continue
            pred = classifier(*(x[idx].to(device) for x in data))
            loss = nn.functional.binary_cross_entropy(pred, labels[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    classifier.eval().requires_grad_(False)


def read_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mode', choices=['smoke', 'full'], default='full')
    p.add_argument('--device', default='auto')
    p.add_argument('--n-agents', type=int, default=3)
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--epochs', type=int)
    p.add_argument('--train-samples', type=int)
    p.add_argument('--eval-samples', type=int, default=512)
    p.add_argument('--data-seed', type=int, default=314159)
    p.add_argument('--eval-seed', type=int, default=271828)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--misreport-steps', type=int)
    p.add_argument('--preference-objective', choices=['tvf', 'entropy', 'quota'], default='tvf')
    p.add_argument('--preference-samples', type=int)
    p.add_argument('--classifier-epochs', type=int)
    p.add_argument('--pairwise-samples', type=int)
    p.add_argument('--self-samples', type=int)
    p.add_argument('--fairness-distance', type=float, default=0.0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--torch-threads', type=int, default=4)
    args = p.parse_args(argv)
    full = args.mode == 'full'
    ci = args.method in CONTEXT_METHODS
    prop = args.method in PROPORTION_METHODS
    defaults = dict(epochs=(80 if ci else 120 if prop else 200) if full else 1,
                    train_samples=(100000 if ci else 640000 if prop else 160000) if full else 16,
                    batch_size=(500 if ci else 2048) if full else 8,
                    misreport_steps=25 if full else 2,
                    preference_samples=80000 if full else 32,
                    classifier_epochs=50 if full else 1,
                    pairwise_samples=5000 if full else 16,
                    self_samples=5000 if full else 8)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for key in ('epochs', 'train_samples', 'batch_size', 'eval_samples',
                'misreport_steps', 'preference_samples', 'classifier_epochs', 'pairwise_samples', 'self_samples'):
        if getattr(args, key) <= 0:
            p.error(f'{key} must be positive')
    if args.batch_size < 2 or args.train_samples < 2:
        p.error('batch/sample counts must be at least two')
    if args.n_agents < 1 or args.n_items < 1:
        p.error('agent and item counts must be positive')
    if args.preference_objective == 'quota' and args.n_agents < 2:
        p.error('Authors quota preference requires at least two agents')
    if args.eval_seed == args.data_seed:
        p.error('training and test seeds must differ')
    if not 0 <= args.fairness_distance <= 1:
        p.error('fairness distance must be in [0,1]')
    return args


def train(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    compare = {k: v for k, v in config.items() if k not in ('resume', 'device', 'torch_threads', 'output_dir')}
    signature = hashlib.sha256(json.dumps(compare, sort_keys=True).encode()).hexdigest()
    previous = out / 'config.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if not args.resume or old['signature'] != signature:
            raise RuntimeError('Existing output: use --resume with the same scientific configuration, or a new directory')
    elif any(out.iterdir()):
        raise RuntimeError('Refusing nonempty output without a matching configuration')
    write_json(previous, {'signature': signature, **config, 'resolved_device': device})
    write_json(out / 'status.json', {'state': 'RUNNING', 'status': 'RUNNING', 'method': args.method})
    args._run_started = True
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Public data use dedicated generators and do not depend on training seed.
    contextual = args.method in CONTEXT_METHODS
    proportion_protocol = args.method in PROPORTION_METHODS
    training = make_data(args.train_samples, args.n_agents, args.n_items, args.data_seed, contextual)
    evaluation = make_data(args.eval_samples, args.n_agents, args.n_items, args.eval_seed, contextual)
    np.savez_compressed(out / 'eval_data.npz', **{k: v.numpy() for k, v in evaluation.items()})
    write_json(out / 'data_manifest.json', {k: {'sha256': hashlib.sha256(v.numpy().tobytes()).hexdigest(),
                                               'shape': list(v.shape)} for k, v in training.items()})
    model = build_model(args.method, args.n_agents, args.n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(10000 + args.seed)
    lam = torch.full((args.n_agents,), 5.0, device=device)
    fair_lam = torch.ones(args.n_items, device=device)
    rho = rho_fair = 1.0
    iteration = start_epoch = 0
    cache = torch.rand(args.train_samples, args.n_agents, args.n_items, generator=generator) if contextual else None
    classifier = None
    pref_data = pref_labels = None
    threshold = None
    reference_generator = torch.Generator().manual_seed(args.data_seed + 531)
    if not contextual:
        reference_data = random_allocations(args.preference_samples, args.n_agents, args.n_items, reference_generator)
        reference_scores = preference_score(reference_data[1], args.preference_objective, args.fairness_distance)
        threshold = float(torch.quantile(reference_scores, 0.8))
        np.savez_compressed(out / 'preference_reference.npz', scores=reference_scores.numpy())
    if args.method == 'PreferenceNet':
        classifier = build_preference_classifier(args.n_agents, args.n_items).to(device)
    resume_path = out / 'training_state.pt'
    resume_action = 'fresh_initialization'
    if args.resume and resume_path.exists():
        resume_action = 'full_state_restored'
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        lam, fair_lam = state['lam'].to(device), state['fair_lam'].to(device)
        rho, rho_fair, start_epoch, iteration = state['rho'], state['rho_fair'], state['epoch'], state['iteration']
        cache = state['cache']
        generator.set_state(state['generator'])
        torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda') and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if classifier is not None:
            classifier.load_state_dict(state['classifier'], strict=True)
            classifier.eval().requires_grad_(False)
            pref_data, pref_labels, threshold = state['pref_data'], state['pref_labels'], state['threshold']
    elif classifier is not None:
        pref_data = reference_data
        scores = preference_score(pref_data[1], args.preference_objective, args.fairness_distance)
        pref_labels = ranking_labels(scores, reference_generator, args.pairwise_samples)
        fit_classifier(classifier, pref_data, pref_labels, args, device, True, generator)

    if args.resume and not resume_path.exists():
        resume_action = 'restart_from_initial_seed_no_completed_epoch'
        print('RESUME_ACTION: no completed-epoch state exists; explicitly restarting from the original seeds.', flush=True)

    started = time.time()
    log_path = out / 'training_curve.csv'
    if start_epoch:
        if not log_path.exists():
            raise RuntimeError('Cannot resume: completed checkpoint has no associated training curve')
        with log_path.open(newline='') as saved_log:
            reader = csv.DictReader(saved_log)
            fields = reader.fieldnames
            old_rows = list(reader)
        keep = [row for row in old_rows if int(row['epoch']) <= start_epoch]
        discarded = [row for row in old_rows if int(row['epoch']) > start_epoch]
        if len(keep) != start_epoch or [int(r['epoch']) for r in keep] != list(range(1,start_epoch+1)):
            raise RuntimeError('Completed checkpoint/training curve epoch history is inconsistent')
        if discarded:
            orphan = out / f'uncheckpointed_training_curve_before_resume_epoch{start_epoch}.csv'
            with orphan.open('w',newline='') as saved_log:
                w=csv.DictWriter(saved_log,fieldnames=fields); w.writeheader(); w.writerows(discarded)
            with log_path.open('w',newline='') as saved_log:
                w=csv.DictWriter(saved_log,fieldnames=fields); w.writeheader(); w.writerows(keep)
    with log_path.open('a' if start_epoch else 'w', newline='') as log:
        columns = ['epoch', 'iteration', 'revenue', 'regret_bidder_mean', 'loss',
                   'paper_unfairness_sum', 'absolute_tvf_sum', 'preference_probability', 'rho', 'rho_fair', 'elapsed_s']
        writer = csv.DictWriter(log, fieldnames=columns)
        if not start_epoch:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            totals = {k: 0.0 for k in ('revenue', 'regret_bidder_mean', 'loss',
                                     'paper_unfairness_sum', 'absolute_tvf_sum', 'preference_probability')}
            seen = 0
            if contextual:
                # Original Setting H samples training indices with replacement.
                indices = torch.randint(args.train_samples,
                                        (math.ceil(args.train_samples / args.batch_size) * args.batch_size,),
                                        generator=generator)
            else:
                indices = torch.randperm(args.train_samples, generator=generator)
            for idx in indices.split(args.batch_size):
                iteration += 1
                values = training['values'][idx].to(device)
                bound = ContextBatchModel(model, training['bidder_context'][idx].to(device),
                                          training['item_context'][idx].to(device)) if contextual else model
                model.train()
                initial = cache[idx].to(device) if contextual else values
                mis = optimize_reports(bound, values, initial, args.misreport_steps,
                                       0.001 if contextual else 0.1, adam=contextual)
                if contextual:
                    cache[idx] = mis.cpu()
                alloc, payment = bound(values)
                truth_util = (values * alloc).sum(-1) - payment
                regret = (deviations(bound, values, mis) - truth_util).clamp_min(0)
                mean_regret = regret.mean(0)
                revenue = payment.sum(1).mean()
                metrics = direct_metrics(alloc, args.fairness_distance)
                pref_prob = torch.zeros((), device=device)
                if contextual:
                    loss = -revenue + (lam * mean_regret).sum() + rho / 2 * mean_regret.square().sum()
                elif proportion_protocol:
                    # Original ProportionNet Eq.(5) squares SUM of expected constraints.
                    unfair = metrics['paper_unfairness_by_item'].mean(0)
                    loss = -revenue + (lam * mean_regret).sum() + rho / 2 * mean_regret.sum().square()
                    if args.method == 'ProportionNet':
                        loss = loss + (fair_lam * unfair).sum() + rho_fair / 2 * unfair.sum().square()
                else:
                    # Matches released PreferenceNet train_loop reductions (mean over B,n).
                    loss = -revenue + (lam * regret).mean() + rho / 2 * regret.square().mean()
                    if classifier is not None:
                        pref_prob = classifier(values, alloc, payment).mean()
                        loss = loss - pref_prob
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at iteration {iteration}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    interval = 2 if contextual else 100 if proportion_protocol else 25
                    if iteration % interval == 0:
                        if proportion_protocol:
                            # Algorithm 1 updates constraints at w^(t+1).
                            updated_alloc,updated_payment = model(values)
                            updated_truth = (values*updated_alloc).sum(-1)-updated_payment
                            updated_regret = (deviations(model,values,mis)-updated_truth).clamp_min(0).mean(0)
                            updated_unfair = direct_metrics(updated_alloc,args.fairness_distance)['paper_unfairness_by_item'].mean(0)
                            lam += rho * updated_regret
                            if args.method == 'ProportionNet':
                                fair_lam += rho_fair * updated_unfair
                        else:
                            lam += rho * mean_regret
                    if not contextual and not proportion_protocol and iteration % 2500 == 0:
                        rho += 1
                count = len(idx)
                values_log = dict(revenue=revenue.item(), regret_bidder_mean=mean_regret.mean().item(),
                                  loss=loss.item(), paper_unfairness_sum=metrics['paper_unfairness_sum'].mean().item(),
                                  absolute_tvf_sum=metrics['absolute_tvf_sum'].mean().item(),
                                  preference_probability=pref_prob.item())
                for key in totals:
                    totals[key] += values_log[key] * count
                seen += count
            if contextual and (epoch + 1) % 2 == 0:
                rho += 5
            if proportion_protocol and (epoch + 1) % 2 == 0:
                # Increment magnitude not specified by original text: explicitly 1.
                rho += 1
                rho_fair += 1
            if classifier is not None and epoch % 5 == 0:
                # Released implementation refreshes at zero-based epochs 0,5,10,...
                sbids = torch.rand(args.self_samples, args.n_agents, args.n_items, generator=generator)
                sa, sp, sl = [], [], []
                model.eval()
                with torch.no_grad():
                    for chunk in sbids.split(args.batch_size):
                        chunk = chunk.to(device)
                        a, p = model(chunk)
                        sa.append(a.cpu()); sp.append(p.cpu())
                        sl.append((classifier(chunk, a, p) > 0.5).float().cpu())
                new_data = (sbids, torch.cat(sa), torch.cat(sp))
                pref_data = tuple(torch.cat([old, new]) for old, new in zip(pref_data, new_data))
                pref_labels = torch.cat([pref_labels, torch.cat(sl)])
                fit_classifier(classifier, pref_data, pref_labels, args, device, False, generator)
            record = dict(epoch=epoch + 1, iteration=iteration, **{k: v / seen for k, v in totals.items()},
                          rho=rho, rho_fair=rho_fair, elapsed_s=time.time() - started)
            writer.writerow(record); log.flush()
            print(json.dumps(record), flush=True)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), lam=lam,
                         fair_lam=fair_lam, rho=rho, rho_fair=rho_fair, epoch=epoch + 1,
                         iteration=iteration, cache=cache, generator=generator.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                         classifier=classifier.state_dict() if classifier is not None else None,
                         pref_data=pref_data, pref_labels=pref_labels, threshold=threshold)
            torch.save(state, out / 'training_state.pt.tmp')
            (out / 'training_state.pt.tmp').replace(resume_path)
            torch.save(model.state_dict(), out / 'checkpoint.pt')
            if classifier is not None:
                torch.save(classifier.state_dict(), out / 'preference_classifier.pt')

    model.eval()
    # A completed resumed run still rewrites portable final metadata below.
    label = {'CITransNet': 'CITransNet_author_architecture_SettingH',
             'CITransNetNoContext': 'CITransNet_zero_context_review_controlled_ablation',
             'PreferenceNet': f'PreferenceNet_author_architecture_{args.preference_objective}_synthetic',
             'ProportionNet': 'ProportionNet_paper_Eq4_Eq5_reimplementation',
             'RegretNetPreferenceBaseline': 'RegretNet_PreferenceNet_matched_revenue_only',
             'RegretNetProportionBaseline': 'RegretNet_ProportionNet_matched_revenue_only'}[args.method]
    metadata = dict(method=args.method, method_label=label, n_agents=args.n_agents, n_items=args.n_items,
                    seed=args.seed, protocol='author_architecture_portable_training_adapter',
                    resume_action=resume_action,
                    contextual=contextual, context_dim=10 if contextual else 0,
                    context_access='zeroed_in_mechanism' if args.method=='CITransNetNoContext' else 'public_context' if contextual else 'none',
                    controlled_ablation=args.method in ('CITransNetNoContext','RegretNetProportionBaseline','RegretNetPreferenceBaseline'),
                    controlled_comparator_for={'CITransNetNoContext':'CITransNet','RegretNetProportionBaseline':'ProportionNet','RegretNetPreferenceBaseline':'PreferenceNet'}.get(args.method),
                    setting=f'{args.n_agents}x{args.n_items}_continuous_context' if contextual else f'{args.n_agents}x{args.n_items}_iid_uniform_additive',
                    author_setting_match=(contextual and (args.n_agents,args.n_items)==(3,10)) or (args.method=='PreferenceNet' and args.n_agents in (2,4) and args.n_items in (2,4)) or (args.method=='ProportionNet' and args.n_agents in range(1,6) and args.n_items in range(2,7)),
                    evaluation_role='shared_review_protocol_not_original_table_number_recovery',
                    evaluation_protocol_changed=True,
                    preference_objective=args.preference_objective if not contextual else None,
                    preference_threshold=threshold, preference_threshold_source='80th percentile of fixed synthetic reference allocations, seed=data_seed+531; independent of training seed',
                    fairness_distance=args.fairness_distance, parameter_count=sum(p.numel() for p in model.parameters()),
                    hidden_layers_total=(getattr(model,'n_hidden_layers',0)+1) if not contextual else None,
                    epochs=args.epochs, iterations=iteration, mode=args.mode, smoke_only=args.mode == 'smoke',
                    checkpoint_sha256=sha256(out / 'checkpoint.pt'), eval_data_sha256=sha256(out / 'eval_data.npz'),
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, device=device),
                    source_manifest_sha256=sha256(Path(__file__).parent / 'SOURCES.json'),
                    adapter_code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                    limitations=['Synthetic preference experiment, not a new human-subject study',
                                 'Empirical regret is a searched lower bound, not a DSIC certificate',
                                 'ProportionNet Eq4 positive-part fairness and absolute TVF are reported separately',
                                 'No human burden score is invented'])
    metadata.update(method_provenance(args.method))
    write_json(out / 'metadata.json', metadata)
    write_json(out / 'status.json', {'state': 'COMPLETED', 'status': 'COMPLETED', 'training_completed': True,
                                    'evaluation_pending': True, 'method': args.method, 'smoke_only': args.mode == 'smoke'})
    return metadata


if __name__ == '__main__':
    arguments = read_args()
    try:
        print(json.dumps(train(arguments), indent=2), flush=True)
    except Exception as exc:
        if getattr(arguments, '_run_started', False):
            write_json(arguments.output_dir / 'status.json', {'state': 'FAILED', 'status': 'FAILED', 'error': str(exc)})
        raise
