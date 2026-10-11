"""Single-bidder deviation search using true valuations and frozen mechanisms.

All gains are empirical lower bounds on worst-case regret. Increasing random
budgets and the strong audit retain all previous candidates, including truth.
"""
from __future__ import annotations
import numpy as np
import torch


def mechanism_output(model, reports):
    allocation, payment = model(reports)
    if allocation.shape != reports.shape or payment.shape != reports.shape[:2]:
        raise ValueError('Author model output dimensions are not allocation[N,A,M], payment[N,A]')
    if not torch.isfinite(allocation).all() or not torch.isfinite(payment).all():
        raise FloatingPointError('Non-finite allocation or payment; no fallback values permitted')
    return allocation, payment


def deviating_utility(model, values, bidder, reports):
    """reports[B,K,M]; only bidder changes; all other reports remain truthful."""
    batch, candidates, items = reports.shape
    all_reports = values[:, None].expand(-1, candidates, -1, -1).clone()
    all_reports[:, :, bidder, :] = reports
    allocation, payment = mechanism_output(model, all_reports.reshape(-1, values.shape[1], items))
    allocation = allocation.reshape(batch, candidates, values.shape[1], items)
    payment = payment.reshape(batch, candidates, values.shape[1])
    return (allocation[:, :, bidder] * values[:, None, bidder]).sum(-1) - payment[:, :, bidder]


def audit_batch(model, values, random_reports, random_budgets, pgd_restarts=8, pgd_steps=100, pgd_lr=0.05, candidate_batch=16):
    """random_reports[B,K,A,M], common across methods/checkpoints.

    PGD starts: truth, largest-budget best random report, then a prefix of the
    shared random candidates. Plain gradient ascent with [0,1] projection,
    retaining the best utility at every iteration (including iteration zero).
    """
    if not torch.isfinite(values).all() or not torch.isfinite(random_reports).all() or torch.any(values < 0) or torch.any(values > 1) or torch.any(random_reports < 0) or torch.any(random_reports > 1):
        raise ValueError('Truth values and deviation candidates must be finite within [0,1]')
    if tuple(random_reports.shape[2:]) != tuple(values.shape[1:]):
        raise ValueError('Candidate shape does not match valuations')
    budgets = sorted(set(int(b) for b in random_budgets))
    if not budgets or budgets[0] < 1 or budgets[-1] > random_reports.shape[1]:
        raise ValueError('Invalid random budgets')
    if not (2 <= pgd_restarts <= random_reports.shape[1] + 2) or pgd_steps < 0 or pgd_lr <= 0:
        raise ValueError('PGD needs 2..K+2 starts, nonnegative steps, and positive lr')
    batch, bidders, items = values.shape
    with torch.no_grad():
        allocation, payment = mechanism_output(model, values)
        truthful = (values * allocation).sum(-1) - payment
    result = {'allocation': allocation, 'payments': payment, 'truth_utility': truthful}
    for budget in budgets:
        result[f'random_gain_budget{budget}'] = torch.zeros_like(truthful)
        result[f'random_best_report_budget{budget}'] = values.clone()
    result['pgd_gain'] = torch.zeros_like(truthful)
    result['pgd_best_report'] = values.clone()
    for bidder in range(bidders):
        best_utility = truthful[:, bidder].clone()
        best_report = values[:, bidder].clone()
        done = 0
        for budget in budgets:
            with torch.no_grad():
                while done < budget:
                    stop = min(done + candidate_batch, budget)
                    candidate = random_reports[:, done:stop, bidder]
                    utility = deviating_utility(model, values, bidder, candidate)
                    block_best, block_idx = utility.max(dim=1)
                    improve = block_best > best_utility
                    best_report[improve] = candidate[torch.arange(batch, device=values.device), block_idx][improve]
                    best_utility = torch.maximum(best_utility, block_best)
                    done = stop
                result[f'random_gain_budget{budget}'][:, bidder] = (best_utility - truthful[:, bidder]).clamp_min(0)
                result[f'random_best_report_budget{budget}'][:, bidder] = best_report
        starts = torch.cat([values[:, None, bidder], best_report[:, None], random_reports[:, :pgd_restarts-2, bidder]], dim=1)
        pgd_best_util = truthful[:, bidder].clone()
        pgd_best_report = values[:, bidder].clone()
        for step in range(pgd_steps + 1):
            report = starts.detach().requires_grad_(step < pgd_steps)
            utility = deviating_utility(model, values, bidder, report)
            with torch.no_grad():
                step_best, step_idx = utility.max(dim=1)
                improve = step_best > pgd_best_util
                pgd_best_report[improve] = report[torch.arange(batch, device=values.device), step_idx][improve]
                pgd_best_util = torch.maximum(pgd_best_util, step_best)
            if step < pgd_steps:
                if utility.requires_grad:
                    gradient, = torch.autograd.grad(utility.sum(), report)
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError('Non-finite deviation gradient')
                    if torch.count_nonzero(gradient) == 0:
                        break  # All deterministic gradient-ascent starts would remain unchanged.
                    starts = (report + pgd_lr * gradient).clamp(0, 1).detach()
                else:
                    starts = report.detach()
        result['pgd_gain'][:, bidder] = (pgd_best_util - truthful[:, bidder]).clamp_min(0)
        result['pgd_best_report'][:, bidder] = pgd_best_report
    result['strong_gain'] = torch.maximum(result['pgd_gain'], result[f'random_gain_budget{budgets[-1]}'])
    choose_random = result[f'random_gain_budget{budgets[-1]}'] > result['pgd_gain']
    result['strong_best_report'] = torch.where(choose_random[:, :, None], result[f'random_best_report_budget{budgets[-1]}'], result['pgd_best_report'])
    strongest = result['strong_gain']
    if not torch.all(strongest + 1e-7 >= result[f'random_gain_budget{budgets[-1]}']):
        raise AssertionError('Strong audit failed candidate-containment check')
    for previous, current in zip(budgets, budgets[1:]):
        if not torch.all(result[f'random_gain_budget{current}'] + 1e-7 >= result[f'random_gain_budget{previous}']):
            raise AssertionError('Nested random audit is not monotone')
    return {name: tensor.detach().cpu().numpy() for name, tensor in result.items()}


def per_profile_metrics(arrays, budgets):
    values, allocation, payments = arrays['values'], arrays['allocation'], arrays['payments']
    utilities = (values * allocation).sum(-1) - payments
    metrics = {
        'revenue': payments.sum(axis=1),
        'welfare': (values * allocation).sum(axis=(1, 2)),
        'ir_violation_sum': np.maximum(-utilities, 0).sum(axis=1),
        'allocation_excess_max': np.maximum(allocation.sum(axis=1) - 1, 0).max(axis=1),
        'allocation_negative_max': np.maximum(-allocation, 0).max(axis=(1, 2)),
        'payment_negative_max': np.maximum(-payments, 0).max(axis=1),
    }
    for label, key in [(f'random_{b}', f'random_gain_budget{b}') for b in budgets] + [('pgd', 'pgd_gain'), ('strong', 'strong_gain')]:
        metrics[f'regret_sum_{label}'] = arrays[key].sum(axis=1)
        metrics[f'regret_bidder_mean_{label}'] = arrays[key].mean(axis=1)
        metrics[f'regret_bidder_max_{label}'] = arrays[key].max(axis=1)
    if any(not np.isfinite(a).all() for a in metrics.values()):
        raise FloatingPointError('Non-finite profile metric')
    return metrics
