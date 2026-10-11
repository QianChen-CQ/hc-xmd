"""Actual GemNet price MILPs (Eq. 6--9), solved with SciPy/HiGHS.

Nonnegative price changes implement the paper's IR-screening variant. The
adjustment for bidder i depends ONLY on the other bidders' grid context.
No allocation scaling, clipping across bidders, or own-bid-dependent solve is
used. Diagnostics are not advertised as a continuous-domain DSIC certificate.
"""
import itertools
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def cartesian_grid(points, dimension, vmax=1.0):
    if points < 2:
        raise ValueError('grid-points must be >= 2')
    return np.asarray(list(itertools.product(np.linspace(0, vmax, points), repeat=dimension)), dtype=np.float64)


def solve_price_milp(own_values, bundles, prices, compatible, margin,
                     time_limit=300.0, mip_gap=0.0, keep_choice_fraction=0.0,
                     zero_violation_skip=False):
    """Minimize sum delta subject to Eq. 7/8 and delta >= 0.

    own_values [L,m], bundles [K,m], prices [K], compatible [L,K].
    The final menu entry must be the null allocation at zero price.
    Binary variables for permanently-negative utilities can be removed because
    nonnegative delta and the null option prevent those entries from winning.
    """
    values = np.asarray(own_values, dtype=np.float64)
    bundles, prices = np.asarray(bundles, dtype=np.float64), np.asarray(prices, dtype=np.float64)
    compatible = np.asarray(compatible, dtype=bool)
    L, K = compatible.shape
    if bundles.shape != (K, values.shape[1]) or prices.shape != (K,):
        raise ValueError('MILP array dimensions disagree')
    if not (np.allclose(bundles[-1], 0) and prices[-1] == 0 and np.all(compatible[:, -1])):
        raise ValueError('The fixed null option must be compatible and have zero price/allocation')
    if margin <= 0 or not np.isfinite(margin):
        raise ValueError('A positive finite utility margin is required')
    utility = values @ bundles.T - prices
    # Prices are already nonnegative for the default ReLU head. For the
    # explicitly requested linear head, final transformed prices are >= 0.
    delta_lower = np.maximum(0.0, -prices)
    original_adjusted = utility-delta_lower[None, :]
    original_choices = original_adjusted.argmax(1)
    original_compatible = compatible[np.arange(L), original_choices]
    if zero_violation_skip and original_compatible.all():
        return delta_lower, dict(solver='paper_zero_violation_skip', status=0,
                                message='All own-grid choices already compatible at this context.',
                                seconds=0.0, rows=L, menu_size=K, n_variables=0,
                                n_binary=0, n_constraints=0, big_m=None,
                                utility_margin=margin, objective=float(delta_lower.sum()),
                                mip_gap=0.0, strict_margin_applied=False,
                                maximum_delta=float(delta_lower.max()), sum_delta=float(delta_lower.sum()),
                                minimum_verified_gap=float(np.min(np.partition(original_adjusted, -2, axis=1)[:, -1]-
                                                                 np.partition(original_adjusted, -2, axis=1)[:, -2])),
                                keep_choice_count=0)
    compatible_rows = np.flatnonzero(original_compatible)
    gaps = np.partition(original_adjusted, -2, axis=1)[:, -1]-np.partition(original_adjusted, -2, axis=1)[:, -2]
    # Sec. 3.3 explicitly says ascending gap; store that choice in diagnostics.
    order = compatible_rows[np.argsort(gaps[compatible_rows], kind='stable')]
    keep_rows = set(int(l) for l in order[:int(len(order)*keep_choice_fraction)])
    candidates = compatible & (utility - delta_lower[None, :] >= -1e-10)
    candidates[:, -1] = True
    for l in keep_rows:
        candidates[l] = False
    binary_pairs = np.argwhere(candidates)
    zids = {(int(l), int(k)): K + L + j for j, (l, k) in enumerate(binary_pairs)}
    nvar = K + L + len(binary_pairs)
    delta_upper = max(1.0, float(utility.max())) + 2 * margin + float(np.abs(prices).max()) + 1
    # This finite M bounds U - (u - delta) over the declared variable bounds;
    # unlike an unexamined constant M=10 it remains valid for larger settings.
    max_value = float((values.max(axis=0) * np.maximum(bundles, 0).max(axis=0)).sum())
    big_m = max(10.0, max_value + float(prices.max()) + delta_upper + margin + 1)
    lower = np.concatenate((delta_lower, np.zeros(L), np.zeros(len(binary_pairs))))
    upper = np.concatenate((np.full(K, delta_upper), np.full(L, max_value + 1), np.ones(len(binary_pairs))))
    lower[K-1] = upper[K-1] = 0
    objective = np.zeros(nvar); objective[:K-1] = 1
    integer = np.zeros(nvar, dtype=np.int32); integer[K+L:] = 1
    rr, cc, vv, row_lower, row_upper = [], [], [], [], []

    def row(terms, lo=-np.inf, hi=np.inf):
        r = len(row_lower)
        for c, v in terms:
            rr.append(r); cc.append(c); vv.append(v)
        row_lower.append(lo); row_upper.append(hi)

    for l in range(L):
        if l in keep_rows:
            best_k = int(original_choices[l])
            row([(K+l, 1), (best_k, 1)], lo=float(utility[l, best_k]), hi=float(utility[l, best_k]))
            for k in range(K):
                if k == best_k:
                    continue
                # Paper KC keeps the original choice using ordinary weak
                # inequalities (no positive gap between compatible choices).
                # A tiny numerical tie buffer is used only against incompatible
                # choices so the actual hard-argmax implementation remains valid.
                tie_buffer = 0.0 if compatible[l, k] else 1e-5
                row([(K+l, 1), (k, 1)], lo=float(utility[l, k])+tie_buffer)
            continue
        winners = []
        for k in range(K):
            z = zids.get((l, k))
            u = float(utility[l, k])
            if z is not None:
                winners.append((z, 1.0))
                # U >= u-delta+(1-z)*s_m.
                row([(K+l, 1), (k, 1), (z, margin)], lo=u+margin)
                # U <= u-delta+(1-z)*M.
                row([(K+l, 1), (k, 1), (z, big_m)], hi=u+big_m)
            elif u + margin > delta_lower[k] + 1e-12:
                # Incompatible OR a compatible option excluded by IR screening.
                row([(K+l, 1), (k, 1)], lo=u+margin)
        row(winners, lo=1, hi=1)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(row_lower), nvar)).tocsc()
    start = time.monotonic()
    result = milp(objective, integrality=integer, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, row_lower, row_upper),
                  options={'time_limit': float(time_limit), 'mip_rel_gap': float(mip_gap), 'disp': False})
    elapsed = time.monotonic() - start
    diagnostics = dict(solver='scipy.optimize.milp/HiGHS', status=int(result.status),
                       message=str(result.message), seconds=elapsed, rows=L,
                       menu_size=K, n_variables=nvar, n_binary=len(binary_pairs),
                       n_constraints=len(row_lower), big_m=big_m, utility_margin=margin,
                       objective=None if result.fun is None else float(result.fun),
                       mip_gap=None if getattr(result, 'mip_gap', None) is None else float(result.mip_gap),
                       keep_choice_fraction=keep_choice_fraction, keep_choice_count=len(keep_rows),
                       keep_choice_order='ascending original utility gap (paper text)',
                       strict_margin_applied=len(keep_rows)==0,
                       optimal=bool(result.status==0))
    if result.x is None or result.status not in (0, 1):
        error = RuntimeError('Price MILP did not reach an optimal solution: ' + json.dumps(diagnostics))
        error.solver_status = int(result.status)
        raise error
    # A time-limited integer incumbent can be used only after independent
    # integrality, bounds, linear-constraint and menu-choice validation. Its
    # non-optimal status and objective gap remain visible in every log.
    ax = matrix @ result.x
    if (np.max(np.abs(result.x[K+L:]-np.rint(result.x[K+L:])), initial=0)>2e-6 or
            np.any(result.x<lower-2e-6) or np.any(result.x>upper+2e-6) or
            np.any(ax<np.asarray(row_lower)-2e-6) or np.any(ax>np.asarray(row_upper)+2e-6)):
        raise RuntimeError('MILP incumbent failed independent linear/integrality validation')
    diagnostics['solution_kind'] = 'OPTIMAL' if result.status==0 else 'TIME_LIMIT_VALIDATED_INTEGER_INCUMBENT_NOT_OPTIMAL'
    delta = result.x[:K]
    if np.any(delta < -1e-7):
        raise RuntimeError('Solver returned a negative price adjustment')
    delta = np.maximum(delta, 0); delta[-1] = 0
    adjusted = utility - delta[None, :]
    choices = adjusted.argmax(1)
    best = adjusted[np.arange(L), choices]
    second = np.partition(adjusted, -2, axis=1)[:, -2]
    if not np.all(compatible[np.arange(L), choices]):
        raise RuntimeError('MILP solution failed independent compatibility validation')
    strict_rows = [l for l in range(L) if l not in keep_rows]
    if strict_rows and np.min((best-second)[strict_rows]) < margin-2e-6:
        raise RuntimeError('MILP solution failed independent utility-margin validation')
    diagnostics.update(minimum_verified_gap=float((best-second).min()),
                       maximum_delta=float(delta.max()), sum_delta=float(delta.sum()))
    return delta, diagnostics


class TransformedGemNet(nn.Module):
    def __init__(self, base, contexts, adjustments, metadata):
        super().__init__()
        self.base = base
        self.n_bidders, self.n_items = base.n_bidders, base.n_items
        self.menu_size = base.menu_size
        self.metadata = metadata
        self.transformed_bidders = sorted(int(i) for i in contexts)
        for i in self.transformed_bidders:
            self.register_buffer(f'contexts_{i}', torch.as_tensor(contexts[i], dtype=torch.float32))
            self.register_buffer(f'adjustments_{i}', torch.as_tensor(adjustments[i], dtype=torch.float32))

    def _delta(self, reports, bidder):
        if bidder not in self.transformed_bidders:
            return reports.new_zeros(reports.shape[0], self.menu_size)
        others = torch.cat((reports[:, :bidder], reports[:, bidder+1:]), 1).reshape(reports.shape[0], -1)
        table = getattr(self, f'adjustments_{bidder}')
        if self.metadata['grid_mode'] == 'cartesian':
            p, vmax = self.metadata['grid_points'], self.metadata['vmax']
            digits = (others / vmax * (p-1)).round().long().clamp(0, p-1)
            powers = torch.tensor([p**j for j in reversed(range(digits.shape[1]))], device=reports.device)
            indices = (digits * powers).sum(1)
        else:
            contexts = getattr(self, f'contexts_{bidder}')
            # L-infinity nearest grid context, excluding own bid by construction.
            indices = torch.cat([torch.cdist(chunk, contexts, p=float('inf')).argmin(1)
                                 for chunk in others.split(128)])
        return table[indices].to(reports.dtype)

    def bidder_menu(self, reports, bidder):
        a, p = self.base.bidder_menu(reports, bidder)
        return a, p + self._delta(reports, bidder)

    def menus(self, reports):
        entries = [self.bidder_menu(reports, i) for i in range(self.n_bidders)]
        return torch.stack([x[0] for x in entries], 1), torch.stack([x[1] for x in entries], 1)

    def forward(self, reports):
        a, p = self.menus(reports)
        utility = (a * reports[:, :, None, :]).sum(-1) - p
        k = utility.argmax(-1)
        return (a.gather(2, k[..., None, None].expand(-1, -1, 1, self.n_items)).squeeze(2),
                p.gather(2, k[..., None]).squeeze(2))

    @classmethod
    def from_file(cls, base, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            contexts = {i: data[f'contexts_{i}'].copy() for i in metadata['transformed_bidders']}
            adjustments = {i: data[f'adjustments_{i}'].copy() for i in metadata['transformed_bidders']}
        return cls(base, contexts, adjustments, metadata)


def transform_model(model, output, grid_points=7, grid_mode='cartesian',
                    context_count=64, own_count=128, seed=271828,
                    time_limit=300, margin_mode='paper', max_contexts=10000,
                    transform_variant='paper-accelerated', resume=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model.eval()
    n, m, K = model.n_bidders, model.n_items, model.menu_size
    dev = next(model.parameters()).device
    vmax = 1.0
    eps = vmax / (grid_points-1)
    if grid_mode == 'cartesian':
        ncontexts = grid_points ** ((n-1)*m)
        nown = grid_points ** m
        if ncontexts > max_contexts or nown > max_contexts:
            raise ValueError(f'Full grid would require {ncontexts} contexts and {nown} own types per MILP. '
                             'Use a smaller original-paper setting or explicitly choose --grid-mode sampled; '
                             'sampled mode has no continuous-domain coverage claim.')
        contexts_all = cartesian_grid(grid_points, (n-1)*m)
        own = cartesian_grid(grid_points, m)
    elif grid_mode == 'sampled':
        rng = np.random.default_rng(seed)
        contexts_all = np.vstack([np.zeros((1, (n-1)*m)), np.ones((1, (n-1)*m)),
                                  rng.random((max(0, context_count-2), (n-1)*m))])
        own = np.vstack([np.zeros((1, m)), np.ones((1, m)), rng.random((max(0, own_count-2), m))])
    else:
        raise ValueError('grid_mode must be cartesian or sampled')
    bounds = model.lipschitz_bounds()
    state_digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        state_digest.update(key.encode())
        state_digest.update(value.detach().cpu().numpy().tobytes())
    la, lp = bounds['allocation'], bounds['price']
    sf = n * eps * la / 2
    sm_paper = la*(m*vmax*eps + m*eps*eps/2) + eps*lp + eps
    sm = sm_paper if margin_mode == 'paper' else sm_paper + (m-1)*eps
    if sf >= 1:
        raise ValueError(f'Allocation safety margin {sf:g} >= 1; use a finer grid.')
    # The two-bidder construction changes only bidder 1; n>2 is sequential.
    order = [0] if n == 2 else list(range(n))
    metadata = dict(paper_url='https://arxiv.org/html/2406.07428v3',
                    implementation='paper_based_reimplementation_not_author_code',
                    n_bidders=n, n_items=m, menu_size=K, grid_mode=grid_mode,
                    model_state_sha256=state_digest.hexdigest(),
                    grid_points=grid_points, vmax=vmax, epsilon=eps,
                    own_type_count=len(own), context_count=len(contexts_all),
                    transformed_bidders=order, seed=seed, allocation_margin=sf,
                    utility_margin=sm, paper_utility_margin=sm_paper,
                    margin_mode=margin_mode, lipschitz=bounds,
                    nonnegative_price_changes=True, solver='SciPy/HiGHS',
                    transform_variant=transform_variant,
                    global_dsic_certified=False, adaptive_grid_implemented=False,
                    scope='finite_cartesian_grid' if grid_mode == 'cartesian' else 'sampled_context_extension',
                    limitation='Finite solver diagnostics and numerical norm bounds are not a continuous-domain DSIC certificate.')
    config_path = output/'transform_config.json'
    if resume:
        if not config_path.exists():
            raise FileNotFoundError('--resume needs transform_config.json and completed context arrays')
        previous = json.loads(config_path.read_text())
        for key in ('model_state_sha256','grid_mode','grid_points','seed','context_count','own_type_count',
                    'transform_variant','margin_mode','allocation_margin','utility_margin'):
            if previous[key] != metadata[key]:
                raise ValueError(f'Transformation resume changed {key}')
    else:
        config_path.write_text(json.dumps(metadata, indent=2))
    contexts, adjustments = {}, {}
    log_path = output/'milp_log.jsonl'
    if log_path.exists() and not resume:
        raise FileExistsError('Transformation log already exists; choose a fresh output directory.')
    for i in order:
        contexts[i] = contexts_all.copy()
        deltas = []
        if resume and (output/f'partial_bidder{i}.npy').exists():
            saved = np.load(output/f'partial_bidder{i}.npy', allow_pickle=False)
            if saved.ndim!=2 or saved.shape[1]!=K or len(saved)>len(contexts_all):
                raise ValueError('Invalid partial price adjustments')
            deltas = list(saved)
        for index, ctx in enumerate(contexts_all):
            if index < len(deltas):
                continue
            profiles = np.zeros((len(own), n, m), dtype=np.float32)
            others = [j for j in range(n) if j != i]
            profiles[:, others, :] = ctx.reshape(n-1, m)
            profiles[:, i, :] = own
            profiles = torch.from_numpy(profiles).to(dev)
            previous = TransformedGemNet(model, {j: contexts[j] for j in adjustments},
                                         adjustments, metadata).to(dev)
            with torch.no_grad():
                chosen = []
                for batch in profiles.split(128):
                    aa, _ = previous(batch)
                    chosen.append(aa.cpu().numpy())
                chosen = np.concatenate(chosen)
                bundles_t, prices_t = model.bidder_menu(profiles[:1], i)
                bundles, prices = bundles_t[0].cpu().numpy(), prices_t[0].cpu().numpy()
            other_sum = chosen[:, others].sum(1)
            cap = np.minimum(1-sf, other_sum)
            compatible = np.all(bundles[None, :, :] + cap[:, None, :] <= 1-sf+1e-10, axis=-1)
            compatible[:, -1] = True
            fractions = [x/100 for x in range(95, -1, -5)] if transform_variant == 'paper-accelerated' else [0.0]
            failed_fractions = []
            for fraction in fractions:
                try:
                    delta, report = solve_price_milp(own, bundles, prices, compatible, sm, time_limit,
                                                    keep_choice_fraction=fraction,
                                                    zero_violation_skip=transform_variant=='paper-accelerated')
                    report['infeasible_keep_choice_fractions'] = failed_fractions
                    break
                except RuntimeError as error:
                    if getattr(error, 'solver_status', None) == 2 and fraction>0:
                        failed_fractions.append(fraction)
                        continue
                    raise
            deltas.append(delta.astype(np.float32))
            report.update(bidder=i, context_index=index, context=ctx.tolist())
            with log_path.open('a') as stream:
                stream.write(json.dumps(report)+'\n')
            # The most recent completed context survives interrupted jobs.
            partial_path = output/f'partial_bidder{i}.npy'
            with partial_path.with_suffix('.tmp').open('wb') as stream:
                np.save(stream, np.asarray(deltas))
            partial_path.with_suffix('.tmp').replace(partial_path)
            print(f'GEMNET_MILP bidder={i} context={index+1}/{len(contexts_all)} '
                  f'binary={report["n_binary"]} seconds={report["seconds"]:.3f}', flush=True)
        adjustments[i] = np.asarray(deltas, dtype=np.float32)
    payload = {'metadata_json': np.asarray(json.dumps(metadata)), 'own_grid': own}
    for i in order:
        payload[f'contexts_{i}'] = contexts[i]
        payload[f'adjustments_{i}'] = adjustments[i]
    np.savez_compressed(output/'price_transform.npz', **payload)
    return TransformedGemNet(model, contexts, adjustments, metadata).to(dev), metadata
