"""Recompute the GemNet Figure3 type-profile slice from the actual V5 models.

Run: python -m common.decision_slices --root PHASE2_RESULTS --device auto --grid-size 51
This is a new shared visualization; it does not copy the paper's numerical plot.
"""
import argparse
import csv
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from .io import sha256, write_json

PAPER = 'https://arxiv.org/html/2406.07428v3'
METHODS = [
    ('RegretNet', 'RegretNet\nOptimaler checkpoint'),
    ('RegretFormer', 'RegretFormer\nauthor checkpoint'),
    ('AMenuNet', 'AMenuNet\npaper reimplementation'),
    ('GemNet', 'GemNet + price transform\npaper reimplementation'),
]


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def make_profiles(grid_size):
    if grid_size < 2:
        raise ValueError('grid_size must be >=2')
    axis = np.linspace(0., 1., grid_size, dtype=np.float32)
    item1, item2 = np.meshgrid(axis, axis, indexing='xy')
    values = np.empty((grid_size*grid_size, 2, 2), dtype=np.float32)
    values[:, 0, 0] = item1.ravel()
    values[:, 0, 1] = item2.ravel()
    values[:, 1] = (0., .6)
    return axis, values


def inherit_common_audit(root, entry, metadata):
    """Failure is monotone: a narrower slice cannot rehabilitate a failed audit."""
    directory = {
        'RegretNet': 'RegretNet_OptimalerBudget_2x2',
        'RegretFormer': 'RegretFormer_AuthorsCheckpoint_2x2',
        'AMenuNet': 'AMenuNet_2x2',
        'GemNet': 'GemNet_2x2',
    }[entry['key']]
    folder = root/'evaluation'/directory/'seed0'
    status_path, metrics_path = folder/'status.json', folder/'metrics.csv'
    provenance_path = folder/'provenance.json'
    status, provenance = read_json(status_path), read_json(provenance_path)
    rows = []
    if metrics_path.is_file():
        with metrics_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
    checkpoint_hash = metadata.get('checkpoint_sha256')
    recorded_hashes = [provenance.get('checkpoint_sha256')]+[r.get('checkpoint_sha256') for r in rows]
    if any(h and h != checkpoint_hash for h in recorded_hashes):
        raise ValueError('Common-audit checkpoint hash does not match the plotted checkpoint')
    if metadata.get('transform_sha256') and provenance.get('transform_sha256') not in (None, metadata['transform_sha256']):
        raise ValueError('Common audit refers to a different GemNet price transformation')
    reasons = []
    if status.get('mechanism_valid_on_checked_reports') is False:
        reasons.append('mechanism_valid_on_checked_reports=false')
    if status and status.get('state') != 'COMPLETED':
        reasons.append('common audit did not complete')
    if str(status.get('result_use', '')).startswith('diagnostic_only'):
        reasons.append('result_use='+status['result_use'])
    boolean_checks = ('feasible_on_checked_reports', 'ir_on_truthful_profiles',
                      'no_positive_transfers_on_checked_reports', 'transform_diagnostic_valid')
    maxima = ('max_truthful_ir_violation', 'max_bidder_ir_violation', 'max_overallocation',
              'max_negative_allocation', 'max_negative_payment')
    for row in rows:
        label = row.get('audit', 'unknown')
        for key in boolean_checks:
            if str(row.get(key, '')).strip().lower() in ('false', '0'):
                reasons.append(label+': '+key+'=false')
        if row.get('evidence_status', '').startswith('diagnostic_only'):
            reasons.append(label+': '+row['evidence_status'])
        for key in maxima:
            if row.get(key) not in (None, '') and float(row[key]) > 1e-5:
                reasons.append(label+': '+key+'='+row[key])
    reasons = list(dict.fromkeys(reasons))
    metadata['common_audit'] = {
        'directory': str(folder), 'available': bool(status or rows),
        'status': status, 'inherited_failure_reasons': reasons,
        'source_files': {p.name: {'path': str(p), 'sha256': sha256(p)}
                         for p in (status_path, metrics_path, provenance_path) if p.is_file()},
        'row_checks': [{k: row.get(k) for k in ('audit', 'evidence_status', 'checkpoint_sha256',
                      *boolean_checks, *maxima)} for row in rows],
    }
    if reasons:
        entry['diagnostic_only'] = True
        message = 'common audit failed: '+'; '.join(reasons)
        entry['diagnostic_reason'] = '; '.join(x for x in (entry.get('diagnostic_reason'), message) if x)


def load_models(root, device):
    """Keep all four requested columns, including explicit unavailable cases."""
    from .official import load_official
    from phase1.models import load_checkpoint as load_phase1
    from gemnet.model import load_checkpoint as load_gemnet
    entries = []
    for method, title in METHODS:
        entry = {'key': method, 'title': title, 'model': None, 'metadata': {},
                 'diagnostic_only': False}
        try:
            if method in ('RegretNet', 'RegretFormer'):
                model, metadata = load_official(method, 0, '2x2', device)
            elif method == 'AMenuNet':
                checkpoint = root/'training/AMenuNet_2x2/seed0/checkpoint.pt'
                model, metadata = load_phase1(checkpoint, device)
                if metadata['config']['model']['n'] != 2 or metadata['config']['model']['m'] != 2:
                    raise ValueError('AMenuNet checkpoint is not the required2x2 model')
                metadata.update(checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                                source_role='paper_reimplementation', seed_role='new_training_seed')
            else:
                checkpoint = root/'training/GemNet_2x2/seed0/checkpoint.pt'
                transform_dir = root/'transforms/GemNet_2x2/seed0'
                transform_path = transform_dir/'price_transform.npz'
                status_path = transform_dir/'transform_status.json'
                transform_status = read_json(status_path)
                checkpoint_hash = sha256(checkpoint)
                if transform_status.get('checkpoint_sha256') not in (None, checkpoint_hash):
                    raise ValueError('GemNet transform was computed for a different model checkpoint')
                model = load_gemnet(checkpoint, device, transform_path)
                if model.n_bidders != 2 or model.n_items != 2:
                    raise ValueError('GemNet checkpoint is not the required2x2 model')
                metadata = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
                    'source_role': 'paper_reimplementation_with_price_transform', 'source_url': PAPER,
                    'seed': 0, 'seed_role': 'new_training_seed',
                    'transform_path': str(transform_path), 'transform_sha256': sha256(transform_path),
                    'transform_status': transform_status,
                    'transform_status_sha256': sha256(status_path) if status_path.is_file() else None}
                validity = transform_status.get('validity', 'UNKNOWN')
                entry['diagnostic_only'] = (transform_status.get('status') != 'COMPLETED' or
                                            validity != 'FINITE_CHECKS_PASSED_NOT_GLOBAL_CERTIFICATE')
                if entry['diagnostic_only']:
                    entry['diagnostic_reason'] = f'GemNet transformation validity={validity}'
            inherit_common_audit(root, entry, metadata)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            entry.update(model=model, metadata=metadata)
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            entry['traceback'] = traceback.format_exc()
        entries.append(entry)
    return entries


def evaluate_slice(model, values, device, batch_size=128):
    allocations, payments = [], []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start+batch_size], device=device)
            allocation, payment = model(batch)
            if allocation.shape != batch.shape or payment.shape != batch.shape[:2]:
                raise ValueError('Invalid model output dimensions for a2x2 slice')
            if not torch.isfinite(allocation).all() or not torch.isfinite(payment).all():
                raise FloatingPointError('Nonfinite model output; visualization was not imputed')
            allocations.append(allocation.cpu().double().numpy())
            payments.append(payment.cpu().double().numpy())
    allocation = np.concatenate(allocations)
    payment = np.concatenate(payments)
    utility = (allocation*values.astype('float64')).sum(-1)-payment
    metrics = {'n_profiles': len(values),
               'allocation_excess_max': float(np.maximum(allocation.sum(1)-1, 0).max()),
               'allocation_negative_max': float(np.maximum(-allocation, 0).max()),
               'payment_negative_max': float(np.maximum(-payment, 0).max()),
               'ir_violation_max': float(np.maximum(-utility, 0).max()),
               'allocation_tolerance': 1e-6, 'utility_tolerance': 1e-6}
    metrics['constraint_violation_on_slice'] = any(metrics[k] > 1e-6 for k in (
        'allocation_excess_max', 'allocation_negative_max', 'payment_negative_max', 'ir_violation_max'))
    return allocation, payment, metrics


def render_slices(entries, output, grid_size=51, device='cpu', smoke=False, batch_size=128):
    """Reusable plot function; fixtures must pass smoke=True and fixture metadata."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    axis, values = make_profiles(grid_size)
    raw = {'axis_item1': axis, 'axis_item2': axis, 'value_profiles': values,
           'fixed_bidder2_values': np.array([0., .6], dtype=np.float32)}
    columns, all_payments = [], []
    for original in entries:
        entry = {k: v for k, v in original.items() if k != 'model'}
        model = original.get('model')
        if model is not None:
            try:
                allocation, payment, metrics = evaluate_slice(model, values, device, batch_size)
                raw[entry['key']+'_allocation'] = allocation.reshape(grid_size, grid_size, 2, 2)
                raw[entry['key']+'_payment'] = payment.reshape(grid_size, grid_size, 2)
                entry['slice_diagnostics'] = metrics
                if metrics['constraint_violation_on_slice']:
                    entry['diagnostic_only'] = True
                    entry['diagnostic_reason'] = entry.get('diagnostic_reason', '')+'; constraint violation on displayed slice'
                all_payments.append(payment[:, 0])
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
                entry['traceback'] = traceback.format_exc()
        elif 'error' not in entry:
            entry['error'] = 'Model not available'
        columns.append(entry)
    smoke = smoke or any(c.get('metadata', {}).get('is_smoke', False) for c in columns)
    paid = np.concatenate(all_payments) if all_payments else np.array([0., 1.])
    p_min, p_max = min(0., float(paid.min())), max(0., float(paid.max()))
    if p_min == p_max:
        p_max = p_min+1.
    normalization = [Normalize(0., 1.), Normalize(0., 1.), Normalize(p_min, p_max)]
    row_labels = ['Allocation probability\n(item 1 to bidder 1)',
                  'Allocation probability\n(item 2 to bidder 1)', 'Payment by bidder 1']
    fig, axes = plt.subplots(3, len(columns), figsize=(3.05*len(columns)+.9, 9.1),
                             squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.88, bottom=.19, top=.86, wspace=.15, hspace=.31)
    for col, entry in enumerate(columns):
        title = entry['title']
        if entry.get('diagnostic_only'):
            title += '\nDIAGNOSTIC ONLY'
        if 'error' in entry:
            title += '\nUNAVAILABLE'
        axes[0, col].set_title(title, fontsize=10,
                              color='#a32626' if entry.get('diagnostic_only') or 'error' in entry else '#222222', pad=10)
        for row in range(3):
            ax = axes[row, col]
            if 'error' in entry:
                ax.set_facecolor('#f2f2f2')
                ax.text(.5, .5, 'No verified output\nSee metadata.json', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#555555')
            else:
                a = raw[entry['key']+'_allocation']
                p = raw[entry['key']+'_payment']
                surface = a[:, :, 0, row] if row < 2 else p[:, :, 0]
                ax.pcolormesh(axis, axis, surface, shading='nearest', cmap='viridis', norm=normalization[row], rasterized=True)
                if entry.get('diagnostic_only'):
                    for spine in ax.spines.values():
                        spine.set_color('#a32626')
                        spine.set_linewidth(1.4)
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            ax.set_aspect('equal')
            ax.set_xticks([0., .5, 1.])
            ax.set_yticks([0., .5, 1.])
            ax.tick_params(labelsize=9)
            if col == 0:
                ax.set_ylabel('Bidder 1 value for item 2', fontsize=9)
            if row == 2:
                ax.set_xlabel('Bidder 1 value for item 1', fontsize=9)
    for row in range(3):
        position = axes[row, -1].get_position()
        color_axis = fig.add_axes([.901, position.y0, .015, position.height])
        bar = fig.colorbar(ScalarMappable(norm=normalization[row], cmap='viridis'), cax=color_axis)
        bar.ax.tick_params(labelsize=8)
        bar.set_label(row_labels[row], fontsize=9)
    label = 'SMOKE — pipeline check; not paper results' if smoke else 'Shared mechanism slice — checkpoint seed0'
    fig.suptitle(label+'\n2 additive bidders × 2 items; bidder 2 fixed at (0, 0.6)', fontsize=13, y=.98)
    fig.text(.075, .055,
             'Same units and shared color limits within each row. No cross-setting normalization.\n'
             'This finite slice visualizes mechanism behavior; it does not establish global IC, feasibility, or human interpretability.\n'
             'New outputs from the listed checkpoints; scenario follows GemNet Figure3, not its reported numerical plot.',
             fontsize=8.3, va='bottom', linespacing=1.6)
    stem = 'decision_slices_2x2_seed0'
    png, pdf, npz = (output/(stem+ext) for ext in ('.png', '.pdf', '.npz'))
    fig.savefig(png, dpi=180, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    raw['method_order'] = np.array([c['key'] for c in columns])
    np.savez_compressed(npz, **raw)
    missing = [c['key'] for c in columns if 'error' in c]
    metadata = {'state': 'PARTIAL_MISSING_MODELS' if missing else 'COMPLETED',
        'is_smoke': bool(smoke), 'setting': '2x2 iidU[0,1], additive', 'checkpoint_seed': 0,
        'scenario_source': PAPER, 'scenario_figure': 'Figure3',
        'scenario': {'x': 'bidder1 item1 value', 'y': 'bidder1 item2 value',
                     'range': [0., 1.], 'grid_size': grid_size,
                     'fixed_bidder2_values': [0., .6],
                     'saved_profiles_axis_order': 'y_index, x_index, bidder, item'},
        'color_ranges': {'allocation_item1': [0., 1.], 'allocation_item2': [0., 1.], 'payment': [p_min, p_max]},
        'columns': columns, 'missing_models': missing,
        'source_script_sha256': sha256(__file__),
        'value_profiles_sha256_raw_float32': __import__('hashlib').sha256(values.tobytes()).hexdigest(),
        'artifacts': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in (png, pdf, npz)},
        'claim_limit': 'Finite mechanism visualization only; no global IC/feasibility or human-understanding inference. Diagnostic columns are retained and explicitly labelled.'}
    write_json(output/'metadata.json', metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--grid-size', type=int, default=51)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.grid_size < 2 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error('grid-size>=2, batch-size>=1, cpu-threads>=1 are required')
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('--root must be an existing PhaseII results directory')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_json(root/'RUN_PLAN.json')
    smoke = plan.get('mode') == 'smoke' or (not plan and 'smoke' in root.name.lower())
    entries = load_models(root, device)
    output = root/'reports'/'decision_slices'
    metadata = render_slices(entries, output, args.grid_size, device, smoke, args.batch_size)
    print(json.dumps({'state': metadata['state'], 'output': str(output),
                      'is_smoke': metadata['is_smoke'], 'missing_models': metadata['missing_models']}), flush=True)
    return 2 if metadata['missing_models'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
