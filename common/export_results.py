"""Export source-traceable, task-local tables and figures from common.evaluate.

No training is run here. Unknown CSV formats are reported, never guessed. Human
scores and cross-task normalization are deliberately absent. Run with:
    python -m common.export_results --root OUT --phase phase1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORE = ('revenue', 'welfare', 'regret_mean', 'regret_max', 'pmr')
STRUCTURE = ('parameter_count', 'menu_size', 'checkpoint_file_bytes', 'price_transform_file_bytes')
CHECKS = ('max_truthful_ir_violation', 'max_overallocation',
          'max_negative_allocation', 'max_negative_payment')
DIAGNOSTICS = (*CHECKS, 'max_bidder_ir_violation', 'ir_violation_fraction')
FAMILY_METRICS = ('paper_unfairness_sum', 'absolute_tvf_sum', 'maximum_l1_allocation_distance',
                  'entropy', 'minimum_normalized_quota', 'tvf_constraint_satisfaction_1e3',
                  'pca', 'preference_ground_truth_score', 'preference_probability',
                  'preference_classifier_probability', 'synthetic_preference_score',
                  'synthetic_proportional_share_gap')
NONMETRICS = {'seed', 'n_profiles', 'n_bidders', 'n_items', 'scope', 'audit',
              'method', 'family', 'setting', 'seed_role', 'source_role',
              'feasible_on_checked_reports', 'ir_on_truthful_profiles',
              'evidence_status', 'checkpoint_sha256', 'test_profile_sha256',
              'evaluation_path', 'metrics_source', 'protocol_id', 'phase',
              'audit_inputs_sha256', 'no_positive_transfers_on_checked_reports',
              'transform_diagnostic_valid', 'training_budget_modified',
              'aggregate_comparison_eligible', 'aggregate_exclusion_reason',
              'group_failed_seed_count', 'group_completed_seed_count'}
FORBIDDEN_METRICS = {'human_proxy_score', 'human_composite', 'human_score',
                     'human_comp', 'human_burden_composite', 'normalized_score'}
ID_FIELDS = ('family', 'setting', 'method', 'seed_role', 'source_role', 'scope', 'protocol_id')
BASE_GROUP = ('family', 'setting', 'scope', 'protocol_id')
SEED_ROLE_LABELS = {'new_training_seed': 'training seeds',
                    'author_checkpoint': 'author checkpoints',
                    'author_checkpoint_directory_label': 'author checkpoints',
                    'author_checkpoint_seed': 'author checkpoints',
                    'released_checkpoint_seed': 'author checkpoints'}
COLORS = ('#315b73', '#b87936', '#666666', '#6f628a', '#51725d', '#944c55')


def load_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: '' if row.get(k) is None else row.get(k, '') for k in fields}
                        for row in rows)


def number(value):
    if value in (None, '', 'NA', 'N/A', 'null', 'None') or isinstance(value, bool):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ValueError(f'Not a boolean: {value!r}')


def stable_slug(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')[:110] or 'unnamed'


def short_name(value):
    aliases = {'AMenuNet_PaperReimplementation': 'AMenuNet (paper reimplementation)',
               'RegretNet_PaperReimplementation': 'RegretNet (paper reimplementation)',
               'GemNet_PaperReimplementation': 'GemNet (paper reimplementation)',
               'RegretFormer_AuthorsCheckpoint': 'RegretFormer (author model)',
               'RegretNet_OptimalerBudget': 'RegretNet (Optimaler budget)',
               'RegretNetPreferenceBaseline': 'RegretNet preference baseline',
               'RegretNetProportionBaseline': 'RegretNet proportion baseline'}
    text = aliases.get(str(value), str(value).replace('_', ' '))
    return '\n'.join(textwrap.wrap(text, width=34))


def tex_escape(value):
    subs = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}'}
    return ''.join(subs.get(c, c) for c in str(value))


def fmt(value):
    if value is None or not math.isfinite(float(value)):
        return 'NA'
    v = float(value)
    if v == 0:
        return '0'
    if abs(v) < .001 or abs(v) >= 100000:
        return f'{v:.3g}'
    return f'{v:.4g}'


def audit_order(value):
    if value.startswith('random_'):
        return (0, int(value.split('_')[-1]))
    return (1 if value == 'strong' else 2, value)


def role_label(value):
    return SEED_ROLE_LABELS.get(value, value.replace('_', ' '))


def protocol_id(config, provenance):
    keys = ('samples', 'n_bidders', 'n_items', 'data_seed', 'candidate_seed',
            'random_budgets', 'pgd_restarts', 'pgd_steps', 'pgd_lr', 'pmr_threshold',
            'test_profile_sha256', 'audit_inputs_sha256', 'context_protocol')
    protocol = {k: config.get(k) for k in keys}
    protocol['evaluation_source'] = {k: v for k, v in provenance['evaluation_code_sha256'].items()
                                     if k in ('common/evaluate.py', 'common/audit.py')}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12], protocol


def find_results(root, phase):
    """Require the common evaluator's metadata instead of accepting trainer CSVs."""
    accepted, reports, protocols, seen = [], [], {}, set()
    for path in sorted(Path(root).rglob('metrics.csv')):
        folder = path.parent
        try:
            required = [folder / n for n in ('status.json', 'config.json', 'provenance.json')]
            if not all(p.is_file() for p in required):
                reports.append({'path': str(path), 'status': 'skipped_non_evaluator',
                                'reason': 'Missing common evaluator status/config/provenance'})
                continue
            status, config, provenance = [load_json(p) for p in required]
            hashes = provenance.get('evaluation_code_sha256', {})
            if 'common/evaluate.py' not in hashes:
                reports.append({'path': str(path), 'status': 'skipped_non_evaluator',
                                'reason': 'No common/evaluate.py in evaluation_code_sha256'})
                continue
            if status.get('state') != 'COMPLETED':
                reports.append({'path': str(path), 'status': 'skipped_incomplete',
                                'reason': f"Evaluator state={status.get('state')}"})
                continue
            pid, protocol = protocol_id(config, provenance)
            protocols[pid] = protocol
            with path.open(encoding='utf-8-sig', newline='') as stream:
                local = list(csv.DictReader(stream))
            if not local:
                raise ValueError('Empty completed metrics.csv')
            prepared = []
            for raw in local:
                for key in (*ID_FIELDS[:-1], 'seed', 'audit', 'n_profiles', 'n_bidders',
                            'n_items', 'evidence_status', 'checkpoint_sha256', *CORE):
                    if key not in raw or raw[key] == '':
                        raise ValueError(f'Missing required field {key}')
                if raw['scope'] != config.get('scope'):
                    raise ValueError('Row scope differs from evaluator config')
                if raw['method'] != provenance.get('method'):
                    raise ValueError('Row method differs from provenance')
                if str(raw['seed']) != str(provenance.get('seed')):
                    raise ValueError('Row seed differs from provenance')
                if raw.get('test_profile_sha256') != config.get('test_profile_sha256'):
                    raise ValueError('Row test data hash differs from config')
                if raw['checkpoint_sha256'] != provenance.get('checkpoint_sha256'):
                    raise ValueError('Row checkpoint hash differs from provenance')
                if float(config.get('pmr_threshold', 1e-6)) != 1e-6:
                    raise ValueError('Exporter PMR caption requires threshold 1e-6')
                allowed = [f'random_{b}' for b in config['random_budgets']] + ['strong']
                if raw['audit'] not in allowed:
                    raise ValueError(f"Unrecognized audit {raw['audit']}")
                row = dict(raw)
                row.update({'protocol_id': pid, 'metrics_source': str(path.resolve()), 'phase': phase,
                            'audit_inputs_sha256': config.get('audit_inputs_sha256', '')})
                for key in (*CORE, *STRUCTURE, *CHECKS, 'n_profiles', 'n_bidders', 'n_items'):
                    val = number(raw.get(key))
                    if key in (*CORE, *CHECKS, 'n_profiles', 'n_bidders', 'n_items') and val is None:
                        raise ValueError(f'Non-finite or missing required numeric field {key}')
                    row[key] = val
                for key, config_key in (('n_profiles', 'samples'), ('n_bidders', 'n_bidders'), ('n_items', 'n_items')):
                    if row[key] != config.get(config_key):
                        raise ValueError(f'Row {key} differs from evaluation config')
                if row['regret_mean'] < 0 or row['regret_max'] < row['regret_mean'] - 1e-10:
                    raise ValueError('Invalid nonnegative regret mean/max ordering')
                if row['regret_max'] > row['n_profiles'] * row['n_bidders'] * row['regret_mean'] + 1e-8:
                    raise ValueError('max regret exceeds B*n*mean regret')
                if not 0 <= row['pmr'] <= 1:
                    raise ValueError('PMR outside [0,1]')
                feasible = boolean(raw['feasible_on_checked_reports'])
                ir = boolean(raw['ir_on_truthful_profiles'])
                claim = raw['evidence_status'] == 'empirical_comparison'
                computed = (row['max_overallocation'] <= 1e-5 and
                            row['max_negative_allocation'] <= 1e-5 and
                            row['max_truthful_ir_violation'] <= 1e-5 and
                            row['max_negative_payment'] <= 1e-5)
                transform_ok = boolean(raw.get('transform_diagnostic_valid',
                    provenance.get('transform_diagnostics', {}).get('valid_for_empirical_revenue_comparison', True)))
                if ('no_positive_transfers_on_checked_reports' in raw and
                    boolean(raw['no_positive_transfers_on_checked_reports']) != (row['max_negative_payment'] <= 1e-5)):
                    raise ValueError('Negative payment flag differs from measured diagnostic')
                computed = computed and transform_ok
                if claim != (feasible and ir and computed):
                    raise ValueError('Constraint evidence status is inconsistent with diagnostics')
                if raw['evidence_status'] not in ('empirical_comparison', 'diagnostic_only_constraint_violation'):
                    raise ValueError('Unknown evidence_status')
                for key in FORBIDDEN_METRICS:
                    row.pop(key, None)
                prepared.append(row)
            if len({r['audit'] for r in prepared}) != len(prepared) or set(r['audit'] for r in prepared) != set(allowed):
                raise ValueError('Duplicate/missing audit rows in one evaluation')
            # Frozen-model performance must be independent of the regret search budget.
            for key in ('revenue', 'welfare'):
                if max(r[key] for r in prepared) - min(r[key] for r in prepared) > 1e-10:
                    raise ValueError(f'{key} differs across audits of the same frozen model')
            ordered = sorted(prepared, key=lambda r: audit_order(r['audit']))
            for left, right in zip(ordered, ordered[1:]):
                for key in ('regret_mean', 'regret_max', 'pmr'):
                    if right[key] + 1e-9 < left[key]:
                        raise ValueError(f'Nested audit {key} decreased')
            signature = tuple((k, prepared[0][k]) for k in ID_FIELDS) + (
                ('seed', prepared[0]['seed']), ('checkpoint', prepared[0]['checkpoint_sha256']))
            if signature in seen:
                reports.append({'path': str(path), 'status': 'skipped_duplicate',
                                'reason': 'Identical method/setting/seed/checkpoint/protocol already loaded'})
                continue
            seen.add(signature)
            accepted.extend(prepared)
            reports.append({'path': str(path), 'status': 'included', 'rows': len(prepared),
                            'evidence_status': prepared[0]['evidence_status']})
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            reports.append({'path': str(path), 'status': 'invalid_evaluation', 'reason': str(exc)})
    return accepted, reports, protocols


def metric_fields(rows):
    fields = list(CORE + STRUCTURE + CHECKS)
    for row in rows:
        for key, value in row.items():
            if key not in NONMETRICS and key not in FORBIDDEN_METRICS and key not in fields:
                if number(value) is not None and not key.endswith('_sha256'):
                    fields.append(key)
    return fields


def apply_group_eligibility(rows):
    """Never turn a failed-seed experiment into a survivor-only comparison.

    Preserve every row's measured evidence_status. One failed completed seed
    excludes its entire method/setting/protocol group from comparison exports.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ID_FIELDS)].append(row)
    result = []
    for members in groups.values():
        units = {(str(r['seed']), r['checkpoint_sha256']) for r in members}
        failed = {(str(r['seed']), r['checkpoint_sha256']) for r in members
                  if r['evidence_status'] != 'empirical_comparison'}
        reason = (f'{len(failed)} of {len(units)} completed seeds/checkpoints failed constraints or transformation checks; '
                  'the entire method/setting/protocol group is diagnostic-only, with no survivor filtering.') if failed else ''
        for row in members:
            result.append({**row, 'aggregate_comparison_eligible': not failed,
                           'aggregate_exclusion_reason': reason,
                           'group_failed_seed_count': len(failed),
                           'group_completed_seed_count': len(units)})
    return result


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ID_FIELDS) + (row['audit'],)].append(row)
    output = []
    fields = metric_fields(rows)
    for key, members in sorted(groups.items()):
        seeds = [str(r['seed']) for r in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Different checkpoints with duplicate seed labels in one aggregate: {key}')
        for metric in fields:
            vals = [number(r.get(metric)) for r in members]
            known = np.asarray([v for v in vals if v is not None], dtype=float)
            # Retain NA even when an entire structural metric is unavailable.
            output.append({**dict(zip(ID_FIELDS, key[:-1])), 'audit': key[-1], 'metric': metric,
                           'n_units': len(members), 'n_available': len(known),
                           'unit_role': role_label(key[3]), 'seed_labels': ';'.join(seeds),
                           'mean': float(known.mean()) if len(known) else None,
                           'sd_across_units': float(known.std(ddof=1)) if len(known) > 1 else None,
                           'sd_interpretation': 'sample SD across listed ' + role_label(key[3]) + '; not CI',
                           'missing_values': len(members) - len(known),
                           'aggregate_comparison_eligible': members[0].get('aggregate_comparison_eligible', True),
                           'aggregate_exclusion_reason': members[0].get('aggregate_exclusion_reason', ''),
                           'group_failed_seed_count': members[0].get('group_failed_seed_count', 0),
                           'group_completed_seed_count': members[0].get('group_completed_seed_count', len(members))})
    return output


def artifact_provenance(rows):
    """Flatten recorded provenance only; unknown training/source facts stay blank."""
    results, seen = [], set()
    for row in rows:
        path = Path(row['metrics_source']).parent
        identity_key = (str(path), row['checkpoint_sha256'], row['protocol_id'])
        if identity_key in seen:
            continue
        seen.add(identity_key)
        prov, cfg = load_json(path / 'provenance.json'), load_json(path / 'config.json')
        source = prov.get('source') or {}
        if not isinstance(source, dict):
            source = {}
        train = prov.get('config') or prov.get('training_config') or {}
        preset = train.get('preset') if isinstance(train.get('preset'), dict) else {}
        author_train = (prov.get('author_model_config') or {}).get('train') or {}
        budget = {k: v for k, v in train.items() if k in {
            'preset', 'steps', 'batch_size', 'train_profiles', 'profiles_per_round', 'learning_rate',
            'inner_steps', 'inner_lr', 'warmup_rounds', 'custom_budget', 'new_setting_extension'}}
        budget.update({k: prov[k] for k in ('epochs', 'iterations', 'mode') if k in prov})
        record = {k: row.get(k, '') for k in (*ID_FIELDS, 'seed', 'evidence_status',
                                             'checkpoint_sha256', 'test_profile_sha256',
                                             'aggregate_comparison_eligible', 'aggregate_exclusion_reason',
                                             'group_failed_seed_count', 'group_completed_seed_count')}
        record.update({
            'seed_label_interpretation': role_label(row['seed_role']),
            'source_url': prov.get('source_url') or source.get('repository') or source.get('paper', ''),
            'paper_url': prov.get('paper_url') or source.get('paper', ''),
            'source_commit': prov.get('source_commit') or source.get('commit', ''),
            'implementation': prov.get('implementation') or source.get('implementation') or prov.get('reimplementation_scope') or prov.get('method_label', ''),
            'training_protocol': prov.get('training_protocol') or prov.get('protocol') or train.get('protocol', ''),
            'observed_checkpoint_step': prov.get('step', prov.get('selected_checkpoint_step', '')),
            'configured_training_steps': train.get('steps', preset.get('steps', '')),
            'recorded_epochs': prov.get('epochs', ''),
            'recorded_iterations': prov.get('iterations', ''),
            'training_budget_json': json.dumps(budget, sort_keys=True) if budget else '',
            'training_budget_modified': row.get('training_budget_modified', ''),
            'training_configuration_json': json.dumps(train, sort_keys=True) if train else '',
            'author_config_max_iter': author_train.get('max_iter', ''),
            'author_config_template_seed': author_train.get('seed', ''),
            'author_training_configuration_json': json.dumps(author_train, sort_keys=True) if author_train else '',
            'checkpoint_path': prov.get('checkpoint_path', ''),
            'checkpoint_file_bytes': row.get('checkpoint_file_bytes', ''),
            'price_transform_sha256': prov.get('transform_sha256', ''),
            'price_transform_file_bytes': row.get('price_transform_file_bytes', ''),
            'global_continuous_certificate': prov.get('global_continuous_certificate', ''),
            'eval_protocol_json': json.dumps({k: cfg[k] for k in (
                'samples', 'n_bidders', 'n_items', 'data_seed', 'candidate_seed', 'random_budgets',
                'pgd_restarts', 'pgd_steps', 'pgd_lr', 'pmr_threshold', 'context_protocol') if k in cfg}, sort_keys=True),
            'source_manifest_sha256': prov.get('source_manifest_sha256', ''),
            'adapter_code_sha256_json': json.dumps(prov['adapter_code_sha256'], sort_keys=True) if 'adapter_code_sha256' in prov else '',
            'evaluation_code_sha256_json': json.dumps(prov.get('evaluation_code_sha256', {}), sort_keys=True),
            'provenance_path': str(path / 'provenance.json'),
            'budget_note': 'Author configuration is a recorded template, not proof of this checkpoint training seed or executed updates.' if author_train else '',
        })
        results.append(record)
    return results


def identity(row):
    return tuple(row[k] for k in ('method', 'seed_role', 'source_role'))


def unit_label(key):
    return f'{short_name(key[0])}\n[{role_label(key[1])}]'


def stats(rows, metric):
    vals = np.array([number(r.get(metric)) for r in rows if number(r.get(metric)) is not None], dtype=float)
    return (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else None) if len(vals) else (None, None)


def title_for(group, phase, diagnostic=False):
    family, setting, scope, pid = group
    prefix = 'SMOKE TEST — NOT PAPER RESULTS | ' if scope == 'smoke_only' else ''
    if diagnostic:
        prefix += 'DIAGNOSTIC ONLY — GROUP CONSTRAINT FAILURE | '
    return f'{prefix}{phase}\n{family} / {setting} | protocol {pid}'


def save_figure(fig, base, footer):
    fig.text(.5, .012, footer, ha='center', va='bottom', fontsize=8, color='#444444', wrap=True)
    fig.tight_layout(rect=(0, .075, 1, .9))
    fig.savefig(str(base) + '.png', dpi=180, facecolor='white')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight', metadata={'Creator': 'HC-XMD source-based exporter'})
    plt.close(fig)


def point_panel(ax, grouped, metric, label):
    keys = list(grouped)
    for i, key in enumerate(keys):
        mean, sd = stats(grouped[key], metric)
        if mean is None:
            ax.text(.02, i, 'NA', transform=ax.get_yaxis_transform(), va='center', color='#666666')
        else:
            ax.errorbar(mean, i, xerr=sd, fmt='o', color=COLORS[i % len(COLORS)],
                        capsize=3, markersize=5, linewidth=1.2)
    labels = []
    for key in keys:
        one = grouped[key][0]
        tick_label = unit_label(key)
        if not one.get('aggregate_comparison_eligible', True):
            tick_label += f"\n[failed units: {one['group_failed_seed_count']}/{one['group_completed_seed_count']}]"
        labels.append(tick_label)
    ax.set_yticks(range(len(keys)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(label, fontsize=9)
    ax.grid(axis='x', color='#e4e4e4', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if metric == 'pmr':
        ax.set_xlim(-.02, 1.02)
    elif metric.startswith('regret'):
        values = [number(r.get(metric)) for rs in grouped.values() for r in rs]
        positive = [v for v in values if v is not None and v > 0]
        if positive and max(positive) / max(min(positive), 1e-15) > 100:
            ax.set_xscale('symlog', linthresh=1e-6)
            ax.set_xlabel(label + '\n(symlog; linear below 1e-6)', fontsize=8)
        else:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)


def make_group_figures(rows, group, directory, phase, diagnostic=False):
    family, setting, scope, pid = group
    suffix = '_diagnostic' if diagnostic else ''
    stem = stable_slug(f'{family}_{setting}_{scope}_{pid}{suffix}')
    strongest = [r for r in rows if r['audit'] == 'strong']
    grouped = defaultdict(list)
    for row in strongest:
        grouped[identity(row)].append(row)
    grouped = dict(sorted(grouped.items()))
    outputs = []
    if not grouped:
        return outputs
    height = max(3.6, 1.9 + .86 * len(grouped))
    panels = [
        ('performance', [('revenue', 'Revenue'), ('welfare', 'Welfare')]),
        ('incentives', [('regret_mean', 'Mean bidder-profile regret'),
                        ('regret_max', 'Maximum searched regret'), ('pmr', 'PMR (> 1e-6)')]),
        ('structure', [('parameter_count', 'Model parameter count'), ('menu_size', 'Declared menu size')]),
        ('artifact_sizes', [('checkpoint_file_bytes', 'Checkpoint file size (bytes)'),
                            ('price_transform_file_bytes', 'Price-transform file size (bytes)')]),
    ]
    if diagnostic or any(number(r.get(metric)) not in (None, 0) for r in strongest for metric in DIAGNOSTICS):
        panels.append(('constraints', [('max_overallocation', 'Max allocation excess'),
                                       ('max_negative_allocation', 'Max negative allocation'),
                                       ('max_truthful_ir_violation', 'Max truthful IR violation')]))
        panels.append(('payment_ir_diagnostics', [('max_negative_payment', 'Max negative payment'),
                        ('max_bidder_ir_violation', 'Max bidder IR violation'),
                        ('ir_violation_fraction', 'Fraction of truthful IR violations')]))
    for name, metrics in panels:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), height), squeeze=False)
        for ax, (metric, label) in zip(axes[0], metrics):
            point_panel(ax, grouped, metric, label)
        fig.suptitle(title_for(group, phase, diagnostic), fontsize=11)
        extra = 'Structure is measured code/model size, not human understanding. ' if name in ('structure', 'artifact_sizes') else ''
        footer = extra + 'Dots: unit means; bars: SD across separately labelled seeds/checkpoints. NA is unavailable; no CI or ranking.'
        base = directory / (stem + '_' + name)
        save_figure(fig, base, footer)
        outputs.extend([str(base) + '.png', str(base) + '.pdf'])
    # Additional family-specific quantities retain their original metric names.
    extras = [k for k in metric_fields(rows) if k in FAMILY_METRICS or
              (k.endswith('_mean') and k[:-5] in FAMILY_METRICS)]
    if family == 'contextual_additive':
        # Fairness quantities may be emitted diagnostically for every allocation,
        # but they are not the contextual model's trained task objective.
        extras = []
    for start in range(0, len(extras), 3):
        page_metrics = extras[start:start + 3]
        fig, axes = plt.subplots(1, len(page_metrics), figsize=(5 * len(page_metrics), height), squeeze=False)
        for ax, metric in zip(axes[0], page_metrics):
            point_panel(ax, grouped, metric, '\n'.join(textwrap.wrap(metric.replace('_', ' '), 29)))
        fig.suptitle(title_for(group, phase, diagnostic), fontsize=11)
        base = directory / (stem + f'_family_metrics_p{start // 3 + 1}')
        save_figure(fig, base, 'Task-specific raw metrics; definitions and synthetic/human-data provenance are in the evaluation metadata.')
        outputs.extend([str(base) + '.png', str(base) + '.pdf'])
    # Objective--revenue relationships are task-local. A baseline is shown only
    # when its evaluation explicitly uses this same family, setting and protocol.
    objectives = [key for key in extras if key in {
        'pca', 'pca_mean', 'preference_ground_truth_score', 'preference_ground_truth_score_mean',
        'preference_classifier_probability', 'preference_classifier_probability_mean',
        'preference_probability', 'preference_probability_mean',
        'paper_unfairness_sum', 'paper_unfairness_sum_mean', 'absolute_tvf_sum', 'absolute_tvf_sum_mean',
        'entropy', 'entropy_mean', 'minimum_normalized_quota', 'minimum_normalized_quota_mean',
        'synthetic_preference_score', 'synthetic_proportional_share_gap'}]
    for metric in objectives:
        fig, ax = plt.subplots(figsize=(7.3, 5))
        for index, (key, unit_rows) in enumerate(grouped.items()):
            paired = [r for r in unit_rows if number(r.get(metric)) is not None]
            if not paired:
                continue
            xs = [number(r[metric]) for r in paired]
            ys = [float(r['revenue']) for r in paired]
            color = COLORS[index % len(COLORS)]
            ax.scatter(xs, ys, color=color, alpha=.35, s=18)
            mean_x, sd_x = stats(paired, metric)
            mean_y, sd_y = stats(paired, 'revenue')
            ax.errorbar(mean_x, mean_y, xerr=sd_x, yerr=sd_y, color=color,
                        fmt='o', capsize=3, markersize=6,
                        label=short_name(key[0]) + '\n[' + role_label(key[1]) + ']')
        ax.set_xlabel(metric.replace('_', ' '))
        ax.set_ylabel('Revenue')
        if metric in ('pca', 'pca_mean', 'preference_probability', 'preference_probability_mean',
                      'preference_classifier_probability', 'preference_classifier_probability_mean',
                      'minimum_normalized_quota', 'minimum_normalized_quota_mean'):
            ax.set_xlim(-.02, 1.02)
            ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(color='#e4e4e4', linewidth=.6)
        ax.legend(fontsize=8, loc='best')
        fig.suptitle(title_for(group, phase, diagnostic), fontsize=10)
        base = directory / (stem + '_' + stable_slug(metric) + '_vs_revenue')
        save_figure(fig, base, 'Faint dots: individual seeds/checkpoints; bars: SD across units, not CI. Only the stated task and protocol are compared.')
        outputs.extend([str(base) + '.png', str(base) + '.pdf'])
    # Thin lines show each frozen checkpoint, thick lines summarize only matching unit roles.
    audits = sorted({r['audit'] for r in rows}, key=audit_order)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric in zip(axes, ('regret_mean', 'regret_max', 'pmr')):
        for color_index, key in enumerate(grouped):
            selected = [r for r in rows if identity(r) == key]
            per_seed = defaultdict(dict)
            for row in selected:
                per_seed[(str(row['seed']), row['checkpoint_sha256'])][row['audit']] = row
            color = COLORS[color_index % len(COLORS)]
            for one in per_seed.values():
                ax.plot(range(len(audits)), [one.get(a, {}).get(metric, np.nan) for a in audits],
                        color=color, alpha=.22, linewidth=.8)
            means, sds = zip(*(stats([r for r in selected if r['audit'] == a], metric) for a in audits))
            ax.errorbar(range(len(audits)), means, yerr=[0 if s is None else s for s in sds],
                        color=color, marker='o', linewidth=1.4, capsize=3,
                        label=short_name(key[0]) + '\n[' + role_label(key[1]) + ']')
        ax.set_xticks(range(len(audits)), [a.replace('_', ' ') for a in audits], rotation=20)
        ax.set_ylabel(metric.replace('_', ' '))
        ax.grid(color='#e4e4e4', linewidth=.6)
        if metric != 'pmr':
            positive = [r[metric] for r in rows if r[metric] > 0]
            if positive and max(positive) / max(min(positive), 1e-15) > 100:
                ax.set_yscale('symlog', linthresh=1e-6)
                ax.set_ylabel(metric.replace('_', ' ') + '\n(symlog; linear below 1e-6)')
            else:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
        else:
            ax.set_ylim(-.03, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), fontsize=7,
               ncol=min(3, max(1, len(labels))))
    fig.suptitle(title_for(group, phase, diagnostic), fontsize=11)
    fig.tight_layout(rect=(0, .25 if len(labels) > 3 else .20, 1, .88))
    fig.text(.5, .005, 'Nested search of the same frozen checkpoint; thin lines = individual units. Strong is a finite-search lower bound.',
             ha='center', fontsize=8)
    base = directory / (stem + '_audit_strength')
    fig.savefig(str(base) + '.png', dpi=180, facecolor='white')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight')
    plt.close(fig)
    outputs.extend([str(base) + '.png', str(base) + '.pdf'])
    return outputs


def make_tex(rows, family, setting, diagnostic=False):
    lines = ['% Generated from verified common.evaluate records. Requires booktabs and graphicx.',
             '% Means and sample SD across listed unit roles; no CI, significance, or cross-task ranking.']
    split = defaultdict(list)
    for row in rows:
        split[(row['scope'], row['protocol_id'])].append(row)
    for (scope, pid), members in sorted(split.items()):
        header = ('SMOKE TEST -- NOT PAPER RESULTS. ' if scope == 'smoke_only' else '')
        if diagnostic:
            header += 'DIAGNOSTIC ONLY: group includes failed seeds; all completed seeds retained. '
        caption = header + family + ' / ' + setting + '; protocol ' + pid + '. Mean and SD over the stated units.'
        lines.extend([r'\begin{table*}[t]', r'\centering', r'\scriptsize',
                      r'\caption{' + tex_escape(caption) + '}', r'\resizebox{\textwidth}{!}{%',
                      r'\begin{tabular}{llllrrrrrr}', r'\toprule',
                      r'Method & Source role & Unit role & Audit & $n$ & Revenue & Welfare & Mean regret & Max regret & PMR \\',
                      r'\midrule'])
        grouped = defaultdict(list)
        for row in members:
            grouped[(identity(row), row['audit'])].append(row)
        for (key, audit), values in sorted(grouped.items(), key=lambda kv: (kv[0][0], audit_order(kv[0][1]))):
            cells = [tex_escape(key[0]), tex_escape(key[2]), tex_escape(role_label(key[1])), tex_escape(audit), str(len(values))]
            for metric in CORE:
                mean, sd = stats(values, metric)
                cells.append(fmt(mean) if sd is None else f'${fmt(mean)} \\pm {fmt(sd)}$')
            lines.append(' & '.join(cells) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}}',
                      r'\par\smallskip\parbox{0.98\textwidth}{\scriptsize '
                      + tex_escape('Regret is an empirical unilateral-search lower bound. PMR threshold = 1e-6. '
                                   'An absent SD indicates one available unit; it is not zero. Source roles and hashes are in results_per_seed.csv / diagnostics_per_seed.csv.')
                      + '}', r'\end{table*}', ''])
    return '\n'.join(lines) + '\n'


def make_metrics_tex(rows, family, setting, diagnostic=False):
    """Keep structural and family-specific fields auditable outside narrow core tables."""
    metrics = [m for m in metric_fields(rows) if m not in CORE]
    groups = defaultdict(list)
    for row in rows:
        if row['audit'] == 'strong':
            groups[(row['scope'], row['protocol_id'])].append(row)
    lines = ['% Raw structure, constraint diagnostics and task-specific metrics. No human composite.']
    for (scope, pid), values in sorted(groups.items()):
        prefix = 'SMOKE TEST -- NOT PAPER RESULTS. ' if scope == 'smoke_only' else ''
        if diagnostic:
            prefix += 'DIAGNOSTIC ONLY -- GROUP INCLUDES FAILED SEEDS. '
        caption = prefix + family + ' / ' + setting + '; protocol ' + pid + '. Additional raw metrics.'
        by_identity = defaultdict(list)
        for row in values:
            by_identity[identity(row)].append(row)
        rendered_rows = []
        for key, members in sorted(by_identity.items()):
            for metric in metrics:
                mean, sd = stats(members, metric)
                text = fmt(mean) if sd is None else '$' + fmt(mean) + r' \pm ' + fmt(sd) + '$'
                available = sum(number(r.get(metric)) is not None for r in members)
                cells = [tex_escape(key[0]), tex_escape(role_label(key[1])), tex_escape(metric),
                         text, str(available), str(len(members))]
                rendered_rows.append(' & '.join(cells) + r' \\')
        for start in range(0, len(rendered_rows), 24):
            page_caption = caption + f' Part {start // 24 + 1}.'
            lines.extend([r'\begin{table*}[t]', r'\centering\scriptsize',
                          r'\caption{' + tex_escape(page_caption) + '}',
                          r'\resizebox{\textwidth}{!}{\begin{tabular}{llllrr}',
                          r'\toprule', r'Method & Unit role & Metric & Mean $\pm$ SD & Available & Total \\',
                          r'\midrule', *rendered_rows[start:start + 24],
                          r'\bottomrule\end{tabular}}', r'\par\smallskip\parbox{0.98\textwidth}{\scriptsize '
                          + tex_escape('NA is unavailable, not zero. Structure and file size do not measure human understanding. '
                                       'Within-profile SD is itself a diagnostic metric and is not a training-seed SD. '
                                       'Task-specific metric definitions and provenance must accompany this table.') + '}',
                          r'\end{table*}', ''])
    return '\n'.join(lines) + '\n'


def training_log_schema(fields):
    fields = set(fields)
    schemas = [
        ({'step', 'loss', 'revenue_train_batch', 'regret_bidder_mean_train_batch'}, 'RegretNet training', 'step',
         ['loss', 'revenue_train_batch', 'regret_bidder_mean_train_batch']),
        ({'step', 'loss', 'surrogate_revenue_train_round'}, 'AMenuNet training surrogate', 'step',
         ['loss', 'surrogate_revenue_train_round']),
        ({'step', 'loss', 'revenue_soft', 'incompatibility_loss'}, 'GemNet soft training', 'step',
         ['loss', 'revenue_soft', 'incompatibility_loss']),
        ({'epoch', 'train_loss', 'validation_objective'}, 'Custom proxy training/validation', 'epoch',
         ['train_loss', 'validation_objective', 'train_candidate_regret_mean']),
        ({'epoch', 'iteration', 'loss', 'revenue', 'regret_bidder_mean', 'rho_fair'},
         'Author architecture / paper-adapter training', '__recorded_update__',
         ['loss', 'revenue', 'regret_bidder_mean']),
    ]
    for required, label, x, metrics in schemas:
        if required <= fields:
            return label, x, [m for m in metrics if m in fields]
    return None


def export_training_logs(root, directory, run_scopes=None):
    reports, outputs = [], []
    paths = sorted({p for name in ('train_log.csv', 'training.csv', 'training_curve.csv')
                    for p in Path(root).rglob(name)})
    for path in paths:
        try:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                schema = training_log_schema(reader.fieldnames or [])
                data = list(reader)
            if schema is None or not data:
                reports.append({'path': str(path), 'status': 'skipped_unknown_or_empty_training_schema'})
                continue
            label, x, metrics = schema
            if x == '__recorded_update__':
                pairs = [(number(r.get('epoch')), number(r.get('iteration'))) for r in data]
                if any(a is None or b is None for a, b in pairs) or any(b <= a for a, b in zip(pairs, pairs[1:])):
                    raise ValueError('Recorded epoch/iteration pairs are not strictly increasing')
                values = {k: [number(r.get(k)) for r in data] for k in metrics}
                values[x] = list(range(1, len(data) + 1))
                x_label = 'Recorded log point (ordered epoch/iteration)'
            else:
                values = {k: [number(r.get(k)) for r in data] for k in [x] + metrics}
                x_label = x
            if any(v is None for vv in values.values() for v in vv):
                raise ValueError('Non-finite training log values')
            if any(b <= a for a, b in zip(values[x], values[x][1:])):
                raise ValueError('Training steps/epochs are not strictly increasing')
            scope = 'scope unknown; training diagnostics only'
            if set(run_scopes or []) == {'smoke_only'}:
                scope = 'SMOKE TEST — NOT PAPER RESULTS'
            for config_path in (path.parent / 'config.json', path.parent / 'run_config.json',
                                path.parent / 'train_config.json', path.parent / 'metadata.json'):
                if config_path.exists():
                    cfg = load_json(config_path)
                    if (cfg.get('smoke') or cfg.get('is_smoke') or cfg.get('smoke_only') or
                        cfg.get('scope') == 'smoke_only' or cfg.get('preset') == 'smoke' or cfg.get('mode') == 'smoke'):
                        scope = 'SMOKE TEST — NOT PAPER RESULTS'
            relative = str(path.relative_to(root))
            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8), squeeze=False)
            for ax, metric in zip(axes[0], metrics):
                ax.plot(values[x], values[metric], color=COLORS[0], linewidth=1)
                ax.set_xlabel(x_label)
                ax.set_ylabel(metric.replace('_', ' '))
                ax.grid(color='#e4e4e4', linewidth=.6)
            fig.suptitle(f'{scope}\n{label} | {relative}', fontsize=10)
            suffix = hashlib.sha256(relative.encode()).hexdigest()[:8]
            base = directory / ('training_' + stable_slug(path.parent.name) + '_' + suffix)
            save_figure(fig, base, 'Training/validation log values only; these are not independent test results or proof of convergence.')
            outputs.extend([str(base) + '.png', str(base) + '.pdf'])
            reports.append({'path': str(path), 'status': 'plotted', 'schema': label, 'metrics': metrics,
                            'last_step_or_epoch': values[x][-1]})
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            reports.append({'path': str(path), 'status': 'invalid_training_log', 'reason': str(exc)})
    return reports, outputs


def export(root, phase):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    tables, graphs = root / 'tables', root / 'graphs'
    tables.mkdir(exist_ok=True)
    graphs.mkdir(exist_ok=True)
    previous_files = []
    if (root / 'export_report.json').is_file():
        previous_report = load_json(root / 'export_report.json')
        if previous_report.get('exporter') == 'common.export_results':
            previous_files = previous_report.get('files', [])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    raw_rows, scans, protocols = find_results(root, phase)
    rows = apply_group_eligibility(raw_rows)
    eligible = [r for r in rows if r['aggregate_comparison_eligible']]
    diagnostic = [r for r in rows if not r['aggregate_comparison_eligible']]
    summary = aggregate(eligible)
    diagnostic_summary = aggregate(diagnostic)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or [*ID_FIELDS, 'audit', *CORE]
    write_csv(tables / 'results_per_seed.csv', eligible, fields)
    write_csv(tables / 'diagnostics_per_seed.csv', diagnostic, fields)
    write_csv(tables / 'all_evaluated_rows.csv', rows, fields)
    provenance_rows = artifact_provenance(rows)
    write_csv(tables / 'artifact_provenance.csv', provenance_rows,
              list(provenance_rows[0]) if provenance_rows else [*ID_FIELDS, 'seed', 'source_url', 'source_commit'])
    summary_fields = list(summary[0]) if summary else (list(diagnostic_summary[0]) if diagnostic_summary else
                     [*ID_FIELDS, 'audit', 'metric', 'n_units', 'n_available', 'mean', 'sd_across_units'])
    write_csv(tables / 'summary.csv', summary, summary_fields)
    write_csv(tables / 'diagnostics_summary.csv', diagnostic_summary, summary_fields)
    files = []
    for subset, is_diagnostic in ((eligible, False), (diagnostic, True)):
        groups = defaultdict(list)
        tex_groups = defaultdict(list)
        for row in subset:
            groups[tuple(row[k] for k in BASE_GROUP)].append(row)
            tex_groups[(row['family'], row['setting'])].append(row)
        for group, members in sorted(groups.items()):
            files.extend(make_group_figures(members, group, graphs, phase, is_diagnostic))
        for (family, setting), members in sorted(tex_groups.items()):
            name = ('diagnostics_' if is_diagnostic else 'comparison_') + stable_slug(f'{family}_{setting}') + '.tex'
            path = tables / name
            path.write_text(make_tex(members, family, setting, is_diagnostic), encoding='utf-8')
            files.append(str(path))
            extra_path = tables / (('diagnostic_metrics_' if is_diagnostic else 'raw_metrics_')
                                   + stable_slug(f'{family}_{setting}') + '.tex')
            extra_path.write_text(make_metrics_tex(members, family, setting, is_diagnostic), encoding='utf-8')
            files.append(str(extra_path))
    training_reports, training_figs = export_training_logs(root, graphs, {r['scope'] for r in rows})
    files.extend(training_figs)
    removed = []
    current_paths = {Path(p).resolve() for p in files}
    for old in previous_files:
        old_path = Path(old).resolve()
        if old_path.parent in (tables, graphs) and old_path not in current_paths and old_path.is_file():
            old_path.unlink()
            removed.append(str(old_path))
    invalid = [r for r in scans if r['status'] == 'invalid_evaluation']
    report = {'exporter': 'common.export_results', 'phase': phase, 'root': str(root),
              'state': 'COMPLETED' if rows and not invalid else ('NO_EVALUATIONS' if not rows and not invalid else 'INVALID_EVALUATIONS_FOUND'),
              'evaluation_rows': len(rows), 'comparison_rows': len(eligible), 'diagnostic_only_rows': len(diagnostic),
              'files': files, 'scan_reports': scans, 'training_logs': training_reports,
              'protocols': protocols, 'removed_stale_export_files': removed,
              'artifact_provenance_rows': len(provenance_rows),
              'missing_source_url_rows': sum(not r['source_url'] for r in provenance_rows),
              'notes': ['Grouped by family, setting, method, seed role, source role, scope, and evaluation protocol.',
                        'Sample SD across labelled units; no confidence interval or significance test.',
                        'Constraint-violating rows appear only in diagnostic exports.',
                        'Any failed completed seed excludes its whole method/setting/protocol group; measured individual validity is retained, without survivor-only aggregation.',
                        'No human composite, cross-family normalization, or algorithm ranking.',
                        'Structure is code/model size; it does not measure human understanding.',
                        'SMOKE tables and figures cannot be used as paper experiment results.']}
    save_json(root / 'export_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['phase1', 'phase2', 'phase3', 'proxy'], required=True)
    args = parser.parse_args()
    report = export(args.root, args.phase)
    print(json.dumps({k: report[k] for k in ('state', 'evaluation_rows', 'comparison_rows', 'diagnostic_only_rows')}, indent=2))
    if report['state'] != 'COMPLETED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
