"""Meaningful export contract tests; fixtures are explicitly synthetic smoke data."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from common.export_results import (aggregate, export, find_results, load_json,
                                   save_json, write_csv, training_log_schema, apply_group_eligibility,
                                   point_panel, identity, plt)


def fixture(root, method='FixtureMethod', seed=0, role='new_training_seed', diagnostic=False,
            scope='smoke_only', candidate_seed=9, transform_invalid=False):
    out = Path(root) / f'{method}_{role}' / f'seed_{seed}' / 'evaluation'
    out.mkdir(parents=True)
    checkpoint = f'fixture_{method}_{seed}'
    config = {'samples': 4, 'n_bidders': 3, 'n_items': 10, 'scope': scope,
              'data_seed': 7, 'candidate_seed': candidate_seed, 'random_budgets': [2, 4],
              'pgd_restarts': 3, 'pgd_steps': 2, 'pgd_lr': .05, 'pmr_threshold': 1e-6,
              'test_profile_sha256': 'fixture_not_a_real_data_hash',
              'audit_inputs_sha256': f'fixture_candidates_{candidate_seed}', 'context_protocol': 'none'}
    provenance = {'method': method, 'seed': seed, 'checkpoint_sha256': checkpoint,
                  'evaluation_code_sha256': {'common/evaluate.py': 'fixture_code', 'common/audit.py': 'fixture_audit'},
                  'transform_diagnostics': {'valid_for_empirical_revenue_comparison': not transform_invalid}}
    save_json(out / 'config.json', config)
    save_json(out / 'provenance.json', provenance)
    save_json(out / 'status.json', {'state': 'COMPLETED'})
    rows = []
    for budget_index, audit in enumerate(('random_2', 'random_4', 'strong'), start=1):
        rows.append({'method': method, 'family': 'fixture_static', 'setting': '3x10_fixture_only',
                     'seed': seed, 'seed_role': role, 'source_role': 'synthetic_test_fixture',
                     'scope': scope, 'audit': audit, 'n_profiles': 4, 'n_bidders': 3, 'n_items': 10,
                     'revenue': 4 + seed, 'revenue_profile_sd': .5, 'welfare': 6 + seed,
                     'regret_mean': .002 * budget_index * (1 + seed),
                     'regret_max': .01 * budget_index * (1 + seed), 'pmr': .1 * budget_index,
                     'parameter_count': 1234, 'menu_size': '', 'checkpoint_file_bytes': 5000,
                     'max_truthful_ir_violation': 0., 'max_overallocation': .2 if diagnostic else 0.,
                     'max_negative_allocation': 0., 'max_negative_payment': 0.,
                     'feasible_on_checked_reports': not diagnostic, 'ir_on_truthful_profiles': True,
                     'transform_diagnostic_valid': not transform_invalid,
                     'evidence_status': 'diagnostic_only_constraint_violation' if diagnostic or transform_invalid else 'empirical_comparison',
                     'checkpoint_sha256': checkpoint, 'test_profile_sha256': config['test_profile_sha256'],
                     'evaluation_path': str(out), 'pca': .8 + .01 * seed,
                     'paper_unfairness_sum': .03 - .01 * seed,
                     'human_proxy_score': 99})
    write_csv(out / 'metrics.csv', rows)
    return out


class ExportTests(unittest.TestCase):
    def test_one_failed_seed_excludes_whole_group_without_changing_raw_validity(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d, method='MixedMethod', seed=0)
            fixture(d, method='MixedMethod', seed=1, diagnostic=True)
            fixture(d, method='AllValidMethod', seed=0)
            raw, _, _ = find_results(d, 'phase2')
            grouped = apply_group_eligibility(raw)
            mixed = [r for r in grouped if r['method'] == 'MixedMethod']
            valid = [r for r in grouped if r['method'] == 'AllValidMethod']
            self.assertEqual(len(mixed), 6)
            self.assertTrue(all(not r['aggregate_comparison_eligible'] for r in mixed))
            self.assertTrue(all(r['group_failed_seed_count'] == 1 and r['group_completed_seed_count'] == 2 for r in mixed))
            self.assertEqual(sum(r['evidence_status'] == 'empirical_comparison' for r in mixed), 3)
            self.assertTrue(all(r['aggregate_comparison_eligible'] for r in valid))
            diagnostics = aggregate(mixed)
            self.assertTrue(all(r['n_units'] == 2 for r in diagnostics))
            self.assertTrue(all('no survivor filtering' in r['aggregate_exclusion_reason'] for r in diagnostics))
            fig, ax = plt.subplots()
            point_panel(ax, {identity(mixed[0]): mixed}, 'revenue', 'Revenue')
            self.assertEqual(ax.get_xlabel(), 'Revenue')
            self.assertIn('failed units: 1/2', ax.get_yticklabels()[0].get_text())
            plt.close(fig)

    def test_roles_protocols_and_sd_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d, seed=0)
            fixture(d, seed=1)
            fixture(d, seed=0, role='author_checkpoint')
            fixture(d, seed=1, role='author_checkpoint', candidate_seed=100)
            rows, scans, protocols = find_results(d, 'phase1')
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(protocols), 2)
            metrics = [r for r in aggregate(rows) if r['metric'] == 'revenue' and r['audit'] == 'strong']
            self.assertEqual(len(metrics), 3)
            training = next(r for r in metrics if r['seed_role'] == 'new_training_seed')
            self.assertEqual(training['n_units'], 2)
            self.assertAlmostEqual(training['mean'], 4.5)
            self.assertAlmostEqual(training['sd_across_units'], math.sqrt(.5))
            self.assertTrue(all('human_proxy_score' not in r for r in rows))
            menu = next(r for r in aggregate(rows) if r['metric'] == 'menu_size')
            self.assertIsNone(menu['mean'])

    def test_frozen_audit_and_nonnested_results_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = fixture(d)
            with (folder / 'metrics.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            rows[2]['regret_mean'] = .001
            rows[2]['regret_max'] = .005
            write_csv(folder / 'metrics.csv', rows)
            found, report, _ = find_results(d, 'phase1')
            self.assertFalse(found)
            self.assertEqual(report[0]['status'], 'invalid_evaluation')
            self.assertIn('Nested audit', report[0]['reason'])

    def test_inconsistent_ir_status_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = fixture(d)
            with (folder / 'metrics.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row['max_truthful_ir_violation'] = .1
            write_csv(folder / 'metrics.csv', rows)
            found, report, _ = find_results(d, 'phase1')
            self.assertFalse(found)
            self.assertIn('Constraint evidence status', report[0]['reason'])

    def test_transform_violation_remains_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d, transform_invalid=True)
            found, report, _ = find_results(d, 'phase2')
            self.assertEqual(len(found), 3)
            self.assertTrue(all(r['evidence_status'] == 'diagnostic_only_constraint_violation' for r in found))

    def test_training_schema_is_explicit(self):
        self.assertIsNone(training_log_schema(['step', 'revenue', 'some_loss']))
        self.assertEqual(training_log_schema(['step', 'loss', 'surrogate_revenue_train_round'])[0],
                         'AMenuNet training surrogate')


def make_visual_fixture(path):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    fixture(root, seed=0)
    fixture(root, seed=1)
    fixture(root, method='FixtureAuthorModel', seed=0, role='author_checkpoint')
    fixture(root, method='FixtureConstraintViolation', seed=0, diagnostic=True)
    train = root / 'fixture_train'
    train.mkdir()
    save_json(train / 'config.json', {'smoke': True})
    write_csv(train / 'train_log.csv', [{'step': i, 'loss': -i / 10, 'surrogate_revenue_train_round': i / 10}
                                       for i in range(1, 4)])
    report = export(root, 'phase2')
    assert report['state'] == 'COMPLETED'
    assert report['comparison_rows'] == 9 and report['diagnostic_only_rows'] == 3
    table = (root / 'tables/comparison_fixture_static_3x10_fixture_only.tex').read_text()
    assert 'SMOKE TEST' in table and 'FixtureConstraintViolation' not in table
    assert (root / 'tables/diagnostics_fixture_static_3x10_fixture_only.tex').exists()
    assert any('pca_vs_revenue' in p for p in report['files'])
    return report


if __name__ == '__main__':
    unittest.main()
