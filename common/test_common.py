"""Scientific checks for unilateral audits, raw results and context handling.

These mechanisms have analytically known behavior; the tests do not assert
that one learned method should outperform another.
Run: python -m unittest common.test_common
"""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .audit import audit_batch
from .evaluate import evaluate


class OtherReportsOnly(torch.nn.Module):
    """Own allocation is fixed; own payment depends solely on others.

    A simultaneous perturbation of all bids can falsely report positive gain,
    while a correct unilateral audit always gives zero.
    """
    def forward(self, reports):
        allocation = torch.ones_like(reports) / reports.shape[1]
        totals = reports.sum(-1)
        payment = 0.1 * (totals.sum(1, keepdim=True) - totals)
        return allocation, payment


class FixedAllocationPayOwnReport(torch.nn.Module):
    """Two bidders receive half each; payment .25*sum(own reports).

    The most profitable report is zero. Exact gain is .25*sum(true values).
    """
    def __init__(self):
        super().__init__()
        self.fraction = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, reports):
        return torch.ones_like(reports) * 0.5, self.fraction * reports.sum(-1)


class InvalidOnlyAwayFromTruth(torch.nn.Module):
    """Truth .25 allocates feasibly; other reports overallocate both bidders."""
    def forward(self, reports):
        different = (reports - 0.25).abs().amax((1, 2)) > 1e-5
        fraction = torch.where(different, 0.75, 0.5)
        return (torch.ones_like(reports) * fraction[:, None, None],
                reports.new_zeros(reports.shape[:2]))


class ContextDependentFixedRule(torch.nn.Module):
    """Public contexts affect output; only opponents' reports affect payment."""
    def forward(self, inputs):
        reports, bidder_context, item_context = inputs
        public = bidder_context[..., 0, None] + item_context[:, None, :, 0]
        allocation = 0.1 + 0.02 * public
        own = reports.sum(-1)
        other = own.sum(1, keepdim=True) - own
        payment = 0.01 * bidder_context[..., 0] * other
        return allocation, payment


def metadata(role='new_training_seed'):
    return dict(method='KnownAnalyticMechanism', n_bidders=2, n_items=3,
                seed=17, seed_role=role, source_role='test_analytic_definition')


class CommonScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unilateral_zero_even_when_joint_changes_can_help(self):
        values = torch.full((3, 2, 3), 0.8, dtype=torch.float64)
        candidates = torch.zeros(3, 4, 2, 3, dtype=torch.float64)
        model = OtherReportsOnly()
        a, p = model(values)
        truth = (a * values).sum(-1) - p
        joint_a, joint_p = model(torch.zeros_like(values))
        self.assertGreater(float(((joint_a * values).sum(-1) - joint_p - truth).max()), 0.1)
        arrays = audit_batch(model, values, candidates, (1, 4), 3, 3, 0.05, 2)
        np.testing.assert_allclose(arrays['strong_gain'], 0, atol=1e-14)

    def test_known_positive_gain_and_nested_budgets(self):
        torch.manual_seed(11)
        values = torch.rand(5, 2, 3, dtype=torch.float64)
        candidates = torch.rand(5, 6, 2, 3, dtype=torch.float64)
        candidates[:, -1] = 0  # Last candidate attains the analytic optimum.
        arrays = audit_batch(FixedAllocationPayOwnReport().double(), values,
                             candidates, (2, 6), 4, 5, 0.1, 2)
        expected = 0.25 * values.sum(-1).numpy()
        np.testing.assert_allclose(arrays['strong_gain'], expected, atol=1e-12)
        self.assertTrue(np.all(arrays['strong_gain'] >= arrays['random_gain_budget6']))
        self.assertTrue(np.all(arrays['random_gain_budget6'] >= arrays['random_gain_budget2']))
        self.assertTrue(np.all(arrays['random_gain_budget2'] >= 0))

    def test_raw_arrays_reproduce_csv_and_model_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FixedAllocationPayOwnReport()
            original = {k: v.clone() for k, v in model.state_dict().items()}
            output = Path(directory) / 'evaluation'
            with contextlib.redirect_stdout(io.StringIO()):
                rows = evaluate(model, metadata(), output, samples=6,
                                budgets=(2, 6), pgd_restarts=3, pgd_steps=5,
                                batch_size=2, candidate_batch=2)
            raw = np.load(output / 'profiles.npz', allow_pickle=False)
            with (output / 'metrics.csv').open(newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 3)
            for row in csv_rows:
                gain_key = ('strong_gain' if row['audit'] == 'strong' else
                            'random_gain_budget' + row['audit'].split('_')[-1])
                gains = raw[gain_key].astype(np.float64)
                self.assertAlmostEqual(float(row['regret_mean']), gains.mean(), places=12)
                self.assertAlmostEqual(float(row['regret_max']), gains.max(), places=12)
                self.assertAlmostEqual(float(row['pmr']), (gains > 1e-6).mean(), places=12)
                expected_revenue = raw['payments'].sum(1).mean(dtype=np.float64)
                self.assertAlmostEqual(float(row['revenue']), expected_revenue, places=12)
                self.assertLessEqual(float(row['regret_mean']), float(row['regret_max']))
            self.assertEqual(len({row['revenue'] for row in csv_rows}), 1)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertTrue(all(torch.equal(original[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(row['evidence_status'] == 'empirical_comparison' for row in rows))
            # Recompute strongest witness utilities directly from saved reports.
            values = torch.from_numpy(raw['values'])
            for bidder in range(2):
                reports = values.clone()
                reports[:, bidder] = torch.from_numpy(raw['strong_best_report'][:, bidder])
                allocation, payment = model(reports)
                utility = (allocation[:, bidder] * values[:, bidder]).sum(-1) - payment[:, bidder]
                gain = np.maximum(utility.numpy() - raw['truth_utility'][:, bidder], 0)
                np.testing.assert_allclose(gain, raw['strong_gain'][:, bidder], atol=1e-7)

    def test_invalid_deviation_reports_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'evaluation'
            with contextlib.redirect_stdout(io.StringIO()):
                rows = evaluate(InvalidOnlyAwayFromTruth(), metadata(), output,
                                samples=4, values=np.full((4, 2, 3), 0.25),
                                budgets=(2, 4), pgd_restarts=3, pgd_steps=0,
                                batch_size=2)
            raw = np.load(output / 'profiles.npz', allow_pickle=False)
            self.assertLessEqual(float(raw['allocation'].sum(1).max()), 1)
            self.assertTrue(all(row['max_overallocation'] > 0.4 for row in rows))
            self.assertTrue(all(not row['feasible_on_checked_reports'] for row in rows))
            self.assertTrue(all(row['evidence_status'] == 'diagnostic_only_constraint_violation' for row in rows))
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['result_use'], 'diagnostic_only_constraint_violation')

    def test_context_is_profile_major_and_fixed_for_all_candidates(self):
        from phase3_author.bridge import ContextBatchModel
        torch.manual_seed(47)
        values = torch.rand(3, 2, 3, dtype=torch.float64)
        candidates = torch.rand(3, 5, 2, 3, dtype=torch.float64)
        bc = torch.tensor([[[0.2], [0.4]], [[0.7], [0.9]], [[1.2], [1.4]]], dtype=torch.float64)
        ic = torch.arange(9, dtype=torch.float64).reshape(3, 3, 1) / 10
        core = ContextDependentFixedRule()
        bound = ContextBatchModel(core, bc, ic)
        submitted = values[:, None].expand(-1, 5, -1, -1).clone()
        submitted[:, :, 0] = candidates[:, :, 0]
        batched_a, batched_p = bound(submitted.reshape(15, 2, 3))
        expected_a, expected_p = [], []
        for profile in range(3):
            for candidate in range(5):
                allocation, payment = core((submitted[profile, candidate: candidate + 1],
                                            bc[profile:profile + 1], ic[profile:profile + 1]))
                expected_a.append(allocation[0]); expected_p.append(payment[0])
        torch.testing.assert_close(batched_a, torch.stack(expected_a))
        torch.testing.assert_close(batched_p, torch.stack(expected_p))
        arrays = audit_batch(bound, values, candidates, (2, 5), 3, 3, .05, 2)
        np.testing.assert_allclose(arrays['strong_gain'], 0, atol=1e-14)

    def test_checkpoint_seed_role_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            for role in ('author_checkpoint_directory_label', 'new_training_seed'):
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = evaluate(FixedAllocationPayOwnReport(), metadata(role),
                                    Path(directory) / role, samples=3,
                                    budgets=(2, 4), pgd_restarts=3, pgd_steps=0,
                                    batch_size=3)
                self.assertTrue(all(row['seed_role'] == role for row in rows))


if __name__ == '__main__':
    unittest.main()
