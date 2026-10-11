import unittest
from pathlib import Path
import sys
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase3_author.bridge import build_model, ContextBatchModel, direct_metrics
from phase3_author.train import make_data, deviations, optimize_reports


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)

    def test_context_does_not_change_on_candidate_expansion(self):
        data = make_data(2, 3, 10, 9, True)
        model = build_model('CITransNet').eval()
        bound = ContextBatchModel(model, data['bidder_context'], data['item_context'])
        truth_a, truth_p = bound(data['values'])
        candidate = data['values'][:, None].expand(-1, 4, -1, -1).reshape(8, 3, 10)
        a, p = bound(candidate)
        self.assertTrue(torch.allclose(a.reshape(2,4,3,10), truth_a[:,None].expand(-1,4,-1,-1), atol=1e-6))
        self.assertTrue(torch.allclose(p.reshape(2,4,3), truth_p[:,None].expand(-1,4,-1), atol=1e-6))

    def test_unilateral_utilities_agree_with_explicit_loop(self):
        model = build_model('PreferenceNet', 3, 4)
        truth, report = torch.rand(5,3,4), torch.rand(5,3,4)
        util = deviations(model, truth, report)
        for i in range(3):
            changed = truth.clone(); changed[:,i] = report[:,i]
            a,p = model(changed)
            expected = (a[:,i]*truth[:,i]).sum(-1)-p[:,i]
            self.assertTrue(torch.allclose(util[:,i], expected, atol=1e-6))

    def test_paper_positive_part_and_l1_are_distinct(self):
        allocation = torch.tensor([[[.8,.2],[.1,.6]]])
        metric = direct_metrics(allocation, .1)
        # Original Eq4: ordered differences .6 and .5, each minus distance .1.
        self.assertAlmostEqual(metric['paper_unfairness_sum'].item(), .9, places=6)
        # Absolute TVF: two ordered pairs each L1=1.1, minus .1.
        self.assertAlmostEqual(metric['absolute_tvf_sum'].item(), 2.0, places=6)

    def test_original_architectures_feasible_and_ir(self):
        for method in ('CITransNet','PreferenceNet','ProportionNet'):
            data=make_data(3,3,10,10,method=='CITransNet')
            model=build_model(method).eval()
            if method=='CITransNet':
                model=ContextBatchModel(model,data['bidder_context'],data['item_context'])
            a,p=model(data['values'])
            self.assertTrue(bool((a.sum(1)<=1+1e-6).all()))
            self.assertTrue(bool((a>=0).all() and (p>=0).all()))
            self.assertTrue(bool(((a*data['values']).sum(-1)-p>=-1e-6).all()))

    def test_context_ablation_ignores_context_without_changing_values(self):
        data=make_data(3,3,10,7,True)
        ablation=build_model('CITransNetNoContext').eval()
        a,p=ablation((data['values'],data['bidder_context'],data['item_context']))
        b,q=ablation((data['values'],torch.rand_like(data['bidder_context']),torch.rand_like(data['item_context'])))
        self.assertTrue(torch.equal(a,b) and torch.equal(p,q))
        self.assertEqual(sum(x.numel() for x in ablation.parameters()),202370)

    def test_proportion_baseline_architecture_and_default_budget_match(self):
        from phase3_author.train import read_args
        torch.manual_seed(42); mechanism=build_model('ProportionNet',2,2)
        torch.manual_seed(42); baseline=build_model('RegretNetProportionBaseline',2,2)
        self.assertTrue(all(torch.equal(v,baseline.state_dict()[k]) for k,v in mechanism.state_dict().items()))
        original=read_args(['--method','ProportionNet','--output-dir','unused'])
        control=read_args(['--method','RegretNetProportionBaseline','--output-dir','unused'])
        for key in ('epochs','train_samples','batch_size','misreport_steps'):
            self.assertEqual(getattr(original,key),getattr(control,key))


if __name__=='__main__':
    unittest.main()
