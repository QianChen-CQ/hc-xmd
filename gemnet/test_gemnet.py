import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .model import GemNet
from .transform import solve_price_milp, TransformedGemNet, transform_model
from .run import parser, train


class GemNetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_own_bid_does_not_change_menu(self):
        model = GemNet(2, 2, 5, 16).eval()
        v = torch.rand(8, 2, 2)
        for i in range(2):
            w = v.clone(); w[:, i] = torch.rand(8, 2)
            a, p = model.bidder_menu(v, i); aa, pp = model.bidder_menu(w, i)
            self.assertTrue(torch.equal(a, aa)); self.assertTrue(torch.equal(p, pp))
        ctx = {0: np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)}
        delta = {0: np.array([[0,.1,.2,.3,0],[.1,0,.3,.2,0],[.2,.4,.3,.1,0],[.3,.2,.1,.4,0]],dtype=np.float32)}
        transformed = TransformedGemNet(model, ctx, delta, dict(grid_mode='cartesian',grid_points=2,vmax=1.0)).eval()
        w = v.clone(); w[:, 0] = torch.rand(8, 2)
        a,p = transformed.bidder_menu(v,0); aa,pp=transformed.bidder_menu(w,0)
        self.assertTrue(torch.equal(a,aa)); self.assertTrue(torch.equal(p,pp))

    def test_actual_unilateral_gain_zero_for_menu_choice(self):
        # This verifies menu utility maximization; feasibility is a separate test.
        model = GemNet(2, 2, 6, 12).eval()
        v = torch.rand(16, 2, 2)
        with torch.no_grad():
            a,p = model(v); truth=(a*v).sum(2)-p
            for i in range(2):
                for _ in range(20):
                    reports=v.clone(); reports[:,i]=torch.rand(16,2)
                    aa,pp=model(reports)
                    gain=(aa[:,i]*v[:,i]).sum(1)-pp[:,i]-truth[:,i]
                    self.assertLessEqual(float(gain.max()),2e-6)

    def test_incompatible_option_really_gets_price_adjusted(self):
        own=np.array([[0.0],[1.0]])
        bundle=np.array([[1.0],[0.5],[0.0]])
        price=np.array([.1,.2,0.0])
        compatible=np.array([[False,True,True],[False,True,True]])
        delta,diagnostics=solve_price_milp(own,bundle,price,compatible,.1)
        selected=(own@bundle.T-price-delta).argmax(1)
        self.assertTrue(compatible[np.arange(2),selected].all())
        self.assertGreater(delta[0],0)
        self.assertGreater(diagnostics['n_binary'],0)

    def test_screened_negative_option_still_obeys_margin(self):
        own=np.array([[0.]])
        bundle=np.array([[1.],[0.]])
        price=np.array([.02,0.])
        delta,_=solve_price_milp(own,bundle,price,np.ones((1,2),bool),.1)
        self.assertAlmostEqual(delta[0],.08,places=6)

    def test_zero_violation_skip_preserves_compatible_menu(self):
        own=np.linspace(0,1,7)[:,None]
        bundle=np.array([[1.],[0.]])
        price=np.array([.2,0.])
        delta,diagnostics=solve_price_milp(own,bundle,price,np.ones((7,2),bool),1/6,
                                         zero_violation_skip=True)
        np.testing.assert_array_equal(delta,np.zeros(2))
        self.assertEqual(diagnostics['solver'],'paper_zero_violation_skip')
        self.assertGreater(((own@bundle.T-price-delta).argmax(1)==0).sum(),0)

    def test_resume_matches_uninterrupted_optimizer_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'continuous',Path(tmp)/'resumed'
            options=['train','--preset','smoke','--device','cpu','--seed','12']
            train(parser().parse_args(options+['--out',str(a),'--steps','4']))
            train(parser().parse_args(options+['--out',str(b),'--steps','2']))
            train(parser().parse_args(options+['--out',str(b),'--steps','4','--resume']))
            x=torch.load(a/'checkpoint_last.pt',weights_only=True)
            y=torch.load(b/'checkpoint_last.pt',weights_only=True)
            self.assertEqual(x['step'],y['step'])
            for key in x['state_dict']:
                self.assertTrue(torch.equal(x['state_dict'][key],y['state_dict'][key]), key)
            self.assertEqual(x['lambda_incompatibility'],y['lambda_incompatibility'])
            for pid in x['optimizer_state_dict']['state']:
                for key in x['optimizer_state_dict']['state'][pid]:
                    self.assertTrue(torch.equal(x['optimizer_state_dict']['state'][pid][key],
                                                y['optimizer_state_dict']['state'][pid][key]))

    def test_transform_resume_restores_completed_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            model=GemNet(2,2,4,8)
            # Initialize spectral-normalization power vectors as training does.
            with torch.no_grad():
                model(torch.zeros(1,2,2))
            model.eval()
            transform_model(model,path,grid_points=3)
            expected=np.load(path/'partial_bidder0.npy').copy()
            # Simulate an interrupted copy with four completed context outputs.
            np.save(path/'partial_bidder0.npy',expected[:4])
            log=(path/'milp_log.jsonl').read_text().splitlines()
            (path/'milp_log.jsonl').write_text('\n'.join(log[:4])+'\n')
            transform_model(model,path,grid_points=3,resume=True)
            np.testing.assert_array_equal(np.load(path/'partial_bidder0.npy'),expected)


if __name__=='__main__':
    unittest.main()
