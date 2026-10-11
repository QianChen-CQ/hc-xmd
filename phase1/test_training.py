"""Behavioral tests for long-run interruption/recovery, not paper results."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import train
from .models import create_model, load_checkpoint


class TrainingTests(unittest.TestCase):
    def test_exact_resume_both_methods(self):
        torch.set_num_threads(1)
        for method in ('RegretNet', 'AMenuNet'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                cfg = train.preset_config(method, 'smoke')
                cfg['seed'] = 31
                full = Path(root)/'full'
                interrupted = Path(root)/'interrupted'
                full.mkdir()
                interrupted.mkdir()
                function = train.train_regretnet if method == 'RegretNet' else train.train_amenu
                torch.manual_seed(cfg['seed'])
                expected = create_model(method, cfg['model']).train()
                with contextlib.redirect_stdout(io.StringIO()):
                    function(expected, cfg, full, 'cpu', None)
                torch.manual_seed(cfg['seed'])
                partial = create_model(method, cfg['model']).train()
                original_write = train.torch_write
                stopped = False
                def stop_after_checkpoint(path, payload):
                    nonlocal stopped
                    original_write(path, payload)
                    if Path(path).name == 'resume_latest.pt' and not stopped:
                        stopped = True
                        raise KeyboardInterrupt('simulated interruption after safe checkpoint')
                with patch.object(train, 'torch_write', stop_after_checkpoint):
                    with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                        function(partial, cfg, interrupted, 'cpu', None)
                resume = torch.load(interrupted/'resume_latest.pt', weights_only=True)
                # Deliberately perturb global state before loading checkpoint.
                torch.manual_seed(999)
                restored = create_model(method, cfg['model']).train()
                restored.load_state_dict(resume['model_state'])
                with contextlib.redirect_stdout(io.StringIO()):
                    function(restored, cfg, interrupted, 'cpu', resume)
                for name, tensor in expected.state_dict().items():
                    self.assertTrue(torch.equal(tensor, restored.state_dict()[name]),
                                    f'{method}: changed weights after resume: {name}')
                model, meta = load_checkpoint(interrupted/'checkpoint.pt')
                self.assertTrue(meta['is_smoke'])
                self.assertTrue(meta['training_complete'])
                values = torch.rand(7, 3, 10)
                allocation, payment = model(values)
                self.assertTrue(torch.isfinite(allocation).all())
                self.assertTrue(torch.isfinite(payment).all())
                self.assertLessEqual(float(allocation.sum(dim=1).max()), 1+1e-6)
                self.assertGreaterEqual(float(((allocation*values).sum(-1)-payment).min()), -2e-5)

    def test_setting_shapes(self):
        for setting in ('3x10', '2x2'):
            for method in ('RegretNet', 'AMenuNet'):
                cfg = train.preset_config(method, 'paper', setting)
                model = create_model(method, cfg['model']).eval()
                x = torch.rand(4, cfg['model']['n'], cfg['model']['m'])
                allocation, payment = model(x)
                self.assertEqual(allocation.shape, x.shape)
                self.assertEqual(payment.shape, x.shape[:2])


if __name__ == '__main__':
    unittest.main()
