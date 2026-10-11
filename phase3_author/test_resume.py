"""Meaningful crash/restart check: identical weights after a discarded epoch log."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from phase3_author.train import train, read_args


class InjectedCrash(RuntimeError):
    pass


class ResumeTests(unittest.TestCase):
    def test_all_state_restored_and_orphan_curve_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in ('CITransNet','PreferenceNet','ProportionNet'):
                common = ['--method',method,'--seed','7','--mode','smoke','--epochs','2',
                          '--n-agents','2','--n-items','2','--eval-samples','4',
                          '--torch-threads','2']
                full = Path(temp)/method/'continuous'
                resumed = Path(temp)/method/'resumed'
                with contextlib.redirect_stdout(io.StringIO()):
                    train(read_args(common+['--output-dir',str(full)]))
                    original_save = torch.save

                    def crashing_save(value,path,*args,**kwargs):
                        if Path(path).name=='training_state.pt.tmp' and value.get('epoch')==2:
                            raise InjectedCrash('After epoch 2 CSV flush, before checkpoint commit')
                        return original_save(value,path,*args,**kwargs)

                    with patch('torch.save',crashing_save):
                        with self.assertRaises(InjectedCrash):
                            train(read_args(common+['--output-dir',str(resumed)]))
                    meta=train(read_args(common+['--output-dir',str(resumed),'--resume']))
                self.assertEqual(meta['resume_action'],'full_state_restored')
                expected=torch.load(full/'checkpoint.pt',weights_only=True)
                actual=torch.load(resumed/'checkpoint.pt',weights_only=True)
                self.assertTrue(all(torch.equal(expected[k],actual[k]) for k in expected),method)
                self.assertTrue((resumed/'uncheckpointed_training_curve_before_resume_epoch1.csv').exists())
                self.assertEqual(len((resumed/'training_curve.csv').read_text().strip().splitlines()),3)
                if method=='PreferenceNet':
                    expected=torch.load(full/'preference_classifier.pt',weights_only=True)
                    actual=torch.load(resumed/'preference_classifier.pt',weights_only=True)
                    self.assertTrue(all(torch.equal(expected[k],actual[k]) for k in expected))


if __name__=='__main__':
    unittest.main()
