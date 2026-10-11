"""One evaluation definition for all compatible additive auction mechanisms."""
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .audit import audit_batch,per_profile_metrics
from .io import write_json,write_csv,sha256,ROOT

class OutputAudit(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
        self.max_overallocation=0.;self.max_negative_allocation=0.;self.max_negative_payment=0.
    def forward(self,values):
        alloc,pay=self.model(values)
        with torch.no_grad():
            self.max_overallocation=max(self.max_overallocation,float((alloc.sum(1)-1).clamp_min(0).max()))
            self.max_negative_allocation=max(self.max_negative_allocation,float((-alloc).clamp_min(0).max()))
            self.max_negative_payment=max(self.max_negative_payment,float((-pay).clamp_min(0).max()))
        return alloc,pay

def evaluate(model,metadata,outdir,*,samples=512,data_seed=20261004,candidate_seed=20261005,
             budgets=(32,128),pgd_restarts=8,pgd_steps=100,pgd_lr=.05,batch_size=16,
             candidate_batch=16,device='cpu',smoke=False,values=None,context_factory=None,
             context_arrays=None,extra_metrics=None):
    out=Path(outdir)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Refusing to overwrite evaluation: {out}')
    out.mkdir(parents=True,exist_ok=True)
    status={'state':'RUNNING','scope':'smoke_only' if smoke else 'requested_evaluation'}
    write_json(out/'status.json',status)
    start=time.time()
    try:
        if samples<2 or batch_size<1 or candidate_batch<1:
            raise ValueError('At least 2 samples and positive batch sizes are required')
        trained_smoke = metadata.get('is_smoke') or metadata.get('smoke_only') or metadata.get('mode') == 'smoke'
        if trained_smoke and not smoke:
            raise ValueError('Smoke-trained checkpoints cannot be exported as a formal evaluation')
        if smoke:samples,budgets,pgd_restarts,pgd_steps,batch_size=4,(2,4),3,2,2
        n,m=int(metadata['n_bidders']),int(metadata['n_items'])
        provided_values=values is not None
        if values is None:values=np.random.default_rng(data_seed).random((samples,n,m)).astype(np.float32)
        else:
            values=np.asarray(values,dtype=np.float32)[:samples]
            if len(values)!=samples:raise ValueError('Saved evaluation data too small for requested samples')
        candidates=np.random.default_rng(candidate_seed).random((samples,max(budgets),n,m)).astype(np.float32)
        np.save(out/'test_profiles.npy',values,allow_pickle=False)
        np.savez_compressed(out/'audit_inputs.npz',values=values,random_reports=candidates,**(context_arrays or {}))
        config={'samples':samples,'n_bidders':n,'n_items':m,'data_seed':None if provided_values else data_seed,'candidate_seed':candidate_seed,
                'input_origin':'provided test data, source/hash in provenance' if provided_values else 'numpy default_rng uniform then float32',
                'random_budgets':list(budgets),'pgd_restarts':pgd_restarts,'pgd_steps':pgd_steps,'pgd_lr':pgd_lr,
                'batch_size':batch_size,'candidate_batch_size':candidate_batch,'pmr_threshold':1e-6,
                'scope':status['scope'],'device':device,'software':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform()},
                'regret_interpretation':'finite-search lower bound; empirical zero is not a proof of DSIC',
                'context_protocol':'public contexts fixed for all unilateral candidates' if context_factory else 'no public context',
                'test_profile_sha256':sha256(out/'test_profiles.npy'),'audit_inputs_sha256':sha256(out/'audit_inputs.npz')}
        write_json(out/'config.json',config)
        model=model.to(device).eval()
        for p in model.parameters():p.requires_grad_(False)
        chunks={};max_checks={'max_overallocation':0.,'max_negative_allocation':0.,'max_negative_payment':0.}
        for lo in range(0,samples,batch_size):
            hi=min(lo+batch_size,samples)
            base=context_factory(model,lo,hi) if context_factory else model
            checked=OutputAudit(base)
            result=audit_batch(checked,torch.from_numpy(values[lo:hi]).to(device),
                torch.from_numpy(candidates[lo:hi]).to(device),budgets,pgd_restarts,pgd_steps,pgd_lr,candidate_batch)
            for key,data in result.items():chunks.setdefault(key,[]).append(data)
            for key in max_checks:max_checks[key]=max(max_checks[key],getattr(checked,key))
            print(f"  {metadata['method']} seed={metadata['seed']} audited {hi}/{samples}",flush=True)
        arrays={key:np.concatenate(parts,axis=0) for key,parts in chunks.items()};arrays['values']=values
        np.savez_compressed(out/'profiles.npz',**arrays)
        profiles=per_profile_metrics(arrays,budgets)
        rows=[];bidder_rows=[]
        for i in range(samples):
            row={'profile_index':i,**{key:float(v[i]) for key,v in profiles.items()}}
            for label,key in [(f'random_{b}',f'random_gain_budget{b}') for b in budgets]+[('strong','strong_gain')]:
                row[f'pmr_{label}']=float((arrays[key][i]>1e-6).mean())
            rows.append(row)
            for bidder in range(n):
                bidder_rows.append({'profile_index':i,'bidder_index':bidder,'truth_utility':float(arrays['truth_utility'][i,bidder]),
                    **{key:float(arrays[key][i,bidder]) for key in [*[f'random_gain_budget{b}' for b in budgets],'pgd_gain','strong_gain']}})
        write_csv(out/'profile_metrics.csv',rows);write_csv(out/'bidder_gains.csv',bidder_rows)
        feasible=(max_checks['max_overallocation']<=1e-5 and max_checks['max_negative_allocation']<=1e-5)
        ir=float(profiles['ir_violation_sum'].max())<=1e-5
        no_positive_transfers=max_checks['max_negative_payment']<=1e-5
        inherited_diagnostics=metadata.get('transform_diagnostics',{})
        transform_ok=inherited_diagnostics.get('valid_for_empirical_revenue_comparison',True)
        valid=feasible and ir and no_positive_transfers and transform_ok
        results=[]
        for label,key in [(f'random_{b}',f'random_gain_budget{b}') for b in budgets]+[('strong','strong_gain')]:
            gain=arrays[key].astype(np.float64)
            assert float(gain.max()) <= float(gain.mean())*gain.size+1e-10
            item={'method':metadata['method'],'setting':metadata.get('setting',f'{n}x{m}'),
                'family':metadata.get('family','static_additive'),'seed':metadata['seed'],'seed_role':metadata.get('seed_role','new_training_seed'),
                'source_role':metadata.get('source_role','paper_reimplementation'),'scope':status['scope'],'audit':label,
                'n_profiles':samples,'n_bidders':n,'n_items':m,'revenue':float(profiles['revenue'].mean(dtype=np.float64)),
                'revenue_profile_sd':float(profiles['revenue'].std(ddof=1,dtype=np.float64)),
                'welfare':float(profiles['welfare'].mean(dtype=np.float64)),
                'regret_mean':float(gain.mean()),'regret_max':float(gain.max()),'pmr':float((gain>1e-6).mean()),
                'parameter_count':sum(p.numel() for p in model.parameters()),'menu_size':metadata.get('menu_size',''),
                'max_truthful_ir_violation':float(profiles['ir_violation_sum'].max()),**max_checks,
                'max_bidder_ir_violation':float(np.maximum(-arrays['truth_utility'],0).max()),
                'ir_violation_fraction':float((arrays['truth_utility'] < -1e-6).mean()),
                'feasible_on_checked_reports':feasible,'ir_on_truthful_profiles':ir,
                'no_positive_transfers_on_checked_reports':no_positive_transfers,
                'transform_diagnostic_valid':transform_ok,
                'training_budget_modified':bool(metadata.get('config',{}).get('custom_budget',False)),
                'evidence_status':'empirical_comparison' if valid else 'diagnostic_only_constraint_violation',
                'checkpoint_sha256':metadata.get('checkpoint_sha256',''),'test_profile_sha256':config['test_profile_sha256'],
                'checkpoint_file_bytes':Path(metadata['checkpoint_path']).stat().st_size if metadata.get('checkpoint_path') else '',
                'price_transform_file_bytes':Path(metadata['transform_path']).stat().st_size if metadata.get('transform_path') else '',
                'evaluation_path':str(out.resolve())}
            if extra_metrics:item.update(extra_metrics(arrays))
            results.append(item)
        write_csv(out/'metrics.csv',results)
        provenance=dict(metadata)
        provenance['evaluation_code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'common').glob('*.py'))}
        provenance['profiles_sha256']=sha256(out/'profiles.npz')
        write_json(out/'provenance.json',provenance)
        status.update({'state':'COMPLETED','wall_seconds':time.time()-start,'mechanism_valid_on_checked_reports':valid,
                       'result_use':'smoke_only' if smoke else ('limited_empirical_comparison' if valid else 'diagnostic_only_constraint_violation')})
        write_json(out/'status.json',status)
        return results
    except BaseException as exc:
        import traceback
        status.update({'state':'FAILED','error_type':type(exc).__name__,'error':str(exc)})
        write_json(out/'status.json',status);(out/'error.txt').write_text(traceback.format_exc());raise
