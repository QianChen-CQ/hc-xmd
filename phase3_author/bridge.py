"""Read original network definitions without executing upstream training entrypoints.

AST extraction below retains the exact author class bodies; only unrelated imports,
training routines, and plotting entrypoints are omitted. No learned results are
embedded here. Checkpoints are tensors only; metadata remains inspectable JSON.
"""
from pathlib import Path
import ast
import hashlib
import json
import types
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
METHODS = ('CITransNet', 'CITransNetNoContext', 'PreferenceNet', 'ProportionNet',
           'RegretNetPreferenceBaseline', 'RegretNetProportionBaseline')
CONTEXT_METHODS = ('CITransNet','CITransNetNoContext')
PROPORTION_METHODS = ('ProportionNet','RegretNetProportionBaseline')


class ZeroContextMechanism(nn.Module):
    """Review-added controlled ablation; the valuation distribution is unchanged."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, batch_data):
        values, bidder_context, item_context = batch_data
        return self.original((values, torch.zeros_like(bidder_context), torch.zeros_like(item_context)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_provenance(method):
    """Source identities from the frozen manifest; paper reconstructions have no upstream commit."""
    if method not in METHODS:
        raise ValueError(method)
    sources = {entry['method']:entry for entry in json.loads((HERE/'SOURCES.json').read_text())['sources']}
    ablation = method in ('CITransNetNoContext','RegretNetProportionBaseline','RegretNetPreferenceBaseline')
    if method in CONTEXT_METHODS:
        source = network_source = sources['CITransNet']
    elif method in PROPORTION_METHODS:
        source, network_source = sources['ProportionNet'], sources['PreferenceNet']
    else:
        source = network_source = sources['PreferenceNet']
    return {
        'source_url':source.get('url',source['paper']),
        'source_commit':source.get('commit',''),
        'paper_url':source['paper'],
        'source_role':'controlled_ablation' if ablation else 'paper_directed_reimplementation' if method=='ProportionNet' else 'author_architecture_training_adapter',
        'source_scope':source['source_role'] + (' Review-added controlled ablation; not an original-paper released variant.' if ablation else ''),
        'network_source_url':network_source['url'],
        'network_source_commit':network_source['commit'],
        'network_source_scope':'Architecture class definitions only; training is the declared review adapter or paper reconstruction.',
    }


def _classes(path, names):
    parsed = ast.parse(path.read_text())
    selected = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f'Missing original class definitions in {path}')
    env = {'torch': torch, 'nn': nn, 'F': F, '__name__': 'phase3_author.original_network_classes'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def build_model(method, n_agents=3, n_items=10):
    manifest = json.loads((HERE / 'SOURCES.json').read_text())
    for entry in manifest['files']:
        if sha256(HERE / entry['path']) != entry['sha256']:
            raise RuntimeError(f"Upstream source hash mismatch: {entry['path']}")
    if method not in METHODS:
        raise ValueError(method)
    if method in CONTEXT_METHODS:
        src = HERE / 'vendor/CITransNet/network.py'
        classes = _classes(src, ['Transformer2DNet', 'TransformerMechanism'])
        model = classes['TransformerMechanism'](None, None, 10, 3, 4, 64,
                                               0, 1, True, False)
        return ZeroContextMechanism(model) if method=='CITransNetNoContext' else model
    src = HERE / 'vendor/PreferenceNet/learnable_preferences/regretnet.py'
    classes = _classes(src, ['View', 'View_Cut', 'RegretNet'])
    hidden = 3
    if method in PROPORTION_METHODS:
        # Original Table 1 total hidden layers (not author's "additional layers").
        table_layers = {(1,m):2 for m in range(2,7)}
        table_layers.update({(2,m):2 if m<=4 else 3 for m in range(2,7)})
        table_layers.update({(3,m):2 if m<=3 else 3 for m in range(2,7)})
        table_layers.update({(4,m):2 if m==2 else 3 if m<=4 else 4 for m in range(2,7)})
        table_layers.update({(5,m):3 if m<=3 else 4 if m<=5 else 5 for m in range(2,7)})
        hidden = table_layers.get((n_agents,n_items),3)
    return classes['RegretNet'](n_agents, n_items, hidden_layer_size=100,
                               n_hidden_layers=hidden-1, activation='relu', separate=False)


def build_preference_classifier(n_agents, n_items):
    src = HERE / 'vendor/PreferenceNet/learnable_preferences/network.py'
    return _classes(src, ['PreferenceNet'])['PreferenceNet'](n_agents, n_items, 100)


class ContextBatchModel(nn.Module):
    """Bind one audit batch's PUBLIC contexts; flattening must be profile-major.

    Common audit uses [B,K,n,m].reshape(B*K,n,m), hence each context is repeated
    K times. Contexts cannot be resampled when a bidder changes their report.
    """
    def __init__(self, model, bidder_context, item_context):
        super().__init__()
        self.model = model
        self.register_buffer('bidder_context', bidder_context)
        self.register_buffer('item_context', item_context)

    def forward(self, reports):
        batch = self.bidder_context.shape[0]
        if reports.shape[0] % batch:
            raise ValueError('Context-bound audit requires whole profile-major batches')
        repeats = reports.shape[0] // batch
        bc = self.bidder_context.repeat_interleave(repeats, 0)
        ic = self.item_context.repeat_interleave(repeats, 0)
        return self.model((reports, bc, ic))


def load_checkpoint(path, device='cpu'):
    path = Path(path)
    if path.is_dir():
        path = path / 'checkpoint.pt'
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    if metadata.get('source_manifest_sha256') and sha256(HERE / 'SOURCES.json') != metadata['source_manifest_sha256']:
        raise RuntimeError('Checkpoint source manifest does not match current source snapshot')
    if metadata.get('checkpoint_sha256') and sha256(path) != metadata['checkpoint_sha256']:
        raise RuntimeError('Checkpoint does not match metadata SHA256')
    model = build_model(metadata['method'], metadata['n_agents'], metadata['n_items']).to(device)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata


def load_preference_classifier(directory, device='cpu'):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    model = build_preference_classifier(metadata['n_agents'], metadata['n_items']).to(device)
    model.load_state_dict(torch.load(directory / 'preference_classifier.pt',
                                    map_location='cpu', weights_only=True), strict=True)
    return model.eval()


def direct_metrics(allocation, distance=0.0):
    """Separate original ProportionNet Eq4 from PreferenceNet's absolute TVF.

    Category is all advertisers. Both sums count ordered item pairs, matching
    the respective definitions. No invented burden or human scores are added.
    """
    difference = allocation.unsqueeze(-1) - allocation.unsqueeze(-2)
    # B,n,m,m -> B,m,m; direction u-v matters in original Eq.(4).
    positive_distance = difference.clamp_min(0).sum(1)
    l1_distance = difference.abs().sum(1)
    paper_unfairness = (positive_distance - distance).clamp_min(0).sum(-1)
    tvf_unfairness = (l1_distance - distance).clamp_min(0).sum(-1)
    norm = allocation.clamp_min(1e-8)
    norm = norm / norm.sum(-1, keepdim=True)
    entropy = -(norm * norm.log()).sum((1, 2))
    norm_bidder = allocation.clamp_min(1e-8)
    norm_bidder = norm_bidder / norm_bidder.sum(1, keepdim=True)
    quota = norm_bidder.amin((1, 2))
    if allocation.shape[1] <= 1:
        # Author diagnostic convention: quota is inapplicable for one agent.
        quota = torch.zeros_like(quota)
    return {'paper_unfairness_by_item': paper_unfairness,
            'paper_unfairness_sum': paper_unfairness.sum(1),
            'absolute_tvf_by_item': tvf_unfairness,
            'absolute_tvf_sum': tvf_unfairness.sum(1),
            'maximum_l1_allocation_distance': l1_distance.amax((1, 2)),
            'entropy': entropy, 'minimum_normalized_quota': quota}


def preference_score(allocation, objective='tvf', distance=0.0):
    metrics = direct_metrics(allocation, distance)
    if objective == 'tvf':
        return -metrics['absolute_tvf_sum']
    if objective == 'entropy':
        return metrics['entropy']
    if objective == 'quota':
        return metrics['minimum_normalized_quota']
    raise ValueError(objective)


def family_metrics(allocation, metadata):
    """Tensor-valued family-specific per-profile metrics for the common exporter."""
    metrics = direct_metrics(allocation, metadata.get('fairness_distance', 0.0))
    result = {key: value for key, value in metrics.items() if value.ndim == 1}
    result['tvf_constraint_satisfaction_1e3'] = (metrics['maximum_l1_allocation_distance'] <= metadata.get('fairness_distance',0.0)+1e-3).to(allocation.dtype)
    threshold = metadata.get('preference_threshold')
    if threshold is not None:
        scores = preference_score(allocation, metadata.get('preference_objective', 'tvf'), metadata.get('fairness_distance',0.0))
        result['pca'] = (scores > threshold).to(allocation.dtype)
        result['preference_ground_truth_score'] = scores
    return result


def evaluation_extras(model, metadata, train_dir, device, samples, smoke):
    """Prepare the root common.evaluate interface and preserve original metrics."""
    import numpy as np
    directory = Path(train_dir)
    count = 4 if smoke else samples
    with np.load(directory / 'eval_data.npz', allow_pickle=False) as saved:
        data = {key: saved[key][:count].copy() for key in saved.files}
    if len(data['values']) != count:
        raise ValueError('Training artifact evaluation data too small; train with --eval-samples matching the suite')
    contextual = bool(metadata['contextual'])
    metadata = dict(metadata)
    metadata.update(method_provenance(metadata['method']))
    metadata.update(n_bidders=metadata['n_agents'], menu_size='',
                    family='contextual_additive' if contextual else 'synthetic_preference_fairness',
                    seed_role='new_training_seed')
    if contextual:
        kwargs = dict(values=data['values'],
                      context_arrays={key: value for key,value in data.items() if key!='values'},
                      context_factory=lambda base,lo,hi: ContextBatchModel(base,
                          torch.as_tensor(data['bidder_context'][lo:hi],device=device),
                          torch.as_tensor(data['item_context'][lo:hi],device=device)))
    else:
        kwargs = dict(values=data['values'])
    classifier = load_preference_classifier(directory, device) if metadata['method']=='PreferenceNet' else None

    def extra_metrics(arrays):
        allocation = torch.from_numpy(arrays['allocation'])
        metrics = family_metrics(allocation, metadata)
        answer = {key: float(value.double().mean()) for key,value in metrics.items()}
        answer.update({key+'_profile_sd': float(value.double().std()) for key,value in metrics.items()})
        # Authors report summed bidder regret; common review metric is bidder mean.
        gain = np.asarray(arrays['strong_gain'],dtype=np.float64)
        answer['paper_regret_sum_mean_strong'] = float(gain.sum(1).mean())
        answer['paper_regret_sum_profile_sd_strong'] = float(gain.sum(1).std(ddof=1))
        answer['paper_regret_bidder_mean_strong'] = float(gain.mean())
        if classifier is not None:
            predictions=[]
            with torch.no_grad():
                for lo in range(0,len(allocation),512):
                    hi=min(lo+512,len(allocation))
                    pred=classifier(torch.as_tensor(arrays['values'][lo:hi],device=device),
                                    allocation[lo:hi].to(device),torch.as_tensor(arrays['payments'][lo:hi],device=device))
                    predictions.append(pred.cpu())
            probability=torch.cat(predictions)
            answer['preference_probability']=float(probability.double().mean())
            answer['preference_probability_profile_sd']=float(probability.double().std())
        return answer
    kwargs['extra_metrics']=extra_metrics
    return metadata,kwargs
