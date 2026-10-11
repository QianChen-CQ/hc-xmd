"""Method-faithful reimplementations, NOT recovered historical checkpoints.

RegretNet follows Duetting et al. ICML 2019 and the authors' MIT-licensed
additive_3x10_uniform configuration. AMenuNet is a new implementation of
Duan et al. NeurIPS 2023, cross-checked against its published architecture.
"""
from pathlib import Path
import torch
from torch import nn


class RegretNet(nn.Module):
    """2019 additive architecture, including dummy bidder AND dummy item."""
    def __init__(self, n=3, m=10, hidden=100, hidden_layers=5, **unused):
        super().__init__()
        self.n, self.m = n, m
        def branch(output):
            layers, width = [], n*m
            for _ in range(hidden_layers):
                layers.extend((nn.Linear(width, hidden), nn.Tanh()))
                width = hidden
            layers.append(nn.Linear(width, output))
            return nn.Sequential(*layers)
        self.allocation_net = branch((n+1)*(m+1))
        self.payment_net = branch(n)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, values):
        x = values.reshape(-1, self.n*self.m)
        logits = self.allocation_net(x).reshape(-1, self.n+1, self.m+1)
        allocation = logits.softmax(dim=1)[:, :self.n, :self.m]
        payment_fraction = self.payment_net(x).sigmoid()
        payments = payment_fraction * (allocation*values).sum(dim=-1)
        return allocation, payments


class AxialBlock(nn.Module):
    """Bidder, item and global features; zero-dropout transformer encoders."""
    def __init__(self, width, heads, output):
        super().__init__()
        self.item_attention = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width, dropout=0.0, batch_first=True)
        self.bidder_attention = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width, dropout=0.0, batch_first=True)
        self.merge = nn.Sequential(nn.Linear(3*width, width), nn.ReLU(),
                                   nn.Linear(width, output))

    def forward(self, x):
        batches, bidders, items, width = x.shape
        along_items = self.item_attention(x.reshape(-1, items, width))
        along_items = along_items.reshape(batches, bidders, items, width)
        along_bidders = self.bidder_attention(
            x.transpose(1, 2).reshape(-1, bidders, width))
        along_bidders = along_bidders.reshape(batches, items, bidders, width).transpose(1, 2)
        global_feature = x.mean(dim=(1, 2), keepdim=True).expand_as(x)
        return self.merge(torch.cat((along_items, along_bidders, global_feature), dim=-1))


class AMenuNet(nn.Module):
    """Non-contextual learned-menu affine maximizer, exact argmax at test time.

    The bidder/item IDs are fixed public representations, not bids. Therefore
    allocation menus, positive weights and boosts are independent of reports.
    Training uses the paper's smooth-choice surrogate; forward ALWAYS applies
    the actual hard AMA allocation and Clarke-pivot payment equations.
    """
    def __init__(self, n=3, m=10, menu_size=512, embed_dim=10, width=64,
                 layers=3, heads=4, allocation_temperature=10.0,
                 choice_temperature=500.0, **unused):
        super().__init__()
        self.n, self.m, self.menu_size = n, m, menu_size
        self.allocation_temperature = allocation_temperature
        self.choice_temperature = choice_temperature
        self.bidder_embedding = nn.Embedding(n+1, embed_dim)
        self.item_embedding = nn.Embedding(m, embed_dim)
        self.context_projection = nn.Sequential(nn.Linear(2*embed_dim, width),
                                                nn.ReLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            AxialBlock(width, heads, width if j+1 < layers else 2*menu_size+1)
            for j in range(layers)])
        self.boost_projection = nn.Sequential(nn.Linear(menu_size, menu_size),
                                              nn.ReLU(), nn.Linear(menu_size, menu_size))
        self._evaluation_menu = None

    def train(self, mode=True):
        self._evaluation_menu = None
        return super().train(mode)

    def _apply(self, function):
        self._evaluation_menu = None
        return super()._apply(function)

    def build_menu(self):
        device = self.bidder_embedding.weight.device
        bidder = self.bidder_embedding(torch.arange(self.n+1, device=device))
        item = self.item_embedding(torch.arange(self.m, device=device))
        pairs = torch.cat((bidder[:, None].expand(-1, self.m, -1),
                           item[None].expand(self.n+1, -1, -1)), dim=-1)
        features = self.context_projection(pairs.unsqueeze(0))
        for block in self.blocks:
            features = block(features)
        features = features.squeeze(0)
        allocation_logits = features[..., :self.menu_size]
        allocations = (allocation_logits*self.allocation_temperature).softmax(dim=0)
        allocations = allocations[:self.n].permute(2, 0, 1)
        weights = features[..., -1].mean(dim=-1).sigmoid()[:self.n]
        boosts = self.boost_projection(features[..., self.menu_size:2*self.menu_size].mean(dim=(0, 1)))
        # The empty allocation/zero boost provides a no-sale alternative.
        allocations = torch.cat((allocations, allocations.new_zeros(1, self.n, self.m)))
        boosts = torch.cat((boosts, boosts.new_zeros(1)))
        if not torch.isfinite(weights).all() or not (weights > 0).all():
            raise FloatingPointError('AMA bidder weights must be finite and strictly positive')
        return allocations, weights, boosts

    def menu(self):
        if self.training:
            return self.build_menu()
        if self._evaluation_menu is None:
            with torch.no_grad():
                # Stable evaluation arithmetic prevents float32 cancellation from
                # appearing as a tiny, artificial DSIC violation.
                self._evaluation_menu = tuple(t.detach().double() for t in self.build_menu())
        return self._evaluation_menu

    def _auction(self, values, soft):
        allocations, weights, boosts = self.menu()
        calculation_values = values.to(dtype=allocations.dtype)
        agent_scores = torch.einsum('knm,bnm->bkn', allocations, calculation_values)*weights
        scores = agent_scores.sum(dim=-1)+boosts
        if soft:
            probability = (scores*self.choice_temperature).softmax(dim=-1)
            allocation = torch.einsum('bk,knm->bnm', probability, allocations)
            chosen_others = (probability.unsqueeze(-1)*(scores.unsqueeze(-1)-agent_scores)).sum(dim=1)
            removed_scores = scores.unsqueeze(-1)-agent_scores
            removed_probability = (removed_scores*self.choice_temperature).softmax(dim=1)
            outside = (removed_probability*removed_scores).sum(dim=1)
        else:
            winner = scores.argmax(dim=1)
            allocation = allocations[winner]
            rows = torch.arange(values.shape[0], device=values.device)
            chosen_others = scores[rows, winner, None]-agent_scores[rows, winner]
            outside = (scores.unsqueeze(-1)-agent_scores).max(dim=1).values
        payments = (outside-chosen_others)/weights
        return allocation, payments

    def forward(self, values):
        return self._auction(values, soft=False)

    def training_forward(self, values):
        return self._auction(values, soft=True)


def create_model(method, config=None):
    config = {} if config is None else config
    cls = {'RegretNet': RegretNet, 'AMenuNet': AMenuNet}.get(method)
    if cls is None:
        raise ValueError(f'Unsupported Phase I method: {method}')
    return cls(**config)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'HCXMD_PHASE1_V5':
        raise ValueError('Not a Phase I V5 checkpoint')
    model = create_model(checkpoint['method'], checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state'}
    return model, metadata
