"""GemNet additive menus, following Wang/Jiang/Parkes (2024), Sec. 3.1/A.5.

Implementation choices not determined by the paper are stored in config.
This module never renormalizes selected allocations across bidders: doing so
would destroy the self-bid-independent menu property.
"""
from pathlib import Path
import math
import torch
from torch import nn

PAPER_URL = 'https://arxiv.org/html/2406.07428v3'


class MenuMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(n_in, hidden), n_power_iterations=5),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden), n_power_iterations=5),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden, n_out), n_power_iterations=5))

    def forward(self, x):
        return self.layers(x)

    def lipschitz_inf_bound(self, output_factor=1.0):
        """Conservative norm-converted bound, not the paper's local estimate.

        Effective weights are inspected after spectral normalization; this does
        not assume finite power iteration produced an exact norm of one.
        GELU has global derivative bound < 1.129; sigmoid <= 1/4.
        Numerical SVD is not a machine-verified real-arithmetic certificate.
        """
        linears = [x for x in self.layers if isinstance(x, nn.Linear)]
        product = 1.129 ** 2 * output_factor * math.sqrt(linears[0].in_features)
        details = []
        for layer in linears:
            weight = layer.weight.detach().double().cpu()
            norm = float(torch.linalg.svdvals(weight)[0])
            product *= norm * (1 + 1e-10)
            details.append(norm)
        return product, details


class GemNet(nn.Module):
    def __init__(self, n_bidders=2, n_items=2, menu_size=300, hidden=1024,
                 price_activation='relu', **ignored):
        super().__init__()
        if n_bidders < 2 or n_items < 1 or menu_size < 2:
            raise ValueError('Need >=2 bidders, >=1 item, >=2 total menu entries.')
        if price_activation not in ('relu', 'linear'):
            raise ValueError('price_activation must be relu or linear')
        self.config = dict(n_bidders=n_bidders, n_items=n_items,
                           menu_size=menu_size, hidden=hidden,
                           price_activation=price_activation)
        self.n_bidders, self.n_items, self.menu_size = n_bidders, n_items, menu_size
        self.price_activation = price_activation
        self.bundle_nets = nn.ModuleList([
            MenuMLP((n_bidders-1)*n_items, (menu_size-1)*n_items, hidden)
            for _ in range(n_bidders)])
        self.price_nets = nn.ModuleList([
            MenuMLP((n_bidders-1)*n_items, menu_size-1, hidden)
            for _ in range(n_bidders)])
        # The paper does not specify price initialization. A positive bias
        # avoids an all-dead ReLU price head at initialization.
        if price_activation == 'relu':
            for net in self.price_nets:
                nn.init.constant_(net.layers[-1].bias, 0.5)

    def bidder_menu(self, reports, bidder):
        b, n, m = reports.shape
        if (n, m) != (self.n_bidders, self.n_items):
            raise ValueError('Unexpected bidder/item dimensions')
        others = torch.cat((reports[:, :bidder], reports[:, bidder+1:]), 1).reshape(b, -1)
        allocations = self.bundle_nets[bidder](others).reshape(b, self.menu_size-1, m).sigmoid()
        prices = self.price_nets[bidder](others)
        if self.price_activation == 'relu':
            prices = prices.relu()
        # The fixed null option is last, exactly as Eq. 3.1 describes.
        return (torch.cat((allocations, allocations.new_zeros(b, 1, m)), 1),
                torch.cat((prices, prices.new_zeros(b, 1)), 1))

    def menus(self, reports):
        entries = [self.bidder_menu(reports, i) for i in range(self.n_bidders)]
        return torch.stack([x[0] for x in entries], 1), torch.stack([x[1] for x in entries], 1)

    def forward(self, reports, temperature=None):
        allocations, prices = self.menus(reports)
        utility = (reports[:, :, None, :] * allocations).sum(-1) - prices
        if temperature is not None:
            probabilities = (utility * float(temperature)).softmax(-1)
            return ((probabilities[..., None] * allocations).sum(-2),
                    (probabilities * prices).sum(-1))
        selected = utility.argmax(-1)
        return (allocations.gather(2, selected[..., None, None].expand(-1, -1, 1, self.n_items)).squeeze(2),
                prices.gather(2, selected[..., None]).squeeze(2))

    def lipschitz_bounds(self):
        self.eval()
        with torch.no_grad():
            self(torch.zeros(1, self.n_bidders, self.n_items, device=next(self.parameters()).device))
        bounds = []
        for i in range(self.n_bidders):
            a, av = self.bundle_nets[i].lipschitz_inf_bound(0.25)
            p, pv = self.price_nets[i].lipschitz_inf_bound(1.0)
            bounds.append(dict(bidder=i, allocation=a, price=p,
                               allocation_layer_spectral_norms=av, price_layer_spectral_norms=pv))
        return dict(allocation=max(x['allocation'] for x in bounds),
                    price=max(x['price'] for x in bounds), per_bidder=bounds,
                    construction='product effective spectral 2-norms, GELU 1.129^2, sigmoid 1/4, sqrt(input_dimension)',
                    machine_verified=False)


def load_checkpoint(path, device='cpu', transform_path=None):
    obj = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = GemNet(**obj['model_config']).to(device)
    model.load_state_dict(obj['state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if transform_path is not None:
        from .transform import TransformedGemNet
        model = TransformedGemNet.from_file(model, transform_path).to(device)
    return model
