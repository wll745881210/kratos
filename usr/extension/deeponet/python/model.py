"""Stacked DeepONet torch module (DeepONet surrogate for Kratos chemistry).

Branch:  [T, rho, X_1..n_spe, F_1..n_hv]  (transformed + z-scored) -> n_out*p
Trunk:   log10(dt) (z-scored)             -> p
Combine: yhat[z] = bias[z] + sum_k B[z*p + k] * Ttrunk[k]

Outputs (z-scored space): [X'_1..n_spe, T']  (n_out = n_spe + 1).
Hidden activation: GELU (erf form). Final layers: linear.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        dims = [n_in] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers += [nn.Linear(dims[-1], n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StackedDeepONet(nn.Module):
    def __init__(self, n_in, n_out, n_basis=32,
                 branch_hidden=(512, 256), trunk_hidden=(32,)):
        super().__init__()
        self.n_in, self.n_out, self.p = n_in, n_out, n_basis
        self.branch = MLP(n_in, branch_hidden, n_out * n_basis)
        self.trunk = MLP(1, trunk_hidden, n_basis)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, xin, tdt):
        """xin [N, n_in] (transformed+z-scored), tdt [N] (z-scored log10 dt)."""
        B = self.branch(xin)                       # [N, n_out*p]
        T = self.trunk(tdt.unsqueeze(1))           # [N, p]
        B = B.view(-1, self.n_out, self.p)         # [N, n_out, p]
        y = self.bias + (B * T.unsqueeze(1)).sum(-1)  # [N, n_out]
        return y
