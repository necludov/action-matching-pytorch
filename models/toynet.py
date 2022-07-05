import math

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    def __init__(self, n_dims=2, n_out=100, n_hid=300, layer=nn.Linear, relu=False):
        super(SmallMLP, self).__init__()
        self._built = False
        self.net = nn.Sequential(
            layer(n_dims, n_hid),
            nn.LeakyReLU(.2) if relu else nn.SiLU(n_hid),
            layer(n_hid, n_hid),
            nn.LeakyReLU(.2) if relu else nn.SiLU(n_hid),
            layer(n_hid, n_out)
        )

    def forward(self, t, x):
        x = x.view(x.size(0), -1)
        t = t.view(t.size(0), 1)
        x = torch.hstack([t,x])
        x = self.net(x)
        out = x.sum(dim=1, keepdim=True)
        return out
