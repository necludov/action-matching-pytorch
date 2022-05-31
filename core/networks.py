import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


class SmallMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, relu=False):
        super(SmallMLP, self).__init__()
        self._built = False
        if relu:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                nn.LeakyReLU(.2),
                layer(n_hid, n_hid),
                nn.LeakyReLU(.2),
                layer(n_hid, n_out)
            )
        else:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                nn.SiLU(n_hid),
                layer(n_hid, n_hid),
                nn.SiLU(n_hid),
                layer(n_hid, n_out)
            )

    def forward(self, x, t):
        x = x.view(x.size(0), -1)
        t = t.view(t.size(0), 1)
        x = torch.hstack([x,t])
        out = self.net(x)
        out = out.squeeze()
        return out
