import torch
import torch.nn as nn
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='anet')
class ActionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # Downsampling block
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(nn.Linear(in_ch*self.all_resolutions[-1]**2, 256))
        self.all_modules = nn.ModuleList(modules)
        
    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.

        # Downsampling block
        h = modules[m_idx](h)
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](h, temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](h)
                m_idx += 1

        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        
        flatten_h = h.reshape([h.shape[0],-1])
        h = modules[m_idx](flatten_h)
        m_idx += 1
        assert m_idx == len(modules)
        
        h = torch.logsumexp(h, dim=1, keepdim=True)
        return h
