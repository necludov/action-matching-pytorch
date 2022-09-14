import torch
import math
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

# @utils.register_model(name='anet')
# class ActionNet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.act = get_act(config)
#         self.temb_act = nn.SiLU()

#         self.nf = nf = config.model.nf
#         self.nc = nc = config.model.num_channels
#         self.n_phases = config.model.n_phases
#         ch_mult = config.model.ch_mult
#         self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
#         self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
#         dropout = config.model.dropout
#         skip_connection = config.model.skip
#         resamp_with_conv = config.model.resamp_with_conv
#         self.num_resolutions = num_resolutions = len(ch_mult)
#         self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

#         AttnBlock = functools.partial(layers.AttnBlock)
#         ResnetBlock = functools.partial(ResnetBlockDDPM, act=self.act, temb_dim=4 * nf, 
#                                         dropout=dropout, skip_connection=skip_connection)
        
#         # condition on time
#         modules = []
#         modules.append(nn.Linear(nf, nf*4))
#         modules[-1].weight.data = default_initializer()(modules[-1].weight.data.shape)
#         nn.init.zeros_(modules[-1].bias)
#         modules.append(nn.Linear(nf*4, nf*4))
#         modules[-1].weight.data = default_initializer()(modules[-1].weight.data.shape)
#         nn.init.zeros_(modules[-1].bias)
#         modules.append(nn.Linear(nf*4, nf*4))
#         modules[-1].weight.data = default_initializer()(modules[-1].weight.data.shape)
#         nn.init.zeros_(modules[-1].bias)

#         self.conditional = config.model.cond_channels > 0
#         modules.append(conv3x3(nc + config.model.cond_channels, nf))
#         hs_c = [nf]
#         in_ch = nf
#         for i_level in range(num_resolutions):
#             # residual blocks for this resolution
#             for i_block in range(num_res_blocks):
#                 out_ch = nf * ch_mult[i_level]
#                 modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
#                 in_ch = out_ch
#                 if all_resolutions[i_level] in attn_resolutions:
#                     modules.append(AttnBlock(channels=in_ch))
#                 hs_c.append(in_ch)
#             if i_level != num_resolutions - 1:
#                 modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
#                 hs_c.append(in_ch)

#         in_ch = hs_c[-1]
#         modules.append(ResnetBlock(in_ch=in_ch))
#         modules.append(AttnBlock(channels=in_ch))
#         modules.append(ResnetBlock(in_ch=in_ch))
#         modules.append(torch.nn.AvgPool2d(self.all_resolutions[-1]))
#         modules.append(nn.Linear(in_ch, 256))
#         modules.append(nn.Linear(256, 1))
#         self.all_modules = nn.ModuleList(modules)
        
#     def forward(self, t, x, condition=None):
#         bs = x.shape[0]
#         t = t.flatten()
#         t = t.expand(bs)
        
#         modules = self.all_modules
#         m_idx = 0
        
#         temb = layers.get_timestep_embedding(t, self.nf)
#         temb = modules[m_idx](temb)
#         m_idx += 1
#         temb = modules[m_idx](self.temb_act(temb))
#         m_idx += 1
#         temb = modules[m_idx](self.temb_act(temb))
#         m_idx += 1
        
#         if self.n_phases is not None:
#             if self.n_phases > 0:
#                 x = layers.get_circular_embedding(x, self.n_phases, self.nc)
#         if condition is not None:
#             x = torch.hstack([x, condition.detach()])

#         # Downsampling block
#         h = modules[m_idx](x)
#         m_idx += 1
#         for i_level in range(self.num_resolutions):
#             # Residual blocks for this resolution
#             for i_block in range(self.num_res_blocks):
#                 h = modules[m_idx](h, temb)
#                 m_idx += 1
#                 if h.shape[-1] in self.attn_resolutions:
#                     h = modules[m_idx](h)
#                     m_idx += 1
#             if i_level != self.num_resolutions - 1:
#                 h = modules[m_idx](h)
#                 m_idx += 1

#         h = modules[m_idx](h, temb)
#         m_idx += 1
#         h = modules[m_idx](h)
#         m_idx += 1
#         h = modules[m_idx](h, temb)
#         m_idx += 1
        
#         h = modules[m_idx](h)
#         m_idx += 1
#         h = h.reshape([h.shape[0],-1])
#         h = modules[m_idx](self.act(h))
#         m_idx += 1
#         h = modules[m_idx](self.act(h))
#         m_idx += 1
#         assert m_idx == len(modules)
#         return h
    
@utils.register_model(name='anet')
class ActionNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)

    self.nf = nf = config.model.nf
    self.nc = nc = config.model.num_channels
    self.n_phases = config.model.n_phases
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    # self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    
    # Condition on noise levels.
    modules = [nn.Linear(nf, nf * 4)]
    modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
    nn.init.zeros_(modules[0].bias)
    modules.append(nn.Linear(nf * 4, nf * 4))
    modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
    nn.init.zeros_(modules[1].bias)

    # Downsampling block
    self.conditional = config.model.cond_channels
    modules.append(conv3x3(nc + config.model.cond_channels, nf))
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

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, nc, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, t, x, condition=None):
    bs = x.shape[0]
    t = t.flatten()
    t = t.expand(bs)
    
    modules = self.all_modules
    m_idx = 0

    # timestep/scale embedding
    temb = layers.get_timestep_embedding(t, self.nf)
    temb = modules[m_idx](temb)
    m_idx += 1
    temb = modules[m_idx](self.act(temb))
    m_idx += 1
    
    if self.n_phases is not None:
        if self.n_phases > 0:
            x = layers.get_circular_embedding(x, self.n_phases, self.nc)
    if condition is not None:
        x = torch.hstack([x, condition.detach()])

    # Downsampling block
    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)
    return 0.5*((h-x[:,:self.nc])**2).sum([1,2,3]).unsqueeze(1)