import math
import torch

beta_0 = 0.1
beta_1 = 20.0

beta = lambda t: (1-t)*beta_0 + t*beta_1

def w1(t):
    return 0.5*(1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))**2

def dw1dt(t):
    out = (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
    out = out*torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))
    return out

def w1_cond(t):
    return w1(t)*w1(1-t)

def dw1dt_cond(t):
    return dw1dt(t)*w1(1-t) - w1(t)*dw1dt(1-t)

def get_q(config):
    diffusion_based = {'diffusion', 'conditional', 'classification'}
    if config.model.task in diffusion_based:
        return get_q_diffusion(config)
    elif 'heat' == config.model.task:
        sigma = lambda t: t
        w = lambda t: 0.5*t**2
        dwdt = lambda t: t
        def q_t(data, t):
            assert (2 == data.dim())
            ydim = config.data.ydim
            if ydim > 0:
                x = data[:,:-ydim]
            else:
                x = data
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            blurred_img = blur(x.reshape([B, C, H, W]),t)
            blurred_img = blurred_img.reshape([B, C*H*W])
            return blurred_img + sigma(t.view([B, 1]))*torch.randn_like(blurred_img)
        return q_t, sigma, w, dwdt
    elif 'color' == config.model.task:
        sigma = lambda t: 1e-1*t
        w = lambda t: 0.5*t**2
        dwdt = lambda t: t
        def q_t(data, t):
            assert (2 == data.dim())
            while (data.dim() > t.dim()): t = t.unsqueeze(-1)
            ydim = config.data.ydim
            if ydim > 0:
                x = data[:,:-ydim]
            else:
                x = data
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            gray_x = x.mean(1,keepdim=True).repeat([1,C,1,1])
            eps = torch.randn_like(x)
            output = t*gray_x + (1-t)*x + sigma(t)*eps
            if config.model.cond_channels > 0:
                output = torch.hstack([output, eps])
                C = C + config.model.cond_channels
            return output.reshape([B, C*H*W])
        return q_t, sigma, w, dwdt
    elif 'superres' == config.model.task:
        sigma = lambda t: 1e-1*t
        w = lambda t: 0.5*t**2
        dwdt = lambda t: t
        def q_t(data, t):
            assert (2 == data.dim())
            while (data.dim() > t.dim()): t = t.unsqueeze(-1)
            ydim = config.data.ydim
            if ydim > 0:
                x = data[:,:-ydim]
            else:
                x = data
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            downscale_x = torch.nn.functional.interpolate(x, size=(H//2,W//2), mode='nearest')
            downscale_x = torch.nn.functional.interpolate(downscale_x, size=(H,W), mode='bilinear')
            eps = torch.randn_like(x)
            output = t*downscale_x + (1-t)*x + sigma(t)*eps
            if config.model.cond_channels > 0:
                output = torch.hstack([output, eps])
                C = C + config.model.cond_channels
            return output.reshape([B, C*H*W])
        return q_t, sigma, w, dwdt
    else:
        raise NameError('config.model.task is undefined')


def get_q_diffusion(config):
    name = config.model.sigma
    if 'vpsde' == name:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: torch.sqrt(1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
        w = w1
        dwdt = dw1dt
    elif 'subvpsde' == name:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: 1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))
        w = w1
        dwdt = dw1dt
    elif 'simple' == name:
        alpha = lambda t: torch.sqrt(1-t)
        sigma = lambda t: torch.sqrt(t)
        w = lambda t: t**2*(1-t)
        dwdt = lambda t: 2*t-3*t**2
    elif 'dimple' == name:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = lambda t: 0.5*t**2
        dwdt = lambda t: t
    else:
        raise NotImplementedError('there is no %' % label)
    def q_t(data, t):
        assert (2 == data.dim())
        while (data.dim() > t.dim()): t = t.unsqueeze(-1)
        ydim = config.data.ydim
        if ydim > 0:
            x = data[:,:-ydim]
        else:
            x = data
        B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size    
        eps = torch.randn_like(x)
        output = x*alpha(t) + sigma(t)*eps
        if config.model.cond_channels > 0:
            output = torch.hstack([output, eps])
            C = C + config.model.cond_channels
        return output.reshape([B, C*H*W])
    return q_t, sigma, w, dwdt


def get_s(net, config):
    label = config.model.s
    C, H, W = config.model.num_channels, config.data.image_size, config.data.image_size
    C_cond, ydim = config.model.cond_channels, config.data.ydim
    if C_cond > 0:
        def s(t,x):
            x = x.view(-1, C+C_cond, H, W)
            return net(t, x[:,:C,:,:], x[:,C:,:,:])
    else:
        def s(t,x):
            return net(t, x.view(-1,C,H,W))
    return s
    
##################################### BLUR ###########################################
    
def dct(x, norm=None):
    B, N = x.shape
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(B,N)
    return V

def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.reshape(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    
    v = torch.fft.ifft(V_r + V_i*1j, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)

def dct_2d(x, norm=None):
    B, C, H, W = x.shape
    x1 = dct(x.reshape(B*C*H, W), norm=norm)
    x1 = x1.reshape(B*C, H, W).transpose(-1, -2)
    x2 = dct(x1.reshape(B*C*H, W), norm=norm)
    return x2.reshape(B, C, H, W).transpose(-1, -2)

def idct_2d(x, norm=None):
    B, C, H, W = x.shape
    x1 = idct(x.reshape(B*C*H, W), norm=norm)
    x1 = x1.reshape(B*C, H, W).transpose(-1, -2)
    x2 = idct(x1.reshape(B*C*H, W), norm=norm)
    return x2.reshape(B, C, H, W).transpose(-1, -2)

def blur(x, t):
    sigma_min, sigma_max = 1e-1, 20
    B, C, H, W = x.shape
    device = x.device
    freqs = math.pi * torch.linspace(0, H-1, H).to(device)/H
    frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
    frequencies_squared = frequencies_squared.unsqueeze(0).unsqueeze(0)
    while (x.dim() > t.dim()): t = t.unsqueeze(-1)
    x_proj = dct_2d(x, norm='ortho')
    t_prime = 0.5*torch.exp(2*(t*math.log(sigma_max) + (1-t)*math.log(sigma_min)))
    x_proj = torch.exp(-frequencies_squared*t_prime)*x_proj
    x_reconstructed = idct_2d(x_proj, norm='ortho')
    return x_reconstructed

######################################################################################
