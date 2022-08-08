import math
import torch

beta_0 = 0.1
beta_1 = 20.0

beta = lambda t: (1-t)*beta_0 + t*beta_1

def w1(t):
    return (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))

def dw1dt(t):
    out = torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))
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
        noise_sigma = lambda t: torch.sqrt(1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
        w = w1
        dwdt = dw1dt
        w = lambda t: torch.ones_like(t)
        dwdt = lambda t: torch.zeros_like(t)
        def heat_eq_forward(data, t):
            assert (2 == data.dim())
            ydim = config.data.ydim
            if ydim > 0:
                u = data[:,:-ydim]
            else:
                u = data
            B, C, H, W = u.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            sigma_min, sigma_max = 1e-1, 20
            u = u.reshape([B, C, H, W])
            device = u.device
            freqs = math.pi * torch.linspace(0, H-1, H).to(device)/H
            frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
            frequencies_squared = frequencies_squared.unsqueeze(0).unsqueeze(0)
            while (u.dim() > t.dim()): t = t.unsqueeze(-1)
            u_proj = dct_2d(u, norm='ortho')
            t_prime = 0.5*torch.exp(2*(t*math.log(sigma_max) + (1-t)*math.log(sigma_min)))
            u_proj = torch.exp(-frequencies_squared*t_prime)*u_proj
            u_reconstucted = idct_2d(u_proj, norm='ortho')
            blurred_img = u_reconstucted.reshape([B, C*H*W])
            t = t.reshape([B, 1])
            return blurred_img + noise_sigma(t)*torch.randn_like(blurred_img)
        return heat_eq_forward, None, None, w, dwdt
    else:
        raise NameError('config.model.task is undefined')


def get_q_diffusion(config):
    name = config.model.evolution
    if 'vpsde' == name:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: torch.sqrt(1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
        w = w1
        dwdt = dw1dt
        w_cond = w1_cond
        dwdt_cond = dw1dt_cond
    elif 'subvpsde' == name:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: 1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))
        w = w1
        dwdt = dw1dt
        w_cond = w1_cond
        dwdt_cond = dw1dt_cond
    elif 'simple' == name:
        alpha = lambda t: torch.sqrt(1-t)
        sigma = lambda t: torch.sqrt(t)
        w = lambda t: t*(1-t)
        dwdt = lambda t: 1-2*t
        w_cond = w
        dwdt_cond = dwdt
    elif 'dimple' == name:
        alpha = lambda t: 1-torch.exp(t*beta_1-beta_1)
        sigma = lambda t: torch.exp(t*beta_1-beta_1)
        w = lambda t: torch.ones_like(t)
        dwdt = lambda t: torch.zeros_like(t)
        w_cond = w
        dwdt_cond = dwdt
    else:
        raise NotImplementedError('there is no %' % label)
    def q_t(data, t):
        assert (2 == data.dim())
        while (data.dim() > t.dim()): t = t.unsqueeze(-1)
        ydim = config.data.ydim
        if ydim > 0:
            x_0 = data[:,:-ydim]
        else:
            x_0 = data
        return x_0*alpha(t) + sigma(t)*torch.randn_like(x_0)
    def q_t_class(data, t):
        assert (2 == data.dim())
        while (data.dim() > t.dim()): t = t.unsqueeze(-1)
        ydim = config.data.ydim
        x_0, y = data[:,:-ydim], data[:,-ydim:]
        y = torch.repeat_interleave(y, math.ceil(32*32/ydim), 1)[:,:32*32]
        x_1 = 2*y + torch.randn_like(x_0)
        x_t = alpha(t)*x_0 + sigma(t)*x_1
        return x_t
    def q_t_cond(data, t):
        assert (2 == data.dim())
        while (data.dim() > t.dim()): t = t.unsqueeze(-1)
        ydim = config.data.ydim
        x, y = data[:,:-ydim], data[:,-ydim:]
        x_t = data.clone()
        mask = (t >= 0.5).squeeze()
        x_t[mask,:-ydim] = x[mask]*alpha(2*(t[mask]-0.5)) + sigma(2*(t[mask]-0.5))*torch.randn_like(x[mask])
        mask = (t < 0.5).squeeze()
        x_t[mask,-ydim:] = y[mask]*alpha(1-2*t[mask]) + sigma(1-2*t[mask])*torch.randn_like(y[mask])
        return x_t
    if 'conditional' == config.model.task:
        return q_t_cond, alpha, sigma, w_cond, dwdt_cond
    if 'classification' == config.model.task:
        return q_t_class, alpha, sigma, w, dwdt
    return q_t, alpha, sigma, w, dwdt


def get_s(net, config):
    label = config.model.s
    C, H, W = config.data.num_channels, config.data.image_size, config.data.image_size
    ydim = config.data.ydim
    assert ('generic' == label)
    def s1(t,x):
        return net(t,x.view(-1,C,H,W))
    def s1_cond(t,x):
        return net(t,x[:,:-ydim].view(-1,C,H,W),x[:,-ydim:])
    if conditional:
        return s1_cond
    else: 
        return s1
    
    
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

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

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
    x = x.reshape(B*C, H, W)
    x1 = dct(x.reshape(B*C*H, W), norm=norm)
    x1 = x1.reshape(B*C, H, W).transpose(-1, -2)
    x2 = dct(x1.reshape(B*C*H, W), norm=norm)
    return x2.reshape(B, C, H, W).transpose(-1, -2)

def idct_2d(x, norm=None):
    B, C, H, W = x.shape
    x = x.reshape(B*C, H, W)
    x1 = idct(x.reshape(B*C*H, W), norm=norm)
    x1 = x1.reshape(B*C, H, W).transpose(-1, -2)
    x2 = idct(x1.reshape(B*C*H, W), norm=norm)
    return x2.reshape(B, C, H, W).transpose(-1, -2)
