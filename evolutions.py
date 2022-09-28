import math
import torch

beta_0 = 0.1
beta_1 = 20.0

beta = lambda t: (1-t)*beta_0 + t*beta_1

def w0(t):
    return torch.ones_like(t)

def dw0dt(t):
    return torch.zeros_like(t)

def w_variance(t):
    return torch.pow(-torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)), 2.5)

def dw_variancedt(t):
    out = 2.5*torch.pow(-torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)), 1.5)
    out = out*torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))
    return out

def w_volume(t):
    return torch.pow(-torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)), 3.0)

def dw_volumedt(t):
    out = 3.0*torch.pow(-torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)), 2.0)
    out = out*torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))
    return out

def w2(t):
    return torch.pow(t, 2.5)*(1-t)

def dw2dt(t):
    return 2.5*torch.pow(t, 1.5) - 3.5*torch.pow(t, 2.5)

def w3(t):
    return t**3

def dw3dt(t):
    return 3*t**2


def get_q(config):
    if 'am' == config.model.objective:
        return get_q_am(config)
    elif 'sm' == config.model.objective:
        return get_q_sm(config)
    else:
        raise NameError('config.model.objective name is %s, which is undefined' % config.model.objective)

def get_q_sm(config):
    if 'vpsde' == config.model.sigma:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: torch.sqrt(-torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
    elif 'subvpsde' == config.model.sigma:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: -torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))
    else:
        raise NotImplementedError('config.model.sigma is %s, which is undefined' % config.model.sigma)
    if 'diffusion' == config.model.task:
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size    
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            eps = torch.randn_like(x)
            output = x*alpha(t) + sigma(t)*eps
            return output.reshape([B, C*H*W]), eps
        return q_t, beta, sigma
    elif 'color' == config.model.task:
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            gray_x = x.mean(1,keepdim=True).repeat([1,C,1,1])
            eps = torch.randn_like(x)
            output = x*alpha(t) + sigma(t)*eps
            output = torch.hstack([output, gray_x])
            return output.reshape([B, 2*C*H*W]), eps
        return q_t, beta, sigma
    elif 'superres' == config.model.task:
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            downscale_x = torch.nn.functional.interpolate(x, size=(H//2,W//2), mode='nearest')
            downscale_x = torch.nn.functional.interpolate(downscale_x, size=(H,W), mode='bilinear')
            eps = torch.randn_like(x)
            output = x*alpha(t) + sigma(t)*eps
            output = torch.hstack([output, downscale_x])
            return output.reshape([B, 2*C*H*W]), eps
        return q_t, beta, sigma
    elif 'inpaint' == config.model.task:
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            mask = torch.zeros_like(x)
            u = (torch.rand((B,1,1,1)) < 0.5).float()
            mask[:,:,:H//2,:], mask[:,:,H//2:,:] = u, 1-u
            eps = torch.randn_like(x)
            output = x*alpha(t) + sigma(t)*eps
            output = torch.hstack([output, x*mask])
            return output.reshape([B, 2*C*H*W]), eps
        return q_t, beta, sigma
    else:
        raise NameError('config.model.task is %s, which is undefined' % config.model.task)

def get_q_am(config):
    if 'diffusion' == config.model.task:
        return get_q_diffusion(config)
    elif 'heat' == config.model.task:
        sigma = lambda t: t
        w = lambda t: 0.5*t**2
        dwdt = lambda t: t
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            blurred_img = blur(x.reshape([B, C, H, W]),t)
            blurred_img = blurred_img.reshape([B, C*H*W])
            return blurred_img + sigma(t.view([B, 1]))*torch.randn_like(blurred_img), None
        return q_t, w, dwdt
    elif 'color' == config.model.task:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = w3
        dwdt = dw3dt
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            gray_x = x.mean(1,keepdim=True).repeat([1,C,1,1])
            eps = torch.randn_like(x)
            output = alpha(t)*x + sigma(t)*(eps + 1e-1*gray_x)
            output = torch.hstack([output, gray_x])
            return output.reshape([B, 2*C*H*W]), None
        return q_t, w, dwdt
    elif 'superres_additive' == config.model.task:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = w3
        dwdt = dw3dt
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            downscale_x = torch.nn.functional.interpolate(x, size=(H//2,W//2), mode='nearest')
            downscale_x = torch.nn.functional.interpolate(downscale_x, size=(H,W), mode='bilinear')
            eps = torch.randn_like(x)
            output = sigma(t)*(downscale_x + 1e-1*eps) + alpha(t)*x
            output = torch.hstack([output, downscale_x])
            return output.reshape([B, 2*C*H*W]), None
        return q_t, w, dwdt
    elif 'torus' == config.model.task:
        sigma = lambda t: t
        w = w3
        dwdt = dw3dt
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            # x to [0.25,0.75]
            x = x*torch.tensor(config.data.norm_std, device=x.device).view(1,C,1,1)
            x = x + torch.tensor(config.data.norm_mean, device=x.device).view(1,C,1,1)
            x = 0.5*x + 0.25
            # add noise
            eps = 5e-1*torch.randn_like(x)
            output = torch.remainder(x + sigma(t)*eps, 1.0)
            return output.reshape([B, C*H*W]), None
        return q_t, w, dwdt
    elif 'inpaint' == config.model.task:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = w3
        dwdt = dw3dt
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            mask = torch.zeros_like(x)
            u = (torch.rand((B,1,1,1)) < 0.5).float()
            mask[:,:,:H//2,:], mask[:,:,H//2:,:] = u, 1-u
            eps = torch.randn_like(x)
            output = mask*x + (1-mask)*(sigma(t)*eps + alpha(t)*x)
            output = torch.hstack([output])
            return output.reshape([B, C*H*W]), None
        return q_t, w, dwdt
    elif 'superres' == config.model.task:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = w3
        dwdt = dw3dt
        def q_t(x, t):
            assert (2 == x.dim())
            B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size
            x = x.reshape([B, C, H, W])
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            downscaled_x = torch.nn.functional.interpolate(x, size=(H//2,W//2), mode='nearest')
            interleaved_x = torch.nn.functional.interpolate(downscaled_x, size=(H,W), mode='nearest')
            interleaved_x[:,:,1::2,:] = torch.randn_like(interleaved_x[:,:,1::2,:])
            interleaved_x[:,:,:,1::2] = torch.randn_like(interleaved_x[:,:,:,1::2])
            output = sigma(t)*interleaved_x + alpha(t)*x
            downscaled_x = torch.nn.functional.interpolate(downscaled_x, size=(H,W), mode='bilinear')
            output = torch.hstack([output, downscaled_x])
            return output.reshape([B, 2*C*H*W]), None
        return q_t, w, dwdt
    else:
        raise NameError('config.model.task is %s, which is undefined' % config.model.task)


def get_q_diffusion(config):
    name = config.model.sigma
    if 'variance' == name:
        alpha = lambda t: torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
        sigma = lambda t: torch.sqrt(-torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
        w = w_variance
        dwdt = dw_variancedt
    elif 'volume' == name:
        alpha = lambda t: torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))
        sigma = lambda t: -torch.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))
        w = w_volume
        dwdt = dw_volumedt
    elif 'simple' == name:
        alpha = lambda t: torch.sqrt(1-t)
        sigma = lambda t: torch.sqrt(t)
        w = w2
        dwdt = dw2dt
    elif 'dimple' == name:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = w3
        dwdt = dw3dt
    elif 'dimple0' == name:
        alpha = lambda t: 1-t
        sigma = lambda t: t
        w = w0
        dwdt = dw0dt
    else:
        raise NotImplementedError('there is no %' % label)
    def q_t(x, t):
        assert (2 == x.dim())
        B, C, H, W = x.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size    
        while (x.dim() > t.dim()): t = t.unsqueeze(-1)
        eps = torch.randn_like(x)
        output = x*alpha(t) + sigma(t)*eps
        output = output.reshape([B, C*H*W])
        return output, None
    return q_t, w, dwdt

def remove_labels(data, t, ydim=0):
    assert (2 == data.dim())
    while (data.dim() > t.dim()): t = t.unsqueeze(-1)
    if ydim > 0:
        x = data[:,:-ydim]
    else:
        x = data
    return x, t
    
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
    freqs = math.pi * torch.linspace(0, H-1, H, device=device)/H
    frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
    frequencies_squared = frequencies_squared.unsqueeze(0).unsqueeze(0)
    while (x.dim() > t.dim()): t = t.unsqueeze(-1)
    x_proj = dct_2d(x, norm='ortho')
    t_prime = 0.5*torch.exp(2*(t*math.log(sigma_max) + (1-t)*math.log(sigma_min)))
    x_proj = torch.exp(-frequencies_squared*t_prime)*x_proj
    x_reconstructed = idct_2d(x_proj, norm='ortho')
    return x_reconstructed

######################################################################################
