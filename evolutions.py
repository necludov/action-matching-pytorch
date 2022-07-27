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
    name = config.model.evolution
    conditional = config.model.conditional
    classification = config.model.classification
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
    def q_t(x_0, t):
        assert (2 == x_0.dim())
        while (x_0.dim() > t.dim()): t = t.unsqueeze(-1)
        ydim = config.data.ydim
        if ydim > 0:
            x_0 = x_0[:,:-ydim]
        eps = torch.empty_like(x_0).normal_()
        return x_0*alpha(t) + sigma(t)*eps
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
    if conditional:
        return q_t_cond, alpha, sigma, w_cond, dwdt_cond
    if classification:
        return q_t_class, alpha, sigma, w, dwdt
    return q_t, alpha, sigma, w, dwdt


# weights for VPSDE when s ~= log p
def w2(t):
    return (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))/beta(t)**2

def dw2dt(t):
    out = torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))/beta(t)**2
    out = out - 2*(1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))/beta(t)**3*(beta_1-beta_0)
    return out

# weights for sub-VPSDE when s ~= s^*
def w3(t):
    return (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))**2

def dw3dt(t):
    out = 2*(1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
    out = out*torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))
    return out

# weights for sub-VPSDE when s ~= log p
def w4(t):
    return 1./beta(t)**2

def dw4dt(t):
    return -2./beta(t)**3*(beta_1-beta_0)


def get_s(net, config):
    label, conditional = config.model.s, config.model.conditional
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

#     def s2(t,x):
#         dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
#         out = -0.5*beta(t).squeeze()*(-net(t,x).squeeze() + 0.5*(x**2).sum(dims_to_reduce))
#         return out
    
#     def s3(t,x):
#         dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
#         out = 0.5*(x**2).sum(dims_to_reduce)
#         out = out - (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))).squeeze()*net(t,x).squeeze()
#         return -0.5*beta(t).squeeze()*out
    
#     if 'generic' == label:
#         return s1
#     elif 'vpsde' == label:
#         return s2
#     elif 'subvpsde' == label:
#         return s3
#     else:
#         raise NotImplementedError('there is no label %' % label)
