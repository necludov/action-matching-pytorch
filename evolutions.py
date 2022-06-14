import math
import torch

beta_0 = 0.1
beta_1 = 20.0

beta = lambda t: (1-t)*beta_0 + t*beta_1

def vpsde(x_0, t):
    while (x_0.dim() > t.dim()): t = t.unsqueeze(-1)
    mean = x_0*torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
    sigma = torch.sqrt(1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
    x_t = mean + sigma*torch.empty_like(x_0).normal_()
    return x_t

# weights for VPSDE when s ~= s^*
def w1(t):
    return (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))

def dw1dt(t):
    out = torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))
    return out

# weights for VPSDE when s ~= log p
def w2(t):
    return (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))/beta(t)**2

def dw2dt(t):
    out = torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))*(beta_0+t*(beta_1-beta_0))/beta(t)**2
    out = out - 2*(1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0)))/beta(t)**3*(beta_1-beta_0)
    return out


def subvpsde(x_0, t):
    while (x_0.dim() > t.dim()): t = t.unsqueeze(-1) 
    mean = x_0*torch.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
    sigma = 1-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))
    x_t = mean + sigma*torch.empty_like(x_0).normal_()
    return x_t

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


def get_s(net, label):
    def s1(t,x):
        return net(t,x)

    def s2(t,x):
        dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
        out = -0.5*beta(t).squeeze()*(-net(t,x).squeeze() + 0.5*(x**2).sum(dims_to_reduce))
        return out
    
    def s3(t,x):
        dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
        out = 0.5*(x**2).sum(dims_to_reduce)
        out = out - (1.0-torch.exp(-t*beta_0-0.5*t**2*(beta_1-beta_0))).squeeze()*net(t,x).squeeze()
        return -0.5*beta(t).squeeze()*out
    
    if 'generic' == label:
        return s1
    elif 'vpsde' == label:
        return s2
    elif 'subvpsde' == label:
        return s3
    else:
        raise NotImplementedError('there is no label %' % label)
