import torch
import math
import numpy as np

from scipy import integrate


def solve_ode_rk(device, s, x, t0=1.0, t1=0.0, atol=1e-5, rtol=1e-5):
    shape = x.shape
    x = x.detach().cpu().numpy().flatten()

    def ode_func(t, x):
        sample = torch.from_numpy(x).reshape(shape).to(device).type(torch.float32)
        t_vec = torch.ones(sample.shape[0], device=sample.device) * t
        sample.requires_grad = True
        dx = torch.autograd.grad(s(t_vec, sample).sum(), sample)[0].detach()
        sample.detach()
        dx = dx.detach().cpu().numpy().flatten()
        return dx
    
    solution = integrate.solve_ivp(ode_func, (t0, t1), x, rtol=rtol, atol=atol, method='RK45')
    return torch.from_numpy(solution['y'][:,-1].reshape(shape)), solution.nfev

def solve_ode(device, s, x, i_inter, ti=1.0, tf=0.0, dt=-1e-3):
    x_inter = []
    t_inter = []
    for i, t in enumerate(np.arange(t0, t1, dt)):
        if i in i_inter:
            x_inter.append(x.clone())
            t_inter.append(t)
        t_vec = (t*torch.ones([x.shape[0],1])).to(device)
        x.requires_grad = True
        x.data += dt * torch.autograd.grad(s(t_vec, x).sum(), x)[0].detach()
        x = x.detach()
    return x, x_inter, t_inter

@torch.no_grad()
def get_likelihood(device, s, x, t0=0.0, t1=1.0, atol=1e-5, rtol=1e-5):
    shape = x.shape
    eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
    x = x.detach().cpu().numpy().flatten()

    def ode_func(t, x):
        sample = torch.from_numpy(x[:-shape[0]]).reshape(shape).to(device).type(torch.float32)
        t_vec = torch.ones(sample.shape[0], device=sample.device) * t
        with torch.enable_grad():
            sample.requires_grad = True
            dx = torch.autograd.grad(s(t_vec, sample).sum(), sample, create_graph=True, retain_graph=True)[0]
            div = (eps*torch.autograd.grad(dx, sample, grad_outputs=eps, retain_graph=True)[0]).sum([1,2,3])
            sample.requires_grad = False
        dx = dx.detach().cpu().numpy().flatten()
        div = div.detach().cpu().numpy().flatten()
        return np.concatenate([dx, div], axis=0)

    init = np.concatenate([x, np.zeros((shape[0],))], axis=0)
    solution = integrate.solve_ivp(ode_func, (t0, t1), init, rtol=rtol, atol=atol, method='RK45')
    
    z = torch.from_numpy(solution['y'][:-shape[0],-1]).reshape(shape).to(device).type(torch.float32)
    delta_logp = torch.from_numpy(solution['y'][-shape[0]:,-1]).to(device).type(torch.float32)
    prior_logp = -0.5*(z**2).sum([1,2,3]) - 0.5*np.prod(shape[1:])*math.log(2*math.pi)
    logp = prior_logp + delta_logp
    return logp, z, solution.nfev

def get_bpd(device, logp, x, lacedaemon):
    shape = x.shape
    D = np.prod(shape[1:])
    bpd = -logp / math.log(2) / D
    bpd = bpd + (torch.log2(torch.exp(-x))-2*torch.log2(1.+torch.exp(-x))).sum([1,2,3])/D
    bpd = bpd - math.log2(1.0-2*lacedaemon) + 8.
    return bpd