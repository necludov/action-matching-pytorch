import torch
import math
import numpy as np

from scipy import integrate

from utils import dotdict
from bpd import discretized_gaussian_log_likelihood


def euler_scheme(ode_func, t0, t1, x, dt):
    solution = dotdict()
    timesteps = np.arange(t0, t1, dt)
    solution.y = np.zeros([len(x), len(timesteps)+1])
    solution.y[:,0] = x
    solution.nfev = 0
    for i, t in enumerate(timesteps):
        dx = ode_func(t, solution.y[:,i])
        solution.y[:,i+1] = solution.y[:,i] + dt*dx
        solution.nfev += 1
    return solution
    
@torch.no_grad()
def solve_ode(device, dxdt, x, t0=1.0, t1=0.0, atol=1e-5, rtol=1e-5, method='RK45', dt=-1e-2):
    shape = x.shape
    def ode_func(t, x_):
        print(f'(solve_ode, method={method}) solving ODE t={t}', flush=True)
        x_ = torch.from_numpy(x_).reshape(shape).to(device).type(torch.float32)
        t_vec = torch.ones(x_.shape[0], device=x_.device) * t
        with torch.enable_grad():
            x_.requires_grad = True
            dx = dxdt(t_vec,x_).detach()
            x_.requires_grad = False
        dx = dx.cpu().numpy().flatten()
        return dx
    
    x = x.detach().cpu().numpy().flatten()
    if 'euler' != method:
        solution = integrate.solve_ivp(ode_func, (t0, t1), x, rtol=rtol, atol=atol, method=method)
    else:
        solution = euler_scheme(ode_func, t0, t1, x, dt)
    return torch.from_numpy(solution.y[:,-1].reshape(shape)), solution.nfev

@torch.no_grad()
def get_likelihood(device, dxdt, x, t0=0.0, t1=1.0, atol=1e-5, rtol=1e-5, method='RK45', dt=1e-2, task='diffusion'):
    assert (2 == x.dim())
    shape = x.shape
    eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
    x = x.detach().cpu().numpy().flatten()

    def ode_func(t, x_):
        print(f'(likelihood, method={method}) solving ODE t={t}', flush=True)
        x_ = torch.from_numpy(x_[:-shape[0]]).reshape(shape).to(device).type(torch.float32)
        t_vec = torch.ones(x_.shape[0], device=x_.device) * t
        with torch.enable_grad():
            x_.requires_grad = True
            dx = dxdt(t_vec,x_)
            div = (eps*torch.autograd.grad(dx, x_, grad_outputs=eps)[0]).sum(1)
            x_.requires_grad = False
        dx = dx.detach().cpu().numpy().flatten()
        div = div.detach().cpu().numpy().flatten()
        return np.concatenate([dx, div], axis=0)

    init = np.concatenate([x, np.zeros((shape[0],))], axis=0)
    if 'euler' != method:
        solution = integrate.solve_ivp(ode_func, (t0, t1), init, rtol=rtol, atol=atol, method=method)
    else:
        solution = euler_scheme(ode_func, t0, t1, init, dt)
    
    z = torch.from_numpy(solution.y[:-shape[0],-1]).reshape(shape).to(device).type(torch.float32)
    delta_logp = torch.from_numpy(solution.y[-shape[0]:,-1]).to(device).type(torch.float32)
    return delta_logp, z, solution.nfev
    
# @torch.no_grad()
# def solve_ode(device, s, x, t0=1.0, t1=0.0, atol=1e-5, rtol=1e-5, method='RK45', dt=-1e-2):
#     shape = x.shape
#     def ode_func(t, x_):
#         x_ = torch.from_numpy(x_).reshape(shape).to(device).type(torch.float32)
#         t_vec = torch.ones(x_.shape[0], device=x_.device) * t
#         with torch.enable_grad():
#             x_.requires_grad = True
#             dx = torch.autograd.grad(s(t_vec, x_).sum(), x_)[0].detach()
#             x_.requires_grad = False
#         dx = dx.cpu().numpy().flatten()
#         return dx
    
#     x = x.detach().cpu().numpy().flatten()
#     if 'euler' != method:
#         solution = integrate.solve_ivp(ode_func, (t0, t1), x, rtol=rtol, atol=atol, method=method)
#     else:
#         solution = euler_scheme(ode_func, t0, t1, x, dt)
#     return torch.from_numpy(solution.y[:,-1].reshape(shape)), solution.nfev

# @torch.no_grad()
# def get_likelihood(device, s, x, t0=0.0, t1=1.0, atol=1e-5, rtol=1e-5, method='RK45', dt=1e-2):
#     assert (2 == x.dim())
#     shape = x.shape
#     eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
#     x = x.detach().cpu().numpy().flatten()

#     def ode_func(t, x_):
#         x_ = torch.from_numpy(x_[:-shape[0]]).reshape(shape).to(device).type(torch.float32)
#         t_vec = torch.ones(x_.shape[0], device=x_.device) * t
#         with torch.enable_grad():
#             x_.requires_grad = True
#             dx = torch.autograd.grad(s(t_vec, x_).sum(), x_, create_graph=True, retain_graph=True)[0]
#             div = (eps*torch.autograd.grad(dx, x_, grad_outputs=eps)[0]).sum(1)
#             x_.requires_grad = False
#         dx = dx.detach().cpu().numpy().flatten()
#         div = div.detach().cpu().numpy().flatten()
#         return np.concatenate([dx, div], axis=0)

#     init = np.concatenate([x, np.zeros((shape[0],))], axis=0)
#     if 'euler' != method:
#         solution = integrate.solve_ivp(ode_func, (t0, t1), init, rtol=rtol, atol=atol, method=method)
#     else:
#         solution = euler_scheme(ode_func, t0, t1, init, dt)
    
#     z = torch.from_numpy(solution.y[:-shape[0],-1]).reshape(shape).to(device).type(torch.float32)
#     delta_logp = torch.from_numpy(solution.y[-shape[0]:,-1]).to(device).type(torch.float32)
#     prior_logp = -0.5*(z**2).sum(1) - 0.5*shape[1]*math.log(2*math.pi)
#     logp = prior_logp + delta_logp
#     return logp, z, solution.nfev

# bpd from this paper https://arxiv.org/pdf/1705.07057.pdf
def get_bpd_(device, logp, x, lacedaemon=5e-2):
    assert (2 == x.dim())
    D = x.shape[1]
    bpd = -logp / math.log(2) / D
    bpd = bpd + (torch.log2(torch.exp(-x))-2*torch.log2(1.+torch.exp(-x))).sum(1)/D
    bpd = bpd - math.log2(1.0-2*lacedaemon) + 8.    
    return bpd

# song's bpd for the data in [0,1] (for data [-1,1] it should be bpd + 7. instead of + 8.)
# this approximates each bin as having a constant density (we now know how to derive this)
def get_bpd_hack(device, logp, x):
    assert (2 == x.dim())
    D = x.shape[1]
    bpd = -logp / math.log(2) / D
    return bpd + 7.

def get_bpd(device, delta_logp, z):
    means = log_scales = torch.zeros_like(z)
    logq1 = discretized_gaussian_log_likelihood(z, means=means, log_scales=log_scales)*math.log2(math.e)
    return -(logq1 + delta_logp)

def disc_gaussian_loglike(z):
    means = log_scales = torch.zeros_like(z)
    return discretized_gaussian_log_likelihood(z, means=means, log_scales=log_scales)

# @torch.no_grad()
# def solve_ode(device, s, x, i_inter=[], t0=1.0, t1=0.0, dt=-1e-2):
#     x_inter = []
#     t_inter = []
#     for i, t in enumerate(np.arange(t0, t1, dt)):
#         if i in i_inter:
#             x_inter.append(x.clone())
#             t_inter.append(t)
#         t_vec = (t*torch.ones([x.shape[0],1])).to(device)
#         with torch.enable_grad():
#             x.requires_grad = True
#             x.data += dt * torch.autograd.grad(s(t_vec, x).sum(), x)[0].detach()
#             x.requires_grad = False
#     return x, x_inter, t_inter
