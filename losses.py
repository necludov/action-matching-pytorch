import torch
import math
import numpy as np
import os
import shutil
import scipy.interpolate

from evolutions import get_q
from utils import DDPAverageMeter, gather, get_rank, get_world_size


def get_loss(net, config):
    if 'am' == config.model.objective:
        return AdaptiveLoss(net, config)
    elif 'sm' == config.model.objective:
        return ScoreLoss(net, config)
    else:
        raise NameError('config.model.objective name is incorrect')
        
def get_s(net, config):
    label = config.model.s
    C, H, W = config.data.num_channels, config.data.image_size, config.data.image_size
    C_cond, ydim = config.model.cond_channels, config.data.ydim
    if C_cond > 0:
        def s(t,x):
            x = x.view(-1, C+C_cond, H, W)
            return net(t, x[:,:C,:,:], x[:,C:,:,:])
    else:
        def s(t,x):
            return net(t, x.view(-1,C,H,W))
    return s

class AdaptiveLoss:
    def __init__(self, net, config, n=100, beta=0.99):
        self.t0, self.t1 = config.model.t0, config.model.t1
        self.alpha, self.beta = config.train.alpha, beta
        self.timesteps = np.linspace(self.t0, self.t1, n)
        self.dt = (self.t1-self.t0)/(n-1)
        self.rank = get_rank()
        self.ws = get_world_size()
        
        self.q_t, self.w, self.dwdt = get_q(config)
        self.boundary_conditions = (self.w(torch.tensor(self.t0)).item() != 0.0,
                                    self.w(torch.tensor(self.t1)).item() != 0.0)
        print('boundary conditions are: ', self.boundary_conditions)
        config.train.boundary_conditions = self.boundary_conditions
        
        self.s = get_s(net, config)
        
        self.buffer = {'values': [],
                       'times': [],
                       'size': 100,
                       'mean': np.zeros_like(self.timesteps),
                       'var': np.ones_like(self.timesteps),
                       'p': np.ones_like(self.timesteps),
                       'u0': 0.5}
        self.construct_dist()
        
        meters = [DDPAverageMeter('train_loss'),
                  DDPAverageMeter('dsdx_std'),
                  DDPAverageMeter('dsdt_std'),
                  DDPAverageMeter('s_1_std'),
                  DDPAverageMeter('s_0_std'),
                  DDPAverageMeter('s_std')]
        self.meters = dict((m.name,m) for m in meters)
        
    def load_state_dict(self, buffer_dict):
        self.buffer = buffer_dict
        self.construct_dist()
        
    def state_dict(self):
        return self.buffer
        
    def construct_dist(self):
        dt, t = self.dt, self.timesteps
        p = self.buffer['p']
        self.fp = scipy.interpolate.interp1d(t, p, kind='linear')
        self.dpdt = scipy.interpolate.interp1d(t, np.concatenate([p[1:]-p[:-1], p[-1:]-p[-2:-1]])/dt, kind='zero')
        intercept = lambda t: self.fp(t)-self.dpdt(t)*t
        t0_interval = scipy.interpolate.interp1d(t, t, kind='zero')
        mass = np.concatenate([np.zeros([1]), ((p[1:]+p[:-1])*dt/2).cumsum()[:-1], np.ones([1])])
        F0_interval = scipy.interpolate.interp1d(t, mass, kind='zero')
        F0_inv = scipy.interpolate.interp1d(mass, t, kind='zero')
        def F(t):
            t0_ = t0_interval(t)
            F0_ = F0_interval(t)
            k, b = self.dpdt(t), intercept(t)
            output = 0.5*k*(t**2-t0_**2) + b*(t-t0_)
            return F0_ + output 

        def F_inv(y):
            t0_ = F0_inv(y)
            F0_ = F0_interval(t0_)
            k, b = self.dpdt(t0_), intercept(t0_)
            c = y - F0_
            c = c + 0.5*k*t0_**2 + b*t0_
            D = np.sqrt(b**2 + 2*k*c)
            output = (-b + D) * (np.abs(k) > 0)  + c/b * (np.abs(k) == 0.0)
            output[np.abs(k) > 0] /= k[np.abs(k) > 0]
            return output
        
        self.F_inv = F_inv
        
    def sample_t(self, n, device):
        u = (self.buffer['u0'] + np.sqrt(2)*np.arange(n*self.ws)) % 1
        self.buffer['u0'] = (self.buffer['u0'] + np.sqrt(2)*n*self.ws) % 1
        u = u[self.rank*n:(self.rank+1)*n]
        t = self.F_inv(u)
        assert ((t < 0.0).sum() == 0) and ((t > 1.0).sum() == 0)
        p_t, dpdt = self.fp(t), self.dpdt(t)
        p_0, p_1 = self.fp(self.t0*np.ones_like(t)), self.fp(self.t1*np.ones_like(t))
        t = torch.from_numpy(t).to(device).float()
        p_t, dpdt = torch.from_numpy(p_t).to(device).float(), torch.from_numpy(dpdt).to(device).float()
        p_0, p_1 = torch.from_numpy(p_0).to(device).float(), torch.from_numpy(p_1).to(device).float()
        return t, p_t, dpdt
    
#     def update_history(self, new_p, t, p_t):
#         new_p, t = new_p.cpu().numpy().flatten(), t.cpu().numpy().flatten()
#         weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#         weights = weights/weights.sum(1,keepdims=True)
#         if self.mean is None:
#             self.mean = weights@new_p
#         else:
#             self.mean = self.beta*self.mean + (1-self.beta)*(weights@new_p)
#         mean_func = scipy.interpolate.interp1d(self.timesteps, self.mean, kind='linear')

#         self.buffer_values.append(new_p)
#         self.buffer_times.append(t)
#         if len(self.buffer_values) > 100:
#             self.buffer_values.pop(0)
#             self.buffer_times.pop(0)
#         var = np.zeros_like(self.timesteps)
#         for i in range(len(self.buffer_values)):
#             t = self.buffer_times[i]
#             weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#             weights = weights/weights.sum(1,keepdims=True)
#             var += weights@((mean_func(t) - self.buffer_values[i])**2)
#         var /= len(self.buffer_values)
#         if len(self.buffer_values) < 100:
#             self.construct_dist(np.ones_like(self.timesteps))
#         else:
#             self.construct_dist(np.sqrt(var))

#     def update_history(self, new_p, t, p_t):
#         new_p, t, p_t = new_p.cpu().numpy().flatten(), t.cpu().numpy().flatten(), p_t.cpu().numpy().flatten()
#         weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#         weights = weights/weights.sum(1,keepdims=True)
#         self.buffer['mean'] += weights@(new_p)/self.buffer['size']

#         self.buffer['values'].append(new_p)
#         self.buffer['times'].append(t)
#         if len(self.buffer['values']) > self.buffer['size']:
#             p = self.buffer['values'].pop(0)
#             t = self.buffer['times'].pop(0)
#             weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#             weights = weights/weights.sum(1,keepdims=True)
#             self.buffer['mean'] -= weights@(p)/self.buffer['size']
#             assert len(self.buffer['values']) == self.buffer['size']
            
#         mean_func = scipy.interpolate.interp1d(self.timesteps, self.buffer['mean'], kind='linear')
#         var = np.zeros_like(self.timesteps)
#         for i in range(len(self.buffer['values'])):
#             p = self.buffer['values'][i]
#             t = self.buffer['times'][i]
#             weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#             weights = weights/weights.sum(1,keepdims=True)
#             var += weights@((mean_func(t) - p)**2)/self.buffer['size']
#         if len(self.buffer['values']) == self.buffer['size']:
#             p = np.sqrt(var)
#             p = (1.0-self.alpha)*p/((p[1:]+p[:-1])*self.dt/2).sum() + self.alpha/(self.t1-self.t0)    
#             self.buffer['p'] = p
#         self.construct_dist()

    def update_history(self, new_p, t, p_t):
        new_p, t, p_t = new_p.cpu().numpy().flatten(), t.cpu().numpy().flatten(), p_t.cpu().numpy().flatten()
        weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
        weights = weights/weights.sum(1,keepdims=True)
        self.buffer['mean'] = self.beta*self.buffer['mean'] + (1-self.beta)*(weights@new_p)
        mean_func = scipy.interpolate.interp1d(self.timesteps, self.buffer['mean'], kind='linear')
        self.buffer['var'] = self.beta*self.buffer['var'] + (1-self.beta)*(weights@((mean_func(t) - new_p)**2))
        p = np.sqrt(self.buffer['var'])
        p = (1.0-self.alpha)*p/((p[1:]+p[:-1])*self.dt/2).sum() + self.alpha/(self.t1-self.t0)
        self.buffer['p'] = p
        self.construct_dist()
    
    def eval_loss(self, x):
        q_t, w, dwdt, s = self.q_t, self.w, self.dwdt, self.s
        assert (2 == x.dim())
        t_0, t_1 = self.t0, self.t1
        device = x.device
        bs = x.shape[0]
        t, p_t, dpdt = self.sample_t(bs, device)
        while (x.dim() > t.dim()): t = t.unsqueeze(-1)
        x_t, _ = q_t(x, t)
        x_t.requires_grad, t.requires_grad = True, True
        s_t = s(t, x_t)
        assert (2 == s_t.dim())
        dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
        x_t.requires_grad, t.requires_grad = False, False
        
        loss = (0.5*(dsdx**2).sum(1, keepdim=True) + dsdt.sum(1, keepdim=True))*w(t)
        self.meters['dsdx_std'].update((0.5*(dsdx**2).sum(1)*w(t).squeeze()).detach().cpu().std())
        self.meters['dsdt_std'].update((dsdt.sum(1)*w(t).squeeze()).detach().cpu().std())
        loss = loss + s_t*dwdt(t)
        self.meters['s_std'].update((s_t*dwdt(t)).squeeze().detach().cpu().std())
        loss = loss.squeeze()/p_t
        time_loss = loss.detach()*p_t
            
        s_1_std, s_0_std = 0.0, 0.0
        if self.boundary_conditions[0]:
            t_0 = t_0*torch.ones([bs, 1], device=device)
            x_0, _ = q_t(x, t_0)
            left_bound = (s(t_0,x_0)*w(t_0)).squeeze()
            loss = loss + left_bound
            self.meters['s_0_std'].update(left_bound.detach().cpu().std())
        if self.boundary_conditions[1]:
            t_1 = t_1*torch.ones([bs, 1], device=device)
            x_1, _ = q_t(x, t_1)
            right_bound = (-s(t_1,x_1)*w(t_1)).squeeze()
            loss = loss + right_bound
            self.meters['s_1_std'].update(right_bound.detach().cpu().std())
            
        self.meters['train_loss'].update(loss.detach().mean().cpu())
        self.update_history(gather(time_loss), gather(t), gather(p_t))
        return loss.mean(), self.meters
    
    def get_dxdt(self):
        def dxdt(t, x):
            return torch.autograd.grad(self.s(t, x).sum(), x, create_graph=True, retain_graph=True)[0]
        return dxdt

    
class ScoreLoss:
    def __init__(self, net, config):
        self.q_t, self.beta, self.sigma = get_q(config)
        self.net = net
        self.C, self.W, self.H = config.data.num_channels, config.data.image_size, config.data.image_size
        self.C_cond = config.model.cond_channels
        meters = [DDPAverageMeter('train_loss')]
        self.meters = dict((m.name,m) for m in meters)
        
    def load_state_dict(self, buffer_dict):
        return
        
    def state_dict(self):
        return {}
        
    def eval_loss(self, x):
        device = x.device
        bs = x.shape[0]

        t = torch.rand([bs], device=device)
        x_t, eps = self.q_t(x, t)
        loss_sm = ((eps - self.net(t, x_t)) ** 2).sum(dim=(1, 2, 3))
        self.meters['train_loss'].update(loss_sm.detach().mean().cpu())
        return loss_sm.mean(), self.meters
    
    def get_dxdt(self):
        C, H, W, C_cond = self.C, self.W, self.H, self.C_cond
        def score(t, x):
            return -self.net(t, x) / self.sigma(t)
        f = lambda t, x: -0.5*self.beta(t)*x
        g = lambda t, x: torch.sqrt(self.beta(t))
        def dxdt(t, x):
            original_shape = x.shape
            x = x.view(-1,C + C_cond,H,W)
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            out = torch.zeros_like(x)
            out[:,:C] = f(t,x[:,:C]) - 0.5*g(t,x[:,:C])**2*score(t,x)
            return out.reshape(-1, (C + C_cond)*H*W)
        return dxdt
