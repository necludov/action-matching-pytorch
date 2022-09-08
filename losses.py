import torch
import math
import numpy as np
import os
import shutil
import scipy.interpolate

from evolutions import get_q


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
    def __init__(self, net, config, n=50, beta=0.99):
        self.t0, self.t1 = config.model.t0, config.model.t1
        self.alpha, self.beta = config.train.alpha, beta
        self.timesteps = np.linspace(self.t0, self.t1, n)
        self.dt = (self.t1-self.t0)/(n-1)
        
        self.q_t, self.w, self.dwdt = get_q(config)
        self.boundary_conditions = (self.w(torch.tensor(self.t0)).item() != 0.0,
                                    self.w(torch.tensor(self.t1)).item() != 0.0)
        print('boundary conditions are: ', self.boundary_conditions)
        config.train.boundary_conditions = self.boundary_conditions
        
        self.s = get_s(net, config)
        
        self.mean = np.zeros_like(self.timesteps)
        self.use_var = config.train.use_var
        self.buffer_values = []
        self.buffer_times = []
        self.buffer_pt = []
        self.buffer_size = 100
        self.construct_dist(np.ones_like(self.timesteps))
        
    def construct_dist(self, p):
        dt, t = self.dt, self.timesteps
        p = (1.0-self.alpha)*p/((p[1:]+p[:-1])*dt/2).sum() + self.alpha/(self.t1-self.t0)
        self.p = p
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
        u = (np.random.uniform() + np.sqrt(2)*np.arange(n)) % 1
        t = self.F_inv(u)
        p_t, dpdt = self.fp(t), self.dpdt(t)
        p_0, p_1 = self.fp(self.t0*np.ones_like(t)), self.fp(self.t1*np.ones_like(t))
        t = torch.from_numpy(t).to(device).float()
        p_t, dpdt = torch.from_numpy(p_t).to(device).float(), torch.from_numpy(dpdt).to(device).float()
        p_0, p_1 = torch.from_numpy(p_0).to(device).float(), torch.from_numpy(p_1).to(device).float()
        return t, p_t
    
#     def update_history(self, new_p, t):
#         new_p, t = new_p.cpu().numpy().flatten(), t.cpu().numpy().flatten()
#         weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#         weights = weights/weights.sum(1,keepdims=True)
#         if self.mean is None:
#             self.mean = weights@new_p
#         else:
#             self.mean = self.beta*self.mean + (1-self.beta)*(weights@new_p)
#         mean_func = scipy.interpolate.interp1d(self.timesteps, self.mean, kind='linear')

#         if self.use_var:
#             self.buffer_values.append(new_p)
#             self.buffer_times.append(t)
#             if len(self.buffer_values) > 100:
#                 self.buffer_values.pop(0)
#                 self.buffer_times.pop(0)
#             var = np.zeros_like(self.timesteps)
#             for i in range(len(self.buffer_values)):
#                 t = self.buffer_times[i]
#                 weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
#                 weights = weights/weights.sum(1,keepdims=True)
#                 var += weights@((mean_func(t) - self.buffer_values[i])**2)
#             var /= len(self.buffer_values)
#             if len(self.buffer_values) < 100:
#                 self.construct_dist(np.ones_like(self.timesteps))
#             else:
#                 self.construct_dist(np.sqrt(var))
#         else:
#             self.construct_dist(self.mean)

    def update_history(self, new_p, t, p_t):
        new_p, t, p_t = new_p.cpu().numpy().flatten(), t.cpu().numpy().flatten(), p_t.cpu().numpy().flatten()
        weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
        weights = weights/weights.sum(0,keepdims=True)/weights.shape[1]
        self.mean += weights@(new_p/p_t)/self.buffer_size

        self.buffer_values.append(new_p)
        self.buffer_times.append(t)
        self.buffer_pt.append(p_t)
        if len(self.buffer_values) > self.buffer_size:
            p = self.buffer_values.pop(0)
            t = self.buffer_times.pop(0)
            p_t = self.buffer_pt.pop(0)
            weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
            weights = weights/weights.sum(0,keepdims=True)/weights.shape[1]
            self.mean -= weights@(p/p_t)/self.buffer_size
            
        mean_func = scipy.interpolate.interp1d(self.timesteps, self.mean, kind='linear')
        var = np.zeros_like(self.timesteps)
        for i in range(len(self.buffer_values)):
            p = self.buffer_values[i]
            t = self.buffer_times[i]
            p_t = self.buffer_pt[i]
            weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
            weights = weights/weights.sum(0,keepdims=True)/weights.shape[1]
            var += weights@((mean_func(t) - p)**2/p_t)/self.buffer_size
        if len(self.buffer_values) < self.buffer_size:
            self.construct_dist(np.ones_like(self.timesteps))
        else:
            self.construct_dist(np.sqrt(var))
    
    def eval_loss(self, x):
        q_t, w, dwdt, s = self.q_t, self.w, self.dwdt, self.s
        assert (2 == x.dim())
        t_0, t_1 = self.t0, self.t1
        device = x.device
        bs = x.shape[0]
        t, p_t = self.sample_t(bs, device)
        while (x.dim() > t.dim()): t = t.unsqueeze(-1)
        x_t, _ = q_t(x, t)
        x_t.requires_grad, t.requires_grad = True, True
        s_t = s(t, x_t)
        assert (2 == s_t.dim())
        dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
        x_t, t = x_t.detach(), t.detach()
        
        loss = (0.5*(dsdx**2).sum(1, keepdim=True) + dsdt.sum(1, keepdim=True))*w(t)
        dsdx_std = 0.5*(dsdx**2).sum(1).detach().cpu().std()
        dsdt_std = dsdt.sum(1).detach().cpu().std()
        loss = loss + s_t*dwdt(t)
        s_std = s_t.sum(1).detach().cpu().std()
        loss = loss.squeeze()/p_t
        time_loss = loss.detach()*p_t
            
        s_1_std, s_0_std = 0.0, 0.0
        if self.boundary_conditions[0]:
            t_0 = t_0*torch.ones([bs, 1])
            x_0, _ = q_t(x, t_0)
            loss = loss + (s(t_0,x_0)*w(t_0)).squeeze()
            s_0_std = s(t_0,x_0).sum(1).detach().cpu().std()
#             time_loss += (s(t_0,x_0)*w(t_0)).squeeze().detach().mean()
        if self.boundary_conditions[1]:
            t_1 = t_1*torch.ones([bs, 1], device=device)
            x_1, _ = q_t(x, t_1)
            loss = loss + (-s(t_1,x_1)*w(t_1)).squeeze()
            s_1_std = s(t_1,x_1).sum(1).detach().cpu().std()
#             time_loss += (-s(t_1,x_1)*w(t_1)).squeeze().detach().mean()
        
        self.update_history(time_loss, t, p_t)
        meters = {'train_loss': loss.detach().mean(),
                  'dsdx_std': dsdx_std,
                  'dsdt_std': dsdt_std,
                  's_std': s_std,
                  's_1_std': s_1_std,
                  's_0_std': s_0_std}
        return loss.mean(), meters
    
    def get_dxdt(self):
        def dxdt(t, x):
            return torch.autograd.grad(self.s(t, x).sum(), x, create_graph=True, retain_graph=True)[0]
        return dxdt

    
class ScoreLoss:
    def __init__(self, net, config):
        self.q_t, self.beta, self.sigma = get_q(config)
        self.net = net
        self.C, self.W, self.H = config.data.num_channels, config.data.image_size, config.data.image_size
        
    def eval_loss(self, x):
        device = x.device
        bs = x.shape[0]

        t = torch.rand([bs], device=device)
        x_t, eps = self.q_t(x, t)
        loss_sm = ((eps - self.net(t, x_t)) ** 2).sum(dim=(1, 2, 3))    
        loss_sm = loss_sm.mean()
        meters = {'train_loss': loss_sm.detach()}
        return loss_sm, meters
    
    def get_dxdt(self):
        C, H, W = self.C, self.W, self.H
        def score(t, x):
            return -self.net(t, x) / self.sigma(t)
        f = lambda t, x: -0.5*self.beta(t)*x
        g = lambda t, x: torch.sqrt(self.beta(t))
        def dxdt(t, x):
            x = x.view(-1,C,H,W)
            while (x.dim() > t.dim()): t = t.unsqueeze(-1)
            return f(t,x) - 0.5*g(t,x)**2*score(t,x)
        return dxdt
