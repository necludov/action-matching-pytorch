import torch
import math
import numpy as np
import wandb
import os
import shutil
import torch.distributions as D

from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm, trange

from evaluation import *
from evolutions import get_s
from utils import stack_imgs

import scipy.interpolate


# class AdaptiveLoss:
#     def __init__(self, t0=0.0, t1=1.0, n=200, alpha=1e-2, beta=0.99):
#         self.t0, self.t1 = t0, t1
#         self.alpha, self.beta = alpha, beta
#         self.timesteps = np.linspace(t0, t1, n)
#         self.dt = (t1-t0)/(n-1)
#         self._update_w(np.ones_like(self.timesteps))
        
#     def _update_w(self, w):
#         dt, t = self.dt, self.timesteps
#         w = w/(w.sum()*dt - w[0]*dt/2 - w[-1]*dt/2)
#         self.w = w
#         self.fw = scipy.interpolate.interp1d(t, w, kind='nearest')
#         mass = np.concatenate([np.zeros([1]), w[:1]*dt/2, w[0]*dt/2 + w[1:-1].cumsum()*dt, np.ones([1])])
#         grid = np.concatenate([self.t0*np.ones([1]), t[:-1] + dt/2, self.t1*np.ones([1])])
#         self.F_inv = scipy.interpolate.interp1d(mass, grid, kind='linear')
        
#     def sample_t(self, n, device):
#         u = (np.random.uniform() + np.sqrt(2)*np.arange(n)) % 1
#         t_samples = self.F_inv(u)
#         w_t = self.fw(t_samples)
#         t_samples = torch.from_numpy(t_samples).to(device).float()
#         w_t = torch.from_numpy(w_t).to(device).float()
#         return t_samples, w_t
    
#     def update_w(self, new_w, t):
#         new_w, t = new_w.cpu().numpy().flatten(), t.cpu().numpy().flatten()
#         weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e3)
#         weights = weights/weights.sum(1,keepdims=True)
#         w = self.beta*(self.w-self.alpha*(self.t1-self.t0)) + (1-self.beta)*np.sqrt(weights@(new_w**2))
#         w = (1.0-self.alpha)*w/(w.sum()*self.dt)+self.alpha*(self.t1-self.t0)
#         self._update_w(w)
        
#     def get_loss(self, s, x, q_t):
#         t_0, t_1 = self.t0, self.t1
#         device = x.device
#         bs = x.shape[0]
#         t, w_t = self.sample_t(bs, device)
#         while (x.dim() > t.dim()): t = t.unsqueeze(-1)
#         x_t = q_t(x, t)
#         x_t.requires_grad, t.requires_grad = True, True
#         s_t = s(t, x_t)
#         dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
#         x_t, t = x_t.detach(), t.detach()

#         t_0 = t_0*torch.ones(bs).to(device)
#         x_0 = q_t(x, t_0)

#         t_1 = t_1*torch.ones(bs).to(device)
#         x_1 = q_t(x, t_1)

#         dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
#         loss = (0.5*(dsdx**2).sum(dims_to_reduce) + dsdt.sum(dims_to_reduce))/w_t
#         loss = loss*(t_1-t_0).squeeze()
#         loss = loss + (-s(t_1,x_1).squeeze() + s(t_0,x_0).squeeze())
        
#         time_loss = torch.abs(0.5*(dsdx**2).sum(dims_to_reduce)).detach()
#         self.update_w(time_loss, t)
#         return loss.mean()


class AdaptiveLoss:
    def __init__(self, t0=0.0, t1=1.0, n=10, alpha=0.0, beta=0.99):
        self.t0, self.t1 = t0, t1
        self.alpha, self.beta = alpha, beta
        self.timesteps = np.linspace(t0, t1, n)
        self.dt = (t1-t0)/(n-1)
        self._update_w(np.ones_like(self.timesteps))
        
    def _update_w(self, w):
        dt, t = self.dt, self.timesteps
#         w = w/((w[1:]+w[:-1])*dt/2).sum()
        self.w = w
        self.fw = scipy.interpolate.interp1d(t, w, kind='linear')
        self.dwdt = scipy.interpolate.interp1d(t, np.concatenate([w[1:]-w[:-1], w[-1:]-w[-2:-1]])/dt, kind='zero')
        intercept = lambda t: self.fw(t)-self.dwdt(t)*t
        t0_interval = scipy.interpolate.interp1d(t, t, kind='zero')
        mass = np.concatenate([np.zeros([1]), ((w[1:]+w[:-1])*dt/2).cumsum()])
        F0_interval = scipy.interpolate.interp1d(t, mass, kind='zero')
        F0_inv = scipy.interpolate.interp1d(mass, t, kind='zero')
        def F(t):
            t0_ = t0_interval(t)
            F0_ = F0_interval(t)
            k, b = self.dwdt(t), intercept(t)
            output = 0.5*k*(t**2-t0_**2) + b*(t-t0_)
            return F0_ + output 

        def F_inv(y):
            t0_ = F0_inv(y)
            F0_ = F0_interval(t0_)
            k, b = self.dwdt(t0_), intercept(t0_)
            c = y - F0_
            c = c + 0.5*k*t0_**2 + b*t0_
            D = np.sqrt(b**2 + 2*k*c)
            output = (-b + D) * (np.abs(k) > 0)  + c/b * (np.abs(k) == 0.0)
            output[np.abs(k) > 0] /= k[np.abs(k) > 0]
            return output
        
        self.F_inv = F_inv
        
#     def sample_t(self, n, device):
#         u = (np.random.uniform() + np.sqrt(2)*np.arange(n)) % 1
#         t_samples = self.F_inv(u)
#         w_t = self.fw(t_samples)
#         t_samples = torch.from_numpy(t_samples).to(device).float()
#         w_t = torch.from_numpy(w_t).to(device).float()
#         return t_samples, w_t

    def sample_t(self, n, device):
        u = (np.random.uniform() + np.sqrt(2)*np.arange(n)) % 1
        t = u*(self.t1-self.t0) + self.t0
#         t = self.F_inv(u)
        w_t, dwdt = self.fw(t), self.dwdt(t)
        w_0, w_1 = self.fw(self.t0*np.ones_like(t)), self.fw(self.t1*np.ones_like(t))
        t = torch.from_numpy(t).to(device).float()
        w_t, dwdt = torch.from_numpy(w_t).to(device).float(), torch.from_numpy(dwdt).to(device).float()
        w_0, w_1 = torch.from_numpy(w_0).to(device).float(), torch.from_numpy(w_1).to(device).float()
        return t, w_t, dwdt, w_1, w_0
    
    def update_w(self, new_w, t):
        new_w, t = new_w.cpu().numpy().flatten(), t.cpu().numpy().flatten()
        weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e3)
        weights = weights/weights.sum(1,keepdims=True)
        w = self.beta*(self.w-self.alpha*(self.t1-self.t0)) + (1-self.beta)*weights@new_w
#         w = (1.0-self.alpha)*w/(w.sum()*self.dt)+self.alpha*(self.t1-self.t0)
        self._update_w(w)
        
    def get_loss(self, s, x, q_t):
        t_0, t_1 = self.t0, self.t1
        device = x.device
        bs = x.shape[0]
        t, w_t, dwdt, w_1, w_0 = self.sample_t(bs, device)
        while (x.dim() > t.dim()): t = t.unsqueeze(-1)
        x_t = q_t(x, t)
        x_t.requires_grad, t.requires_grad = True, True
        s_t = s(t, x_t)
        dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
        x_t, t = x_t.detach(), t.detach()

        t_0 = t_0*torch.ones(bs).to(device)
        x_0 = q_t(x, t_0)

        t_1 = t_1*torch.ones(bs).to(device)
        x_1 = q_t(x, t_1)

        dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
        loss = (0.5*(dsdx**2).sum(dims_to_reduce) + dsdt.sum(dims_to_reduce))*w_t
        loss = loss.squeeze() + s_t.squeeze()*dwdt
        loss = loss*(t_1-t_0).squeeze()
        loss = loss + (-s(t_1,x_1).squeeze()*w_1 + s(t_0,x_0).squeeze()*w_0)
        
#         time_loss = torch.abs(0.5*(dsdx**2).sum(dims_to_reduce) + dsdt.sum(dims_to_reduce)).detach()
        time_loss = torch.abs(0.5*(dsdx**2).sum(dims_to_reduce)).detach()
        self.update_w(1./(time_loss+1e-2), t)
        return loss.mean()


def loss_AM(s, x, w, dwdt, q_t):
    t_0, t_1 = 0.0, 1.0
    device = x.device
    bs = x.shape[0]
    u = (torch.rand([1,1]) + math.sqrt(2)*torch.arange(bs).view(-1,1)) % 1
    t = u*(t_1-t_0) + t_0
    t = t.to(device)
    while (x.dim() > t.dim()): t = t.unsqueeze(-1)
    x_t = q_t(x, t)
    x_t.requires_grad, t.requires_grad = True, True
    s_t = s(t, x_t)
    dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
    x_t, t = x_t.detach(), t.detach()

    t_0 = t_0*torch.ones(bs).to(device)
    x_0 = q_t(x, t_0)

    t_1 = t_1*torch.ones(bs).to(device)
    x_1 = q_t(x, t_1)
    
    dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
    loss = (0.5*(dsdx**2).sum(dims_to_reduce, keepdim=True) + dsdt.sum(dims_to_reduce, keepdim=True))*w(t)
    loss = loss.squeeze() + s_t.squeeze()*dwdt(t).squeeze()
    loss = loss*(t_1-t_0).squeeze()
    loss = loss + (-s(t_1,x_1).squeeze()*w(t_1).squeeze() + s(t_0,x_0).squeeze()*w(t_0).squeeze())
    return loss.mean()
        
def train(net, train_loader, val_loader, optim, ema, epochs, device, config):
    s, w, dwdt, q_t = get_s(net, config.model.s), config.model.w, config.model.dwdt, config.model.q_t
    loss_AM = AdaptiveLoss()
    step = 0
    for epoch in trange(epochs):
        net.train()
        for x, _ in train_loader:
            x = x.to(device)
#             loss_total = loss_AM(s, x, w, dwdt, q_t)
            loss_total = loss_AM.get_loss(s, x, q_t)
            optim.zero_grad()
            loss_total.backward()

            if config.train.warmup > 0:
                for g in optim.param_groups:
                    g['lr'] = config.train.lr * np.minimum(step / config.train.warmup, 1.0)
            if config.train.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.train.grad_clip)
            optim.step()
            ema.update(net.parameters())
            wandb.log({'train_loss': loss_total}, step=step)
            step += 1
        torch.save({'model': net.state_dict(), 'ema': ema.state_dict(), 'optim': optim.state_dict()}, config.model.savepath)
        
        if ((epoch % config.train.eval_every) == 0) and (epoch >= config.train.first_eval):
            net.eval()
            evaluate(step, epoch, net, s, val_loader, device, config)
        
def evaluate(step, epoch, net, s, val_loader, device, config):
    x_1 = torch.randn(64, config.data.num_channels, config.data.image_size, config.data.image_size).to(device)
    img, nfe_gen = solve_ode_rk(device, s, x_1)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    x_0, y_0 = next(iter(val_loader))
    x_0 = x_0.to(device)[:64]
    logp, z, nfe_ll = get_likelihood(device, s, x_0)
    bpd = get_bpd(device, logp, x_0, lacedaemon=config.data.lacedaemon)
    bpd = bpd.mean().cpu().numpy()

    wandb.log({'epoch': epoch, 
               'RK_function_evals_generation': nfe_gen,
               'RK_function_evals_likelihood': nfe_ll,
               'likelihood(BPD)': bpd,
               'examples': [wandb.Image(stack_imgs(img))]}, step=step)

    
def loss_AM_conditional(s, x, y, w, dwdt, q_t):
    t_0, t_1 = 0.0, 1.0
    device = x.device
    bs = x.shape[0]
    u = (torch.rand([1,1]) + math.sqrt(2)*torch.arange(bs).view(-1,1)) % 1
    u = torch.rand([bs,1])
    t = u*(t_1-t_0) + t_0
    t = t.to(device)
    while (x.dim() > t.dim()): t = t.unsqueeze(-1)
    x_t, y_t = q_t(x, y, t)
    x_t.requires_grad, y_t.requires_grad, t.requires_grad = True, True, True
    s_t = s(t, x_t, y_t)
    dsdt, dsdx, dsdy = torch.autograd.grad(s_t.sum(), [t, x_t, y_t], create_graph=True, retain_graph=True)
    x_t, y_t, t = x_t.detach(), y_t.detach(), t.detach()

    t_0 = t_0*torch.ones(bs).to(device)
    x_0, y_0 = q_t(x, y, t_0)

    t_1 = t_1*torch.ones(bs).to(device)
    x_1, y_1 = q_t(x, y, t_1)
    
    dims_to_reduce = [i + 1 for i in range(x.dim()-1)]
    loss = 0.5*(dsdx**2).sum(dims_to_reduce, keepdim=True)
    loss = loss + dsdt.sum(dims_to_reduce, keepdim=True)
    dims_to_reduce = [i + 1 for i in range(y.dim()-1)]
    loss = loss + 0.5*(dsdy**2).sum(dims_to_reduce, keepdim=True)
    loss = loss*w(t)
    loss = loss.squeeze() + s_t.squeeze()*dwdt(t).squeeze()
    loss = loss*(t_1-t_0).squeeze()
    loss = loss + (-s(t_1,x_1,y_1).squeeze()*w(t_1).squeeze() + s(t_0,x_0,y_0).squeeze()*w(t_0).squeeze())
    return loss.mean()
