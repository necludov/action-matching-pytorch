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
from evolutions import get_s, get_q
from utils import stack_imgs

import scipy.interpolate


class AdaptiveLoss:
    def __init__(self, t0=0.0, t1=1.0, n=50, alpha=1e-2, beta=0.99):
        self.t0, self.t1 = t0, t1
        self.alpha, self.beta = alpha, beta
        self.timesteps = np.linspace(t0, t1, n)
        self.dt = (t1-t0)/(n-1)
        self.mean = None
        self.construct_dist(np.ones_like(self.timesteps))
        
    def construct_dist(self, p):
        dt, t = self.dt, self.timesteps
        p = (1.0-self.alpha)*p/((p[1:]+p[:-1])*dt/2).sum() + self.alpha/(self.t1-self.t0)
        self.p = p
        self.fp = scipy.interpolate.interp1d(t, p, kind='linear')
        self.dpdt = scipy.interpolate.interp1d(t, np.concatenate([p[1:]-p[:-1], p[-1:]-p[-2:-1]])/dt, kind='zero')
        intercept = lambda t: self.fp(t)-self.dpdt(t)*t
        t0_interval = scipy.interpolate.interp1d(t, t, kind='zero')
        mass = np.concatenate([np.zeros([1]), ((p[1:]+p[:-1])*dt/2).cumsum()])
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
    
    def update_history(self, new_w, t):
        new_w, t = new_w.cpu().numpy().flatten(), t.cpu().numpy().flatten()
        weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
        weights = weights/weights.sum(1,keepdims=True)
        if self.mean is None:
            self.mean = weights@new_w
        else:
            self.mean = self.beta*self.mean + (1-self.beta)*(weights@new_w)
        self.construct_dist(self.mean)
    
    def get_loss(self, s, x, q_t, omega, dodt):
        assert (2 == x.dim())
        t_0, t_1 = self.t0, self.t1
        device = x.device
        bs = x.shape[0]
        t, p_t = self.sample_t(bs, device)
        while (x.dim() > t.dim()): t = t.unsqueeze(-1)
        x_t = q_t(x, t)
        x_t.requires_grad, t.requires_grad = True, True
        s_t = s(t, x_t)
        assert (2 == s_t.dim())
        dsdt, dsdx = torch.autograd.grad(s_t.sum(), [t, x_t], create_graph=True, retain_graph=True)
        x_t, t = x_t.detach(), t.detach()

        t_0 = t_0*torch.ones([bs, 1]).to(device)
        x_0 = q_t(x, t_0)

        t_1 = t_1*torch.ones([bs, 1]).to(device)
        x_1 = q_t(x, t_1)

        loss = (0.5*(dsdx**2).sum(1, keepdim=True) + dsdt.sum(1, keepdim=True))*omega(t)
        loss = loss + s_t*dodt(t)
        loss = loss.squeeze()/p_t
        loss = loss + (-s(t_1,x_1)*omega(t_1) + s(t_0,x_0)*omega(t_0)).squeeze()

#         time_loss = (0.5*(dsdx**2).sum(1) + (dsdt).sum(1)).detach()
        time_loss = (0.5*(dsdx**2).sum(1)).detach()
        self.update_history(time_loss, t)
        return loss.mean()

    
def train(net, train_loader, val_loader, optim, ema, epochs, device, config):
    s = get_s(net, config)
    q_t, _, _, w, dwdt = get_q(config)
    loss_AM = AdaptiveLoss(alpha=config.train.alpha)
    step = 0
    for epoch in trange(epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(train_loader.batch_size, -1)
            y = torch.nn.functional.one_hot(y).float()
            x = torch.hstack([x, y])
            loss_total = loss_AM.get_loss(s, x, q_t, w, dwdt)
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
            ema.store(net.parameters())
            ema.copy_to(net.parameters())
            net.eval()
            evaluate(step, epoch, s, val_loader, device, config)
            ema.restore(net.parameters())

# def evaluate(step, epoch, s, val_loader, device, config):
#     B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
#     ydim = config.data.ydim
#     x_1 = torch.randn(B, C*W*H).to(device)
#     if config.model.conditional:
#         y_1 = torch.repeat_interleave(torch.eye(ydim), math.ceil(B/ydim), dim=0)[:B]
#         y_1 = y_1.to(device)
#         x_1 = torch.hstack([x_1, y_1])
#     if config.model.classification:
#         y_1 = torch.repeat_interleave(torch.eye(ydim), math.ceil(B/ydim), dim=0)[:B]
#         y_1 = y_1.to(device)
#         y_1 = torch.repeat_interleave(y_1, math.ceil(32*32/ydim), 1)[:,:32*32]
#         x_1 = x_1 + 2*y_1
#     img, nfe_gen = solve_ode_rk(device, s, x_1)
#     if config.model.conditional:
#         label = img[:,-ydim:]
#         img = img[:,:-ydim]
#     img = img.view(B, C, W, H)
#     img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
#     img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

#     x_0, y_1 = next(iter(val_loader))
#     x_0, y_1 = x_0.to(device)[:B], y_1.to(device)[:B]
#     x_0 = x_0.view(B, C*W*H)
#     x_1, nfe_ll = solve_ode_rk(device, s, x_0, t0=0.0, t1=1.0)
#     preds = x_1.to(device)
#     labels = torch.repeat_interleave(2*torch.eye(ydim).to(device), math.ceil(32*32/ydim), 1)[:,:32*32]
#     dists = pairwise_distances(preds.unsqueeze(0), labels.unsqueeze(0))[0]
    
#     meters = {'epoch': epoch, 
#               'RK_function_evals_generation': nfe_gen,
#               'RK_function_evals_likelihood': nfe_ll,
#               'examples': [wandb.Image(stack_imgs(img))],
#               'acc': (torch.argmin(dists,1) == y_1).float().mean()}
#     wandb.log(meters, step=step)            

def evaluate(step, epoch, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    ydim = config.data.ydim
    x_1 = torch.randn(B, C*W*H).to(device)
    if config.model.conditional:
        y_1 = torch.repeat_interleave(torch.eye(ydim), math.ceil(B/ydim), dim=0)[:B]
        y_1 = y_1.to(device)
        x_1 = torch.hstack([x_1, y_1])
    if config.model.classification:
        y_1 = torch.repeat_interleave(torch.eye(ydim), math.ceil(B/ydim), dim=0)[:B]
        y_1 = y_1.to(device)
        y_1 = torch.repeat_interleave(y_1, math.ceil(32*32/ydim), 1)[:,:32*32]
        x_1 = 1e-1*x_1 + y_1
    img, nfe_gen = solve_ode_rk(device, s, x_1)
    if config.model.conditional:
        label = img[:,-ydim:]
        img = img[:,:-ydim]
    img = img.view(B, C, W, H)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    x_0, y_1 = next(iter(val_loader))
    x_0, y_1 = x_0.to(device)[:B], y_1.to(device)[:B]
    x_0 = x_0.view(B, C*W*H)
    if config.model.conditional:
        x_0 = torch.hstack([x_0, torch.randn(B, ydim).to(device)])
    logp, z, nfe_ll = get_likelihood(device, s, x_0)
    bpd = get_bpd(device, logp, x_0, lacedaemon=config.data.lacedaemon)
    bpd = bpd.mean().cpu().numpy()

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'RK_function_evals_likelihood': nfe_ll,
              'likelihood(BPD)': bpd,
              'examples': [wandb.Image(stack_imgs(img))]}
    if config.model.conditional:
        preds = z[:,-ydim:]
        dists = pairwise_distances(preds.unsqueeze(0), torch.eye(ydim).to(device).unsqueeze(0))[0]
        meters['acc'] = (torch.argmin(dists,1) == y_1).float().mean()
    elif config.model.classification:
        preds = z
        labels = torch.repeat_interleave(torch.eye(ydim).to(device), math.ceil(32*32/ydim), 1)[:,:32*32]
        dists = pairwise_distances(preds.unsqueeze(0), labels.unsqueeze(0))[0]
        meters['acc'] = (torch.argmin(dists,1) == y_1).float().mean()
    
    wandb.log(meters, step=step)
    
    
def pairwise_distances(x, y):
    '''
    Input: x is a BatchSize x N x d matrix
           y is a BatchSize x M x d matirx
    Output: dist is a BatchSize x N x M matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    assert x.shape[0] == y.shape[0]
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    norm = ((x - y)**2).sum(3)
    return norm
