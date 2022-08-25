import torch
import math
import numpy as np
import wandb
import os
import shutil
import torch.distributions as D

from copy import deepcopy
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
    def __init__(self, t0=0.0, t1=1.0, n=50, alpha=1e-2, beta=0.99, use_var=False):
        self.t0, self.t1 = t0, t1
        self.alpha, self.beta = alpha, beta
        self.timesteps = np.linspace(t0, t1, n)
        self.dt = (t1-t0)/(n-1)
        self.mean = None
        self.use_var = use_var
        self.buffer_values = []
        self.buffer_times = []
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
    
    def update_history(self, new_w, t):
        new_w, t = new_w.cpu().numpy().flatten(), t.cpu().numpy().flatten()
        weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
        weights = weights/weights.sum(1,keepdims=True)
        if self.mean is None:
            self.mean = weights@new_w
        else:
            self.mean = self.beta*self.mean + (1-self.beta)*(weights@new_w)
        mean_func = scipy.interpolate.interp1d(self.timesteps, self.mean, kind='linear')

        if self.use_var:
            self.buffer_values.append(new_w)
            self.buffer_times.append(t)
            if len(self.buffer_values) > 100:
                self.buffer_values.pop(0)
                self.buffer_times.pop(0)
            var = np.zeros_like(self.timesteps)
            for i in range(len(self.buffer_values)):
                t = self.buffer_times[i]
                weights = np.exp(-np.abs(self.timesteps.reshape(-1, 1) - t.reshape(1,-1))*1e2)
                weights = weights/weights.sum(1,keepdims=True)
                var += weights@((mean_func(t) - self.buffer_values[i])**2)
            var /= len(self.buffer_values)
            if len(self.buffer_values) < 100:
                self.construct_dist(np.ones_like(self.timesteps))
            else:
                self.construct_dist(var)
        else:
            self.construct_dist(self.mean)
    
    def get_loss(self, s, x, q_t, w, dwdt, boundary_conditions=True):
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
        
        loss = (0.5*(dsdx**2).sum(1, keepdim=True) + dsdt.sum(1, keepdim=True))*w(t)
        dsdx_std = 0.5*(dsdx**2).sum(1).detach().cpu().std()
        dsdt_std = dsdt.sum(1).detach().cpu().std()
        loss = loss + s_t*dwdt(t)
        s_std = s_t.sum(1).detach().cpu().std()
        loss = loss.squeeze()/p_t

        s_1_std, s_0_std = 0.0, 0.0
        if boundary_conditions:
            t_0 = t_0*torch.ones([bs, 1]).to(device)
            x_0 = q_t(x, t_0)

            t_1 = t_1*torch.ones([bs, 1]).to(device)
            x_1 = q_t(x, t_1)

            loss = loss + (-s(t_1,x_1)*w(t_1) + s(t_0,x_0)*w(t_0)).squeeze()
            s_1_std = s(t_1,x_1).sum(1).detach().cpu().std()
            s_0_std = s(t_0,x_0).sum(1).detach().cpu().std()

        dmetricdt = ((dsdx**2).sum(1)).detach()
        self.update_history(dmetricdt, t)
        return loss.mean(), (dsdx_std, dsdt_std, s_std, s_1_std, s_0_std)

    
def train(net, train_loader, val_loader, optim, ema, device, config):
    s = get_s(net, config)
    q_t, sigma, w, dwdt = get_q(config)
    boundary_conditions = True
    if (w(torch.tensor(config.model.t0)) == 0.0) and (w(torch.tensor(config.model.t1)) == 0.0):
        config.train.boundary_conditions = 'off'
        print('boundary conditions are off')
        boundary_conditions = False
    else:
        config.train.boundary_conditions = 'on'
        print('boundary conditions are on')
    print('w0, w1 = %.5e, %.5e' % (w(torch.tensor(config.model.t0)), w(torch.tensor(config.model.t1))))
    
    loss_AM = AdaptiveLoss(config.model.t0, config.model.t1, alpha=config.train.alpha, use_var=config.train.use_var)
    step = config.train.current_step
    for epoch in trange(config.train.current_epoch, config.train.n_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = flatten_data(x, y, config)
            loss_total, stds = loss_AM.get_loss(s, x, q_t, w, dwdt, boundary_conditions)
            optim.zero_grad()
            loss_total.backward()

            if config.train.warmup > 0:
                for g in optim.param_groups:
                    g['lr'] = config.train.lr * np.minimum(step / config.train.warmup, 1.0)
            if config.train.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.train.grad_clip)
            optim.step()
            ema.update(net.parameters())
            dsdx_std, dsdt_std, s_std, s_1_std, s_0_std = stds
            if (step % 50) == 0:
                wandb.log({'train_loss': loss_total,
                           'dsdx_std': dsdx_std,
                           'dsdt_std': dsdt_std,
                           's_std': s_std,
                           's_1_std': s_1_std,
                           's_0_std': s_0_std}, step=step)
            step += 1
        if ((epoch % config.train.save_every) == 0):
            save(step, epoch, net, ema, optim, config)
        
        if ((epoch % config.train.eval_every) == 0) and (epoch >= config.train.first_eval):
            evaluate(step, epoch, q_t, net, ema, s, val_loader, device, config)
    save(step, epoch, net, ema, optim, config)
    evaluate(step, epoch, q_t, net, ema, s, val_loader, device, config)

            
def evaluate(step, epoch, q_t, net, ema, s, val_loader, device, config):
    ema.store(net.parameters())
    ema.copy_to(net.parameters())
    net.eval()
    if 'diffusion' == config.model.task:
        evaluate_diffusion(step, epoch, q_t, s, val_loader, device, config)
    elif 'heat' == config.model.task:
        evaluate_heat(step, epoch, s, val_loader, device, config)
    elif 'conditional' == config.model.task:
        evaluate_cond(step, epoch, s, val_loader, device, config)
    elif 'augmented' == config.model.task:
        evaluate_augmented(step, epoch, s, val_loader, device, config)
    elif 'color' == config.model.task:
        evaluate_color(step, epoch, s, val_loader, device, config)
    elif 'superres' == config.model.task:
        evaluate_superres(step, epoch, s, val_loader, device, config)
    else:
        raise NameError('config.model.task name is incorrect')
    ema.restore(net.parameters())   
        
def evaluate_diffusion(step, epoch, q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    t0, t1 = config.model.t0, config.model.t1
    ydim = config.data.ydim
    
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = flatten_data(x, y, config)
    x_1 = q_t(x, t1*torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = img.view(B, C, H, W)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

#     x_0, y_1 = next(iter(val_loader))
#     x_0, y_1 = x_0.to(device)[:B], y_1.to(device)[:B]
#     x_0 = x_0.view(B, C*W*H)
#     logp, z, nfe_ll = get_likelihood(device, s, x_0)
#     bpd = get_bpd(device, logp, x_0, lacedaemon=config.data.lacedaemon)
#     bpd = bpd.mean().cpu().numpy()

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
#               'RK_function_evals_likelihood': nfe_ll,
#               'likelihood(BPD)': bpd,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)
    
def evaluate_heat(step, epoch, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    ydim = config.data.ydim
    x_0, y_1 = next(iter(val_loader))
    x_0, y_1 = x_0.to(device)[:B], y_1.to(device)[:B]
    q_t, sigma, w, dwdt = get_q(config)
    x_0 = x_0.view(B, C*W*H)
    y_1 = torch.nn.functional.one_hot(y_1, num_classes=config.data.ydim).float()
    x_0 = torch.hstack([x_0, y_1])
    x_1 = q_t(x_0, torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1)
    img = img.view(B, C, W, H)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)

def evaluate_color(step, epoch, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    C_cond = config.model.cond_channels
    ydim = config.data.ydim
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = x.view(B, C*W*H)
    y = torch.nn.functional.one_hot(y, num_classes=config.data.ydim).float()
    x = torch.hstack([x, y])
    q_t, sigma, w, dwdt = get_q(config)
    x_1 = q_t(x, torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1)
    img = img.view(B, C + C_cond, W, H)
    if C_cond > 0:
        img = img[:,:C,:,:]
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)
    
def evaluate_superres(step, epoch, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    C_cond = config.model.cond_channels
    ydim = config.data.ydim
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = x.view(B, C*W*H)
    y = torch.nn.functional.one_hot(y, num_classes=config.data.ydim).float()
    x = torch.hstack([x, y])
    q_t, sigma, w, dwdt = get_q(config)
    x_1 = q_t(x, torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1)
    img = img.view(B, C + C_cond, W, H)
    if C_cond > 0:
        img = img[:,:C,:,:]
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)
    
def evaluate_cond(step, epoch, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    ydim = config.data.ydim
    x_1 = torch.randn(B, C*W*H).to(device)
    y_1 = torch.repeat_interleave(torch.eye(ydim), math.ceil(B/ydim), dim=0)[:B]
    y_1 = y_1.to(device)
    x_1 = torch.hstack([x_1, y_1])
    img, nfe_gen = solve_ode(device, s, x_1, method='euler')
    label = img[:,-ydim:]
    img = img[:,:-ydim]
    img = img.view(B, C, W, H)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    x_0, y_1 = next(iter(val_loader))
    x_0, y_1 = x_0.to(device)[:B], y_1.to(device)[:B]
    x_0 = x_0.view(B, C*W*H)
    x_0 = torch.hstack([x_0, torch.randn(B, ydim).to(device)])
    logp, z, nfe_ll = get_likelihood(device, s, x_0, method='euler')
    bpd = get_bpd(device, logp, x_0, lacedaemon=config.data.lacedaemon)
    bpd = bpd.mean().cpu().numpy()

    preds = z[:,-ydim:]
    dists = pairwise_distances(preds.unsqueeze(0), torch.eye(ydim).to(device).unsqueeze(0))[0]
    meters['acc'] = (torch.argmin(dists,1) == y_1).float().mean()
    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'RK_function_evals_likelihood': nfe_ll,
              'likelihood(BPD)': bpd,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)

    
def save(step, epoch, net, ema, optim, config):
    config.model.last_checkpoint = config.model.savepath + '_%d.cpt' % epoch
    config.train.current_epoch = epoch
    config.train.current_step = step
    torch.save({'model': net.state_dict(), 
                'ema': ema.state_dict(), 
                'optim': optim.state_dict()}, config.model.last_checkpoint)
    torch.save(config, config.model.savepath + '.config')
    
def flatten_data(x,y,config):
    bs = x.shape[0]
    x = x.view(bs, -1)
    y = torch.nn.functional.one_hot(y, num_classes=config.data.ydim).float()
    return torch.hstack([x, y])
    
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
