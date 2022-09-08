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

from losses import get_loss
from evaluation import *
from evolutions import get_q
from utils import stack_imgs

import scipy.interpolate


def train(net, train_loader, val_loader, optim, ema, device, config):
    loss = get_loss(net, config)
    step = config.train.current_step
    for epoch in trange(config.train.current_epoch, config.train.n_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = flatten_data(x, y, config)
            loss_total, meters = loss.eval_loss(x)
            optim.zero_grad(set_to_none=True)
            loss_total.backward()

            if config.train.warmup > 0:
                for g in optim.param_groups:
                    g['lr'] = config.train.lr * np.minimum(step / config.train.warmup, 1.0)
            if config.train.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.train.grad_clip)
            optim.step()
            ema.update(net.parameters())
            if (step % 50) == 0:
                wandb.log(meters, step=step)
            step += 1

        if ((epoch % config.train.save_every) == 0):
            save(step, epoch, net, ema, optim, config)
        if ((epoch % config.train.eval_every) == 0) and (epoch >= config.train.first_eval):
            evaluate(step, epoch, net, ema, loss.get_dxdt(), val_loader, device, config)
    save(step, epoch, net, ema, optim, config)
    evaluate(step, epoch, net, ema, loss.get_dxdt(), val_loader, device, config)
            
def evaluate(step, epoch, net, ema, s, val_loader, device, config):
    q_t, _, _ = get_q(config)
    ema.store(net.parameters())
    ema.copy_to(net.parameters())
    net.eval()
    if 'diffusion' == config.model.task:
        evaluate_generic(step, epoch, q_t, s, val_loader, device, config)
    elif 'torus' == config.model.task:
        evaluate_torus(step, epoch, q_t, s, val_loader, device, config)
    elif 'heat' == config.model.task:
        evaluate_generic(step, epoch, q_t, s, val_loader, device, config)
    elif 'color' == config.model.task:
        evaluate_generic(step, epoch, q_t, s, val_loader, device, config)
    elif 'superres' == config.model.task:
        evaluate_generic(step, epoch, q_t, s, val_loader, device, config)
    else:
        raise NameError('config.model.task name is incorrect')
    ema.restore(net.parameters())
    
def evaluate_generic(step, epoch, q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    t0, t1 = config.model.t0, config.model.t1
    ydim, C_cond = config.data.ydim, config.model.cond_channels
    
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = flatten_data(x, y, config)
    x_1, _ = q_t(x, t1*torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = img.view(B, C + C_cond, W, H)
    if C_cond > 0:
        img = img[:,:C,:,:]
    img = img*torch.tensor(config.data.norm_std).view(1,C,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,C,1,1).to(img.device)

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)

    
def evaluate_torus(step, epoch, q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    t0, t1 = config.model.t0, config.model.t1
    ydim = config.data.ydim
    
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = flatten_data(x, y, config)
    x_1, _ = q_t(x, t1*torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = torch.remainder(img, 1.0)
    img = img.view(B, C, H, W)
    img = torch.clamp(img, 0.25, 0.75)
    img = 2*(img - 0.25)

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=step)

    
def evaluate_diffusion(step, epoch, q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    t0, t1 = config.model.t0, config.model.t1
    ydim = config.data.ydim
    
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = flatten_data(x, y, config)
    x_1, _ = q_t(x, t1*torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = img.view(B, C, H, W)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    x_0, _ = q_t(x, t0*torch.ones([B, 1]).to(device))
    logp, z, nfe_ll = get_likelihood(device, s, x_0)
    bpd = get_bpd(device, logp, x_0)
    bpd = bpd.mean().cpu().numpy()

    meters = {'epoch': epoch, 
              'RK_function_evals_generation': nfe_gen,
              'RK_function_evals_likelihood': nfe_ll,
              'likelihood(BPD)': bpd,
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
