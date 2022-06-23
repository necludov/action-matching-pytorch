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
from scipy import integrate
from PIL import Image
from tqdm.auto import tqdm, trange

from evaluation import *
from evolutions import get_s
from utils import stack_imgs


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
    step = 0
    for epoch in trange(epochs):
        net.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss_total = loss_AM(s, x, w, dwdt, q_t)
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
            evaluate(epoch, net, s, val_loader, device, config)
        
def evaluate(epoch, net, s, val_loader, device, config):
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
