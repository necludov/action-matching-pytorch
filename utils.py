from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from models import anet
from models import ema
from evolutions import get_s
from tqdm.auto import tqdm, trange
import torch
import math
import numpy as np
import wandb
import os
import shutil
from PIL import Image
import torch.distributions as D


class ANet(nn.Module):
    def __init__(self, config):
        super(ANet, self).__init__()
        
        self.anet = anet.ActionNet(config)
        
    def forward(self, t, x):
        bs = x.shape[0]
        
        t = t.reshape(-1)
        t = t.expand(bs)
        return self.anet(x, t)

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
        
def train(net, train_loader, optim, ema, epochs, device, config):
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
            log_losses('Train', loss_total, step)
            step += 1
        torch.save({'model': net.state_dict(), 'ema': ema.state_dict(), 'optim': optim.state_dict()}, config.model.savepath)
        wandb.log({'epoch': epoch}, step=step)
        
        net.eval()
        x_1 = torch.randn(64, x.shape[1], x.shape[2], x.shape[3]).to(device)
        img, _, _ = solve_ode(device, s, x_1, [])
        wandb.log({"examples": [wandb.Image(stack_imgs(img))]})


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_dataset_CIFAR10(config):

    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = CIFAR10(root='../data/', train=True, download=True, transform=transform)
    val_data = CIFAR10(root='../data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    return train_loader, val_loader

def get_dataset_MNIST(config):

    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = MNIST(root='../data/', train=True, download=True, transform=transform)
    val_data = MNIST(root='../data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    return train_loader, val_loader

def log_losses(title, loss, step):
    title = f'{title}_losses'
    
    if loss is not None:
        wandb.log({f'{title}/loss_sm': loss}, step=step)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def rmdir(path):
    shutil.rmtree(path)

def save_img(p, path, num, scale):
    mkdir(path)
    m = torch.tensor(scale[0])
    sc = torch.tensor(scale[1])
    for i in range(len(sc)):
        p[i,:,:] = p[i,:,:]*sc[i] + m[i]
    p = p * 255
    p = p.clamp(0, 255)
    p = p.detach().cpu().numpy()
    p = p.astype(np.uint8)
    p = p.transpose((1,2,0))
    if p.shape[-1] == 3:
        p = Image.fromarray(p, mode='RGB')
    elif p.shape[-1] == 1:
        p = p.squeeze(2)
        p = Image.fromarray(p, mode='L')
    p.save(f"{path}/{num}.png", format="png")
    
def save_batch(x, path, num, scale):
    for p in x:
        save_img(p, path, num, scale)
        num += 1
    return num

def save_dataloader(loader, path, scale, n=2048):
    m = 0
    for x, _ in loader:
        m = save_batch(x, path, m, scale)
        if m >= n:
            break
            
def save_callable(foo, path, scale, n=2048):
    m = 0
    while m < n:
        m = save_batch(foo(), path, m, scale)
        
@torch.no_grad()
def calc_fid(foo):
    path_1 = "data_1"
    path_2 = "data_2"
    
    save_dataloader(train_loader, path_1, 16*1024)
    save_callable(foo, path_2, 16*1024)
    
    res = fid_score.calculate_fid_given_paths(
        paths=[path_1, path_2],
        batch_size=128,
        device=device,
        dims=2048
    )
    
#     rmdir(path_1)
#     rmdir(path_2)
    
    return res

def solve_ode(device, s, x, i_inter, ts=1.0, tf=0.0, dt=-1e-3):
    x_inter = []
    t_inter = []
    for i, t in enumerate(np.arange(ts, tf, dt)):
        if i in i_inter:
            x_inter.append(x.clone())
            t_inter.append(t)
        tt = (t*torch.ones([x.shape[0],1])).to(device)
        x.requires_grad = True
        x.data += dt * torch.autograd.grad(s(tt, x).sum(), x)[0].detach()
        x = x.detach()
    return x, x_inter, t_inter

def stack_imgs(x):
    big_img = np.zeros((8*32,8*32),dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            y = 0.5*x + 0.5
            p = y[i*8+j] * 255
            p = p.clamp(0, 255)
            p = p.detach().cpu().numpy()
            p = p.astype(np.uint8)
            p = p.transpose((1,2,0))
            p = p.squeeze(2)
            big_img[i*32:(i+1)*32, j*32:(j+1)*32] = p
    return big_img