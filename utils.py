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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    """methods to enable pickling"""
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    
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
        num_workers=8,
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True, 
        pin_memory=True
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
        num_workers=8,
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True, 
        pin_memory=True
    )
    return train_loader, val_loader

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

def stack_imgs(x):
    big_img = np.zeros((8*32,8*32,x.shape[1]),dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            p = x[i*8+j] * 255
            p = p.clamp(0, 255)
            p = p.detach().cpu().numpy()
            p = p.astype(np.uint8)
            p = p.transpose((1,2,0))
            big_img[i*32:(i+1)*32, j*32:(j+1)*32] = p
    return big_img