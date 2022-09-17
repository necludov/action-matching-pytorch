import torch
import math
import numpy as np
import wandb
import os
import shutil
import torch.distributions as D
import torch.distributed as dist

from torch import nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10, MNIST, CelebA
from torch.utils.data import DataLoader
from scipy import integrate
from PIL import Image
from tqdm.auto import tqdm, trange

def gather(x):
    if dist.is_initialized():
        x_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(x_list, x)
        x = torch.vstack(x_list)
    return x

class DDPAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = torch.tensor([0.0])
        self.avg = torch.tensor([0.0])
        self.sum = torch.tensor([0.0])
        self.count = torch.tensor([0])

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def get_val(self):
        if dist.is_initialized():
            val = self.val.clone().cuda()
            avg = self.avg.clone().cuda()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
            self.val = val.item() / dist.get_world_size()
            self.avg = avg.item() / dist.get_world_size()
        return self.val

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_host():
    return get_rank() == 0

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
        num_workers=4,
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True, 
        pin_memory=True
    )
    return train_loader, val_loader


def get_dataset_CIFAR10_DDP(config):
    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = CIFAR10(root='../data/', train=True, download=True, transform=transform)
    val_data = CIFAR10(root='../data/', train=False, download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=True, drop_last=True)
    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True, 
        pin_memory=True, 
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True, 
        pin_memory=True, 
        sampler=val_sampler
    )
    return train_loader, val_loader, train_sampler

class Crop:
    def __init__(self, hs, ws, h, w):
        self.p = (hs, ws, h, w)

    def __call__(self, img):
        return TF.crop(img, *self.p)

def get_dataset_CelebA(config):
    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        Crop(40, 15, 148, 148),
        transforms.Resize((config.data.image_size,config.data.image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = CelebA(root='/ssd003/home/kirill/learning-continuity/data/celeba_pytorch/', split='train', download=False, transform=transform)
    val_data = CelebA(root='/ssd003/home/kirill/learning-continuity/data/celeba_pytorch/', split='test', download=False, transform=transform)
    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True, 
        pin_memory=True
    )
    return train_loader, val_loader

def get_dataset_CelebA_DDP(config):
    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        Crop(40, 15, 148, 148),
        transforms.Resize((config.data.image_size,config.data.image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = CelebA(root='/ssd003/home/kirill/learning-continuity/data/celeba_pytorch/', split='train', download=False, transform=transform)
    val_data = CelebA(root='/ssd003/home/kirill/learning-continuity/data/celeba_pytorch/', split='test', download=False, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=True, drop_last=True)
    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True, 
        pin_memory=True, 
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True, 
        pin_memory=True, 
        sampler=val_sampler
    )
    return train_loader, val_loader, train_sampler

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
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True, 
        pin_memory=True
    )
    return train_loader, val_loader

def get_dataset_MNIST_DDP(config):
    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = MNIST(root='../data/', train=True, download=True, transform=transform)
    val_data = MNIST(root='../data/', train=False, download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=True, drop_last=True)
    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True, 
        pin_memory=True, 
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True, 
        pin_memory=True, 
        sampler=val_sampler
    )
    return train_loader, val_loader, train_sampler

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
    im_size = x.shape[2]
    big_img = np.zeros((8*im_size,8*im_size,x.shape[1]),dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            p = x[i*8+j] * 255
            p = p.clamp(0, 255)
            p = p.detach().cpu().numpy()
            p = p.astype(np.uint8)
            p = p.transpose((1,2,0))
            big_img[i*im_size:(i+1)*im_size, j*im_size:(j+1)*im_size] = p
    return big_img