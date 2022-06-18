from utils import *
import os
import shutil
from PIL import Image
from pytorch_fid import fid_score
import copy
import torch.distributions as D
import torch.optim.lr_scheduler as lrsc
import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from config_cifar10_32 import *
config = get_configs()

device = torch.device('cuda')
wandb.login()
train_loader, val_loader = get_dataset_CIFAR10(config)

net = nn.DataParallel(ANet(config))
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=config.train.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=0.9999)

wandb.init(project='cifar')
train(net, train_loader, optim, ema_, 1000, device, config)
