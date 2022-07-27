import os
import shutil
from PIL import Image
from pytorch_fid import fid_score
import copy

import torch
import torch.distributions as D
import torch.optim.lr_scheduler as lrsc
import wandb

from torch import nn
from tqdm.auto import tqdm, trange

from models import anet
from models import ema
from train import train


from utils import get_dataset_MNIST
from config_mnist import get_configs
config = get_configs()

device = torch.device('cuda')
wandb.login()
train_loader, val_loader = get_dataset_MNIST(config)

net = nn.DataParallel(anet.ActionNet(config))
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=config.train.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=0.9999)

wandb.init(project='mnist')
train(net, train_loader, val_loader, optim, ema_, 1000, device, config)
