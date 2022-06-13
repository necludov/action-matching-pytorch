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

from config_mnist import *
config = get_configs()

device = torch.device('cuda')
wandb.login()
train_loader, val_loader = get_dataset_MNIST(config)

eps_th = nn.DataParallel(ANet(config))
eps_th.to(device)

optim = torch.optim.Adam(eps_th.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
ema_ = ema.ExponentialMovingAverage(eps_th.parameters(), decay=0.9999)
sched = lrsc.StepLR(optim, step_size=30, gamma=0.1)

wandb.init(project='mnist')
train(eps_th, train_loader, optim, ema_, 1000, 0, 1.0, device, config)


# key='71e4c7875e89fd0aaf1b292d4cdb511699b90b42'