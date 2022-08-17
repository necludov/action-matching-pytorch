import os
import shutil
from PIL import Image
from pytorch_fid import fid_score
import copy
import argparse

import torch
import torch.distributions as D
import torch.optim.lr_scheduler as lrsc
import wandb

from torch import nn
from tqdm.auto import tqdm, trange

from models import anet
from models import ema
from train import train

from utils import get_dataset_CIFAR10
from config_cifar10_32 import get_configs

def main(args):
    config = get_configs()
    config.model.save_path = os.path.join(args.checkpoint_dir, config.model.savepath)
    torch.save(config, config.model.save_path + '.config')

    device = torch.device('cuda')
    wandb.login()
    train_loader, val_loader = get_dataset_CIFAR10(config)

    net = nn.DataParallel(anet.ActionNet(config))
    net.to(device)

    optim = torch.optim.Adam(net.parameters(), lr=config.train.lr, betas=config.train.betas, eps=1e-8, weight_decay=0)
    ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=config.eval.ema)

    wandb.init(project='cifar')
    train(net, train_loader, val_loader, optim, ema_, 1000, device, config)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description=''
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='path to save and look for the checkpoint file',
        default=os.getcwd()
    )
    
    main(parser.parse_args())
