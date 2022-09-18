import os
import shutil
from pytorch_fid import fid_score
import copy
import argparse
import random

import torch
import torch.distributions as D
import torch.distributed as dist
import torch.optim.lr_scheduler as lrsc
import numpy as np
import wandb

from torch import nn
from tqdm.auto import tqdm, trange
from configs import job_configs

from losses import get_loss
from models import anet, ddpm
from models import ema
from train_utils import evaluate_final

    
def launch(args, config):
    device = torch.device('cuda')
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)

    wandb.login()
    if 'mnist' == args.dataset:
        from utils import get_dataset_MNIST as get_dataset
    elif 'cifar' == args.dataset:
        from utils import get_dataset_CIFAR10 as get_dataset
    elif 'celeba' == args.dataset:
        from utils import get_dataset_CelebA as get_dataset
    else:
        raise NameError('unknown dataset')
    config.data.batch_size = 64
    train_loader, val_loader = get_dataset(config)

    if 'am' == config.model.objective:
        net = nn.DataParallel(anet.ActionNet(config))
    elif 'sm' == config.model.objective:
        net = nn.DataParallel(ddpm.DDPM(config))
    else:
        raise NameError('config.model.objective name is incorrect')
    net.to(device)

    optim = torch.optim.Adam(net.parameters(), lr=config.train.lr, betas=config.train.betas, 
                             eps=1e-8, weight_decay=config.train.wd)
    ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=config.eval.ema)
    loss = get_loss(net, config)
    
    if config.model.last_checkpoint is not None:
        state = torch.load(config.model.last_checkpoint, map_location=device)
        print('starting from existing checkpoint')
        net.load_state_dict(state['model'], strict=True)
        ema_.load_state_dict(state['ema'])
        optim.load_state_dict(state['optim'])
        loss.load_state_dict(state['loss'])
        print('dicts are successfully loaded')

    wandb.init(id=config.train.wandbid, 
               project=args.dataset + '_' + config.model.task, 
               resume="allow",
               config=config)
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = config.train.wandbid
    evaluate_final(net, loss, train_loader, ema_, device, config)
    
def main(args):
    config_name = args.config_path
    print('starting from existing config:', config_name)    
    config = torch.load(config_name)
    launch(args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description=''
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='dataset: mnist or cifar',
        default='celeba'
    )
    
    parser.add_argument(
        '--config_path',
        type=str,
        help='path to config in case of starting from checkpoint',
        default=None
    )
    main(parser.parse_args())
