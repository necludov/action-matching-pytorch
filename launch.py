import os
import shutil
from pytorch_fid import fid_score
import copy
import argparse
import random

import torch
import torch.distributions as D
import torch.optim.lr_scheduler as lrsc
import numpy as np
import wandb

from torch import nn
from tqdm.auto import tqdm, trange

from configs import job_configs
from models import anet, ddpm
from models import ema
from losses import get_loss
from train_utils import train

    
def launch_traininig(args, config, state=None):
    device = torch.device('cuda')
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)

    wandb.login()
    if 'mnist' == args.dataset:
        from utils import get_dataset_MNIST as get_dataset
    elif 'cifar' == args.dataset:
        from utils import get_dataset_CIFAR10 as get_dataset
    else:
        raise NameError('unknown dataset')
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
    
    if state is not None:
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
    train(net, loss, train_loader, val_loader, optim, ema_, device, config)
    
def main(args):
    filenames = os.listdir(args.checkpoint_dir)
    configs = list(filter(lambda f: '.config' in f, filenames))
    has_config = len(configs) > 0
    if has_config:
        assert len(configs) == 1
        config_name = os.path.join(args.checkpoint_dir, configs[0])
        print('starting from existing config:', config_name)
        config = torch.load(config_name)
        if config.model.last_checkpoint is not None:
            state = torch.load(config.model.last_checkpoint)
            print('starting from existing checkpoint')
        else:
            state = None
    elif args.job_config_name is not None:
        config = job_configs[args.job_config_name]
        config.model.savepath = os.path.join(args.checkpoint_dir, config.model.savepath)
        config.train.wandbid = wandb.util.generate_id()
        torch.save(config, config.model.savepath + '.config')
        state = None
    else:
        raise NameError()
    launch_traininig(args, config, state)


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
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='dataset: mnist or cifar',
        default='cifar'
    )

    parser.add_argument(
        '--job_config_name',
        type=str,
        help='name of config to load from configs.py',
        default='experimental'
    )
    
    main(parser.parse_args())
