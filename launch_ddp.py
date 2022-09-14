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

from models import anet, ddpm
from models import ema
from train_utils import train
from utils import is_main_host, get_world_size

    
def launch_traininig(args, config):
    local_gpu = int(os.environ["LOCAL_RANK"])
    rank =  int(os.environ["RANK"])
    print(rank, "Use GPU: {} for training".format(local_gpu))
    config.train.seed = config.train.seed + rank
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    
    dist.init_process_group(backend='nccl', init_method="env://")
    config.data.batch_size = config.data.batch_size//get_world_size()
    torch.cuda.set_device(local_gpu)
    device = torch.device(local_gpu)

    if 'mnist' == args.dataset:
        from utils import get_dataset_MNIST_DDP as get_dataset
    elif 'cifar' == args.dataset:
        from utils import get_dataset_CIFAR10_DDP as get_dataset
    else:
        raise NameError('unknown dataset')
    train_loader, val_loader, train_sampler = get_dataset(config)

    if 'am' == config.model.objective:
        net = anet.ActionNet(config)
    elif 'sm' == config.model.objective:
        net = ddpm.DDPM(config)
    else:
        raise NameError('config.model.objective name is incorrect')
    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_gpu])

    optim = torch.optim.Adam(net.parameters(), lr=config.train.lr, betas=config.train.betas, 
                             eps=1e-8, weight_decay=config.train.wd)
    ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=config.eval.ema)
    
    if config.model.last_checkpoint is not None:
        state = torch.load(config.model.last_checkpoint, map_location=device)
        print('starting from existing checkpoint')
        net.load_state_dict(state['model'], strict=True)
        ema_.load_state_dict(state['ema'])
        optim.load_state_dict(state['optim'])
        print('dicts are successfully loaded')

    if is_main_host():
        wandb.login()
        wandb.init(id=config.train.wandbid, 
                   project=args.dataset + '_' + config.model.task, 
                   resume="allow",
                   config=config)
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = config.train.wandbid
    train(net, train_loader, val_loader, optim, ema_, device, config, train_sampler)
    
def main(args):
    filenames = os.listdir(args.checkpoint_dir)
    configs = list(filter(lambda f: '.config' in f, filenames))
    has_config = len(configs) > 0
    if args.config_path is not None:
        # starting from checkpoint
        config_name = args.config_path
        print('starting from existing config:', config_name)
        config = torch.load(config_name)
    elif has_config:
        # preemption case
        assert len(configs) == 1
        config_name = os.path.join(args.checkpoint_dir, configs[0])
        print('starting from existing config:', config_name)
        config = torch.load(config_name)
    else:
        # init from default config
        if 'mnist' == args.dataset:
            from config_mnist import get_configs
        elif 'cifar' == args.dataset:
            from config_cifar10_32 import get_configs
        else:
            raise NameError('unknown dataset')
        config = get_configs()
        if is_main_host():
            config.model.savepath = os.path.join(args.checkpoint_dir, config.model.savepath)
            config.train.wandbid = wandb.util.generate_id()
            torch.save(config, config.model.savepath + '.config')
    launch_traininig(args, config)


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
        '--config_path',
        type=str,
        help='path to config in case of starting from checkpoint',
        default=None
    )
    
    main(parser.parse_args())
