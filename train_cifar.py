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
from train_utils import train

    
def launch_traininig(args, config, state=None):
    device = torch.device('cuda')
    wandb.login()
    if 'mnist' == args.dataset:
        from utils import get_dataset_MNIST as get_dataset
    elif 'cifar' == args.dataset:
        from utils import get_dataset_CIFAR10 as get_dataset
    else:
        raise NameError('unknown dataset')
    train_loader, val_loader = get_dataset(config)

    net = nn.DataParallel(anet.ActionNet(config))
    net.to(device)

    optim = torch.optim.Adam(net.parameters(), lr=config.train.lr, betas=config.train.betas, eps=1e-8, weight_decay=0)
    ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=config.eval.ema)
    
    if state is not None:
        net.load_state_dict(state['model'], strict=True)
        ema_.load_state_dict(state['ema'], strict=True)
        optim.load_state_dict(state['optim'], strict=True)
        print('dicts are successfully loaded')

    wandb.init(id=config.train.wandbid, project=args.dataset + '_' + config.model.task, resume="allow")
    train(net, train_loader, val_loader, optim, ema_, config.train.n_epochs, device, config)
    
def main(args):
    filenames = os.listdir(args.checkpoint_dir)
    configs = list(filter(lambda f: '.config' in f, filenames))
    has_config = len(configs) > 0
    if has_config:
        assert len(configs) == 1
        config = torch.load(os.path.join(args.checkpoint_dir, configs[0]))
        print('starting from existing config')
        if config.model.last_checkpoint is not None:
            state = torch.load(config.model.last_checkpoint)
            print('starting from existing checkpoint')
        else:
            state = None
    else:
        if 'mnist' == args.dataset:
            from config_mnist import get_configs
        elif 'cifar' == args.dataset:
            from config_cifar10_32 import get_configs
        else:
            raise NameError('unknown dataset')
        config = get_configs()
        config.model.savepath = os.path.join(args.checkpoint_dir, config.model.savepath)
        config.train.wandbid = wandb.util.generate_id()
        torch.save(config, config.model.savepath + '.config')
        state = None
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
    
    main(parser.parse_args())
