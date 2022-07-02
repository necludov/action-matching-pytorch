import os
import shutil
from PIL import Image
from pytorch_fid import fid_score
import copy

import torch
import torch.distributions as D
import torch.optim.lr_scheduler as lrsc

from torch import nn
from tqdm.auto import tqdm, trange

from models import anet
from models import ema
from train import train

from evaluation import *
from evolutions import *

from utils import get_dataset_MNIST
from config_mnist import get_configs
config = get_configs()

device = torch.device('cuda')

config.data.batch_size = config.eval.batch_size
train_loader, val_loader = get_dataset_MNIST(config)

net = nn.DataParallel(anet.ANet(config))
net.to(device)

state = torch.load(config.model.savepath)
net.load_state_dict(state['model'], strict=True)
if config.eval.ema:
    ema_ = ema.ExponentialMovingAverage(net.parameters(), decay=0.9999)
    ema_.load_state_dict(state['ema'])
    ema_.copy_to(net.parameters())
net.eval()
s = get_s(net, config.model.s)

n_imgs = len(val_loader)*val_loader.batch_size
norm_const = torch.zeros([n_imgs, config.eval.n_tries])
bpd = torch.zeros([n_imgs, config.eval.n_tries])
for try_i in trange(config.eval.n_tries):
    k = 0
    for x_0, _ in val_loader:
        bs = x_0.shape[0]
        x_0 = x_0.to(device)
        s_0 = s(torch.zeros(x_0.shape[0]).to(device), x_0).squeeze().detach()
        logp, z, nfe = get_likelihood(device, s, x_0)
        bpd[k:k+bs, try_i] = get_bpd(device, logp, x_0, lacedaemon=config.data.lacedaemon).cpu()
        norm_const[k:k+bs, try_i] = (-2*s_0/beta_0 - 0.5*(x_0**2).sum([1,2,3]) - logp).cpu()
        k += bs

print('bits per dimenstion: %.4e' % bpd.mean())
print('norm constant: %.4e (%.4e)' % (norm_const.mean(), norm_const.std()))
print('norm constant std/mean: %.4e' % torch.abs(norm_const.std()/norm_const.mean()))
