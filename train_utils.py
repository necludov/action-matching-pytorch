import torch
import math
import numpy as np
import wandb
import os
import shutil
import torch.distributions as D
import torch.distributed as dist

from copy import deepcopy
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm, trange
from pytorch_fid import fid_score

from eval_utils import *
from evolutions import get_q
from utils import stack_imgs, is_main_host, gather, save_batch
from losses import get_s

import scipy.interpolate


def train(net, loss, train_loader, val_loader, optim, ema, device, config, train_sampler=None):
    epoch = 0
    while config.train.current_step < config.train.n_steps:
        net.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = flatten_data(x, y, config)
            loss_total, meters = loss.eval_loss(x)
            optim.zero_grad(set_to_none=True)
            loss_total.backward()

            if config.train.warmup > 0:
                for g in optim.param_groups:
                    g['lr'] = config.train.lr * np.minimum(config.train.current_step / config.train.warmup, 1.0)
            if config.train.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.train.grad_clip)
            optim.step()
            ema.update(net.parameters())
            if (config.train.current_step % 50) == 0:
                logging_dict = dict((k, meters[k].get_val()) for k in meters)
                if is_main_host():
                    wandb.log(logging_dict, step=config.train.current_step)
                for k in meters: meters[k].reset()
            config.train.current_step += 1
            
            if config.train.current_step >= config.train.n_steps:
                if is_main_host():
                    save(net, ema, optim, loss, config)
                evaluate(net, ema, loss.get_dxdt(), val_loader, device, config)
                return

            if ((config.train.current_step % config.train.save_every) == 0) and is_main_host():
                save(net, ema, optim, loss, config)
            if ((config.train.current_step % config.train.eval_every) == 0):
                evaluate(net, ema, loss.get_dxdt(), val_loader, device, config)
        epoch += 1
            
def evaluate(net, ema, s, val_loader, device, config):
    q_t, _, _ = get_q(config)
    ema.store(net.parameters())
    ema.copy_to(net.parameters())
    net.eval()
    if 'diffusion' == config.model.task:
        evaluate_generic(q_t, s, val_loader, device, config)
    elif 'torus' == config.model.task:
        evaluate_torus(q_t, s, val_loader, device, config)
    elif 'heat' == config.model.task:
        evaluate_generic(q_t, s, val_loader, device, config)
    elif 'color' == config.model.task:
        evaluate_generic(q_t, s, val_loader, device, config)
    elif 'superres' == config.model.task:
        evaluate_generic(q_t, s, val_loader, device, config)
    elif 'inpaint' == config.model.task:
        evaluate_generic(q_t, s, val_loader, device, config)
    else:
        raise NameError('config.model.task name is incorrect')
    ema.restore(net.parameters())
    
def evaluate_generic(q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    if dist.is_initialized():
        B = B//dist.get_world_size()
    t0, t1 = config.model.t0, config.model.t1
    ydim, C_cond = config.data.ydim, config.model.cond_channels
    
    x, y = next(iter(val_loader))
    x, y = x.to(device, non_blocking=True)[:B], y.to(device, non_blocking=True)[:B]
    x = flatten_data(x, y, config)
    x_1, _ = q_t(x, t1*torch.ones([B, 1], device=device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = img.view(B, C + C_cond, W, H)
    if C_cond > 0:
        img = img[:,:C,:,:]
    
    img = img*torch.tensor(config.data.norm_std, device=img.device).view(1,C,1,1)
    img = img + torch.tensor(config.data.norm_mean, device=img.device).view(1,C,1,1)
    img = gather(img.cuda()).cpu()

    if is_main_host():
        meters = {'RK_function_evals_generation': nfe_gen,
                  'examples': [wandb.Image(stack_imgs(img))]}
        wandb.log(meters, step=config.train.current_step)

    
def evaluate_torus(q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    if dist.is_initialized():
        B = B//dist.get_world_size()
    t0, t1 = config.model.t0, config.model.t1
    ydim = config.data.ydim
    
    x, y = next(iter(val_loader))
    x, y = x.to(device, non_blocking=True)[:B], y.to(device, non_blocking=True)[:B]
    x = flatten_data(x, y, config)
    x_1, _ = q_t(x, t1*torch.ones([B, 1], device=device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = torch.remainder(img, 1.0)
    img = img.view(B, C, H, W)
    img = torch.clamp(img, 0.25, 0.75)
    img = 2*(img - 0.25)
    img = gather(img.cuda()).cpu()

    if is_main_host():
        meters = {'RK_function_evals_generation': nfe_gen,
                  'examples': [wandb.Image(stack_imgs(img))]}
        wandb.log(meters, step=config.train.current_step)

    
def evaluate_diffusion(q_t, s, val_loader, device, config):
    B, C, W, H = 64, config.data.num_channels, config.data.image_size, config.data.image_size
    t0, t1 = config.model.t0, config.model.t1
    ydim = config.data.ydim
    
    x, y = next(iter(val_loader))
    x, y = x.to(device)[:B], y.to(device)[:B]
    x = flatten_data(x, y, config)
    x_1, _ = q_t(x, t1*torch.ones([B, 1]).to(device))
    img, nfe_gen = solve_ode(device, s, x_1, t0=t1, t1=t0, method='euler')
    img = img.view(B, C, H, W)
    img = img*torch.tensor(config.data.norm_std).view(1,config.data.num_channels,1,1).to(img.device)
    img = img + torch.tensor(config.data.norm_mean).view(1,config.data.num_channels,1,1).to(img.device)

    x_0, _ = q_t(x, t0*torch.ones([B, 1]).to(device))
    logp, z, nfe_ll = get_likelihood(device, s, x_0)
    bpd = get_bpd(device, logp, x_0)
    bpd = bpd.mean().cpu().numpy()

    meters = {'RK_function_evals_generation': nfe_gen,
              'RK_function_evals_likelihood': nfe_ll,
              'likelihood(BPD)': bpd,
              'examples': [wandb.Image(stack_imgs(img))]}
    wandb.log(meters, step=config.train.current_step)
    


#     if task == 'diffusion':
#         prior_logp = -0.5*(z**2).sum(1) - 0.5*shape[1]*math.log(2*math.pi)
#     elif task == 'torus':
#         prior_logp = 0.0
#     else:
#         raise ValueError(f'task={task} but should be either diffusion or torus')
# 
#     logp = prior_logp + delta_logp
#     print(f'prior_logp={prior_logp}, delta_logp={delta_logp}', flush=True)

    
def evaluate_final(net, loss, val_loader, ema, device, config, total_steps, args):
    integration_method = args.integration_method
    C, W, H = config.data.num_channels, config.data.image_size, config.data.image_size
    t0, t1 = config.model.t0, config.model.t1
    ydim, C_cond = config.data.ydim, config.model.cond_channels
    
    dir_path = '/'.join(config.model.savepath.split('/')[:-1])
    gen_path, test_path = os.path.join(dir_path, 'gen_dir'), os.path.join(dir_path, 'test_dir')
    
    if config.eval.ema:
        ema.store(net.parameters())
        ema.copy_to(net.parameters())
    net.eval()
    q_t, _, _ = get_q(config)
    test_i, gen_i = 0, 0
    loglike = 0.0
    bpd = 0.0
    fid_val = 0.0
    gen_evals, likelihood_evals = 0, 0
    step = step_bpd = 0
    print('starting val loop')
    for x, y in val_loader:
        print(f'starting step={step}', flush=True)
        B = x.shape[0]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = flatten_data(x, y, config)
        x_1, _ = q_t(x, t1*torch.ones([B, 1], device=device))
        if config.model.objective =='am':
            t0 = -4e-2
        elif config.model.objective =='sm':
            t0 = 1e-5
        else:
            raise ValueError()
        print(f'FORCING t0 TO BE {t0}', flush=True)
        x_0, nfe_gen = solve_ode(device, loss.get_dxdt(), x_1, t0=t1, t1=t0, method=integration_method)
        gen_evals += nfe_gen
        x_0 = x_0.view(B, C + C_cond, W, H)[:,:C,:,:]
        x_0 = x_0*torch.tensor(config.data.norm_std, device=x_0.device).view(1,C,1,1)
        x_0 = x_0 + torch.tensor(config.data.norm_mean, device=x_0.device).view(1,C,1,1)
        print(f'saving gen_i={gen_i}', flush=True)
        gen_i = save_batch(x_0, gen_path, gen_i)
        
        if config.model.task in ['diffusion', 'torus'] and step*B < args.num_images_bpd:
            print('calculating likelihood', flush=True)
            if config.model.task == 'torus':
                x_clone, _ = q_t(x, 0*torch.ones([B, 1], device=device))
            else:
                x_clone = x.clone()
            delta_logp, z, nfe = get_likelihood(device, loss.get_dxdt(), x_clone,
                                                task=config.model.task, method=integration_method,
                                                t0=t0, t1=t1)
            likelihood_evals += nfe

            if config.model.task == 'diffusion':
                log_q1_cont = -0.5*(z**2).sum(1) - 0.5*z.shape[1]*math.log(2*math.pi)
            elif config.model.task == 'torus':
                log_q1_cont = 0.0
            else:
                raise ValueError()

            step += 1
            loglike += (log_q1_cont + delta_logp).cpu().mean(0)
            bpd = get_bpd_hack(device, loglike/step, z)
            step_bpd = step
            print(f'step={step} | bpd={bpd} | gen_i={gen_i} | loglike={loglike/step_bpd}', flush=True)
        else:
            step += 1
            
        x = x.view(B, C, W, H)
        x = x*torch.tensor(config.data.norm_std, device=x.device).view(1,C,1,1)
        x = x + torch.tensor(config.data.norm_mean, device=x.device).view(1,C,1,1)

        if config.model.dataset != 'mnist':
            print(f'saving test_i={test_i}', flush=True)
            test_i = save_batch(x, test_path, test_i)

        if step >= total_steps:
            break

    gen_evals /= step
    likelihood_evals /= step
    loglike /= step
    if args.num_images_bpd > 0:
        bpd = get_bpd_hack(device, loglike, z)
    if config.model.dataset != 'mnist':
        print('Evaluating FID', flush=True)
        fid_val = fid_score.calculate_fid_given_paths(
            paths=[gen_path, test_path],
            batch_size=args.batch_size,
            device=device,
            dims=2048
        )
    meters = {'FID': fid_val,
              'function_evals_gen': gen_evals,
              'loglikelihood': loglike/step}
    if config.model.task in ['diffusion', 'torus']:
        meters['likelihood(BPD)'] = bpd
        meters['function_evals_likelihood'] = likelihood_evals
    print(meters, flush=True)
    wandb.log(meters, step=config.train.current_step)
    if config.eval.ema:
        ema.restore(net.parameters())

    
def save(net, ema, optim, loss, config):
    config.model.last_checkpoint = config.model.savepath + '_%d.cpt' % config.train.current_step
    torch.save({'model': net.state_dict(), 
                'ema': ema.state_dict(), 
                'optim': optim.state_dict(),
                'loss': loss.state_dict()}, config.model.last_checkpoint)
    torch.save(config, config.model.savepath + '.config')
    
def flatten_data(x,y,config):
    bs = x.shape[0]
    return x.view(bs, -1)
    
def pairwise_distances(x, y):
    '''
    Input: x is a BatchSize x N x d matrix
           y is a BatchSize x M x d matirx
    Output: dist is a BatchSize x N x M matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    assert x.shape[0] == y.shape[0]
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    norm = ((x - y)**2).sum(3)
    return norm
