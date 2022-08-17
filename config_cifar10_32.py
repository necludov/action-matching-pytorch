from utils import dotdict
from evolutions import *

def get_configs():
    model_dict = dotdict()
    model_dict.nf = 128
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 3
    model_dict.attn_resolutions = (16, 8)
    model_dict.dropout = 0.1
    model_dict.resamp_with_conv = True
    model_dict.task = 'superres'
    model_dict.sigma = 'simple'
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join(['am', 'cifar', model_dict.task])
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 3
    data_dict.centered = True
    data_dict.batch_size = 128
#     data_dict.norm_mean = (0.4914, 0.4822, 0.4465)
#     data_dict.norm_std = (0.2470, 0.2435, 0.2616)
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 1e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.eval_every = 10
    train_dict.first_eval = 10
    train_dict.alpha = 1e-2
    
    eval_dict = dotdict()
    eval_dict.batch_size = 100
    eval_dict.ema = 0.9999
    eval_dict.n_tries = 1
    
    config_dict = dotdict()
    config_dict.model = model_dict
    config_dict.data = data_dict
    config_dict.train = train_dict
    config_dict.eval = eval_dict
    return config_dict
