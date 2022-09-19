from utils import dotdict
from evolutions import *

def make_am_celeba_diffusion():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'am' # am or sm
    model_dict.task = 'diffusion'
    model_dict.sigma = 'variance'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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

def make_am_celeba_inpaint():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'am' # am or sm
    model_dict.task = 'inpaint'
    model_dict.sigma = 'variance'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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

def make_am_celeba_superres():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'am' # am or sm
    model_dict.task = 'superres'
    model_dict.sigma = 'variance'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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

def make_am_celeba_torus():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = 12
    model_dict.n_freqs = 4
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'am' # am or sm
    model_dict.task = 'torus'
    model_dict.sigma = 'variance'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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

def make_sm_celeba_diffusion():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'diffusion'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = None
    train_dict.wandbid = None
    
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

def make_sm_celeba_superres():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 3
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'superres'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = None
    train_dict.wandbid = None
    
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

def make_sm_celeba_inpaint():
    model_dict = dotdict()
    model_dict.nf = 64
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 3
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'inpaint'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'celeba', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 48
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 6000
    train_dict.save_every = 3000
    train_dict.first_eval = 0
    train_dict.alpha = None
    train_dict.wandbid = None
    
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

def make_sm_cifar_diffusion():
    model_dict = dotdict()
    model_dict.nf = 128
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'diffusion'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'cifar', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465) true mean
    data_dict.norm_std = (0.5, 0.5, 0.5) # (0.2470, 0.2435, 0.2616) true std
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 5
    train_dict.save_every = 1
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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


def make_sm_cifar_color():
    model_dict = dotdict()
    model_dict.nf = 128
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 3
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'color'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'cifar', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465) true mean
    data_dict.norm_std = (0.5, 0.5, 0.5) # (0.2470, 0.2435, 0.2616) true std
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 5
    train_dict.save_every = 1
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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


def make_sm_cifar_superres():
    model_dict = dotdict()
    model_dict.nf = 128
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 3
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (16,8)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'superres'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'cifar', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 3
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465) true mean
    data_dict.norm_std = (0.5, 0.5, 0.5) # (0.2470, 0.2435, 0.2616) true std
    data_dict.lacedaemon = 5e-2
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 5
    train_dict.save_every = 1
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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


def make_sm_mnist_diffusion():
    model_dict = dotdict()
    model_dict.nf = 32
    model_dict.ch_mult = (1, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 1
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (8,)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'sm' # am or sm
    model_dict.task = 'diffusion'
    model_dict.sigma = 'vpsde'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'mnist', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 1
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5) # (0.1309) true mean
    data_dict.norm_std = (0.5) # (0.2893) true std
    data_dict.lacedaemon = 1e-6
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 5
    train_dict.save_every = 1
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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


def make_am_mnist_diffusion():
    model_dict = dotdict()
    model_dict.nf = 32
    model_dict.ch_mult = (1, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 1
    model_dict.cond_channels = 0
    model_dict.n_phases = None
    model_dict.n_freqs = None
    model_dict.attn_resolutions = (8,)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'am' # am or sm
    model_dict.task = 'diffusion'
    model_dict.sigma = 'variance'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'mnist', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 1
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5) # (0.1309) true mean
    data_dict.norm_std = (0.5) # (0.2893) true std
    data_dict.lacedaemon = 1e-6
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 1e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 5
    train_dict.save_every = 1
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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


def make_am_mnist_torus():
    model_dict = dotdict()
    model_dict.nf = 32
    model_dict.ch_mult = (1, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.num_channels = 1
    model_dict.cond_channels = 0
    model_dict.n_phases = 12
    model_dict.n_freqs = 4
    model_dict.attn_resolutions = (8,)
    model_dict.dropout = 0.1
    model_dict.t0, model_dict.t1 = 0.0, 1.0
    model_dict.resamp_with_conv = True
    model_dict.objective = 'am' # am or sm
    model_dict.task = 'diffusion'
    model_dict.sigma = 'variance'
    model_dict.uniform = False
    model_dict.skip = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = '_'.join([model_dict.objective, 'mnist', model_dict.task])
    model_dict.last_checkpoint = None
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 1
    data_dict.total_batch_size = 128
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5) # (0.1309) true mean
    data_dict.norm_std = (0.5) # (0.2893) true std
    data_dict.lacedaemon = 1e-6
    data_dict.ydim = 10
    
    train_dict = dotdict()
    train_dict.seed = 1
    train_dict.current_step = 0
    train_dict.n_steps = int(3e6)
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 1e-4
    train_dict.betas = (0.9, 0.999)
    train_dict.wd = 0.0
    train_dict.eval_every = 5
    train_dict.save_every = 1
    train_dict.first_eval = 0
    train_dict.alpha = 1e-2
    train_dict.wandbid = None
    
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


job_configs = dotdict()
job_configs.sm_cifar_diffusion = make_sm_cifar_diffusion()
job_configs.sm_cifar_color = make_sm_cifar_color()
job_configs.sm_cifar_superres = make_sm_cifar_superres()

job_configs.sm_mnist_diffusion = make_sm_mnist_diffusion()
job_configs.am_mnist_diffusion = make_am_mnist_diffusion()
job_configs.am_mnist_torus = make_am_mnist_torus()

job_configs.sm_celeba_diffusion = make_sm_celeba_diffusion()
job_configs.sm_celeba_superres = make_sm_celeba_superres()
job_configs.sm_celeba_inpaint = make_sm_celeba_inpaint()
job_configs.am_celeba_diffusion = make_am_celeba_diffusion()
job_configs.am_celeba_inpaint = make_am_celeba_inpaint()
job_configs.am_celeba_superres = make_am_celeba_superres()
job_configs.am_celeba_torus = make_am_celeba_torus()
