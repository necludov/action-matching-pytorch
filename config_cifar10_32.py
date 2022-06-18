from utils import dotdict

def get_configs():
    model_dict = dotdict()
    model_dict.nf = 128
    model_dict.ch_mult = (1, 2, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.attn_resolutions = (16,)
    model_dict.dropout = 0.1
    model_dict.resamp_with_conv = True
    model_dict.conditional = True
    model_dict.nonlinearity = 'swish'
    model_dict.savepath = 'am_cifar_vpsde'
    model_dict.s = 'generic'
    model_dict.w = w1
    model_dict.dwdt = dw1dt
    model_dict.q_t = vpsde
    
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 3
    data_dict.centered = True
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5, 0.5, 0.5)
    data_dict.norm_std = (0.5, 0.5, 0.5)
    train_dict = dotdict()
    train_dict.grad_clip = 1.0
    train_dict.warmup = 0
    train_dict.lr = 1e-4
    config_dict = dotdict()
    config_dict.model = model_dict
    config_dict.data = data_dict
    config_dict.train = train_dict
    return config_dict