# Pytorch Implementation of [Action Matching](https://arxiv.org/abs/2210.06662)

## Disclaimer!!!

This repository might contain bugs, non-working code, horrible coding styles, and other side products of the research activity! Use it at your own risk! This repository is a _draft_ version of the official [Action Matching repository](https://github.com/necludov/jam). The only reason this repository is made public is the large number of requests I received asking for the PyTorch version of Action Matching. I won't have time to prepare a clean version of this repository in the foreseeable future. I'm sorry for this. On the other hand, implementing the PyTorch version of Action Matching is very welcomed if someone is looking for an entrance point into studies of generative modelling.

## Animations
### VP-DDPM
The density changes as follows.
$$q_t = \mathcal{N}\bigg(x_t|x_0e^{-\frac{1}{2}\int dt'\;\beta(t')},\mathbf{I}(1-e^{-\int dt'\;\beta(t')})\bigg)$$
The dynamics goes backward in time (from $t=1$ to $t=0$), and follows the antigradient of the potential.

<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/vpddpm.gif" alt="drawing" width="600"/>
<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/mnist_vp.gif" alt="drawing" width="300"/>

### subVP-DDPM
The density changes as follows.
$$q_t = \mathcal{N}\bigg(x_t|x_0e^{-\frac{1}{2}\int dt'\;\beta(t')},\mathbf{I}(1-e^{-\int dt'\;\beta(t')})^2\bigg)$$
The dynamics goes backward in time (from $t=1$ to $t=0$), and follows the antigradient of the potential.

<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/subvpddpm.gif" alt="drawing" width="700"/>

## useful snippets

### launch baselines
```bash
sbatch ./launch_job_v_ddp.sh --dataset cifar --job_config_name sm_cifar_color
sbatch ./launch_job_v_ddp.sh --dataset cifar --job_config_name sm_cifar_diffusion
sbatch ./launch_job_v_ddp.sh --dataset cifar --job_config_name sm_cifar_superres
sbatch ./launch_job_v_ddp.sh --dataset mnist --job_config_name sm_mnist_diffusion
```

### launch AM for mnist
```
sbatch ./launch_job_v_ddp.sh --dataset mnist --job_config_name am_mnist_diffusion
sbatch ./launch_job_v_ddp.sh --dataset mnist --job_config_name am_mnist_torus
```
