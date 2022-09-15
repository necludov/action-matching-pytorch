# learning-continuity

## VP-DDPM
The density changes as follows.
$$q_t = \mathcal{N}\bigg(x_t|x_0e^{-\frac{1}{2}\int dt'\;\beta(t')},\mathbf{I}(1-e^{-\int dt'\;\beta(t')})\bigg)$$
The dynamics goes backward in time (from $t=1$ to $t=0$), and follows the antigradient of the potential.

<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/vpddpm.gif" alt="drawing" width="600"/>
<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/mnist_vp.gif" alt="drawing" width="300"/>

## subVP-DDPM
The density changes as follows.
$$q_t = \mathcal{N}\bigg(x_t|x_0e^{-\frac{1}{2}\int dt'\;\beta(t')},\mathbf{I}(1-e^{-\int dt'\;\beta(t')})^2\bigg)$$
The dynamics goes backward in time (from $t=1$ to $t=0$), and follows the antigradient of the potential.

<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/subvpddpm.gif" alt="drawing" width="700"/>

## useful snippets

# launch baselines
```bash
sbatch ./launch_job_v_ddp.sh --dataset cifar --job_config_name sm_cifar_color
sbatch ./launch_job_v_ddp.sh --dataset cifar --job_config_name sm_cifar_diffusion
sbatch ./launch_job_v_ddp.sh --dataset cifar --job_config_name sm_cifar_superres
sbatch ./launch_job_v_ddp.sh --dataset mnist --job_config_name sm_mnist_diffusion
```

# launch AM for mnist
```
sbatch ./launch_job_v_ddp.sh --dataset mnist --job_config_name am_mnist_diffusion
sbatch ./launch_job_v_ddp.sh --dataset mnist --job_config_name am_mnist_torus
```
