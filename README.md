# learning-continuity

## Diffusion
The variance changes from $\sigma_0 = 5$ to $\sigma_1 = 1$ linearly in time $t\in[0,1]$. The dynamics goes forward in time and follows the gradient of the potential.

<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/diffusion.gif" alt="drawing" width="700"/>

## VP-DDPM
The density changes as follows.
$$q_t = \mathcal{N}\bigg(x_t|x_0e^{-\frac{1}{2}\int dt'\;\beta(t')},\mathbf{I}(1-e^{-\int dt'\;\beta(t')})\bigg)$$
The dynamics goes backward in time (from $t=1$ to $t=0$), and follows the antigradient of the potential.

<img src="https://github.com/necludov/learning-continuity/blob/main/notebooks/gifs/vpddpm.gif" alt="drawing" width="700"/>
