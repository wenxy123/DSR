import torch
import numpy as np
import torch.nn as nn
import os
from torch.linalg import vector_norm
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm
import matplotlib.pyplot as plt
import time


### Generate data

# Default parameter vector 'a' if not provided
    # a: 64-dimensional:
    # a[0] = 1.0
    # a[30], a[58] are non-zero
    # all others 0.0

a_true = torch.zeros(64, dtype=torch.float32)
nonzero_indices = [0, 30, 58]
for idx, val in zip(nonzero_indices, [1.0, 1.2, 1.5]):
    a_true[idx] = val


def preanm_simulator_64d(true_function, n, x_lower, x_upper, noise_std,
                         noise_dist, train, device=torch.device("cpu"),
                         a=a_true, noise_corr=0):
    """Data simulator for a pre-additive noise model (pre-ANM) with 64-dimensional covariates.

    Args:
        true_function (str or callable, optional): the true function g*. Defaults to "softplus".
            Choices: ["softplus", "square", "log", "cubic"] or a callable.
        n (int, optional): sample size. Defaults to 10000.
        x_lower (float, optional): lower bound of the training support. Defaults to 0.
        x_upper (float, optional): upper bound of the training support. Defaults to 2.
        noise_std (float, optional): standard deviation of the noise. Defaults to 1.
        noise_dist (str, optional): noise distribution. "gaussian" or "uniform". Defaults to "gaussian".
        train (bool, optional): if True, generates data for training. If False, generates evaluation data.
        device (str or torch.device, optional): device to place tensors. Defaults to CPU.
        a (torch.Tensor, optional): a 64-dim vector for the linear transformation.
            If None, the vector is constructed as follows:
            a[0] = 1.0
            a[2], a[30], a[58] = some non-zero values
            all other elements = 0.0
        noise_corr (float, optional): length-scale parameter for the squared exponential kernel.
            If noise_corr=0, no correlation between pixels.

    Returns:
        For train=True:
            x (torch.Tensor): shape (n,64), input data
            y (torch.Tensor): shape (n,1), output data
        For train=False:
            x_eval (torch.Tensor): shape (n,64), evaluation inputs
            y_eval_med (torch.Tensor): shape (n,1), median (or deterministic) function values
            y_eval_mean (torch.Tensor): shape (n,1), mean output after noise
    """

    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(true_function, str):
        if true_function == "softplus":
            true_function = lambda x: nn.Softplus()(x)
        elif true_function == "square":
            true_function = lambda x: (nn.functional.relu(x)).pow(2)/20
        elif true_function == "log":
            true_function = lambda x: (x/3 + np.log(3) - 2/3)*(x <= 2) + (torch.log(1 + x*(x > 2)))*(x > 2)
        elif true_function == "cubic":
            true_function = lambda x: x.pow(3)/30

    a = a_true.to(device)
    nonzero_indices = [0, 30, 58]  # exactly 10 indices
    nonzero_indices = torch.tensor(nonzero_indices, dtype=torch.long, device=device)

    all_indices = torch.arange(64, device=device)
    mask = torch.ones(64, dtype=torch.bool, device=device)
    mask[nonzero_indices] = False
    zero_indices = all_indices[mask]  # these are the remaining 54 indices

    def pixel_distance(i, j, size=8):
        i_row, i_col = divmod(i, size)
        j_row, j_col = divmod(j, size)
        dist = np.sqrt((i_row - j_row)**2 + (i_col - j_col)**2)
        return dist

    def build_cov_matrix(dim=64, noise_std=1, noise_corr=0):
        cov = torch.zeros(dim, dim, device=device)
        if noise_corr == 0:
            # No correlation, just diagonal
            cov = (noise_std**2)*torch.eye(dim, device=device)
            return cov

        # If noise_corr > 0, use squared exponential kernel
        # cov[i,j] = noise_std^2 * exp(-(distance(i,j)^2)/(2*noise_corr^2))
        for i in range(dim):
            for j in range(dim):
                dist = pixel_distance(i, j, size=8)
                cov[i,j] = (noise_std**2)*np.exp(-(dist**2)/(2*(noise_corr**2)))
        return cov

    def generate_correlated_noise(n_samples, effect=True, noise_dist="gaussian", noise_std=1, noise_corr=0):
        if effect:
            dim=3
            a_effect = a[nonzero_indices]
            cov_matrix_init = build_cov_matrix(dim=3, noise_std=noise_std, noise_corr=noise_corr)
            var_factor = a_effect @ cov_matrix_init @ a_effect
        else:
            dim=61
            cov_matrix_init = build_cov_matrix(dim=61, noise_std=noise_std, noise_corr=noise_corr)
            var_factor = torch.tensor(1)

        cov_matrix = (1/var_factor.item()) * cov_matrix_init ## modify the variance to account for the effect of inner product
        L = torch.linalg.cholesky(cov_matrix)
        z = torch.randn(n_samples, dim, device=device)
        ERR = z @ L.T
        if noise_dist == "gaussian":
            eps = ERR
        else:
            # Transform Gaussian to uniform(-0.5,0.5), then scale
            eps = torch.distributions.Normal(0, 1).cdf(ERR) - 0.5
            eps = eps * np.sqrt(12)
        return eps

    if train:
        x_eff = (x_upper - x_lower) * torch.rand(n, 3, device=device) + x_lower
        x_noneff = (x_upper - x_lower) * torch.rand(n, 61, device=device) + x_lower

        x = torch.empty(n, 64, device=device)
        x[:, nonzero_indices] = x_eff
        x[:, zero_indices] = x_noneff

        #x = (x_upper - x_lower) * torch.rand(n, 64, device=device) + x_lower

        eps_eff = generate_correlated_noise(n, effect=True, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)
        eps_noneff = generate_correlated_noise(n, effect=False, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)

        eps = torch.empty(n, 64, device=device)
        eps[:, nonzero_indices] = eps_eff
        eps[:, zero_indices] = eps_noneff

        xn = x + eps

        s = xn @ a.unsqueeze(1)

        y = true_function(s)

        return x.to(device), y.to(device)

    else:

        x_eval_eff = torch.linspace(x_lower, x_upper, n, device=device).unsqueeze(1).repeat(1, 3)
        x_eval_noneff = torch.zeros(61,device=device)

        x_eval = torch.empty(n, 64, device=device)
        x_eval[:, nonzero_indices] = x_eval_eff
        x_eval[:, zero_indices] = x_eval_noneff

        #x_eval = torch.linspace(x_lower, x_upper, n, device=device).unsqueeze(1).repeat(1, 64)

        s_eval = x_eval @ a.unsqueeze(1)
        y_eval_med = true_function(s_eval)

        gen_sample_size = 10000
        x_rep = torch.repeat_interleave(x_eval, gen_sample_size, dim=0)

        #eps = generate_correlated_noise(x_rep.size(0), 64, noise_dist, noise_std, noise_corr)
        eps_eff = generate_correlated_noise(x_rep.size(0), effect=True, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)
        eps_noneff = generate_correlated_noise(x_rep.size(0), effect=False, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)

        eps = torch.empty(x_rep.size(0), 64, device=device)
        eps[:, nonzero_indices] = eps_eff
        eps[:, zero_indices] = eps_noneff

        xn_rep = x_rep + eps
        s_rep = xn_rep @ a.unsqueeze(1)

        y_rep = true_function(s_rep)
        y_eval_mean = y_rep.view(n, gen_sample_size).mean(dim=1).unsqueeze(1)

        return x_eval.to(device), y_eval_med.to(device), y_eval_mean.to(device)