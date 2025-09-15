import torch
import numpy as np
import torch.nn as nn
import os
from torch.linalg import vector_norm
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## data generator

def preanm_simulator(true_function, n, x_lower, x_upper, noise_std, noise_dist, train=True, device=device, a=torch.tensor([1.0, 1.2, 1.5]),noise_corr=0):
    """Data simulator for a pre-additive noise model (pre-ANM) with 3-dimensional covariates and noise.

    Args:
        true_function (str, optional): true function g*. Defaults to "softplus". Choices: ["softplus", "square", "log"].
        n (int, optional): sample size. Defaults to 10000.
        x_lower (int, optional): lower bound of the training support. Defaults to 0.
        x_upper (int, optional): upper bound of the training support. Defaults to 2.
        noise_std (int, optional): standard deviation of the noise. Defaults to 1.
        noise_dist (str, optional): noise distribution. Defaults to "gaussian". Choices: ["gaussian", "uniform"].
        train (bool, optional): generate data for training. Defaults to True.
        device (str or torch.device, optional): device. Defaults to torch.device("cpu").
        a (torch.Tensor, optional): a linear vector to transform input. Defaults to torch.tensor([1,0.4,0.3]).
        noise_corr (float, optional): pairwise correlation between noise components. Defaults to 0.

    Returns:
        tuple of torch.Tensors: data simulated from a pre-ANM.
    """

    if isinstance(true_function, str):
        if true_function == "softplus":
            true_function = lambda x: nn.Softplus()(x)
        elif true_function == "square":
            true_function = lambda x: (nn.functional.relu(x)).pow(2)/7.4
        elif true_function == "log":
            true_function = lambda x: (x/3 + np.log(3) - 2/3)*(x <= 2) + (torch.log(1 + x*(x > 2)))*(x > 2)
        elif true_function == "cubic":
            true_function = lambda x: x.pow(3)/11.1

    if isinstance(device, str):
        device = torch.device(device)


    if a is None:
        a = torch.ones(3)
    #a = a.to(device)

    def generate_correlated_noise(n_samples, dim, noise_dist, noise_std, noise_corr):
        cov_matrix = (1/(2.45 + 4.84*noise_corr)) * noise_std**2 * ((1 - noise_corr) * torch.eye(dim) + noise_corr * torch.ones(dim, dim))
        L = torch.linalg.cholesky(cov_matrix)
        z = torch.randn(n_samples, dim)
        ERR = z @ L.T
        if noise_dist == "gaussian":
            eps = ERR
        else:
            eps = torch.distributions.Normal(0, 1).cdf(ERR) - 0.5 ## transfer such that the noise is distributed as Unif(-0.5,0.5)
            eps = eps * np.sqrt(12)
        return eps

    if train:
        x = torch.rand(n, 3)*(x_upper - x_lower) + x_lower

        # Generate 3-dimensional noise 'eps'
        if noise_dist == "gaussian":
            eps = generate_correlated_noise(n, 3, "gaussian", noise_std, noise_corr)
        else:
            assert noise_dist == "uniform"
            eps = generate_correlated_noise(n, 3, "uniform", noise_std, noise_corr)

        xn = x + eps

        s = xn @ a.unsqueeze(1)

        y = true_function(s)

        return x.to(device), y.to(device)

    else:

        x_eval = torch.linspace(x_lower, x_upper, n).unsqueeze(1).repeat(1, 3)

        s_eval = x_eval @ a.unsqueeze(1)

        y_eval_med = true_function(s_eval)

        gen_sample_size = 10000

        x_rep = torch.repeat_interleave(x_eval, gen_sample_size, dim=0)

        if noise_dist == "gaussian":
            eps = generate_correlated_noise(x_rep.size(0), 3, "gaussian", noise_std, noise_corr)
        else:
            assert noise_dist == "uniform"
            eps = generate_correlated_noise(x_rep.size(0), 3, "uniform", noise_std, noise_corr)

        xn_rep = x_rep + eps
        s_rep = xn_rep @ a.unsqueeze(1)

        y_rep = true_function(s_rep)
        y_eval_mean = y_rep.view(n, gen_sample_size).mean(dim=1).unsqueeze(1)

        return x_eval.to(device), y_eval_med.to(device), y_eval_mean.to(device)