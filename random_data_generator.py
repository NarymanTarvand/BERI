from torch.distributions import Normal
import torch
import numpy as np


def univariate_normal_data_generator(n_samples: int):
    x = torch.stack([torch.ones(n_samples), torch.randn(n_samples)], -1)
    beta = torch.tensor([2.0, -3.0])
    sig = 2
    y = Normal(x.matmul(beta), sig).rsample()

    return x, y


def masegosa_sampleData(samples, variance):
    x = np.linspace(-10.5, 10.5, samples).reshape(-1, 1)
    r = 1 + np.float32(np.random.normal(size=(samples, 1), scale=variance))
    y = np.float32(x * 1.0 + r * 1.0)

    x = np.transpose(x)[0]
    y = np.transpose(y)[0]
    x = torch.stack([torch.ones(samples), torch.tensor(x)], -1)
    y = torch.tensor(y)

    return x, y


def SinusoidalData(samples, variance=10):
    x = np.linspace(-10.5, 10.5, samples).reshape(-1, 1)
    r = 1 + np.float32(np.random.normal(size=(samples, 1), scale=variance))
    y = np.float32(np.sin(0.75 * x) * 7.0 + x * 0.5 + r * 1.0)

    x = torch.tensor(x)
    # y = np.transpose(y)[0]
    y = torch.tensor(y)

    return x, y
