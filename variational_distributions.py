import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn

torch.set_default_dtype(torch.float64)


# Multivariate Normal Variational Distribution, with independent components
class VariationalNormal:
    def __init__(self, size, mu=None, log_s=None):
        # we take log_standard deviation instead of standard deviation. This allows for unconstrained optimisation.
        if mu is None:
            mu = torch.randn(size)
        if log_s is None:
            log_s = torch.randn(size)  # log of the standard deviation
        # Variational parameters
        self.var_params = torch.stack([mu, log_s])  # are always unconstrained
        self.var_params.requires_grad = True

    def dist(self):
        return Normal(self.var_params[0], self.var_params[1].exp())

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_q(self, x):
        return self.dist().log_prob(x).mean(0).sum()
