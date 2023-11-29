import torch
from torch.distributions import Normal, Gamma, Binomial, MultivariateNormal

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
        return Normal(self.var_params[0], 1)

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_q(self, x):
        return self.dist().log_prob(x).mean(0).sum()


# Multivariate Normal Variational Distribution, with dependent components
class VariationalMultivariateNormal:
    def __init__(self, size):
        self.size = size

        mean = torch.zeros(1, size)
        cov_matrix = torch.eye(size)
        self.var_params = torch.cat([mean, cov_matrix])
        self.var_params.requires_grad = True

    def dist(self):
        return MultivariateNormal(self.var_params[:1], scale_tril=self.var_params[1:])

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_q(self, x):
        return self.dist().log_prob(x).mean(0).sum()


# Normal Variational Distribution, with independent components used in bayesian NN
class VariationalNormalNN:
    def __init__(self, num_hidden):
        # we take log_standard deviation instead of standard deviation. This allows for unconstrained optimisation.
        self.in_weights_mean = torch.normal(
            0, 0.05, size=(1, num_hidden)
        ).requires_grad_(True)
        self.in_bias_mean = torch.normal(0, 0.05, size=(1, num_hidden)).requires_grad_(
            True
        )
        self.in_weights_stdev = torch.nn.Softplus(
            torch.normal(-10, 0.05, size=(1, num_hidden))
        ).beta.requires_grad_(True)
        self.in_bias_stdev = torch.nn.Softplus(
            torch.normal(-10, 0.05, size=(1, num_hidden))
        ).beta.requires_grad_(True)

        self.out_weights_mean = torch.normal(
            0, 0.05, size=(num_hidden, 1)
        ).requires_grad_(True)
        self.out_bias_mean = torch.normal(0, 0.05, size=(1, 1)).requires_grad_(True)
        self.out_weights_stdev = torch.nn.Softplus(
            torch.normal(-10, 0.05, size=(num_hidden, 1))
        ).beta.requires_grad_(True)
        self.out_bias_stdev = torch.nn.Softplus(
            torch.normal(-10, 0.05, size=(1, 1))
        ).beta.requires_grad_(True)

    def rsample(self, n=torch.Size([])):
        return (
            Normal(self.in_weights_mean, torch.exp(self.in_weights_stdev)).rsample(n),
            Normal(self.in_bias_mean, torch.exp(self.in_bias_stdev)).rsample(n),
            Normal(self.out_weights_mean, torch.exp(self.out_weights_stdev)).rsample(n),
            Normal(self.out_bias_mean, torch.exp(self.out_bias_stdev)).rsample(n),
        )

    def log_q(
        self, in_weights_sample, in_bias_sample, out_weights_sample, out_bias_sample
    ):
        return (
            Normal(self.in_weights_mean, torch.exp(self.in_weights_stdev))
            .log_prob(in_weights_sample)
            .mean(0)
            .sum(),
            Normal(self.in_bias_mean, torch.exp(self.in_bias_stdev))
            .log_prob(in_bias_sample)
            .mean(0)
            .sum(),
            Normal(self.out_weights_mean, torch.exp(self.out_weights_stdev))
            .log_prob(out_weights_sample)
            .mean(0)
            .sum(),
            Normal(self.out_bias_mean, torch.exp(self.out_bias_stdev))
            .log_prob(out_bias_sample)
            .mean(0)
            .sum(),
        )
