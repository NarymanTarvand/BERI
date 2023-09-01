import torch
from torch.distributions import Normal, Gamma, Binomial
from variational_distributions import VariationalNormal
import torch
from torch.distributions import Normal, Gamma

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)


class BayesianLinearRegression:
    def __init__(self, x_data, y_data, theta_samples) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.n = self.x_data.size(0)
        self.m = self.x_data.size(1)
        self.variational_distributions = {
            "beta": VariationalNormal(size=2),
            "sig": VariationalNormal(size=1),
        }
        self.theta_samples = torch.Size([theta_samples])

    def log_prior(self, variational_samples):
        log_prior_beta = 0
        log_prior_log_sigma = 0
        for beta_sample, sig_sample in variational_samples:
            log_prior_beta += Normal(0, 1).log_prob(beta_sample).sum()
            log_prior_log_sigma += Normal(0, 1).log_prob(sig_sample).sum()

        return (log_prior_beta + log_prior_log_sigma) / self.theta_samples[0]

    def log_likelihood(self, variational_samples):
        loglik = []
        for beta_sample, sig_sample in variational_samples:
            loglik.append(
                torch.reshape(
                    Normal(self.x_data.matmul(beta_sample), sig_sample).log_prob(
                        self.y_data
                    ),
                    (-1, 1),
                )
            )
        return torch.cat(loglik, -1)

    def log_q(self, variational_samples):
        log_q_beta = 0
        log_q_sigma = 0

        for beta_sample, sig_sample in variational_samples:
            log_q_beta += self.variational_distributions["beta"].log_q(beta_sample)
            log_q_sigma += self.variational_distributions["sig"].log_q(sig_sample)

        return (log_q_beta + log_q_sigma) / self.theta_samples[0]

    def likelihood(self, variational_samples):
        likelihood = []
        for beta_sample, sig_sample in variational_samples:
            likelihood.append(
                torch.reshape(
                    torch.exp(
                        Normal(self.x_data.matmul(beta_sample), sig_sample).log_prob(
                            self.y_data
                        )
                    ),
                    (-1, 1),
                )
            )

        return torch.cat(likelihood, -1)

    def elbo(self):
        beta_samples = self.variational_distributions["beta"].rsample(
            self.theta_samples
        )
        sig_samples = (
            self.variational_distributions["sig"].rsample(self.theta_samples).exp()
        )
        variational_samples = list(zip(beta_samples, sig_samples))

        log_likelihood = self.log_likelihood(variational_samples).mean()
        log_prior = self.log_prior(variational_samples)
        log_posterior = self.log_q(variational_samples)

        elbo = log_likelihood - log_posterior / self.n + log_prior / self.n

        return elbo

    def elbo_variance(self):
        beta_samples = self.variational_distributions["beta"].rsample(
            self.theta_samples
        )
        sig_samples = (
            self.variational_distributions["sig"].rsample(self.theta_samples).exp()
        )
        variational_samples = list(zip(beta_samples, sig_samples))

        log_likelihood = self.log_likelihood(variational_samples).mean()
        log_prior = self.log_prior(variational_samples)
        log_posterior = self.log_q(variational_samples)

        elbo = log_likelihood - log_posterior / self.n + log_prior / self.n

        likelihood = self.likelihood(variational_samples)

        diff_2 = ((likelihood - likelihood.mean(1).reshape(-1, 1)) ** 2).mean(1)
        max_denom = 2 * (likelihood**2).max(1).values

        var_component = (diff_2 / max_denom).mean()

        return elbo + var_component

    def optimise(self):
        optimizer = torch.optim.Adam(
            [
                self.variational_distributions[key].var_params
                for key in self.variational_distributions
            ],
            lr=0.11,
        )
        elbo_hist = []

        max_iter = 3000
        minibatch_size = 100
        torch.manual_seed(1)

        # This basically creates a progress bar for SGD. max_iter determines the final iteration.
        # mininterval determines how often the progress bar is updated (every 1 second here).
        iters = trange(max_iter, mininterval=1)

        # Stochastic gradient descent
        for t in iters:
            sample_with_replacement = minibatch_size > self.n
            idx = np.random.choice(
                self.n, minibatch_size, replace=sample_with_replacement
            )
            loss = -self.elbo_variance() / self.n
            elbo_hist.append(-loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress bar
            iters.set_description("ELBO: {}".format(elbo_hist[-1]), refresh=False)

        nsamps = 1000
        sig_post = (
            self.variational_distributions["sig"]
            .rsample([nsamps])
            .exp()
            .detach()
            .numpy()
        )
        # print('True beta: {}'.format(beta.detach().numpy()))
        print(
            "beta mean: {}".format(
                self.variational_distributions["beta"].var_params[0].detach().numpy()
            )
        )
        print(
            "beta sd: {}".format(
                self.variational_distributions["beta"]
                .var_params[1]
                .exp()
                .detach()
                .numpy()
            )
        )

        # print('True sigma: {}'.format(sig))
        print("sig mean: {} | sig sd: {}".format(sig_post.mean(), sig_post.std()))


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(0)
    N = 1000
    x = torch.stack([torch.ones(N), torch.randn(N)], -1)
    k = x.shape[1]
    beta = torch.tensor([2.0, -3.0])
    sig = 0.5
    y = Normal(x.matmul(beta), sig).rsample()

    blr_class = BayesianLinearRegression(x_data=x, y_data=y, theta_samples=10)
    blr_class.optimise()
