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
    def __init__(self, x_data, y_data) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.n = self.x_data.size(0)
        self.m = self.x_data.size(1)
        self.variational_distributions = {
            "beta": VariationalNormal(size=2),
            "sig": VariationalNormal(size=1),
        }

    def log_prior(self, variational_samples):
        log_prior_beta = Normal(0, 1).log_prob(variational_samples["beta"]).sum()
        log_prior_log_sigma = Normal(0, 1).log_prob(variational_samples["sig"]).sum()

        return log_prior_beta + log_prior_log_sigma

    def loglike(self, variational_samples):
        beta = variational_samples["beta"]
        sig = variational_samples["sig"]
        return Normal(self.x_data.matmul(beta), sig).log_prob(self.y_data).sum()

    def log_q(self, variational_samples):
        # out = 0.0
        # for key in self.variational_distributions:
        #     out += self.variational_distributions[key].log_q(variational_samples[key])
        log_q_beta = self.variational_distributions["beta"].log_q(
            variational_samples["beta"]
        )
        log_q_sigma = self.variational_distributions["sig"].log_q(
            variational_samples["sig"]
        )
        return log_q_beta + log_q_sigma

    def elbo(self):
        variational_samples = {}
        for key in self.variational_distributions:
            variational_samples[key] = self.variational_distributions[key].rsample()

        variational_samples = {
            "beta": variational_samples["beta"],
            "sig": variational_samples["sig"].exp(),
        }

        out = self.loglike(variational_samples)
        out += self.log_prior(variational_samples)
        out -= self.log_q(variational_samples)

        return out

    def optimise(self):
        optimizer = torch.optim.Adam(
            [
                self.variational_distributions[key].var_params
                for key in self.variational_distributions
            ],
            lr=0.1,
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
            loss = -self.elbo() / self.n
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
    # Generate data
    # set random number generator seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(0)
    N = 1000
    x = torch.stack([torch.ones(N), torch.randn(N)], -1)
    k = x.shape[1]
    beta = torch.tensor([2.0, -3.0])
    sig = 0.5
    y = Normal(x.matmul(beta), sig).rsample()

    blr_class = BayesianLinearRegression(x_data=x, y_data=y)
    blr_class.optimise()
