import torch
from torch.distributions import Normal, Gamma, Binomial
from random_data_generator import masegosa_sampleData, univariate_normal_data_generator
from variational_distributions import VariationalMultivariateNormal, VariationalNormal
import torch
from torch.distributions import Normal, Gamma, MultivariateNormal

import numpy as np
from tqdm import trange

torch.set_default_dtype(torch.float64)


class BayesianLinearRegression:
    def __init__(self, x_train, y_train, x_test, y_test, num_var_samples, loss) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.n = self.x_train.size(0)
        self.m = self.x_train.size(1)

        self.variational_distribution = VariationalMultivariateNormal(dim=self.m)
        self.num_var_samples = num_var_samples
        self.loss = loss

    def log_prob(self, variational_samples):
        return Normal(variational_samples @ self.x_train.t(), 1).log_prob(self.y_train)

    def log_prior(self, variational_samples):
        return (
            MultivariateNormal(torch.zeros(self.m), scale_tril=torch.eye(self.m))
            .log_prob(variational_samples)
            .mean()
        )

    def log_loss(self, variational_samples):
        return -self.log_prob(variational_samples).mean()

    def log_q(self, variational_samples):
        return self.variational_distribution.log_q(variational_samples)

    def likelihood(self, variational_samples):
        return self.log_prob(variational_samples).exp()

    def variance(self, variational_samples, variational_samples_prime):
        log_prob = self.log_prob(variational_samples)
        log_prob_prime = self.log_prob(variational_samples_prime)

        max_ = torch.maximum(log_prob, log_prob_prime) + 0.01

        var_ = torch.exp(2 * log_prob - 2 * max_) - torch.exp(
            log_prob + log_prob_prime - 2 * max_
        )

        return var_.mean()

    def pac_bayes_bound(self):
        variational_samples = self.variational_distribution.rsample(
            self.num_var_samples
        )

        log_loss = self.log_loss(variational_samples)
        log_posterior = self.log_q(variational_samples)
        log_prior = self.log_prior(variational_samples)

        pac_bayes_bound = log_loss + log_posterior / self.n - log_prior / self.n

        return pac_bayes_bound

    def PAC2_variational_bound(self):
        variational_samples = self.variational_distribution.rsample(
            self.num_var_samples
        )
        variational_samples_prime = self.variational_distribution.rsample(
            self.num_var_samples
        )

        log_loss = self.log_loss(variational_samples)
        variational_var = self.variance(variational_samples, variational_samples_prime)
        log_prior = self.log_prior(variational_samples)
        log_posterior = self.log_q(variational_samples)

        pac_variational_bound = (
            log_loss - variational_var + log_posterior / self.n - log_prior / self.n
        )

        return pac_variational_bound

    def negative_test_log_likelihood(self, variational_samples):
        return -(
            Normal(variational_samples @ self.x_test.t(), 1)
            .log_prob(self.y_test)
            .mean()
        )

    def optimise(self):
        optimiser = torch.optim.Adam(
            self.variational_distribution.var_params(),
            lr=0.005,
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
            if self.loss == "PAC BAYES":
                loss = self.pac_bayes_bound()
            else:
                loss = self.PAC2_variational_bound()
            elbo_hist.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Print progress bar
            iters.set_description(
                "PAC Bayes Bound: {}".format(elbo_hist[-1]), refresh=False
            )
        print(
            "theta mean: {}".format(self.variational_distribution.mean.detach().numpy())
        )

        M_aux = torch.tril(self.variational_distribution.M).detach()
        print("theta sd: {}".format(M_aux @ M_aux.t()))

        variational_samples = self.variational_distribution.rsample(
            self.num_var_samples
        )
        print(
            f"Test Log-Likelihood: {self.negative_test_log_likelihood(variational_samples)}"
        )
        # return elbo_hist


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    x_train, y_train = masegosa_sampleData(100, 5)

    x_test, y_test = masegosa_sampleData(10000, 5)

    blr_class = BayesianLinearRegression(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_var_samples=100,
        loss="PAC2 BAYES",
    )
    blr_class.optimise()

# TODO: write a parser for the hyperparameters, something like a grid search.
