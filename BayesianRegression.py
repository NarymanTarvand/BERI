import torch
from torch.distributions import Normal, Gamma, Binomial
from random_data_generator import masegosa_sampleData, univariate_normal_data_generator
from variational_distributions import VariationalMultivariateNormal, VariationalNormal
import torch
from torch.distributions import Normal, Gamma

import numpy as np
from tqdm import trange

torch.set_default_dtype(torch.float64)


class BayesianLinearRegression:
    def __init__(self, x_train, y_train, x_test, y_test, theta_samples, loss) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.n = self.x_train.size(0)
        self.m = self.x_train.size(1)

        self.variational_distribution = VariationalMultivariateNormal(size=self.m)
        self.theta_samples = torch.Size([theta_samples])
        self.loss = loss

    def log_prior(self, variational_samples):
        log_prior = 0
        for theta_sample in variational_samples:
            log_prior += Normal(0, 1).log_prob(theta_sample).sum()

        return log_prior / self.theta_samples[0]

    def log_likelihood(self, variational_samples):
        loglik = []
        for theta_sample in variational_samples:
            loglik.append(
                torch.reshape(
                    Normal(
                        self.x_train.matmul(torch.transpose(theta_sample, 0, 1)), 1
                    ).log_prob(self.y_train),
                    (-1, 1),
                )
            )
        return torch.cat(loglik, -1)

    def log_q(self, variational_samples):
        log_q_theta = 0

        for theta_sample in variational_samples:
            log_q_theta += self.variational_distribution.log_q(theta_sample)

        return log_q_theta / self.theta_samples[0]

    def likelihood(self, variational_samples):
        likelihood = []
        for theta_sample in variational_samples:
            likelihood.append(
                torch.reshape(
                    torch.exp(
                        Normal(self.x_train.matmul(theta_sample), 1).log_prob(
                            self.y_train
                        )
                    ),
                    (-1, 1),
                )
            )

        return torch.cat(likelihood, -1)

    def neg_log_likelihood(self, variational_samples):
        likelihood = []
        for theta_sample in variational_samples:
            likelihood.append(
                torch.reshape(
                    Normal(self.x_test.matmul(theta_sample), 1).log_prob(self.y_test),
                    (-1, 1),
                )
            )

        likelihood = torch.cat(likelihood, -1)
        # TODO: CE over the test set - equation (1) in masegosa
        # TODO: replace with: torch.sum(torch.logsumexp(likelihood, -1) - torch.log(torch.tensor(10)))
        return torch.cat(likelihood, -1).mean()

    def elbo(self):
        variational_samples = self.variational_distribution.rsample(self.theta_samples)
        log_likelihood = self.log_likelihood(variational_samples).mean()
        log_prior = self.log_prior(variational_samples)
        log_posterior = self.log_q(variational_samples)

        elbo = log_likelihood + log_posterior / self.n - log_prior / self.n

        return elbo

    def elbo_variance(self):
        variational_samples = self.variational_distribution.rsample(self.theta_samples)

        log_likelihood = self.log_likelihood(variational_samples).mean()
        log_prior = self.log_prior(variational_samples)
        log_posterior = self.log_q(variational_samples)

        elbo = log_likelihood + log_posterior / self.n - log_prior / self.n

        likelihood = self.likelihood(variational_samples)

        diff_2 = ((likelihood - likelihood.mean(1).reshape(-1, 1)) ** 2).mean(1)
        max_denom = 2 * (likelihood**2).max(1).values

        var_component = torch.nan_to_num(diff_2 / max_denom).mean()

        return elbo - var_component

    def optimise(self):
        optimizer = torch.optim.Adam(
            [self.variational_distribution.var_params],
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
            if self.loss == "ELBO":
                loss = -self.elbo()
            else:
                loss = -self.elbo_variance()
            elbo_hist.append(-loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress bar
            iters.set_description("ELBO: {}".format(elbo_hist[-1]), refresh=False)
        # print('True theta: {}'.format(theta.detach().numpy()))
        print(
            "theta mean: {}".format(
                self.variational_distribution.var_params[:1].detach().numpy()
            )
        )
        print(
            "theta sd: {}".format(
                self.variational_distribution.var_params[1:].detach().numpy()
            )
        )

        variational_samples = self.variational_distribution.rsample(self.theta_samples)
        print(
            f"Negative Log-Likelihood: {self.neg_log_likelihood(variational_samples)}"
        )
        # return elbo_hist


if __name__ == "__main__":
    torch.manual_seed(2)
    np.random.seed(1)
    x_test, y_test = univariate_normal_data_generator(100)

    torch.manual_seed(1)
    np.random.seed(0)
    x_train, y_train = univariate_normal_data_generator(1000)

    blr_class = BayesianLinearRegression(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        theta_samples=10,
        loss="ELBO",
    )
    blr_class.optimise()

# TODO: write a parser for the hyperparameters, something like a grid search.
