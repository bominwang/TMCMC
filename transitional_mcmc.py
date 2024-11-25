"""
------------------------------------------------------------------------------------------------------------------------
BearGangLab
------------------------------------------------------------------------------------------------------------------------
Transitional Markov chain Monte Carlo (a.k.a Sequential Monte carlo)

from:
'J. Ching and Y.-C. Chen. Transitional Markov Chain Monte Carlo Method for Bayesian
Model Updating, Model Class Selection, and Model Averaging. Journal of Engineering
Mechanics, 133(7):816-832, 2007'

Author: BoMin Wang

Date: 20240324
------------------------------------------------------------------------------------------------------------------------
The annealing parameter is used to control transition of T-MCMC
from:
'Konstantin M. Zuev and James L. Beck. Asymptotically Independent Markov Sampling:
A New MCMC Scheme for Bayesian Inference, pages 2022-2031'
------------------------------------------------------------------------------------------------------------------------
C.O.V of weights = std(weights) / mean(weights)
The annealing parameter is choice when the C.O.V = 1.
Dichotomy method:
1. define the boundary [min, max]
2. calculate the mid by 0.5 * (min + max)
3. if the mid is target return mid
4. if mid < target min = mid
5. if mid > target max = mid
6. if min > max return false
------------------------------------------------------------------------------------------------------------------------
compute the weights by annealing parameter:
weights = likelihood^(delta)
        = exp(ln(likelihood^(delta)))
        = exp(delta * ln(likelihood))
------------------------------------------------------------------------------------------------------------------------
Resample based on multinomial distribution
------------------------------------------------------------------------------------------------------------------------
"""
import torch
from TMCMC_ALBNN.mcmc.mcmc import MCMC
import matplotlib.pyplot as plt


# verified
def plausible_weights_calculator(log_likelihood: torch.Tensor, delta: float) -> torch.Tensor:
    """
    :param log_likelihood: likelihood of current samples
    :param delta:  current annealing parameter - old annealing parameter
    :return: weights
    """
    size, _ = log_likelihood.shape
    weights = torch.exp(delta * (log_likelihood - log_likelihood.max())).reshape(size, 1)
    weights = weights / torch.sum(weights)
    return weights


# verified
def annealing_parameter_calculator(lll: torch.Tensor, current_annealing_parameter: float) -> float:
    """
    :param lll: log_likelihood
    :param current_annealing_parameter: annealing parameter of the current stage
    :return: annealing parameter of the next stage and delta
    """
    next_annealing_parameter = None
    min_annealing_parameter = current_annealing_parameter
    max_annealing_parameter = 2.0
    while max_annealing_parameter - min_annealing_parameter > 1e-8:
        next_annealing_parameter = 0.5 * (min_annealing_parameter + max_annealing_parameter)
        delta = next_annealing_parameter - current_annealing_parameter
        weights = plausible_weights_calculator(lll, delta)
        cv = torch.std(weights) / torch.mean(weights)
        if cv > 1:
            max_annealing_parameter = next_annealing_parameter
        else:
            min_annealing_parameter = next_annealing_parameter

    return min(next_annealing_parameter, 1.0)


def resampler(samples: torch.Tensor, weights: torch.Tensor, size: int) -> tuple:
    """
    :param samples: current samples
    :param weights: normalized weights
    :param size: number of resample
    :return: resampled samples
    """
    indices = torch.multinomial(weights.view(-1), size, replacement=True)
    return samples[indices], weights[indices]


def covariance_calculator(samples, weights, beta):
    size, dim = samples.shape
    scaled_factor = beta ** 2
    weighted_samples = weights * samples
    mean_samples = torch.sum(weighted_samples, dim=0, keepdim=True)

    difference = torch.t(samples - mean_samples)
    covariance = torch.zeros(size=[dim, dim])
    for i in range(size):
        cov = torch.mm(difference[:, i].reshape(dim, 1), difference[:, i].reshape(1, dim))
        covariance += cov * weights[i]
    covariance = covariance * scaled_factor
    return covariance


def covariance_metrix_calculator(weights, samples, scaling_factor):
    scaling_factor = scaling_factor ** 2
    size, dim = samples.shape
    weights = weights.reshape(size, 1)
    weighted_samples = weights * samples
    mu = torch.sum(weighted_samples, dim=0, keepdim=True)
    diff = samples - mu
    weighted_diff = weights * diff
    cov = scaling_factor * torch.mm(weighted_diff.t(), diff)
    return cov


def metropolis_hastings(likelihood_function: callable, prior_function: callable,
                        annealing_parameter: float,
                        current_state: torch.Tensor, covariance: torch.Tensor,
                        num_steps: int):
    size, dim = current_state.shape
    # trajectory = current_state
    proposal_samples = torch.zeros_like(current_state)
    acceptance_rate = 0.0
    print('Running MH algorithm phase:')
    for i in range(num_steps):
        proposal_distribution = torch.distributions.MultivariateNormal(loc=current_state,
                                                                       covariance_matrix=covariance)
        proposal_samples = proposal_distribution.sample()
        # for j in range(size):
        #     proposal_distribution = torch.distributions.MultivariateNormal(loc=current_state[j, :],
        #                                                                    covariance_matrix=covariance)
        #     proposal_samples[j, :] = proposal_distribution.sample()

        alpha = annealing_parameter * likelihood_function(proposal_samples) + prior_function(proposal_samples)
        alpha = alpha - annealing_parameter * likelihood_function(current_state) - prior_function(current_state)
        u = torch.rand(size=[size, 1])
        condition = (alpha > torch.log(u))
        index = torch.where(condition)[0]
        acceptance_rate += len(index) / size
        current_state[index, :] = proposal_samples[index, :]
        progress = int((i + 1) / num_steps * 20)
        print("\r[{}{}] {:.2f}%".format("=" * progress, " " * (20 - progress), (i + 1) / num_steps * 100),
              end="", flush=True)
    print()
    # trajectory = torch.vstack([trajectory, current_state])
    return current_state, acceptance_rate / num_steps


def dynamic_scale_factor(acceptance_rate: float) -> float:
    return 1 / 9 + (acceptance_rate * 8 / 9)


class TransitionalMetropolisHastings(MCMC):
    def __init__(self):
        super().__init__()

        self.log_likelihood = None
        self.log_prior = None
        self.size = None
        self.dim = None
        self.annealing_param = 0.0
        self.beta = 0.2
        self.num_stage = 0
        self.samples = None
        self.acceptance_rate = 0.0
        self.history_annealing_param = []

    def __reshape_log_likelihood(self, states):
        log_likelihood = self.log_likelihood(states).reshape(self.size, 1)
        return log_likelihood

    def __reshape_log_prior(self, states):
        log_prior = self.log_prior(states).reshape(self.size, 1)
        return log_prior

    def sampling(self, log_likelihood: callable, log_prior: callable,
                 init_states: torch.Tensor, steps_mh: int = 20):
        print("Ensure that the initial state size is number of chains × state dimension.")
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.size, self.dim = init_states.shape

        self.history_annealing_param.append(self.annealing_param)
        self.samples = init_states.unsqueeze(0)

        current_states = init_states
        # perform the annealing sample
        while self.annealing_param < 1.0:
            self.num_stage += 1
            # calculate the likelihood
            lll = self.__reshape_log_likelihood(current_states)
            annealing_param = annealing_parameter_calculator(lll=lll,
                                                             current_annealing_parameter=self.annealing_param)
            # store the annealing parameter
            self.history_annealing_param.append(annealing_param)
            delta = annealing_param - self.annealing_param
            self.annealing_param = annealing_param
            print(f' T-MCMC: Iteration {self.num_stage}, tempering parameter = {self.annealing_param}')

            # calculate the weights
            weights = plausible_weights_calculator(log_likelihood=lll,
                                                   delta=delta)
            # resample
            current_states, weights = resampler(samples=current_states,
                                                weights=weights,
                                                size=self.size)
            # calculate covariance of proposal distribution
            covariance = covariance_calculator(samples=current_states,
                                               weights=weights,
                                               beta=self.beta)
            # metropolis hastings sampling
            current_states, self.acceptance_rate = metropolis_hastings(
                likelihood_function=self.__reshape_log_likelihood,
                prior_function=self.__reshape_log_prior,
                annealing_parameter=self.annealing_param,
                current_state=current_states,
                covariance=covariance,
                num_steps=steps_mh)
            self.samples = torch.cat([self.samples, current_states.unsqueeze(0)], dim=0)
        return self.samples

    def get_samples(self):
        return self.samples[-1, :, :]

    def trajectory(self, index_dim):
        pass
