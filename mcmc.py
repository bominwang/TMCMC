"""
------------------------------------------------------------------------------------------------------------------------
BearGangLab
------------------------------------------------------------------------------------------------------------------------
class of Markov chain Monte Carlo sampling

Author: BoMin Wang

Date: 20240323
------------------------------------------------------------------------------------------------------------------------
"""


class MCMC(object):
    def __init__(self, *args):
        self.log_likelihood = None
        self.log_prior = None

    def __reshape_log_likelihood(self, states):
        pass

    def __reshape_log_prior(self, states):
        pass

    def sampling(self, *args):
        pass

    def get_samples(self, *args):
        pass

    def trajectory(self, *args):
        pass


class PriorFunction(object):
    def __init__(self, *args):
        pass

    def density(self, *args):
        pass

    def log_density(self, *args):
        pass
