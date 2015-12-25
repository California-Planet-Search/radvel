from .likelihood  import Likelihood

class Posterior(Likelihood):
    def __init__(self,likelihood):
        self.likelihood = likelihood
        self.params = likelihood.params
        self.vary = likelihood.vary
        self.priors = []
    
    def __repr__(self):
        s = super(Posterior, self).__repr__()
        s += "Priors\n"
        s += "------\n"
        for prior in self.priors:
            s +=  prior.__repr__() + "\n"
        return s

    def logprob(self):
        _logprob = self.likelihood.logprob()
        for prior in self.priors:
            _logprob += prior( self.params )

        return _logprob

    def logprob_array(self, params_array):
        self.likelihood.set_vary_params(params_array)
        _logprob = self.logprob()
        return _logprob

import numpy as np
