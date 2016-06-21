from .likelihood  import Likelihood
import copy
import time
import numpy as np

class Posterior(Likelihood):
    """Posterior object

    Posterior object to be sent to the fitting routines.
    It is essentially the same as the Liklihood object,
    but priors are applied here.

    Args:
        likelihood (radvel.likelihood.Likelihood): Likelihood object
        params (radvel.model.RVParameters): parameters object

    Note:
        Append `radvel.prior.Prior` objects to the Posterior.priors list
        to apply priors in the likelihood calculations.
    """
    
    def __init__(self,likelihood):
        self.likelihood = likelihood
        self.params = likelihood.params
        self.vary = likelihood.vary
        self.uparams = likelihood.uparams
        self.priors = []
    
    def __repr__(self):
        s = super(Posterior, self).__repr__()
        s += "Priors\n"
        s += "------\n"
        for prior in self.priors:
            s +=  prior.__repr__() + "\n"
        return s

    def logprob(self):
        """Log probability

        Log-probability for the likelihood given the list
        of priors in `Posterior.priors`.

        Returns:
            float: log probability of the likelihood + priors
        """
            
        _logprob = self.likelihood.logprob()
        for prior in self.priors:
            _logprob += prior( self.params )

        return _logprob

    def logprob_array(self, params_array):
        """Log probability for parameter vector

        Same as `self.logprob`, but will take a vector of
        parameter values. Useful as the objective function
        for routines that optimize a vector of parameter values
        instead of the dictionary-like format of the `radvel.model.RVParameters` object.

        Returns:
            float: log probability of the likelihood + priors

        """
        
        self.likelihood.set_vary_params(params_array)
        _logprob = self.logprob()
                
        return _logprob
