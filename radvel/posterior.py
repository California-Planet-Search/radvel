from .likelihood  import Likelihood
import numpy as np
import pickle
import radvel
from radvel.gp import CeleriteKernel

class Posterior(Likelihood):
    """Posterior object

    Posterior object to be sent to the fitting routines.
    It is essentially the same as the Likelihood object,
    but priors are applied here.

    Args:
        likelihood (radvel.likelihood.Likelihood): Likelihood object
        params (radvel.model.Parameters): parameters object

    Note:
        Append `radvel.prior.Prior` objects to the Posterior.priors list
        to apply priors in the likelihood calculations.
    """
    
    def __init__(self,likelihood):
        self.likelihood = likelihood
        self.params = likelihood.params
        self.uparams = likelihood.uparams
        self.priors = []

        self.vparams_order = self.list_vary_params()
    
    def __repr__(self):
        s = super(Posterior, self).__repr__()
        s += "\nPriors\n"
        s += "------\n"
        for prior in self.priors:
            s += prior.__repr__() + "\n"
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

    def logprob_array(self, param_values_array):
        """Log probability for parameter vector

        Same as `self.logprob`, but will take a vector of
        parameter values. Useful as the objective function
        for routines that optimize a vector of parameter values
        instead of the dictionary-like format of the `radvel.model.Parameters` object.

        Returns:
            float: log probability of the likelihood + priors

        """
        self.likelihood.set_vary_params(param_values_array)
        _logprob = self.logprob()
                
        return _logprob

    def writeto(self, filename):
        """
        Save posterior object to pickle file.

        Args:
            filename (string): full path to outputfile
        """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def residuals(self):
        """Overwrite inherited residuals method that does not work"""

        return self.likelihood.residuals()

    def bic(self):
        """Moved to Likelihood.bic"""

        return self.likelihood.bic()

    def aic(self):
        """Moved to Likelihood.aic"""

        raise self.likelihood.aic()


def load(filename):
    """
    Load posterior object from pickle file.

    Args:
        filename (string): full path to pickle file
    """
    with open(filename, 'rb') as f:
        post = pickle.load(f)

    for key,val in post.params.items():
        if val is None:
            del post.params[key]

    return post
