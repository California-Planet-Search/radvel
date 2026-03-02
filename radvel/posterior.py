from .likelihood  import Likelihood
import numpy as np
import dill as pickle
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
        self.model = self.likelihood.model
        self.vector = self.likelihood.vector
        self.vector.dict_to_vector()
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
        _logprob=0
        for prior in self.priors:
            _logprob += prior(self.params, self.vector)
        if np.isfinite(_logprob):
            return _logprob + self.likelihood.logprob()
        return _logprob

    def get_prior_dict(self):
        """Prior dictionary
        Returns:
            dict: Dictionary mapping parameters to a list of their priors
        """
        prior_dict = dict()
        for prior in self.priors:
            try:
                param = prior.param
            except AttributeError:
                continue
            if param not in prior_dict:
                prior_dict[param] = [prior]
            else:
                prior_dict[param] += [prior]
        return prior_dict

    def check_proper_priors(self):
        """Checks that the priors are proper for nested sampling.
        Checks that the priors are properly normalized and that there is only one prior per parameter.
        Runs internally before nested sampling.
        """
        vary_params = self.name_vary_params()
        prior_dict = self.get_prior_dict()
        for pname in vary_params:
            prior_list = prior_dict.get(pname, [])
            num_priors = len(prior_list)
            if num_priors == 0:
                raise ValueError("No prior specified for free parameter {}".format(pname))
            else:
                num_prior_transforms = 0
                for prior in prior_list:
                    num_prior_transforms += int(not prior.extra_constraint)
                if num_prior_transforms == 0:
                    raise ValueError(
                        "All priors for free parameter {} are 'extra_constraint' priors. "
                        "Prior transform required for nested sampling.".format(pname)
                    )
                elif num_prior_transforms > 1:
                    raise ValueError(
                        "Multiple prior transforms specified for free parameter {}. "
                        "Not supported for nested sampling. "
                        "Use priors with `extra_constraint=True` to put additional constraints on a parameter".format(pname)
                    )
        for pname in prior_dict:
            for prior in prior_dict[pname]:
                if pname not in vary_params and not prior.extra_constraint:
                    raise ValueError(
                        "Prior transform specified for fixed parameter {}. "
                        "Use `extra_constraint` to constrain fixed or deterministic parameters".format(pname)
                    )


    def prior_transform(self, u, inplace=False):
        """Prior transform for all model parameters
        Takes an array of uniform values between 0 and 1 and converts them to parametre values
        through each parameter's prior transform.

        **Note: If using this outside of RadVel's nested sampling module, make sure to call `check_proper_priors` first!**

        Args:
            u (np.ndarray): Array of uniform values between 0 and 1 for each parameter

        Returns:
            Array of parameter values derived
        """
        if inplace:
            x = u
        else:
            x = np.array(u)
        vary_param_names = self.name_vary_params()
        prior_dict = self.get_prior_dict()
        for ind, pname in enumerate(vary_param_names):
            prior_list = prior_dict[pname]
            for prior in prior_list:
                if not prior.extra_constraint:
                    x[ind] = prior.transform(u[ind])
                    break

        return x

    def extra_likelihood(self):
        """Computes "extra constraint" priors to add them to the likelihood
        This runs internally to add priors such as PositiveK as likelihood constraint
        for nested sampling.
        Called by Posterior.likelihood_ns_array and Posterior.extra_likelihood_array.

        Returns:
            float for the extra priors' contribution to the likelihood
        """
        _logprob = 0
        for prior in self.priors:
            if prior.extra_constraint:
                _logprob += prior(self.params, self.vector, finite=True)
        return _logprob

    def extra_likelihood_array(self, param_values_array):
        """Calls Posterior.extra_likelihood with a vector of parameter values.

        Args:
            param_values_array (np.ndarray): Array of parameter values

        Returns:
            float for the extra priors' contribution to the likelihood
        """
        self.likelihood.set_vary_params(param_values_array, fast=True)
        return self.extra_likelihood()

    def likelihood_ns_array(self, param_values_array):
        """Likelihood of the model, with 'extra prior' constraints applied.

        This is basically a combined call to `self.likelihood.logprob()` and `self.extra_likelihood()`.

        Args:
            param_values_array (np.ndarray): Array of parameter values

        Returns:
            Log probability of the likelihood + extra priors
        """
        self.likelihood.set_vary_params(param_values_array, fast=True)
        extra_likelihood = self.extra_likelihood()
        if np.isfinite(extra_likelihood):
            return extra_likelihood + self.likelihood.logprob()
        # Ultranest requires finite values, so return very large negative instead of -inf
        return -1e100

    def logprob_array(self, param_values_array):
        """Log probability for parameter vector
        Same as `self.logprob`, but will take a vector of
        parameter values. Useful as the objective function
        for routines that optimize a vector of parameter values
        instead of the dictionary-like format of the `radvel.model.Parameters` object.
        Returns:
            float: log probability of the likelihood + priors
        """
        self.likelihood.set_vary_params(param_values_array, fast=True)
        _logprob = self.logprob()
        # if not np.isfinite(_logprob):
        #     raise ValueError("logprob is NaN for the following posterior:\n{}\n{}".format(self.vary_params,
        #                                                                                   self.get_vary_params()))

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
