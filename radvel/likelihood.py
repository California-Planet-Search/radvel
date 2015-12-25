import numpy as np

class Likelihood(object):
    """
    Generic Likelihood
    """
    def __init__(self, model, x, y, yerr, extra_params=[]):
        self.model = model
        self.params = model.params
        self.x = x
        self.y = y
        self.yerr = yerr
        self.params = model.params
        for key in extra_params:
            self.params[key] = None

        vary = {}
        for key in self.params.keys():
            vary[key] = True
        self.vary = vary
        
    def __repr__(self):
        s = ""
        s +=  "{:<10s}{:>20s} {:>10s}\n".format(
            'parameter', 'value', 'vary'
            )
        keys = self.params.keys()
        keys.sort()
        for key in keys:
            s +=  "{:10s}{:20g} {:>10s}\n".format(
                key, self.params[key], str(self.vary[key])
                 )
#        s+="{}".format(self.logprob())
        return s

    def set_vary_params(self, params_array):
        i = 0
        for key in self.params.keys():
            if self.vary[key]:
                self.params[key] = params_array[i]
                i+=1
        assert i==len(params_array), \
            "length of array must match number of varied parameters"

    def get_vary_params(self):
        params_array = []
        for key in self.params.keys():
            if self.vary[key]:
                params_array += [ self.params[key] ] 
        params_array = np.array(params_array)
        return params_array

    def list_vary_params(self):
        return [key for key in self.params if self.vary[key] ]

    def residuals(self):
        return self.y - self.model(self.x) 

    def neglogprob(self):
        return -1.0 * self.logprob()

    def neglogprob_array(self, params_array):
        self.set_vary_params(params_array)
        return self.neglogprob()

    def logprob_array(self, params_array):
        self.set_vary_params(params_array)
        _logprob = self.logprob()
        return _logprob

class RVLikelihood(Likelihood):
    def __init__(self, model, t, vel, errvel, suffix=''):
        self.gamma_param = 'gamma'+suffix
        self.logjit_param = 'logjit'+suffix
        extra_params = [self.gamma_param, self.logjit_param]
        super(RVLikelihood, self).__init__(
            model, t, vel, errvel, extra_params=extra_params
            )

    def residuals(self):
        return self.y - self.params[self.gamma_param] - self.model(self.x)

    def logprob(self):
        """
        Return log-likelihood given

        Returns
        -------
        loglike : Natural log of likelihood
        """
        sigma_jit = np.exp( self.params[self.logjit_param] )
        residuals = self.residuals()
        loglike = loglike_jitter(residuals, self.yerr, sigma_jit)
        
        return loglike

def loglike_jitter(residuals, sigma, sigma_jit):
    """
    Log-likelihood incorporating jitter

    See equation (1) in Howard et al. 2014. Returns loglikelihood, where 
    sigma**2 is replaced by sigma**2 + sigma_jit**2. It penalizes
    excessively large values of jitter
    
    Parameters
    ----------
    residuals : array
    sigma : float "measurement errors"
    sigma_jit : float "jitter"
    """

    sum_sig_quad = sigma**2 + sigma_jit**2
    penalty = np.sum( np.log( np.sqrt( 2 * np.pi * sum_sig_quad ) ) )
    chi2 = np.sum(residuals**2 / sum_sig_quad)
    loglike = -0.5 * chi2 - penalty 
    return loglike


