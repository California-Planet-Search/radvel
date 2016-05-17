import numpy as np

class Likelihood(object):
    """
    Generic Likelihood
    """
    def __init__(self, model, x, y, yerr, extra_params=[]):
        self.model = model
        self.params = model.params

        self.x = np.array(x) # Variables must be arrays.
        self.y = np.array(y) # Pandas data structures lead to problems.
        self.yerr = np.array(yerr)
        self.params.update({}.fromkeys(extra_params, np.nan) ) 
        
        vary = {}.fromkeys(self.params.keys(), True)
        self.vary = vary

    def __repr__(self):
        s = ""
        s +=  "{:<20s}{:>20s} {:>10s}\n".format(
            'parameter', 'value', 'vary'
            )
        keys = self.params.keys()
        keys.sort()
        for key in keys:
            if key in self.vary.keys():
                vstr = str(self.vary[key])
            else:
                vstr = ""
                
            s +=  "{:20s}{:20g} {:>10s}\n".format(
                key, self.params[key], vstr
                 )
            
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

    def bic(self):
        return -2.0 * self.logprob() + len(self.get_vary_params()) + np.log(len(self.y))

    def neglogprob_array(self, params_array):
        return -self.logprob_array(params_array)
        #self.set_vary_params(params_array)
        #return self.neglogprob()

    def logprob_array(self, params_array):
        self.set_vary_params(params_array)
        _logprob = self.logprob()
        return _logprob

class CompositeLikelihood(Likelihood):
    def __init__(self, like_list):
        self.nlike = len(like_list)

        like0 = like_list[0]
        params = like0.params
        self.model = like0.model
        self.x = like0.x
        self.y = like0.y - params[like0.gamma_param]
        self.yerr = np.sqrt(like0.yerr**2 + np.exp(like0.params[like0.logjit_param])**2)
        self.telvec = like0.telvec
        self.extra_params = like0.extra_params
        self.suffixes = like0.suffix
        
        for i in range(1,self.nlike):
            like = like_list[i]
            
            self.x = np.append(self.x,like.x)
            self.y = np.append(self.y, like.y - like.params[like.gamma_param])
            self.yerr = np.append(self.yerr, np.sqrt(like.yerr**2 + np.exp(like.params[like.logjit_param])**2))
            self.telvec = np.append(self.telvec, like.telvec)
            self.extra_params = np.append(self.extra_params, like.extra_params)
            self.suffixes = np.append(self.suffixes, like.suffix)
            
            assert like.model is like0.model, \
                "Likelihoods must use the same model"

            for k in like.params:
                if params.has_key(k):
                    assert like.params[k] is params[k]
                else:
                    params[k] = like.params[k]


        self.extra_params = list(set(self.extra_params))
        self.params = params
        self.vary = {}.fromkeys(params.keys(),True)
        self.like_list = like_list

    def logprob(self):
        _logprob = 0
        for like in self.like_list:
            _logprob += like.logprob()
        return _logprob

    def residuals(self):
        res = self.like_list[0].residuals()
        for like in self.like_list[1:]:
            res = np.append(res,like.residuals())

        return res

class RVLikelihood(Likelihood):
    def __init__(self, model, t, vel, errvel, suffix=''):
        self.gamma_param = 'gamma'+suffix
        self.logjit_param = 'logjit'+suffix

        if suffix.startswith('_'): self.suffix = suffix[1:]
        else: self.suffix = suffix

        self.telvec = np.array([self.suffix]*len(t))
        
        self.extra_params = [self.gamma_param, self.logjit_param]
        super(RVLikelihood, self).__init__(
            model, t, vel, errvel, extra_params=self.extra_params
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


