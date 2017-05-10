import numpy as np

class Likelihood(object):
    """
    Generic Likelihood
    """
    def __init__(self, model, x, y, yerr, extra_params=[], decorr_params=[], decorr_vectors=[]):
        self.model = model
        self.params = model.params

        self.x = np.array(x) # Variables must be arrays.
        self.y = np.array(y) # Pandas data structures lead to problems.
        self.yerr = np.array(yerr)
        self.dvec = [np.array(d) for d in decorr_vectors]
        self.params.update({}.fromkeys(extra_params, np.nan) )
        self.params.update({}.fromkeys(decorr_params, 0.0) )
        self.uparams = None
        
        vary = {}.fromkeys(self.params.keys(), True)
        self.vary = vary

    def __repr__(self):
        s = ""
        if self.uparams is None:
            s +=  "{:<20s}{:>15s}{:>10s}\n".format(
                'parameter', 'value', 'vary'
                )
            keys = self.params.keys()
            #keys.sort()
            for key in keys:
                if key == 'meta':
                    continue
                if key in self.vary.keys():
                    vstr = str(self.vary[key])
                else:
                    vstr = ""
                
                if (key.startswith('tc') or key.startswith('tp')) and self.params[key] > 1e6:
                    par = self.params[key] - 2450000
                else:
                    par = self.params[key]

                s +=  "{:20s}{:15g} {:>10s}\n".format(
                    key, par, vstr
                     )
        else:
            s = ""
            s +=  "{:<20s}{:>15s}{:>10s}{:>10s}\n".format(
                'parameter', 'value', '+/-', 'vary'
                )
            keys = self.params.keys()
            #keys.sort()
            for key in keys:
                if key == 'meta':
                    continue
                if key in self.vary.keys():
                    vstr = str(self.vary[key])
                else:
                    vstr = ""
                if key in self.uparams.keys():
                    err = self.uparams[key]
                else:
                    err = 0
                    
                if (key.startswith('tc') or key.startswith('tp')) and self.params[key] > 1e6:
                    par = self.params[key] - 2450000
                else:
                    par = self.params[key]
                    
                s +=  "{:20s}{:15g}{:10g}{:>10s}\n".format(
                    key, par, err, vstr
                     )

                
        return s

    
    def set_vary_params(self, params_array):
        i = 0
        for key in self.list_vary_params():
            # flip sign for negative jitter
            if key.startswith('jit') and params_array[i] < 0:
                params_array[i] = -params_array[i]
                    
            self.params[key] = params_array[i]
            i+=1

        assert i==len(params_array), \
            "length of array must match number of varied parameters"

    def get_vary_params(self):
        params_array = []
        for key in self.list_vary_params():
            if key != 'meta' and self.vary[key]:
                params_array += [ self.params[key] ]
                
        params_array = np.array(params_array)
        return params_array

    def list_vary_params(self):
        return [key for key in self.params.keys() if key != 'meta'
                    and key in self.vary.keys() and self.vary[key] ]

    def residuals(self):
        return self.y - self.model(self.x) 

    def neglogprob(self):
        return -1.0 * self.logprob()

    def neglogprob_array(self, params_array):
        return -self.logprob_array(params_array)

    def logprob_array(self, params_array):
        self.set_vary_params(params_array)
        _logprob = self.logprob()
        return _logprob

class CompositeLikelihood(Likelihood):
    def __init__(self, like_list):
        """Composite Likelihood

        A thin wrapper to combine multiple `Likelihood`
        objects. One `Likelihood` applies to a dataset from
        a particular instrument.

        Args:
            like_list (list): list of `radvel.likelihood.RVLikelihood` objects
        """
    
        
        self.nlike = len(like_list)

        like0 = like_list[0]
        params = like0.params
        self.model = like0.model
        self.x = like0.x
        self.y = like0.y - params[like0.gamma_param]
        #self.yerr = np.sqrt(like0.yerr**2 + like0.params[like0.jit_param]**2)
        self.yerr = like0.yerr
        self.telvec = like0.telvec
        self.extra_params = like0.extra_params
        #self.decorr_params = like0.decorr_params
        #self.decorr_vectors = like0.decorr_vectors
        self.suffixes = like0.suffix
        self.uparams = like0.uparams
        
        for i in range(1,self.nlike):
            like = like_list[i]
            
            self.x = np.append(self.x,like.x)
            self.y = np.append(self.y, like.y - like.params[like.gamma_param])
            #self.yerr = np.append(self.yerr, np.sqrt(like.yerr**2 + like.params[like.jit_param]**2))
            self.yerr = np.append(self.yerr, like.yerr)
            self.telvec = np.append(self.telvec, like.telvec)
            self.extra_params = np.append(self.extra_params, like.extra_params)
            #self.decorr_params = np.append(self.decorr_params, like.decorr_params)
            #self.decorr_vectors = np.append(self.decorr_vectors, like.decorr_vectors)
            self.suffixes = np.append(self.suffixes, like.suffix)
            try:
                self.uparams = self.uparams.update(like.uparams)
            except AttributeError:
                self.uparams = None
            
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
        """
        See `radvel.likelihood.RVLikelihood.logprob`
        """
        
        _logprob = 0
        for like in self.like_list:
            _logprob += like.logprob()
        return _logprob

    def residuals(self):
        """
        See `radvel.likelihood.RVLikelihood.residuals`
        """

        res = self.like_list[0].residuals()
        for like in self.like_list[1:]:
            res = np.append(res,like.residuals())

        return res

    def errorbars(self):
        """
        See `radvel.likelihood.RVLikelihood.errorbars`
        """
        err = self.like_list[0].errorbars()
        for like in self.like_list[1:]:
            err = np.append(err,like.errorbars())

        return err

        

class RVLikelihood(Likelihood):
    """RV Likelihood

    The Likelihood object for a radial velocity dataset

    Args:
        model (radvel.model.RVModel): RV model object
        t (array): time array
        vel (array): array of velocities
        errvel (array): array of velocity uncertainties
        suffix (string): suffix to identify this Likelihood object
           useful when constructing a `CompositeLikelihood` object.

    """
    
    def __init__(self, model, t, vel, errvel, suffix='',
                     decorr_vars=[], decorr_vectors=[]):
        self.gamma_param = 'gamma'+suffix
        self.jit_param = 'jit'+suffix

        if suffix.startswith('_'):
            self.suffix = suffix[1:]
        else:
            self.suffix = suffix

        self.telvec = np.array([self.suffix]*len(t))
        
        self.extra_params = [self.gamma_param, self.jit_param]
        self.decorr_params = []
        self.decorr_vectors = decorr_vectors
        if len(decorr_vars) > 0:
            self.decorr_params += ['c1_'+d+suffix for d in decorr_vars]
            #self.decorr_params += ['c0_'+d+suffix for d in decorr_vars]

        super(RVLikelihood, self).__init__(
            model, t, vel, errvel, extra_params=self.extra_params,
            decorr_params = self.decorr_params, decorr_vectors=self.decorr_vectors
            )

    def residuals(self):
        """Residuals

        Data minus model
        """

        res = self.y - self.params[self.gamma_param] - self.model(self.x)
        
        if len(self.decorr_params) > 0:
            for parname in self.decorr_params:
                var = parname.split('_')[1]
                pars = []
                for par in self.decorr_params:
                    if var in par:
                        pars.append(self.params[par])
                pars.append(0.0)
                if np.isfinite(self.decorr_vectors[var]).all():
                    vec = self.decorr_vectors[var] - np.mean(self.decorr_vectors[var])
                    p = np.poly1d(pars)
                    res -= p(vec)
                    
        return res

    def errorbars(self):
        """
        Return uncertainties with jitter added
        in quadrature.

        Returns:
            array: uncertainties
        
        """
        return np.sqrt(self.yerr**2 + self.params[self.jit_param]**2)

    def logprob(self):
        """
        Return log-likelihood given the data and model.
        Priors are not applied here.

        Returns:
            float: Natural log of likelihood
        """
        
        sigma_jit = self.params[self.jit_param]
        residuals = self.residuals()
        loglike = loglike_jitter(residuals, self.yerr, sigma_jit)
        
        return loglike

def loglike_jitter(residuals, sigma, sigma_jit):
    """
    Log-likelihood incorporating jitter

    See equation (1) in Howard et al. 2014. Returns loglikelihood, where 
    sigma**2 is replaced by sigma**2 + sigma_jit**2. It penalizes
    excessively large values of jitter
    
    Args:
        residuals (array): array of residuals
        sigma (array): array of measurement errors
        sigma_jit (float): jitter

    Returns:
        float: log-likelihood
    """
    sum_sig_quad = sigma**2 + sigma_jit**2
    penalty = np.sum( np.log( np.sqrt( 2 * np.pi * sum_sig_quad ) ) )
    chi2 = np.sum(residuals**2 / sum_sig_quad)
    loglike = -0.5 * chi2 - penalty
    
    return loglike

