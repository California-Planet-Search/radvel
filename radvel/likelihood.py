import numpy as np
import radvel.model
from radvel import gp
from scipy.linalg import cho_factor, cho_solve
from scipy import matrix

_has_celerite = gp._try_celerite()
if _has_celerite:
    import celerite

class Likelihood(object):
    """
    Generic Likelihood
    """
    def __init__(self, model, x, y, yerr, extra_params=[], decorr_params=[],
                 decorr_vectors=[]):
        self.model = model
        self.params = model.params
        self.x = np.array(x)  # Variables must be arrays.
        self.y = np.array(y)  # Pandas data structures lead to problems.
        self.yerr = np.array(yerr)
        self.dvec = [np.array(d) for d in decorr_vectors]
        for key in extra_params:
            self.params[key] = radvel.model.Parameter(value=np.nan)
        for key in decorr_params:
            self.params[key] = radvel.model.Parameter(value=0.0)
        self.uparams = None

        self.params_order = self.list_vary_params()

    def __repr__(self):
        s = ""
        if self.uparams is None:
            s += "{:<20s}{:>15s}{:>10s}\n".format(
                'parameter', 'value', 'vary'
                )
            keys = self.params.keys()
            for key in keys:
                vstr = str(self.params[key].vary)    
                if (key.startswith('tc') or key.startswith('tp')) and self.params[key].value > 1e6:
                    par = self.params[key].value - 2450000
                else:
                    par = self.params[key].value

                s += "{:20s}{:15g} {:>10s}\n".format(
                    key, par, vstr
                     )
        else:
            s = ""
            s += "{:<20s}{:>15s}{:>10s}{:>10s}\n".format(
                'parameter', 'value', '+/-', 'vary'
                )
            keys = self.params.keys()
            for key in keys:
                vstr = str(self.params[key].vary)
                if key in self.uparams.keys():
                    err = self.uparams[key]
                else:
                    err = 0
                if (key.startswith('tc') or key.startswith('tp')) and \
                        self.params[key].value > 1e6:
                    par = self.params[key].value - 2450000
                else:
                    par = self.params[key].value
                    
                s += "{:20s}{:15g}{:10g}{:>10s}\n".format(
                    key, par, err, vstr
                     )
        return s

    def set_vary_params(self, param_values_array):
        i = 0
        for key in self.list_vary_params():
            # flip sign for negative jitter
            if key.startswith('jit') and param_values_array[i] < 0:
                param_values_array[i] = -param_values_array[i]
            self.params[key].value = param_values_array[i]
            i += 1
        assert i == len(param_values_array), \
            "Length of array must match number of varied parameters"

    def get_vary_params(self):
        params_array = []
        for key in self.list_vary_params():
            if self.params[key].vary:
                params_array += [self.params[key].value]     
        params_array = np.array(params_array)
        return params_array

    def list_vary_params(self):
        keys = self.list_params()

        return [key for key in keys if self.params[key].vary]

    def list_params(self):
        try:
            keys = self.params_order
        except AttributeError:
            keys = list(self.params.keys())
            self.params_order = keys
        return keys

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

    def bic(self):
        """
        Calculate the Bayesian information criterion

        Returns:
            float: BIC
        """

        n = len(self.y)
        k = len(self.get_vary_params())
        _bic = np.log(n) * k - 2.0 * self.logprob()
        return _bic

    def aic(self):
        """
        Calculate the Aikike information criterion
        The Small Sample AIC (AICC) is returned because for most RV data sets n < 40 * k
        (see Burnham & Anderson 2002 S2.4).

        Returns:
            float: AICC
        """

        n = len(self.y)
        k = len(self.get_vary_params())
        aic = - 2.0 * self.logprob() + 2.0 * k
        # Small sample correction
        _aicc = aic
        denom = (n - k - 1.0)
        if denom > 0:
            _aicc += (2.0 * k * (k + 1.0)) / denom
        else:
            print("Warning: The number of free parameters is greater than or equal to")
            print("         the number of data points. The AICc comparison calculations")
            print("         will fail in this case.")
            _aicc = np.inf
        return _aicc


class CompositeLikelihood(Likelihood):
    """Composite Likelihood

    A thin wrapper to combine multiple `Likelihood`
    objects. One `Likelihood` applies to a dataset from
    a particular instrument.

    Args:
        like_list (list): list of `radvel.likelihood.RVLikelihood` objects
    """
    def __init__(self, like_list):
        self.nlike = len(like_list)

        like0 = like_list[0]
        params = like0.params
        self.model = like0.model
        self.x = like0.x
        self.y = like0.y
        self.yerr = like0.yerr
        self.telvec = like0.telvec
        self.extra_params = like0.extra_params
        self.suffixes = like0.suffix
        self.uparams = like0.uparams
        
        for i in range(1, self.nlike):
            like = like_list[i]
            
            self.x = np.append(self.x, like.x)
            self.y = np.append(self.y, like.y - like.params[like.gamma_param].value)
            self.yerr = np.append(self.yerr, like.yerr)
            self.telvec = np.append(self.telvec, like.telvec)
            self.extra_params = np.append(self.extra_params, like.extra_params)
            self.suffixes = np.append(self.suffixes, like.suffix)
            try:
                self.uparams = self.uparams.update(like.uparams)
            except AttributeError:
                self.uparams = None
            
            assert like.model is like0.model, \
                "Likelihoods must use the same model"

            for k in like.params:
                if k in params:
                    assert like.params[k]._equals(params[k]), "Name={} {} != {}".format(k, like.params[k], params[k])
                else:
                    params[k] = like.params[k]

        self.extra_params = list(set(self.extra_params))
        self.params = params
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
            res = np.append(res, like.residuals())

        return res
        
    def errorbars(self):
        """
        See `radvel.likelihood.RVLikelihood.errorbars`
        """
        err = self.like_list[0].errorbars()
        for like in self.like_list[1:]:
            err = np.append(err, like.errorbars())

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
    def __init__(self, model, t, vel, errvel, suffix='', decorr_vars=[],
                 decorr_vectors=[], **kwargs):
        self.gamma_param = 'gamma'+suffix
        self.jit_param = 'jit'+suffix
        self.extra_params = [self.gamma_param, self.jit_param]

        if suffix.startswith('_'):
            self.suffix = suffix[1:]
        else:
            self.suffix = suffix

        self.telvec = np.array([self.suffix]*len(t))
        
        self.decorr_params = []
        self.decorr_vectors = decorr_vectors
        if len(decorr_vars) > 0:
            self.decorr_params += ['c1_'+d+suffix for d in decorr_vars]

        super(RVLikelihood, self).__init__(
            model, t, vel, errvel, extra_params=self.extra_params,
            decorr_params=self.decorr_params, decorr_vectors=self.decorr_vectors
            )

    def residuals(self):
        """Residuals

        Data minus model
        """
        res = self.y - self.params[self.gamma_param].value - self.model(self.x)
        
        if len(self.decorr_params) > 0:
            for parname in self.decorr_params:
                var = parname.split('_')[1]
                pars = []
                for par in self.decorr_params:
                    if var in par:
                        pars.append(self.params[par].value)
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
        return np.sqrt(self.yerr**2 + self.params[self.jit_param].value**2)

    def logprob(self):
        """
        Return log-likelihood given the data and model.
        Priors are not applied here.

        Returns:
            float: Natural log of likelihood
        """

        sigma_jit = self.params[self.jit_param].value
        residuals = self.residuals()
        loglike = loglike_jitter(residuals, self.yerr, sigma_jit)
        
        return loglike


class GPLikelihood(RVLikelihood):
    """GP Likelihood

    The Likelihood object for a radial velocity dataset modeled with a GP

    Args:
        model (radvel.model.GPModel): GP model object
        t (array): time array
        vel (array): array of velocities
        errvel (array): array of velocity uncertainties
        hnames (list of string): keys corresponding to radvel.Parameter 
           objects in model.params that are GP hyperparameters
        suffix (string): suffix to identify this Likelihood object;
           useful when constructing a `CompositeLikelihood` object
    """
    def __init__(self, model, t, vel, errvel, 
                 hnames=['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'],
                 suffix='', kernel_name="QuasiPer", **kwargs):

        self.suffix = suffix
        super(GPLikelihood, self).__init__(
              model, t, vel, errvel, suffix=self.suffix, 
              decorr_vars=[], decorr_vectors={}
            )
        assert kernel_name in gp.KERNELS.keys(), \
            'GP Kernel not recognized: ' + kernel_name + '\n' + \
            'Available kernels: ' + str(gp.KERNELS.keys())

        self.hnames = hnames  # list of string names of hyperparameters
        self.hyperparams = {k: self.params[k] for k in self.hnames}

        self.kernel_call = getattr(gp, kernel_name + "Kernel") 
        self.kernel = self.kernel_call(self.hyperparams)

        self.kernel.compute_distances(self.x, self.x)
        self.N = len(self.x)

    def update_kernel_params(self):
        """ Update the Kernel object with new values of the hyperparameters 
        """
        for key in self.list_vary_params():
            if key in self.hnames: 
                hparams_key = key.split('_')
                hparams_key = hparams_key[0] + '_' + hparams_key[1]
                self.kernel.hparams[hparams_key].value = self.params[key].value

    def _resids(self):
        """Residuals for internal GP calculations

        Data minus orbit model. For internal use in GP calculations ONLY.
        """
        res = self.y - self.params[self.gamma_param].value - self.model(self.x)
        return res

    def residuals(self):
        """Residuals

        Data minus (orbit model + predicted mean of GP noise model). For making GP plots.
        """
        mu_pred, _ = self.predict(self.x)
        res = self.y - self.params[self.gamma_param].value - self.model(self.x) - mu_pred
        return res

    def logprob(self):
        """
        Return GP log-likelihood given the data and model.

        log-likelihood is computed using Cholesky decomposition as:

        .. math::

           lnL = -0.5r^TK^{-1}r - 0.5ln[det(K)] - 0.5N*ln(2pi)
           
        where r = vector of residuals (GPLikelihood._resids), 
        K = covariance matrix, and N = number of datapoints. 

        Priors are not applied here. 
        Constant has been omitted.

        Returns:
            float: Natural log of likelihood

        """
        # update the Kernel object hyperparameter values
        self.update_kernel_params()

        r = self._resids()      

        self.kernel.compute_covmatrix(self.errorbars())

        K = self.kernel.covmatrix

        # solve alpha = inverse(K)*r
        try:
            alpha = cho_solve(cho_factor(K),r)

            # compute determinant of K
            (s,d) = np.linalg.slogdet(K)

            # calculate likelihood
            like = -.5 * (np.dot(r, alpha) + d + self.N*np.log(2.*np.pi))

            return like

        except (np.linalg.linalg.LinAlgError, ValueError):
            print("Warning: non-positive definite kernel detected.")
            return -np.inf 


    def predict(self, xpred):
        """ Realize the GP using the current values of the hyperparameters at values x=xpred.
            Used for making GP plots.

            Args:
                xpred (np.array): numpy array of x values for realizing the GP
            Returns:
                tuple: tuple containing:
                    np.array: the numpy array of predictive means \n
                    np.array: the numpy array of predictive standard deviations
        """

        self.update_kernel_params()

        r = matrix(self._resids()).T

        self.kernel.compute_distances(self.x, self.x)
        K = self.kernel.compute_covmatrix(self.errorbars())

        self.kernel.compute_distances(xpred, self.x)
        Ks = self.kernel.compute_covmatrix(0.)

        L = cho_factor(K)
        alpha = cho_solve(L, r)
        mu = np.array(Ks*alpha).flatten()

        self.kernel.compute_distances(xpred, xpred)
        Kss = self.kernel.compute_covmatrix(0.)
        B = cho_solve(L, Ks.T) 
        var = np.array(np.diag(Kss - Ks * matrix(B))).flatten()
        stdev = np.sqrt(var)

        # set the default distances back to their regular values
        self.kernel.compute_distances(self.x, self.x)

        return mu, stdev
        

class CeleriteLikelihood(GPLikelihood):
    """Celerite GP Likelihood

    The Likelihood object for a radial velocity dataset modeled with a GP
    whose kernel is an approximation to the quasi-periodic kernel.

    See celerite.readthedocs.io and Foreman-Mackey et al. 2017. AJ, 154, 220
    (equation 56) for more details.

    See `radvel/example_planets/k2-131_celerite.py` for an example of a setup
    file that uses this Likelihood object.

    Args:
        model (radvel.model.RVModel): RVModel object
        t (array): time array
        vel (array): array of velocities
        errvel (array): array of velocity uncertainties
        hnames (list of string): keys corresponding to radvel.Parameter 
           objects in model.params that are GP hyperparameters
        suffix (string): suffix to identify this Likelihood object;
           useful when constructing a `CompositeLikelihood` object
    """

    def __init__(self, model, t, vel, errvel, hnames, suffix='', **kwargs):

        super(CeleriteLikelihood, self).__init__(
            model, t, vel, errvel, hnames,
            suffix=suffix, kernel_name='Celerite'
        )

        # Sort inputs in time order. Required for celerite calculations.
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.y = self.y[order]
        self.yerr = self.yerr[order]
        self.N = len(self.x)

    def logprob(self):

        self.update_kernel_params()

        try:
            solver = self.kernel.compute_covmatrix(self.errorbars())

            # calculate log likelihood
            lnlike = -0.5 * (solver.dot_solve(self._resids()) + solver.log_determinant() + self.N*np.log(2.*np.pi))
        
            return lnlike

        except celerite.solver.LinAlgError:
            print("WARNING: non-positive definite kernel encountered!")

            return -np.inf

    def predict(self,xpred):
        """ Realize the GP using the current values of the hyperparameters at values x=xpred.
            Used for making GP plots. Wrapper for `celerite.GP.predict()`.

            Args:
                xpred (np.array): numpy array of x values for realizing the GP
            Returns:
                tuple: tuple containing:
                    np.array: numpy array of predictive means \n
                    np.array: numpy array of predictive standard deviations
        """

        self.update_kernel_params()

        B = self.kernel.hparams['gp_B'].value
        C = self.kernel.hparams['gp_C'].value
        L = self.kernel.hparams['gp_L'].value
        Prot = self.kernel.hparams['gp_Prot'].value

        # build celerite kernel with current values of hparams
        kernel = celerite.terms.JitterTerm(
                log_sigma = np.log(self.params[self.jit_param].value)
                )

        kernel += celerite.terms.RealTerm(
            log_a=np.log(B*(1+C)/(2+C)),
            log_c=np.log(1/L)
        )

        kernel += celerite.terms.ComplexTerm(
            log_a=np.log(B/(2+C)),
            log_b=-np.inf,
            log_c=np.log(1/L),
            log_d=np.log(2*np.pi/Prot)
        )

        gp = celerite.GP(kernel)
        gp.compute(self.x, self.yerr)
    #    mu, var = gp.predict(self.y-self.params[self.gamma_param].value, xpred, return_var=True)
        mu, var = gp.predict(self._resids(), xpred, return_var=True)

        stdev = np.sqrt(var)

        return mu, stdev


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
