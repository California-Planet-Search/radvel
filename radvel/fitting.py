from scipy import optimize
import numpy as np
import copy
import collections


def maxlike_fitting(post, verbose=True):
    """Maximum Likelihood Fitting

    Perform a maximum likelihood fit.

    Args:
        post (radvel.Posterior): Posterior object with initial guesses
        verbose (bool): (optional) Print messages and fitted values?

    Returns: radvel.Posterior : Posterior object with parameters
        updated their maximum likelihood values

    """

    post0 = copy.copy(post)
    if verbose:
        print("Initial loglikelihood = %f" % post0.logprob())
        print("Performing maximum likelihood fit...")

    res = optimize.minimize(
        post.neglogprob_array, post.get_vary_params(), method='Powell',
        options=dict(maxiter=200, maxfev=100000, xtol=1e-8)
    )

    cpspost = copy.copy(post)
    cpsparams = post.params.basis.to_cps(post.params, noVary = True) # setting "noVary" assigns each new parameter a vary attribute
    cpspost.params.update(cpsparams)                                 # of '', for printing purposes
 
    if verbose:
        print("Final loglikelihood = %f" % post.logprob())
        print("Best-fit parameters:")
        print(cpspost)
        
    return post
    

def model_comp(post, verbose=False):
    """Model Comparison

    Fit for planets adding one at a time.  Save results as list of
    posterior objects.

    Args:
        post (radvel.Posterior): posterior object for final best-fit solution 
            with all planets
        verbose (bool): (optional) print out statistics
        
    Returns:

        list of dictionaries : List of dictionaries with fit
            statistics. Each value in the dictionary is a tuple with
            the statistic value as the first element and a description
            of that statistic in the second element.
    """
    ipost = copy.deepcopy(post)
    
    num_planets = post.likelihood.model.num_planets

    statsdict = []
    
    for n in range(num_planets+1):
        pdict = collections.OrderedDict()
        if verbose:
            print("Testing %d planet model.\n\n" % n)

        for par in post.params.keys():
            try:
                num = int(par[-1])
                if num > n:
                    if par.startswith('k') or par.startswith('logk'):
                        post.params[par].value = 0.0
                    post.params[par].vary = False
                else:
                    post.params[par].value = ipost.params[par].value
                    post.params[par].vary = ipost.params[par].vary
            except (ValueError, KeyError):
                pass

        for par in post.params:
            if par.startswith('jit'):
                post.params[par].vary = False
            
        post = maxlike_fitting(post, verbose=False)

        ndata = len(post.likelihood.y)
        nfree = len(post.get_vary_params())
        # chi = np.sum((post.likelihood.residuals()/post.likelihood.yerr)**2)
        chi = np.sum((post.likelihood.residuals()/post.likelihood.errorbars())**2)
        chi_red = chi / (ndata - nfree)

        if verbose:
            print(post)
            print("N_free = %d" % nfree)
            print("RMS = %4.2f" % np.std(post.likelihood.residuals()))
            print("logprob (jitter fixed) = %4.2f" % post.logprob())
            print("chi (jitter fixed) = %4.2f" % chi)
            print("chi_red (jitter fixed) = %4.2f" % chi_red)
            print("BIC (jitter fixed) = %4.2f" % post.bic())
        
        pdict['$N_{\\rm data}$'] = (ndata, 'number of measurements')
        pdict['$N_{\\rm free}$'] = (nfree, 'number of free parameters')
        pdict['RMS'] = (
            np.round(np.std(post.likelihood.residuals()), 2), 
            'RMS of residuals in m s$^{-1}$'
        )
        pdict['$\\chi^{2}$'] = (np.round(chi,2), "jitter fixed")
        pdict['$\\chi^{2}_{\\nu}$'] = (
            np.round(chi_red,2), "jitter fixed"
        )
        pdict['$\\ln{\\mathcal{L}}$'] = (
            np.round(post.logprob(),2), "natural log of the likelihood"
        )
        pdict['BIC'] = (
            np.round(post.bic(),2), 
            'Bayesian information criterion'
        )
        
        statsdict.append(pdict)

    return statsdict
