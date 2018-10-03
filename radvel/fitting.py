import scipy.optimize
import numpy as np
import copy
import collections
import itertools
import radvel.likelihood

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', \
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def maxlike_fitting(post, verbose=True, method='Powell'):
    """Maximum Likelihood Fitting

    Perform a maximum likelihood fit.

    Args:
        post (radvel.Posterior): Posterior object with initial guesses
        verbose (bool [optional]): Print messages and fitted values?
        method (string [optional]): Minimization method. See documentation for `scipy.optimize.minimize` for available
            options.

    Returns: 
        radvel.Posterior : Posterior object with parameters
        updated to their maximum likelihood values

    """

    if verbose:
        print("Initial loglikelihood = %f" % post.logprob())
        print("Performing maximum likelihood fit...")
    res = scipy.optimize.minimize(
        post.neglogprob_array, post.get_vary_params(), method=method,
        options=dict(xtol=1e-8, maxiter=200, maxfev=100000)
    )
    synthparams = post.params.basis.to_synth(post.params, noVary = True) # setting "noVary" assigns each new parameter a vary attribute
    post.params.update(synthparams)                                 # of '', for printing purposes

    if verbose:
        print("Final loglikelihood = %f" % post.logprob())
        print("Best-fit parameters:")
        print(post)
        
    return post
    



def model_comp(post, params=[], mc_list=[], verbose=False):
    """Model Comparison

    Vary the presence of additional parameters and check how the improve the model fit
    Save results as list of dictionaries of posterior statistics.

    Args:
        post (radvel.Posterior): posterior object for final best-fit solution 
            with all planets
        params (list of strings): (optional) type of comparison to make via bic/aic
        mc_list (list of OrderedDicts): (optional) list of dictionaries from different
            model comparisons. Each value in the dictionary is a tuple with a statistic
            as the first element and a description as the second element.
        verbose (bool): (optional) print out statistics
        
    Returns:
        list of OrderedDicts: 
            List of dictionaries with fit statistics. Each value in the 
            dictionary is a tuple with the statistic value as the first 
            element and a description of that statistic in the second element.
    """

    assert isinstance(post, radvel.likelihood.Likelihood), \
        "model_comp requires a radvel likelihood object as the first argument"
    assert isinstance(mc_list, list), \
        "mc_list must be either an empty list or a list of model comparison dictionaries"
    assert isinstance(params, list), \
        "The params argument must contain a list of parameters for model comparison." 
    

    VALID_MC_ARGS = ['e', 'nplanets', 'trend', 'jit', 'gp']
    for element in params: 
        assert element in VALID_MC_ARGS, \
            "The valid model comparison strings in the params argument are: " \
            + ", ".join(VALID_MC_ARGS) 

 
    # If there are no parameters to compare simply do a maximum likelihood fit
    #   to get BIC and AIC values among other diagnostics. 
    if not params:
        
        fitpost = maxlike_fitting(post, verbose=verbose)

        ndata = len(fitpost.likelihood.y)
        nfree = len(fitpost.get_vary_params())
        chi = np.sum((fitpost.likelihood.residuals()/fitpost.likelihood.errorbars())**2)

        chi_red = chi / (ndata - nfree)

        if verbose:
            print(fitpost)
            print("N_free = %d" % nfree)
            print("RMS = %4.2f" % np.std(fitpost.likelihood.residuals()))
            print("logprob = %4.2f" % fitpost.logprob())
            print("chi = %4.2f" % chi)
            print("chi_red = %4.2f" % chi_red)
            print("BIC = %4.2f" % fitpost.likelihood.bic())
            print("AIC = %4.2f" % fitpost.likelihood.aic())
       
        comparison_parameters = ['Free Params', '$N_{\\rm free}$', '$N_{\\rm data}$',\
            'RMS', '$\\ln{\\mathcal{L}}$',\
            'BIC', 'AICc']
        pdict=collections.OrderedDict.fromkeys(comparison_parameters) 
        pdict['$N_{\\rm data}$'] = (ndata, 'number of measurements')
        pdict['$N_{\\rm free}$'] = (nfree, 'number of free parameters')
        pdict['RMS'] = (
            np.round(np.std(fitpost.likelihood.residuals()), 2), 
            'RMS of residuals in m s$^{-1}$'
        )
        #pdict['$\\chi^{2}$'] = (np.round(chi,2), "jitter fixed")
        #pdict['$\\chi^{2}_{\\nu}$'] = (
        #    np.round(chi_red,2), "jitter fixed"
        #)
        pdict['$\\ln{\\mathcal{L}}$'] = (
            np.round(fitpost.logprob(),2), "natural log of the likelihood"
        )
        pdict['BIC'] = (
            np.round(fitpost.likelihood.bic(),2),
            'Bayesian information criterion'
        )
        pdict['AICc'] = (
            np.round(fitpost.likelihood.aic(),2),
            'Aikaike information (small sample corrected) criterion'
        )
        num_planets = fitpost.likelihood.model.num_planets
        freepar = []
        thisbasis = fitpost.params.basis.name
        eparams = fitpost.params.basis.get_eparams()
        circparams = fitpost.params.basis.get_circparams()
        eparam = eparams[0]
        leparam = len(eparam)
        circparam = circparams[-1]
        lcparam = len(circparam)
        planet_letters = fitpost.likelihood.params.planet_letters
        if planet_letters is None:
            planet_letters = [ALPHABET[i] for i in range(num_planets+1)] 
        jitterchecked = False
        for pari in fitpost.params:
            if (len(pari) >= lcparam) and (pari[0:lcparam] == circparam) and (fitpost.params[pari].vary == True): 
                freepar.append('$K_{'+planet_letters[int(pari[lcparam+0:])]+'}$')
            if (len(pari) >= leparam) and (pari[0:leparam] == eparam) and (fitpost.params[pari].vary == True):
                freepar.append('$e_{'+planet_letters[int(pari[leparam+0:])]+'}$')
            if (pari == 'dvdt') and (fitpost.params[pari].vary == True):
                freepar.append(r'$\dot{\gamma}$')
            if (pari == 'curv') and (fitpost.params[pari].vary == True):
                freepar.append('$\ddot{\gamma}$')
            if (len(pari) >= 3) and (pari[0:3] == 'jit') and (fitpost.params[pari].vary == True) \
                    and (jitterchecked == False):
                partex = '\{$\sigma$\}'
                freepar.append(partex)
                jitterchecked = True
            if (len(pari) >= 6) and (pari[0:6] == 'gp_amp') and (fitpost.params[pari].vary == True):
                freepar.append('GP')

        pdict['Free Params'] = (freepar, "The free parameters in this model")
        mc_list.append(pdict)
        return mc_list


    # Otherwise parse the different parameter comparison options and perform a maximum 
    #   likelihood model comparison for each case
    
    elif 'gp' in params: 
        newparams = [pi for pi in params if pi != 'gp'] 
        if isinstance(post.likelihood, radvel.likelihood.GPLikelihood):
            print("Warning: BIC/AIC comparisons with and without GP are only implemented for "\
                + "kernels where the amplitude of the GP is described by the 'gp_amp' "\
                + "hyper parameter")
            gpparamlist = post.hnames
            ipost = copy.deepcopy(post)
            allfixed = False
            for gpparam in gpparamlist:
                if len(gpparam) >= 6 and post.params[gpparam][0:6] == 'gp_amp':
                    ipost.params[gpparam].value = 0.
                if post.params[gpparam].vary == True:
                    allfixed = False
                    ipost.params[par].vary = False
            if not allfixed:
                mc_list = model_comp(ipost, newparams, mc_list=mc_list)
            mc_list = model_comp(post, newparams, mc_list=mc_list)
            return mc_list
        else:
            if verbose:
                print("Warning: You requested a GP BIC/AIC comparison")
                print("   However, your model does not include GPs")
            mc_list = model_comp(post, newparams, mc_list=mc_list)
            return mc_list
         

    elif 'jit' in params:
        ipost = copy.deepcopy(post)
        cpost = copy.deepcopy(ipost)
        newparams = [pi for pi in params if pi != 'jit'] 
        anyjitteron = False
        for parami in ipost.params:
            if len(parami) >= 3 and parami[:3] == 'jit' and ipost.params[parami].vary == True:
                cpost.params[parami].value = 0.
                cpost.params[parami].vary = False
                anyjitteron = True
        if anyjitteron:
            mc_list = model_comp(cpost, newparams, mc_list=mc_list)
        else:
            if verbose:
                print("Warning: You requested a jitter BIC/AIC comparison")
                print("   However, your model has a fixed jitter")
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list


    elif 'trend' in params:
        ipost = copy.deepcopy(post)
        newparams = [pi for pi in params if pi != 'trend'] 
        trendparamlist = ['curv', 'dvdt']
        anytrendparam = False
        for cparam in trendparamlist:
            if ipost.params[cparam].vary == True:
                ipost.params[cparam].value = 0.
                cpost = copy.deepcopy(ipost)
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
                ipost.params[cparam].vary = False
                anytrendparam = True
        if not anytrendparam:
            if verbose:
                print("Warning: You requested a trend BIC/AIC comparison")
                print("   However, your model has a fixed dv/dt and curv")
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list


    elif 'nplanets' in params:
        eparams = post.params.basis.get_eparams()
        circparams = post.params.basis.get_circparams()
        allparams = eparams+circparams  

        ipost = copy.deepcopy(post)
        newparams = [pi for pi in params if pi != 'nplanets'] 
        num_planets = post.likelihood.model.num_planets
        pllist = [pl+1 for pl in range(num_planets)]
        plgroups = ()
        for p in [pl+1 for pl in range(num_planets)]:
            plgroups = itertools.chain(plgroups, itertools.combinations(pllist, p)) 
        plparams = []
        for plgroup in plgroups:
            suffixes = [str(pl) for pl in plgroup]
            plparams.append([ [pari+''+sufi for pari in allparams] for sufi in suffixes ])
        for plparamset in plparams:
            if all( [any([post.params[pari].vary for pari in pparam]) for pparam in plparamset] ):
                cpost = copy.deepcopy(post)
                for pparam in plparamset:
                    for pari in pparam:
                        if pari[0] == 'k':
                            cpost.params[pari].value = 0.
                        if len(pari) >= 4 and pari[0:4] == 'logk':
                            cpost.params[pari].value = -np.inf
                        cpost.params[pari].vary = False
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list
    

    elif 'e' in params:
        eparams = post.params.basis.get_eparams()
        lepar0 = len(eparams[0])
        lepar1 = len(eparams[1])

        ipost = copy.deepcopy(post)
        newparams = [pi for pi in params if pi != 'e'] 
        num_planets = post.likelihood.model.num_planets
        pllist = [pl+1 for pl in range(num_planets)]
        plgroups = ()
        for p in [pl+1 for pl in range(num_planets)]:
            plgroups = itertools.chain(plgroups, itertools.combinations(pllist, p)) 
        plparams = []
        for plgroup in plgroups:
            suffixes = [str(pl) for pl in plgroup]
            plparams.append([ [pari+''+sufi for pari in eparams] for sufi in suffixes ])
        anyefree = False
        for plparamset in plparams:
            if all( [any([post.params[pari].vary for pari in pparam]) for pparam in plparamset] ):
                cpost = copy.deepcopy(post)
                for pparam in plparamset:
                    for pari in pparam:
                        cpost.params[pari].value = 0.
                        cpost.params[pari].vary = False
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
                anyefree = True
        if not anyefree:
            if verbose:
                print("Warning: You requested an eccentricity BIC/AIC comparison")
                print("   However, your model has fixed e for all planets")
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list


    else:
        errorstring = 'The given params argument was:\n' + ' '.join(params)
        errorstring += '\n'
        errorstring += 'The only valid comparison parameters are:\n'\
            +' '.join(VALID_MC_ARGS)
        raise NotImplementedError(errorstring)





