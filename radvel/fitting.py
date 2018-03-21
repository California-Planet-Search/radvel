import scipy.optimize
import numpy as np
import copy
import collections
import itertools
import radvel.likelihood



def maxlike_fitting(post, verbose=True):
    """Maximum Likelihood Fitting

    Perform a maximum likelihood fit.

    Args:
        post (radvel.Posterior): Posterior object with initial guesses
        verbose (bool [optional]): Print messages and fitted values?

    Returns: 
        radvel.Posterior : Posterior object with parameters
        updated to their maximum likelihood values

    """

    post0 = copy.copy(post)
    if verbose:
        print("Initial loglikelihood = %f" % post0.logprob())
        print("Performing maximum likelihood fit...")
    res = scipy.optimize.minimize(
        post.neglogprob_array, post.get_vary_params(), method='Nelder-Mead',
        options=dict(xatol=1e-8, maxiter=200, maxfev=100000)
    )
    synthpost = copy.copy(post)
    synthparams = post.params.basis.to_synth(post.params, noVary = True) # setting "noVary" assigns each new parameter a vary attribute
    synthpost.params.update(synthparams)                                 # of '', for printing purposes

    if verbose:
        print("Final loglikelihood = %f" % post.logprob())
        print("Best-fit parameters:")
        print(synthpost)
        
    return post
    



def model_comp(post, params=[], verbose=False, mc_list=[]):
    """Model Comparison

    Fit for planets adding one at a time.  Save results as list of
    posterior objects.

    Args:
        post (radvel.Posterior): posterior object for final best-fit solution 
            with all planets
        params: list of parameters to compare with bic/aic
        mc_list: list of dictionaries from different model comparison
            statistics
        verbose (bool): (optional) print out statistics
        
    Returns:
        list of dictionaries: 
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
    
    VALID_MC_ARGS = ['e', 'nplanet', 'curv', 'jit', 'gp']
    for element in params: 
        assert element in VALID_MC_ARGS, \
            "The valid model comparison strings in the params argument are: " \
            + ", ".join(VALID_MC_ARGS) 
 
    # If there are no parameters to compare simply do a maximum likelihood fit
    #   to get BIC and AIC values among other diagnostics. 
    if not params:
        
        fitpost = maxlike_fitting(post, verbose=False)

        ndata = len(fitpost.likelihood.y)
        nfree = len(fitpost.get_vary_params())
        chi = np.sum((fitpost.likelihood.residuals()/fitpost.likelihood.errorbars())**2)
        chi_red = chi / (ndata - nfree)

        if verbose:
            print(fitpost)
            print("N_free = %d" % nfree)
            print("RMS = %4.2f" % np.std(fitpost.likelihood.residuals()))
            print("logprob (jitter fixed) = %4.2f" % fitpost.logprob())
            print("chi (jitter fixed) = %4.2f" % chi)
            print("chi_red (jitter fixed) = %4.2f" % chi_red)
            print("BIC (jitter fixed) = %4.2f" % fitpost.bic())
            print("AIC (jitter fixed) = %4.2f" % fitpost.aic())
       
        pdict={} 
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
            np.round(fitpost.bic(),2), 
            'Bayesian information criterion'
        )
        pdict['AIC'] = (
            np.round(fitpost.aic(),2), 
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
        for pari in fitpost.params:
            if pari[0:lcparam] == circparam and fitpost.params[pari].vary == True: 
                freepar.append('Planet '+planet_letters[int(pari[lcparam+0:])])
            if pari[0:leparam] == eparam and fitpost.params[pari].vary == True:
                freepar.append('Eccentricity '+planet_letters[int(pari[leparam+0:])])
            if (pari == 'curv' or pari == 'dvdt') and fitpost.params[pari].vary == True:
                freepar.append(pari)

        pdict['Free Params'] = freepar
        mc_list.append(pdict)
        return mc_list


    # Otherwise parse the different parameter comparison options and perform a maximum 
    #   likelihood model comparison for each case
    
    elif 'gp' in params: 
        newparams = copy.copy(params) 
        newparams.remove('gp')
        if isinstance(post, radvel.likelihood.GPLikelihood):
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
            print "Warning: You requested a GP BIC/AIC comparison"
            print "   However, you're model does not include GPs"
            mc_list = model_comp(post, newparams, mc_list=mc_list)
            return mc_list
         


    elif 'jit' in params:
        ipost = copy.deepcopy(post)
        cpost = copy.deepcopy(ipost)
        newparams = copy.copy(params) 
        newparams.remove('jit')
        anyjitteron = False
        for parami in ipost.params:
            if len(parami) >= 3 and parami[:3] == 'jit' and ipost.params[parami].vary == True:
                cpost.params[parami].value = 0.
                cpost.params[parami].vary = False
                anyjitteron = True
        if anyjitteron:
            mc_list = model_comp(cpost, newparams, mc_list=mc_list)
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list


    elif 'trend' in params:
        ipost = copy.deepcopy(post)
        newparams = copy.copy(params) 
        newparams.remove('trend')
        trendparamlist = ['curv', 'dvdt']
        for cparam in trendparamlist:
            if post.params[cparam].vary == True:
                cpost = copy.deepcopy(ipost)
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
                ipost.params[cparam].vary = False
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list


    elif 'nplanet' in params:
        eparams = post.params.basis.get_eparams()
        circparams = post.params.basis.get_circparams()
        allparams = eparams+circparams  

        ipost = copy.deepcopy(post)
        newparams = copy.copy(params) 
        newparams.remove('nplanet')
        num_planets = post.likelihood.model.num_planets
        pllist = [pl+1 for pl in range(num_planets)]
        plgroups = ()
        for p in [pl+1 for pl in range(num_planets)]:
            plgroups = itertools.chain(plgroups, itertools.combinations(pllist, p)) 
        plparams = []
        for plgroup in plgroups:
            suffixes = [str(pl) for pl in plgroup]
            plparams.append([ [pari+''+sufi for pari in allparams] for sufi in suffixes ])
        print plparams
        for plparamset in plparams:
            if all( [any([post.params[pari].vary for pari in pparam]) for pparam in plparamset] ):
                cpost = copy.deepcopy(post)
                for pparam in plparamset:
                    for pari in pparam:
                        print pari
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
        newparams = copy.copy(params) 
        newparams.remove('e')
        num_planets = post.likelihood.model.num_planets
        pllist = [pl+1 for pl in range(num_planets)]
        plgroups = ()
        for p in [pl+1 for pl in range(num_planets)]:
            plgroups = itertools.chain(plgroups, itertools.combinations(pllist, p)) 
        plparams = []
        for plgroup in plgroups:
            suffixes = [str(pl) for pl in plgroup]
            plparams.append([ [pari+''+sufi for pari in eparams] for sufi in suffixes ])
        print plparams
        for plparamset in plparams:
            if all( [any([post.params[pari].vary for pari in pparam]) for pparam in plparamset] ):
                cpost = copy.deepcopy(post)
                for pparam in plparamset:
                    for pari in pparam:
                        cpost.params[pari].value = 0.
                        cpost.params[pari].vary = False
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list



    else:
        print "The given params argument was:"
        print params
        print "The only valid comparison parameters are:"
        print VALID_MC_ARGS
        raise





