import scipy.optimize
import numpy as np
import copy
import collections
import itertools
import radvel.likelihood

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def maxlike_fitting(post, verbose=True, method='Powell'):
    """Maximum A Posteriori Fitting

    Perform a maximum a posteriori fit.

    Args:
        post (radvel.Posterior): Posterior object with initial guesses
        verbose (bool [optional]): Print messages and fitted values?
        method (string [optional]): Minimization method. See documentation for `scipy.optimize.minimize` for available
            options.

    Returns:
        radvel.Posterior : Posterior object with parameters
        updated to their maximum a posteriori values

    """

    if verbose:
        print("Initial loglikelihood = %f" % post.logprob())
        print("Performing maximum a posteriori fit...")
    _ = scipy.optimize.minimize(
        post.neglogprob_array, post.get_vary_params(), method=method,
        options=dict(maxiter=200, maxfev=100000, xtol=1e-8))
    post.vector.vector_to_dict()

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

    valid_mc_args = ['e', 'nplanets', 'trend', 'jit', 'gp']
    for element in params:
        assert element in valid_mc_args, \
            "The valid model comparison strings in the params argument are: " \
            + ", ".join(valid_mc_args)

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

        comparison_parameters = ['Free Params', r'$N_{\rm free}$', r'$N_{\rm data}$',
                                 'RMS', r'$\ln{\mathcal{L}}$', 'BIC', 'AICc']
        pdict = collections.OrderedDict.fromkeys(comparison_parameters)
        pdict[r'$N_{\rm data}$'] = (ndata, 'number of measurements')
        pdict[r'$N_{\rm free}$'] = (nfree, 'number of free parameters')
        pdict['RMS'] = (
            np.round(np.std(fitpost.likelihood.residuals()), 2),
            'RMS of residuals in m s$^{-1}$'
        )
        # pdict['$\\chi^{2}$'] = (np.round(chi,2), "jitter fixed")
        # pdict['$\\chi^{2}_{\\nu}$'] = (
        #    np.round(chi_red,2), "jitter fixed"
        # )
        pdict[r'$\ln{\mathcal{L}}$'] = (
            np.round(fitpost.logprob(), 2), "natural log of the likelihood"
        )
        pdict['BIC'] = (
            np.round(fitpost.likelihood.bic(), 2),
            'Bayesian information criterion'
        )
        pdict['AICc'] = (
            np.round(fitpost.likelihood.aic(), 2),
            'Aikaike information (small sample corrected) criterion'
        )
        num_planets = fitpost.likelihood.model.num_planets
        freepar = []
        eparams = fitpost.vector.params.basis.get_eparams()
        circparams = fitpost.vector.params.basis.get_circparams()
        eparam = eparams[0]
        leparam = len(eparam)
        circparam = circparams[-1]
        lcparam = len(circparam)
        planet_letters = fitpost.likelihood.vector.params.planet_letters
        if planet_letters is None:
            planet_letters = [ALPHABET[i] for i in range(num_planets+1)]
        jitterchecked = False
        for pari in fitpost.vector.names:
            if (len(pari) >= lcparam) and (pari[0:lcparam] == circparam) \
                    and fitpost.vector.vector[fitpost.vector.indices[pari]][1]:
                freepar.append('$K_{'+planet_letters[int(pari[lcparam+0:])]+'}$')
            if (len(pari) >= leparam) and (pari[0:leparam] == eparam) \
                    and fitpost.vector.vector[fitpost.vector.indices[pari]][1]:
                freepar.append('$e_{'+planet_letters[int(pari[leparam+0:])]+'}$')
            if (pari == 'dvdt') \
                    and fitpost.vector.vector[fitpost.vector.indices[pari]][1]:
                freepar.append(r'$\dot{\gamma}$')
            if (pari == 'curv') \
                    and fitpost.vector.vector[fitpost.vector.indices[pari]][1]:
                freepar.append(r'$\ddot{\gamma}$')
            if (len(pari) >= 3) and (pari[0:3] == 'jit') \
                    and fitpost.vector.vector[fitpost.vector.indices[pari]][1] \
                    and (not jitterchecked):
                partex = r'{$\sigma$}'
                freepar.append(partex)
                jitterchecked = True
            if (len(pari) >= 6) and (pari[0:6] == 'gp_amp') \
                    and fitpost.vector.vector[fitpost.vector.indices[pari]][1]:
                freepar.append(r'GP$_{\rm %s}$' % pari[6:].replace('_', ''))

        pdict['Free Params'] = (freepar, "The free parameters in this model")
        mc_list.append(pdict)
        return mc_list

    # Otherwise parse the different parameter comparison options and perform a maximum
    #   likelihood model comparison for each case

    elif 'gp' in params:
        newparams = [pi for pi in params if pi != 'gp']
        if verbose:
            print("Warning: BIC/AIC comparisons with and without GP are only implemented for "
                  + "kernels where the amplitude of the GP is described by the 'gp_amp' "
                  + "hyper parameter")
        have_gpamp = False
        for param in post.vector.names:
            if 'gp_amp' in param:
                have_gpamp = True
                break
            else:
                continue
        if have_gpamp:
            gpparamlist = post.likelihood.hnames
            ipost = copy.deepcopy(post)
            allfixed = False
            for gpparam in gpparamlist:
                if len(gpparam) >= 6 and gpparam.startswith('gp_amp'):
                    ipost.vector.vector[ipost.vector.indices[gpparam]][0] = 0.
                if ipost.vector.vector[ipost.vector.indices[gpparam]][1]:
                    allfixed = False
                    ipost.vector.vector[ipost.vector.indices[gpparam]][0] = False
            if not allfixed:
                mc_list = model_comp(ipost, newparams, mc_list=mc_list)
            post.list_vary_params()
            post.likelihood.list_vary_params()
            mc_list = model_comp(post, newparams, mc_list=mc_list)
            return mc_list
        else:
            if verbose:
                print("Warning: You requested a GP BIC/AIC comparison")
                print("   However, your model does not include GPs")
            post.list_vary_params()
            post.likelihood.list_vary_params()
            mc_list = model_comp(post, newparams, mc_list=mc_list)
            return mc_list

    elif 'jit' in params:
        ipost = copy.deepcopy(post)
        cpost = copy.deepcopy(ipost)
        newparams = [pi for pi in params if pi != 'jit']
        anyjitteron = False
        for parami in ipost.vector.names:
            if len(parami) >= 3 and parami[:3] == 'jit' \
                    and ipost.vector.vector[ipost.vector.indices[parami]][1]:
                cpost.vector.vector[cpost.vector.indices[parami]][0] = 1e-6
                cpost.vector.vector[cpost.vector.indices[parami]][1] = 0
                anyjitteron = True
        if anyjitteron:
            cpost.list_vary_params()
            cpost.likelihood.list_vary_params()
            mc_list = model_comp(cpost, newparams, mc_list=mc_list)
        else:
            if verbose:
                print("Warning: You requested a jitter BIC/AIC comparison")
                print("   However, your model has a fixed jitter")
        ipost.list_vary_params()
        ipost.likelihood.list_vary_params()
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list

    elif 'trend' in params:
        ipost = copy.deepcopy(post)
        newparams = [pi for pi in params if pi != 'trend']
        trendparamlist = ['curv', 'dvdt']
        anytrendparam = False
        for cparam in trendparamlist:
            if ipost.vector.vector[ipost.vector.indices[cparam]][1]:
                ipost.vector.vector[ipost.vector.indices[cparam]][0] = 0
                cpost = copy.deepcopy(ipost)
                cpost.list_vary_params()
                cpost.likelihood.list_vary_params()
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
                ipost.vector.vector[ipost.vector.indices[cparam]][1] = 0
                anytrendparam = True
        if not anytrendparam:
            if verbose:
                print("Warning: You requested a trend BIC/AIC comparison")
                print("   However, your model has a fixed dv/dt and curv")
        ipost.list_vary_params()
        ipost.likelihood.list_vary_params()
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list

    elif 'nplanets' in params:
        eparams = post.vector.params.basis.get_eparams()
        circparams = post.vector.params.basis.get_circparams()
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
            plparams.append([[pari+''+sufi for pari in allparams] for sufi in suffixes])
        for plparamset in plparams:
            if all([any([post.vector.vector[post.vector.indices[pari]][1] for pari in pparam]) for pparam in plparamset]):
                cpost = copy.deepcopy(post)
                for pparam in plparamset:
                    for pari in pparam:
                        if pari[0] == 'k':
                            cpost.vector.vector[cpost.vector.indices[pari]][0] = 0.
                        if len(pari) >= 4 and pari[0:4] == 'logk':
                            cpost.vector.vector[cpost.vector.indices[pari]][0] = -np.inf
                        cpost.vector.vector[cpost.vector.indices[pari]][1] = 0
                del cpost.vary_params
                cpost.list_vary_params()
                cpost.likelihood.list_vary_params()
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
        ipost.list_vary_params()
        ipost.likelihood.list_vary_params()
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list

    elif 'e' in params:
        eparams = post.vector.params.basis.get_eparams()

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
            plparams.append([[pari+''+sufi for pari in eparams] for sufi in suffixes])
        anyefree = False
        for plparamset in plparams:
            if all([any([post.vector.vector[post.vector.indices[pari]][1] for pari in pparam]) for pparam in plparamset]):
                cpost = copy.deepcopy(post)
                for pparam in plparamset:
                    for pari in pparam:
                        cpost.vector.vector[cpost.vector.indices[pari]][0] = 0.
                        cpost.vector.vector[cpost.vector.indices[pari]][1] = 0.
                cpost.list_vary_params()
                cpost.likelihood.list_vary_params()
                mc_list = model_comp(cpost, newparams, mc_list=mc_list)
                anyefree = True
        if not anyefree:
            if verbose:
                print("Warning: You requested an eccentricity BIC/AIC comparison")
                print("   However, your model has fixed e for all planets")
        ipost.list_vary_params()
        ipost.likelihood.list_vary_params()
        mc_list = model_comp(ipost, newparams, mc_list=mc_list)
        return mc_list

    else:
        errorstring = 'The given params argument was:\n' + ' '.join(params)
        errorstring += '\n'
        errorstring += 'The only valid comparison parameters are:\n'\
            + ' '.join(valid_mc_args)
        raise NotImplementedError(errorstring)
