import os
import sys
import time
import copy

from multiprocessing import Pool

import pandas as pd
import numpy as np
from scipy import optimize

import emcee

from radvel import utils, likelihood


# Maximum G-R statistic to stop burn-in period
burnGR = 1.03

# Maximum G-R statistic for chains to be deemed well-mixed
maxGR = 1.01

# Minimum number of steps per walker before
# convergence tests are performed
minsteps = 1000

class StateVars(object):
    def __init__(self):
        pass

statevars = StateVars()

def _status_message(statevars):
    msg = (
        "{:d}/{:d} ({:3.1f}%) steps complete; "
        "Running {:.2f} steps/s; Mean acceptance rate = {:3.1f}%; "
        "Min Tz = {:.1f}; Max G-R = {:4.2f}      \r"
    ).format(statevars.ncomplete, statevars.totsteps, statevars.pcomplete,
                 statevars.rate, statevars.ar, statevars.mintz, statevars.maxgr)

    sys.stdout.write(msg)
    sys.stdout.flush()


def convergence_check(samplers):
    """Check for convergence
    Check for convergence for a list of emcee samplers
    
    Args:
        samplers (list): List of emcee sampler objects
    """
    
    statevars.ar = 0
    statevars.ncomplete = statevars.nburn
    statevars.tchains = np.empty((statevars.ndim,
                        statevars.samplers[0].flatlnprobability.shape[0],
                        statevars.ensembles))
    statevars.lnprob = []
    for i,sampler in enumerate(statevars.samplers):
        statevars.ncomplete += sampler.flatlnprobability.shape[0]
        statevars.ar += sampler.acceptance_fraction.mean() * 100
        statevars.tchains[:,:,i] = sampler.flatchain.transpose()
        statevars.lnprob.append(sampler.flatlnprobability)
    statevars.ar /= statevars.ensembles

    statevars.pcomplete = statevars.ncomplete/float(statevars.totsteps) * 100
    statevars.rate = (statevars.checkinterval*statevars.nwalkers*statevars.ensembles) / statevars.interval

    if statevars.ensembles < 3:
        # if less than 3 ensembles then GR between ensembles does
        # not work so just calculate is on the last sampler
        statevars.tchains = sampler.chain.transpose()

    # Must have compelted at least 5% or 1000 steps per walker before
    # attempting to calculate GR
    if statevars.pcomplete < 5 and sampler.flatlnprobability.shape[0] <= minsteps*statevars.nwalkers:
        (statevars.ismixed, statevars.maxgr, statevars.mintz) = 0, np.inf, -1
    else:
        (statevars.ismixed, gr, tz) = gelman_rubin(statevars.tchains)
        statevars.mintz = min(tz)
        statevars.maxgr = max(gr)
        if statevars.ismixed:
            statevars.mixcount += 1
        else:
            statevars.mixcount = 0

    _status_message(statevars)

def _domcmc(input_tuple):
    # Function to be run in parallel on different CPUs
    #
    # Input is a tuple: first element is an emcee sampler object, second is an array of 
    #   initial positions, third is number of steps to run before doing a convergence check
    sampler = input_tuple[0]
    ipos = input_tuple[1]
    check_interval = input_tuple[2]
    sampler.run_mcmc(ipos, check_interval)
    
    return sampler

def mcmc(post, nwalkers=50, nrun=10000, ensembles=8,
             checkinterval=50):
    """Run MCMC
    Run MCMC chains using the emcee EnsambleSampler
    Args:
        post (radvel.posterior): radvel posterior object
        nwalkers (int): number of MCMC walkers
        nrun (int): number of steps to take
        ensembles (int): number of ensembles to run. Will be run
            in parallel on separate CPUs
        checkinterval (int): check MCMC convergence statistics every 
            `checkinterval` steps
    Returns:
        DataFrame: DataFrame containing the MCMC samples
    """

   # server = pp.Server(ncpus=ensembles)

   # statevars.server = server
    statevars.ensembles = ensembles
    statevars.nwalkers = nwalkers
    statevars.checkinterval = checkinterval
    
    nrun = int(nrun)
        
    # Get an initial array value
    pi = post.get_vary_params()
    statevars.ndim = pi.size

    if nwalkers < 2*statevars.ndim:
        print("WARNING: Number of walkers is less than 2 times number \
of free parameters. Adjusting number of walkers to {}".format(2*statevars.ndim))
        statevars.nwalkers = 2*statevars.ndim

    # set up perturbation size
    pscales = []
    for par in post.list_vary_params():
        val = post.params[par].value
        if post.params[par].mcmcscale is None:
            if par.startswith('per'):
                pscale = np.abs(val * 1e-5*np.log10(val))
            #    pscale_per = pscale
            elif par.startswith('logper'):
                pscale = np.abs(1e-5 * val)
            #    pscale_per = pscale
            elif par.startswith('tc'):
                pscale = 0.1
            else:
                pscale = np.abs(0.10 * val)
            post.params[par].mcmc_scale = pscale
        else:
            pscale = post.params[par].mcmcscale
        pscales.append(pscale)
    pscales = np.array(pscales)
    
    statevars.samplers = []
    statevars.initial_positions = []
    for e in range(ensembles):
        lcopy = copy.deepcopy(post)
        pi = lcopy.get_vary_params()
        p0 = np.vstack([pi]*statevars.nwalkers)
        p0 += [np.random.rand(statevars.ndim)*pscales for i in range(statevars.nwalkers)]
        statevars.initial_positions.append(p0)
        statevars.samplers.append(emcee.EnsembleSampler( 
            statevars.nwalkers, statevars.ndim, lcopy.logprob_array, threads=1))

        
    num_run = int(np.round(nrun / checkinterval))
    statevars.totsteps = nrun*statevars.nwalkers*statevars.ensembles
    statevars.mixcount = 0
    statevars.ismixed = 0
    statevars.burn_complete = False
    statevars.nburn = 0
    statevars.ncomplete = statevars.nburn
    statevars.pcomplete = 0
    statevars.rate = 0
    statevars.ar = 0
    statevars.mintz = -1
    statevars.maxgr = np.inf
    statevars.t0 = time.time()

    for r in range(num_run):
        t1 = time.time()
        mcmc_input_array = []
        for i, sampler in enumerate(statevars.samplers):
            if sampler.flatlnprobability.shape[0] == 0:
                p1 = statevars.initial_positions[i]
            else:
                p1 = None
            mcmc_input = (sampler, p1, checkinterval)
            mcmc_input_array.append(mcmc_input)

        pool = Pool(statevars.ensembles)
        statevars.samplers = pool.map(_domcmc, mcmc_input_array)
        pool.close() #terminates worker processes once all work is done
        pool.join() #waits for all processes to finish before proceeding

        t2 = time.time()
        statevars.interval = t2 - t1

        convergence_check(statevars.samplers)



        # Burn-in complete after maximum G-R statistic first reaches burnGR
        # reset samplers
        if not statevars.burn_complete and statevars.maxgr <= burnGR:
            for i, sampler in enumerate(statevars.samplers):
                statevars.initial_positions[i] = sampler._last_run_mcmc_result[0]
                sampler.reset()
                statevars.samplers[i] = sampler
            msg = (
                "\nDiscarding burn-in now that the chains are marginally "
                "well-mixed\n"
            )
            print(msg)
            statevars.nburn = statevars.ncomplete
            statevars.burn_complete = True

        if statevars.mixcount >= 5:
            tf = time.time()
            tdiff = tf - statevars.t0
            tdiff,units = utils.time_print(tdiff)
            msg = (
                "\nChains are well-mixed after {:d} steps! MCMC completed in "
                "{:3.1f} {:s}"
            ).format(statevars.ncomplete, tdiff, units)
            print(msg)
            break

            
    print("\n")        
    if statevars.ismixed and statevars.mixcount < 5: 
        msg = (
            "MCMC: WARNING: chains did not pass 5 consecutive convergence "
            "tests. They may be marginally well=mixed."
        )
        print(msg)
    elif not statevars.ismixed: 
        msg = (
            "MCMC: WARNING: chains did not pass convergence tests. They are "
            "likely not well-mixed."
        )
        print(msg)
        
    df = pd.DataFrame(
        statevars.tchains.reshape(statevars.ndim,statevars.tchains.shape[1]*statevars.tchains.shape[2]).transpose(),
        columns=post.list_vary_params())
    df['lnprobability'] = np.hstack(statevars.lnprob)


    return df

def draw_models_from_chain(mod, chain, t, nsamples=50):
    """Draw Models from Chain
    
    Given an MCMC chain of parameters, draw representative parameters
    and synthesize models.
    Args:
        mod (radvel.RVmodel) : RV model
        chain (DataFrame): pandas DataFrame with different values from MCMC 
            chain
        t (array): time range over which to synthesize models
        nsamples (int): number of draws
    
    Returns:
        array: 2D array with the different models as different rows
    """

    np.random.seed(0)
    chain_samples = chain.ix[np.random.choice(chain.index, nsamples)]
    models = []
    for i in chain_samples.index:
        params = np.array( chain.ix[i, mod.vary_parameters] )
        params = mod.array_to_params(params)
        models += [mod.model(params, t)]
    models = np.vstack(models)
    return models


def gelman_rubin(pars0, minTz=1000, maxGR=maxGR):
    """Gelman-Rubin Statistic
    Calculates the Gelman-Rubin statistic and the number of
    independent draws for each parameter, as defined by Ford et
    al. (2006) (http://adsabs.harvard.edu/abs/2006ApJ...642..505F).
    The chain is considered well-mixed if all parameters have a
    Gelman-Rubin statistic of <= 1.03 and >= 1000 independent draws.
    History: 
        2010/03/01 - Written: Jason Eastman - The Ohio State University        
        2012/10/08 - Ported to Python by BJ Fulton - University of Hawaii, 
            Institute for Astronomy
        2016/04/20 - Adapted for use in radvel. Removed "angular" parameter.
    Args:
        pars0 (array): A 3 dimensional array (NPARS,NSTEPS,NCHAINS) of
            parameter values
        minTz (int): (optional) minimum Tz to consider well-mixed
        maxGR (float): (optional) maximum Gelman-Rubin statistic to
            consider well-mixed
    Returns:
        (tuple): tuple containing:
            ismixed (bool): Are the chains well-mixed?
            gelmanrubin (array): An NPARS element array containing the
                Gelman-Rubin statistic for each parameter (equation
                25)
            Tz (array): An NPARS element array containing the number
                of independent draws for each parameter (equation 26)
    """


    pars = pars0.copy() # don't modify input parameters
    
    sz = pars.shape
    msg = 'MCMC: GELMAN_RUBIN: ERROR: pars must have 3 dimensions'
    assert pars.ndim == 3, msg 

    npars = float(sz[0])
    nsteps = float(sz[1])
    nchains = float(sz[2])

    msg = 'MCMC: GELMAN_RUBIN: ERROR: NSTEPS must be greater than 1'
    assert nsteps > 1, msg

    # Equation 21: W(z) in Ford 2006
    variances = np.var(pars,axis=1, dtype=np.float64)
    meanofvariances = np.mean(variances,axis=1)
    withinChainVariances = np.mean(variances, axis=1)
    
    # Equation 23: B(z) in Ford 2006
    means = np.mean(pars,axis=1)
    betweenChainVariances = np.var(means,axis=1, dtype=np.float64) * nsteps
    varianceofmeans = np.var(means,axis=1, dtype=np.float64) / (nchains-1)
    varEstimate = (
        (1.0 - 1.0/nsteps) * withinChainVariances 
        + 1.0 / nsteps * betweenChainVariances
    )
    
    bz = varianceofmeans * nsteps

    # Equation 24: varhat+(z) in Ford 2006
    varz = (nsteps-1.0)/bz + varianceofmeans

    # Equation 25: Rhat(z) in Ford 2006
    gelmanrubin = np.sqrt(varEstimate/withinChainVariances)

    # Equation 26: T(z) in Ford 2006
    vbz = varEstimate / bz
    tz = nchains*nsteps*vbz[vbz < 1]
    if tz.size == 0:
        tz = [-1]

    # well-mixed criteria
    ismixed = min(tz) > minTz and max(gelmanrubin) < maxGR
        
    return (ismixed, gelmanrubin, tz)
