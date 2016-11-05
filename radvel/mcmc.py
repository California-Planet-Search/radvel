import emcee
import pandas as pd
import numpy as np
from scipy import optimize
import sys
import time
from radvel import utils

# Maximum G-R statistic to stop burn-in period
burnGR = 1.10

# Maximum G-R statistic for chains to be deemed well-mixed
maxGR = 1.03

def mcmc(likelihood, nwalkers=50, nrun=10000, threads=1, checkinterval=50):
    """Run MCMC

    Run MCMC chains using the emcee EnsambleSampler

    Args:
        likelihood (radvel.likelihood): radvel likelihood object
        nwalkers (int): number of MCMC walkers
        nrun (int): number of steps to take
        threads (int): number of CPU threads to utilize
        checkinterval (int): check MCMC convergence statistics every 
            `checkinterval` steps

    Returns:
        DataFrame: DataFrame containing the MCMC samples

    """
    nrun = int(nrun)
        
    # Get an initial array value
    p0 = likelihood.get_vary_params()
    ndim = p0.size
    p0 = np.vstack([p0]*nwalkers)
    p0 += [np.random.rand(ndim)*1e-5 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler( 
        nwalkers, ndim, likelihood.logprob_array, threads=threads
    )

    pos = p0    
    num_run = int(np.round(nrun / checkinterval))
    totsteps = nrun*nwalkers
    mixcount = 0
    burn_complete = False
    nburn = 0
    t0 = time.time()
    for r in range(num_run):
        t1 = time.time()
        pos, prob, state = sampler.run_mcmc(pos, checkinterval)
        t2 = time.time()

        rate = (checkinterval*nwalkers) / (t2-t1)
        ncomplete = sampler.flatlnprobability.shape[0] + nburn
        pcomplete = ncomplete/float(totsteps) * 100
        ar = sampler.acceptance_fraction.mean() * 100.

        tchains = sampler.chain.transpose()
        
        (ismixed, gr, tz) = gelman_rubin(tchains)
        mintz = min(tz)
        maxgr = max(gr)
        if ismixed: mixcount += 1
        else: mixcount = 0

        # Burn-in complete after maximum G-R statistic first reaches 1.10
        # reset sampler
        if not burn_complete and maxgr <= burnGR:
            sampler.reset()
            msg = (
                "\nDiscarding burn-in now that the chains are marginally "
                "well-mixed\n"
            )
            print msg
            nburn = ncomplete
            burn_complete = True

        if mixcount >= 5:
            tf = time.time()
            tdiff = tf - t0
            tdiff,units = utils.time_print(tdiff)
            msg = (
                "\nChains are well-mixed after {:d} steps! MCMC completed in "
                "{:3.1f} {:s}"
            ).format(ncomplete, tdiff, units)
            print msg
            break
        else:
            msg = (
                "{:d}/{:d} ({:3.1f}%) steps complete; "
                "Running {:.2f} steps/s; Mean acceptance rate = {:3.1f}%; "
                "Min Tz = {:.1f}; Max G-R = {:4.2f} \r"
            ).format(ncomplete, totsteps, pcomplete, rate, ar, mintz, maxgr)
            
            sys.stdout.write(msg)
            sys.stdout.flush()
            
    print "\n"        
    if ismixed and mixcount < 5: 
        msg = (
            "MCMC: WARNING: chains did not pass 5 consecutive convergence "
            "tests. They may be marginally well=mixed."
        )
        print msg
    elif not ismixed: 
        msg = (
            "MCMC: WARNING: chains did not pass convergence tests. They are "
            "likely not well-mixed."
        )
        print msg
    
    df = pd.DataFrame(
        sampler.flatchain,columns=likelihood.list_vary_params()
        )
    df['lnprobability'] = sampler.flatlnprobability
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
