import emcee
import pandas as pd
import numpy as np
from scipy import optimize
import sys

def mcmc(likelihood, nwalkers=50, nburn=1000, nrun=10000, threads=1, checkinterval=100):
    """Run MCMC

    Run MCMC chains using emcee

    Args:
        likelihood (radvel.likelihood): radvel likelihood object
        nwalkers (int): number of MCMC walkers
        nburn (int): number of burn-in steps to run and discard before starting the full chains
        nrun (int): number of steps to take
        threads (int): number of CPU threads to utilize
        checkinterval (int): check MCMC convergence statistics every `checkinterval` steps

    Returns:
        DataFrame: DataFrame containing the MCMC samples

    """
    
    # Get an initial array value
    p0 = likelihood.get_vary_params()
    ndim = p0.size
    p0 = np.vstack([p0]*nwalkers)
    p0 += [np.random.rand(ndim)*0.0001 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler( 
        nwalkers, 
        ndim, 
        likelihood.logprob_array, 
        threads=threads
        )

    # Run the MCMC
    # Burn-in first
    print "Performing burn-in..."
    pos,prob,state = sampler.run_mcmc(p0,nburn)
    sampler.reset()
    print "Discarding burn-in"

    num_run = int(np.round(nrun / checkinterval))
    totsteps = nrun*nwalkers
    mixcount = 0
    for r in range(num_run):
        pos, prob, state = sampler.run_mcmc(pos, checkinterval)

        ncomplete = sampler.flatlnprobability.shape[0]
        pcomplete = ncomplete/float(totsteps) * 100
        ar = sampler.acceptance_fraction.mean() * 100.
        tchains = sampler.chain.transpose()

        (ismixed, gr, tz) = gelman_rubin(tchains)
        mintz = min(tz)
        maxgr = max(gr)
        if ismixed: mixcount += 1
        else: mixcount = 0

        if mixcount >= 5:
            print "\nChains are well-mixed after %d steps! MCMC complete" % ncomplete
            break
        else:
            sys.stdout.write("%d/%d (%3.1f%%) steps complete; Mean acceptance rate = %3.1f%%; Min Tz = %.1f; Max G-R = %4.2f \r" % (ncomplete, totsteps, pcomplete, ar, mintz, maxgr))
            sys.stdout.flush()

    print "\n"        
    if ismixed and mixcount < 5: print "MCMC: WARNING: chains did not pass 5 consecutive convergence tests. They may be marginally well=mixed."
    elif not ismixed: print "MCMC: WARNING: chains did not pass convergence tests. They are likely not well-mixed."
            
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
        chain (DataFrame): pandas DataFrame with different values from MCMC chain
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


def gelman_rubin(pars0, minTz=1000, maxGR=1.10):
    """Gelman-Rubin Statistic

    Calculates the Gelman-Rubin statistic and the number of
    independent draws for each parameter, as defined by Ford et al. (2006) (http://adsabs.harvard.edu/abs/2006ApJ...642..505F).
    The chain is considered well-mixed if all parameters have a
    Gelman-Rubin statistic of <= 1.10 and >= 1000 independent draws.

    History:
        2010/03/01 - Written: Jason Eastman - The Ohio State University
        
        2012/10/08 - Ported to Python by BJ Fulton - University of Hawaii, Institute for Astronomy
        
        2016/04/20 - Adapted for use in radvel. Removed "angular" parameter.

    Args:
        pars0 (array): A 3 dimensional array (NPARS,NSTEPS,NCHAINS) of
                 parameter values
        minTz (int): (optional) minimum Tz to consider well-mixed
        maxGR (float): (optional) maximum Gelman-Rubin statistic to consider well-mixed

    Returns:
             tuple :
                     :ismixed (bool): Are the chains well-mixed?
                     
                     :G-R (array): An NPARS element array containing the Gelman-Rubin
                         statistic for each parameter (equation 25)
                         
                     :Tz (array): An NPARS element array containing the number of
                         independent draws for each parameter (equation 26)
"""


    pars = pars0.copy() # don't modify input parameters
    
    sz = pars.shape
    assert pars.ndim == 3, 'MCMC: GELMAN_RUBIN: ERROR: pars must have 3 dimensions'

    npars = float(sz[0])
    nsteps = float(sz[1])
    nchains = float(sz[2])

    assert nsteps > 1, 'MCMC: GELMAN_RUBIN: ERROR: NSTEPS must be greater than 1'

    # Equation 21: W(z) in Ford 2006
    variances = np.var(pars,axis=1, dtype=np.float64)
    meanofvariances = np.mean(variances,axis=1)
    withinChainVariances = np.mean(variances, axis=1)
    
    # Equation 23: B(z) in Ford 2006
    means = np.mean(pars,axis=1)
    betweenChainVariances = np.var(means,axis=1, dtype=np.float64) * nsteps
    varianceofmeans = np.var(means,axis=1, dtype=np.float64) / (nchains-1)
    varEstimate = (1 - 1./nsteps) * withinChainVariances + (1./nsteps) * betweenChainVariances
    
    bz = varianceofmeans*nsteps

    # Equation 24: varhat+(z) in Ford 2006
    varz = (nsteps-1.)/bz + varianceofmeans

    # Equation 25: Rhat(z) in Ford 2006
    gelmanrubin = np.sqrt(varEstimate/withinChainVariances)

    # Equation 26: T(z) in Ford 2006
    vbz = varEstimate / bz
    tz = nchains*nsteps*vbz[vbz < 1]
    if tz.size == 0: tz = [-1]

    # well-mixed criteria
    ismixed = min(tz) > minTz and max(gelmanrubin) < maxGR
        
    return (ismixed, gelmanrubin, tz)
