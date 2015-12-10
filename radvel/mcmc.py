import emcee
import pandas as pd
import numpy as np
from scipy import optimize

def mcmc(mod, t, vel, errvel, nwalkers=200, nburn=100, nrun=200, threads=8):
    p0 = mod.params_to_array(mod.params0)
    ndim = p0.size
    p0 = np.vstack([p0]*nwalkers)
    p0 += [np.random.rand(ndim)*0.0001 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler( 
        nwalkers, 
        ndim, 
        mod.lnprob_array, 
        args=(t, vel, errvel,),
        threads=threads
        )

    # Run the MCMC
    pos,prob,state = sampler.run_mcmc(p0,nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nrun)

    df = pd.DataFrame(sampler.flatchain,columns=mod.vary_parameters)
    df['lnprobability'] = sampler.flatlnprobability
    return df

def draw_models_from_chain(mod, chain, t, nsamples=50):
    """
    Draw Models from Chain
    
    Given an MCMC chain of parameters, draw representative parameters
    and synthesize models.

    Parameters
    ----------
    mod : RV model
    chain : pandas DataFrame with different values from MCMC chain
    t : time range over which to synthesize models
    nsamples : number of draws
    
    Returns
    -------
    models : 2D array with the different models as different rows
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


def test_mcmc():
    t, vel, errvel, mod = generate_synthetic_single_planet_model()

    def obj(*args):
        return -1.0 * mod.lnprob_array(*args) 

    p0 = mod.params_to_array()
    p1 = optimize.fmin_l_bfgs_b(obj, p0, args=(t, vel, errvel,) ,approx_grad=1)
    p1 = p1[0]
    mod.params0 = mod.array_to_params(p1)

    df = mcmc(mod, t, vel, errvel, nwalkers=200, nburn=200, nrun=200, threads=8)

    corner.corner(
        df[mod.vary_parameters],
        labels=mod.vary_parameters,
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        smooth=False,
        bins=40,
        levels=[0.68,0.95],
        )

    return df

