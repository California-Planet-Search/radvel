import sys
import time
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import emcee
print(emcee.__version__)

import gc
gc.enable()

from radvel import utils
import radvel

class StateVars(object):
    def __init__(self):
        pass

statevars = StateVars()

def _status_message(statevars):
    msg = (
        "{:d}/{:d} ({:3.1f}%) steps complete; "
        "Running {:.2f} steps/s; Mean acceptance rate = {:3.1f}%; "
        "Convergence Progress: Is max(tau_ac={:5.3f})*(tau_ac multiplier={:d})<(Nsteps={:d})? Is max rel. change in (tau_ac={:6.4f})<(threshold act={:6.4f})?\r"
    ).format(statevars.ncomplete, statevars.totsteps, statevars.pcomplete,
                 statevars.rate, statevars.ar, statevars.maxautocorrnow,int(statevars.autocorrmin),statevars.ncomplete,statevars.maxrelthresh,statevars.autocorrrelthreshmax)
    sys.stdout.write(msg)
    sys.stdout.flush()
    if statevars.ncomplete>0:
        n = statevars.checkinterval*np.arange(1,statevars.ncomplete/statevars.checkinterval+1)
        y= np.asarray(statevars.avgautocorr)
        y2= np.asarray(statevars.maxautocorr)
        y3= np.asarray(statevars.minautocorr)
        plt.plot(n, n / statevars.autocorrmin, "--k")
        plt.plot(n, y)
        plt.plot(n, y2)
        plt.plot(n, y3)
        plt.xlim(0, n.max())
        if np.isfinite(y2.max()) and np.isfinite(y3.min()):
            plt.ylim(0, y2.max() + 0.1*(y2.max() - y3.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"min,mean,max ${\tau}$ (solid); nsteps/${\tau}$_ac_scale=${\tau}$_ac (dashed)");
        plt.savefig("MCMCprogress.png")
        plt.clf()

def convergence_check(sampler, autocorrmin, minsteps):
    """Check for convergence
    Check for convergence for the emcee sampler, using get_autocorrtime
    
    Args:
        samplers (list): List of emcee sampler objects
        autocorrmin (float):  Minimum ration for the autocorrelation time-scale to the number of steps.
        minsteps (int): Minimum number of steps per walker before convergence tests are performed
    """
    msg = ("DEBUG: Convergence check: pre-amble.")
    print(msg)    
    statevars.ar = 0
    statevars.ncomplete=sampler.backend.iteration # *statevars.nwalkers
    statevars.ar = sampler.acceptance_fraction.mean() * 100
    statevars.pcomplete = float(statevars.ncomplete)/float(statevars.totsteps)*100 
    statevars.rate = (statevars.checkinterval*statevars.nwalkers) / statevars.interval 

    # Must have completed at least 5% or 1000 steps per walker before attempting to calculate convergence
    # emcee.backend will error out if ncomplete < numwalkers*50 with tol=0 not specified; tol=50 is default:
    # emcee.autocorr.AutocorrError: The chain is shorter than 50 times the integrated autocorrelation time for 25 parameter(s). Use this estimate with caution and run a longer chain!
    minsteps=np.max([50,minsteps])
    #msg = ("DEBUG: ncomplete: {:d}; pcomplete: {:3.1f}; minsteps: {:d}; nwalkers {:d}").format(statevars.ncomplete, statevars.pcomplete,minsteps,statevars.nwalkers)
    #print(msg)

    msg = ("DEBUG: Convergence check: getting log prob.")
    print(msg)
    statevars.lnprob = sampler.get_log_prob(discard=statevars.nburn,flat=True,thin=statevars.thin)
    msg = ("DEBUG: Convergence check: getting auto-corr time.")
    print(msg)
    statevars.autocorrnow = sampler.get_autocorr_time(discard=statevars.nburn,thin=statevars.thin,tol=0,quiet=True)
    msg = ("DEBUG: autocorr values:")
    print(msg)
    print(str(statevars.autocorrnow))
    msg = ("DEBUG: Convergence check: math.")
    print(msg)
    statevars.avgautocorrnow=np.mean(statevars.autocorrnow)
    statevars.minautocorrnow=np.min(statevars.autocorrnow)
    statevars.maxautocorrnow=np.max(statevars.autocorrnow)
    statevars.avgautocorr.append(statevars.avgautocorrnow)
    statevars.minautocorr.append(statevars.minautocorrnow)
    statevars.maxautocorr.append(statevars.maxautocorrnow)
    statevars.avgrelthresh = np.abs(statevars.avgautocorrold-statevars.avgautocorrnow)/statevars.avgautocorrnow
    statevars.maxrelthresh = np.abs(statevars.maxautocorrold-statevars.maxautocorrnow)/statevars.maxautocorrnow
    statevars.minrelthresh = np.abs(statevars.minautocorrold-statevars.minautocorrnow)/statevars.minautocorrnow
    
    # this is the actual convergence check following the emcee v3rc2 readthedocs:
    # in order to be converged, the MAX autocorr timescale must be < nsteps/ac-multipler-min, e.g., nsteps > MAX autocorr* ac-multipler min (MAX is over all parameters).
    # in order to be converged, the relative change in the auto corr timescale must be < the threshold, with 0.01 conservative...

    msg = ("DEBUG C: ncomplete: {:d}; pcomplete: {:3.1f}; minsteps: {:d}; nwalkers {:d}").format(statevars.ncomplete, statevars.pcomplete,minsteps,statevars.nwalkers)
    print(msg)
    if statevars.pcomplete > 5 and statevars.ncomplete > minsteps:
        statevars.converged = np.all(statevars.autocorrnow * statevars.autocorrmin < statevars.ncomplete)
        statevars.converged &= np.all(np.abs(statevars.autocorrold-statevars.autocorrnow)/statevars.autocorrnow < statevars.autocorrrelthreshmax)
        print(str(statevars.converged))
        print(str(statevars.autocorrnow))
        print(str(statevars.autocorrold))
     
    statevars.autocorrold=statevars.autocorrnow    
    statevars.avgautocorrold=statevars.avgautocorrnow
    statevars.maxautocorrold=statevars.maxautocorrnow
    statevars.minautocorrold=statevars.minautocorrnow
    _status_message(statevars)

def _domcmc(input_tuple):
    """Function to be run in parallel on different CPUs
    Input is a tuple: first element is an emcee sampler object, second is an array of 
    initial positions, third is number of steps to run before doing a convergence check
    """
    sampler = input_tuple[0]
    ipos = input_tuple[1]
    check_interval = input_tuple[2]
    sampler.run_mcmc(ipos, check_interval)
    return sampler

def mcmc(post, nwalkers=30, nrun=10000, ensembles=8, checkinterval=200, nburn=0, autocorrmin=100, autocorrrelthreshmax=0.01, minsteps=100, thin=1, serial=False):
    """Run MCMC
    Run MCMC chains using the emcee EnsambleSampler
    Args:
        post (radvel.posterior): radvel posterior object
        nwalkers (int): (optional) number of MCMC walkers
        nrun (int): (optional) number of steps to take
        ensembles (int): (optional) number of ensembles to run. Will be run
            in parallel on separate CPUs
        checkinterval (int): (optional) check MCMC convergence statistics every
            `checkinterval` steps
        nburn (int): (optional) burn the initial part of the MCMC for mixing.
        autocorrmin (float): (optional) MCMC convergence criteria reached if autocorrelation time is > autocorrmin * Nsteps 
        thin (int): (optional) save one sample every N steps (default=1, save every sample)
        serial (bool): set to true if MCMC should be run in serial
    Returns:
        DataFrame: DataFrame containing the MCMC samples
    """
    # check if one or more likelihoods are GPs
    if isinstance(post.likelihood, radvel.likelihood.CompositeLikelihood):
        check_gp = [like for like in post.likelihood.like_list if isinstance(like, radvel.likelihood.GPLikelihood)]
    else:
        check_gp = isinstance(post.likelihood, radvel.likelihood.GPLikelihood)  

    np_info = np.__config__.blas_opt_info
    if 'extra_link_args' in np_info.keys() \
       and check_gp \
       and ('-Wl,Accelerate' in np_info['extra_link_args']) \
       and serial == False:
        print("WARNING: Parallel processing with Gaussian Processes will not work with your current"
                      + " numpy installation. See radvel.readthedocs.io/en/latest/OSX-multiprocessing.html"
                      + " for more details. Running in serial with " + str(ensembles) + " ensembles.")
        serial = True

    nrun = int(nrun)
        
    statevars.ensembles = ensembles
    statevars.nwalkers = nwalkers
    statevars.checkinterval = checkinterval 
    statevars.thin = thin

    statevars.autocorrrelthreshmax=autocorrrelthreshmax
    statevars.autocorrmin=autocorrmin
    statevars.autocorrnow=np.inf
    statevars.autocorrold=np.inf
    statevars.avgautocorr=[]
    statevars.avgautocorrnow=np.inf 
    statevars.avgautocorrold=np.inf
    statevars.maxautocorr=[]
    statevars.maxautocorrnow=np.inf 
    statevars.maxautocorrold=np.inf
    statevars.minautocorr=[]
    statevars.minautocorrnow=np.inf 
    statevars.minautocorrold=np.inf
    statevars.converged=False
    
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
            elif par.startswith('logper'):
                pscale = np.abs(1e-5 * val)
            elif par.startswith('tc'):
                pscale = 0.1
            else:
                pscale = np.abs(0.10 * val)
            post.params[par].mcmc_scale = pscale
        else:
            pscale = post.params[par].mcmcscale
        pscales.append(pscale)
    pscales = np.array(pscales)

    #statevars.samplers = []
    #statevars.initial_positions = []
    
    # PPP: 12/5/2018 - move over to backend for chains for each ensemble.
    #filename = "PPPchains_ensemble_"+str(e)+".h5"
    filename = "PPPchains.h5"
    backend = emcee.backends.HDFBackend(filename, name='mcmc',read_only=False)
    backend.reset(statevars.nwalkers, statevars.ndim)

    pi = post.get_vary_params()
    p0 = np.vstack([pi]*statevars.nwalkers)
    p0 += [np.random.rand(statevars.ndim)*pscales for i in range(statevars.nwalkers)]
    statevars.initial_positions=p0
    if serial:
        statevars.sampler=emcee.EnsembleSampler(statevars.nwalkers, statevars.ndim, post.logprob_array,backend=backend)
    else:
        pool = mp.Pool(statevars.ensembles)
        statevars.sampler=emcee.EnsembleSampler(statevars.nwalkers, statevars.ndim, post.logprob_array,pool=pool,backend=backend)

    num_run = int(np.round(nrun / checkinterval))
    statevars.totsteps = nrun # *statevars.nwalkers #*statevars.ensembles
    statevars.mixcount = 0
    statevars.ncomplete = 0
    statevars.pcomplete = 0
    statevars.rate = 0
    statevars.ar = 0
    statevars.t0 = time.time()
    statevars.nburn = 0

    #print(statevars.samplers)

    # THIS is the loop that needs better memory management. PPP
    for r in range(num_run):
        t1 = time.time()
        #gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
        gc.collect()
        #print("Uncollectable garbage", gc.garbage)
        mcmc_input = (statevars.sampler, statevars.initial_positions, checkinterval)
        statevars.sampler = _domcmc(mcmc_input)
        t2 = time.time()
        statevars.interval = t2 - t1

        convergence_check(statevars.sampler, autocorrmin, minsteps=minsteps)

        if nburn>0:
            if statevars.ncomplete > nburn*statevars.nwalkers:  # burn-in complete
                statevars.nburn=nburn
        if statevars.converged:
            tf = time.time()
            tdiff = tf - statevars.t0
            tdiff,units = utils.time_print(tdiff)
            msg = (
                "\nChains are converged after {:d} steps! MCMC completed in "
                "{:3.1f} {:s}"
            ).format(statevars.ncomplete, tdiff, units)
            print(msg)
            break
    
    # END FOR LOOP
    print("\n")
    if statevars.converged==False: 
        msg = ("MCMC: WARNING: chains did not converge.")
        print(msg)
        
    chain=statevars.sampler.backend.get_chain(discard=statevars.nburn,flat=False,thin=statevars.thin)
    [nsteps,nchains,ndim] = chain.shape
    df = pd.DataFrame(chain.reshape(nsteps*nchains,ndim),columns=post.list_vary_params())
    df['lnprobability'] = np.hstack(statevars.lnprob)

    return df