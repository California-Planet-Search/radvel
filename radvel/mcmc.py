import time
import curses
import sys
import os

import multiprocessing as mp

import pandas as pd
import numpy as np

import emcee
import h5py

from radvel import utils
import radvel


class StateVars(object):
    def __init__(self):
        self.oac = 0
        self.autosamples = []
        self.automean = []
        self.automin = []
        self.automax = []
        self.proceed_started = 0

    def reset(self):
        self.__init__()

statevars = StateVars()


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def _closescr():
    if isnotebook() == False:
        try:
            curses.endwin()
        except:
             pass


def _progress_bar(step, totsteps, width=50):
    fltot = float(totsteps)
    numsym = int(np.round(width * (step / fltot)))

    bar = ''.join(["=" for s in range(numsym)])
    bar += ''.join([" " for s in range(width - numsym)])

    msg = "[" + bar + "]"

    return(msg)


def _status_message_NB(statevars):

    msg1 = (
        "{:d}/{:d} ({:3.1f}%) steps complete; "
        "Running {:.2f} steps/s; Mean acceptance rate = {:3.1f}%; "
        "Min Auto Factor = {:3.0f}; Max Auto Relative-Change = {:5.3}; "
        "Min Tz = {:.1f}; Max G-R = {:5.3f}\r"
    ).format(statevars.ncomplete, statevars.totsteps, statevars.pcomplete, statevars.rate, statevars.ar,
             statevars.minafactor, statevars.maxarchange, statevars.mintz, statevars.maxgr)

    sys.stdout.write(msg1)
    sys.stdout.flush()


def _status_message_CLI(statevars):

    statevars.screen = curses.initscr()

    statevars.screen.clear()

    barline = _progress_bar(statevars.ncomplete, statevars.totsteps)

    msg1 = (
            barline + " {:d}/{:d} ({:3.1f}%) steps complete; "
    ).format(statevars.ncomplete, statevars.totsteps, statevars.pcomplete)

    msg2 = (
        "Running {:.2f} steps/s; Mean acceptance rate = {:3.1f}%; "
        "Min Auto Factor = {:3.0f}; \nMax Auto Relative-Change = {:5.3}; "
        "Min Tz = {:.1f}; Max G-R = {:5.3f}\n"
    ).format(statevars.rate, statevars.ar, statevars.minafactor, statevars.maxarchange,
             statevars.mintz, statevars.maxgr)

    statevars.screen.addstr(0, 0, msg1+ '\n' + msg2)

    statevars.screen.refresh()


def convergence_check(minAfactor, maxArchange, maxGR, minTz, minsteps, minpercent):
    """Check for convergence

    Check for convergence for a list of emcee samplers

    Args:
        minAfactor (float): Minimum autocorrelation time factor for chains to be deemed well-mixed and halt the MCMC run
        maxArchange (float): Maximum relative change in the autocorrelative time to be deemed well-mixed and
            halt the MCMC run
        maxGR (float): Maximum G-R statistic for chains to be deemed well-mixed and halt the MCMC run
        minTz (int): Minimum Tz to consider well-mixed
        minsteps (int): Minimum number of steps per walker before convergence tests are performed. Convergence checks
            will start after the minsteps threshold or the minpercent threshold has been hit.
        minpercent (float): Minimum percentage of total steps before convergence tests are performed. Convergence checks
            will start after the minsteps threshold or the minpercent threshold has been hit.
    """

    statevars.ar = 0
    statevars.ncomplete = statevars.nburn
    statevars.tchains = np.empty((statevars.ndim,
                        statevars.samplers[0].get_log_prob(flat=True).shape[0],
                        statevars.ensembles))
    statevars.lnprob = []
    statevars.autocorrelation = []
    statevars.chains = []
    for i,sampler in enumerate(statevars.samplers):
        statevars.ncomplete += sampler.get_log_prob(flat=True).shape[0]
        statevars.ar += sampler.acceptance_fraction.mean() * 100
        statevars.tchains[:,:,i] = sampler.flatchain.transpose()
        statevars.chains.append(sampler.get_chain()[:,:,:].T)
        statevars.lnprob.append(sampler.get_log_prob(flat=True))
    statevars.ar /= statevars.ensembles

    statevars.pcomplete = statevars.ncomplete/float(statevars.totsteps) * 100
    statevars.rate = (statevars.checkinterval*statevars.nwalkers*statevars.ensembles) / statevars.interval

    if statevars.ensembles < 3:
        # if less than 3 ensembles then GR between ensembles does
        # not work so just calculate it on the last sampler
        statevars.tchains = sampler.chain.transpose()

    # Must have completed at least 5% or minsteps steps per walker before
    # attempting to calculate GR
    if statevars.pcomplete < minpercent and sampler.get_log_prob(flat=True).shape[0] <= minsteps*statevars.nwalkers:
        (statevars.ismixed, statevars.minafactor, statevars.maxarchange, statevars.maxgr,
            statevars.mintz) = 0, -1.0, np.inf, np.inf, -1.0
    else:
        (statevars.ismixed, afactor, archange, oac, gr, tz) \
            = convergence_calculate(statevars.tchains, statevars.chains,
                                    oldautocorrelation=statevars.oac, minAfactor=minAfactor, maxArchange=maxArchange,
                                    maxGR=maxGR, minTz=minTz)
        statevars.mintz = min(tz)
        statevars.maxgr = max(gr)
        statevars.minafactor = np.amin(afactor)
        statevars.maxarchange = np.amax(archange)
        statevars.oac = oac

        if statevars.burn_complete:
            statevars.autosamples.append(len(statevars.chains)*statevars.chains[0].shape[2])
            statevars.automean.append(np.mean(statevars.oac))
            statevars.automin.append(np.amin(statevars.oac))
            statevars.automax.append(np.amax(statevars.oac))

        if statevars.ismixed:
            statevars.mixcount += 1
        else:
            statevars.mixcount = 0

    if isnotebook():
        _status_message_NB(statevars)
    else:
        _status_message_CLI(statevars)


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


def mcmc(post, nwalkers=50, nrun=10000, ensembles=8, checkinterval=50, minAfactor=50, maxArchange=.07, burnAfactor=25,
         burnGR=1.03, maxGR=1.01, minTz=1000, minsteps=1000, minpercent=5, thin=1, serial=False, save=True,
         savename=None, proceed=False, proceedname=None):
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
        minAfactor (float): Minimum autocorrelation time factor to deem chains as well-mixed and halt the MCMC run
        maxArchange (float): Maximum relative change in autocorrelation time to deem chains and well-mixed
        burnAfactor (float): Minimum autocorrelation time factor to stop burn-in period. Burn-in ends once burnGr
            or burnAfactor are reached.
        burnGR (float): (optional) Maximum G-R statistic to stop burn-in period. Burn-in ends once burnGr or
            burnAfactor are reached.
        maxGR (float): (optional) Maximum G-R statistic for chains to be deemed well-mixed and halt the MCMC run
        minTz (int): (optional) Minimum Tz to consider well-mixed
        minsteps (int): Minimum number of steps per walker before convergence tests are performed. Convergence checks
            will start after the minsteps threshold or the minpercent threshold has been hit.
        minpercent (float): Minimum percentage of total steps before convergence tests are performed. Convergence checks
            will start after the minsteps threshold or the minpercent threshold has been hit.
        thin (int): (optional) save one sample every N steps (default=1, save every sample)
        serial (bool): set to true if MCMC should be run in serial
        save (bool): set to true to save MCMC chains that can be continued in a future run
        savename (string): location of h5py file where MCMC chains will be saved for future use
        proceed (bool): set to true to continue a previously saved run
        proceedname (string): location of h5py file with previously MCMC run chains
    Returns:
        DataFrame: DataFrame containing the MCMC samples
    """

    try:
        if save and savename is None:
            raise ValueError('save set to true but no savename provided')

        if save:
            h5f = h5py.File(savename, 'a')

        if proceed:
            if proceedname is None:
                raise ValueError('proceed set to true but no proceedname provided')
            else:
                h5p = h5py.File(savename, 'r')
                msg = 'Loading chains and run information from previous MCMC'
                print(msg)
            if len(h5p.keys()) != (3*ensembles + 6):
                raise ValueError('number of ensembles must be equal to that from previous run: {}'.format(((len(h5f.keys()) - 6)/3)))
            statevars.prechains = []
            statevars.prelog_probs = []
            statevars.preaccepted = []
            statevars.preburned = h5p['burned'][0]
            statevars.minafactor = h5p['crit'][0]
            statevars.maxarchange = h5p['crit'][1]
            statevars.mintz = h5p['crit'][2]
            statevars.maxgr = h5p['crit'][3]
            statevars.autosamples = list(h5p['autosample'])
            statevars.automin = list(h5p['automin'])
            statevars.automean = list(h5p['automean'])
            statevars.automax = list(h5p['automax'])
            for i in range(0,int((len(h5p.keys()) - 6)/3)):
                str_chain = str(i) + '_chain'
                str_log_prob = str(i) + '_log_prob'
                str_accepted = str(i) + '_accepted'
                statevars.prechains.append(h5p[str_chain])
                statevars.prelog_probs.append(h5p[str_log_prob])
                statevars.preaccepted.append(h5p[str_accepted])

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

        statevars.ensembles = ensembles
        statevars.nwalkers = nwalkers
        statevars.checkinterval = checkinterval - 1

        nrun = int(nrun)

        # Get an initial array value
        pi = post.get_vary_params()
        statevars.ndim = pi.size

        if nwalkers < 2*statevars.ndim:
            print("WARNING: Number of walkers is less than 2 times number of free parameters. Adjusting number of walkers to {}".format(2*statevars.ndim))
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

        statevars.samplers = []
        statevars.samples = []
        statevars.initial_positions = []
        for e in range(ensembles):
            pi = post.get_vary_params()
            p0 = np.vstack([pi]*statevars.nwalkers)
            p0 += [np.random.rand(statevars.ndim)*pscales for i in range(statevars.nwalkers)]
            if not proceed:
                statevars.initial_positions.append(p0)
            else:
                statevars.initial_positions.append(statevars.prechains[i][-1,:,:])
            statevars.samplers.append(emcee.EnsembleSampler(statevars.nwalkers, statevars.ndim, post.logprob_array,
                                                                threads=1))

        if proceed:
            for i, sampler in enumerate(statevars.samplers):
                sampler.backend.grow(statevars.prechains[i].shape[0], None)
                sampler.backend.chain = statevars.prechains[i]
                sampler.backend.log_prob = statevars.prelog_probs[i]
                sampler.backend.accepted = statevars.preaccepted[i]
                sampler.backend.iteration = statevars.prechains[i].shape[0]

        num_run = int(np.round(nrun / (checkinterval -1)))
        statevars.totsteps = nrun*statevars.nwalkers*statevars.ensembles
        statevars.mixcount = 0
        statevars.ismixed = 0
        if proceed == True and statevars.preburned != 0:
            statevars.burn_complete = True
            statevars.nburn = statevars.preburned
        else:
            statevars.burn_complete = False
            statevars.nburn = 0
        statevars.ncomplete = statevars.nburn
        statevars.pcomplete = 0
        statevars.rate = 0
        statevars.ar = 0
        statevars.minAfactor = -1
        statevars.maxArchange = np.inf
        statevars.mintz = -1
        statevars.maxgr = np.inf
        statevars.t0 = time.time()

        for r in range(num_run):
            t1 = time.time()
            mcmc_input_array = []
            for i, sampler in enumerate(statevars.samplers):
                if sampler.iteration <= 1 or statevars.proceed_started == 0:
                    p1 = statevars.initial_positions[i]
                    statevars.proceed_started = 1
                else:
                    p1 = sampler.get_last_sample()
                for sample in sampler.sample(p1, store=True):
                    mcmc_input = (sampler, p1, (checkinterval - 1))
                    mcmc_input_array.append(mcmc_input)

            if serial:
                statevars.samplers = []
                for i in range(ensembles):
                    result = _domcmc(mcmc_input_array[i])
                    statevars.samplers.append(result)
            else:
                pool = mp.Pool(statevars.ensembles)
                statevars.samplers = pool.map(_domcmc, mcmc_input_array)
                pool.close()  # terminates worker processes once all work is done
                pool.join()   # waits for all processes to finish before proceeding

            t2 = time.time()
            statevars.interval = t2 - t1

            convergence_check(minAfactor=minAfactor, maxArchange=maxArchange, maxGR=maxGR, minTz=minTz,
                          minsteps=minsteps, minpercent=minpercent)

            if save==True:
                for i, sampler in enumerate(statevars.samplers):
                    str_chain = str(i) + '_chain'
                    str_log_prob = str(i) + '_log_prob'
                    str_accepted = str(i) + '_accepted'
                    if str_chain in h5f.keys():
                        del h5f[str_chain]
                    if str_log_prob in h5f.keys():
                        del h5f[str_log_prob]
                    if str_accepted in h5f.keys():
                        del h5f[str_accepted]
                    if 'crit' in h5f.keys():
                        del h5f['crit']
                    if 'autosample' in h5f.keys():
                        del h5f['autosample']
                    if 'automin' in h5f.keys():
                        del h5f['automin']
                    if 'automean' in h5f.keys():
                        del h5f['automean']
                    if 'automax' in h5f.keys():
                        del h5f['automax']
                    if 'burned' in h5f.keys():
                        del h5f['burned']
                    h5f.create_dataset(str_chain, data=sampler.get_chain())
                    h5f.create_dataset(str_log_prob, data=sampler.get_log_prob())
                    h5f.create_dataset(str_accepted, data=sampler.backend.accepted)
                    h5f.create_dataset('crit', data=[statevars.minafactor, statevars.maxarchange, statevars.mintz,
                                                     statevars.maxgr])
                    h5f.create_dataset('autosample', data=statevars.autosamples)
                    h5f.create_dataset('automin', data=statevars.automin)
                    h5f.create_dataset('automean', data=statevars.automean)
                    h5f.create_dataset('automax', data=statevars.automax)
                    if statevars.burn_complete==True:
                        h5f.create_dataset('burned', data=[statevars.nburn])
                    else:
                        h5f.create_dataset('burned', data=[0])

            # Burn-in complete after maximum G-R statistic first reaches burnGR or minAfactor reaches burnAfactor
            # reset samplers
            if not statevars.burn_complete and (statevars.maxgr <= burnGR or burnAfactor <= statevars.minafactor):
                for i, sampler in enumerate(statevars.samplers):
                    statevars.initial_positions[i] = sampler.get_last_sample()
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
                _closescr()
                print(msg)
                break

        print("\n")
        if statevars.ismixed and statevars.mixcount < 5:
            msg = (
                "MCMC: WARNING: chains did not pass 5 consecutive convergence "
                "tests. They may be marginally well=mixed."
            )
            _closescr()
            print(msg)
        elif not statevars.ismixed:
            msg = (
                "MCMC: WARNING: chains did not pass convergence tests. They are "
                "likely not well-mixed."
            )
            _closescr()
            print(msg)

        preshaped = np.dstack(statevars.chains)
        df = pd.DataFrame(
            preshaped.reshape(preshaped.shape[0], preshaped.shape[1]*preshaped.shape[2]).transpose(),
            columns=post.list_vary_params())
        df['lnprobability'] = np.hstack(statevars.lnprob)
        df = df.iloc[::thin]

        statevars.factor = [minAfactor] * len(statevars.autosamples)

        return df

    except KeyboardInterrupt:
        curses.endwin()


def convergence_calculate(pars0, chains, oldautocorrelation, minAfactor, maxArchange, minTz, maxGR):
    """Calculate Convergence Criterion

    Calculates the Gelman-Rubin statistic, autocorrelation time factor,
    relative change in autocorrellation time, and the number of
    independent draws for each parameter, as defined by Ford et
    al. (2006) (http://adsabs.harvard.edu/abs/2006ApJ...642..505F).
    The chain is considered well-mixed if all parameters have a
    Gelman-Rubin statistic of <= 1.03, the min autocorrelation time factor >= 75,
    a max relative change in autocorrelation time <= .01, and >= 1000 independent draws.

    Args:
        pars0 (array): A 3 dimensional array (NPARS,NSTEPS,NCHAINS) of
            parameter values
        chains (array): A 3 dimensional array of parameter values shaped to calculate
            autocorrelation time
        oldautocorrelation (float): previously calculated autocorrelation time
        minAfactor (float): minimum autocorrelation
            time factor to consider well-mixed
        maxArchange (float): maximum relative change in
            autocorrelation time to consider well-mixed
        minTz (int): minimum Tz to consider well-mixed
        maxGR (float): maximum Gelman-Rubin statistic to
            consider well-mixed
    Returns:
        tuple: tuple containing:
            ismixed (bool):
                Are the chains well-mixed?
            afactor (array):
                A matrix containing the
                autocorrelation time factor for each parameter and ensemble combination
            archange (matrix):
                A matrix containing the relative
                change in the autocorrelation time factor for each parameter and ensemble combination
            autocorrelation (matrix):
                A matrix containing the autocorrelation time for each parameter and ensemble combination
            gelmanrubin (array):
                An NPARS element array containing the
                Gelman-Rubin statistic for each parameter (equation
                25)
            Tz (array):
                An NPARS element array containing the number
                of independent draws for each parameter (equation 26)

    History:
        2010/03/01:
            Written: Jason Eastman - The Ohio State University
        2012/10/08:
            Ported to Python by BJ Fulton - University of Hawaii,
            Institute for Astronomy
        2016/04/20:
            Adapted for use in RadVel. Removed "angular" parameter.
        2019/10/24:
            Adapted to calculate and consider autocorrelation times
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

    chains = np.dstack(chains)
    chains = np.swapaxes(chains, 0, 2)

    autocorrelation = emcee.autocorr.integrated_time(chains, tol = 0)

    afactor = np.divide(chains.shape[0], autocorrelation)

    archange = np.divide(np.abs(np.subtract(autocorrelation, oldautocorrelation)), oldautocorrelation)

    # well-mixed criteria
    ismixed = min(tz) > minTz and max(gelmanrubin) < maxGR and np.amin(afactor) > minAfactor and np.amax(archange) < maxArchange

    return (ismixed, afactor, archange, autocorrelation, gelmanrubin, tz)
