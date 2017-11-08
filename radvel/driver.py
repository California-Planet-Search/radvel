"""
Driver functions for the radvel pipeline.\
These functions are meant to be used only with\
the `cli.py` command line interface.
"""
from __future__ import print_function
import os
import sys
import copy
from collections import OrderedDict
if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser
import pandas as pd

import numpy as np 

import radvel


def plots(args):
    """
    Generate plots

    Args:
        args (ArgumentParser): command line arguments
    """
    
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(
        args.outputdir,"{}_radvel.stat".format(conf_base)
    )

    status = load_status(statfile)

    assert status.getboolean('fit', 'run'), \
      "Must perform max-liklihood fit before plotting"
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    for ptype in args.type:
        print("Creating {} plot for {}".format(ptype, conf_base))

        if ptype == 'rv':
            args.plotkw['uparams'] = post.uparams
            saveto = os.path.join(
                args.outputdir,conf_base+'_rv_multipanel.pdf'
            )
            P,_ = radvel.utils.initialize_posterior(config_file)
            if hasattr(P, 'bjd0'):
                args.plotkw['epoch'] = P.bjd0
            radvel.plotting.rv_multipanel_plot(
                post, saveplot=saveto, **args.plotkw
            )

        if ptype == 'corner' or ptype == 'trend':
            assert status.getboolean('mcmc', 'run'), \
            "Must run MCMC before making corner or trend plots"

            chains = pd.read_csv(status.get('mcmc', 'chainfile'))

        if ptype == 'corner':
            saveto = os.path.join(args.outputdir, conf_base+'_corner.pdf')
            radvel.plotting.corner_plot(post, chains, saveplot=saveto)

        if ptype == 'trend':
            nwalkers = status.getint('mcmc', 'nwalkers')

            saveto = os.path.join(args.outputdir, conf_base+'_trends.pdf')
            radvel.plotting.trend_plot(post, chains, nwalkers, saveto)

        if ptype == 'derived':
            assert status.has_section('derive'), \
            "Must run `radvel derive` before plotting derived parameters"

            P,_ = radvel.utils.initialize_posterior(config_file)
            chains = pd.read_csv(status.get('derive', 'chainfile'))
            saveto = os.path.join(
                args.outputdir,conf_base+'_corner_derived_pars.pdf'
            )
            radvel.plotting.corner_plot_derived_pars(
                chains, P, saveplot=saveto
            )

        savestate = {'{}_plot'.format(ptype): os.path.abspath(saveto)}
        save_status(statfile, 'plot', savestate)
            
        
def fit(args):
    """Perform maximum-likelihood fit

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    print("Performing max-likelihood fitting for {}".format(conf_base))

    P, post = radvel.utils.initialize_posterior(config_file, decorr=args.decorr)

    post = radvel.fitting.maxlike_fitting(post, verbose=True)
    
    postfile = os.path.join(args.outputdir,
                            '{}_post_obj.pkl'.format(conf_base))
    post.writeto(postfile)
    
    savestate = {'run': True,
                 'postfile': os.path.abspath(postfile)}
    save_status(os.path.join(args.outputdir,
                             '{}_radvel.stat'.format(conf_base)),
                             'fit', savestate)


def mcmc(args):
    """Perform MCMC error analysis

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.outputdir,
                            "{}_radvel.stat".format(conf_base))

    status = load_status(statfile)

    if status.getboolean('fit', 'run'):
        print("Loading starting positions from previous max-likelihood fit")

        post = radvel.posterior.load(status.get('fit', 'postfile'))
    else:
        P, post = radvel.utils.initialize_posterior(config_file,
                                                        decorr=args.decorr)

    msg = "Running MCMC for {}, N_walkers = {}, N_steps = {}, N_ensembles = {} ...".format(
        conf_base, args.nwalkers, args.nsteps, args.ensembles)
    print(msg)

    chains = radvel.mcmc(
            post, nwalkers=args.nwalkers, nrun=args.nsteps, ensembles=args.ensembles)

    # Convert chains into CPS basis
    cpschains = chains.copy()
    for par in post.params.keys():
        if not post.params[par].vary:
            cpschains[par] = post.params[par].value

    cpschains = post.params.basis.to_cps(cpschains)
    cps_quantile = cpschains.quantile([0.159, 0.5, 0.841])

    # Get quantiles and update posterior object to median 
    #   values returned by MCMC chains
    post_summary=chains.quantile([0.159, 0.5, 0.841])        

    for k in chains.keys():
        if k in post.params.keys():
            post.params[k].value = post_summary[k][0.5]

    print("Performing post-MCMC maximum likelihood fit...")
    post = radvel.fitting.maxlike_fitting(post, verbose=False)

    cpspost = copy.deepcopy(post)
    cpsparams = post.params.basis.to_cps(post.params)
    cpspost.params.update(cpsparams)


    print("Calculating uncertainties...")
    cpspost.uparams = {}
    cpspost.medparams = {}
    cpspost.maxparams = {}
    for par in cpspost.params.keys():
        maxlike = cpspost.params[par].value
        med = cps_quantile[par][0.5]
        high = cps_quantile[par][0.841] - med
        low = med - cps_quantile[par][0.159]
        err = np.mean([high,low])
        err = radvel.utils.round_sig(err)
        med, err, errhigh = radvel.utils.sigfig(med, err)
        maxlike, err, errhigh = radvel.utils.sigfig(maxlike, err)
        cpspost.uparams[par] = err
        cpspost.medparams[par] = med
        cpspost.maxparams[par] = maxlike


    print("Final loglikelihood = %f" % post.logprob())
    print("Final RMS = %f" % post.likelihood.residuals().std())
    print("Best-fit parameters:")
    print(cpspost)

    print("Saving output files...")
    saveto = os.path.join(args.outputdir, conf_base+'_post_summary.csv')
    post_summary.to_csv(saveto, sep=',')

    postfile = os.path.join(args.outputdir,
                            '{}_post_obj.pkl'.format(conf_base))
    cpspost.writeto(postfile)

    csvfn = os.path.join(args.outputdir, conf_base+'_chains.csv.tar.bz2')
    chains.to_csv(csvfn, compression='bz2')


    savestate = {'run': True,
                 'postfile': os.path.abspath(postfile),
                 'chainfile': os.path.abspath(csvfn),
                 'summaryfile': os.path.abspath(saveto),
                 'nwalkers': args.nwalkers,
                 'nsteps': args.nsteps}
    save_status(statfile, 'mcmc', savestate)


def bic(args):
    """Compare different models and comparative statistics

    Args:
        args (ArgumentParser): command line arguments
    """

    
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.outputdir,
                            "{}_radvel.stat".format(conf_base))


    status = load_status(statfile)
    savestate = {}

    assert status.getboolean('fit', 'run'), \
      "Must perform max-liklihood fit before running BIC comparisons"
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    for btype in args.type:
        print("Performing bic comparison: {}".format(btype))

        if btype == 'nplanets':
            statsdict = radvel.fitting.model_comp(post, verbose=False)
            savestate['nplanets'] = statsdict


    save_status(statfile, 'bic', savestate)


def tables(args):
    """Generate TeX code for tables in summary report

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.outputdir,
                            "{}_radvel.stat".format(conf_base))
    status = load_status(statfile)

    assert status.getboolean('mcmc', 'run'), \
        "Must run MCMC before making tables"

    P, post = radvel.utils.initialize_posterior(config_file)
    post = radvel.posterior.load(status.get('fit', 'postfile'))
    chains = pd.read_csv(status.get('mcmc', 'chainfile'))
    report = radvel.report.RadvelReport(P, post, chains)

    for tabtype in args.type:
        print("Generating LaTeX code for {} table".format(tabtype))

        if tabtype == 'params':
            tex = report.tabletex(tabtype=tabtype)

        if tabtype == 'priors':
            tex = report.tabletex(tabtype=tabtype)

        if tabtype == 'nplanets':
            assert status.has_option('bic', 'nplanets'), \
                "Must run BIC comparison before making comparison tables"

            compstats = eval(status.get('bic', 'nplanets'))
            report = radvel.report.RadvelReport(
                P, post, chains, compstats=compstats
            )
            tex = report.tabletex(tabtype='nplanets')

        saveto = os.path.join(
            args.outputdir, '{}_{}_.tex'.format(conf_base,tabtype)
        )
        with open(saveto, 'w') as f:
            print(tex, file=f)

        savestate = {'{}_tex'.format(tabtype): os.path.abspath(saveto)}
        save_status(statfile, 'table', savestate)


def derive(args):
    """Derive physical parameters from posterior samples

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.outputdir,
                            "{}_radvel.stat".format(conf_base))
    status = load_status(statfile)

    msg = "Multiplying mcmc chains by physical parameters for {}".format(
        conf_base
    )
    print(msg)

    assert status.getboolean('mcmc', 'run'), \
        "Must run MCMC before making tables"

    P, post = radvel.utils.initialize_posterior(config_file)
    post = radvel.posterior.load(status.get('fit', 'postfile'))
    chains = pd.read_csv(status.get('mcmc', 'chainfile'))

    mstar = np.random.normal(
        loc=P.stellar['mstar'], scale=P.stellar['mstar_err'], 
        size=len(chains)
        )

    # Convert chains into CPS basis
    cpschains = chains.copy()
    for par in post.params.keys():
        if not post.params[par].vary:
            cpschains[par] = post.params[par].value


    cpschains = post.params.basis.to_cps(cpschains)

    savestate = {'run': True}
    outcols = []
    for i in np.arange(1, P.nplanets +1, 1):
        # Grab parameters from the chain
        def _has_col(key):
            cols = list(cpschains.columns)
            return cols.count('{}{}'.format(key,i))==1

        def _get_param(key):
            if _has_col(key):
                return cpschains['{}{}'.format(key,i)]
            else:
                return P.params['{}{}'.format(key,i)].value

        def _set_param(key, value):
            chains['{}{}'.format(key,i)] = value

        def _get_colname(key):
            return '{}{}'.format(key,i)

        per = _get_param('per')
        k = _get_param('k')
        e = _get_param('e')

        mpsini = radvel.utils.Msini(k, per, mstar, e, Msini_units='earth')
        _set_param('mpsini',mpsini)

        outcols.append(_get_colname('mpsini'))

        try:
            rp = np.random.normal(
                loc=P.planet['rp{}'.format(i)], 
                scale=P.planet['rp_err{}'.format(i)],
                size=len(chains)
            )

            _set_param('rp',rp)
            _set_param('rhop', radvel.utils.density(mpsini, rp))

            outcols.append(_get_colname('rhop'))
        except (AttributeError, KeyError):
            pass

    print("Derived parameters:", outcols)

    csvfn = os.path.join(args.outputdir, conf_base+'_derived.csv.tar.bz2')
    chains.to_csv(csvfn, columns=outcols, compression='bz2')
    savestate['chainfile'] = os.path.abspath(csvfn)

    save_status(statfile, 'derive', savestate)


def report(args):
    """Generate summary report

    Args:
        args (ArgumentParser): command line arguments
    """

    
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    print("Assembling report for {}".format(conf_base))

    statfile = os.path.join(args.outputdir,
                            "{}_radvel.stat".format(conf_base))

    status = load_status(statfile)

    P, post = radvel.utils.initialize_posterior(config_file)
    post = radvel.posterior.load(status.get('fit', 'postfile'))
    chains = pd.read_csv(status.get('mcmc', 'chainfile'))

    try:
        compstats = eval(status.get('bic', args.comptype))
    except:
        print("WARNING: Could not find {} BIC model comparison\
in {}.\nPlease make sure that you have run `radvel bic -t {}` if you would\
like to include\nthe model comparison table in the report.".format(args.comptype,
                                                            statfile,
                                                            args.comptype))
        compstats = None

    report = radvel.report.RadvelReport(P, post, chains, compstats=compstats)
    report.runname = conf_base

    report_depfiles = []
    for ptype,pfile in status.items('plot'):
        report_depfiles.append(pfile)

    with radvel.utils.working_directory(args.outputdir):
        rfile = os.path.join(conf_base+"_results.pdf")
        report_depfiles = [os.path.basename(p) for p in report_depfiles]
        report.compile(
            rfile, depfiles=report_depfiles, latex_compiler=args.latex_compiler
        )

    
def save_status(statfile, section, statevars):
    """Save pipeline status

    Args:
        statfile (string): name of output file
        section (string): name of section to write
        statevars (dict): dictionary of all options to populate
           the specified section
    """

    config = configparser.RawConfigParser()
    
    if os.path.isfile(statfile):
        config.read(statfile)

    if not config.has_section(section):
        config.add_section(section)
        
    for key,val in statevars.items():
        config.set(section, key, val)

    with open(statfile, 'w') as f:
        config.write(f)


def load_status(statfile):
    """Load pipeline status

    Args:
        statfile (string): name of configparser file

    Returns:
        configparser.RawConfigParser
    """
    
    config = configparser.RawConfigParser()
    gl = config.read(statfile)

    return config
