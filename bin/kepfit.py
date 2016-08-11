#!/usr/bin/env python
import os
import sys
import imp
import pickle
import argparse
import copy
import warnings

import pandas as pd
import numpy as np
from scipy import optimize

import radvel
import radvel.likelihood
import radvel.plotting
import radvel.utils
import radvel.fitting

warnings.filterwarnings("ignore")
warnings.simplefilter('once', DeprecationWarning)
def initialize_posterior(P):
    params = P.params.basis.from_cps(P.params, P.fitting_basis, keep=False)

    for key in params.keys():
        if key.startswith('logjit'):
            msg = """
Fitting log(jitter) is depreciated. Please convert your config
files to initialize 'jit' instead of 'logjit' parameters.
Converting 'logjit' to 'jit' for you now.
"""
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            newkey = key.replace('logjit', 'jit')
            params[newkey] = np.exp(params[key])
            del params[key]

    iparams = params.copy()

    # Make sure we don't have duplicate indicies in the DataFrame
    P.data = P.data.reset_index(drop=True)
    
    # initialize RVmodel object
    mod = radvel.RVModel(params, time_base=P.time_base)   

    # initialize RVlikelihood objects for each instrument
    telgrps = P.data.groupby('tel').groups
    likes = {}
    for inst in P.instnames:
        likes[inst] = radvel.likelihood.RVLikelihood(
            mod, P.data.iloc[telgrps[inst]].time,
            P.data.iloc[telgrps[inst]].mnvel,
            P.data.iloc[telgrps[inst]].errvel, suffix='_'+inst
        )
        likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
        likes[inst].params['jit_'+inst] = iparams['jit_'+inst]

    like = radvel.likelihood.CompositeLikelihood(likes.values())

    # Set fixed/vary parameters
    like.vary.update(P.vary)
    
    # Initialize Posterior object
    post = radvel.posterior.Posterior(like)
    post.priors = P.priors
    return post

if __name__ == '__main__':
    psr = argparse.ArgumentParser(description='Fit an RV dataset')
    psr.add_argument(
        metavar='planet', dest='planet', action='store', 
        help='''
        Planet name (should be file name contained in the planets directory)
        ''',
        type=str
    )
    psr.add_argument(
        '--nsteps', dest='nsteps', action='store',default=20000, type=float, 
        help='Number of steps per chain [20000]',)
    psr.add_argument(
        '--nwalkers', dest='nwalkers', action='store', default=50, type=int,
        help='Number of walkers. [50]', 
    )
    psr.add_argument(
        '--noplots', dest='noplot', action='store_true', 
        help='No plots will be created or saved [False]'
    )
    psr.add_argument(
        '--plotkw', dest='plotkw',action='store', default='{}', type=str,
        help='''
        Dictionary of keywords sent to rv_multipanel_plot. 
        E.g. --plotkw "{'yscale_auto': True}"'
        ''',
    )
    psr.add_argument(
        '--nomcmc', dest='nomcmc', action='store_true', 
        help='Skip MCMC? [False]'
    )
    psr.add_argument(
        '--outputdir', dest='outputdir', action='store', default='./',
        help='Directory to save output files [./]', 
    )
    opt = psr.parse_args()
    opt.plotkw = eval(opt.plotkw)
    
    system_name = os.path.basename(opt.planet).split('.')[0]
    P = imp.load_source(system_name, os.path.abspath(opt.planet))
    system_name = P.starname

    opt.plotkw['epoch'] = P.bjd0

    post = initialize_posterior(P)
    post = radvel.fitting.maxlike_fitting(post, verbose=True)
    like = post.likelihood
    writedir = os.path.join(opt.outputdir, P.starname)
    curdir = os.getcwd()
    if not os.path.isdir(writedir):
        os.mkdir(writedir)

    if not opt.noplot:
        saveto = os.path.join(writedir, P.starname+'_rv_multipanel.pdf')
        radvel.plotting.rv_multipanel_plot(post, saveplot=saveto, **opt.plotkw)

    # Check if stellar or planet properties are defined, if not add nan
    if not hasattr(P, 'stellar'):
        P.stelar = dict(mstar=np.nan, mstar_err=np.nan)

    if not hasattr(P, 'planet'):
        planet = {}
        for i in range(1, P.nplanets+1):
            planet['rp{}'.format(i)] = np.nan
        for i in range(1, P.nplanets+1):
            planet['rp_err{}'.format(i)] = np.nan
        P.planet = planet

    if not opt.nomcmc:
        msg = "Running MCMC, nwalkers = {}, nsteps = {} ...".format(
            opt.nwalkers, opt.nsteps
        )
        print msg
        chains = radvel.mcmc(
            post, threads=1, nwalkers=opt.nwalkers, nrun=opt.nsteps
        )
        
        mstar = np.random.normal(
            loc=P.stellar['mstar'], scale=P.stellar['mstar_err'], 
            size=len(chains)
            )

            
        for i in np.arange(1, P.nplanets +1, 1):
            # Grab parameters from the chain
            def _has_col(key):
                cols = list(chains.columns)
                return cols.count('{}{}'.format(key,i))==1

            def _get_param(key):
                if _has_col(key):
                    return chains['{}{}'.format(key,i)]
                else:
                    return P.params['{}{}'.format(key,i)]

            def _set_param(key, value):
                chains['{}{}'.format(key,i)] = value

            per = _get_param('per')
            k = _get_param('k')
            if _has_col('e') or 'e{}'.format(i) in P.params.keys():
                e = _get_param('e')
            if _has_col('ecosw') or 'ecosw{}'.format(i) in P.params.keys():
                secosw = _get_param('secosw')
                sesinw = _get_param('sesinw')
                e, _ = radvel.orbit.sqrtecosom_sqrtesinom_to_e_om(ecosw,esinw)  

            mpsini = radvel.orbit.Msini(k, per, mstar, e, Msini_units='earth')
            _set_param('mpsini',mpsini)
            mpsini50 = np.median(_get_param('mpsini'))
            
            rp = np.random.normal(
                loc=P.planet['rp{}'.format(i)], 
                scale=P.planet['rp_err{}'.format(i)],
                size=len(chains)
            )

            _set_param('rp',rp)
            chains['rhop' + str(i)] = radvel.orbit.density(mpsini, rp)


        report_depfiles = []
        if not opt.noplot:
            cp_saveto = os.path.join(writedir, P.starname+'_corner.pdf')
            radvel.plotting.corner_plot(post, chains, saveplot=cp_saveto)
            cp_derived_saveto = os.path.join(
                writedir, P.starname+'_corner_derived_pars.pdf'
            )
            report_depfiles.append(cp_saveto)
            radvel.plotting.corner_plot_derived_pars(
                chains, P, saveplot=cp_derived_saveto
            )
            report_depfiles.append(cp_derived_saveto)

#Evan updates end here
################# 
            
        post_summary=chains.quantile([0.1587, 0.5, 0.8413])
        print '\n Posterior Summary...\n'
        print post_summary
        saveto = os.path.join(writedir, P.starname+'_post_summary.csv')
        post_summary.to_csv(saveto, sep=',')
        print '\n Posterior Summary saved:' , saveto  


        csvfn = os.path.join(writedir, P.starname+'_chains.csv.tar.bz2')
        chains.to_csv(csvfn, compression='bz2')

        for k in chains.keys():
            if k in post.params.keys():
                post.params[k] = post_summary[k][0.5]
        
        print "Performing post-MCMC maximum likelihood fit..."
        post = radvel.fitting.maxlike_fitting(post, verbose=False)
        
        cpspost = copy.deepcopy(post)
        cpsparams = post.params.basis.to_cps(post.params)
        cpspost.params.update(cpsparams)

        report = radvel.report.RadvelReport(P, post, chains)

        cpspost.uparams = {}
        for par in cpspost.params.keys():
            med = report.quantiles[par][0.5]
            high = report.quantiles[par][0.841] - med
            low = med - report.quantiles[par][0.159]
            err = np.mean([high,low])
            err = radvel.utils.round_sig(err)
            med, err, errhigh = radvel.utils.sigfig(med, err)
            cpspost.uparams[par] = err

        print "Final loglikelihood = %f" % post.logprob()
        print "Final RMS = %f" % post.likelihood.residuals().std()
        print "Best-fit parameters:"
        print cpspost

        if not opt.noplot:
            opt.plotkw['uparams'] = cpspost.uparams
            mp_saveto = os.path.join(writedir, P.starname+'_rv_multipanel.pdf')
            radvel.plotting.rv_multipanel_plot(post, saveplot=mp_saveto, **opt.plotkw)
            saveto = os.path.join(writedir, P.starname+'_trends.pdf')
            radvel.plotting.trend_plot(post, chains, opt.nwalkers, saveto)
            report_depfiles.append(mp_saveto)

        with radvel.utils.working_directory(writedir):
            rfile = os.path.join(P.starname+"_results.pdf")
            report_depfiles = [os.path.basename(p) for p in report_depfiles]
            report.compile(rfile, depfiles=report_depfiles)

    # Save posterior object as binary pickle file
    pkl = open(os.path.join(writedir, P.starname+'_post_obj.pkl'), 'wb')
    pickle.dump(post, pkl)
    pkl.close()
    
