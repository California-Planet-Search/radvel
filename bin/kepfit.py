#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import sys
import argparse
import imp
import pickle
import warnings

from scipy import optimize
import copy

import radvel
import radvel.likelihood
import radvel.plotting
import radvel.utils
import radvel.fitting

warnings.simplefilter('once', DeprecationWarning)

def initialize_posterior(P):
    params = P.params.basis.from_cps(P.params, P.fitting_basis, keep=False)

    for key in params.keys():
        if key.startswith('logjit'):
            warnings.warn("""Fitting log(jitter) is depreciated. Please convert your config \
files to initialize 'jit' instead of 'logjit' parameters. \
Converting 'logjit' to 'jit' for you now.""", DeprecationWarning, stacklevel=2)
            newkey = key.replace('logjit', 'jit')
            params[newkey] = np.exp(params[key])
            P.vary[newkey] = P.vary[key]
            del P.vary[key]
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
        likes[inst] = radvel.likelihood.RVLikelihood(mod, P.data.iloc[telgrps[inst]].time,
                                               P.data.iloc[telgrps[inst]].mnvel,
                                               P.data.iloc[telgrps[inst]].errvel, suffix='_'+inst)
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
    parser = argparse.ArgumentParser(description='Fit an RV dataset')
    parser.add_argument(metavar='planet',dest='planet',action='store',help='Planet name (should be file name contained in the planets directory)',type=str)
    parser.add_argument('--nsteps', dest='nsteps',action='store',help='Number of steps per chain [20000]',default=20000,type=float)
    parser.add_argument('--nwalkers', dest='nwalkers',action='store',help='Number of walkers. [50]',default=50,type=int)
    parser.add_argument('--noplots', dest='noplot',action='store_true',help='No plots will be created or saved [False]')
    parser.add_argument('--plotkw', dest='plotkw',action='store',help='Dictionary of keywords sent to rv_multipanel_plot. E.g. --plotkw "{\'yscale_auto\': True}"', default='{}', type=str)
    parser.add_argument('--nomcmc', dest='nomcmc',action='store_true',help='Skip MCMC? [False]')
    parser.add_argument('--outputdir', dest='outputdir',action='store',help='Directory to save output files [./]', default='./')
    opt = parser.parse_args()

    opt.plotkw = eval(opt.plotkw)
    
    system_name = os.path.basename(opt.planet).split('.')[0]
    P = imp.load_source(system_name, os.path.abspath(opt.planet))
    system_name = P.starname

    opt.plotkw['epoch'] = P.bjd0

    post = initialize_posterior(P)

    
    post = radvel.fitting.maxlike_fitting(post, verbose=True)
    
    #statsdict = radvel.fitting.model_comp(post)
    
    like = post.likelihood

    writedir = os.path.join(opt.outputdir, P.starname)
    curdir = os.getcwd()
    if not os.path.isdir(writedir):
        os.mkdir(writedir)

    if not opt.noplot:
        saveto = os.path.join(writedir, P.starname+'_rv_multipanel.pdf')
        radvel.plotting.rv_multipanel_plot(post, saveplot=saveto, **opt.plotkw)

################# 
#Evan updates start here
    if not hasattr(P, 'mstar'):
        P.mstar = 1.0
        
    if not opt.nomcmc:
        print '\n Running MCMC, nwalkers = %s, nsteps = %s ...'  %(opt.nwalkers, opt.nsteps)
        chains = radvel.mcmc(post,threads=1,nwalkers=opt.nwalkers,nrun=opt.nsteps)

        if not hasattr(P, 'mstar_err'):
            mstar = copy.deepcopy(P.mstar)
        else:
            mstar = np.random.normal(loc=P.mstar, scale=P.mstar_err, size=len(chains))
            
        for i in np.arange(1, P.nplanets +1, 1):
            if hasattr(chains, 'per'+ str(i)):
                per = chains['per' + str(i)]
            else:
                per = P.params['per' + str(i)]
            if hasattr(chains, 'k'+ str(i)):
                k = chains['k' + str(i)]
            else:
                k = P.params['k' + str(i)]
            if hasattr(chains, 'e'+ str(i)):
                e = chains['e' + str(i)]
            elif hasattr(chains, 'secosw' + str(i)) and hasattr(chains, 'sesinw' + str(i)):
                e, _ = radvel.orbit.sqrtecosom_sqrtesinom_to_e_om(chains['secosw'+str(i)], chains['sesinw'+str(i)])
            else:
                if 'e' + str(i) in P.params.keys():
                    e = P.params['e' + str(i)]
                if ('secosw' + str(i) in P.params.keys()) and ('sesinw' + str(i) in P.params.keys()):
                    e, _ = radvel.orbit.sqrtecosom_sqrtesinom_to_e_om(P.params['secosw'+str(i)], P.params['sesinw'+str(i)])            
            chains['mpsini' + str(i)] = radvel.orbit.Msini(k, per, mstar, e, Msini_units='earth')
            print "mpsini" + str(i) + " = " + str(np.median(chains['mpsini' + str(i)]))

            if hasattr(P, 'rp' + str(i)):
                if hasattr(P, 'rp_err' + str(i)):
                    chains['rp' + str(i)] = np.random.normal(loc= getattr(P, 'rp' + str(i)), scale=getattr(P, 'rp_err' + str(i)), size=len(chains))
                    chains['rhop' + str(i)] = radvel.orbit.density(chains['mpsini' + str(i)], chains['rp' + str(i)])
                
        report_depfiles = []
        if not opt.noplot:
            cp_saveto = os.path.join(writedir, P.starname+'_corner.pdf')
            radvel.plotting.corner_plot(post, chains, saveplot=cp_saveto)
            cp_derived_saveto = os.path.join(writedir, P.starname+'_corner_derived_pars.pdf')
            report_depfiles.append(cp_saveto)
            radvel.plotting.corner_plot_derived_pars(chains, P, saveplot=cp_derived_saveto)
            report_depfiles.append(cp_derived_saveto)

#Evan updates end here
################# 
            
        post_summary=chains.quantile([0.1587, 0.5, 0.8413])
        print '\n Posterior Summary...\n'
        print post_summary
        saveto = os.path.join(writedir, P.starname+'_post_summary.csv')
        post_summary.to_csv(saveto, sep=',')
        print '\n Posterior Summary saved:' , saveto  

        chains.to_csv(os.path.join(writedir, P.starname+'_chains.csv.tar.bz2'), compression='bz2')

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
    
