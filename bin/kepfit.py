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

from scipy import optimize
import copy

import radvel
import radvel.likelihood
import radvel.plotting


def initialize_posterior(P):
    params = P.params.basis.from_cps(P.params, P.fitting_basis, keep=False)
    iparams = params.copy()

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
        likes[inst].params['logjit_'+inst] = iparams['logjit_'+inst]
        
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
    parser.add_argument('--nburns', dest='nburns',action='store',help='Number of burns. [1000]',default=1000,type=int)
    parser.add_argument('--noplots', dest='noplot',action='store_true',help='No plots will be created or saved [False]')
    parser.add_argument('--nomcmc', dest='nomcmc',action='store_true',help='Skip MCMC? [False]')
    parser.add_argument('--outputdir', dest='outputdir',action='store',help='Directory to save output files [./]', default='./')
    opt = parser.parse_args()

    system_name = os.path.basename(opt.planet).split('.')[0]
    P = imp.load_source(system_name, os.path.abspath(opt.planet))
    system_name = P.starname
        
    post = initialize_posterior(P)
    
    post0 = copy.deepcopy(post)
    print "Initial loglikelihood = %f" % post0.logprob()
    print "Performing maximum likelihood fit..."
    res  = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Powell',
                         options=dict(maxiter=10,maxfev=100000,xtol=1e-8) )

    cpspost = copy.deepcopy(post)
    cpsparams = post.params.basis.to_cps(post.params)
    cpspost.params.update(cpsparams)
    
    print "Final loglikelihood = %f" % post.logprob()
    print "Best-fit parameters:"
    print cpspost
    
    like = post.likelihood

    writedir = os.path.join(opt.outputdir, P.starname)
    if not os.path.isdir(writedir):
        os.mkdir(writedir)

    if not opt.noplot:
        saveto = os.path.join(writedir, P.starname+'_rv_multipanel.pdf')
        radvel.plotting.rv_multipanel_plot(post, saveplot=saveto)

    if not opt.nomcmc:
        print '\n Running MCMC, nwalkers = %s, nsteps = %s, nburn = %s ...'  %(opt.nwalkers, opt.nsteps, opt.nburns)
        chains = radvel.mcmc(post,threads=1,nburn=opt.nburns,nwalkers=opt.nwalkers,nrun=opt.nsteps)

        saveto = os.path.join(writedir, P.starname+'_corner.pdf')
        radvel.plotting.corner_plot(post, chains, saveplot=saveto)
        
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
        res  = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Powell',
                         options=dict(maxiter=10,maxfev=100000,xtol=1e-8) )

        cpspost = copy.deepcopy(post)
        cpsparams = post.params.basis.to_cps(post.params)
        cpspost.params.update(cpsparams)
        
        print "Final loglikelihood = %f" % post.logprob()
        print "Best-fit parameters:"
        print cpspost

        if not opt.noplot:
            saveto = os.path.join(writedir, P.starname+'_rv_multipanel.pdf')
            radvel.plotting.rv_multipanel_plot(post, saveplot=saveto)
            saveto = os.path.join(writedir, P.starname+'_trends.pdf')
            radvel.plotting.trend_plot(post, chains, opt.nwalkers, saveto)
            

    # Save posterior object as binary pickle file
    pkl = open(os.path.join(writedir, P.starname+'_post_obj.pkl'), 'wb')
    pickle.dump(post, pkl)
    pkl.close()
    
