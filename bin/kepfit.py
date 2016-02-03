#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

from scipy import optimize
import corner
import copy

import radvel
import radvel.likelihood

def args():
    import argparse

    parser = argparse.ArgumentParser(description='Fit an RV dataset')
    parser.add_argument(metavar='planet',dest='planet',action='store',help='Planet name (should be file name contained in the planets directory)',type=str)
    parser.add_argument('--nsteps', dest='steps',action='store',help='Number of steps per chain',default=1000,type=float)
    parser.add_argument('--nwalkers', dest='nwalkers',action='store',help='Number of walkers.',default=40,type=int)
    #parser.add_argument('--thin', dest='thin',action='store',help='Record only every Nth link in chain. (saves memory)',default=1,type=int)
    #parser.add_argument('--plot', dest='plot',action='store_true',help='Plot in real time? (Very slow)',default=False)
    
    opt = parser.parse_args()
    opt.planet = import_string('radvel.planets.'+opt.planet)    

    return opt

def import_string(name):
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

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
    opt = args()

    P = opt.planet

    post = initialize_posterior(P)
    
    post0 = copy.deepcopy(post)
    print "Initial loglikelihood = %f" % post0.logprob()
    print "Performing maximum likelihood fit..."

    res  = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Powell',
                         options=dict(maxiter=100000,maxfev=100000,xtol=1e-8) )

    print "Final loglikelihood = %f" % post.logprob()
    print "Best-fit parameters:"
    print post
