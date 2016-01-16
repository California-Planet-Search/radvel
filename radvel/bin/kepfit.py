#!/usr/bin/env python

import pandas as pd
import numpy as np

from scipy import optimize
import corner
import copy

import radvel
import radvel.likelihood


def import_string(name):
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def args():
    import argparse

    parser = argparse.ArgumentParser(description='Fit an RV dataset')
    parser.add_argument(metavar='planet',dest='planet',action='store',help='Planet name (should be file name contained in the planets directory)',default='HD164922',type=str)
    parser.add_argument('--nsteps', dest='steps',action='store',help='Number of steps per chain',default=1000,type=float)
    parser.add_argument('--nwalkers', dest='nwalkers',action='store',help='Number of walkers.',default=40,type=int)
    #parser.add_argument('--thin', dest='thin',action='store',help='Record only every Nth link in chain. (saves memory)',default=1,type=int)
    #parser.add_argument('--plot', dest='plot',action='store_true',help='Plot in real time? (Very slow)',default=False)
    
    opt = parser.parse_args()
    opt.planet = import_string('planets.'+opt.planet)    

    return opt


if __name__ == '__main__':
    opt = args()

    P = opt.planet

    post0 = copy.deepcopy(P.post)

    print P.post.params.basis
        
    print "Initial loglikelihood = %f" % post0.logprob()
    print "Performing maximum likelihood fit..."

    res  = optimize.minimize(P.post.neglogprob_array, P.post.get_vary_params(), method='Powell',
                         options=dict(maxiter=100000,maxfev=100000,xtol=1e-8) )

    print "Final loglikelihood = %f" % P.post.logprob()
    print "Best-fit parameters:"
    print P.post

