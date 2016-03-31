#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import pylab as py
import pdb

from scipy import optimize
import corner
import copy

import radvel
import radvel.likelihood
import radvel.plotting

def args():
    import argparse

    parser = argparse.ArgumentParser(description='Fit an RV dataset')
    parser.add_argument(metavar='planet',dest='planet',action='store',help='Planet name (should be file name contained in the planets directory)',type=str)
    parser.add_argument('--nsteps', dest='nsteps',action='store',help='Number of steps per chain',default=1000,type=float)
    parser.add_argument('--nwalkers', dest='nwalkers',action='store',help='Number of walkers.',default=40,type=int)
    parser.add_argument('--nburns', dest='nburns',action='store',help='Number of burns.',default=1000,type=int)
    parser.add_argument('--noplots', dest='noplot',action='store_true',help='No plots will be created or saved')
    parser.add_argument('--nomcmc', dest='nomcmc',action='store_true',help='Skip MCMC?')
    # parser.add_argument('--plot', dest='plot',action='store_true',help='Save plots?')
    # parser.add_argument('--mcmc', dest='mcmc',action='store_true',help='Run MCMC?')
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


def plot_results(like,color,label,bjd0):
    fig = py.gcf()
    axL = fig.get_axes()
    jit = np.exp(like.params[like.logjit_param])
    yerr = np.sqrt(like.yerr**2 + jit**2)
    ti = np.linspace(np.min(like.x), np.max(like.x), 1000)
    py.sca(axL[0])
    py.errorbar(
        like.x-bjd0, like.model(like.x)+like.residuals(), 
        yerr=yerr, fmt='o',color=color,label=label
        )
    py.plot(ti-bjd0, like.model(ti))
    fig.set_tight_layout(True)
    py.sca(axL[1])
    py.errorbar(
        like.x-bjd0, like.residuals(), 
        yerr=yerr, fmt='o'
        )
    return


def plot_maxlike(like, tel, bjd0, saveto):

    py.close('all')
    fig,axL = py.subplots(nrows=2,figsize=(12,8),sharex=True)
    if 'j' in tel:
        plot_results(like.like_list[np.where(np.array(tel)=='j')[0]],'black','hires_rj', bjd0) # plot best fit model
    if 'k' in tel:
        plot_results(like.like_list[np.where(np.array(tel)=='k')[0]],'Tomato','hires_rk', bjd0) # plot best fit model
    if 'a' in tel:
        plot_results(like.like_list[np.where(np.array(tel)=='a')[0]],'RoyalBlue','apf', bjd0) # plot best fit model
        
    axL[0].legend()
    py.xlabel('BJD_TBD - %i' % bjd0)
    py.ylabel('RV')
    [ax.grid() for ax in axL]
    py.savefig(saveto, bbox_inches='tight', pad_inches=0.1)
    print '\n Plot of max likelihood fit saved: ', saveto 
    return


    
if __name__ == '__main__':

    opt = args()

    P = opt.planet
    
    post = initialize_posterior(P)
    
    post0 = copy.deepcopy(post)
    print "Initial loglikelihood = %f" % post0.logprob()
    print "Performing maximum likelihood fit..."

    res  = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Powell',
                         options=dict(maxiter=10,maxfev=100000,xtol=1e-8) )

    print "Final loglikelihood = %f" % post.logprob()
    print "Best-fit parameters:"
    print post

    cpsparams = post.params.basis.to_cps(post.params)

    print post.likelihood.like_list
    
    like = post.likelihood

    writedir = radvel.DATADIR + '/output/'
    if not os.path.isdir(writedir):
        os.mkdir(writedir)

    if not opt.noplot:
        saveto = writedir + P.starname + '_bestfit.pdf'
        plot_maxlike(like, P.instnames, P.bjd0, saveto)
        py.close('all')
        saveto = writedir + P.starname + '_rv_multipanel.pdf'
        radvel.plotting.rv_multipanel_plot(post, saveplot=saveto)
        py.savefig(saveto, bbox_inches='tight', pad_inches=0.1)
        print '\n RV multipanel plot saved: ', saveto
        py.close('all')

    if not opt.nomcmc:
        print '\n Running MCMC, nwalkers = %s, nsteps = %s, nburns = %s ...'  %(opt.nwalkers, opt.nsteps, opt.nburns)
        df = radvel.mcmc(post,threads=1,nburn=opt.nburns,nwalkers=opt.nwalkers,nrun=opt.nsteps)

        labels = [k for k in post.vary.keys() if post.vary[k]]
        fig = corner.corner(df[labels],
                            labels=labels,
                            levels=[0.68,0.95],
                            plot_datapoints=False,
                            smooth=True,
                            bins=20, quantiles=[.16,.5,.84])
        saveto = writedir + P.starname + '_corner.pdf'
        py.savefig(saveto, bbox_inches='tight', pad_inches=0.1)
        print '\n Plot of max likelihood fit saved: ', saveto, '\n' 
        py.close('all')
        df_summary=df[labels].describe(percentiles=[.1587,.5,.8413])
        print '\n Posterior Summary...\n'
        print df_summary
        saveto = writedir + P.starname + '_post_summary.txt'
        df_summary.to_csv(saveto, sep=',')
        print '\n Posterior Summary saved:' , saveto  
