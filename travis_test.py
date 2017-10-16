# Preliminary imports
import os
os.chdir("..")

import radvel
import radvel.likelihood
import radvel.plotting
import copy
import pandas as pd
from scipy import optimize
import corner
from astropy.time import Time
import numpy as np
from numpy import *
import copy

def bin_same_night(rv):
    rv['jd_date'] = rv['time'].apply(lambda x: np.int(np.floor(x)))
    rv_mean = rv.groupby('jd_date',as_index=False).mean()
    rv_n = rv.groupby('jd_date',as_index=False).size()
    print(rv['tel'].iloc[0], len(rv_n))
    rv_mean['errvel'] = rv_mean['errvel'] / np.array(np.sqrt(rv_n))
    return rv_mean

#path = os.path.join(radvel.DATADIR,'164922_fixed.txt')
#data = pd.read_csv(path, usecols=(0,1,2,3), delim_whitespace=True)
data = pd.read_csv('example_data/164922_fixed.txt', usecols=(0,1,2,3), delim_whitespace=True)

rv_hires_rk = data.query("tel == 'k'")
rv_hires_rj = data.query("tel == 'j'")
rv_apf = data.query("tel == 'a'")

rv_hires_rj = bin_same_night(rv_hires_rj)
rv_hires_rk = bin_same_night(rv_hires_rk)
rv_apf = bin_same_night(rv_apf)

t_start = min(data['time'].values)
t_stop = max(data['time'].values)
ti = linspace(t_start,t_stop,10000)

#Some conveinence functions
def initialize_model():
    time_base = 2456778
    params = radvel.Parameters(2,basis='per tc secosw sesinw logk')
    params['per1'] = radvel.Parameter(value=1201.1 + 0.4)
    params['tc1'] = radvel.Parameter(2456778 + 1)
    params['secosw1'] = radvel.Parameter(0.01)
    params['sesinw1'] = radvel.Parameter(0.01)
    params['logk1'] = radvel.Parameter(1)
    params['per2'] = radvel.Parameter(75.765 + 0.1)
    params['tc2'] = radvel.Parameter(2456277.6)
    params['secosw2'] = radvel.Parameter(0.01)
    params['sesinw2'] = radvel.Parameter(0.01)
    params['logk2'] = radvel.Parameter(1)
    params['dvdt'] = radvel.Parameter(0)
    params['curv'] = radvel.Parameter(0)
    mod = radvel.RVModel(params, time_base=time_base)
    return mod

def plot_results(like,color,label):
    fig = gcf()
    axL = fig.get_axes()
    jit = like.params[like.jit_param].value
    yerr = np.sqrt(like.yerr**2 + jit**2)
    bjd0 = 2450000

    sca(axL[0])
    errorbar(
        like.x-bjd0, like.model(like.x)+like.residuals(), 
        yerr=yerr, fmt='o',color=color,label=label
        )
    plot(ti-bjd0, like.model(ti), 'g-')
    fig.set_tight_layout(True)
    sca(axL[1])
    errorbar(
        like.x-bjd0, like.residuals(), 
        yerr=yerr, fmt='o', color=color
        )


def initialize_likelihood(rv,suffix):
    like = radvel.likelihood.RVLikelihood( 
        mod, rv.time, rv.mnvel, rv.errvel,suffix=suffix)
    return like
mod = initialize_model()

# Build up HIRES >2004 likelihood
like_hires_rj = initialize_likelihood(rv_hires_rj,'_hires_rj')
like_hires_rj.params['gamma_hires_rj'] = radvel.Parameter(value=1.0)
like_hires_rj.params['jit_hires_rj'] = radvel.Parameter(value=np.log(1))

# Build up HIRES <2004 likelihood
like_hires_rk = initialize_likelihood(rv_hires_rk,'_hires_rk')
like_hires_rk.params['gamma_hires_rk'] = radvel.Parameter(value=1.0)
like_hires_rk.params['jit_hires_rk'] = radvel.Parameter(value=np.log(1))

# Build up APF
like_apf = initialize_likelihood(rv_apf,'_apf')
like_apf.params['gamma_apf'] = radvel.Parameter(value=1.0)
like_apf.params['jit_apf'] = radvel.Parameter(value=np.log(1))

# Build composite likelihood
like = radvel.likelihood.CompositeLikelihood(
    [like_hires_rj,like_hires_rk,like_apf])

# Set initial values for jitter
like.params['jit_hires_rk'] = radvel.Parameter(value=log(2.6))
like.params['jit_hires_rj'] = radvel.Parameter(value=log(2.6))
like.params['jit_apf'] = radvel.Parameter(value=log(2.6))

# Do not vary dvdt or jitter (Fulton 2015)
like.params['dvdt'].vary = False
like.params['curv'].vary = False
like.params['jit_hires_rk'].vary = False
like.params['jit_hires_rj'].vary = False
like.params['jit_apf'].vary = False



# Instantiate posterior
post = radvel.posterior.Posterior(like)
post0 = copy.deepcopy(post)

# Add in priors
post.priors += [radvel.prior.EccentricityPrior( 2 )] # Keeps eccentricity < 1

# Perform Max-likelihood fitting
res  = optimize.minimize(
    post.neglogprob_array,
    post.get_vary_params(), 
    method='Powell',
    options=dict(maxiter=100000,maxfev=100000,xtol=1e-8)
    )


print("Initial loglikelihood = %f" % post0.logprob())
print("Final loglikelihood = %f" % post.logprob())
print(post)


plt.close('all')
fig,axL = subplots(nrows=2,figsize=(12,8),sharex=True)
plot_results(like.like_list[0],'black','hires_rj') # plot best fit model
plot_results(like.like_list[1],'Tomato','hires_rk') # plot best fit model
plot_results(like.like_list[2],'RoyalBlue','apf') # plot best fit model
axL[0].legend()
bjd0 = 2450000
xlabel('BJD_TBD - %i' % bjd0)
ylabel('RV')
[ax.grid() for ax in axL]
draw()

df = radvel.mcmc(post, nrun=1000)

df_cps = post.params.basis.to_cps(df)
labels = 'per1 tc1 e1 k1 per2 tc2 e2 k2 gamma_hires_rj gamma_hires_rk gamma_apf '.split()
df_cps[labels].quantile([0.14,0.5,0.84]).T

#,close all
labels = 'per1 tc1 e1 k1 per2 tc2 e2 k2 gamma_hires_rj gamma_hires_rk gamma_apf '.split()
rc('font',size=8)
dims = len(labels)
fig,axL = subplots(nrows=dims,ncols=dims,figsize=(10,10))
corner_kw = dict(
    labels=labels,
    levels=[0.68,0.95],
    plot_datapoints=False,
    smooth=True,
    bins=20,
    )
corner.corner(df_cps[labels],fig=fig,**corner_kw)

reload(radvel.plotting)
radvel.plotting.rv_multipanel_plot(post)
