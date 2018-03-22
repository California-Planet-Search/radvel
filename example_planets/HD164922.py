# Example Keplerian fit configuration file

# Required packages for setup
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/seanmills/Desktop/radvel_smmills/radvel/')

import radvel

# Define global planetary system and dataset parameters
starname = 'HD164922'
nplanets = 2    # number of planets in the system
instnames = ['k', 'j', 'a']    # list of instrument names. Can be whatever you like but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw logk'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 0   # reference epoch for RV timestamps (i.e. this number has been subtracted off your timestamps)
planet_letters = {1: 'b', 2: 'c'}   # map the numbers in the Parameters keys to planet letters (for plotting and tables)

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)    # initialize Parameters object

anybasis_params['per1'] = radvel.Parameter(value=1206.3)      # period of 1st planet
anybasis_params['tc1'] = radvel.Parameter(value=2456779.)     # time of inferior conjunction of 1st planet
anybasis_params['e1'] = radvel.Parameter(value=0.01)          # eccentricity of 1st planet
anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)      # argument of periastron of the star's orbit for 1st planet
anybasis_params['k1'] = radvel.Parameter(value=10.0)          # velocity semi-amplitude for 1st planet
anybasis_params['per2'] = radvel.Parameter(value=75.771)      # same parameters for 2nd planet ...
anybasis_params['tc2'] = radvel.Parameter(value=2456277.6)
anybasis_params['e2'] = radvel.Parameter(value=0.01)
anybasis_params['w2'] = radvel.Parameter(value=np.pi/2.)
anybasis_params['k2'] = radvel.Parameter(value=1)

time_base = 2456778          # abscissa for slope and curvature terms (should be near mid-point of time baseline)
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)         # slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
anybasis_params['curv'] = radvel.Parameter(value=0.0)        # curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)

#anybasis_params['gamma_k'] = radvel.Parameter(value=0.0)       # velocity zero-point for hires_rk
#anybasis_params['gamma_j'] = radvel.Parameter(value=1.0)       # "                   "   hires_rj
#anybasis_params['gamma_a'] = radvel.Parameter(value=0.0)       # "                   "   hires_apf

#anybasis_params['jit_k'] = radvel.Parameter(value=2.6)        # jitter for hires_rk
#anybasis_params['jit_j'] = radvel.Parameter(value=2.6)         # "      "   hires_rj
#anybasis_params['jit_a'] = radvel.Parameter(value=2.6)         # "      "   hires_apf

# Convert input orbital parameters into the fitting basis
params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

# Set the 'vary' attributes of each of the parameters in the fitting basis. A parameter's 'vary' attribute should
# be set to False if you wish to hold it fixed during the fitting process. By default, all 'vary' parameters
# are set to True.
params['dvdt'].vary = True 
params['curv'].vary = True


# Load radial velocity data, in this example the data is contained in
# an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
#data = pd.read_csv(os.path.join(radvel.DATADIR,'164922_fixed.txt'), sep=' ')
data = pd.read_csv(os.path.join('/Users/seanmills/Desktop/radvel_smmills/radvel/example_data/','164922_fixed.txt'), sep=' ')


# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.Gaussian('tc1', params['tc1'].value, 300.0),    # Gaussian prior on tc1 with center at tc1 and width 300 days
    radvel.prior.HardBounds('jit', 0.0, 10.0),
#    radvel.prior.HardBounds('jit_k', 0.0, 10.0),
#    radvel.prior.HardBounds('jit_j', 0.0, 10.0),
#    radvel.prior.HardBounds('jit_a', 0.0, 10.0)
]


# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=0.874, mstar_err=0.012)





########################

print data

t = np.array(data['time'])
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])



mod = radvel.RVModel(params, time_base=time_base)
like = radvel.likelihood.RVLikelihood(mod, t, vel, errvel)

like.params['jit'] = radvel.Parameter(value=2.6)
like.params['gamma'] = radvel.Parameter(value=1.0)

post = radvel.posterior.Posterior(like)
post.priors += priors
#post.priors += [radvel.prior.Gaussian( 'jit', np.log(3), 0.5)]
#post.priors += [radvel.prior.Gaussian( 'logk2', np.log(5), 10)]
#post.priors += [radvel.prior.Gaussian( 'logk1', np.log(5), 10)]
#post.priors += [radvel.prior.Gaussian( 'gamma', 0, 10)]

print (post)
print "here"
import pylab as pl
ti = np.linspace(data.time.iloc[0]-5,data.time.iloc[-1]+5,10000)
def plot_results(like):
    fig = pl.figure(figsize=(12,4))
    fig = pl.gcf()
    fig.set_tight_layout(True)
    pl.errorbar(
        like.x, like.model(t)+like.residuals(), 
        yerr=like.yerr, fmt='o'
        )
    pl.plot(ti, like.model(ti))
    pl.xlabel('Time')
    pl.ylabel('RV')
    pl.draw()
    pl.show()

from scipy import optimize

plot_results(like)             # plot best fit model
res  = optimize.minimize(
    post.neglogprob_array,     # objective function is negative log likelihood
    post.get_vary_params(),    # initial variable parameters
    method='Powell',           # Nelder-Mead also works
    )

plot_results(like)             # plot best fit model
print(post)

bic = post.bic()
aic = post.aic()
print "bic:", bic
print "aic:", aic


#diclist = radvel.model_comp(post, params=['trend'], verbose=True)
#diclist = radvel.model_comp(post, params=['trend', 'nplanet'], verbose=True)
#diclist = radvel.model_comp(post, params=['nplanet'], verbose=True)
#diclist = radvel.model_comp(post, params=['e'], verbose=True)
#diclist = radvel.model_comp(post, params=['jit'], verbose=True)
#diclist = radvel.model_comp(post, params=['gp'], verbose=True)
diclist = radvel.model_comp(post, params=['x'], verbose=True)
for dic in diclist:
  print ""
  print dic




