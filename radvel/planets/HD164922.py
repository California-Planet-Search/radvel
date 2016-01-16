# Example Keplerian fit configuration file

# Required packages for setup
import radvel
from radvel.utils import bintels
import os
import numpy as np


# Define global planetary system parameters
nplanets = 2    # number of planets in the system
instnames = ['hires_rk', 'hires_rj', 'apf']    # list of instrument names. Can be whatever you like, but should be kept consistent.
ntels = len(instnames)       # number of instruments with unique velocity zero-points


# Define prior centers (initial guesses) here.
params = radvel.RVParameters(nplanets,basis='per tc e w k')    # initialize RVparameters object

params['per1'] = 1206.3      # period of 1st planet
params['tc1'] = 2456779.     # time of inferior conjunction of 1st planet
params['e1'] = 0.01          # eccentricity of 1st planet
params['w1'] = np.pi/2.      # argument of periastron of the star's orbit for 1st planet
params['k1'] = 10.0          # velocity semi-amplitude for 1st planet
params['per2'] = 75.771      # same parameters for 2nd planet ...
params['tc2'] = 2456277.6
params['e2'] = 0.01
params['w2'] = 0.01
params['k2'] = 1

time_base = 2457000          # abscissa for slope and curvature terms (should be near mid-point of time baseline)
params['dvdt'] = 0.0         # slope
params['curv'] = 0.0         # curvature

params['gamma_hires_rk'] = 0 # velocity zero-point for hires_rk
params['gamma_hires_rj'] = 0 # "                   "   hires_rj
params['gamma_apf'] = 0      # "                   "   hires_apf

params['logjit_hires_rk'] = np.log(2.6)   # jitter for hires_rk
params['logjit_hires_rj'] = np.log(2.6)   # "      "   hires_rj
params['logjit_apf'] = np.log(2.6)        # "      "   hires_apf

mod = radvel.RVModel(params, time_base=time_base)    # initialize RVmodel object


# Load radial velocity data, in this example the data is contained in an ASCII file
data = np.genfromtxt(os.path.join(radvel.DATADIR,'164922_fixed.txt'), names=True, dtype=None, usecols=(0,1,2,3))
time = data['BJD_TDB']
vel = data['mnvel']
err = data['errvel']
tel = data['tel']

time_bin, vel_bin, err_bin, tel_bin = bintels(time, vel, err, tel, binsize=0.5)   # Bin together data taken within 0.5 days on a single instrument 

# Separate data by insturment to construct RVlikelihood objects
rk = tel_bin == 'k'
rj = tel_bin == 'j'
apf = tel_bin == 'a'

t_rk = time_bin[rk]
v_rk = vel_bin[rk]
e_rk = err_bin[rk]

t_rj = time_bin[rj]
v_rj = vel_bin[rj]
e_rj = err_bin[rj]

t_apf = time_bin[apf]
v_apf = vel_bin[apf]
e_apf = err_bin[apf]


# initialize RVlikelihood objects for each instrument
like_rk = radvel.likelihood.RVLikelihood(mod, t_rk, v_rk, e_rk,suffix='_hires_rk')
like_rj = radvel.likelihood.RVLikelihood(mod, t_rj, v_rj, e_rj,suffix='_hires_rj')
like_apf = radvel.likelihood.RVLikelihood(mod, t_apf, v_apf, e_apf,suffix='_apf')
like = radvel.likelihood.CompositeLikelihood([like_rk, like_rj, like_apf])


# Set parameters to be held constant (default is for all parameters to vary)
like.vary['dvdt'] = False
like.vary['curv'] = False
#like.vary['logjit_hires_rk'] = False
#like.vary['logjit_hires_rj'] = False
#like.vary['logjit_apf'] = False


# Initialize Posterior object
post = radvel.posterior.Posterior(like)


# Define prior shapes and widths here
post.priors += [radvel.prior.EccentricityPrior( nplanets )]          # Keeps eccentricity < 1
post.priors += [radvel.prior.Gaussian('tc1', params['tc1'], 300.0)]   # Gaussian prior on tc1 with center at tc1 and width 300 days
