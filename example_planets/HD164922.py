# Example Keplerian fit configuration file

# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel

# Define global planetary system and dataset parameters
starname = 'HD164922'
nplanets = 2    # number of planets in the system
instnames = ['k', 'j', 'a']    # list of instrument names. Can be whatever you like but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw logk'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 0   # reference epoch for RV timestamps (i.e. this number has been subtracted off your timestamps)
planet_letters = {1: 'b', 2: 'c'}   # map the numbers in the RVParameters keys to planet letters (for plotting and tables)


# Define prior centers (initial guesses), and their 'vary' attributes. Each parameter has a 'vary' attribute, which
#   tells RadVel whether to hold the parameter fixed during the fitting process. The 'vary' attributes of parameters
#   you want RadVel to hold fixed should be set to False, and the vary attributes of those you want to float should be 
#   set to True. Orbital parameters not in the fitting basis are derived from other parameters, so their vary attributes
#   should be set to ''.

params = radvel.Parameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)    # initialize RVparameters object

params['per1'] = radvel.Parameter(value=1206.3, vary=True)      # period of 1st planet
params['tc1'] = radvel.Parameter(value=2456779., vary=True)     # time of inferior conjunction of 1st planet
params['e1'] = radvel.Parameter(value=0.01, vary='')          # eccentricity of 1st planet
params['w1'] = radvel.Parameter(value=np.pi/2., vary='')      # argument of periastron of the star's orbit for 1st planet
params['k1'] = radvel.Parameter(value=10.0, vary='')          # velocity semi-amplitude for 1st planet
params['per2'] = radvel.Parameter(value=75.771, vary=True)      # same parameters for 2nd planet ...
params['tc2'] = radvel.Parameter(value=2456277.6, vary=True)
params['e2'] = radvel.Parameter(value=0.01, vary='')
params['w2'] = radvel.Parameter(value=np.pi/2., vary='')
params['k2'] = radvel.Parameter(value=1, vary='')

time_base = 2456778          # abscissa for slope and curvature terms (should be near mid-point of time baseline)
params['dvdt'] = radvel.Parameter(value=0.0, vary=False)         # slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
params['curv'] = radvel.Parameter(value=0.0, vary=False)        # curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)

params['gamma_k'] = radvel.Parameter(value=0.0)       # velocity zero-point for hires_rk
params['gamma_j'] = radvel.Parameter(value=1.0)       # "                   "   hires_rj
params['gamma_a'] = radvel.Parameter(value=0.0)       # "                   "   hires_apf

params['jit_k'] = radvel.Parameter(value=2.6, vary=True)        # jitter for hires_rk
params['jit_j'] = radvel.Parameter(value=2.6, vary=True)         # "      "   hires_rj
params['jit_a'] = radvel.Parameter(value=2.6, vary=True)         # "      "   hires_apf


# Load radial velocity data, in this example the data is contained in
# an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
data = pd.read_csv(os.path.join(radvel.DATADIR,'164922_fixed.txt'), sep=' ')


# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.Gaussian('tc1', params['tc1'].value, 300.0),    # Gaussian prior on tc1 with center at tc1 and width 300 days
    radvel.prior.HardBounds('jit_k', 0.0, 10.0),
    radvel.prior.HardBounds('jit_j', 0.0, 10.0),
    radvel.prior.HardBounds('jit_a', 0.0, 10.0)
]


# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=0.874, mstar_err=0.012)
