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

# Define prior centers (initial guesses) here.
params = radvel.RVParameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)    # initialize RVparameters object

params['per1'] = 1206.3      # period of 1st planet
params['tc1'] = 2456779.     # time of inferior conjunction of 1st planet
params['e1'] = 0.01          # eccentricity of 1st planet
params['w1'] = np.pi/2.      # argument of periastron of the star's orbit for 1st planet
params['k1'] = 10.0          # velocity semi-amplitude for 1st planet
params['per2'] = 75.771      # same parameters for 2nd planet ...
params['tc2'] = 2456277.6
params['e2'] = 0.01
params['w2'] = np.pi/2.
params['k2'] = 1

time_base = 2456778          # abscissa for slope and curvature terms (should be near mid-point of time baseline)
params['dvdt'] = 0.0         # slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
params['curv'] = 0.0         # curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)

params['gamma_k'] = 0.0      # velocity zero-point for hires_rk
params['gamma_j'] = 1.0      # "                   "   hires_rj
params['gamma_a'] = 0.0      # "                   "   hires_apf

params['jit_k'] = np.log(2.6)        # jitter for hires_rk
params['jit_j'] = np.log(2.6)        # "      "   hires_rj
params['jit_a'] = np.log(2.6)        # "      "   hires_apf


# Load radial velocity data, in this example the data is contained in an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
data = pd.read_csv(os.path.join(radvel.DATADIR,'164922_fixed.txt'), sep=' ')


# Set parameters to be held constant (default is for all parameters to vary). Must be defined in the fitting basis
vary = dict(
    dvdt = False,
    curv = False,
    jit_k = True,
    jit_j = True,
    jit_a = True,
)


# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.Gaussian('tc1', params['tc1'], 300.0),    # Gaussian prior on tc1 with center at tc1 and width 300 days
    radvel.prior.HardBounds('jit_k', 0.0, 10.0),
    radvel.prior.HardBounds('jit_j', 0.0, 10.0),
    radvel.prior.HardBounds('jit_a', 0.0, 10.0)
]


# optional argument that can contain stellar mass and
# uncertainties. If not set, mstar will be set to nan.
stellar = dict(mstar=0.874, mstar_err=0.012)
