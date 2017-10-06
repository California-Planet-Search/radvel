# Example Keplerian fit configuration file
# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel
import os

# Define global planetary system and dataset parameters
starname = 'epic203771098'
nplanets = 2    # number of planets in the system
instnames = ['j']    # list of instrument names. Can be whatever you like but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw k'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 2454833.
planet_letters = {1: 'b', 2:'c'}

# Define prior centers (initial guesses), and their 'vary' attributes. Each parameter has a 'vary' attribute, which
#   tells RadVel whether or not to hold the parameter fixed during the fitting process. The 'vary' attributes of parameters
#   you want RadVel to hold fixed should be set to False, and the vary attributes of those you want to float should be 
#   set to True. Orbital parameters in the input basis but not the fitting basis are derived from other parameters, so 
#   their vary attributes should be set to ''.

params = radvel.Parameters(nplanets,basis='per tc e w k')    # initialize Parameters object

params['per1'] = radvel.Parameter(value=20.885258, vary=False)    # period of 1st planet
params['tc1'] = radvel.Parameter(value=2072.79438, vary=False)    # time of inferior conjunction of 1st planet
params['e1'] = radvel.Parameter(value=0.01, vary='')          # eccentricity of 'per tc secosw sesinw logk'1st planet
params['w1'] = radvel.Parameter(value=np.pi/2., vary='')      # argument of periastron of the star's orbit for 1st planet
params['k1'] = radvel.Parameter(value=10.0, vary=True)         # velocity semi-amplitude for 1st planet
params['per2'] = radvel.Parameter(value=42.363011, vary=False)    # same parameters for 2nd planet ...
params['tc2'] = radvel.Parameter(value=2082.62516, vary=False)
params['e2'] = radvel.Parameter(value=0.01, vary='')
params['w2'] = radvel.Parameter(value=np.pi/2., vary='')
params['k2'] = radvel.Parameter(value=10.0, vary=True)

params['dvdt'] = radvel.Parameter(value=0.0, vary=True)        # slope
params['curv'] = radvel.Parameter(value=0.0, vary=True)         # curvature

params['gamma_j'] = radvel.Parameter(1.0, vary=True)      # "                   "   hires_rj
params['jit_j'] = radvel.Parameter(value=2.6, vary=True)        # "      "   hires_rj


# Here we set parameters in the fitting basis (but not in the input basis) to be held fixed during the fitting process.
params['secosw1'] = radvel.Parameter(value=None, vary=False)
params['sesinw1'] = radvel.Parameter(value=None, vary=False)
params['secosw2'] = radvel.Parameter(value=None, vary=False)
params['sesinw2'] = radvel.Parameter(value=None, vary=False)



# Load radial velocity data, in this example the data is contained in an hdf file,
# the resulting dataframe or must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
path = os.path.join(radvel.DATADIR,'epic203771098.csv')
data = pd.read_csv(path)
data['time'] = data.t
data['mnvel'] = data.vel
data['errvel'] = data.errvel
data['tel'] = 'j'

# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.PositiveKPrior( nplanets ),             # Keeps K > 0
    radvel.prior.Gaussian('tc1', params['tc1'].value, 0.01), # Gaussian prior on tc1 with center at tc1 and width 0.01 days
    radvel.prior.Gaussian('per1', params['per1'].value, 0.01),
    radvel.prior.Gaussian('tc2', params['tc2'].value, 0.01),
    radvel.prior.Gaussian('per2', params['per2'].value, 0.01),
    radvel.prior.HardBounds('jit_j', 0.0, 15.0)
]

# abscissa for slope and curvature terms (should be near mid-point of time baseline)
time_base = np.mean([np.min(data.time), np.max(data.time)])  


# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=1.12, mstar_err= 0.05)

# optional argument that can contain planet radii,
# used for computing densities. Values should be given
# in units of Earth radii
planet = dict(
    rp1=5.68, rp_err1=0.56,
    rp2=7.82, rp_err2=0.72,
)

