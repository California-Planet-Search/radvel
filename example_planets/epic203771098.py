# Example Keplerian fit configuration file

# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel

# Define global planetary system and dataset parameters
starname = 'epic203771098'
nplanets = 2    # number of planets in the system
instnames = ['j']    # list of instrument names. Can be whatever you like (no spaces) but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw k'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 2454833.
planet_letters = {1: 'b', 2:'c'}

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k')    # initialize Parameters object

anybasis_params['per1'] = radvel.Parameter(value=20.885258)    # period of 1st planet
anybasis_params['tc1'] = radvel.Parameter(value=2072.79438)    # time of inferior conjunction of 1st planet
anybasis_params['e1'] = radvel.Parameter(value=0.01)          # eccentricity of 'per tc secosw sesinw logk'1st planet
anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)      # argument of periastron of the star's orbit for 1st planet
anybasis_params['k1'] = radvel.Parameter(value=10.0)         # velocity semi-amplitude for 1st planet
anybasis_params['per2'] = radvel.Parameter(value=42.363011)    # same parameters for 2nd planet ...
anybasis_params['tc2'] = radvel.Parameter(value=2082.62516)
anybasis_params['e2'] = radvel.Parameter(value=0.01)
anybasis_params['w2'] = radvel.Parameter(value=np.pi/2.)
anybasis_params['k2'] = radvel.Parameter(value=10.0)

anybasis_params['dvdt'] = radvel.Parameter(value=0.0)        # slope
anybasis_params['curv'] = radvel.Parameter(value=0.0)         # curvature

anybasis_params['gamma_j'] = radvel.Parameter(1.0)      # velocity zero-point for hires_rj
anybasis_params['jit_j'] = radvel.Parameter(value=2.6)        # jitter for hires_rj


# Convert input orbital parameters into the fitting basis
params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

# Set the 'vary' attributes of each of the parameters in the fitting basis. A parameter's 'vary' attribute should
# 	be set to False if you wish to hold it fixed during the fitting process. By default, all 'vary' parameters
#	are set to True.
params['secosw1'].vary = False
params['sesinw1'].vary = False
params['secosw2'].vary = False
params['sesinw2'].vary = False
params['tc1'].vary = False
params['per1'].vary = False
params['tc2'].vary = False
params['per2'].vary = False

# Load radial velocity data, in this example the data is contained in an hdf file,
# the resulting dataframe or must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
path = os.path.join(radvel.DATADIR, 'epic203771098.csv')
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

