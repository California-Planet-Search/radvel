# Example Keplerian fit configuration file

# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel
import pdb
from cpsutils import io
from Evan_test import mkobsdb_keck
from Evan_test import KECKVSTDIR

# Define global planetary system and dataset parameters
starname = 'epic204221263'
nplanets = 2    # number of planets in the system
instnames = ['j']    # list of instrument names. Can be whatever you like but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw k'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 2454833

# Define prior centers (initial guesses) here.
params = radvel.RVParameters(nplanets,basis='per tc e w k')    # initialize RVparameters object

params['per1'] = 4.01628     # period of 1st planet
params['tc1'] = 2063.8753    # time of inferior conjunction of 1st planet
params['e1'] = 0.01          # eccentricity of 'per tc secosw sesinw logk'1st planet
params['w1'] = np.pi/2.      # argument of periastron of the star's orbit for 1st planet
params['k1'] = 10.0          # velocity semi-amplitude for 1st planet
params['per2'] = 10.56098      # same parameters for 2nd planet ...
params['tc2'] = 2067.4746
params['e2'] = 0.01
params['w2'] = np.pi/2.
params['k2'] = 1

params['dvdt'] = 0.0         # slope
params['curv'] = 0.0         # curvature

params['gamma_k'] = 0.0      # velocity zero-point for hires_rk
params['gamma_j'] = 1.0      # "                   "   hires_rj
params['gamma_a'] = 0.0      # "                   "   hires_apf

params['logjit_k'] = np.log(2.6)        # jitter for hires_rk
params['logjit_j'] = np.log(2.6)        # "      "   hires_rj
params['logjit_a'] = np.log(2.6)        # "      "   hires_apf


# Load radial velocity data, in this example the data is contained in an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
vstfile = KECKVSTDIR + '/vst' + 'epic204221263' + '.dat'
df = io.read_vst(vstfile)
df = df[['jd','mnvel','errvel']]
df.columns.values[df.columns.values=='jd'] = 'time'
df.time = df.time - bjd0
df['tel']='j'
datafile = os.path.join(radvel.DATADIR+'/epic204221263/','epic204221263.txt')
df.to_csv(datafile, sep=',')
data = pd.read_csv(os.path.join(radvel.DATADIR+'/epic204221263/','epic204221263.txt'), sep=',')
data = data[0:14]

# Set parameters to be held constant (default is for all parameters to vary). Must be defined in the fitting basis
vary = dict(
    dvdt = True,
    curv = False,
    logjit_k = False,
    logjit_j = False,
    logjit_a = False,
    per1 = False,
    per2 = False,
    tc1 = False,
    tc2 = False,
    e1 = False,
    e2 = False,
    w1= False,
    w2 = False
    
)


# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.Gaussian('tc1', params['tc1'], 300.0)    # Gaussian prior on tc1 with center at tc1 and width 300 days
]


time_base = np.mean([np.min(df.time), np.max(df.time)])   # abscissa for slope and curvature terms (should be near mid-point of time baseline)
