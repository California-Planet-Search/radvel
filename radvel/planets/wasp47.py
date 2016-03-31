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
starname = 'wasp47'
nplanets = 4    # number of planets in the system
instnames = ['j']    # list of instrument names. Can be whatever you like but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc e w k'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 2454833

# Define prior centers (initial guesses) here.
params = radvel.RVParameters(nplanets,basis='per tc e w k')    # initialize RVparameters object

params['per1'] = 4.1591409     # period of 1st planet
params['tc1'] = 2149.97697    # time of inferior conjunction of 1st planet
params['e1'] = 0.01          # eccentricity of 'per tc secosw sesinw logk'1st planet
params['w1'] = np.pi/2.      # argument of periastron of the star's orbit for 1st planet
params['k1'] = 140.0          # velocity semi-amplitude for 1st planet

params['per2'] = 572.0    #Neveu-VanMalle et al 2015
params['tc2'] = 1148.0
params['e2'] = 0.13
params['w2'] = np.pi/2.
params['k2'] = 30.0          

params['per3'] = 0.789518    
params['tc3'] = 2148.3450
params['e3'] = 0.01
params['w3'] = np.pi/2.
params['k3'] = 8.0

params['per4'] = 9.03164     # Dai et al 2015
params['tc4'] = 2155.3075
params['e4'] = 0.01
params['w4'] = np.pi/2.
params['k4'] = 3.0




params['dvdt'] = 0.0         # slope
params['curv'] = 0.0         # curvature

params['gamma_k'] = 0.0      # velocity zero-point for hires_rk
params['gamma_j'] = 1.0      # "                   "   hires_rj
params['gamma_a'] = 0.0      # "                   "   hires_apf

params['logjit_k'] = np.log(2.6)        # jitter for hires_rk
params['logjit_j'] = np.log(2.6)        # "      "   hires_rj
params['logjit_a'] = np.log(2.6)        # "      "   hires_apf


# Load radial velocity data, in this example the data is contained in an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
vstfile = KECKVSTDIR + '/vst' + starname + '.dat'
df = io.read_vst(vstfile)
df = df[['jd','mnvel','errvel']]
df.columns.values[df.columns.values=='jd'] = 'time'
df.time = df.time - bjd0
df['tel']='j'

datadir = os.path.join(radvel.DATADIR+'/' + starname)
if not os.path.isdir(datadir):
    os.mkdir(datadir)
datafile =  datadir + '/' + starname + '.txt'
df.to_csv(datafile, sep=',')
data = pd.read_csv(datafile)
data=data.drop(range(27)).reset_index(drop=True)
print data

# Set parameters to be held constant (default is for all parameters to vary). Must be defined in the fitting basis
vary = dict(
    dvdt = True,
    curv = False,
    logjit_k = False,
    logjit_j = False,
    logjit_a = False,
    per1 = False,
    tc1 = False,
#secosw1 =False,
#sesinw1 = False,
    e1=False,
    w1=False,
    per2 = True,
    tc2 = True,
#secosw2 = True,
#sesinw2 = True,
    e2=False,
    w2=False,
    per3 = False,
    tc3 = False,
#secosw3 = False,
#sesinw3 = False,
    e3=False,
    w3=False,
    per4 = False,
    tc4 = False,
#    secosw4 = False,
#    sesinw4 = False
    e4=False,
    w4=False
)


# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
# radvel.prior.PositiveKPrior( nplanets ),             # Keeps K > 0
    radvel.prior.Gaussian('tc1', params['tc1'], 0.01), # Gaussian prior on tc1 with center at tc1 and width 0.01 days
    radvel.prior.Gaussian('per1', params['per1'], 0.01),
    radvel.prior.Gaussian('tc2', params['tc2'], 16.0),
    radvel.prior.Gaussian('per2', params['per2'], 7.0),
    radvel.prior.Gaussian('e2', 0.13, 0.1),
    radvel.prior.Gaussian('tc3', params['tc3'], 0.01),
    radvel.prior.Gaussian('per3', params['per3'], 0.01)
]


time_base = np.mean([np.min(df.time), np.max(df.time)])   # abscissa for slope and curvature terms (should be near mid-point of time baseline)

