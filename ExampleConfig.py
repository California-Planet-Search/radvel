# Example Keplerian fit configuration file
# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel

# Define global planetary system and dataset parameters
starname = 'ExampleConfig'
nplanets = 1    # number of planets in the system
instnames = ['j']    # list of instrument names. Can be whatever you like but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw k'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 2454833.
planet_letters = {1: 'b'}

stellar = {'mstar':0.94,
           'mstar_err':0.05
           }

planet = {'rp1':1.61,
          'rp_err1':0.13,
          #'rp2':1.0,
          #'rp_err2':1.0
          }
    
# Define prior centers (initial guesses) here.
params = radvel.RVParameters(nplanets,basis='per tc e w k')    # initialize RVparameters object

params['per1'] = 2.369193     # period of 1st planet
params['tc1'] = 2148.64248 + 2454833.   # time of inferior conjunction of 1st planet
params['e1'] = 0.0          # eccentricity of 'per tc secosw sesinw logk'1st planet
params['w1'] = np.pi/2.      # argument of periastron of the star's orbit for 1st planet
params['k1'] = 2.0         # velocity semi-amplitude for 1st planet

params['dvdt'] = 0.0         # slope
params['curv'] = 0.0         # curvature

params['gamma_j'] = 1.0      # "                   "   hires_rj
params['jit_j'] = 2.6        # "      "   hires_rj

gp_kernel = 'SqExp'
gp_hyperparams = {'j': [10., 33.]
                 }

gp = radvel.GP(gp_kernel, gp_hyperparams, ['amp', 'scale'], shared = [], xpred=None, plot_sigma = [1])  # initialize GP object

#gp_kernel = 'Periodic'
#gp_hyperparams = {'j': [6.0, 30., 2.0],
                  #'sim': [6.0, 30., 2.0]
#                      }

#gp = radvel.GP(gp_kernel, gp_hyperparams, ['amp', 'prot', 'roughness'], shared = [], xpred=None, plot_sigma = [1])  # initialize GP object


# Load radial velocity data, must have 'time', 'mnvel', 'errvel', and 'tel' keys
datadir = os.path.join(radvel.ROOTDIR + 'data/')
if not os.path.isdir(datadir):
    os.mkdir(datadir)

datafile =  os.path.join(datadir, 'epic206011496.txt')
data = pd.read_csv(datafile)



# Set parameters to be held constant (default is for all parameters to vary). Must be defined in the fitting basis
vary = dict(
    dvdt = False,
    curv = False,
    per1 = False,
    tc1 = False,
    secosw1 = False,
    sesinw1 = False,
)


# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.PositiveKPrior( nplanets ),             # Keeps K > 0
    radvel.prior.Gaussian('tc1', params['tc1'], 0.01), # Gaussian prior on tc1 with center at tc1 and width 0.01 days
    radvel.prior.Gaussian('per1', params['per1'], 0.01),
    radvel.prior.HardBounds('scale_j', 0., 10000.),
    radvel.prior.HardBounds('amp_j', 0., 10000.),
#    radvel.prior.HardBounds('amp_j', 0., 10000.),
#    radvel.prior.HardBounds('prot_j', 0., 10000.),
#    radvel.prior.HardBounds('roughness_j', 0., 10000.),
]

time_base = np.mean([np.min(data.time), np.max(data.time)])   # abscissa for slope and curvature terms (should be near mid-point of time baseline)


