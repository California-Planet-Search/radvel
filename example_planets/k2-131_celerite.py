import numpy as np
import pandas as pd
import os
import radvel

# This setup file is provided to illustrate how to set up
# a config file for use with Dan Foreman-Mackey's `celerite`
# package. The results it produces are not meant to be
# compared with those of Dai et al. (2017).

# Data from Dai+ 2017
instnames = ['harps-n','pfs'] 
data = pd.read_csv(os.path.join(radvel.DATADIR,'k2-131.txt'), sep=' ')
t = np.array(data['time']) 
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])
telgrps = data.groupby('tel').groups
bjd = 0. 

# Constraints from transits
Porb = 0.3693038 # [days]
Porb_unc = 0.0000091 
Tc = 2457582.9360 # [BJD]
Tc_unc = 0.0011

starname = 'k2-131'
ntels = len(instnames)      
planet_letters = {1: 'b'}   

nplanets = 1
fitting_basis = 'per tc secosw sesinw k'

# Set initial parameter value guesses.
params = radvel.Parameters(nplanets,basis=fitting_basis, planet_letters=planet_letters)

params['per1'] = radvel.Parameter(value=Porb)
params['tc1'] = radvel.Parameter(value=Tc)
params['sesinw1'] = radvel.Parameter(value=0.,vary=False) # hold eccentricity fixed at 0
params['secosw1'] = radvel.Parameter(value=0.,vary=False)
params['k1'] = radvel.Parameter(value=6.55)
params['dvdt'] = radvel.Parameter(value=0.,vary=False)
params['curv'] = radvel.Parameter(value=0.,vary=False)
time_base = np.median(t)

# Define Celerite GP hyperparameters.

# First Celerite Term. AC>=BD must be true to ensure 
# positive-definiteness. (A == params['1_logA'].value, etc.)
params['1_logA'] = radvel.Parameter(value=np.log(26.))
params['1_logB'] = radvel.Parameter(value=np.log(.0005)) 
params['1_logC'] = radvel.Parameter(value=np.log(.5))
params['1_logD'] = radvel.Parameter(value=np.log(.0005))

# Second Celerite Term (real). Setting vary=False for 1_logB and 
# 1_logD ensures that this term remains real throughout the fitting process.
params['2_logA'] = radvel.Parameter(value=np.log(26.)) 
params['2_logB'] = radvel.Parameter(value=np.log(0.),vary=False)
params['2_logC'] = radvel.Parameter(value=np.log(0.)) 
params['2_logD'] = radvel.Parameter(value=np.log(0.1),vary=False)

hnames = {}
for tel in instnames:
  hnames[tel] = ['1_logA','1_logB','1_logC','1_logD',
                 '2_logA','2_logB','2_logC','2_logD'
                 ]

kernel_name = {'harps-n':"Celerite", 
               'pfs':"Celerite"}

jit_guesses = {'harps-n':2.0, 'pfs':5.0}

def initialize_instparams(tel_suffix):

    indices = telgrps[tel_suffix]

    params['gamma_'+tel_suffix] = radvel.Parameter(value=np.mean(vel[indices]))
    params['jit_'+tel_suffix] = radvel.Parameter(value=jit_guesses[tel_suffix]) 


for tel in instnames:
    initialize_instparams(tel)

priors = [radvel.prior.Gaussian('per1', Porb, Porb_unc),
          radvel.prior.Gaussian('tc1', Tc, Tc_unc),
          radvel.prior.Jeffreys('k1', 0.01, 10.),
          radvel.prior.Jeffreys('jit_pfs', 0.01, 10.),
          radvel.prior.Jeffreys('jit_harps-n', 0.01,10.)
          ]

