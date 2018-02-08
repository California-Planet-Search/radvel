import numpy as np
import pandas as pd
import os
import radvel

# Data from Dai+ 2017
instnames = ['harps-n','pfs'] 
data = pd.read_csv(os.path.join(radvel.DATADIR,'k2-131.txt'), sep=' ')
t = np.array(data['time']) 
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])
telgrps = data.groupby('tel').groups
bjd = 0. 

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

# Define Celerite GP hyperparameters. # at end is array index.
# First Celerite Term
params['a_real_0'] = radvel.Parameter(value=1.)
params['c_real_0'] = radvel.Parameter(value=gp_explength_mean) 

# Second Celerite Term
params['a_comp_0'] = radvel.Parameter(value=gp_per_mean) 
params['b_comp_0'] = radvel.Parameter(value=gp_perlength_mean)
params['c_comp_0'] = radvel.Parameter(value=gp_per_mean) 
params['d_comp_0'] = radvel.Parameter(value=gp_perlength_mean)

# Second Celerite Term
params['a_comp_1'] = radvel.Parameter(value=gp_per_mean) 
params['b_comp_1'] = radvel.Parameter(value=gp_perlength_mean)
params['c_comp_1'] = radvel.Parameter(value=gp_per_mean) 
params['d_comp_1'] = radvel.Parameter(value=gp_perlength_mean)


hnames = {}
for tel in instnames:
  hnames[tel] = ['a_real_0', 'c_real_0', 'a_comp_0', 'b_comp_0', 'c_comp_0', 'd_comp_0', 
                 'a_comp_1', 'b_comp_1', 'c_comp_1', 'd_comp_1']

kernel_name = {'harps-n':"Celerite", 
               'pfs':"Celerite"}

jit_guesses = {'harps-n':2.0, 'pfs':5.3}

def initialize_instparams(tel_suffix):

    indices = telgrps[tel_suffix]

    params['gamma_'+tel_suffix] = radvel.Parameter(value=np.mean(vel[indices]))
    params['jit_'+tel_suffix] = radvel.Parameter(value=jit_guesses[tel_suffix]) 


for tel in instnames:
    initialize_instparams(tel)

priors = [radvel.prior.Gaussian('per1', Porb, Porb_unc),
          radvel.prior.Gaussian('tc1', Tc, Tc_unc),
          radvel.prior.Jeffreys('k1', 0.01, 10.),
          radvel.prior.Jeffreys('gp_amp', 0.01, 100.),
          radvel.prior.Jeffreys('jit_pfs', 0.01, 10.),
          radvel.prior.Jeffreys('jit_harps-n', 0.01,10.)]

