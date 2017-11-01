# Example Gaussian Process fit configuration file

import numpy as np
import pandas as pd
import os
import radvel

starname = 'Kepler21'
stellar = dict(mstar=1.34, mstar_err=0.06)

# Data from Lopez-Moralez+ 2016
instnames = ['hires', 'harps'] 
data = pd.read_csv(os.path.join(radvel.DATADIR,'kepler21b.txt'), sep=' ')
t = np.array(data['time']) # BJD - 2400000
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])
telgrps = data.groupby('tel').groups

t[telgrps['hires']] += 40000. # put HIRES dates in BJD - 2400000 form

time_base = np.mean([np.min(t), np.max(t)])  

nplanets = 1
fitting_basis = 'per tc se w k'

# Relevant numbers from Lopez-Morales+ 2017
Ptrans = 2.7858212 # period in days
Ptrans_err = 0.0000032 
Tc = 2455093.83716 - 2400000. # time of inferior conjunction
Tc_err = 0.00085
Tev = 24.0 # spot evolution time, in days
Tev_err = 0.1
Prot = 12.62 # stellar rotation period, in days
Prot_err = 0.03
sigma_rv = np.std(vel[telgrps['harps']])

# Initialize Parameters object.
params = radvel.Parameters(nplanets,basis=fitting_basis)

# Set initial parameter value guesses.
params['per1'] = radvel.Parameter(value=Ptrans)
params['tc1'] = radvel.Parameter(value=2456798.7188 - 2400000)
params['se1'] = radvel.Parameter(value=np.sqrt(0.02))
params['w1'] = radvel.Parameter(value=-15)
params['k1'] = radvel.Parameter(value=2.0)
params['dvdt'] = radvel.Parameter(value=0.)
params['curv'] = radvel.Parameter(value=0.)

# Define vary attributes
params['dvdt'].vary = False
params['curv'].vary = False

for tel in instnames:
    indices = telgrps[tel]
    params['gamma_'+tel] = radvel.Parameter(value=np.mean(vel[indices]))

params['jit_harps'] = radvel.Parameter(value=1.0) 
params['jit_hires'] = radvel.Parameter(value=5.0)

# Priors from Lopez-Morales+ Table 3
priors = [
    radvel.prior.Gaussian('per1', Ptrans, Ptrans_err),
    radvel.prior.Gaussian('tc1', Tc, Tc_err),
    radvel.prior.ModifiedJeffreys('k1', sigma_rv, 2*sigma_rv),
    radvel.prior.EccentricityPrior( nplanets ),
    radvel.prior.ModifiedJeffreys('gp_amp', sigma_rv, 2*sigma_rv),
    radvel.prior.Gaussian('gp_explength', Tev, Tev_err),
    radvel.prior.Gaussian('gp_per', Prot, Prot_err),
    radvel.prior.Gaussian('gp_perlength', 0.5, 0.05),
    radvel.prior.Jeffreys('jit_hires', 0.01, 10.),
    radvel.prior.Gaussian('jit_harps', 0.9, 0.1)
]
# (Table 3 of Lopez-Morales+ says they use a Jeffrey's prior 
#  for jit_harps, but in the text they say they use a Gaussian.)

# Define GP hyperparameters. 
params['gp_amp'] = radvel.Parameter(value=9.0, isGP=True, telsshared=instnames) 
params['gp_explength'] = radvel.Parameter(value=Tev, isGP=True, telsshared=instnames) 
params['gp_per'] = radvel.Parameter(value=Prot, isGP=True, telsshared=instnames) 
params['gp_perlength'] = radvel.Parameter(value=0.5, isGP=True, telsshared=instnames) 
# TIPS: - Set "isGP" attribute for GP hyperparameters. 
#         - "telsshared" should be a list of keys for 
#           telescopes that share this hyperparameter.
