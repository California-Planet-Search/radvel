import numpy as np
import radvel.model
from radvel import gp
from scipy.linalg import cho_factor, cho_solve
from scipy import matrix

import numpy as np
import pandas as pd
from scipy import optimize
import os
import radvel
from radvel import gp
from radvel import model

# Data from Lopez-Moralez+ 2016
instnames = ['hires', 'harps'] 
data = pd.read_csv(os.path.join(radvel.DATADIR,'kepler21b.txt'), sep=' ')
t = np.array(data['time']) # BJD - 2400000
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])
telgrps = data.groupby('tel').groups

t[telgrps['hires']] += 40000. # put HIRES dates in same form as HARPS dates

#gran_err = 1.50
#errvel[telgrps['harps']] = np.sqrt((errvel[telgrps['harps']])**2 + (gran_err)**2)

nplanets = 1
basis_in  = 'per tc e w k'
fitting_basis = 'per tc se w k'

# Initialize Parameters object
params = radvel.Parameters(nplanets,basis=basis_in)

# Set initial parameter value guesses.
params['per1'] = radvel.Parameter(value=2.78578)
params['tc1'] = radvel.Parameter(value=2456798.7188 - 2400000)
params['e1'] = radvel.Parameter(value=0.02)
params['w1'] = radvel.Parameter(value=-15)
params['k1'] = radvel.Parameter(value=1.99)
params['dvdt'] = radvel.Parameter(value=0.)
params['curv'] = radvel.Parameter(value=0.)

# Convert to fitting basis
params = params.basis.to_any_basis(params,fitting_basis)

# Define vary attributes
params['dvdt'].vary = False
params['curv'].vary = False

# Define GP hyperparameters. Setting the "isGP" attribute tells
#	radvel this Parameter is a GP hyperparameter. The "telsshared"
#	attribute should contain a list of the telescopes that share
#	this hyperparameter.
params['gp_amp'] = radvel.Parameter(value=8.6, isGP=True, telsshared=instnames) 
params['gp_explength'] = radvel.Parameter(value=23.95, isGP=True, telsshared=instnames) 
params['gp_per'] = radvel.Parameter(value=12.63, isGP=True, telsshared=instnames) 
params['gp_perlength'] = radvel.Parameter(value=0.45, isGP=True, telsshared=instnames) 

# Instantiate a model object, with Parameters object as attribute
gpmodel = model.RVModel(params)

likes = []
def initialize(tel_suffix):

	# Instantiate a separate likelihood object for each instrument.
	# Each likelihood must use the same radvel.model object.
	indices = telgrps[tel_suffix]
	like = radvel.likelihood.GPLikelihood(gpmodel, t[indices], vel[indices], 
		                                  errvel[indices], suffix='_'+tel_suffix,
		                                  kernel_name="QuasiPer"
		                                  )

	# Add in instrument parameters
	like.params['gamma_'+tel_suffix] = radvel.Parameter(value=np.mean(vel[indices]))
	like.params['jit_harps'] = radvel.Parameter(value=1.0) 
	like.params['jit_hires'] = radvel.Parameter(value=5.4)
#	like.params['gran_harps'] = radvel.Parameter(value=0.) # granulation error term

	likes.append(like)

for tel in instnames:
	initialize(tel)

# Instantiate a CompositeLikelihood object that contains both instrument likelihoods
gplike = radvel.likelihood.CompositeLikelihood(likes)

# Instantiate a Posterior object
gppost = radvel.posterior.Posterior(gplike)

# Numbers for priors used in Lopez-Morales+ 2017
Ptrans = 2.7858212 # period in days
Ptrans_err = 0.0000032 
Tc = 2455093.83716 - 2400000. # time of inferior conjunction
Tc_err = 0.00085
Tev = 24.0 # spot evolution time, in days
Tev_err = 0.1
Prot = 12.62 # stellar rotation period, in days
Prot_err = 0.03
sigma_rv = np.std(vel[telgrps['harps']])

# Add in priors (Lopez-Morales+ Table 3)
gppost.priors += [radvel.prior.Gaussian('per1', Ptrans, Ptrans_err)]
gppost.priors += [radvel.prior.Gaussian('tc1', Tc, Tc_err)]
gppost.priors += [radvel.prior.ModifiedJeffreys('k1', sigma_rv, 2*sigma_rv)]
gppost.priors += [radvel.prior.EccentricityPrior( nplanets )]
gppost.priors += [radvel.prior.ModifiedJeffreys('gp_amp', sigma_rv, 2*sigma_rv)]
gppost.priors += [radvel.prior.Gaussian('gp_explength', Tev, Tev_err)]
gppost.priors += [radvel.prior.Gaussian('gp_per', Prot, Prot_err)]
gppost.priors += [radvel.prior.Gaussian('gp_perlength', 0.5, 0.05)]
gppost.priors += [radvel.prior.Jeffreys('jit_hires', 0.01, 10.)]
gppost.priors += [radvel.prior.Gaussian('jit_harps', 0.9, 0.1)]
# (Table 3 of Lopez-Morales+ says they use a Jeffrey's prior 
#  for jit_harps, but in the text they say they use a Gaussian.)

print gppost
print gppost.logprob()
print gppost.get_vary_params()
gppost.set_vary_params(gppost.get_vary_params())
print gppost.neglogprob_array(gppost.get_vary_params())

res = optimize.minimize(gppost.neglogprob_array, gppost.get_vary_params(), method="Powell")
