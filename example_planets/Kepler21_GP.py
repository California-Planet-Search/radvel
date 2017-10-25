import numpy as np
import pandas as pd
from scipy import optimize
import os
import radvel
from radvel import gp
from radvel import model

# data from Lopez-Moralez+ 2016
instnames = ['harps'] # for now, we'll just use data from one telescope
data = pd.read_csv(os.path.join(radvel.DATADIR,'kepler21b.txt'), sep=' ')
t = np.array(data['time'])
vel = np.array(data['mnvel'])
errvel = np.array(data['errvel'])


# RV orbital parameters
basis_fit  = 'per tc e w k'
params = radvel.Parameters(1,basis=basis_fit)

# Initialize values
params['per1'] = radvel.Parameter(value=1201.1 + 0.4)
params['tc1'] = radvel.Parameter(value=2456778. + 1.)
params['e1'] = radvel.Parameter(value=0.01)
params['w1'] = radvel.Parameter(value=0.01)
params['k1'] = radvel.Parameter(value=1.0)
params['dvdt'] = radvel.Parameter(value=0.)
params['curv'] = radvel.Parameter(value=0.)

params['dvdt'].vary = False
params['curv'].vary = False

# Add in noise parameters
params['gamma_hires'] = radvel.Parameter(value=1.0)
params['gamma_harps'] = radvel.Parameter(value=1.0)


params['gp_length_hires'] = radvel.Parameter(value=1.0) #
params['gp_amp_share'] = radvel.Parameter(value=1.0) # 
params['gp_per_share'] = radvel.Parameter(value=1.0) # 
params['jit_hires'] = radvel.Parameter(value=1.0) # 

# instantiate a GP model object, with hyperparameters as attributes
gpmodel = model.GPModel(params, kernel_name="QuasiPer") 

gplike = radvel.likelihood.GPLikelihood(gpmodel, t, vel, errvel, suffix='hires')
gppost = radvel.posterior.Posterior(gplike)

# find max likelihood orbital parameters, noise parameters, and hyperparameters. 
res = optimize.minimize(gppost.neglogprob_array, gppost.get_vary_params(), method="Powell")


# TODO: add methods to GPModel object to calculate mean, draw samples from posterior
# TODO: add posterior characterization methods (choices = MCMC & Evan's implemented multivariate Gaussian sampler)
# TODO: add in Evan's plotting tools
