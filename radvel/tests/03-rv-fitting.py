from k2phot.config import bjd0
import pandas as pd
from loadpars import vst
import emcee
import corner
from scipy import optimize
from matplotlib.pylab import *
import radvel.model 
import radvel.plotting
import radvel.mcmc 
import radvel.fitting
time_base = 2420
t = np.array( vst.jd ) - bjd0
nptsi= 1000
ti = linspace( t[0] - 1, t[-1] + 1,nptsi)    
vel = np.array( vst.mnvel )
errvel = np.array( vst.errvel )
planet = pd.read_hdf('planet-parameters.hdf','planet')

def initialize_model():
    mod = radvel.model.Model_logK(2, time_base=time_base)
    mod.params0['P1'].value = planet.ix['P','b']
    mod.params0['tc1'].value = planet.ix['t0','b']
    mod.params0['sqrtecosom1'].value = 0.01
    mod.params0['sqrtesinom1'].value = 0.01
    mod.params0['logk1'].value = 0.0
    mod.params0['P2'].value = planet.ix['P','c']
    mod.params0['tc2'].value = planet.ix['t0','c']
    mod.params0['sqrtecosom2'].value = 0.01
    mod.params0['sqrtesinom2'].value = 0.01
    mod.params0['logk2'].value = 0.0
    mod.params0['gamma'].value = 2
    mod.params0['dvdt'].value = 0.1
    mod.params0['logjitter'].value = np.log10(3)
    mod.params0['P1'].vary = False
    mod.params0['tc1'].vary = False
    mod.params0['P2'].vary = False
    mod.params0['tc2'].vary = False
    return mod

rc('lines',linewidth=1.5)
rc('axes',linewidth=1.0)

print "\n" * 5
print "Circular orbit with variable jitter"
mod = initialize_model()
mod.params0['sqrtecosom1'].vary = False
mod.params0['sqrtesinom1'].vary = False
mod.params0['sqrtecosom2'].vary = False
mod.params0['sqrtesinom2'].vary = False
mod.set_vary_parameters()
params_maxlike, maxlnlike = radvel.fitting.maxlike_fitting(mod, t, vel, errvel)
radvel.plotting.rvplot(mod, params_maxlike , t, vel, errvel)
fig = gcf()
axL = fig.get_axes()
sca(axL[0])
xlabel('BJD - %i' % bjd0)
ylabel('Radial Velocity')
xlim(2360,2480)

sca(axL[1])
title('Planet b')

sca(axL[2])
title('Planet c')

# Start MCMC near the best fit
#mod.params0 = params_maxlike
#chain = rv.mcmc.mcmc(
#   mod, t, vel, errvel, nwalkers=100, nburn=1000, nrun=2000, threads=8
#   )
#chain.to_hdf('rvfit.hdf','rv-logk-circ')

print "\n" * 5
print "Eccentric orbit with variable jitter"
mod = initialize_model()
mod.set_vary_parameters()
params_maxlike, maxlnlike = radvel.fitting.maxlike_fitting(mod, t, vel, errvel)
sca(axL[0])
radvel.plotting.plotfit(mod, params_maxlike, t, vel, errvel,linestyle='--',zorder=1.95,color='RoyalBlue')
fig.set_tight_layout(True)
#fig.savefig('/Users/petigura/Research/K2/Papers/EPIC2037/EPIC2037/fig_rv.pdf')
# Start MCMC near the best fit
# mod.params0 = mod.array_to_params(p1)
# chain = rv.model.mcmc(
#    mod, t, vel, errvel, nwalkers=100, nburn=1000, nrun=2000, threads=8
#    )
# chain.to_hdf('rvfit.hdf','rv-logk')


print "\n" * 5
print "Circular orbit with fixed jitter"
mod = initialize_model()
mod.params0['sqrtecosom1'].vary = False
mod.params0['sqrtesinom1'].vary = False
mod.params0['sqrtecosom2'].vary = False
mod.params0['sqrtesinom2'].vary = False
mod.params0['logjitter'].vary = False 
mod.params0['logjitter'].value = params_maxlike['logjitter'].value
mod.set_vary_parameters()
params_maxlike, maxlnlike = radvel.fitting.maxlike_fitting(mod, t, vel, errvel)

print "\n" * 5
print "Eccentric orbit with fixed jitter "
mod = initialize_model()
mod.set_vary_parameters()
mod.params0['logjitter'].vary = False 
mod.params0['logjitter'].value = params_maxlike['logjitter'].value
mod.set_vary_parameters()
params_maxlike, maxlnlike = radvel.fitting.maxlike_fitting(mod, t, vel, errvel)

#chain = pd.read_hdf('rvfit.hdf','rv-logk')
#chain =  chain.query('logk1 > 0 and logk2 > 0')
#corner.corner(
#    chain[mod.vary_parameters],
#    labels=mod.vary_parameters,
#    quantiles=[0.16, 0.5, 0.84],
#    plot_datapoints=False,
#    smooth=False,
#    bins=40,
#    levels=[0.68,0.95],
#    )
#

