"""
RV Model

Define a class that can do the following things:

1. (done) Synthesize RV models with an arbitrary number of planets.
2. (done) Given data, errors, and parameters return a likelihood
3. (done) Ability to float/fix parameters
4. (todo) Ability to add priors to parameters.
5. (todo) Support multiple "bases" i.e. re-parameterizations of P, tp, e, om, K
6. (todo) Easily return single planet model
7. (todo) Performant (perhaps re-write core aspects of rvkep.py in C)
"""

from scipy import optimize 
from lmfit import Parameters, minimize, fit_report
import rvkep
from matplotlib.pylab import * 
import copy
import pandas as pd
import copy_reg
import types

import lmfit

class Parameters(lmfit.Parameters):
    def __repr__(self):
        s = "  Name Value Vary\n"
        for i, key in enumerate(self.keys()):
            s+="{0} {1} {2} {3}\n".format(
                i, key, self[key].value, self[key].vary
                )
        return s

class ModelBase(object):
    def __init__(self, num_planets, time_base=0):
        params0 = Parameters()

        # Load up the best fitting circular orbit
        print "Initializing model with {} planets".format(num_planets)
        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                params0.add('{0}{1}'.format(parameter,num_planet),vary=True)
        
        params0.add('gamma')
        params0.add('dvdt')
        params0.add('logjitter')

        self.num_planets = num_planets
        self.params0 = params0
        self.time_base = time_base

    def set_vary_parameters(self):
        """
        """
        keys = []
        for k in self.params0.iterkeys():
            if self.params0[k].vary:
                keys += [k]

        self.vary_parameters = keys
        self.nvary_parameters = len(keys)

        print "{} floating parameters:".format(self.nvary_parameters)
        print self.vary_parameters
        print "if you want to change, instantiate a new Model object"

    def model_single_planet(self, params, t, num_planet):
        """
        Model Single Planet
        
        Removes contribution from other planets as well as trends
        
        Parameters 
        ----------
        params : lmfit-style parameters
        t : time values to synthesize model
        num_planet : which planet to model (starting at 1).
        
        Returns
        -------
        vel : velocities
        """

        params_single_planet = copy.deepcopy(params)
        for num in range(1, self.num_planets+1):
            if num!=num_planet:
                params_single_planet['k%i' % num].value = 0

        for key in trend_parameters:
            params_single_planet[key].value = 0

        return self.model(params_single_planet, t)

    
    def residuals(self, params, t, vel):
        _resid = vel - self.model( params, t )
        return _resid

    def lnprob(self, params, t, vel, errvel):
        """
        Return log-likelihood given parameters, time, rv, and rv errors
 
        Parameters
        ----------
        params : parameter object
        t : time
        vel : radial velocities
        errvel : uncertanties

        Returns
        -------
        lnlike : Natural log of likelihood
        """

        for num_planet in range(1, self.num_planets+1):
            sqrtecosom = params['sqrtecosom%i' % num_planet].value
            sqrtesinom = params['sqrtesinom%i' % num_planet].value
            ecc = sqrtecosom**2 + sqrtesinom**2
            if ecc > 1.0:
                return -np.inf
            
        sigma_jit = 10**params['logjitter'].value
        # Eccentricity prior
        residuals = self.residuals(params, t, vel)
        lnlike = lnlike_jitter(residuals, errvel, sigma_jit)
        return lnlike

    def lnprob_array(self, params_array, t, vel, errvel):
        """
        Compute log-likelihood from array.
        
        Parameters
        ----------
        params_array : numpy array of parameters (order matters)
        t : time
        vel : velocities
        errvel : errors on the velocities

        Returns
        -------
        lnprob : Natural Log of Likelihood
        """
        params = self.array_to_params(params_array)
        lnprob = self.lnprob(params, t, vel, errvel)
        return lnprob

    def array_to_params(self, params_array):
        """
        Convert numpy array of parameters into Parameters object

        Parameters
        ----------
        params_array : numpy array of parameters
        
        Returns
        -------
        params : Parameters object with values from numpy array passed in
        """
        
        params = copy.deepcopy(self.params0)
        for i, key in enumerate(self.vary_parameters):
            params[key].value = params_array[i]

        return params

    def params_to_array(self, params):
        """
        Convert Parameters object into a numpy array.

        Parameters
        ----------
        params : Parameters object

        Returns
        -------
        params_array : numpy array representation
        """
        params_array = []
        for i, key in enumerate(self.vary_parameters):
            params_array += [ self.params0[key].value ]
        params_array = np.array(params_array)
        return params_array

class Model(ModelBase):
    planet_parameters = 'P tc sqrtecosom sqrtesinom k'.split()
    trend_parameters = 'gamma dvdt'.split()

    def model_single_planet(self, params, t, num_planet):
        """
        """
        params_single_planet = copy.deepcopy(params)
        for num in range(1, self.num_planets+1):
            if num!=num_planet:
                params_single_planet['k%i' % num].value = 0.0

        for key in self.trend_parameters:
            params_single_planet[key].value = 0

        return self.model(params_single_planet, t)

    def model(self, params, t):
        vals = params.valuesdict()

        # Build up list of orbital elelments
        vel = np.zeros(len(t))
        for num_planet in range(1,self.num_planets+1):
            orbel_tcsqrtecos = []
            for parameter in self.planet_parameters:
                val = vals['{0}{1}'.format(parameter,num_planet)]
                orbel_tcsqrtecos += [ val ] 

            orbel_tcsqrtecos = np.array(orbel_tcsqrtecos)
            orbel_tcsqrtecos = np.hstack( [orbel_tcsqrtecos, np.zeros(3)] )
            orbel_tcecos = basis_tcsqrtecos_to_tcecos(orbel_tcsqrtecos)
            orbel_cps = rvkep.basis_tcecos2cps(orbel_tcecos)
            vel += rvkep.rv_drive(t, orbel_cps, time_base=0 )

        vel += (vals['gamma'] + vals['dvdt']*( t - self.time_base ))
        return vel 

class Model_logK(ModelBase):
    planet_parameters = 'P tc sqrtecosom sqrtesinom logk'.split()
    trend_parameters = 'gamma dvdt'.split()

    def model(self, params, t):
        vals = params.valuesdict()

        # Build up list of orbital elelments
        vel = np.zeros(len(t))
        for num_planet in range(1,self.num_planets+1):
            orbel = [] # P tc sqrtecosom sqrtesinom logk
            for parameter in self.planet_parameters:
                val = vals['{0}{1}'.format(parameter,num_planet)]
                orbel += [ val ] 

            orbel = np.array(orbel)            
            orbel[4] = 10**orbel[4] # logK -> K
            orbel = np.hstack( [orbel, np.zeros(3)] )

            # Convert to P tc ecosom esinom K
            orbel_tcecos = basis_tcsqrtecos_to_tcecos(orbel)

            # Convert to P tp e om K
            orbel_cps = rvkep.basis_tcecos2cps(orbel_tcecos)
            vel += rvkep.rv_drive(t, orbel_cps, time_base=0 )

        vel += (vals['gamma'] + vals['dvdt']*( t - self.time_base ))
        return vel 

    def model_single_planet(self, params, t, num_planet):
        """
        """

        params_single_planet = copy.deepcopy(params)
        for num in range(1, self.num_planets+1):
            if num!=num_planet:
                params_single_planet['logk%i' % num].value = -9.0

        for key in self.trend_parameters:
            params_single_planet[key].value = 0

        return self.model(params_single_planet, t)



def lnlike_jitter(residuals, sigma, sigma_jit):
    """
    Log-likelihood incorporating jitter

    See equation (1) in Howard et al. 2014. Returns lnlikelihood, where 
    sigma**2 is replaced by sigma**2 + sigma_jit**2. It penalizes
    excessively large values of jitter
    
    Parameters
    ----------
    residuals : array
    sigma : float "measurement errors"
    sigma_jit : float "jitter"
    """

    sum_sig_quad = sigma**2 + sigma_jit**2
    penalty = np.sum( np.log( np.sqrt( 2 * np.pi * sum_sig_quad ) ) )
    chi2 = np.sum(residuals**2 / sum_sig_quad)
    lnlike = -0.5 * chi2 - penalty 
    return lnlike

def basis_tcsqrtecos_to_tcecos(orbel_tcsqrtecos):
    orbel_tcecos = orbel_tcsqrtecos.copy() 
    e = orbel_tcsqrtecos[2]**2 + orbel_tcsqrtecos[3]**2
    se = np.sqrt(e)
    orbel_tcecos[2] = orbel_tcsqrtecos[2] * se
    orbel_tcecos[3] = orbel_tcsqrtecos[3] * se
    return orbel_tcecos



def generate_synthetic_single_planet_model():
    """
    Test the RV model by generating some synthetic data and fitting it.
    """
    npts = 100
    sigma = 0.1
    sigma_jit = 0.3

    trange = [1000,1002] # Set to explore the effect of changing tbase
    t = linspace(*(trange+[npts]))

    np.random.seed(0)
    mod = Model(1, time_base=1000)
    mod.params0['P1'].value = 1 
    mod.params0['tc1'].value = 0
    mod.params0['sqrtecosom1'].value = 0.3
    mod.params0['sqrtesinom1'].value = 0.3
    mod.params0['k1'].value = 1
    mod.params0['gamma'].value = 1
    mod.params0['dvdt'].value = 0.1
    mod.params0['logjitter'].value = np.log10(1)
    mod.params0['P1'].vary = False
    mod.params0['tc1'].vary = False
    mod.set_vary_parameters()

    # Simulate rv and plot
    vel = mod.model(mod.params0, t)
    errvel = np.ones(npts) * sigma
    vel += np.random.randn( npts ) * sigma
    vel += np.random.randn( npts ) * sigma_jit

    # Perturb the parameters to simulate ignorance of real values
    mod.params0['sqrtecosom1'].value = 0.01
    mod.params0['sqrtesinom1'].value = 0.01
    mod.params0['k1'].value = 2
    mod.params0['gamma'].value = 0.1
    mod.params0['dvdt'].value = 0.1
    mod.params0['logjitter'].value = -3
    return t, vel, errvel, mod

def test_rv_fitting():
    nptsi = 1000
    t, vel, errvel, mod = generate_synthetic_single_planet_model()
    ti = linspace( t[0] - 1, t[-1] + 1,nptsi)    
    p0 = mod.params_to_array()
    veli_0 = mod.model(mod.params0, ti)
    def obj(*args):
        _obj = -1.0 * mod.lnprob_array(*args) 
        return _obj

    #    ['sqrtecosom1', 'sqrtesinom1', 'k1', 'gamma', 'dvdt', 'logjitter']    
    bounds = [(-.99,.99), (-.99,.99), (-inf,inf), (-inf,inf), (-inf,inf), (-inf,inf),]
    p1 = optimize.fmin_l_bfgs_b(
        obj, p0, args=(t, vel, errvel,) ,approx_grad=1,bounds=bounds
        )
    p1 = p1[0]
    params_bestfit = mod.array_to_params(p1) 
    veli = mod.model(params_bestfit, ti)

    plot(t, vel, '.')
    plot(ti, veli_0, 'r', alpha=0.8, label='initial fit')
    plot(ti, veli, 'b', alpha=0.8, label='final fit')
    legend()
    return mod, t, vel, vel_err 


# I had to add these methods to get the model object to be
# pickle-able, so we could run the mcmc in as a in multi-threaded
# mode.
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
