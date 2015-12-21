"""
RV Model

Define a class that can do the following things:
"""

from lmfit import Parameters
import rvkep
from matplotlib.pylab import * 
import copy_reg
import types

class Parameters(Parameters):
    def __repr__(self):
        s = "  Name Value Vary\n"
        for i, key in enumerate(self.keys()):
            s+="{0} {1} {2} {3}\n".format(
                i, key, self[key].value, self[key].vary
                )
        return s

class GenericModel(object):
    def set_vary_params(self, params_array):
        i = 0
        for key,param in self.params.iteritems():
            if param.vary:
                param.value = params_array[i]
                i+=1
        assert i==len(params_array), \
            "length of array must match number of varied parameters"

    def get_vary_params(self):
        params_array = []
        for key,param in self.params.iteritems():
            if param.vary:
                params_array += [param.value] 
        params_array = np.array(params_array)
        return params_array

class GenericRVModel(GenericModel):
    """
    Generic RV Model

    This class defines the methods common to all RV modeling
    classes. The different RV models, having different
    parameterizations inherit from this class.
    """

    def __init__(self, num_planets, time_base=0):
        params = Parameters()

        # Load up the best fitting circular orbit
        print "Initializing model with {} planets".format(num_planets)
        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                params.add('{0}{1}'.format(parameter, num_planet),vary=True)
        
        params.add('dvdt')
        self.num_planets = num_planets
        self.params = params
        self.time_base = time_base

    def __call__(self, t):
        """
        Compute the radial velocity due to all Keplerians and
        additional trends.
        """
        vals = self.params.valuesdict()
        vel = np.zeros(len(t))
        for num_planet in range(1, self.num_planets+1):
            vel += self.rv_keplerian(t, num_planet)

        vel += vals['dvdt']*( t - self.time_base )
        return vel

class RVModelA(GenericRVModel):
    # Five parameters corresponding to the RV of planet
    planet_parameters = 'per tc secosw sesinw logk'.split()
    def rv_keplerian(self, t, num_planet):
        """
        Radial Velocity due to single planet. Handles the change of basis.
        """
        vals = self.params.valuesdict()
        print vals['secosw1'],vals['secosw2'],vals['logk2']
        orbel = [] # P tc sqrtecosom sqrtesinom logk
        for parameter in self.planet_parameters:
            val = vals['{0}{1}'.format(parameter,num_planet)]
            orbel += [ val ] 

        secosw = vals['secosw{}'.format(num_planet)]
        sesinw = vals['sesinw{}'.format(num_planet)]
        ecc = np.sqrt(secosw**2 + sesinw**2)
            
        orbel = np.array(orbel)            
        orbel[4] = 10**orbel[4] # logK -> K
        orbel = np.hstack( [orbel, np.zeros(3)] )

        # Convert to P tc ecosom esinom K
        orbel_tcecos = basis_tcsqrtecos_to_tcecos(orbel)

        # Convert to P tp e om K
        orbel_cps = rvkep.basis_tcecos2cps(orbel_tcecos)
        vel = rvkep.rv_drive(t, orbel_cps, time_base=0 )
        return vel 

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
