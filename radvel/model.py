"""
RV Model

Define a class that can do the following things:
"""

from lmfit import Parameters
import rvkep
from matplotlib.pylab import * 
import copy_reg
import types
import numpy as np
class RVParameters(dict):
    def __init__(self, num_planets, basis='per tc secosw sesinw logk'):
        self.basis = basis
        self.planet_parameters = basis.split()
        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                self.__setitem__(self._sparameter(parameter, num_planet), None)

        self.num_planets = num_planets
        
    def _sparameter(self, parameter, num_planet):
        return '{0}{1}'.format(parameter, num_planet)

    def to_cps(self, num_planet):
        def _getpar(key):
            return self['{}{}'.format(key,num_planet)]

        if self.basis=='per tc secosw sesinw logk':
            # pull out parameters
            per = _getpar('per')
            tc = _getpar('tc')
            secosw = _getpar('secosw')
            sesinw = _getpar('sesinw')
            logk = _getpar('logk')
            
            # transform into CPS basis
            k = np.exp(logk)
            ecc = secosw**2 + sesinw**2
            w = np.arctan2( sesinw , secosw ) / pi * 180
            orbel = np.array([per, tc, ecc, w, k, 0, 0, 0])
            return orbel

        if self.basis=='per tc secosw sesinw k':
            # pull out parameters
            per = _getpar('per')
            tc = _getpar('tc')
            secosw = _getpar('secosw')
            sesinw = _getpar('sesinw')
            k = _getpar('k')
            
            # transform into CPS basis
            ecc = secosw**2 + sesinw**2
            w = np.arctan2( sesinw , secosw ) / pi * 180
            orbel = np.array([per, tc, ecc, w, k, 0, 0, 0])
            return orbel

            
class RVModel(object):
    """
    Generic RV Model

    This class defines the methods common to all RV modeling
    classes. The different RV models, having different
    parameterizations inherit from this class.
    """

    def __init__(self, params, time_base=0):
        self.num_planets = params.num_planets
        self.params = params
        self.params['dvdt'] = 0
        self.time_base = time_base

    def __call__(self, t):
        """
        Compute the radial velocity due to all Keplerians and
        additional trends.
        """
        vel = np.zeros(len(t))
        for num_planet in range(1, self.num_planets+1):
            vel += self.rv_keplerian(t, num_planet)
        vel += self.params['dvdt']*( t - self.time_base )
        return vel

    def rv_keplerian(self, t, num_planet):
        """
        Radial Velocity due to single planet. Handles the change of basis.
        """
        orbel_cps = self.params.to_cps(num_planet)
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
