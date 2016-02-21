"""
RV Model

Define a class that can do the following things:
"""

import rvkep
from matplotlib.pylab import * 
import copy_reg
import types
import numpy as np
from .basis import Basis

class RVParameters(dict):
    """
    Object to store the physical parameters of the transit.

    :param num_planets: Number of planets in model
    :type num_planets: int

    :param basis: parameterization of orbital parameters    
    :type basis: str

    .. doctest::
	
       >>> import radvel
       >>> params = radvel.RVParameters(2)

    """
    def __init__(self, num_planets, basis='per tc secosw sesinw logk'):
        self.basis = Basis(basis,num_planets)
        self.planet_parameters = basis.split()
        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                self.__setitem__(self._sparameter(parameter, num_planet), None)

        self.num_planets = num_planets
        
    def _sparameter(self, parameter, num_planet):
        return '{0}{1}'.format(parameter, num_planet)

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

    def __call__(self, t, planet_num=None):
        """
        Compute the radial velocity due to all Keplerians and
        additional trends.
        """
        vel = np.zeros(len(t))

        params_cps = self.params.basis.to_cps(self.params)

        if planet_num == None:
            planets = range(1, self.num_planets+1)
        else:
            planets = [planet_num]
        
        for num_planet in planets:
            per = params_cps['per{}'.format(num_planet)]
            tp = params_cps['tp{}'.format(num_planet)]
            e = params_cps['e{}'.format(num_planet)]
            w = params_cps['w{}'.format(num_planet)]
            k = params_cps['k{}'.format(num_planet)]
            orbel_cps = np.array([per, tp, e, w, k, 0, 0, 0])
            vel += rvkep.rv_drive(t, orbel_cps, time_base=0 )

        vel += self.params['dvdt']*( t - self.time_base ) + self.params['curv']*( t - self.time_base)**2
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
