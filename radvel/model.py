import rvkep
import copy_reg
import types
import numpy as np
from .basis import Basis

texdict = {'per': 'P',
           'logk': '\\log{K}',
           'tc': 'Tconj',
           'secosw': '\\sqrt{e}\\cos{\\omega}',
           'sesinw': '\\sqrt{e}\\cos{\\omega}'}

class RVParameters(dict):
    """
    Object to store the orbital parameters.

    :param num_planets: Number of planets in model
    :type num_planets: int

    :param basis: parameterization of orbital parameters. See radvel.Basis._print_valid_basis() for a list of valid basis strings.
    :type basis: str

    :param tex_labels: Dictionary maping RVParameters keys to their TeX code representations
    :type tex_labels: dict
    
    :Example:
    
    .. doctest::
	
       >>> import radvel
       >>> params = radvel.RVParameters(2)

    """
    def __init__(self, num_planets, basis='per tc secosw sesinw logk'):
        self.basis = Basis(basis,num_planets)
        self.planet_parameters = basis.split()
        self.tex_labels = {}
        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                self.__setitem__(self._sparameter(parameter, num_planet), None)
                self.tex_labels.__setitem__(self._sparameter(parameter, num_planet), self._texlabel(parameter, num_planet))
                
        self.num_planets = num_planets
        
    def _sparameter(self, parameter, num_planet):
        return '{0}{1}'.format(parameter, num_planet)

    def _texlabel(self, parameter, num_planet):
        pname = texdict.get(parameter, parameter)
        lett_planet = chr(int(num_planet)+97)
        return '$%s_{%s}$' % (pname, lett_planet) 

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
        self.params['curv'] = 0
        self.time_base = time_base

    def __call__(self, t, planet_num=None):
        """
        Compute the radial velocity due to all Keplerians and
        additional trends.

        :param t: Timestamps to calculate the RV model
        :type t: float array

        :param planet_num: (optional) calculate the RV model for a single planet within a multi-planet system
        :type planet_num: int

        :return vel: Radial velocity at each timestamp in the t input array
        :type vel: float array
        
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
