import numpy as np
import copy_reg
import types
from collections import OrderedDict
import lmfit

from . import kepler
from .basis import Basis

texdict = {
    'per': 'P',
    'logk': '\\ln{K}',
    'tc': 'T\\rm{conj}',
    'secosw': '\\sqrt{e}\\cos{\\omega}',
    'sesinw': '\\sqrt{e}\\sin{\\omega}',
    'ecosw': 'e\\cos{\\omega}',
    'esinw': 'e\\sin{\\omega}',
    'e': 'e',
    'w': '\\omega',
    'tp': 'T\\rm{peri}',
    'k': 'K',
    'gamma_': '\\gamma_{\\rm ',
    'logjit_': '\\ln{\\sigma_{\\rm jit}}_{\\rm ',
    'jit_': '\\sigma_{\\rm ',
    'dvdt': '\\dot{\\gamma}',
    'curv': '\\ddot{\\gamma}'
}

class RVParameters(OrderedDict):
    """Object to store the orbital parameters.

    Parameters to describe a radial velocity orbit
    stored as an OrderedDict

    Args:
        num_planets (int): Number of planets in model
        basis (string): parameterization of orbital parameters. See 
            ``radvel.basis.Basis`` for a list of valid basis strings.
        planet_letters (Dictionary[optional): custom map to match the planet 
            numbers in the RVParameter object to planet letters.
            Default {1: 'b', 2: 'c', etc.}. The keys of this dictionary must 
            all be integers.

    Attributes:
        basis (radvel.Basis): Basis object
        planet_parameters (list): orbital parameters contained within the 
            specified basis
        num_planets (int): number of planets in the model
    
    Examples:
       >>> import radvel
       # create a RVParameters object for a 2-planet system with
       # custom planet number to letter mapping
       >>> params = radvel.RVParameters(2, planet_letters={1:'d', 2:'e'})

    """
    def __init__(self, num_planets, basis='per tc secosw sesinw logk', 
                 planet_letters=None):
        super(RVParameters, self).__init__()
        
        basis = Basis(basis,num_planets)
        self.planet_parameters = basis.name.split()

        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                self.__setitem__(self._sparameter(parameter, num_planet), None)

                
        if planet_letters is not None:
            for k in planet_letters.keys():
                assert isinstance(k, int), """\
RVParameters: ERROR: The planet_letters dictionary \
should have only integers as keys."""

        self.basis = basis
        self.num_planets = num_planets
        self.planet_letters = planet_letters
        #self.__setitem__('meta', meta)

    def __reduce__(self):

        red = (self.__class__, (self.num_planets,
                                self.basis.name,
                                self.planet_letters),
                                None,None,self.iteritems())
        return red

    def tex_labels(self, param_list=None):
        """Map RVParameters keys to pretty TeX code representations.

        Args:
            param_list (list): (optional) Manually pass a list of parameter labels
        
        Returns:
            dict: dictionary mapping RVParameters keys to TeX code

        """

        if param_list is None:
            param_list = self.keys()
        
        tex_labels = {}
        for k in param_list:
            n = k[-1]
            p = k[:-1]
            if n.isdigit(): tex_labels[k] = self._planet_texlabel(p, n)
            elif k in texdict.keys(): tex_labels[k] = "$%s$" % texdict[k]
            elif p not in self.planet_parameters:
                for tex in texdict.keys():
                    if tex in k and len(tex) > 1:
                        tex_labels[k] = "$%s}$" % k.replace(tex, texdict[tex])
                        
            if k not in tex_labels.keys():
                tex_labels[k] = k

        return tex_labels
        
    def _sparameter(self, parameter, num_planet):
        return '{0}{1}'.format(parameter, num_planet)

    def _planet_texlabel(self, parameter, num_planet):        
        pname = texdict.get(parameter, parameter)
        if self.planet_letters is not None:
            lett_planet = self.planet_letters[int(num_planet)]
        else:
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
        """Compute the radial velocity
        
        Includes all Keplerians and additional trends.

        Args:
            t (array of floats): Timestamps to calculate the RV model
            planet_num (Optional[int]): calculate the RV model for a single 
                planet within a multi-planet system

        Returns:
            vel (array of floats): Radial velocity at each time in `t`
        """
        vel = np.zeros( len(t) )
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
            orbel_cps = np.array([per, tp, e, w, k])
            vel+=kepler.rv_drive(t, orbel_cps)

        vel+=self.params['dvdt'] * ( t - self.time_base )
        vel+=self.params['curv'] * ( t - self.time_base )**2
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
