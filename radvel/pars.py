import numpy as np
from collections import OrderedDict

from radvel import kepler
from radvel.basis import Basis

texdict = {
    'per': 'P',
    'logper': '\\ln{P}',
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
    'curv': '\\ddot{\\gamma}',
    'gp_amp': '\\eta_{1}',
    'gp_explength': '\\eta_{2}',
    'gp_per': '\\eta_{3}',
    'gp_perlength': '\\eta_{4}',
    'gp_length':'\\eta_{2}',
    'gp_B': 'B',
    'gp_C': 'C',
    'gp_L': 'L',
    'gp_Prot': 'P_{\\rm rot}',
}

class Parameter(object):
    """Object to store attributes of each orbital parameter

    Attributes:
        value (float): value of parameter.
        vary (Bool): True if parameter is allowed to vary in
            MCMC or max likelihood fits, false if fixed
        minbound (float): minimum value the parameter is allowed to take on
        maxbound (float): maximum value the parameter is allowed to take on
        mcmcscale (float): step size to be used for MCMC fitting
        linear (bool): if vary=False and linear=True for gamma parameters then they will be calculated analytically
            using the `trick <http://cadence.caltech.edu/~bfulton/share/Marginalizing_the_likelihood.pdf>`_. derived by Timothy Brandt.
    """

    def __init__(self, value=None, vary=True, minbound=None, maxbound=None, mcmcscale=None, linear=False):
        self.value = value
        self.vary = vary
        self.minbound = minbound
        self.maxbound = maxbound
        self.mcmcscale = mcmcscale
        self.linear = linear

    def _equals(self, other):
        """method to assess the equivalence of two Parameter objects"""
        if isinstance(other, self.__class__):
            return (self.value == other.value) \
                   and (self.vary == other.vary) \
                   and (self.minbound == other.minbound) \
                   and (self.maxbound == other.maxbound) \
                   and (self.mcmcscale == other.mcmcscale)

    def __repr__(self):
        s = (
            "Parameter object: value = {}, vary = {}, minbound = {}, maxbound = {}, mcmc scale = {}"
        ).format(self.value, self.vary, self.minbound, self.maxbound, self.mcmcscale)
        return s

    def __float__(self):
        return self.value

class Parameters(OrderedDict):

    """Object to store the model parameters.

    Parameters to describe a radial velocity orbit
    stored as an OrderedDict.

    Args:
        num_planets (int): Number of planets in model
        basis (string): parameterization of orbital parameters. See
            ``radvel.basis.Basis`` for a list of valid basis strings.
        planet_letters (dict [optional): custom map to match the planet
            numbers in the Parameter object to planet letters.
            Default {1: 'b', 2: 'c', etc.}. The keys of this dictionary must
            all be integers.

    Attributes:
        basis (radvel.Basis): Basis object
        planet_parameters (list): orbital parameters contained within the
            specified basis
        num_planets (int): number of planets in the model

    Examples:
       >>> import radvel
       # create a Parameters object for a 2-planet system with
       # custom planet number to letter mapping
       >>> params = radvel.Parameters(2, planet_letters={1:'d', 2:'e'})

    """
    def __init__(self, num_planets, basis='per tc secosw sesinw logk',
                 planet_letters=None):
        super(Parameters, self).__init__()

        basis = Basis(basis,num_planets)
        self.planet_parameters = basis.name.split()

        for num_planet in range(1,1+num_planets):
            for parameter in self.planet_parameters:
                new_name = self._sparameter(parameter, num_planet)
                self.__setitem__(new_name, Parameter())

        if planet_letters is not None:
            for k in planet_letters.keys():
                assert isinstance(k, int), """\
Parameters: ERROR: The planet_letters dictionary \
should have only integers as keys."""

        self.basis = basis
        self.num_planets = num_planets
        self.planet_letters = planet_letters

    def __reduce__(self):

        red = (self.__class__, (self.num_planets,
                                self.basis.name,
                                self.planet_letters),
                                None,None,iter(self.items()))
        return red

    def tex_labels(self, param_list=None):
        """Map Parameters keys to pretty TeX code representations.

        Args:
            param_list (list [optional]): Manually pass a list of parameter labels

        Returns:
            dict: dictionary mapping Parameters keys to TeX code

        """

        if param_list is None:
            param_list = self.keys()

        tex_labels = {}
        for k in param_list:
            n = k[-1]
            p = k[:-1]
            if n.isdigit() and (not 'gamma' in p and not 'jit' in p):
                tex_labels[k] = self._planet_texlabel(p, n)
            elif k in texdict.keys():
                tex_labels[k] = "$%s$" % texdict[k]
            elif p not in self.planet_parameters:
                for tex in texdict.keys():
                    if tex in k and len(tex) > 1:
                        tex_labels[k] = "$%s}$" % k.replace(tex, texdict[tex])
                        if k.startswith('gp_'):
                            tex_labels[k] = tex_labels[k].replace("}_", ", \\rm ")

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