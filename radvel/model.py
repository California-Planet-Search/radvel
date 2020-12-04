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
        mcmcscale (float): step size to be used for MCMC fitting
        linear (bool): if vary=False and linear=True for gamma parameters then they will be calculated analytically
            using the `trick <http://cadence.caltech.edu/~bfulton/share/Marginalizing_the_likelihood.pdf>`_. derived by Timothy Brandt.
    """

    def __init__(self, value=None, vary=True, mcmcscale=None, linear=False):
        self.value = value
        self.vary = vary
        self.mcmcscale = mcmcscale
        self.linear = linear

    def _equals(self, other):
        """method to assess the equivalence of two Parameter objects"""
        if isinstance(other, self.__class__):
            return (self.value == other.value) \
                   and (self.vary == other.vary) \
                   and (self.mcmcscale == other.mcmcscale)

    def __repr__(self):
        s = (
            "Parameter object: value = {}, vary = {}, mcmc scale = {}"
        ).format(self.value, self.vary, self.mcmcscale)
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


if __name__ == "__main__":
    a = Parameter(value=1.3)
    a.mcmcscale = 100.
    print(a)


class Vector(object):

    def __init__(self, params):

        self.params = params

        self.init_index_dict()
        self.dict_to_vector()
        self.vector_names()

    def init_index_dict(self):
        indices = dict()
        n = 0
        for k in self.params.keys():
            if k.startswith('gamma') or k.startswith('jit') or k.startswith('logjit') or k.startswith('gp'):
                indices.update({k:2 + n + (5*self.params.num_planets)})
                n += 1
        for num_planet in range(1, self.params.num_planets+1):
            indices.update({'per'+str(num_planet):-5+(5*num_planet),'logper'+str(num_planet):-5+(5*num_planet),
                         'tc'+str(num_planet):-4+(5*num_planet),'tp'+str(num_planet):-4+(5*num_planet),
                         'secosw'+str(num_planet):-3+(5*num_planet),'ecosw'+str(num_planet):-3+(5*num_planet),
                         'e'+str(num_planet):-3+(5*num_planet),'se'+str(num_planet):-3+(5*num_planet),
                         'sesinw'+str(num_planet):-2+(5*num_planet),'esinw'+str(num_planet):-2+(5*num_planet),
                         'w'+str(num_planet):-2+(5*num_planet),'k'+str(num_planet):-1+(5*num_planet),
                         'logk'+str(num_planet):-1+(5*num_planet)})
        indices.update({'dvdt':(5*self.params.num_planets),'curv':1+(5*self.params.num_planets)})
        self.indices = indices

    def dict_to_vector(self):
        n = 0
        if 'dvdt' not in self.params.keys():
            n += 1
        if 'curv' not in self.params.keys():
            n += 1
        vector = np.zeros((len(self.params.keys())+n,4))
        for key in self.params.keys():
            try:
                vector[self.indices[key]][0] = self.params[key].value
                vector[self.indices[key]][1] = self.params[key].vary
                if self.params[key].mcmcscale == None:
                    vector[self.indices[key]][2] = 0
                else:
                    vector[self.indices[key]][2] = self.params[key].mcmcscale
                vector[self.indices[key]][3] = self.params[key].linear
            except KeyError:
                pass
        self.vector = vector

    def vector_names(self):
        names = [0] * (len(self.params.keys()) + 2)
        for key in self.params.keys():
            try:
                names[self.indices[key]] = key
            except KeyError:
                pass
        self.names = [i for i in names if type(i) == str]

    def vector_to_dict(self):
        for key in self.params.keys():
            try:
                self.params[key].value = self.vector[self.indices[key]][0]
            except KeyError:
                pass


class GeneralRVModel(object):
    """
    A generalized Model

    Args:
        params (radvel.Parameters): The parameters upon which the RV model depends.
        forward_model (callable): 
            The function that defines the signal as a function of time and parameters.
            The forward model is called as
            
                ``forward_model(time, params, *args, **kwargs) -> float``
        time_base (float): time relative to which 'dvdt' and 'curv' terms are computed.
    Examples:
        >>> import radvel
        #  In this example, we'll assume a function called 'my_rv_function' that
        #  computes RV values has been defined elsewhere. We'll assume that 
        #  'my_rv_function' depends on planets' usual RV parameters
        #  contained in radvel.Parameters as well as some additional
        #  parameter, 'my_param'.
        >>> params = radvel.Parameters(2)
        >>> params['my_param'] = rv.Parameter(my_param_value,vary=True)
        >>> rvmodel = radvel.GeneralRVModel(myparams,my_rv_function)
        >>> rv = rvmodel(10)
    """
    def __init__(self,params,forward_model,time_base=0):
        self.params = params
        self.vector = Vector(self.params)
        self.time_base = time_base
        self._forward_model = forward_model
        assert callable(forward_model)
    def __call__(self,t,*args,**kwargs):
        """Compute the signal

        Args:
            t (array of floats): Timestamps to calculate the model

        Returns:
            vel (array of floats): model at each time in `t`
        """
        vel = self._forward_model(t,self.params,self.vector,*args,**kwargs)
        vel += self.vector.vector[self.vector.indices['dvdt']][0] * (t - self.time_base)
        vel += self.vector.vector[self.vector.indices['curv']][0] * (t - self.time_base)**2
        return vel

    def array_to_params(self,param_values):
    
    	new_params = self.params
    	
    	vary_parameters = self.list_vary_params()
    	
    	for i in range(len(vary_parameters)):
    		new_params[vary_parameters[i]] = Parameter(value=param_values[i])
    		
    	return new_params
             
    def list_vary_params(self):
        keys = self.list_params()

        return [key for key in keys if self.params[key].vary]

    def list_params(self):
        try:
            keys = self.params_order
        except AttributeError:
            keys = list(self.params.keys())
            self.params_order = keys
        return keys

        
def _standard_rv_calc(t,params,vector,planet_num=None):
        vel = np.zeros(len(t))
        params_synth = params.basis.v_to_synth(vector)
        if planet_num is None:
            planets = range(1, params.num_planets+1)
        else:
            planets = [planet_num]

        for num_planet in planets:
            #index values
            #per: -5 + (5*num_planet)
            #tp: -4 + (5*num_planet)
            #e: -3 + (5*num_planet)
            #w: -2 + (5*num_planet)
            #k: -1 + (5*num_planet)
            per = params_synth[-5+(5*num_planet)][0]
            tp = params_synth[-4+(5*num_planet)][0]
            e = params_synth[-3+(5*num_planet)][0]
            w = params_synth[-2+(5*num_planet)][0]
            k = params_synth[-1+(5*num_planet)][0]
            orbel_synth = np.array([per, tp, e, w, k])
            vel += kepler.rv_drive(t, orbel_synth)
        return vel


class RVModel(GeneralRVModel):
    """
    Generic RV Model

    This class defines the methods common to all RV modeling
    classes. The different RV models, with different
    parameterizations, all inherit from this class.
    """
    def __init__(self, params, time_base=0):
        super(RVModel,self).__init__(params,_standard_rv_calc,time_base)
        self.num_planets=params.num_planets
