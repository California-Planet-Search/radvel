import numpy as np
from radvel import model

class Prior(object):
    def __repr__(self):
        return "Generic Prior"

class Gaussian(Prior):
    """Gaussian prior

    Guassian prior on a given parameter.

    Args:
        param (string): parameter label
        mu (float): center of Gaussian prior
        sigma (float): width of Gaussian prior
    """
    
    def __init__(self, param, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.param = param
    def __call__(self, params):
        x = params[self.param].value
        return -0.5 * ( ( x - self.mu ) / self.sigma )**2
    def __repr__(self):
        s = "Gaussian prior on {}, mu={}, sigma={}".format(
            self.param, self. mu, self.sigma
            )
        return s
    def __str__(self):
        try:
            d = {self.param: self.mu}
            tex = model.Parameters(9).tex_labels(param_list=[self.param])[self.param]
            
            s = "Gaussian prior on {}: ${} \\pm {}$ \\\\".format(tex, self. mu, self.sigma)
        except KeyError:
            s = self.__repr__()
            
        return s

class EccentricityPrior(Prior):
    """Physical eccentricities

    Prior to keep eccentricity between 0 and a specified upper limit.

    Args:
        num_planets (int or list): Planets to apply the eccentricity prior.
            If an integer is given then all planets with indexes up to and including
            the specified integer will be included in the prior. If a list is given then
            the prior will only be applied to the specified planets.
        upperlims (float or list of floats): List of eccentricity upper limits to assign
            to each of the planets. If a float is given then all planets must have
            eccentricities less then this value. If a list of floats is given then
            each planet can have a different eccentricity upper limit.
    """

    
    def __repr__(self):
        msg = ""
        for i,num_planet in enumerate(self.planet_list):
            msg += "e{} constrained to be < {}\n".format(num_planet, self.upperlims[i])

        return msg[:-1]
    
    def __str__(self):
        tex = model.Parameters(9, basis='per tc e w k').tex_labels()

        msg = ""
        for i,num_planet in enumerate(self.planet_list):
            par = "e{}".format(num_planet)
            label = tex[par]
            msg += "{} constrained to be $<{}$ \\\\\\\\\n".format(label, self.upperlims[i])

        return msg[:-5]
        
    def __init__(self, num_planets, upperlims=0.99):

        if type(num_planets) == int:
            self.planet_list = range(1,num_planets+1)
            npl = len(self.planet_list)
        else:
            self.planet_list = num_planets
            npl = num_planets
            
        if type(upperlims) == float:
            self.upperlims = [upperlims] * npl
        else:
            assert len(upperlims) == len(self.planet_list), "Number of eccentricity \
upper limits must match number of planets."
            self.upperlims = upperlims
    
    def __call__(self, params):
        def _getpar(key, num_planet):
            return params['{}{}'.format(key,num_planet)].value

        parnames = params.basis.name.split()
        
        for i,num_planet in enumerate(self.planet_list):
            if 'e' in parnames:
                ecc = _getpar('e', num_planet)
            elif 'secosw' in parnames:
                secosw = _getpar('secosw',num_planet)
                sesinw = _getpar('sesinw',num_planet)
                ecc = secosw**2 + sesinw**2 
            elif 'ecosw' in parnames:
                ecosw = _getpar('ecosw',num_planet)
                esinw = _getpar('esinw',num_planet)
                ecc = np.sqrt(ecosw**2 + esinw**2)

                
            if ecc > self.upperlims[i] or ecc < 0.0:
                return -np.inf
        
        return 0
        
class PositiveKPrior(Prior):
    """K must be positive

    A prior to prevent K going negative.
    Be careful with this as it can introduce a bias to larger K
    values.

    Args:
        num_planets (int): Number of planets. Used to ensure K for each
            planet is positive
    """
    
    def __repr__(self):
        return "K constrained to be > 0"
    def __str__(self):
        return "$K$ constrained to be $>0$"

    def __init__(self, num_planets):
        self.num_planets = num_planets
    
    def __call__(self, params):
        def _getpar(key, num_planet):
            return params['{}{}'.format(key,num_planet)].value

        for num_planet in range(1,self.num_planets+1):
            try:
                k = _getpar('k', num_planet)
            except KeyError:
                k = np.exp(_getpar('logk',num_planet))

            if k < 0.0:    
                return -np.inf
        return 0

class HardBounds(Prior):
    """Prior for hard boundaries

    This prior allows for hard boundaries to be established
    for a given parameter.

    Args:
        param (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    """
    
    def __init__(self, param, minval, maxval):
        self.minval = minval
        self.maxval = maxval
        self.param = param
    def __call__(self, params):
        x = params[self.param].value
        if x < self.minval or x > self.maxval:
            return -np.inf
        else:
            return 0.0
    def __repr__(self):
        s = "Bounded prior on {}, min={}, max={}".format(
            self.param, self.minval, self.maxval
            )
        return s
    def __str__(self):
        try:
            d = {self.param: self.minval}
            tex = model.Parameters(9).tex_labels(param_list=[self.param])[self.param]
            
            s = "Bounded prior: ${} < {} < {}$".format(self.minval,
                                                       tex.replace('$',''),
                                                       self.maxval)
        except KeyError:
            s = self.__repr__()
            
        return s
