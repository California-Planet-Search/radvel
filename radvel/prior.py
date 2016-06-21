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
        x = params[self.param]
        return -0.5 * ( ( x - self.mu ) / self.sigma )**2
    def __repr__(self):
        s = "Gaussian prior on {}, mu={}, sigma={}".format(
            self.param, self. mu, self.sigma
            )
        return s
    def __str__(self):
        try:
            d = {self.param: self.mu}
            tex = model.RVParameters(9).tex_labels(param_list=[self.param])[self.param]
            
            s = "Gaussian prior on {}: ${} \\pm {}$".format(tex, self. mu, self.sigma)
        except KeyError:
            s = self.__repr__()
            
        return s

class EccentricityPrior(Prior):
    """Physical eccentricities

    Prior to keep eccentricity between 0 and 1.

    Args:
        num_planets (int): Number of planets. Used to ensure e for each
            planet is a phsyical value.
    """

    
    def __repr__(self):
        return "Eccentricity constrained to be < 0.99"
    def __str__(self):
        return "Eccentricity constrained to be $<0.99$"

    
    def __init__(self, num_planets):
        self.num_planets = num_planets
    
    def __call__(self, params):
        def _getpar(key, num_planet):
            return params['{}{}'.format(key,num_planet)]

        for num_planet in range(1,self.num_planets+1):
            try:
                ecc = _getpar('e', num_planet)
            except KeyError:
                secosw = _getpar('secosw',num_planet)
                sesinw = _getpar('sesinw',num_planet)
                ecc = secosw**2 + sesinw**2 

            if ecc > 0.99 or ecc < 0.0:
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
            return params['{}{}'.format(key,num_planet)]

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
        x = params[self.param]
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
            tex = model.RVParameters(9).tex_labels(param_list=[self.param])[self.param]
            
            s = "Bounded prior: ${} < {} < {}$".format(self.minval, tex.replace('$',''), self.maxval)
        except KeyError:
            s = self.__repr__()
            
        return s
