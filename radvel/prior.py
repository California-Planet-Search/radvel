import numpy as np
from radvel import model

class Prior(object):
    def __repr__(self):
        return "Generic Prior"

class Gaussian(Prior):
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
            tex = model.RVParameters(9).tex_labels()[self.param]
        
            s = "Gaussian prior on {}: ${} \\pm {}$".format(tex, self. mu, self.sigma)
        except KeyError:
            s = self.__repr__(self)
            
        return s

class EccentricityPrior(Prior):
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
                return -1e25
        
        return 0
        
class PositiveKPrior(Prior):
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

            if k < 0.0:     # This was ecc > 0.99. We should probably allow for eccentricities as high as 0.99
                return -np.inf
        return 0
