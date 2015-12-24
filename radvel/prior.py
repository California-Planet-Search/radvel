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
        s = "Gaussian Prior on {}, mu={}, sigma={}".format(
            self.param, self. mu, self.sigma
            )
        return s 

class EccentricityPrior(Prior):
    def __init__(self, num_planets):
        self.num_planets = num_planets
    
    def __call__(self, params):
        def _getpar(key, num_planet):
            return params['{}{}'.format(key,num_planet)]

        for num_planet in range(1,self.num_planets+1):
            secosw = _getpar('secosw',num_planet)
            sesinw = _getpar('sesinw',num_planet)
            ecc = secosw**2 + sesinw**2 
            if ecc > .99:
                return -1e25
        return 0
        
