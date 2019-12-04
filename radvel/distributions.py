import numpy as np
import pymc3 as pm
import theano.tensor as tt
import radvel


class Jeffreys(pm.distributions.continuous.BoundedContinuous):

    def __init__(self, lower, upper, **kwargs):
        self.lower = lower
        self.upper = upper
        self.mean = (upper + lower)/2
        self.median = self.mean

        kwargs["testval"] = kwargs.pop("testval", self.mean)
        super(Jeffreys, self).__init__(**kwargs)

    def logp(self, value):

        lower = self.lower
        upper = self.upper

        normalization = 1. / pm.math.log(upper / lower)

        if tt.gt(lower, value) or tt.gt(value, upper):
            return -np.inf
        else:
            return tt.log(normalization) - tt.log(value)

class ModifiedJeffreys(pm.distributions.continuous.BoundedContinuous):

    def __init__(self, kneeval, lower, upper, **kwargs):
        self.minval = lower
        self.maxval = upper
        self.kneeval = kneeval
        self.mean = (upper + lower)/2
        self.median = self.mean

        assert self.minval > self.kneeval, "ModifiedJeffreys prior requires minval>kneeval."

        kwargs["testval"] = kwargs.pop("testval", self.mean)
        super(ModifiedJeffreys, self).__init__(**kwargs)

    def logp(self, value):

        normalization = 1. / np.log((self.maxval - self.kneeval) / (self.minval - self.kneeval))

        if tt.gt(value, self.maxval) or tt.gt(self.minval, value):
            return -np.inf
        else:
            return np.log(normalization) - np.log(value - self.kneeval)


def eccpotential(planet_list, upperlims):

    for i, num_planet in enumerate(planet_list):
        for param in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
            if str(param)[0:3] == 'ecc' and int(str(param)[-1]) == num_planet:
                value = param
            if tt.gt(value, upperlims[i]) or tt.gt(0, value):
                return -np.inf

    return 0


def kpotential(num_planets):

    for num_planet in range(1, num_planets + 1):
        for param in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
            if str(param)[0] == 'k' and int(str(param)[-1]) == num_planet:
                value = param
            if tt.gt(0, value):
                return -np.inf

    return 0


def secondaryeclipsepotential(planet_num, ts_self, ts_err):

    for param in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
        if str(param)[0:2] == 'tp' and int(str(param)[-1]) == planet_num:
            tp = param
        if str(param)[0:3] == 'per' and int(str(param)[-1]) == planet_num:
            per = param
        if str(param)[0:3] == 'ecc' and int(str(param)[-1]) == planet_num:
            ecc = param
        if str(param)[0] == 'w' and int(str(param)[-1]) == planet_num:
            omega = param

    def t_to_phase(tp, per, t):
        phase = (((t - tp) / per) - pm.math.floor((t - tp) / per))
        return phase

    ts = radvel.orbit.timeperi_to_timetrans(tp, per, ecc, omega, secondary=True)
    ts_phase = t_to_phase(tp, per, ts)

    pts = t_to_phase(tp, per, ts_self)
    epts = ts_err / per

    penalty = -0.5 * ((ts_phase - pts) / epts) ** 2 - 0.5 * pm.math.log((epts ** 2) * 2. * np.pi)

    return penalty


def numericalpotential(param_list, pdf_estimate):

    x = []
    for param in param_list:
        for par in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
            if str(par) == param:
                x.append(par)

    val = np.log(pdf_estimate(x))

    return val[0]


def userpotential(param_list, func):

    x = []
    for param in param_list:
        for par in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
            if str(par) == param:
                x.append(par)

    return func(x)


def informativebaselinepotential(planet_num, baseline, duration):

    for param in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
        if int(str(param)[-1]) == planet_num:
            per = param

    if pm.math.gt(baseline, (per - duration)):
        return 0
    else:
        return pm.math.log((baseline + duration) / per)

