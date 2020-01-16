import numpy as np
import pymc3 as pm
import theano.tensor as tt
import radvel


class Jeffreys(pm.distributions.continuous.BoundedContinuous):

    def __init__(self, lower=.1, upper=1, *args, **kwargs):

        self.lower = lower = tt.as_tensor_variable(lower)
        self.upper = upper = tt.as_tensor_variable(upper)
        self.mean = (upper + lower) / 2.
        self.median = self.mean

        kwargs["testval"] = kwargs.pop("testval", self.mean)
        super().__init__(lower=lower, upper=upper, *args, **kwargs)

    def logp(self, value):

        lower = self.lower
        upper = self.upper

        normalization = 1. / pm.math.log(upper / lower)

        return tt.log(normalization) - tt.log(value)

class ModifiedJeffreys(pm.distributions.continuous.BoundedContinuous):

    def __init__(self, kneeval, lower, upper, **kwargs):

        assert lower > kneeval, "ModifiedJeffreys prior requires minval>kneeval."

        self.lower = tt.as_tensor_variable(lower)
        self.upper = tt.as_tensor_variable(upper)
        self.kneeval = tt.as_tensor_variable(kneeval)
        self.mean = (upper + lower)/2
        self.median = self.mean

        kwargs["testval"] = kwargs.pop("testval", self.mean)
        super().__init__(kneeval=kneeval, lower=lower, upper=upper, **kwargs)

    def logp(self, value):

        normalization = 1. / np.log((self.maxval - self.kneeval) / (self.minval - self.kneeval))

        return tt.log(normalization) - tt.log(value - self.kneeval)


def eccpotential(model, planet_list, upperlims):
    lp = 0
    ecount = 0

    if planet_list != None:
        for i, num_planet in enumerate(planet_list):
            for par in model.unobserved_RVs:
                if str(par).startswith('ecc') and str(par).endswith(str(num_planet)):
                    ecc = par
                    ecount += 1

            if ecount == 0:
                ecc = globals()['ecc' + str(num_planet)]

            lp += tt.switch(tt.gt(ecc, 0), 0, -10e10) + tt.switch(tt.gt(ecc, upperlims[i]), -10e10, 0)

    else:
        for par in (model.unobserved_RVs):
            if str(par.startswith('ecc')):
                ecc = par
                ecount += 1

        if ecount == 0:
            ecc = globals()['ecc']

        lp += tt.switch(tt.gt(ecc, 0), 0, -10e10)
        lp += tt.switch(tt.gt(ecc, upperlims), -10e10, 0)

    pm.Potential('eccpotential', lp, model=model)


def kpotential(model, num_planets):
    lp = 0
    kcount = 0

    if num_planets != None:
        for i in range(1, num_planets + 1):
            for par in (model.unobserved_RVs):
                if str(par).startswith('k') and str(par).endswith(str(i)):
                    k = par
                    kcount += 1

            if kcount == 0:
                k = globals()['k' + str(i)]

            lp += tt.switch(tt.gt(k, 0), 0, -10e10)

    else:
        for par in (model.unobserved_RVs):
            if str(par).startswith('k'):
                k = par
                kcount += 1

        if kcount == 0:
            k = globals()['k']

        lp += tt.switch(tt.gt(k, 0), 0, -10e10)

    pm.Potential('kpotential', lp, model=model)


def secondaryeclipsepotential(model, planet_num, ts_self, ts_err):

    tcount = 0
    pcount = 0
    ecount = 0
    wcount = 0

    if planet_num != None:

        for par in (model.unobserved_RVs):
            if str(par).startswith('tp') and str(par).endswith(str(planet_num)):
                tp = par
                tcount += 1
            if str(par).startswith('per') and str(par).endswith(str(planet_num)):
                per = par
                pcount += 1
            if str(par).startswith('ecc') and str(par).endswith(str(planet_num)):
                ecc = par
                ecount += 1
            if str(par).startswith('w') and str(par).endswith(str(planet_num)):
                omega = par
                wcount += 1

        if ecount == 0:
            ecc = globals()['ecc' + str(planet_num)]
        if pcount == 0:
            per = globals()['per' + str(planet_num)]
        if wcount == 0:
            omega = globals()['w' + str(planet_num)]
        if tcount == 0:
            tp = globals()['tp' + str(planet_num)]

    else:
        for par in (model.unobserved_RVs):
            if str(par).startswith('tp'):
                tp = par
                tcount += 1
            if str(par).startswith('per'):
                per = par
                pcount += 1
            if str(par).startswith('ecc'):
                ecc = par
                ecount += 1
            if str(par).startswith('w'):
                omega = par
                wcount += 1

        if ecount == 0:
            ecc = globals()['ecc']
        if pcount == 0:
            per = globals()['per']
        if wcount == 0:
            omega = globals()['w']
        if tcount == 0:
            tp = globals()['tp']

    def t_to_phase(tp, per, t):
        phase = (((t - tp) / per) - tt.floor((t - tp) / per))
        return phase

    ts = radvel.orbit.timeperi_to_timetrans(tp, per, ecc, omega, secondary=True, eval=False)
    ts_phase = t_to_phase(tp, per, ts)

    pts = t_to_phase(tp, per, ts_self)
    epts = ts_err / per

    penalty = -0.5 * ((ts_phase - pts) / epts) ** 2 - 0.5 * pm.math.log((epts ** 2) * 2. * np.pi)

    return tt.as_tensor_variable(penalty)


def numericalpotential(param_list, pdf_estimate):

    x = []
    for param in param_list:
        for par in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
            if str(par) == param:
                x.append(par)

    val = np.log(pdf_estimate(x))

    return tt.as_tensor_variable(val[0])


def userpotential(param_list, func):

    x = []
    for param in param_list:
        for par in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
            if str(par) == param:
                x.append(par)

    return tt.as_tensor_variable(func(x))


def informativebaselinepotential(planet_num, baseline, duration):

    for param in (radvel.mcmc.model.free_RVs + radvel.mcmc.model.deterministics):
        if int(str(param)[-1]) == planet_num:
            per = param

    if pm.math.gt(baseline, (per - duration)):
        return 0
    else:
        return pm.math.log((baseline + duration) / per)

