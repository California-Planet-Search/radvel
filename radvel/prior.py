
import warnings

import numpy as np
from scipy.stats import gaussian_kde

import radvel
from radvel import model
from radvel import orbit
from radvel import utils

import pymc3 as pm
import theano.tensor as tt


commands = {
    "Jeffreys": radvel.distributions.Jeffreys,
    "ModifiedJeffreys": radvel.distributions.ModifiedJeffreys,
    "eccpotential": radvel.distributions.eccpotential,
    "kpotential": radvel.distributions.kpotential,
    "secondaryeclipsepotential": radvel.distributions.secondaryeclipsepotential,
    "numericalpotential": radvel.distributions.numericalpotential,
    "userpotential": radvel.distributions.userpotential,
    "informativebaselinepotential": radvel.distributions.informativebaselinepotential,
    "Gaussian": pm.Normal,
    "Uniform": pm.Uniform,
    "Flat": pm.Flat,
    "HalfFlat": pm.HalfFlat,
    "Normal": pm.Normal,
    "TruncatedNormal": pm.TruncatedNormal,
    "HalfNormal": pm.HalfNormal,
    "SkewNormal": pm.SkewNormal,
    "Beta": pm.Beta,
    "Kumaraswamy": pm.Kumaraswamy,
    "Exponential": pm.Exponential,
    "Laplace": pm.Laplace,
    "StudentT": pm.StudentT,
    "HalfStudentT": pm.HalfStudentT,
    "Cauchy": pm.Cauchy,
    "HalfCauchy": pm.HalfCauchy,
    "Gamma": pm.Gamma,
    "InverseGamma": pm.InverseGamma,
    "Weibull": pm.Weibull,
    "Lognromal": pm.Lognormal,
    "ChiSquared": pm.ChiSquared,
    "Wald": pm.Wald,
    "Pareto": pm.Pareto,
    "ExGaussian": pm.ExGaussian,
    "VonMises": pm.VonMises,
    "Triangular": pm.Triangular,
    "Gumbel": pm.Gumbel,
    "Rice": pm.Rice,
    "Logistic": pm.Logistic,
    "LogitNormal": pm.LogitNormal,
    "Interpolated": pm.Interpolated,
    "Potential": pm.Potential,
}


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

    def __repr__(self):
        s = "Gaussian prior on {}, mu={}, sigma={}".format(
            self.param, self.mu, self.sigma
            )
        return s

    def __str__(self):
        try:
            tex = model.Parameters(9).tex_labels(param_list=[self.param])[self.param]

            s = "Gaussian prior on {}: ${} \\pm {}$ \\\\".format(tex, self. mu, self.sigma)
        except KeyError:
            s = self.__repr__()

        return s

    def __call__(self):
        return ['Gaussian', self.param, self.mu, self.sigma]


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
        for i, num_planet in enumerate(self.planet_list):
            msg += "e{} constrained to be < {}\n".format(num_planet, self.upperlims[i])

        return msg[:-1]

    def __str__(self):
        tex = model.Parameters(9, basis='per tc e w k').tex_labels()

        msg = ""
        for i, num_planet in enumerate(self.planet_list):
            par = "e{}".format(num_planet)
            label = tex[par]
            msg += "{} constrained to be $<{}$ \\\\\\\\\n".format(label, self.upperlims[i])

        return msg[:-5]

    def __init__(self, num_planets, upperlims=0.99):

        if type(num_planets) == int:
            self.planet_list = range(1, num_planets+1)
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

    def __call__(self):

        return ['Potential', 'eccpotential', self.planet_list, self.upperlims]


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

    def __call__(self):

        return ['Potential', 'kpotential', self.num_planets]


class SecondaryEclipsePrior(Prior):
    """Secondary eclipse prior

    Implied prior on eccentricity and omega by specifying measured secondary eclipse time

    Args:
        planet_num (int): Number of planet with measured secondary eclipse
        ts (float): Secondary eclipse midpoint time.
            Should be in the same units as the timestamps of your data.
        ts_err (float): Uncertainty on secondary eclipse time
    """

    def __repr__(self):
        msg = ""
        msg += "secondary eclipse constraint: {} +/- {}\n".format(self.ts, self.ts_err)

        return msg[:-1]

    def __str__(self):
        msg = "secondary eclipse prior: ${} \\pm {}$ \\\\\\\\\n".format(self.ts, self.ts_err)

        return msg[:-5]

    def __init__(self, planet_num, ts, ts_err):

        self.planet_num = planet_num
        self.ts = ts
        self.ts_err = ts_err

    def __call__(self, params):

        return ['Potential', 'secondaryeclipse', self.planet_num, self.ts, self.ts_err]


class ModifiedJeffreys(Prior):
    """Modified Jeffry's prior

    This prior follows the distribution:

    .. math::
        p(x) \\propto \\frac{1}{x-x_0}

    with upper bound.

    Args:
        param (string): parameter label
        kneeval (float): "knee" of Jeffrey's prior (:math:`x_0` in eq above)
        minval (float): minimum allowed value. `minval` must be larger than `kneeval`
        maxval (float): maximum allowed value

    """

    def __init__(self, param, minval, maxval, kneeval):
        self.maxval = maxval
        self.param = param
        self.kneeval = kneeval
        self.minval = minval

        self.normalization = 1./np.log((self.maxval-self.kneeval)/(self.minval-self.kneeval))

        assert self.minval > self.kneeval, "ModifiedJeffreys prior requires minval>kneeval."

    def __call__(self, params):

        return ['ModifiedJeffreys', self.param, self.kneeval, self.minval, self.maxval]

    def __repr__(self):
        s = "Modified Jeffrey's prior on {}, knee={}, min={}, max={}".format(
            self.param, self.kneeval, self.minval, self.maxval
            )
        return s

    def __str__(self):
        try:
            tex = model.Parameters(9).tex_labels(param_list=[self.param])[self.param]

            s = "Modified Jeffrey's prior: knee = {}; ${} < {} < {}$".format(
                self.kneeval, self.minval, tex.replace('$',''), self.maxval
                )
        except KeyError:
            s = self.__repr__()

        return s


class NumericalPrior(Prior):
    """Prior defined by an input array of values

    Wrapper for scipy.stats.gaussian_kde.

    This prior uses Gaussian Kernel Density Estimation to
    estimate the probability density function from which
    a set of values are randomly drawn.

    Useful for defining a prior given a posterior obtained
    from a complementary fitting process. For example, you
    might use transit data to obtain constraints on secosw and
    sesinw, then use the posterior on secosw as a prior for
    a RadVel fit.

    Args:
        param_list (list of str): list of parameter label(s).
        values (numpy array of float): values of ``param`` you
            wish to use to define this prior. For example, this
            might be a posterior array of values of secosw
            derived from transit data. In case of univariate data
            this is a 1-D array, otherwise a 2-D array with shape
            (# of elements in param_list, # of data points).
        bw_method (str, scalar, or callable [optional]): see
            scipy.stats.gaussian_kde

    Note: the larger the input array of values, the longer it will
    take for calls to this prior to be evaluated. Consider thinning
    large input arrays to speed up performance.

    """

    def __init__(self, param_list, values, bw_method=None):
        self.param_list = param_list
        self.pdf_estimate = gaussian_kde(values, bw_method=bw_method)

    def __call__(self, params):

        return ['Potential', 'numericalpotential', self.param_list, self.pdf_estimate]

    def __repr__(self):
        s = "Numerical prior on {}".format(
            self.param_list
            )
        return s

    def __str__(self):
        try:
            tex = model.Parameters(9).tex_labels(param_list=self.param_list)
            t=[tex[key] for key in tex.keys()]
            if len(self.param_list) == 1:
                str2print = '{0}'.format(*t)
            elif len(self.param_list) == 2:
                str2print = '{} and {}'.format(*t)
            else:
                str2print = ''
                for el in np.arange(len(self.param_list) - 1):
                    str2print += '{}, '.format(t[el])
                str2print += 'and {}'.format(t[el+1])
            s = "Numerical prior on " + str2print + \
                ", defined using Gaussian kernel density estimation."
        except KeyError:
            s = self.__repr__()

        return s


class UserDefinedPrior(Prior):
    """Interface for user to define a prior
       with an arbitrary functional form.

    Args:
        param_list (list of str): list of parameter label(s).
        func (function): a Python function that takes in  a list
            of values (ordered as in ``param_list``), and returns
            the corresponding log-value of a pdf.
        tex_rep (str): TeX-readable string representation of
            this prior, to be passed into radvel report and
            plotting code.

    Example:
        >>> def myPriorFunc(inp_list):
        ...     if inp_list[0] > 0. and inp_list[0] < 1.:
        ...         return 0.
        ...     else:
        ...         return -np.inf
        >>> myTexString = 'Uniform Prior on $\\sqrt{e}$'
        >>> myPrior = radvel.prior.UserDefinedPrior(['se'], myPriorFunc, myTexString)

    Note:
        ``func`` must be properly normalized; i.e. integrating over the
        entire parameter space must give a probability of 1.
    """

    def __init__(self, param_list, func, tex_rep):
        self.param_list = param_list
        self.func = func
        self.tex_rep = tex_rep

    def __call__(self, params):

        return ['Potential', 'userpotential', self.param_list, self.func]

    def __repr__(self):
        s = "User-defined prior on {}".format(
            self.param_list
            )
        return s

    def __str__(self):
        s = self.tex_rep
        return s


class InformativeBaselinePrior(Prior):
    """ Informative baseline prior acting on the period, suggested
    by A. Vanderburg (see Blunt et al. 2019).

    This prior follows the distribution:

    .. math::
        p(x) \\propto 1\\, \\mathrm{{if}}\\, x-t_{{d}} \\lt B

             \\propto (B+t_{{d}})/x\\, \mathrm{{else}}

    with upper bound.

    Args:
        planet_num (integer): number of planet to apply prior on
        baseline (float): :math:`B` in eq above
        duration (float): :math:`t_{{d}}` in eq above (default: 0.0)

    """

    def __init__(self, planet_num, baseline, duration=0.0):
        self.planet_num = planet_num
        self.baseline = baseline
        self.duration = duration

    def __repr__(self):
        s = "Informative baseline prior on per{}, baseline={}, duration={}".format(
            self.planet_num, self.baseline, self.duration
            )
        return s

    def __call__(self, params):

        return ['Potential', "informativebaselinepotential", self.planet_num, self.baseline, self.duration]


class PMDistribution(Prior):
    """Interface for the user to define a prior using
    any continuous distribution supported by PyMC3.

    Args:
        name (string): name of the distribution
        param (string): parameter to place the prior on
        **kargs: all required arguments for that distribution

    """

    def __init__(self, name, param, **kwargs):

        self.name = name
        self.param = param
        self.kwargs = kwargs

    def __call__(self):

        return("PMDistribution", self.param, self.name, self.kwargs)

    def __repr__(self):

        s = "{} prior on {} with arguments {}".format(
            self.name, self.param, self.kwargs
        )