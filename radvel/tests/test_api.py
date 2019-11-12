import sys
import copy
import warnings

import radvel
import radvel.driver
import numpy as np
import scipy
import radvel.prior

warnings.simplefilter('ignore')

class _args(object):
    def __init__(self):
        self.outputdir = '/tmp/'
        self.decorr = False
        self.name_in_title = False
        self.gp = False
        self.simple = False

        self.nwalkers = 50
        self.nsteps = 100
        self.ensembles = 8
        self.maxGR = 1.10
        self.burnGR = 1.30
        self.burnAfactor = 15
        self.minAfactor = 75
        self.maxArchange = .01
        self.minTz = 1000
        self.minsteps = 100
        self.minpercent = 5
        self.thin = 1
        self.serial = False
        self.save = True
        self.savename = 'rawchains.h5'
        self.proceed = False
        self.proceedname = None

class _args2(object):
    def __init__(self):
        self.outputdir = '/tmp/'
        self.decorr = False
        self.name_in_title = False
        self.gp = False
        self.simple = False

        self.nwalkers = 50
        self.nsteps = 100
        self.ensembles = 8
        self.maxGR = 1.10
        self.burnGR = 1.30
        self.burnAfactor = 15
        self.minAfactor = 75
        self.maxArchange = .01
        self.minTz = 1000
        self.minsteps = 100
        self.minpercent = 5
        self.thin = 1
        self.serial = False
        self.save = True
        self.savename = 'rawchains.h5'
        self.proceed = True
        self.proceedname = 'rawchains.h5'

def _standard_run(setupfn, arguments):
    """
    Run through all of the standard steps
    """

    args = arguments
    args.setupfn = setupfn
    radvel.driver.fit(args)

    args.serial = True
    args.nensembles = 1
    radvel.driver.mcmc(args)
    
    radvel.driver.derive(args)

    args.type = ['trend', 'jit', 'e', 'nplanets', 'gp']
    args.verbose = True
    radvel.driver.ic_compare(args)

    args.type = ['params', 'priors', 'rv', 'ic_compare', 'derived', 'crit']
    radvel.driver.tables(args)

    args.type = ['rv', 'corner', 'auto', 'trend', 'derived']
    args.plotkw = {'highlight_last': True, 'show_rms': True}
    radvel.driver.plots(args)

    args.comptype = 'ic_compare'
    args.latex_compiler = 'pdflatex'
    radvel.driver.report(args)

def test_k2(setupfn='example_planets/epic203771098.py'):
    """
    Run through K2-24 example
    """
    _standard_run(setupfn, _args())

def test_proceed(setupfn='example_planets/epic203771098.py'):
    """
    Run through K2-24 example with the proceed argument, pulling from the previous run.
    """
    _standard_run(setupfn, _args2())

def test_hd(setupfn='example_planets/HD164922.py'):
    """
    Check multi-instrument fit
    """
    args = _args()
    args.setupfn = setupfn
    radvel.driver.fit(args)

    args.serial = False
    args.nensembles = 4
    radvel.driver.mcmc(args)

    args.type = ['rv']
    args.plotkw = {}
    radvel.driver.plots(args)


def test_k2131(setupfn='example_planets/k2-131.py'):
    """
    Check GP fit
    """
    args = _args()
    args.setupfn = setupfn

    radvel.driver.fit(args)

    args.type = ['gp']
    args.verbose = True
    radvel.driver.ic_compare(args)

    args.type = ['rv']
    args.gp = True
    args.plotkw = {}
    radvel.driver.plots(args)


def test_celerite(setupfn='example_planets/k2-131_celerite.py'):
    """
    Check celerite GP fit
    """
    args = _args()
    args.setupfn = setupfn

    radvel.driver.fit(args)

    args.type = ['rv']
    args.gp = True
    args.plotkw = {'plot_likelihoods_separately':True}
    radvel.driver.plots(args)


def test_basis():
    """
    Test basis conversions
    """

    basis_list = radvel.basis.BASIS_NAMES
    default_basis = 'per tc e w k'

    anybasis_params = radvel.Parameters(1, basis=default_basis)

    anybasis_params['per1'] = radvel.Parameter(value=20.885258)
    anybasis_params['tc1'] = radvel.Parameter(value=2072.79438)
    anybasis_params['e1'] = radvel.Parameter(value=0.01)
    anybasis_params['w1'] = radvel.Parameter(value=1.6)
    anybasis_params['k1'] = radvel.Parameter(value=10.0)

    anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
    anybasis_params['curv'] = radvel.Parameter(value=0.0)

    anybasis_params['gamma_j'] = radvel.Parameter(1.0)
    anybasis_params['jit_j'] = radvel.Parameter(value=2.6)

    for new_basis in basis_list:
        iparams = radvel.basis._copy_params(anybasis_params)
        if new_basis != default_basis:
            new_params = iparams.basis.to_any_basis(iparams, new_basis)
            tmp = radvel.basis._copy_params(new_params)

            old_params = tmp.basis.to_any_basis(tmp, default_basis)

            for par in iparams:
                before = iparams[par].value
                after = old_params[par].value
                assert (before - after) <= 1e-5,\
                    "Parameters do not match after basis conversion: \
{}, {} != {}".format(par, before, after)


def test_kernels():
    """
    Test basic functionality of all standard GP kernels
    """

    kernel_list = radvel.gp.KERNELS

    for kernel in kernel_list:
        hnames = kernel_list[kernel] # gets list of hyperparameter name strings
        hyperparams = {k: radvel.Parameter(value=1.) for k in hnames}
        kernel_call = getattr(radvel.gp, kernel + "Kernel")
        test_kernel = kernel_call(hyperparams)

        x = np.array([1.,2.,3.])
        test_kernel.compute_distances(x,x)
        test_kernel.compute_covmatrix(x.T)

        print("Testing {}".format(kernel_call(hyperparams)))

        sys.stdout.write("Testing error catching with dummy hyperparameters... \n")

        fakeparams1 = {}
        fakeparams1['dummy'] = radvel.Parameter(value=1.0)
        try:
            kernel_call(fakeparams1)
            raise Exception('Test #1 failed for {}'.format(kernel))
        except AssertionError:
            sys.stdout.write("passed #1\n")

        fakeparams2 = copy.deepcopy(hyperparams)
        fakeparams2[hnames[0]] = 1.
        try:
            kernel_call(fakeparams2)
            raise Exception('Test #2 failed for {}'.format(kernel))
        except AttributeError:
            sys.stdout.write("passed #2\n")


def test_priors():
    """
    Test basic functionality of all Priors
    """

    params = radvel.Parameters(1)
    params['per1'] = radvel.Parameter(10.0)
    params['tc1'] = radvel.Parameter(0.0)
    params['secosw1'] = radvel.Parameter(0.0)
    params['sesinw1'] = radvel.Parameter(0.0)
    params['logk1'] = radvel.Parameter(1.5)

    testTex = r'Delta Function Prior on $\sqrt{e}\cos{\omega}_{b}$'

    def userdef_prior_func(inp_list):
        if inp_list[0] >= 0. and inp_list[0] < 1.:
            return 0.
        else:
            return -np.inf

    prior_tests = {
        radvel.prior.EccentricityPrior(1):                  1/.99,
        radvel.prior.PositiveKPrior(1):                     1.0,
        radvel.prior.Gaussian('per1', 9.9, 0.1):            scipy.stats.norm(9.9,0.1).pdf(10.),
        radvel.prior.HardBounds('per1', 1.0, 9.0):          0.,
        radvel.prior.HardBounds('per1', 1.0, 11.0):         1./10.,
        radvel.prior.Jeffreys('per1', 0.1, 100.0):          (1./10.)/np.log(100./0.1),
        radvel.prior.ModifiedJeffreys('per1', 0.1, 100.0, 0.):  (1./10.)/np.log(100./0.1),
        radvel.prior.ModifiedJeffreys('per1', 2., 100.0, 1.):  (1./9.)/np.log(99.),
        radvel.prior.SecondaryEclipsePrior(1, 5.0, 10.0):    1./np.sqrt(2.*np.pi),
        radvel.prior.NumericalPrior(
            ['sesinw1'],
            np.random.randn(1,5000000)
        ):                                                  scipy.stats.norm(0, 1).pdf(0.),
        radvel.prior.UserDefinedPrior(
            ['secosw1'], userdef_prior_func, testTex
        ):                                                  1.0,
        radvel.prior.InformativeBaselinePrior(
            'per1', 5.0, duration=1.0
        ):                                                  6./10.

    }

    for prior, val in prior_tests.items():
        print(prior.__repr__())
        print(prior.__str__())
        tolerance = .01
        print(abs(np.exp(prior(params))))
        print(val)
        assert abs(np.exp(prior(params)) - val) < tolerance, \
            "Prior output does not match expectation"


def test_kepler():
    """
    Profile and test C-based Kepler solver
    """
    radvel.kepler.profile()


def test_model_comp(setupfn='example_planets/HD164922.py'):
    """
    Test some additional model_comp lines
    """

    args = _args()
    args.setupfn = setupfn
    radvel.driver.fit(args)

    # also check some additional lines of model_comp
    args.verbose = True
    args.type = ['trend']
    radvel.driver.ic_compare(args)

    args.simple = True
    args.type = ['e']
    radvel.driver.ic_compare(args)

    args.simple = False
    args.type = ['something_else']
    try:
        radvel.driver.ic_compare(args)
        raise Exception("Unexpected result from model_comp.")
    except AssertionError:  # expected result
        return


if __name__ == '__main__':
    test_k2()
    test_proceed()
    test_hd()
    test_model_comp()
    test_k2131()
    test_celerite()
    test_basis()
    test_kernels()
    test_kepler()
