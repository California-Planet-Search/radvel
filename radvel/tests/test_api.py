import sys
import copy
import warnings
import types
import pytest

import radvel
import radvel.driver
import numpy as np
import scipy
import radvel.prior

warnings.simplefilter('ignore')


class _args(types.SimpleNamespace):
    outputdir = '/tmp/'
    decorr = False
    name_in_title = False
    gp = False
    simple = False

    nwalkers = 50
    nsteps = 100
    ensembles = 8
    maxGR = 1.10
    burnGR = 1.30
    burnAfactor = 25
    minAfactor = 50
    maxArchange = .07
    minTz = 1000
    minsteps = 100
    minpercent = 5
    thin = 1
    serial = False
    save = True
    savename = 'rawchains.h5'
    proceed = False
    proceedname = None
    headless=False


def _standard_run(setupfn, arguments):
    """
    Run through all of the standard steps
    """

    args = arguments
    args.setupfn = setupfn

    radvel.driver.fit(args)
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
    args = _args()
    args.setupfn = setupfn
    _standard_run(setupfn, args)

    # set the proceed flag and continue
    args.proceed = True
    radvel.driver.mcmc(args)

    args.ensembles = 1
    try:
        radvel.driver.mcmc(args)
    except ValueError:  # expected error when changing number of ensembles with proceed flag
        pass

    args.serial = True
    args.proceed = False
    radvel.driver.mcmc(args)


def test_hd(setupfn='example_planets/HD164922.py'):
    """
    Check multi-instrument fit
    """
    args = _args()
    args.setupfn = setupfn

    radvel.driver.fit(args)
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
        ivector = radvel.Vector(iparams)
        if new_basis != default_basis:
            new_vector = iparams.basis.v_to_any_basis(ivector, new_basis)
            new_params = iparams.basis.to_any_basis(iparams, new_basis)
            tmpv = new_vector.copy()
            tmp = radvel.basis._copy_params(new_params)

            old_vector = tmp.basis.v_to_any_basis(tmpv, default_basis)
            old_params = tmp.basis.to_any_basis(tmp, default_basis)

            for par in iparams:
                before = iparams[par].value
                after = old_params[par].value
                assert (before - after) <= 1e-5,\
                    "Parameters do not match after basis conversion: \
{}, {} != {}".format(par, before, after)

            for i in range(ivector.vector.shape[0]):
                before = ivector.vector[i][0]
                after = old_vector[i][0]
                assert (before - after) <= 1e-5, \
                    "Vectors do not match after basis conversion: \
{} row, {} != {}".format(i, before, after)



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


@pytest.fixture
def params_and_vector_for_priors():
    params = radvel.Parameters(1, 'per tc secosw sesinw logk')
    params['per1'] = radvel.Parameter(10.0)
    params['tc1'] = radvel.Parameter(0.0)
    params['secosw1'] = radvel.Parameter(0.0)
    params['sesinw1'] = radvel.Parameter(0.0)
    params['logk1'] = radvel.Parameter(1.5)

    vector = radvel.Vector(params)

    return params, vector

def test_priors(params_and_vector_for_priors):
    """
    Test basic functionality of all Priors
    """

    params, vector = params_and_vector_for_priors

    testTex = r'Delta Function Prior on $\sqrt{e}\cos{\omega}_{b}$'

    def userdef_prior_func(inp_list):
        if inp_list[0] >= 0. and inp_list[0] < 1.:
            return 0.
        else:
            return -np.inf

    prior_tests = {
        radvel.prior.EccentricityPrior(1):                  1/.99,
        radvel.prior.EccentricityPrior([1]):                1/.99,
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
        print(abs(np.exp(prior(params, vector))))
        print(val)
        assert abs(np.exp(prior(params, vector)) - val) < tolerance, \
            "Prior output does not match expectation"


def test_prior_transforms(params_and_vector_for_priors):

    rng = np.random.default_rng(3245)
    u = rng.uniform(size=100)

    prior_dict = {
        radvel.prior.Gaussian("per1", 9.9, 0.1): scipy.stats.norm.ppf(u, 9.9, 0.1),
        radvel.prior.HardBounds("per1", 1.0, 9.0): scipy.stats.uniform.ppf(u, 1.0, 9.0 - 1.0),
        radvel.prior.Jeffreys("per1", 0.1, 100.0): scipy.stats.loguniform.ppf(u, 0.1, 100.0),
        radvel.prior.ModifiedJeffreys(
            "per1", 0.0, 100.0, -0.1
        ): scipy.stats.loguniform.ppf(u, 0.0+0.1, 100.0+0.1, loc=-0.1),
        radvel.prior.UserDefinedPrior(
            ["per1"],
            lambda x: scipy.stats.lognorm.pdf(x, 1e-1, 1e1),
            "lognorm",
            transform_func=lambda x: scipy.stats.lognorm.ppf(x, 1e-1, 1e1),
        ): scipy.stats.lognorm.ppf(u, 1e-1, 1e1),

    }

    for prior, expected_val in prior_dict.items():
        np.testing.assert_allclose(
            prior.transform(u),
            expected_val,
            err_msg=f"Prior transform failed for {prior}")

    with pytest.raises(TypeError):
        radvel.prior.UserDefinedPrior(
            ["per1"],
            lambda x: scipy.stats.lognorm.pdf(x, 1e-1, 1e1),
            "lognorm",
        ).transform(u)

    # Make sure other priors raise error for transform
    no_transform_priors = [
        radvel.prior.EccentricityPrior(1),
        radvel.prior.PositiveKPrior(1),
        radvel.prior.SecondaryEclipsePrior(1, 5.0, 10.0),
        radvel.prior.NumericalPrior(['sesinw1'], np.random.randn(1,5000000)),
        radvel.prior.InformativeBaselinePrior('per1', 5.0, duration=1.0),
    ]
    for prior in no_transform_priors:
        with pytest.raises(NotImplementedError):
            prior.transform(u)


@pytest.fixture
def likelihood_for_pt(params_and_vector_for_priors):
    params = params_and_vector_for_priors[0]
    t = np.linspace(0, 10, num=100)
    vel = np.ones_like(t)
    errvel = np.ones_like(t) * 0.1
    mod = radvel.RVModel(params)
    mod.params['dvdt'] = radvel.Parameter(value=-0.02)
    mod.params['curv'] = radvel.Parameter(value=0.01)
    like = radvel.likelihood.RVLikelihood(mod, t, vel, errvel)
    like.params['gamma'] = radvel.Parameter(value=0.1, vary=False)
    like.params['jit'] = radvel.Parameter(value=1.0)
    like.params['secosw1'].vary = False
    like.params['sesinw1'].vary = False
    like.params['per1'].vary = False
    like.params['tc1'].vary = False
    return like


def test_prior_transform_all_params(likelihood_for_pt):

    # This should work
    post = radvel.posterior.Posterior(likelihood_for_pt)
    post.priors += [radvel.prior.Gaussian( 'dvdt', 0, 1.0)]
    post.priors += [radvel.prior.HardBounds( 'curv', 0.0, 1.0)]
    post.priors += [radvel.prior.ModifiedJeffreys( 'jit', 0, 10.0, -0.1)]
    post.priors += [radvel.prior.Gaussian( 'logk1', np.log(5), 5)]

    post.check_proper_priors()

    post = radvel.posterior.Posterior(likelihood_for_pt)
    post.priors += [radvel.prior.Gaussian( 'dvdt', 0, 1.0)]
    post.priors += [radvel.prior.HardBounds( 'curv', 0.0, 1.0)]
    post.priors += [radvel.prior.ModifiedJeffreys( 'jit', 0, 10.0, -0.1)]
    with pytest.raises(ValueError, match="No prior"):
        post.check_proper_priors()

    post = radvel.posterior.Posterior(likelihood_for_pt)
    post.priors += [radvel.prior.Gaussian( 'dvdt', 0, 1.0)]
    post.priors += [radvel.prior.HardBounds( 'curv', 0.0, 1.0)]
    post.priors += [radvel.prior.ModifiedJeffreys( 'jit', 0, 10.0, -0.1)]
    post.priors += [radvel.prior.Gaussian( 'logk1', np.log(5), 5)]
    post.priors += [radvel.prior.Gaussian( 'logk1', 8, 5)]
    with pytest.raises(ValueError, match="Multiple priors"):
        post.check_proper_priors()



def test_prior_transform_order(likelihood_for_pt):

    post = radvel.posterior.Posterior(likelihood_for_pt)
    post.priors += [radvel.prior.Gaussian( 'dvdt', 0, 1.0)]
    post.priors += [radvel.prior.HardBounds( 'curv', 0.0, 1.0)]
    post.priors += [radvel.prior.ModifiedJeffreys( 'jit', 0, 10.0, -0.1)]
    post.priors += [radvel.prior.Gaussian( 'logk1', np.log(5), 5)]

    rng = np.random.default_rng(3245)
    u = rng.uniform(size=(len(post.vary_params), 100))
    p = post.prior_transform(u)

    prior_param_names = [prior.param for prior in post.priors]

    assert prior_param_names != post.name_vary_params(), "Parameters and priors should have different order for this test"

    for prior in post.priors:
        param_name = prior.param
        param_ind = post.name_vary_params().index(param_name)
        np.testing.assert_allclose(prior.transform(u[param_ind]), p[param_ind], err_msg="Prior transform failed for {}".format(prior))

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
    #test_k2()
    #test_hd()
    #test_model_comp()
    test_k2131()
    #test_celerite()
    # test_basis()
    #test_kernels()
    #test_kepler()
    #test_priors()
