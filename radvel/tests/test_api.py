import sys
import copy
import warnings
import types
import pytest

import celerite
from celerite import terms
import radvel
import radvel.driver
import numpy as np
import scipy
import radvel.prior

warnings.simplefilter('ignore')


class QPTerm(terms.Term):
    parameter_names = ("log_B", "log_C", "log_L", "log_Prot")

    def get_real_coefficients(self, params):
        B, C, L, _ = np.exp(params)
        return (
            B * (1.0 + C) / (2.0 + C), 1 / L,
        )

    def get_complex_coefficients(self, params):
        B, C, L, Prot = np.exp(params)
        return (
            B / (2.0 + C), 0.0,
            1 / L, 2 * np.pi / Prot,
        )


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


def test_celerite_fit(setupfn='example_planets/k2-131_celerite.py'):
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


def constant_rv(t, params, vector):
    """'RV' model that returns 0 for test GP liklihoods """
    return np.zeros_like(t)


@pytest.fixture()
def celerite_data():
    """
    Random example dataset inspired from the "First steps" celerite tutorial
    Link: https://celerite.readthedocs.io/en/stable/tutorials/first/
    """

    t = np.sort(
        np.append(
            np.random.uniform(0, 3.8, 57),
            np.random.uniform(5.5, 10, 68),
        )
    )  # The input coordinates must be sorted
    yerr = np.random.uniform(0.08, 0.22, len(t))
    y = (
        0.2 * (t-5)
        + np.sin(3*t + 0.1*(t-5)**2)
        + yerr * np.random.randn(len(t))
     )

    return t, y, yerr


def test_celerite_qp(celerite_data, tol=1e-7):
    """
    Check that QP kernel gives same cov matrix and prediction as celerite on
    random test data.
    """
    # Define celerite GP
    hparams = {
        "gp_B": 1.0,
        "gp_C": 1e-3,
        "gp_L": 0.1,
        "gp_Prot": 0.2,
    }

    t, y, yerr = celerite_data

    # Compute matrix purely with celerite
    cker = QPTerm(
        log_B=np.log(hparams["gp_B"]),
        log_C=np.log(hparams["gp_C"]),
        log_L=np.log(hparams["gp_L"]),
        log_Prot=np.log(hparams["gp_Prot"]),
    )
    gp = celerite.GP(cker)
    gp.compute(t, yerr)
    cmat = gp.get_matrix()

    # Compute matrix with RadVel
    rker = radvel.gp.CeleriteQuasiPerKernel(
        {k: radvel.Parameter(value=v) for k, v in hparams.items()}
    )
    rker.compute_distances(t, t)
    rker.compute_covmatrix(yerr)
    rmat = rker.get_matrix(errors=yerr)

    assert np.all(np.abs(rmat - cmat) < tol)

    # Radvel likelihood to compare prediction
    params = radvel.Parameters(0)
    for pname in hparams:
        params[pname] = radvel.Parameter(value=hparams[pname])
    mod = radvel.GeneralRVModel(params, forward_model=constant_rv)
    mod.params["dvdt"] = radvel.Parameter(value=0.0, vary=False)
    mod.params["curv"] = radvel.Parameter(value=0.0, vary=False)
    mod.num_planets = 0
    like = radvel.CeleriteLikelihood(
        mod, t, y, yerr, hnames=list(hparams), kernel_name="CeleriteQuasiPer"
    )
    # Add constant offset and white noise
    like.params["gamma"] = radvel.Parameter(value=y.mean())
    like.params["jit"] = radvel.Parameter(value=0.0)
    like.vector.dict_to_vector()

    cpred, cvar = gp.predict(
        y - like.params["gamma"].value, t, return_var=True
    )
    cstd = np.sqrt(cvar)
    rpred, rstd = like.predict(t)

    assert np.all(np.abs(rpred - cpred) < tol)
    assert np.all(np.abs(rstd - cstd) < tol)


# Test Q above and below 0.5
@pytest.mark.parametrize("Q", [3.0, 0.2])
def test_celerite_sho(celerite_data, Q, tol=1e-7):
    """
    Check that SHO kernel gives same cov matrix and prediction as celerite on
    random test data.
    """
    # Define celerite GP
    hparams = {
        "gp_S0": 1.0,
        "gp_Q": Q,
        "gp_w0": 7.0,
    }

    t, y, yerr = celerite_data

    # Compute matrix purely with celerite
    cker = terms.SHOTerm(
        log_S0=np.log(hparams["gp_S0"]),
        log_Q=np.log(hparams["gp_Q"]),
        log_omega0=np.log(hparams["gp_w0"])
    )
    gp = celerite.GP(cker, mean=np.mean(y))
    gp.compute(t, yerr)
    cmat = gp.get_matrix()

    # Compute matrix with RadVel
    rker = radvel.gp.CeleriteSHOKernel(
        {k: radvel.Parameter(value=v) for k, v in hparams.items()}
    )
    rker.compute_distances(t, t)
    rker.compute_covmatrix(yerr)
    rmat = rker.get_matrix(errors=yerr)

    assert np.all(np.abs(rmat - cmat) < tol)

    # Radvel likelihood to compare prediction
    params = radvel.Parameters(0)
    for pname in hparams:
        params[pname] = radvel.Parameter(value=hparams[pname])
    mod = radvel.GeneralRVModel(params, forward_model=constant_rv)
    mod.params["dvdt"] = radvel.Parameter(value=0.0, vary=False)
    mod.params["curv"] = radvel.Parameter(value=0.0, vary=False)
    mod.num_planets = 0
    like = radvel.CeleriteLikelihood(
        mod, t, y, yerr, hnames=list(hparams), kernel_name="CeleriteSHO"
    )
    # Add constant offset and white noise
    like.params["gamma"] = radvel.Parameter(value=y.mean())
    like.params["jit"] = radvel.Parameter(value=0.0)
    like.vector.dict_to_vector()

    cpred, cvar = gp.predict(
        y - like.params["gamma"].value, t, return_var=True
    )
    cstd = np.sqrt(cvar)
    rpred, rstd = like.predict(t)

    # The values are not exactly the same, so check within fraction of error
    ptol = 1e-1 * np.min([rstd, cstd], axis=0)
    assert np.all(np.abs(rpred - cpred) < ptol)
    assert np.all(np.abs(rstd - cstd) < ptol)


@pytest.fixture()
def celerite_data_line():
    """
    Random example dataset inspired from the "First steps" celerite tutorial
    Link: https://celerite.readthedocs.io/en/stable/tutorials/first/
    """

    t = np.sort(
        np.append(
            np.random.uniform(0, 3.8, 57),
            np.random.uniform(5.5, 10, 68),
        )
    )  # The input coordinates must be sorted
    yerr = np.random.uniform(0.08, 0.22, len(t))
    y = (
        0.2 * (t-5)
        + np.sin(t)
        + yerr * np.random.randn(len(t))
     )

    return t, y, yerr


def test_celerite_matern32(celerite_data_line, tol=1e-7):
    """
    Check that Matern 3/2 kernel gives same cov matrix and prediction as
    celerite on random test data.
    """
    # Define celerite GP
    hparams = {
        "gp_sigma": 2.0,
        "gp_rho": 0.1,
    }

    t, y, yerr = celerite_data_line

    # Compute matrix purely with celerite
    cker = terms.Matern32Term(
        log_sigma=np.log(hparams["gp_sigma"]),
        log_rho=np.log(hparams["gp_rho"]),
    )
    gp = celerite.GP(cker)
    gp.compute(t, yerr)
    cmat = gp.get_matrix()

    # Compute matrix with RadVel
    rker = radvel.gp.CeleriteMatern32Kernel(
        {k: radvel.Parameter(value=v) for k, v in hparams.items()}
    )
    rker.compute_distances(t, t)
    rker.compute_covmatrix(yerr)
    rmat = rker.get_matrix(errors=yerr)

    assert np.all(np.abs(rmat - cmat) < tol)

    # Radvel likelihood to compare prediction
    params = radvel.Parameters(0)
    for pname in hparams:
        params[pname] = radvel.Parameter(value=hparams[pname])
    mod = radvel.GeneralRVModel(params, forward_model=constant_rv)
    mod.params["dvdt"] = radvel.Parameter(value=0.0, vary=False)
    mod.params["curv"] = radvel.Parameter(value=0.0, vary=False)
    mod.num_planets = 0
    like = radvel.CeleriteLikelihood(
        mod, t, y, yerr, hnames=list(hparams), kernel_name="CeleriteMatern32"
    )
    # Add constant offset and white noise
    like.params["gamma"] = radvel.Parameter(value=y.mean())
    like.params["jit"] = radvel.Parameter(value=0.0)
    like.vector.dict_to_vector()

    cpred, cvar = gp.predict(
        y - like.params["gamma"].value, t, return_var=True
    )
    cstd = np.sqrt(cvar)
    rpred, rstd = like.predict(t)

    assert np.all(np.abs(rpred - cpred) < tol)
    assert np.all(np.abs(rstd - cstd) < tol)


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


def test_priors():
    """
    Test basic functionality of all Priors
    """

    params = radvel.Parameters(1, 'per tc secosw sesinw logk')
    params['per1'] = radvel.Parameter(10.0)
    params['tc1'] = radvel.Parameter(0.0)
    params['secosw1'] = radvel.Parameter(0.0)
    params['sesinw1'] = radvel.Parameter(0.0)
    params['logk1'] = radvel.Parameter(1.5)

    vector = radvel.Vector(params)

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
    #test_k2131()
    #test_celerite()
    test_basis()
    #test_kernels()
    #test_kepler()
    #test_priors()
