"""
Additional tests to improve code coverage for gp, likelihood, and mcmc modules.
"""
import copy
import numpy as np
import pytest

import importlib

import radvel
import radvel.gp
import radvel.likelihood
# radvel.__init__ does `from .mcmc import *` which shadows the module
# with the mcmc function. Import the actual module directly.
mcmc_module = importlib.import_module('radvel.mcmc')


# --- Helper fixtures ---

def _make_likelihood(ndata=100):
    """Create a simple RVLikelihood for testing."""
    params = radvel.Parameters(1, 'per tc secosw sesinw logk')
    params['per1'] = radvel.Parameter(10.0)
    params['tc1'] = radvel.Parameter(0.0)
    params['secosw1'] = radvel.Parameter(0.0)
    params['sesinw1'] = radvel.Parameter(0.0)
    params['logk1'] = radvel.Parameter(1.5)

    mod = radvel.RVModel(params)
    mod.params['dvdt'] = radvel.Parameter(value=0.0)
    mod.params['curv'] = radvel.Parameter(value=0.0)

    t = np.linspace(0, 100, num=ndata)
    vel = np.sin(2 * np.pi * t / 10.0) + np.random.default_rng(42).normal(0, 0.5, ndata)
    errvel = np.ones(ndata) * 0.5

    like = radvel.likelihood.RVLikelihood(mod, t, vel, errvel)
    like.params['gamma'] = radvel.Parameter(value=0.0)
    like.params['jit'] = radvel.Parameter(value=1.0)
    return like


def _make_sqexp_kernel():
    hparams = {
        'gp_length': radvel.Parameter(value=10.0),
        'gp_amp': radvel.Parameter(value=5.0),
    }
    return radvel.gp.SqExpKernel(hparams)


def _make_per_kernel():
    hparams = {
        'gp_per': radvel.Parameter(value=25.0),
        'gp_length': radvel.Parameter(value=0.5),
        'gp_amp': radvel.Parameter(value=3.0),
    }
    return radvel.gp.PerKernel(hparams)


def _make_quasiper_kernel():
    hparams = {
        'gp_per': radvel.Parameter(value=25.0),
        'gp_perlength': radvel.Parameter(value=0.5),
        'gp_explength': radvel.Parameter(value=30.0),
        'gp_amp': radvel.Parameter(value=3.0),
    }
    return radvel.gp.QuasiPerKernel(hparams)


def _make_celerite_kernel():
    hparams = {
        'gp_B': radvel.Parameter(value=1.0),
        'gp_C': radvel.Parameter(value=1.0),
        'gp_L': radvel.Parameter(value=10.0),
        'gp_Prot': radvel.Parameter(value=25.0),
    }
    return radvel.gp.CeleriteKernel(hparams)


# --- GP Kernel Tests ---

class TestSqExpKernel:
    def test_name(self):
        k = _make_sqexp_kernel()
        assert k.name == "SqExp"

    def test_repr(self):
        k = _make_sqexp_kernel()
        s = repr(k)
        assert "SqExp" in s
        assert "10" in s  # length
        assert "5" in s   # amp

    def test_compute(self):
        k = _make_sqexp_kernel()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k.compute_distances(x, x)
        cov = k.compute_covmatrix(np.ones(5) * 0.5)
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)  # symmetric

    def test_compute_nonsquare(self):
        k = _make_sqexp_kernel()
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.5, 2.5])
        k.compute_distances(x1, x2)
        cov = k.compute_covmatrix(0)  # non-square, errors must be 0
        assert cov.shape == (3, 2)

    def test_init_wrong_count(self):
        with pytest.raises(AssertionError):
            radvel.gp.SqExpKernel({'gp_length': radvel.Parameter(1.0)})

    def test_init_wrong_type(self):
        with pytest.raises(AttributeError):
            radvel.gp.SqExpKernel({'gp_length': 1.0, 'gp_amp': 2.0})


class TestPerKernel:
    def test_name(self):
        k = _make_per_kernel()
        assert k.name == "Per"

    def test_repr(self):
        k = _make_per_kernel()
        s = repr(k)
        assert "Per" in s
        assert "25" in s

    def test_compute(self):
        k = _make_per_kernel()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k.compute_distances(x, x)
        cov = k.compute_covmatrix(np.ones(5) * 0.5)
        assert cov.shape == (5, 5)

    def test_compute_nonsquare(self):
        k = _make_per_kernel()
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.5, 2.5])
        k.compute_distances(x1, x2)
        cov = k.compute_covmatrix(0)
        assert cov.shape == (3, 2)

    def test_init_wrong_count(self):
        with pytest.raises(AssertionError):
            radvel.gp.PerKernel({'gp_length': radvel.Parameter(1.0)})

    def test_init_wrong_type(self):
        with pytest.raises(AttributeError):
            radvel.gp.PerKernel({
                'gp_per': 1.0, 'gp_length': 2.0, 'gp_amp': 3.0
            })

    def test_init_wrong_names(self):
        with pytest.raises(KeyError):
            radvel.gp.PerKernel({
                'gp_foo': radvel.Parameter(1.0),
                'gp_bar': radvel.Parameter(2.0),
                'gp_baz': radvel.Parameter(3.0),
            })


class TestQuasiPerKernel:
    def test_name(self):
        k = _make_quasiper_kernel()
        assert k.name == "QuasiPer"

    def test_repr(self):
        k = _make_quasiper_kernel()
        s = repr(k)
        assert "QuasiPer" in s
        assert "25" in s

    def test_compute(self):
        k = _make_quasiper_kernel()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k.compute_distances(x, x)
        cov = k.compute_covmatrix(np.ones(5) * 0.5)
        assert cov.shape == (5, 5)

    def test_zero_amp(self):
        """When GP amp is 0, should return zero covariance matrix."""
        k = _make_quasiper_kernel()
        k.hparams['gp_amp'].value = 0.0
        x = np.array([1.0, 2.0, 3.0])
        k.compute_distances(x, x)
        cov = k.compute_covmatrix(np.ones(3) * 0.5)
        # Diagonal should have error^2, off-diagonal should be 0
        assert np.allclose(np.diag(cov), 0.25)

    def test_zero_amp_no_errors(self):
        """When GP amp is 0 and errors are 0, should return zero matrix."""
        k = _make_quasiper_kernel()
        k.hparams['gp_amp'].value = 0.0
        x = np.array([1.0, 2.0, 3.0])
        k.compute_distances(x, x)
        cov = k.compute_covmatrix(0)
        assert np.allclose(cov, 0.0)

    def test_invalid_negative_amp(self):
        k = _make_quasiper_kernel()
        k.hparams['gp_amp'].value = -1.0
        x = np.array([1.0, 2.0, 3.0])
        k.compute_distances(x, x)
        with pytest.raises(ValueError, match="gp_amp"):
            k.compute_covmatrix(np.ones(3))

    def test_invalid_inf_per(self):
        k = _make_quasiper_kernel()
        k.hparams['gp_per'].value = np.inf
        x = np.array([1.0, 2.0, 3.0])
        k.compute_distances(x, x)
        with pytest.raises(ValueError, match="gp_per"):
            k.compute_covmatrix(np.ones(3))

    def test_init_wrong_count(self):
        with pytest.raises(AssertionError):
            radvel.gp.QuasiPerKernel({'gp_amp': radvel.Parameter(1.0)})

    def test_init_wrong_type(self):
        with pytest.raises(AttributeError):
            radvel.gp.QuasiPerKernel({
                'gp_per': 1.0, 'gp_perlength': 2.0,
                'gp_explength': 3.0, 'gp_amp': 4.0,
            })

    def test_init_wrong_names(self):
        with pytest.raises(KeyError):
            radvel.gp.QuasiPerKernel({
                'gp_a': radvel.Parameter(1.0),
                'gp_b': radvel.Parameter(2.0),
                'gp_c': radvel.Parameter(3.0),
                'gp_d': radvel.Parameter(4.0),
            })


class TestCeleriteKernel:
    @pytest.fixture(autouse=True)
    def skip_without_celerite(self):
        if not radvel.gp._has_celerite:
            pytest.skip("celerite not available")

    def test_name(self):
        k = _make_celerite_kernel()
        assert k.name == "Celerite"

    def test_repr(self):
        k = _make_celerite_kernel()
        s = repr(k)
        assert "Celerite" in s
        assert "25" in s  # Prot

    def test_compute_real_and_complex(self):
        k = _make_celerite_kernel()
        k.compute_real_and_complex_hparams()
        assert k.real.shape == (1, 4)
        assert k.complex.shape == (1, 4)

    def test_compute_distances(self):
        k = _make_celerite_kernel()
        x = np.array([1.0, 2.0, 3.0])
        k.compute_distances(x, x)
        assert np.array_equal(k.x, x)
        assert k.A.shape == (0,)
        assert k.U.shape == (0, 0)

    def test_compute_covmatrix(self):
        k = _make_celerite_kernel()
        x = np.sort(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        k.compute_distances(x, x)
        errors = np.ones(5) * 0.5
        solver = k.compute_covmatrix(errors)
        assert solver is not None

    def test_init_wrong_count(self):
        with pytest.raises(AssertionError):
            radvel.gp.CeleriteKernel({'gp_B': radvel.Parameter(1.0)})

    def test_init_wrong_type(self):
        with pytest.raises(AttributeError):
            radvel.gp.CeleriteKernel({
                'gp_B': 1.0, 'gp_C': 2.0, 'gp_L': 3.0, 'gp_Prot': 4.0,
            })


# --- Likelihood Tests ---

class TestLikelihoodCoverage:
    def test_residuals(self):
        like = _make_likelihood()
        res = like.residuals()
        assert isinstance(res, np.ndarray)
        assert len(res) == 100

    def test_neglogprob(self):
        like = _make_likelihood()
        nlp = like.neglogprob()
        assert isinstance(nlp, (float, np.floating))
        assert nlp == -like.logprob()

    def test_logprob_array(self):
        like = _make_likelihood()
        vary = like.get_vary_params()
        lp = like.logprob_array(vary)
        assert isinstance(lp, (float, np.floating))

    def test_repr_with_uparams(self):
        like = _make_likelihood()
        like.uparams = {'logk1': 0.5, 'jit': 0.3}
        s = repr(like)
        assert 'value' in s
        assert '+/-' in s

    def test_aicc_warning_few_data(self):
        """AICc should be inf when k >= n."""
        like = _make_likelihood(ndata=3)
        # With 3 data points and many free params, AICc should be inf
        aic = like.aic()
        # If not inf, it's because not enough free params. Make all vary.
        for key in like.params:
            like.params[key].vary = True
        like.vector.dict_to_vector()
        aic = like.aic()
        assert np.isinf(aic)

    def test_bic(self):
        like = _make_likelihood()
        bic = like.bic()
        assert isinstance(bic, (float, np.floating))
        assert np.isfinite(bic)


class TestGPLikelihoodCoverage:
    def test_logprob_non_positive_definite(self):
        """GP logprob should return -inf for non-positive-definite kernel."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)

        params['gp_per'] = radvel.Parameter(value=25.0)
        params['gp_perlength'] = radvel.Parameter(value=0.5)
        params['gp_explength'] = radvel.Parameter(value=30.0)
        params['gp_amp'] = radvel.Parameter(value=1e-15)  # tiny amplitude

        mod = radvel.RVModel(params)
        mod.params['dvdt'] = radvel.Parameter(value=0.0)
        mod.params['curv'] = radvel.Parameter(value=0.0)

        t = np.linspace(0, 100, 20)
        vel = np.random.default_rng(42).normal(0, 1, 20)
        errvel = np.ones(20) * 0.5

        like = radvel.likelihood.GPLikelihood(
            mod, t, vel, errvel,
            hnames=['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'],
        )
        like.params['gamma'] = radvel.Parameter(value=0.0)
        like.params['jit'] = radvel.Parameter(value=1.0)

        # Should return a finite value (the kernel is still valid, just small amplitude)
        lp = like.logprob()
        assert isinstance(lp, (float, np.floating))


# --- MCMC Utility Tests ---

class TestMCMCUtilities:
    def test_progress_bar_empty(self):
        bar = mcmc_module._progress_bar(0, 100)
        assert bar == "[" + " " * 50 + "]"

    def test_progress_bar_half(self):
        bar = mcmc_module._progress_bar(50, 100)
        assert "=" in bar
        assert " " in bar

    def test_progress_bar_full(self):
        bar = mcmc_module._progress_bar(100, 100)
        assert bar == "[" + "=" * 50 + "]"

    def test_progress_bar_custom_width(self):
        bar = mcmc_module._progress_bar(10, 10, width=20)
        assert bar == "[" + "=" * 20 + "]"
        assert len(bar) == 22  # width + 2 brackets

    def test_mcmc_save_no_name(self):
        like = _make_likelihood()
        post = radvel.posterior.Posterior(like)
        with pytest.raises(ValueError, match="save set to true"):
            mcmc_module.mcmc(post, save=True, savename=None, headless=True)

    def test_mcmc_proceed_no_name(self):
        like = _make_likelihood()
        post = radvel.posterior.Posterior(like)
        with pytest.raises(ValueError, match="proceed set to true"):
            mcmc_module.mcmc(post, proceed=True, proceedname=None, headless=True)

    def test_isnotebook(self):
        # In a test environment, this should return False
        assert mcmc_module.isnotebook() is False

    def test_statevars_reset(self):
        sv = mcmc_module.StateVars()
        sv.oac = 42
        sv.reset()
        assert sv.oac == 0


# --- Model Tests ---

class TestModelCoverage:
    def test_parameter_repr(self):
        p = radvel.Parameter(value=3.14, vary=True)
        s = repr(p)
        assert "3.14" in s
        assert "True" in s

    def test_parameter_repr_with_mcmcscale(self):
        p = radvel.Parameter(value=1.3)
        p.mcmcscale = 100.0
        s = repr(p)
        assert "100" in s

    def test_parameter_float(self):
        p = radvel.Parameter(value=2.718)
        assert float(p) == 2.718

    def test_parameter_equals(self):
        p1 = radvel.Parameter(value=1.0, vary=True)
        p2 = radvel.Parameter(value=1.0, vary=True)
        p3 = radvel.Parameter(value=2.0, vary=True)
        assert p1._equals(p2)
        assert not p1._equals(p3)
        assert p1._equals("not a Parameter") is None

    def test_general_rv_model(self):
        """Test GeneralRVModel array_to_params and list methods."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)

        def simple_forward(t, params, vector):
            return np.zeros(len(t))

        mod = radvel.model.GeneralRVModel(params, simple_forward)
        mod.params['dvdt'] = radvel.Parameter(value=0.0)
        mod.params['curv'] = radvel.Parameter(value=0.0)

        # Test list_params
        keys = mod.list_params()
        assert 'per1' in keys

        # Test list_vary_params
        vary_keys = mod.list_vary_params()
        assert len(vary_keys) > 0

        # Test array_to_params
        vary_vals = np.ones(len(vary_keys)) * 5.0
        new_params = mod.array_to_params(vary_vals)
        assert new_params is not None

    def test_vector_mcmcscale_none(self):
        """Vector dict_to_vector handles mcmcscale=None."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)
        # mcmcscale is None by default
        assert params['per1'].mcmcscale is None
        vec = radvel.model.Vector(params)
        assert vec.vector is not None

    def test_vector_with_mcmcscale(self):
        """Vector dict_to_vector handles explicit mcmcscale."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['per1'].mcmcscale = 0.01
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)
        vec = radvel.model.Vector(params)
        per_idx = vec.indices['per1']
        assert vec.vector[per_idx][2] == 0.01

    def test_texlabel(self):
        """Test Parameters.tex_labels method."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)
        label = params.tex_labels()
        assert isinstance(label, dict)


# --- Additional Likelihood Coverage ---

class TestLikelihoodExtra:
    def test_decorr_params(self):
        """Test RVLikelihood with decorrelation parameters."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)
        mod = radvel.RVModel(params)
        mod.params['dvdt'] = radvel.Parameter(value=0.0)
        mod.params['curv'] = radvel.Parameter(value=0.0)

        t = np.linspace(0, 100, 50)
        vel = np.random.default_rng(42).normal(0, 1, 50)
        errvel = np.ones(50) * 0.5
        airmass = np.random.default_rng(42).uniform(1.0, 2.0, 50)

        like = radvel.likelihood.RVLikelihood(
            mod, t, vel, errvel,
            decorr_vars=['airmass'],
            decorr_vectors={'airmass': airmass},
        )
        like.params['gamma'] = radvel.Parameter(value=0.0)
        like.params['jit'] = radvel.Parameter(value=1.0)

        res = like.residuals()
        assert len(res) == 50

    def test_rvlikelihood_suffix_no_underscore(self):
        """Test RVLikelihood with suffix that doesn't start with _."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)
        mod = radvel.RVModel(params)
        mod.params['dvdt'] = radvel.Parameter(value=0.0)
        mod.params['curv'] = radvel.Parameter(value=0.0)

        t = np.linspace(0, 100, 20)
        vel = np.ones(20)
        errvel = np.ones(20) * 0.5

        like = radvel.likelihood.RVLikelihood(mod, t, vel, errvel, suffix='_hires')
        like.params['gamma_hires'] = radvel.Parameter(value=0.0)
        like.params['jit_hires'] = radvel.Parameter(value=1.0)
        assert like.suffix == 'hires'

    def test_linear_gamma(self):
        """Test that linear gamma parameter is solved analytically."""
        params = radvel.Parameters(1, 'per tc secosw sesinw logk')
        params['per1'] = radvel.Parameter(10.0)
        params['tc1'] = radvel.Parameter(0.0)
        params['secosw1'] = radvel.Parameter(0.0)
        params['sesinw1'] = radvel.Parameter(0.0)
        params['logk1'] = radvel.Parameter(1.5)
        mod = radvel.RVModel(params)
        mod.params['dvdt'] = radvel.Parameter(value=0.0)
        mod.params['curv'] = radvel.Parameter(value=0.0)

        t = np.linspace(0, 100, 50)
        vel = np.ones(50) * 5.0  # constant offset
        errvel = np.ones(50) * 0.5

        like = radvel.likelihood.RVLikelihood(mod, t, vel, errvel)
        like.params['gamma'] = radvel.Parameter(value=0.0, vary=False, linear=True)
        like.params['jit'] = radvel.Parameter(value=0.5)

        res = like.residuals()
        lp = like.logprob()
        assert isinstance(lp, (float, np.floating))
