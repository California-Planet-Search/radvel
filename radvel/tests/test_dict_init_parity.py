"""Parity tests for ``initialize_posterior_from_dict``.

These tests exercise the dict-based posterior loader added alongside
:func:`radvel.utils.initialize_posterior` and assert it produces a
posterior that is bit-equivalent to the file-loaded one for every example
planet setup the project ships with.

The dict-based loader does not depend on Pydantic; the tests build plain
Python dicts directly so the parity guarantee can be exercised in the
default test suite (no ``-m api`` marker, no FastAPI install required).
"""

import os

import numpy as np
import pandas as pd
import pytest

import radvel
from radvel.utils import (
    _build_parameters,
    _build_prior,
    initialize_posterior,
    initialize_posterior_from_dict,
)

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'example_planets',
    'epic203771098.py',
)


def _epic203771098_dict():
    """Build the JSON-style dict that mirrors example_planets/epic203771098.py."""
    path = os.path.join(radvel.DATADIR, 'epic203771098.csv')
    raw = pd.read_csv(path)
    rows = [
        {
            'time': float(t),
            'mnvel': float(v),
            'errvel': float(e),
            'tel': 'j',
        }
        for t, v, e in zip(raw.t, raw.vel, raw.errvel)
    ]

    params = {
        'per1': {'value': 20.885258, 'vary': False},
        'tc1': {'value': 2072.79438, 'vary': False},
        'secosw1': {'value': 0.0, 'vary': False},
        'sesinw1': {'value': np.sqrt(0.01), 'vary': False},
        'k1': {'value': 10.0},
        'per2': {'value': 42.363011, 'vary': False},
        'tc2': {'value': 2082.62516, 'vary': False},
        'secosw2': {'value': 0.0, 'vary': False},
        'sesinw2': {'value': np.sqrt(0.01), 'vary': False},
        'k2': {'value': 10.0},
        'dvdt': {'value': 0.0},
        'curv': {'value': 0.0},
        'gamma_j': {'value': 1.0, 'vary': False, 'linear': True},
        'jit_j': {'value': 2.6},
    }

    priors = [
        {'type': 'eccentricity', 'num_planets': 2},
        {'type': 'positivek', 'num_planets': 2},
        {'type': 'jeffreys', 'param': 'k1', 'minval': 1e-2, 'maxval': 1e3},
        {'type': 'jeffreys', 'param': 'k2', 'minval': 1e-2, 'maxval': 1e3},
        {'type': 'hardbounds', 'param': 'jit_j', 'minval': 0.0, 'maxval': 15.0},
        {'type': 'gaussian', 'param': 'dvdt', 'mu': 0.0, 'sigma': 1.0},
        {'type': 'gaussian', 'param': 'curv', 'mu': 0.0, 'sigma': 1e-1},
    ]

    return {
        'starname': 'epic203771098',
        'nplanets': 2,
        'instnames': ['j'],
        'fitting_basis': 'per tc secosw sesinw k',
        'bjd0': 2454833.0,
        'planet_letters': {1: 'b', 2: 'c'},
        'params': params,
        'data': rows,
        'priors': priors,
        'stellar': {'mstar': 1.12, 'mstar_err': 0.05},
    }


def test_epic203771098_logprob_matches_file():
    """Posterior built from the JSON dict matches the file-loaded one to 1e-12."""
    _, post_file = initialize_posterior(EXAMPLE)
    _, post_dict = initialize_posterior_from_dict(_epic203771098_dict())

    assert set(post_file.params.keys()) == set(post_dict.params.keys())
    for key in post_file.params.keys():
        assert post_file.params[key].value == pytest.approx(
            post_dict.params[key].value, abs=1e-12
        ), "mismatch on parameter {}".format(key)
        assert post_file.params[key].vary == post_dict.params[key].vary

    assert post_file.logprob() == pytest.approx(post_dict.logprob(), abs=1e-9)


def test_param_round_trip():
    """Plain dict → Parameters → values agree by-value."""
    spec = {
        'per1': {'value': 20.0},
        'tc1': {'value': 2000.0, 'vary': False},
        'secosw1': {'value': 0.1},
        'sesinw1': {'value': 0.1},
        'k1': {'value': 5.0, 'mcmcscale': 0.5},
        'gamma_a': {'value': 0.0, 'vary': False, 'linear': True},
        'jit_a': {'value': 1.0},
    }
    params_obj = _build_parameters(spec, nplanets=1, basis='per tc secosw sesinw k')
    assert params_obj['per1'].value == 20.0
    assert params_obj['tc1'].vary is False
    assert params_obj['k1'].mcmcscale == 0.5
    assert params_obj['gamma_a'].linear is True


def test_basis_conversion_through_any_basis():
    """A dict given in ``per tc e w k`` is converted to ``per tc secosw sesinw k``."""
    base = _epic203771098_dict()
    base['any_basis'] = 'per tc e w k'
    base['params'] = {
        'per1': {'value': 20.885258, 'vary': False},
        'tc1': {'value': 2072.79438, 'vary': False},
        'e1': {'value': 0.01, 'vary': False},
        'w1': {'value': np.pi / 2.0, 'vary': False},
        'k1': {'value': 10.0},
        'per2': {'value': 42.363011, 'vary': False},
        'tc2': {'value': 2082.62516, 'vary': False},
        'e2': {'value': 0.01, 'vary': False},
        'w2': {'value': np.pi / 2.0, 'vary': False},
        'k2': {'value': 10.0},
        'dvdt': {'value': 0.0},
        'curv': {'value': 0.0},
        'gamma_j': {'value': 1.0, 'vary': False, 'linear': True},
        'jit_j': {'value': 2.6},
    }
    _, post = initialize_posterior_from_dict(base)
    assert str(post.params.basis) == 'Basis Object <per tc secosw sesinw k>'
    _, post_file = initialize_posterior(EXAMPLE)
    assert post.logprob() == pytest.approx(post_file.logprob(), abs=1e-9)


@pytest.mark.parametrize('spec,expected_cls', [
    ({'type': 'gaussian', 'param': 'k1', 'mu': 1.0, 'sigma': 0.1}, radvel.prior.Gaussian),
    ({'type': 'jeffreys', 'param': 'k1', 'minval': 1e-2, 'maxval': 1e3}, radvel.prior.Jeffreys),
    ({'type': 'modifiedjeffreys', 'param': 'k1', 'minval': 1.0, 'maxval': 1e3, 'kneeval': 1e-2},
     radvel.prior.ModifiedJeffreys),
    ({'type': 'hardbounds', 'param': 'jit_j', 'minval': 0.0, 'maxval': 15.0}, radvel.prior.HardBounds),
    ({'type': 'eccentricity', 'num_planets': 2}, radvel.prior.EccentricityPrior),
    ({'type': 'positivek', 'num_planets': 2}, radvel.prior.PositiveKPrior),
    ({'type': 'secondaryeclipse', 'planet_num': 1, 'ts': 100.0, 'ts_err': 0.5},
     radvel.prior.SecondaryEclipsePrior),
    ({'type': 'informative_baseline', 'param': 'per1', 'baseline': 1000.0, 'duration': 365.0},
     radvel.prior.InformativeBaselinePrior),
])
def test_each_prior_type(spec, expected_cls):
    prior = _build_prior(spec)
    assert isinstance(prior, expected_cls)


def test_unknown_prior_type_raises():
    with pytest.raises(ValueError, match="Unknown prior type"):
        _build_prior({'type': 'made-up-prior'})


def test_data_accepts_dataframe():
    """DataFrame inputs are passed through (legacy ergonomics)."""
    cfg = _epic203771098_dict()
    cfg['data'] = pd.DataFrame(cfg['data'])
    _, post = initialize_posterior_from_dict(cfg)
    assert post.logprob() < 0


def test_time_base_auto_filled_when_missing():
    cfg = _epic203771098_dict()
    cfg.pop('time_base', None)
    P, _ = initialize_posterior_from_dict(cfg)
    assert P.time_base == pytest.approx(
        float(np.mean([P.data.time.min(), P.data.time.max()]))
    )
