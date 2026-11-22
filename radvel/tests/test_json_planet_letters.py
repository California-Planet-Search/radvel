from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import radvel
from radvel.fitting import model_comp
from radvel.plot.mcmc_plots import DerivedPlot
from radvel.utils import initialize_posterior_from_dict
from test_dict_init_parity import _epic203771098_dict


@pytest.mark.parametrize("parameter_objects", [False, True])
def test_dict_setup_preserves_labels_for_reordered_planets(parameter_objects):
    setup = _epic203771098_dict()
    raw = setup["params"].copy()
    source, original = initialize_posterior_from_dict(setup)
    for key, value in raw.items():
        if key[-1] in ("1", "2"):
            setup["params"][key[:-1] + ("2" if key[-1] == "1" else "1")] = value
    if parameter_objects:
        _, reordered_unlabelled = initialize_posterior_from_dict(setup)
        setup["params"] = reordered_unlabelled.params
    setup["planet_letters"] = {"1": "c", "2": "b"}

    configured, reordered = initialize_posterior_from_dict(setup)

    times = source.data.time.to_numpy()
    np.testing.assert_allclose(
        reordered.model(times), original.model(times), atol=1e-12, rtol=0,
    )
    assert configured.planet_letters == {1: "c", 2: "b"}
    labels = original.params.tex_labels(["per1", "per2"])
    assert reordered.params.tex_labels(["per1", "per2"]) == {
        "per1": labels["per2"], "per2": labels["per1"],
    }
    if parameter_objects:
        assert setup["params"].planet_letters == {1: "b", 2: "c"}


DEFAULT_LABELS = {"per1": "$P_{b}$", "per2": "$P_{c}$"}


@pytest.mark.parametrize("planet_letters, expected", [
    ({}, ("b", "c")),
    ({"1": "b"}, ("b", "c")),
    ({"1": "d"}, ("d", "c")),
    ({"2": "d"}, ("b", "d")),
    ({"1": "c", "2": "b"}, ("c", "b")),
])
def test_dict_setup_partial_planet_letters_fall_back_to_defaults(
    planet_letters, expected,
):
    setup = _epic203771098_dict()
    setup["planet_letters"] = planet_letters

    namespace, post = initialize_posterior_from_dict(setup)

    assert post.params.tex_labels(["per1", "per2"]) == {
        "per1": "$P_{%s}$" % expected[0], "per2": "$P_{%s}$" % expected[1],
    }
    if not planet_letters:
        # An empty mapping is treated like None: nothing is configured.
        assert not hasattr(namespace, "planet_letters")
        assert post.params.planet_letters is None


def test_parameters_empty_planet_letters_use_defaults():
    params = radvel.Parameters(2, planet_letters={})

    assert params.tex_labels(["per1", "per2"]) == DEFAULT_LABELS


def test_model_comp_free_params_use_default_letter_for_missing_planet():
    setup = _epic203771098_dict()
    setup["planet_letters"] = {"1": "d"}
    _, post = initialize_posterior_from_dict(setup)

    # model_comp's default mc_list is shared between calls; pass a fresh one.
    free = model_comp(post, mc_list=[])[0]["Free Params"][0]

    assert "$K_{d}$" in free and "$K_{c}$" in free


@pytest.mark.parametrize("planet_letters, expected", [
    (None, ["$M_b\\sin i$", "$M_c\\sin i$"]),
    ({}, ["$M_b\\sin i$", "$M_c\\sin i$"]),
    ({1: "d"}, ["$M_d\\sin i$", "$M_c\\sin i$"]),
])
def test_derived_plot_labels_use_default_letter_for_missing_planet(
    planet_letters, expected,
):
    chains = pd.DataFrame({"mpsini1": [1.0, 2.0], "mpsini2": [3.0, 4.0]})
    config = SimpleNamespace(nplanets=2)
    if planet_letters is not None:
        config.planet_letters = planet_letters

    assert DerivedPlot(chains, config).texlabels == expected
