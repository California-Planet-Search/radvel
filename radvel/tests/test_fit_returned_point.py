from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from radvel.fitting import maxlike_fitting
from radvel.utils import initialize_posterior


def test_maxlike_fitting_keeps_optimizer_solution():
    setup = Path(__file__).resolve().parents[2] / "example_planets" / "epic203771098.py"
    _, reference = initialize_posterior(setup)
    _, post = initialize_posterior(setup)
    # Use the shipped transit ephemerides and fix the trend, fitting the two
    # semi-amplitudes and jitter to the published RVs.
    free = {"k1", "k2", "jit_j"}
    for candidate in (reference, post):
        for name, param in candidate.params.items():
            param.vary = name in free
        candidate.vector.dict_to_vector()
        candidate.list_vary_params()
        candidate.likelihood.list_vary_params()

    result = minimize(
        reference.neglogprob_array, reference.get_vary_params(),
        method="Nelder-Mead", options={"maxiter": 200, "maxfev": 100000},
    )
    assert result.success, result.message
    reference.neglogprob_array(result.x)

    maxlike_fitting(post, verbose=False, method="Nelder-Mead")

    np.testing.assert_array_equal(post.get_vary_params(), result.x)
    np.testing.assert_array_equal(
        [post.params[name].value for name in post.name_vary_params()], result.x,
    )
    assert post.params["gamma_j"].value == reference.vector.vector[
        reference.vector.indices["gamma_j"], 0
    ]
    assert post.logprob() == -result.fun
