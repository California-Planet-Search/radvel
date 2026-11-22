from pathlib import Path

import numpy as np

from radvel import RVModel
from radvel.utils import initialize_posterior


def test_converting_parameters_preserves_the_source_rv_model():
    setup = Path(__file__).resolve().parents[2] / "example_planets" / "HD164922.py"
    _, post = initialize_posterior(setup)
    model = post.likelihood.model
    params = model.params
    times = post.likelihood.x
    values = model(times)
    source_basis = params.basis.name

    converted = params.basis.to_any_basis(params, "logper tc secosw sesinw logk")
    # As in the shipped setup files, set the fitting flags after conversion.
    for param in converted.values():
        if param.vary == "":
            param.vary = False
    converted_model = RVModel(converted, time_base=model.time_base)

    np.testing.assert_allclose(converted_model(times), values, atol=1e-11, rtol=0)
    np.testing.assert_array_equal(model(times), values)
    assert params.basis.name == source_basis
    assert params.basis.params == source_basis.split()
