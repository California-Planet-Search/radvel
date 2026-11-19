import numpy as np
import pytest

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
