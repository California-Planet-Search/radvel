import copy
from pathlib import Path
import sys
from types import SimpleNamespace

from radvel import cli, driver, fitting, posterior


def test_cli_ic_holds_jitter_at_the_map_value(tmp_path, monkeypatch):
    setup = Path(__file__).resolve().parents[2] / "example_planets" / "epic203771098.py"
    driver.fit(SimpleNamespace(setupfn=str(setup), outputdir=str(tmp_path), decorr=False))
    fitted = posterior.load(str(tmp_path / "epic203771098_post_obj.pkl"))

    reference = copy.deepcopy(fitted)
    jitter_index = reference.vector.indices["jit_j"]
    nominal_jitter = reference.vector.vector[jitter_index, 0]
    reference.vector.vector[jitter_index, 1] = False
    # The trend comparison starts its full-model refit with zero curvature.
    reference.vector.vector[reference.vector.indices["curv"], 0] = 0
    reference.list_vary_params()
    reference.likelihood.list_vary_params()
    fitting.maxlike_fitting(reference, verbose=False)
    assert reference.vector.vector[jitter_index, 0] == nominal_jitter
    expected_bic = round(reference.likelihood.bic(), 2)

    results = {}
    monkeypatch.setattr(driver, "save_status", lambda path, section, data: results.update(data))
    args = ["radvel", "ic", "-s", str(setup), "-d", str(tmp_path), "-t", "trend"]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert [row[r"$N_{\rm free}$"][0] for row in results["ic"]] == [5, 4, 3]

    monkeypatch.setattr(sys, "argv", args + ["--fixjitter"])
    cli.main()
    assert [row[r"$N_{\rm free}$"][0] for row in results["ic"]] == [4, 3, 2]
    assert results["ic"][0]["BIC"][0] == expected_bic
