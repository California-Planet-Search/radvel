from pathlib import Path
import pickle

import numpy as np
import pytest

from radvel.likelihood import CeleriteLikelihood
from radvel.utils import initialize_posterior


def test_celerite_likelihood_handles_newest_first_observations():
    pytest.importorskip("celerite")
    setup = Path(__file__).resolve().parents[2] / "example_planets" / "k2-131_celerite.py"
    _, post = initialize_posterior(setup)
    for original in post.likelihood.like_list:
        like = CeleriteLikelihood(
            original.model, original.x[::-1], original.y[::-1],
            original.yerr[::-1], original.hnames, suffix="_" + original.suffix,
        )
        np.testing.assert_array_equal(like.x, original.x)
        np.testing.assert_array_equal(like.y, original.y)
        np.testing.assert_array_equal(like.yerr, original.yerr)

        # Dense evaluation of the documented celerite kernel for the same data.
        hparams = original.kernel.hparams
        amplitude, scale, decay, period = [
            hparams[name].value for name in ("gp_B", "gp_C", "gp_L", "gp_Prot")
        ]
        dt = np.abs(original.x[:, None] - original.x)
        covariance = amplitude / (2 + scale) * np.exp(-dt / decay) * (
            np.cos(2 * np.pi * dt / period) + 1 + scale
        )
        covariance[np.diag_indices_from(covariance)] += original.errorbars() ** 2
        gamma = original.vector.vector[original.gamma_index, 0]
        residuals = original.y - gamma - original.model(original.x)
        expected = -0.5 * (
            residuals @ np.linalg.solve(covariance, residuals)
            + np.linalg.slogdet(covariance)[1] + len(dt) * np.log(2 * np.pi)
        )
        assert original.logprob() == pytest.approx(expected, rel=0, abs=1e-8)
        assert like.logprob() == pytest.approx(expected, rel=0, abs=1e-8)

        # Older posteriors saved the input order on the kernel.
        like.kernel.compute_distances(like.x[::-1], like.x[::-1])
        restored = pickle.loads(pickle.dumps(like))
        assert restored.logprob() == pytest.approx(expected, rel=0, abs=1e-8)
