"""Conjunction times on the orbital cycle after periastron."""
import numpy as np
import pytest

from radvel.orbit import timeperi_to_timetrans


@pytest.mark.parametrize('omega, phase', [(0., .75), (np.pi / 2, .5), (np.pi, .25)])
def test_secondary_eclipse_scalar_cycle(omega, phase):
    for dtype in [float, np.float32, np.float64]:
        actual = timeperi_to_timetrans(dtype(100.), dtype(10.), dtype(0.),
                                       dtype(omega), secondary=True)
        assert np.ndim(actual) == 0
        assert actual == pytest.approx(100. + 10. * phase, abs=2e-6)


@pytest.mark.parametrize('omega', [np.array([0., np.pi]),
                                 np.array([[0., np.pi], [np.pi / 2, 2 * np.pi]]),
                                 np.array([np.pi, np.pi / 2]), np.array([])])
def test_secondary_eclipse_array_cycle(omega):
    expected = 10. * ((.75 - omega / (2 * np.pi)) % 1.)
    actual = timeperi_to_timetrans(0., 10., 0., omega, secondary=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual.shape == omega.shape


def test_secondary_eclipse_broadcast_cycle():
    tp = np.array([[100.], [200.]])
    per = np.array([[4.], [10.]])
    omega = np.array([0., np.pi / 2, np.pi])
    actual = timeperi_to_timetrans(tp, per, 0., omega, secondary=True)
    np.testing.assert_allclose(actual, tp + per * np.array([.75, .5, .25]))
