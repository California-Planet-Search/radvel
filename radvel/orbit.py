import numpy as np
import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo


def timetrans_to_timeperi(tc, per, ecc, omega, eval=True):
    """
    Convert Time of Transit to Time of Periastron Passage

    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)
        eval (Boolean): if true, normal value will be returned,
            if false, will be returned as theano tensor

    Returns:
        float: time of periastron passage

    """
    tc = tt.as_tensor_variable(tc)
    per = tt.as_tensor_variable(per)
    ecc = tt.as_tensor_variable(ecc)
    omega = tt.as_tensor_variable(omega)

    try:
        if tt.switch(tt.ge(ecc, 1), 1, 0).eval() == 1:
            if eval:
                return float(tc.eval())
            else:
                return tc
    except ValueError:
        pass

    f = np.pi/2 - omega
    ee = 2 * tt.arctan(np.tan(f/2) * tt.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (ee - ecc*tt.sin(ee))      # time of periastron

    if eval:
        return float(tp.eval())
    else:
        return tp


def timeperi_to_timetrans(tp, per, ecc, omega, secondary=False, eval=True):
    """
    Convert Time of Periastron to Time of Transit

    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)
        secondary (bool): calculate time of secondary eclipse instead
        eval (Boolean): if true, normal value will be returned,
            if false, will be returned as theano tensor

    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)

    """
    tp = tt.as_tensor_variable(tp)
    per = tt.as_tensor_variable(per)
    ecc = tt.as_tensor_variable(ecc)
    omega = tt.as_tensor_variable(omega)

    try:
        if tt.switch(tt.ge(ecc, 1), 1, 0).eval() == 1:
            if eval:
                return float(tp.eval())
            else:
                return tp
    except ValueError:
        pass

    if secondary:
        f = 3 * np.pi / 2 - omega  # true anomaly during secondary eclipse
        ee = 2 * tt.arctan(tt.tan(f / 2) * tt.sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        ee = np.array([ee.eval()])
        if len(ee) == 1:
            ee = tt.as_tensor_variable(ee + 2 * np.pi)
        else:
            ee[0 > ee] = ee + 2 * np.pi
            ee = tt.as_tensor_variable(ee)
    else:
        f = np.pi / 2 - omega  # true anomaly during transit
        ee = 2 * tt.arctan(tt.tan(f / 2) * tt.sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly

    tc = tp + per / (2 * np.pi) * (ee - ecc * tt.sin(ee))  # time of conjunction

    if eval:
        return float(tc.eval())
    else:
        return tc


def true_anomaly(t,e,tp,per,w,eval=True):
    """
    Calculate the true anomaly for a given time, period, eccentricity.

    Args:
        t (array): array of times in JD
        e (float): eccentricity
        tp (float): time of periastron, same units as t
        per (float): orbital period in days
        w (float): omega
        eval (Boolean): if true, normal value will be returned,
            if false, will be returned as theano tensor

    Returns:
        array: true anomaly at each time
    """

    n = tt.as_tensor_variable(2 * np.pi) / per
    ecc = tt.as_tensor_variable(e)
    omega = tt.as_tensor_variable(w)
    cos_omega = tt.cos(omega)
    sin_omega = tt.sin(omega)
    opsw = 1 + sin_omega
    E0 = 2 * tt.arctan2(tt.sqrt(1 - ecc) * cos_omega, tt.sqrt(1 + ecc) * opsw)
    M0 = E0 - ecc * tt.sin(E0)
    t_periastron = tt.as_tensor_variable(tp)
    t0 = t_periastron + M0 / n
    tref = t_periastron - t0
    warp_times = tt.shape_padright(t) - t0
    M = (warp_times - tref) * n
    ec = ecc + tt.zeros_like(M)

    if eval:
        return xo.orbits.get_true_anomaly(M, ec).eval()
    else:
        return xo.orbits.get_true_anomaly(M, ec)
