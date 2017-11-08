import numpy as np
from scipy.optimize import  brentq
from radvel.kepler import kepler
from astropy.time import Time
from astropy import constants as c
from astropy import units as u


# Normalization. 
#RV m/s of a 1.0 Jupiter mass planet tugging on a 1.0
# solar mass star on a 1.0 year orbital period
K_0 = 28.4329


def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

    Args:
        tc (float): time of transit    
        per (float): period [days]
        ecc (float): eccecntricity
        omega (float): longitude of periastron (radians)
    
    Returns:
        float: time of periastron passage

    """
    try:
        if ecc >= 1: return tc
    except ValueError:
        pass
    
    f = np.pi/2   - omega
    EE = 2 * np.arctan( np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)) )  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (EE - ecc*np.sin(EE))      # time of periastron
    
    return tp
    

def timeperi_to_timetrans(tp, per, ecc, omega, secondary=0):
    """
    Convert Time of Periastron to Time of Transit

    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)

    Returns:
        float: time of inferior conjuntion (time of transit if system is transiting)
    
    """
    try:
        if ecc >= 1: return tp
    except ValueError:
        pass
    
    if secondary:
        f = 3*np.pi/2 - omega                      # true anomaly during secondary eclipse
    else:
        f = np.pi/2   - omega                      # true anomaly during transit

    EE = 2 * np.arctan( np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)) )  # eccentric anomaly
    tc = tp + per/(2*np.pi) * (EE - ecc*np.sin(EE))         # time of conjunction

    return tc


def true_anomaly(t, P, e):
    """
    Calculate the true annomoly for a given time, period, eccentricity.

    :param t: time (BJD_TDB)
    :type t: float

    :param P: period [days]
    :type P: float

    :param e: eccentricity
    :type e: float


    :return: True Annomoly
    
    """

    
    # f in Murray and Dermott p. 27
    tp = 0
    t = np.array([t])
    m = 2. * np.pi * (((t - tp) / P) - np.floor((t - tp) / P))
    e1 = kepler(m, np.array([e]))
    n1 = 1.0 + e
    n2 = 1.0 - e
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.e0))
    if nu < 0:
        nu+=2*np.pi
    return nu
