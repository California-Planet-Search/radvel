
import numpy as np
import radvel


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
        if ecc >= 1:
            return tc
    except ValueError:
        pass
    
    f = np.pi/2 - omega
    EE = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
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


def true_anomaly(t, tp, per, e):
    """
    Calculate the true anomoly for a given time, period, eccentricity.

    Args:
        t (array): array of times in JD
        tp (float): time of periastron, same units as t
        per (float): orbital period in days
        e (float): eccentricity

    Returns:
        array: true anomoly at each time
    """

    # f in Murray and Dermott p. 27
    M = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros(t.size) + e
    E1 = radvel.kepler.kepler(M, eccarr)
    n1 = 1.0 + e
    n2 = 1.0 - e
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(E1 / 2.0))

    return nu
