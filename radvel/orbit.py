import numpy as np
from scipy.optimize import  brentq
from kepler import kepler
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



def next_epoch(P, epoch, now, nnext=10):
    """
    Given a reference epoch and a period, compute the next times this period and epoch will occur
    """
    
    phase_now = np.mod((now - epoch) / P,1)
    i = np.arange(1,nnext+1)
    time_from_now = (i - phase_now) * P
    time = now + time_from_now
    return time


def sqrtecosom_sqrtesinom_to_e_om(sqrtecosom,sqrtesinom):
    e = sqrtecosom**2 + sqrtesinom**2
    om = np.arctan2(sqrtesinom,sqrtecosom) * 360 / 2 / np.pi
    om = np.mod(om + 360, 360) # omega is positive
    return e,om

def e_om_to_ecosom_esinom(e,om):
    dtorad = 2 * np.pi / 360.
    ecosom = e * np.cos( dtorad*om )
    esinom = e * np.sin( dtorad*om )
    return ecosom, esinom

def semi_amplitude(Msini, P, Mtotal, e, Msini_units='jupiter'):
    """
    Compute Doppler semi-amplitude

    :param Msini: mass of planet [Mjup]
    :type Msini: float
    
    :param P: Orbital period [days]
    :type P: float
    
    :param Mtotal: Mass of star + mass of planet [Msun]
    :type Mtotal: float
    
    :param e: eccentricity
    :type e: float

    :param Msini_units: Units of returned Msini. Must be 'earth', or 'jupiter' (default 'jupiter').
    :type Msini_units: string

    :return: Doppler semi-amplitude [m/s]
    """
    if Msini_units.lower() == 'jupiter':
        K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**(-1/3.) * \
            Mtotal**(-2/3.)
    elif Msini_units.lower() == 'earth':
        K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**(-1/3.) * \
            Mtotal**-(-2/3.)*(c.M_earth/c.M_jup).value
    else: 
        raise Exception("Msini_units must be 'earth', or 'jupiter'")
        
    return K


def Msini(K, P, Mtotal, e, Msini_units='earth'):
    """Calculate Msini

    Calculate Msini for a given K, P, stellar mass, and e
    
    Args:
        K (float): Doppler semi-amplitude [m/s]
        P (float): Orbital period [days]
        Mtotal (float): Mass of star + mass of planet [Msun]
        e (float): eccentricity
        Msini_units = (optional) Units of returned Msini. Must be 'earth', or 'jupiter' (default 'earth'). 
    Returns:
        float: Msini [units = Msini_units]
    
    """

    if Msini_units.lower() == 'jupiter':
        Msini = K / K_0 * np.sqrt(1.0 - e**2.0) * Mtotal**(2/3.) * \
            (P/365.0)**(1/3.)
    elif Msini_units.lower() == 'earth':
        Msini = K / K_0 * np.sqrt(1.0 - e**2.0) * Mtotal**(2/3.) * \
            (P/365.0)**(1/3.)*(c.M_jup/c.M_earth).value
    else: 
        raise Exception("Msini_units must be 'earth', or 'jupiter'")
    
    return Msini


def density(mass,radius, MR_units='earth'):
    """
    :param mass: mass, units = MR_units 
    :type mass: float

    :param radius: radius, units = MR_units 
    :type radius: float

    :param MR_units: (optional) units of mass and radius. Must be 'earth', or 'jupiter' (default 'earth').

    :return: density (g/cc)
    """
    mass = np.array(mass)
    radius = np.array(radius)
    if MR_units.lower() == 'earth':
        vol = 4./3.*np.pi * (radius * c.R_earth)**3
        rho = ((mass * c.M_earth / vol).to(u.g / u.cm**3)).value
    elif MR_units.lower() == 'jupiter':
        vol = 4./3.*np.pi * (radius * c.R_jup)**3
        rho = ((mass * c.M_jup / vol).to(u.g / u.cm**3)).value
    else: 
        raise Exception("MR_units must be 'earth', or 'jupiter'")
    return rho


def Lstar(Rstar,Teff):
    """
    :param Rstar: Radius (solar units)
    :type Rstar: float
    
    :param Teff: Teff (K)
    :type Teff: float
    

    :return: Luminosity (solar units)
    """
    return (Rstar)**2*(Teff/5770)**4


def Sinc(Lstar,A):
    """
    :param Lstar: Luminosity (solar-units)
    :type Lstar: float
    
    :param A: Semi-major axis (AU)
    :type A: float


    :return: Insolation (Earth-units)
    """
    Sinc = Lstar / A**2
    return Sinc


def Teq(Sinc):
    """
    :param Sinc: Insolation (Earth-units)
    :type Sinc: float

    :return: Equilibrium temperature
    """
    Sinc = np.array(Sinc)
    Se = 1300*u.W*u.m**-2
    Teq = (Sinc*Se / 4 / c.sigma_sb)**0.25 # Kelvin
    Teq = Teq.to(u.K).value
    return Teq
         
