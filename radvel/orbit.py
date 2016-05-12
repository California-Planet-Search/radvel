import numpy as np
from scipy.optimize import  brentq
from kepler import kepler
from astropy.time import Time
from astropy import constants as c
from astropy import units as u

def timetrans_to_timeperi(tc, per, ecc, w):
    """
    Convert Time of Transit to Time of Periastron Passage

    :param tc: time of transit
    :type tc: float
    
    :param per: period [days]
    :type per: float
    
    :param ecc: eccecntricity
    :type ecc: float
    
    :param w: longitude of periastron (radians)
    :type w: float

    
    :return: time of periastron passage

    """

    # Angular distance between true anomaly and peri as a function of time
    diff = lambda t : true_anomaly(t, per, ecc) + w - 2*np.pi
    a = 0
    b = (1-1e-9) * per
    t_diff = brentq(diff,a,b) # time of transit occurs t_diff after tp
    tp = tc - t_diff + per
    return tp


def timeperi_to_timetrans(tp, per, ecc, omega, secondary=0):
    """
    Convert Time of Periastron to Time of Transit


    :param tp: time of periastron
    :type tp: float
    
    :param P: period [days]
    :type P: float
    
    :param e: eccentricity
    :type e: float
    
    :param w: argument of peri (radians)
    :type w: float


    :return: time of inferior conjuntion (time of transit if system is transiting)
    
    """

    if secondary:
        f = 3*np.pi/2 - omega/360*2*np.pi                      # true anomaly during secondary eclipse
    else:
        f = np.pi/2   - omega/360*2*np.pi                      # true anomaly during transit
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



# Normalization. RV m/s of a 1.0 Jupiter mass planet tugging on a 1.0
# Jupiter mass star on a 1.0 year orbital period
K_0 = 28.4329 

def semi_amplitude(Msini, P, Mtotal, e):
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


    :return: Doppler semi-amplitude [m/s]
    """
    K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**-0.33 * Mtotal**-0.66
    return K

def Msini(K, P, Mtotal, e):
    """Calculate Msini

    Calculate Msini for a given K, P, stellar mass, and e
    
    Args:
        K (float): Doppler semi-amplitude [m/s]
        P (float): Orbital period [days]
        Mstar (float): Mass of star [Msun]
        e (float): eccentricity
    Returns:
        float: Msini, Jupiter masses
    
    """
    Msini = K / K_0 * np.sqrt(1.0 - e**2.0) * Mtotal**0.66 * (P/365.0)**0.33
    return Msini

def density(mass,radius):
    """
    :param mass: mass in Earth masses
    :type mass: float

    :param radius: radius in Earth radii
    :type radius: float

    :return: density (g/cc)
    """
    mass = np.array(mass)
    radius = np.array(radius)
    vol = 4./3.*np.pi * (radius * c.R_earth)**3
    rho = ((mass * c.M_earth / vol).to(u.g / u.cm**3)).value
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
         
