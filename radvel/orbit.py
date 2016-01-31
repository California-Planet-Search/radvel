import numpy as np
from scipy.optimize import  brentq
from astropy.time import Time
from astropy import constants as c
from astropy import units as u
def timetrans_to_timeperi(tt, P, e, w):
    """
    Convert Time of Transit to Time of Periastron Passage

    Parameters
    ----------
    tt : time of transit
    P : period [days]
    e : eccecntricity
    w : longitude of peri (radians)
    
    Returns
    -------
    tp : time of periastron passage. After tt
    """

    # Angular distance between true anomaly and peri as a function of time
    diff = lambda t : true_anomaly(t, P, e) + w - 2*np.pi
    a = 0
    b = (1-1e-9) * P
    t_diff = brentq(diff,a,b) # time of transit occurs t_diff after tp
    tp = tt - t_diff + P
    return tp

def true_anomaly(t, P, e):
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
    # Given a reference epoch and a period, compute the next times
    # this period and epoch will occur
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

    Parameters 
    ----------
    Msini : mass of planet [Mjup]
    P : Orbital period [days]
    Mtotal : Mass of star + mass of planet [Msun]
    e : eccentricity

    Returns
    -------
    K : Doppler semi-amplitude [m/s]
    """
    K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**-0.33 * Mtotal**-0.66
    return K

def Msini(K, P, Mtotal, e):
    """
    Parameters
    ----------
    K : m/s
    P : Orbital period [days]
    Mstar : Mass of star [Msun]
    e : eccentricity

    Returns
    -------
    Msini : Jupiter masses
    """
    Msini = K / K_0 * np.sqrt(1.0 - e**2.0) * Mtotal**0.66 * (P/365.0)**0.33
    return Msini

def density(mass,radius):
    """
    mass : mass in earth masses
    radius : radius in earth radii
    
    Returns
    -------
    rho : density (g/cc)
    """
    mass = np.array(mass)
    radius = np.array(radius)
    vol = 4./3.*np.pi * (radius * c.R_earth)**3
    rho = ((mass * c.M_earth / vol).to(u.g / u.cm**3)).value
    return rho

def Lstar(Rstar,Teff):
    """
    Rstar : Radius (solar units)
    Teff : Teff (K)
    
    Returns
    -------
    Lstar : Luminosity (solar units)
    """
    return (Rstar)**2*(Teff/5770)**4

def Sinc(Lstar,A):
    """
    Lstar : Luminosity (solar-units)
    A : Semi-major axis (AU)
    """
    Sinc = Lstar / A**2
    return Sinc

def Teq(Sinc):
    """
    Sinc : Insolation (Earth-units)
    """
    Sinc = np.array(Sinc)
    Se = 1300*u.W*u.m**-2
    Teq = (Sinc*Se / 4 / c.sigma_sb)**0.25 # Kelvin
    Teq = Teq.to(u.K).value
    return Teq
         
