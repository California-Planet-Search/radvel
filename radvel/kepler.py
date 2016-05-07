import numpy as np
from _kepler import kepler_array_cext

def rv_drive(t, orbel):
    """RV Drive
    
    Args:
        t (array of floats): times of observations
        orbel (array of floats): [per, tp, e, om, K]. Omega is expected to be
            in degrees

    Returns:
        rv: (array of floats): radial velocity
    
    """
    
    # unpack array
    per, tp, e, om, k = orbel
    om = om / 180 * np.pi
    curv  = 0
    
    # Error checking
    if per < 0: per = 1e-4
    if e < 0: e = 0
    if e > 1: e = 0.999

    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    M = 2 * np.pi * ( ((t - tp) / per) - np.floor( (t - tp) / per ) )
    eccarr = np.zeros(t.size) + e
    #E1 = kepler(M, eccarr)
    print orbel
    E1 = kepler_array_cext(M, e)
    
    # Calculate nu
    nu = 2 * np.arctan( ( (1+e) / (1-e) )**0.5 * np.tan( E1 / 2 ) )
    
    # Calculate the radial velocity
    rv = k * ( np.cos( nu + om ) + e * np.cos( om ) ) 
    return rv

def kepler(inbigM, inecc):
    """Kepler's Equation
    """
    
    Marr = inbigM  # protect inputs; necessary?
    eccarr = inecc
    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)  
    convd = np.abs(fiarr) > conv  # which indices have not converged
    nd = np.sum(convd == True) # number of converged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1
        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E) # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc * np.sin(E) # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip 
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp/ 2.0 + d2 * d2 * fippp / 6.0) 
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv, nd #test for convergence
        nd = np.sum(convd == True)

    if Earr.size > 1: 
        return Earr
    else: 
        return Earr[0]
