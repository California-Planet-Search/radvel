def kepler(inbigM, inecc):
    #import numpy as np
    from numpy import sign,cos,sin,sum
    Marr = inbigM  # protect inputs; necessary?
    eccarr = inecc
    conv = 1e-12  # convergence criterion
    k = 0.85

    Earr = Marr + sign(sin(Marr)) * k * eccarr  # first guess at E
    fiarr = (Earr-eccarr*sin(Earr)-Marr)  # should go to zero when converges
    convd = abs(fiarr) > conv  # which indices have not  converged
    nd = sum(convd == True) # number of converged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1
        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*sin(E)-M    ; should go to 0
        fip = 1-ecc*cos(E) # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc*sin(E) # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1-fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        d1 = -fi/fip #first  order correction to E
        d2 = -fi/(fip+d1*fipp/2.) #second order correction to E
        d3 = -fi/(fip+d2*fipp/2.+d2*d2*fippp/6.) #third  order correction to E
        E = E + d3
        Earr[convd] = E
        fiarr = (Earr-eccarr*sin(Earr)-Marr)  #how well did we do?
        convd = abs(fiarr) > conv, nd            #test for convergence
        nd = sum(convd == True)
    if Earr.size > 1: return Earr
    else: return Earr[0]

def rv_drive(t, orbel, time_base=14000, basis='cps'):
    """
    RV Drive

    Parameters
    ----------
    t : time
    orbel : orbital elements either in the 
            cps basis: [P, tp, e, om, K, gamma, dvdt, curv]

    Note
    ----
    Omega is expected to be in degrees
    """

    from numpy import zeros,pi,floor,arctan,tan,cos
    #from kepler import *
    if len(orbel) == 6: # in case the trend isn't specified
        neworbel = zeros(7)
        neworbel[0:6] = orbel
        orbel = neworbel
    rv = t*0
    neo = len(orbel)
    npl = neo/7  # number of planets
    nt = len(t)  # number of data points

    for i in range(npl):
        p, tp, e, om, k, gamma, dvdt = orbel[i*7:i*7+7]
        if i > 0: gamma, dvdt = 0,0
        om = om / 180 * pi
        curv  = 0
        if i == 0 and neo/7*7 == neo-1: curv = orbel[neo-1]

        # Error checking
        if p < 0: p = 1e-4
        if e < 0: e = 0
        if e > 1: e = 0.999

        # Calculate the approximate eccentric anomaly, E1, via the mean anomaly, M.
        M = 2*pi*( ((t-tp)/p) - floor((t-tp)/p) )
        eccarr = zeros(nt)+e
        E1 = kepler(M, eccarr)

        # Calculate nu
        nu = 2*arctan(((1+e)/(1-e))**0.5*tan(E1/2))
        
        # Calculate the radial velocity
        rv += k*(cos(nu+om)+e*cos(om))+gamma+dvdt*(t-time_base)+curv*(t-time_base)**2

    return rv
