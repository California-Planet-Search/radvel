#!/usr/bin/env python

import numpy as np
from kepler import *

def calcnu(P, tp, e, t):
    #Calculates the true anomoly from the time and orbital elements.
    #P = period
    #tp = time of periastron passage
    #e = eccentricity
    #t = time
  
    #In Wright & Howard, nu is denoted by little 'f'.  Here we must distinguish from big F, a matrix, so we use nu

    #Ported to Python from rvlin.py by BJ Fulton (09/2012)

    phase = (t-tp)/P
    M = 2.*np.pi*((phase) - np.floor(phase))
    E1 = kepler(M, e)

    n1 = 1. + e
    n2 = 1. - e
    nu = 2.*np.arctan(np.sqrt(n1/n2)*np.tan(E1/2.))
  
    return nu

if __name__ == '__main__':
    import pylab as pl
    t = np.arange(50.)
    phase,nu = calcnu(3.5,0.0,0.5,t)
    print nu
    pl.plot(phase,nu,'k-')
    pl.show()
