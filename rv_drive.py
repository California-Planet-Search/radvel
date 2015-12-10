#!/usr/bin/env python

from numpy import *
from kepler import kepler

def rv_drive(t, orbel, quiet=True, nplanets=1,T0_trend=None):
    """
    # Calculates barycentric velocities of an SB1 from the orbital
    # parameters.  For multiple planets, simply adds Keplerians.

    #orbel is a 6n-element vector containing P,tp,e,om,k,gamma
    #units are days, days, dimensionless, radians , m/s, m/s
    
    #Ported to Python by BJ Fulton (09/2012)

    """
   
    rv = t * 0
    try:
        neo = orbel.size
    except AttributeError:
        orbel = array(orbel)
        neo = orbel.size
    

    pp = 5
    #n = neo / pp #number of planets
    #print "RV inputs for %d planets" % n
    n = nplanets

    #print neo, pp
    if neo < pp:   
        print 'Bad inputs to rv_drive, too few parameters.'
        return
   
    for i in range(n):
   
        p = orbel[0 + i * pp]
        tp = orbel[1 + i * pp]
        e = orbel[2 + i * pp]
        om = orbel[3 + i * pp]
        k = orbel[4 + i * pp]
        try:
            gamma = orbel[5 + i * pp]
        except:
            print "Error", orbel, i, pp, nplanets
        dvdt = orbel[-3]
        curv = orbel[-2]
        if n > 1 and i < n-1: gamma,dvdt,curv = 0.0,0.0,0.0

        #if (i == 0) and (n * pp == neo - 1):   
            #print "RV_DRIVE: setting curve = ", orbel[-1]
            #curv = orbel[-1]
      
        if not quiet and (p < 0 or e < 0 or e >= 1 or k < 0):   
            print 'Bad inputs to rv_drive, out of range'
            rv = zeros(len(t))

        #Error checking
        #if p < 0:   
            #p = 1e-2
        #if e < 0:   
            #e = 0
        #if e >= 1:   
            #e = 0.99
        #if k < 0:   
            #k = 1e-2      
      
        #Calculate the approximate eccentric anomaly, E1, via the mean
        # anomaly, M.
        m = 2. * pi * (((t - tp) / p) - floor((t - tp) / p))
        if e == 0: e1 = m
        else: e1 = kepler(m, e)
        
        #Calculate nu
        n1 = 1. + e
        n2 = 1. - e
        nu = 2. * arctan((n1 / n2) ** 0.5 * tan(e1 / 2.e0))

        #Calculate the radial velocity
        #T0_trend=2456000.
        if T0_trend == None: T0_trend = round((max(t) - min(t))/2.) + round(min(t))
        #print t[0] - 2454000
        if not quiet: print i,p,tp,e,om,k,gamma,dvdt,curv #, T0_trend
        rv = rv + k * (cos(nu + om) + e * cos(om)) + gamma + dvdt * (t-T0_trend) + curv * (t-T0_trend) ** 2  #Default epoch for planet search epoch
    
    return rv



if __name__ == '__main__':
    import numpy as np
    P = 10.2
    tp = 2454100.0
    e = 0.5
    om = radians(100.0)
    k = 50.0
    gamma = 0.0

    orbel = array([P,tp,e,om,k])
    t = arange(2454000.0,2455000.0,0.997)
    t = t[array(np.round(random.uniform(0,size=len(t))-0.2),dtype=bool)]

    P = 300.8
    tp = 2454150.000
    e = 0.1
    om = radians(200.0)
    k = 10.0

    gamma = 0.0
    dvdt = 0.0
    curv = 0.0
    jitter = 0.0

    orbel = concatenate((orbel,array([P,tp,e,om,k])))
    #t = linspace(2454000.0,2455000.0,0.997)

    orbel = append(orbel,[gamma,dvdt,curv,jitter])

    rv = rv_drive(t,orbel)

    rv += random.normal(size=len(rv)) * 5.0
    rverr = ones(len(rv)) * 3.0

    print '%s' % '_'.join(array(orbel,dtype=str))

    savetxt('/Users/bfulton/code/r-m/data/fake_%s.txt' % '%s' % '_'.join(array(orbel,dtype=str)),transpose((t-2440000.,rv,rverr)))

    import pylab as pl
    pl.plot(t,rv,'k.-')
    pl.xlabel('Time (Days)')
    pl.ylabel('RV (m/s)')
    pl.show()

