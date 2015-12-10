#!/usr/bin/env python

#Solve Kepler's equation

#Iteratively solve for E (anomoly) given M (mean anomoly) and e (eccentricity)

#JTW Written at Berkeley 
#Adopted for TEDI Dec 2007
#From Murray & Dermott p. 35, from Danby (1988)

#Ported to Python by BJ Fulton (09/2012)

import numpy as np

def fan(x,nrows):
    try:
        y = np.ones((nrows,len(x)))
    except TypeError:
        y = np.ones((nrows,))
    new = x * y
    return new

def kepler(inbigM, inecc):

  try:
      Marr = inbigM.copy()
      nm = Marr.size
  except AttributeError:
      Marr = inbigM
      nm = 1
  try:
      eccarr = inecc.copy()
      nec = eccarr.size
  except AttributeError:
      eccarr = inecc
      nec = 1
  if nec == 1 and nm > 1: eccarr = fan(eccarr, nm)
  if nec > 1 and nm == 1: marr = fan(Marr, nec)
  conv = 1.e-12
  k = 0.85
  from restrict import restrict
  mphase = restrict(Marr/2/np.pi, 0, 1)
  #ssm = (mphase <= 0.5) - ((mphase >= 0.5) or (mphase == 0))*1# = sign(sin(Marr)), but faster
  ssm = (mphase <= 0.5) - (np.bitwise_or((mphase >= 0.5), (mphase == 0))) * 1
  Earr = Marr+ssm*k*eccarr  #first guess at E
  fiarr = (Earr-eccarr*np.sin(Earr)-Marr)  #E - e*sin(E)-M    # should go to 0 when converges
  convd = np.where(np.abs(fiarr) > conv)[0] #which indices have converged?
  nd = convd.size
  count = 0
  while nd > 0:                #while unconverged elements exist...
    count += 1
    M = Marr[convd]                     #just the unconveged elements, please...
    ecc = eccarr[convd]
    E = Earr[convd]

    fi = fiarr[convd]  #fi = E - e*sin(E)-M    # should go to 0
    fip = 1-ecc*np.cos(E) #d/dE(fi) #i.e.,  fi^(prime)
    fipp = ecc*np.sin(E)  #d/dE(d/dE(fi)) #i.e.,  fi^(\prime\prime)
    fippp = 1-fip #d/dE(d/dE(d/dE(fi))) #i.e.,  fi^(\prime\prime\prime)
    
    d1 = -fi/fip                             #first  order correction to E
    d2 = -fi/(fip+d1*fipp/2.)                #second order correction to E
    d3 = -fi/(fip+d2*fipp/2.+d2*d2*fippp/6.) #third  order correction to E
    E += d3
    Earr[convd] = E
    fiarr = (Earr-eccarr*np.sin(Earr)-Marr)     #how well did we do?
    convd = np.where(np.abs(fiarr) > conv)[0]         #test for convergence
    nd = convd.size
    if count > 30:
      print 'WARNING!  Keplers equation not solved!!!'
      nd = 0

  if inbigM.size == 1: return Earr[0]
  else: return Earr

