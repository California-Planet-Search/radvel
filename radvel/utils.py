import numpy as np
from numpy import *

def timebin(time, meas, meas_err, binsize):
#  This routine bins a set of times, measurements, and measurement errors 
#  into time bins.  All inputs and outputs should be floats or double. 
#  binsize should have the same units as the time array.
#  - from Andrew Howard, ported to Python by BJ Fulton

    ind_order = np.argsort(time)
    time = time[ind_order]
    meas = meas[ind_order]
    meas_err = meas_err[ind_order]

    ct=0
    while ct < len(time):
        ind = np.where((time >= time[ct]) & (time < time[ct]+binsize))[0]
        num = len(ind)
        wt = (1./meas_err[ind])**2.     #weights based in errors
        wt = wt/np.sum(wt)              #normalized weights
        if ct == 0:
            time_out     = [np.sum(wt*time[ind])]
	    meas_out     = [np.sum(wt*meas[ind])]
	    meas_err_out = [1./np.sqrt(np.sum(1./(meas_err[ind])**2))]
        else:
            time_out.append(np.sum(wt*time[ind]))
	    meas_out.append(np.sum(wt*meas[ind]))
	    meas_err_out.append(1./np.sqrt(np.sum(1./(meas_err[ind])**2)))
        ct += num

    return time_out, meas_out, meas_err_out


def bintels(t, vel, err, telvec, binsize=1/2.):
    # Bin RV data with bins of with binsize in the units of t.
    # Will not bin data from different telescopes together since there may
    # be offsets between them.

    ntels = len(np.unique(telvec))
    if ntels == 1:
        t_bin, vel_bin, err_bin = timebin(t, vel, err, binsize=binsize)
        return t_bin, vel_bin, err_bin, telvec
    
    uniqorder = np.argsort(np.unique(telvec,return_index=1)[1])
    uniqsort = np.unique(telvec)[uniqorder]
    rvtimes = np.array([])
    rvdat = np.array([])
    rverr = np.array([])
    newtelvec = np.array([])
    for i,tel in enumerate(uniqsort):
        pos = np.where(telvec == tel)
        t_bin, vel_bin, err_bin = timebin(t[pos], vel[pos], err[pos], binsize=binsize)
        rvtimes = np.hstack((rvtimes, t_bin))
        rvdat = np.hstack((rvdat, vel_bin))
        rverr = np.hstack((rverr, err_bin))
        newtelvec = np.hstack((newtelvec, np.array([tel]*len(t_bin))))
        
    return rvtimes, rvdat, rverr, newtelvec

def Tp2Tc(per, tp, ecc, omega, secondary=0):
    # calculate a transit time from orbital parameters in the 'cps' basis

    if secondary:
        f = 3*pi/2 - omega/360*2*pi                      # true anomaly during secondary eclipse
    else:
        f = pi/2   - omega/360*2*pi                      # true anomaly during transit
    EE = 2 * arctan( tan(f/2) * sqrt((1-ecc)/(1+ecc)) )  # eccentric anomaly
    tc = tp + per/(2*pi) * (EE - ecc*sin(EE))         # time of conjunction

    return tc
