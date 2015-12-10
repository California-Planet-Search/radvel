class LikelihoodSettings(object):
#    def __init__(self, basis='cps'):
#        self.attr = attr
    
    from numpy import array
#    # parameter basis; 'cps' = [P, tp, e, om, K] ...  [gamma, [dvdt, [curve]]]
#    if basis == 'cps':       
#        parname = ['Per (days)', 'Tp', 'e', 'om', 'K (m/s)']
#    # parameter basis; 'tcecos' = [P, tc, ecos(om), esin(om), K] ...  [gamma, [dvdt, [curve]]]
#    elif basis == 'tcecos':  
#        parname = ['Per (days)', 'Tc', 'ecos(om)', 'esin(om)', 'K (m/s)']
    
    basis = ''                    # parameter basis
    parname = []                  # parameter names
    time_base = 14000             # T0 for rv_drive
    nplanets = 1                  # number of planets
    jitter = 0                    # set to >1 if jitter should be added in quadrature is allowed in fit; =1 - all telescope use same jitter, >1 - telescopes use different jitter
    trend = 0                     # set to 1 if a trend is allowed in fit
    curv = 0                      # set to 1 if curvature allowed in fit
    jitter = 0                    # set to 1 to allow jitter in likelihood calculation
    prior_p = 0                   # orbital period [implemented?]
    prior_p_sig = 0               # orbital period error [implemented?]
    prior_tc = 0                  # Transit time [implemented?]
    prior_tc_sig = 0              # Transit time error [implemented?]
    telvec = array([-1])          # telescope vector; must be same length as t [implemented?]
    index_telvec = array([-1])    # indices of parameter arrays for telvec parameters [implemented?]
    index_jitter = array([-1])    # indices of parameter arrays for jitter parameters [implemented?]

# def set_basis(settings, basis):
#     settings.basis = basis
#     if settings.basis == 'cps': # parameter basis; 'cps' = [P, tp, e, om, K] ...  [gamma, [dvdt, [curve]]]      
#         parname = ['Per (days)', 'Tp', 'e', 'om', 'K (m/s)']
#     elif settings.basis == 'tcecos':  # parameter basis; 'tcecos' = [P, tc, ecos(om), esin(om), K] ...  [gamma, [dvdt, [curve]]]
#         parname = ['Per (days)', 'Tc', 'ecos(om)', 'esin(om)', 'K (m/s)']
# 

def lnlike_rv(params, t, rv, rv_sig, settings):
    from numpy import zeros,arange,sum,sqrt,log,pi,arctan

    # fill in orbital elements into format for rv_drive
    orbel = zeros(settings.nplanets*7 + settings.curv)
    if settings.basis == 'cps':
        for p in arange(settings.nplanets): 
            orbel[p*7:p*7+5] = params[p*5:p*5+5] # parameters for each planet
    elif settings.basis == 'tcecos':
        for p in arange(settings.nplanets): 
            orbel[p*7:p*7+5] = params[p*5:p*5+5] # parameters for each planet
            orbel[p*7+2] = sqrt(params[p*5+2]**2+params[p*5+2]**2) # ecc = sqrt(esin(om)^2+ecos(om)^2)
            orbel[p*7+3] = arctan(params[p*5+2]/params[p*5+2]) # om[deg] = arctan(esin(om)/ecos(om))
    orbel[5] = params[settings.nplanets*5]      # gamma
    if settings.trend: orbel[6] = params[settings.nplanets*5+1] # trend
    if settings.curv:  orbel[settings.nplanets*7] = params[settings.nplanets*5+2] # curv
    # add telvec, basis logic later ################

    # add jitter to errors
    if settings.jitter: 
        jitter = params[-1]
        rv_err = sqrt(rv_sig**2 + jitter**2)
        params = params[0:len(params)-1]
    else: rv_err = rv_sig

    # compute model
    rv_model = rv_drive(t,orbel,time_base=settings.time_base)

    # compute likelihood
    lnlike = -0.5*sum(((rv-rv_model)/(rv_err))**2) # sum of normalized, squared residuals
    if settings.jitter: 
        lnlike -= sum(log(sqrt(2*pi*rv_err**2))) # to allow for variable jitter

    return lnlike
    
def lnprior_rv(pars, settings):  
    from numpy import arange, inf, sqrt
    lnp = 0.0 #* orbel
    for p in arange(settings.nplanets): # positive K and e = [0,1]
        if pars[p*5+4] < 0: lnp = -inf
        if settings.basis == 'cps': 
            if (pars[p*5+2] < 0) or (pars[p*5+2] > 1): lnp = -inf
        elif settings.basis == 'tcecos':
            if sqrt(pars[p*5+2]**2 + pars[p*5+3]**2) > 1: lnp = -inf

#    lnp -= 0.5*((pars[0] - settings.prior_p)/settings.prior_p_sig)**2 # period prior
#    dtc = (pars[1] - settings.prior_tc) % settings.prior_p # time from closest transit
#    dtc = min([dtc, settings.prior_p - dtc])
#    lnp -= 0.5*(dtc/settings.prior_tc_sig)**2 # timing prior
        
    return lnp    

def lnprob_rv(pars, t, rv, rvsig, settings):
    from numpy import isfinite,inf
    lp = lnprior_rv(pars, settings)
    if not isfinite(lp):
        return -inf
    return lnlike_rv(pars, t, rv, rvsig, settings) + lp

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
        fipp = ecc*sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1-fip      # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        d1 = -fi/fip                             #first  order correction to E
        d2 = -fi/(fip+d1*fipp/2.)                #second order correction to E
        d3 = -fi/(fip+d2*fipp/2.+d2*d2*fippp/6.) #third  order correction to E
        E = E+d3
        Earr[convd] = E
        fiarr = (Earr-eccarr*sin(Earr)-Marr)  #how well did we do?
        convd = abs(fiarr) > conv, nd            #test for convergence
        nd = sum(convd == True)

        # add conditional statement to check for convergence after 100 iterations

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

def transit_time(pars, secondary=0):
    # calculate a transit time from orbital parameters in the 'cps' basis

    from numpy import zeros, arctan, tan, sqrt, sin, arange, pi
    np = len(pars)/7 # number of planets
    tc = zeros(np)

    for p in arange(np):
        per   = float(pars[p*7+0]) # actually, float means double here
        tp    = float(pars[p*7+1])
        ecc   = float(pars[p*7+2])
        omega = float(pars[p*7+3])
        if secondary:
            f = 3*pi/2 - omega/360*2*pi                      # true anomaly during secondary eclipse
        else:
            f = pi/2   - omega/360*2*pi                      # true anomaly during transit
        EE = 2 * arctan( tan(f/2) * sqrt((1-ecc)/(1+ecc)) )  # eccentric anomaly
        tc[p] = tp + per/(2*pi) * (EE - ecc*sin(EE))         # time of conjunction
        if np == 1: tc = tc.item(0)
    return tc

def basis_cps2tcecos(parin):
    # convert Keplerian parameters in the 'cps' basis:
    # [P, tp, e, om, K, gamma, dvdt, curv]  (allowing for multiple planets)
    # to the 'tcecos' basis:
    # [P, tc, e*cos(om), e*sin(om), K, gamma, dvdt, curv]  (allowing for multiple planets by expanding array in the usual way)

    from numpy import arange, pi, cos, sin
    par = parin
    parout = par
    np = len(par)/7
    for p in arange(np):
        #print par[p*7+3]
        parout[p*7+1] = transit_time(par[p*7:p*7+7])
        parout[p*7+2] = par[p*7+2] * cos(par[p*7+3]/180.*pi) # e*cos(om)
        parout[p*7+3] = par[p*7+2] * sin(par[p*7+3]/180.*pi) # e*sin(om)
        print "parin[1:4]  = ", parin[1:4]
        print "par[1:4]    = ", par[1:4]
        print "parout[1:4] = ", parout[1:4]
        print "om = ", par[p*7+3], "; om = ", par[p*7+3]/180.*pi, "; cos(om) = ", cos(par[p*7+3]/180.*pi)
    return parout

def basis_tcecos2cps(parin):
    # convert Keplerian parameters from the 'tcecos' basis:
    # [P, tc, e*cos(om), e*sin(om), K, gamma, dvdt, curv]  (allowing for multiple planets)
    # to the 'cps' basis:
    # [P, tp, e, om, K, gamma, dvdt, curv]  (allowing for multiple planets)

    from numpy import arange, pi, sqrt, arctan2, tan, arctan, sin
    par = parin
    parout = par
    np = len(par)/7
    for p in arange(np):
        # converting ecosom, esinom to e, omega (degrees)
        ecc = sqrt((par[p*7+2])**2 + (par[p*7+3])**2)
        om = arctan2( par[p*7+3] , parin[p*7+2] ) / pi * 180

        while om < 0.:
             om += 360
        parout[p*7+2] = ecc
        parout[p*7+3] = om
        # om in [0.,360)

        f = pi/2 - parout[p*7+3]/180*pi 
        # true anomaly during conjunction
        EE = 2 * arctan( tan(f/2) * sqrt((1-parout[p*7+2])/(1.+parout[p*7+2]))) 

        # eccentric anomaly
        parout[p*7+1] = par[p*7+1] - par[p*7+0]/(2*pi) * (EE - parout[p*7+2]*sin(EE)) # tc (transit time)

    return parout

def timebin(time, meas, meas_err, binsize):
#    import numpy as np
    from numpy import argsort,where,logical_and,array,append,sqrt,sum

    # This routine bins a set of times, measurements, and measurement errors 
    # into time bins.  All inputs and outputs should be floats or double. 
    # binsize should have the same units as the time array.

    # Andrew Howard, Jan. 2014 (translated from IDL)

    if binsize == 0: 
        return time, meas, meas_err

    # order times and data
    #ind_order = ind = argsort(time)
    ind_order = argsort(time)
    time = time[ind_order]
    meas = meas[ind_order]
    meas_err = meas_err[ind_order]

    ct=0
    while ct < len(time):
        #ind = where(time >= time[ct] and time < time[ct]+binsize, num) 
        ind = where(logical_and(time >= time[ct], time < time[ct]+binsize))
        num = len(time[ind])
        wt = (1./meas_err[ind])**2.    # weights based in errors
        wt = wt/sum(wt)                # normalized weights
        if ct == 0:
            time_out     = array([sum(wt*time[ind])])
            meas_out     = array([sum(wt*meas[ind])])
            meas_err_out = array([1/sqrt(sum(1/(meas_err[ind])**2))])
        else:
            time_out     = append(time_out, sum(wt*time[ind]))
            meas_out     = append(meas_out, sum(wt*meas[ind]))
            meas_err_out = append(meas_err_out, 1/sqrt(sum(1/(meas_err[ind])**2)))
        ct += num

    return time_out, meas_out, meas_err_out
    
def read_vst(filename='',star='', binsize=0, ctsmin=0, errmax_frac=0):
    import scipy.io as sio
    import numpy as np

    if (filename == ''):
        filename = '/mir3/vel/vst'+star+'.dat'
    vst = sio.readsav(filename)
    cf3 = vst.cf3

    if (ctsmin > 0):
        cf3 = cf3[cf3.cts > ctsmin]
    if (errmax_frac > 1):
        cf3 = cf3[(cf3.errvel < errmax_frac*np.median(cf3.errvel))]

    if (binsize > 0):
        t,rv,rv_sig = timebin(cf3.jd,cf3.mnvel,cf3.errvel,1./12)
    else:
        t,rv,rv_sig = cf3.jd,cf3.mnvel,cf3.errvel

    return t,rv,rv_sig

def plot_rvs(t,v,e, orbel=[0]):
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.time import Time
    
    tt = Time(t+2440000,format='jd')
    
    plt.figure()

    if len(orbel) > 1:
        tfine = min(t) + np.linspace(0,max(t)-min(t),2000)
        rvfine = rv_drive(tfine,orbel,time_base=14000)
        yrfine = Time(tfine+2440000,format='jd')
        plt.plot(yrfine.jyear, rvfine, 'b')
        #if plot_resid == 1:
        #    f, (ax1, ax2) = plt.subplots(3, sharex=True, sharey=True)

#    else:
#        plt.figure()

    plt.errorbar(tt.jyear, v, yerr=e, fmt='r.') #plot(t, rv, 'r.')
    
    
    plt.xlabel('Year')
    plt.ylabel('RV (m s$^{-1}$)')
    #plt.show()
    return plt
    

def plot_periodogram(t, data, errors, pmin=1.1, pmax=10000, type=None):
    # periodogram of data and residuals
    import numpy as np
    import scipy.signal as sp
    import matplotlib.pyplot as plt
    pmin = 1.1
    pmax = 10000# 30*365.
    f = np.linspace(2*np.pi/pmin, 2*np.pi/pmax, 100000)
    pers = 2*np.pi/f
    
    if type == 'GLS':
        from PyAstronomy.pyTiming import pyPeriod
        freq = np.linspace(1/pmin, 1/pmax, 100000)
        rvs = pyPeriod.TimeSeries(t, data, errors)
        clp = pyPeriod.Gls(rvs, ofac=10, hifac=1, norm="Cumming", freq=freq)
        power = clp.power
    else: # Lomb-Scargle
        power = sp.lombscargle(t, data-np.median(data), freq)
    
    plt.figure()
    plt.plot(pers, power,'r')
    plt.xlabel('Period (d)')
    plt.ylabel('Power')
    plt.xscale('log')
    #plt.show()
    return plt
