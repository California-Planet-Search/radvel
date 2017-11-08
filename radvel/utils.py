
import imp
import os
from decimal import Decimal
from contextlib import contextmanager
import warnings

import numpy as np
from datetime import datetime, timedelta
from astropy import constants as c
from astropy import units as u

import radvel

# Normalization.
# RV m/s of a 1.0 Jupiter mass planet tugging on a 1.0
# solar mass star on a 1.0 year orbital period
K_0 = 28.4329


def initialize_posterior(config_file, decorr=False):

    system_name = os.path.basename(config_file).split('.')[0]
    P = imp.load_source(system_name, os.path.abspath(config_file))
    system_name = P.starname

    params = P.params
    assert str(params.basis) == "Basis Object <{}>".format(P.fitting_basis), """
Parameters in config file must be converted to fitting basis.
"""

    if decorr:
        try:
            decorr_vars = P.decorr_vars
        except:
            raise Exception("--decorr option selected,\
 but decorr_vars is not found in your setup file.")
    else:
        decorr_vars = []
    
    for key in params.keys():
        if key.startswith('logjit'):
            msg = """
Fitting log(jitter) is depreciated. Please convert your config
files to initialize 'jit' instead of 'logjit' parameters.
Converting 'logjit' to 'jit' for you now.
"""
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            newkey = key.replace('logjit', 'jit')
            params[newkey] = radvel.model.Parameter(value=np.exp(params[key].value), vary=params[key].vary)
            del params[key]

    iparams = radvel.basis._copy_params(params)
    
    # Make sure we don't have duplicate indicies in the DataFrame
    P.data = P.data.reset_index(drop=True)

    # initialize RVmodel object
    mod = radvel.RVModel(params, time_base=P.time_base)

    # initialize RVlikelihood objects for each instrument
    telgrps = P.data.groupby('tel').groups
    likes = {}
    for inst in P.instnames:
        assert inst in P.data.groupby('tel').groups.keys(), \
            "No data found for instrument '{}'.\nInstruments found in this dataset: {}".format(inst,
                                            list(telgrps.keys()))
        decorr_vectors = {}
        if decorr:
            for d in decorr_vars:
                decorr_vectors[d] = P.data.iloc[telgrps[inst]][d].values
        likes[inst] = radvel.likelihood.RVLikelihood(
            mod, P.data.iloc[telgrps[inst]].time,
            P.data.iloc[telgrps[inst]].mnvel,
            P.data.iloc[telgrps[inst]].errvel, suffix='_'+inst,
            decorr_vars=decorr_vars, decorr_vectors=decorr_vectors
        )
        likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
        likes[inst].params['jit_'+inst] = iparams['jit_'+inst]

    like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

    # Initialize Posterior object
    post = radvel.posterior.Posterior(like)
    post.priors = P.priors


    return P, post


def round_sig(x, sig=2):
    """
    Round to the requested number of significant figures.

    Args:
        x (float): value
        sig (int): (optional) desired number of significant figures

    Returns:
        float: `x` rounded to `sig` significant figures
    """

    if x == 0 or np.isnan(x): return 0.0
    return round(x, sig-int(floor(log10(abs(x))))-1)


def sigfig(med, errlow, errhigh=None):
    """
    Format values with errors into an equal number of signficant figures.

    Args:
        med (float): median value
        errlow (float): lower errorbar
        errhigh (float): upper errorbar

    Returns:
        tuple: (med,errlow,errhigh) rounded to the lowest number of significant figures

    """
    
    if errhigh==None: errhigh = errlow
        
    ndec = Decimal(str(errlow)).as_tuple().exponent
    if abs(Decimal(str(errhigh)).as_tuple().exponent) > abs(ndec):
        ndec = Decimal(str(errhigh)).as_tuple().exponent
    if ndec < -1:
            tmpmed = round(med,abs(ndec))
            p = 0
            while tmpmed == 0:
                tmpmed = round(med, abs(ndec)+p)
                p += 1
            med = tmpmed
    elif (ndec == -1 and str(errhigh)[-1] == '0') and (ndec == -1 and str(errlow)[-1] == '0') or ndec == 0:
            errlow = int(round_sig(errlow))
            errhigh = int(round(errhigh))
            med = int(round(med))
    else: med = round(med,abs(ndec))

    return med, errlow, errhigh


def time_print(tdiff):
    units = 'seconds'
    if tdiff > 60:
        tdiff /= 60
        units = 'minutes'
        if tdiff > 60:
            tdiff /= 60
            units = 'hours'
            if tdiff > 24:
                tdiff /= 24
                units = 'days'
    return tdiff, units


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
            time_out = [np.sum(wt*time[ind])]
            meas_out = [np.sum(wt*meas[ind])]
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
        t_bin, vel_bin, err_bin = timebin(
            t[pos], vel[pos], err[pos], binsize=binsize
        )
        rvtimes = np.hstack((rvtimes, t_bin))
        rvdat = np.hstack((rvdat, vel_bin))
        rverr = np.hstack((rverr, err_bin))
        newtelvec = np.hstack((newtelvec, np.array([tel]*len(t_bin))))
        
    return rvtimes, rvdat, rverr, newtelvec


def fastbin(x,y,nbins=30):
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    bindat = sy / n
    binerr = np.sqrt(sy2/n - bindat*bindat) / np.sqrt(n)
    bint = (_[1:] + _[:-1])/2.

    binN = n
    pos = binN >= 3# 0.5 * np.mean(binN)
    bint = bint[pos]
    bindat = bindat[pos]
    binerr = binerr[pos]

    pos = bint>0
    bint = bint[pos]
    bindat = bindat[pos]
    binerr = binerr[pos]
    return bint,bindat,binerr


def round_sig(x, sig=2):
    if x == 0: return 0.0
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def t_to_phase(params, t, num_planet, cat=False):
    if ('tc%i' % num_planet) in params:
        timeparam = 'tc%i' % num_planet
    elif ('tp%i' % num_planet) in params:
        timeparam = 'tp%i' % num_planet
        
    P = params['per%i' % num_planet].value
    tc = params[timeparam].value
    phase = np.mod(t - tc, P) 
    phase /= P
    if cat: phase = np.concatenate((phase,phase+1))
    return phase


@contextmanager
def working_directory(dir):
    """Do something in a directory

    Function to use with `with` statements.

    Args:
       dir (string): name of directory to work in
    
    Example:
        >>> with workdir('/temp'):
            # do something within the /temp directory
    """
    cwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(cwd)


def cmd_exists(cmd):
    return any(
        os.access(os.path.join(path, cmd), os.X_OK) 
        for path in os.environ["PATH"].split(os.pathsep))


def date2jd(date):
    """
    Convert datetime object to JD"

    Args:
        date (datetime.datetime): date to convert
    Returns:
        float: Julian date
     """
    
    jd_td = date - datetime(2000,1,1,12,0,0)
    jd = 2451545.0 + jd_td.days + jd_td.seconds/86400.0
    return jd


def jd2date(jd):
    """
    Convert JD to datetime.datetime object

    Args:
        jd (float): Julian date
    Returns:
        datetime.datetime: calendar date
    """
    
    mjd = jd - 2400000.5
    td = timedelta(days=mjd)
    dt = datetime(1858,11,17,0,0,0) + td

    return dt


def geterr(vec, angular=False):
    """
    Calculate median, 15.9, and 84.1 percentile values
    for a given vector."

    Args:
        vec (array): vector, usually an MCMC chain for one parameter
        angular (bool): (optional) Is this an angular parameter?
            if True vec should be in radians. This will perform
            some checks to ensure proper boundary wrapping.

    Returns:
        tuple: 50, 15.9 and 84.1 percentiles
    """

    if angular:
        val, edges = np.histogram(vec, bins=50)
        med = edges[np.argmax(val)]
        if med > np.radians(90):
            vec[vec<np.radians(0)] = vec[vec<np.radians(0)] + np.radians(360)
        if med <= np.radians(-90):
            vec[vec>=np.radians(0)] = vec[vec>=np.radians(0)] - np.radians(360)
        med = np.median(vec)
    else:
        med = np.median(vec)
        
    s = sorted(vec)
    errlow = med - s[int(0.159*len(s))]
    errhigh = s[int(0.841*len(s))] - med
            
    return med, errlow, errhigh


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
        K = K_0 * (1 - e ** 2) ** -0.5 * Msini * (P / 365.0) ** (-1 / 3.) * \
            Mtotal ** (-2 / 3.)
    elif Msini_units.lower() == 'earth':
        K = K_0 * (1 - e ** 2) ** -0.5 * Msini * (P / 365.0) ** (-1 / 3.) * \
            Mtotal ** -(-2 / 3.) * (c.M_earth / c.M_jup).value
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
        Msini = K / K_0 * np.sqrt(1.0 - e ** 2.0) * Mtotal ** (2 / 3.) * \
                (P / 365.0) ** (1 / 3.)
    elif Msini_units.lower() == 'earth':
        Msini = K / K_0 * np.sqrt(1.0 - e ** 2.0) * Mtotal ** (2 / 3.) * \
                (P / 365.0) ** (1 / 3.) * (c.M_jup / c.M_earth).value
    else:
        raise Exception("Msini_units must be 'earth', or 'jupiter'")

    return Msini


def density(mass, radius, MR_units='earth'):
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
        vol = 4. / 3. * np.pi * (radius * c.R_earth) ** 3
        rho = ((mass * c.M_earth / vol).to(u.g / u.cm ** 3)).value
    elif MR_units.lower() == 'jupiter':
        vol = 4. / 3. * np.pi * (radius * c.R_jup) ** 3
        rho = ((mass * c.M_jup / vol).to(u.g / u.cm ** 3)).value
    else:
        raise Exception("MR_units must be 'earth', or 'jupiter'")
    return rho
