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
    """Initialize Posterior object

    Parse a setup file and initialize the RVModel, Likelihood, Posterior and priors.

    Args:
        config_file (string): path to config file
        decorr (bool): (optional) decorrelate RVs against columns defined in the decorr_vars list

    Returns:
        tuple: (object representation of config file, radvel.Posterior object)
    """

    system_name = os.path.basename(config_file).split('.')[0]
    P = imp.load_source(system_name, os.path.abspath(config_file))

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

    # initialize Likelihood objects for each instrument
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

        try:
            hnames = P.hnames[inst]
            liketype = radvel.likelihood.GPLikelihood
            try:
                kernel_name = P.kernel_name[inst]
                # if kernel_name == "Celerite":
                #     liketype = radvel.likelihood.CeleriteLikelihood
                if kernel_name == "Celerite":
                     liketype = radvel.likelihood.CeleriteLikelihood
            except AttributeError:
                kernel_name = "QuasiPer"
        except AttributeError:
            liketype = radvel.likelihood.RVLikelihood
            kernel_name = None
            hnames = None
        likes[inst] = liketype(
            mod, P.data.iloc[telgrps[inst]].time,
            P.data.iloc[telgrps[inst]].mnvel,
            P.data.iloc[telgrps[inst]].errvel, hnames=hnames, suffix='_'+inst, 
            kernel_name=kernel_name, decorr_vars=decorr_vars, 
            decorr_vectors=decorr_vectors
        )
        likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
        likes[inst].params['jit_'+inst] = iparams['jit_'+inst]

    like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

    # Initialize Posterior object
    post = radvel.posterior.Posterior(like)
    post.priors = P.priors

    return P, post


def round_sig(x, sig=2):
    """Round by significant figures
    Args:
        x (float): number to be rounded
        sig (int): (optional) number of significant figures to retain
    Returns:
        float: x rounded to sig significant figures
    """

    if x == 0:
        return 0.0
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


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
    
    if errhigh is None:
        errhigh = errlow
        
    ndec = Decimal(str(errlow)).as_tuple().exponent
    if abs(Decimal(str(errhigh)).as_tuple().exponent) > abs(ndec):
        ndec = Decimal(str(errhigh)).as_tuple().exponent
    if ndec < -1:
            tmpmed = round(med, abs(ndec))
            p = 0
            if med != 0:
                while tmpmed == 0:
                    tmpmed = round(med, abs(ndec)+p)
                    p += 1
                med = tmpmed
    elif (ndec == -1 and str(errhigh)[-1] == '0') and (ndec == -1 and str(errlow)[-1] == '0') or ndec == 0:
            errlow = int(round_sig(errlow))
            errhigh = int(round(errhigh))
            med = int(round(med))
    else:
        med = round(med, abs(ndec))

    return med, errlow, errhigh


def time_print(tdiff):
    """Print time

    Helper function to print time remaining in sensible units.

    Args:
        tdiff (float): time in seconds

    Returns:
        tuple: (float time, string units)
    """
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
    """Bin in equal sized time bins

    This routine bins a set of times, measurements, and measurement errors
    into time bins.  All inputs and outputs should be floats or double.
    binsize should have the same units as the time array.
    (from Andrew Howard, ported to Python by BJ Fulton)

    Args:
        time (array): array of times
        meas (array): array of measurements to be comined
        meas_err (array): array of measurement uncertainties
        binsize (float): width of bins in same units as time array

    Returns:
        tuple: (bin centers, binned measurements, binned uncertainties)
    """

    ind_order = np.argsort(time)
    time = time[ind_order]
    meas = meas[ind_order]
    meas_err = meas_err[ind_order]
    ct = 0
    while ct < len(time):
        ind = np.where((time >= time[ct]) & (time < time[ct]+binsize))[0]
        num = len(ind)
        wt = (1./meas_err[ind])**2.     # weights based in errors
        wt = wt/np.sum(wt)              # normalized weights
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
    """Bin velocities by instrument

    Bin RV data with bins of with binsize in the units of t.
    Will not bin data from different telescopes together since there may
    be offsets between them.

    Args:
        t (array): array of timestamps
        vel (array): array of velocities
        err (array): array of velocity uncertainties
        telvec (array): array of strings corresponding to the instrument name for each velocity
        binsize (float): (optional) width of bin in units of t (default=1/2.)

    Returns:
        tuple: (bin centers, binned measurements, binned uncertainties, binned instrument codes)
    """

    # Bin RV data with bins of with binsize in the units of t.
    # Will not bin data from different telescopes together since there may
    # be offsets between them.

    ntels = len(np.unique(telvec))
    if ntels == 1:
        t_bin, vel_bin, err_bin = timebin(t, vel, err, binsize=binsize)
        return t_bin, vel_bin, err_bin, telvec
    
    uniqorder = np.argsort(np.unique(telvec, return_index=1)[1])
    uniqsort = np.unique(telvec)[uniqorder]
    rvtimes = np.array([])
    rvdat = np.array([])
    rverr = np.array([])
    newtelvec = np.array([])
    for i, tel in enumerate(uniqsort):
        pos = np.where(telvec == tel)
        t_bin, vel_bin, err_bin = timebin(
            t[pos], vel[pos], err[pos], binsize=binsize
        )
        rvtimes = np.hstack((rvtimes, t_bin))
        rvdat = np.hstack((rvdat, vel_bin))
        rverr = np.hstack((rverr, err_bin))
        newtelvec = np.hstack((newtelvec, np.array([tel]*len(t_bin))))
        
    return rvtimes, rvdat, rverr, newtelvec


def fastbin(x, y, nbins=30):
    """Fast binning

    Fast binning function for equally spaced data

    Args:
        x (array): independent variable
        y (array): dependent variable
        nbins (int): number of bins

    Returns:
        tuple: (bin centers, binned measurements, binned uncertainties)
    """

    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    bindat = sy / n
    binerr = np.sqrt(sy2/n - bindat*bindat) / np.sqrt(n)
    bint = (_[1:] + _[:-1])/2.

    binN = n
    pos = binN >= 3  # 0.5 * np.mean(binN)
    bint = bint[pos]
    bindat = bindat[pos]
    binerr = binerr[pos]

    pos = bint > 0
    bint = bint[pos]
    bindat = bindat[pos]
    binerr = binerr[pos]
    return bint, bindat, binerr


def t_to_phase(params, t, num_planet, cat=False):
    if ('tc%i' % num_planet) in params:
        timeparam = 'tc%i' % num_planet
    elif ('tp%i' % num_planet) in params:
        timeparam = 'tp%i' % num_planet
        
    P = params['per%i' % num_planet].value
    tc = params[timeparam].value
    phase = np.mod(t - tc, P) 
    phase /= P
    if cat:
        phase = np.concatenate((phase, phase+1))
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
    
    jd_td = date - datetime(2000, 1, 1, 12, 0, 0)
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
    dt = datetime(1858, 11, 17, 0, 0, 0) + td

    return dt


def geterr(vec, angular=False):
    """
    Calculate median, 15.9, and 84.1 percentile values
    for a given vector.

    Args:
        vec (array): vector, usually an MCMC chain for one parameter
        angular (bool [optioanl]): Is this an angular parameter?
            if True vec should be in radians. This will perform
            some checks to ensure proper boundary wrapping.

    Returns:
        tuple: 50, 15.9 and 84.1 percentiles
    """

    if angular:
        val, edges = np.histogram(vec, bins=50)
        med = edges[np.argmax(val)]
        if med > np.radians(90):
            vec[vec < np.radians(0)] = vec[vec < np.radians(0)] + np.radians(360)
        if med <= np.radians(-90):
            vec[vec >= np.radians(0)] = vec[vec >= np.radians(0)] - np.radians(360)
        med = np.median(vec)
    else:
        med = np.median(vec)
        
    s = sorted(vec)
    errlow = med - s[int(0.159*len(s))]
    errhigh = s[int(0.841*len(s))] - med
            
    return med, errlow, errhigh

def semi_amplitude(Msini, P, Mtotal, e, Msini_units='jupiter'):
    """Compute Doppler semi-amplitude

    Args:
        Msini (float): mass of planet [Mjup]
        P (float): Orbital period [days]
        Mtotal (float): Mass of star + mass of planet [Msun]
        e (float): eccentricity
        Msini_units (Optional[str]): Units of Msini {'earth','jupiter'}
            default: 'jupiter'

    Returns:
        Doppler semi-amplitude [m/s]

    """

    # convert inputs to array so they work with units
    P = np.array(P)
    Msini = np.array(Msini)
    Mtotal = np.array(Mtotal)
    e = np.array(e)

    P = (P * u.d).to(u.year).value
    if Msini_units.lower() == 'jupiter':
        pass 
    elif Msini_units.lower() == 'earth':
        Msini = (Msini * u.M_earth).to(u.M_jup).value
    else:
        raise Exception("Msini_units must be 'earth', or 'jupiter'")

    K = K_0*(1 - e**2)**-0.5*Msini*P**(-1.0/3.0)*Mtotal**(-2.0 / 3.0)

    return K

def semi_major_axis(P, Mtotal):
    """Semi-major axis

    Kepler's third law

    Args: 
        P (float): Orbital period [days]
        Mtotal (float): Mass [Msun]

    Returns:
        float or array: semi-major axis in AU
    """

    # convert inputs to array so they work with units
    P = np.array(P)
    Mtotal = np.array(Mtotal)

    Mtotal = Mtotal*u.Msun
    P = P * u.d
    a = (c.G * Mtotal * P**2 / 4.0 / np.pi**2)**(1.0/3.0)
    a = a.to(u.AU).value

    return a


def Msini(K, P, Mtotal, e, Msini_units='earth'):
    """Calculate Msini

    Calculate Msini for a given K, P, stellar mass, and e

    Args:
        K (float): Doppler semi-amplitude [m/s]
        P (float): Orbital period [days]
        Mtotal (float): Mass of star + mass of planet [Msun]
        e (float): eccentricity
        Msini_units (Optional[str]): Units of Msini {'earth','jupiter'} 
            default: 'earth'

    Returns:
        float: Msini [units = Msini_units]

    """
    # convert inputs to array so they work with units
    P = np.array(P)
    Mtotal = np.array(Mtotal)
    K = np.array(K)
    e = np.array(e)

    P = (P * u.d).to(u.year).value
    Msini = K / K_0 * np.sqrt(1.0 - e**2.0)*Mtotal**(2.0 / 3.0)*P**(1 / 3.0) 
    if Msini_units.lower() == 'jupiter':
        pass 
    elif Msini_units.lower() == 'earth':
        Msini = (Msini * u.M_jup).to(u.M_earth).value
    else:
        raise Exception("Msini_units must be 'earth', or 'jupiter'")

    return Msini


def density(mass, radius, MR_units='earth'):
    """Compute density from mass and radius

    Args:
        mass (float): mass [MR_units]
        radius (float): radius [MR_units]
        MR_units (string): (optional) units of mass and radius. Must be 'earth', or 'jupiter' (default 'earth').

    Returns:
        float: density in g/cc
    """

    mass = np.array(mass)
    radius = np.array(radius)

    if MR_units.lower() == 'earth':
        uradius = u.R_earth
        umass = u.M_earth
    elif MR_units.lower() == 'jupiter':
        uradius = u.R_jup
        umass = u.M_jup
    else:
        raise Exception("MR_units must be 'earth', or 'jupiter'")

    vol = 4. / 3. * np.pi * (radius * uradius) ** 3
    rho = ((mass * umass / vol).to(u.g / u.cm ** 3)).value
    return rho


def draw_models_from_chain(mod, chain, t, nsamples=50):
    """Draw Models from Chain

    Given an MCMC chain of parameters, draw representative parameters
    and synthesize models.

    Args:
        mod (radvel.RVmodel) : RV model
        chain (DataFrame): pandas DataFrame with different values from MCMC
            chain
        t (array): time range over which to synthesize models
        nsamples (int): number of draws

    Returns:
        array: 2D array with the different models as different rows
    """

    np.random.seed(0)
    chain_samples = chain.ix[np.random.choice(chain.index, nsamples)]
    models = []
    for i in chain_samples.index:
        params = np.array(chain.ix[i, mod.vary_parameters])
        params = mod.array_to_params(params)
        models += [mod.model(params, t)]
    models = np.vstack(models)
    return models

