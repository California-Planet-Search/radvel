
import numpy as np
import radvel

# Try to import Kepler's equation solver written in C
try:
    from . import _kepler 
    cext = True
except ImportError:
    print("WARNING: KEPLER: Unable to import C-based Kepler's\
equation solver. Falling back to the slower NumPy implementation.")
    cext = False


def rv_drive(t, orbel, use_c_kepler_solver=cext):
    """RV Drive
    
    Args:
        t (array of floats): times of observations
        orbel (array of floats): [per, tp, e, om, K].\
            Omega is expected to be\
            in radians
        use_c_kepler_solver (bool): (default: True) If \
            True use the Kepler solver written in C, else \
            use the Python/NumPy version.
    Returns:
        rv: (array of floats): radial velocity model
    
    """
    
    # unpack array of parameters
    per, tp, e, om, k = orbel
    
    # Performance boost for circular orbits
    if e == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + om)
    
    if per < 0:
        per = 1e-4
    if e < 0:
        e = 0
    if e > 0.99:
        e = 0.99

    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    if use_c_kepler_solver:
        rv = _kepler.rv_drive_array(t, per, tp, e, om, k)
    else:
        nu = radvel.orbit.true_anomaly(t, tp, per, e)
        rv = k * (np.cos(nu + om) + e * np.cos(om))
    
    return rv


def kepler(inbigM, inecc):
    """Solve Kepler's Equation

    Args:
        inbigM (array): input Mean anomaly
        inecc (array): eccentricity

    Returns:
        array: eccentric anomaly
    
    """
    
    Marr = inbigM  # protect inputs; necessary?
    eccarr = inecc
    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)  
    convd = np.where(np.abs(fiarr) > conv)[0]  # which indices have not converged
    nd = len(convd)  # number of unconverged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1
        
        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E)  # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc * np.sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip 
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv  # test for convergence
        nd = np.sum(convd is True)
        
    if Earr.size > 1: 
        return Earr
    else: 
        return Earr[0]


def profile():
    # Profile and compare C-based Kepler solver with
    # Python/Numpy implementation

    import timeit
    
    ecc = 0.1
    numloops = 5000
    print("\nECCENTRICITY = {}".format(ecc))

    for size in [10, 30, 100, 300, 1000]:

        setup = """\
from radvel.kepler import rv_drive
import numpy as np
gc.enable()
ecc = %f
orbel = [32.468, 2456000, ecc, np.pi/2, 10.0]
t = np.linspace(2455000, 2457000, %d)
""" % (ecc, size)

        print("\nProfiling pure C code for an RV time series with {} "
              "observations".format(size))
        tc = timeit.timeit('rv_drive(t, orbel, use_c_kepler_solver=True)',
                           setup=setup, number=numloops)
        print("Ran %d model calculations in %5.3f seconds" % (numloops, tc))

        print("Profiling Python code for an RV time series with {} "
              "observations".format(size))
        tp = timeit.timeit('rv_drive(t, orbel, use_c_kepler_solver=False)',
                           setup=setup, number=numloops)
        print("Ran %d model calculations in %5.3f seconds" % (numloops, tp))
        print("The C version runs %5.2f times faster" % (tp/tc))

    ecc = 0.7
    numloops = 5000
    print("\nECCENTRICITY = {}".format(ecc))

    for size in [30]:
            setup = """\
from radvel.kepler import rv_drive
import numpy as np
gc.enable()
ecc = %f
orbel = [32.468, 2456000, ecc, np.pi/2, 10.0]
t = np.linspace(2455000, 2457000, %d)
    """ % (ecc, size)

            print("\nProfiling pure C code for an RV time series with {} "
                  "observations".format(size))
            tc = timeit.timeit('rv_drive(t, orbel, use_c_kepler_solver=True)',
                               setup=setup, number=numloops)
            print("Ran %d model calculations in %5.3f seconds" % (numloops, tc))

            print("Profiling Python code for an RV time series with {} "
                  "observations".format(size))
            tp = timeit.timeit('rv_drive(t, orbel, use_c_kepler_solver=False)',
                               setup=setup, number=numloops)
            print("Ran %d model calculations in %5.3f seconds" % (numloops, tp))
            print("The C version runs %5.2f times faster" % (tp / tc))