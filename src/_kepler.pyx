# cimport the Cython declarations for numpy
cimport numpy as np
import numpy as np
import cython

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# Wrapping kepler(M,e) a simple function that takes two doubles as
# arguments and returns a double
cdef extern from "kepler.c":
    double kepler(double M, double e)
    double rv_drive(double t, double per, double tp, double e, double om, double k )

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# create the wrapper code, with numpy type annotations
@cython.boundscheck(False)
def kepler_array(double [:,] M, double e):
    cdef int size, i

    size = M.shape[0]
    cdef np.ndarray[double, ndim=1] E = \
        np.ndarray(shape=(size,), dtype=np.float64) 

    for i in range(size):
        E[i] = kepler(M[i], e)

    return E 

# create the wrapper code, with numpy type annotations
@cython.boundscheck(False)
def rv_drive_array(np.ndarray[DTYPE_t, ndim=1] t, double per, double tp, 
                   double e, double om, double k):
    cdef int size, i 
    size = t.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] rv = t.copy()
    for i in range(size):
        rv[i] = rv_drive(t[i], per, tp, e, om, k)

    return rv

