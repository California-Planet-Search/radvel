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

def kepler_cext(M, e):
    return kepler(M, e)

# cdefine the signature of our c function
cdef extern from "kepler.c":
    void kepler_array(double * M_array, double e, double * E_array, int size)


# create the wrapper code, with numpy type annotations
@cython.boundscheck(False)
def kepler_array_cext(double [:,] M, double e):
    cdef int size = M.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] E = \
        np.empty(size, dtype=np.float64)

    cdef int i
    for i in range(size):
        E[i] = kepler(M[i], e)

    return E 

