import sys
import radvel
import scipy
from scipy import spatial
import abc
import numpy as np
import warnings

warnings.simplefilter('once')

# implemented kernels & examples of possible names for associated hyperparameters
KERNELS = {
    "SqExp": ['gp_length','gp_amp'],
    "Per": ['gp_per','gp_length','gp_amp'],
    "QuasiPer": ['gp_per','gp_perlength','gp_explength','gp_amp'],
    "Celerite": ['1_logA','1_logB','1_logC','1_logD']
}

if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC


# celerite is an optional dependency
def _try_celerite():
    try:
        import celerite
        from celerite.solver import CholeskySolver
        return True
    except ImportError:
        warnings.warn("celerite not installed. GP kernals using celerite will not work.\n\
Try installing celerite using 'pip install celerite'", ImportWarning)
        return False

_has_celerite = _try_celerite()
if _has_celerite:
    import celerite
    from celerite.solver import CholeskySolver

class Kernel(ABC):
    """
    Abstract base class to store kernel info and compute covariance matrix.
    All kernel objects inherit from this class.

    Note:
        To implement your own kernel, create a class that inherits
        from this class. It should have hyperparameters that follow
        the name scheme 'gp_NAME_SUFFIX'.

    """

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def compute_distances(self, x1, x2):
        pass

    @abc.abstractmethod
    def compute_covmatrix(self, errors):
        pass


class SqExpKernel(Kernel):
    """
    Class that computes and stores a squared exponential kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly two objects, 'gp_length*'
            and 'gp_amp*', where * is a suffix identifying
            these hyperparameters with a likelihood object.

    """

    @property
    def name(self):
        return "SqExp"

    def __init__(self, hparams):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_length'):
                self.hparams['gp_length'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams['gp_amp'] = hparams[par]

        assert len(hparams) == 2, \
        "SqExpKernel requires exactly 2 hyperparameters with names" \
        + "'gp_length*' and 'gp_amp*'."

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
        except KeyError:
            raise KeyError("SqExpKernel requires hyperparameters 'gp_length*'" \
                           + " and 'gp_amp*'.")
        except AttributeError:
            raise AttributeError("SqExpKernel requires dictionary of" \
                                 + " radvel.Parameter objects as input.")

    def __repr__(self):
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        return "SqExp Kernel with length: {}, amp: {}".format(length, amp)

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature. 
        """
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value

        K = scipy.matrix(amp**2 * scipy.exp(-self.dist/(length**2)))

        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError: # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix


class PerKernel(Kernel):
    """
    Class that computes and stores a periodic kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -\\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3^2 } ) }{ 2\\eta_2^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly three objects, 'gp_length*',
            'gp_amp*', and 'gp_per*', where * is a suffix identifying
            these hyperparameters with a likelihood object.

    """

    @property
    def name(self):
        return "Per"

    def __init__(self, hparams):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_length'):
                self.hparams['gp_length'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams['gp_amp'] = hparams[par]
            if par.startswith('gp_per'):
                self.hparams['gp_per'] = hparams[par]

        assert len(hparams) == 3, \
        "PerKernel requires exactly 3 hyperparameters with names 'gp_length*'," \
        + " 'gp_amp*', and 'gp_per*'."

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
        except KeyError:
            raise KeyError("PerKernel requires hyperparameters 'gp_length*'," \
                           + " 'gp_amp*', and 'gp_per*'.")
        except AttributeError:
            raise AttributeError("PerKernel requires dictionary of " \
                                 + "radvel.Parameter objects as input.")

    def __repr__(self):
        length= self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        return "Per Kernel with length: {}, amp: {}, per: {}".format(
            length, amp, per
        )

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')

    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature. 
        """
        length= self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value

        K = scipy.matrix(amp**2 * scipy.exp(-np.sin(np.pi*self.dist/per)**2.
                                                 / (2.*length**2)))
        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError: # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix

class QuasiPerKernel(Kernel):
    """
    Class that computes and stores a quasi periodic kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } 
                           - \\frac{ \\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3^2 } ) }{ 2\\eta_4^2 } )     

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly four objects, 'gp_explength*',
            'gp_amp*', 'gp_per*', and 'gp_perlength*', where * is a suffix 
            identifying these hyperparameters with a likelihood object.

    """
    @property
    def name(self):
        return "QuasiPer"

    def __init__(self, hparams):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_perlength'):
                self.hparams['gp_perlength'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams['gp_amp'] = hparams[par]
            if par.startswith('gp_per') and not 'length' in par:
                self.hparams['gp_per'] = hparams[par]
            if par.startswith('gp_explength'):
                self.hparams['gp_explength'] = hparams[par]

        assert len(hparams) == 4, \
        "QuasiPerKernel requires exactly 4 hyperparameters with names" \
        + " 'gp_perlength*', 'gp_amp*', 'gp_per*', and 'gp_explength*'."

        try:
            self.hparams['gp_perlength'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except KeyError:
            raise KeyError("QuasiPerKernel requires hyperparameters" \
                           + " 'gp_perlength*', 'gp_amp*', 'gp_per*', " \
                           + "and 'gp_explength*'.")
        except AttributeError:
            raise AttributeError("QuasiPerKernel requires dictionary of" \
                                 + " radvel.Parameter objects as input.")

    def __repr__(self):
        perlength = self.hparams['gp_perlength'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        msg = (
            "QuasiPer Kernel with amp: {}, per length: {}, per: {}, " 
            "exp length: {}"
        ).format(amp, perlength, per, explength)
        return msg

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature. 
        """
        perlength = self.hparams['gp_perlength'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        K = scipy.matrix(amp**2
                         * scipy.exp(-self.dist_se/(explength**2))
                         * scipy.exp((-np.sin(np.pi*self.dist_p/per)**2.)
                                      / (2.*perlength**2)))
        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError: # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix

class CeleriteKernel(Kernel):
    """ A wrapper for the celerite.solver.CholeskySolver() methods and attributes.

    Class that computes and stores a kernel that can be modeled as the sum of 
    celerite terms, or kernels of the following form:

    .. math::

        C_{ij} = \\sum_{n=1}^{J} \\frac{1}{2}(a_n + ib_n)e^{-(c_n + id_n)\\tau_{nm}} 
                               + \\frac{1}{2}(a_n - ib_n)e^{-(c_n - id_n)\\tau_{nm}}

    where :math:`J` is the number of terms in the sum, and :math:`\\tau_{nm}=|t_i - t_j|`. 
    
    The hyperparameters of this kernel are :math:`a_{n}` , :math:`b_{n}` , :math:`c_{n}` , and :math:`d_n` 
    for each term :math:`C_{ij}` in the sum. 

    See celerite.readthedocs.io for more information about celerite kernels and
    computation.

    Note: 
        For this kernel to be positive-definite, we must have :math:`a_nc_n \\ge b_nd_n`
        at all times. The CeleriteLikelihood object will raise a warning
        if it ever detects a non-positive-definite kernel.

    :param: hparams (dict of radvel.Parameter): dictionary containing
        radvel.Parameter objects that are GP hyperparameters
        of this kernel. Must contain a multiple of 4 Parameter object
        with the following names:
                - `k_logA*`: the natural log of :math:`a_{k}`. 
                - `k_logB*`: the natural log of :math:`b_{k}`.
                - `k_logC*`: the natural log of :math:`c_{k}`. 
                - `k_logD*`: the natural log of :math:`d_{k}`. 
        (where k is a 1-indexed integer identifying coefficients of a 
        particular term, :math:`1 <= k <= N_{terms}`, and * is an optional 
        suffix, e.g. 'hires'). Suffix is useful when fitting several individual 
        GPLikelihoods with different hyperparameters using the 
        CompositeLikelihood object.

    """

    @property
    def name(self):
        return "Celerite"

    def __init__(self, hparams):

        assert len(hparams) > 0 and len(hparams) % 4 == 0, \
            "CeleriteKernel requires a positive integer number of terms, each" \
             + "with 4 coefficients. See CeleriteKernel documentation."
        self.num_terms = int(len(hparams) / 4)
        self.hparams = np.zeros((self.num_terms, 4))

        # set up hyperparameter arrays
        try:
            for par in hparams:
                index = int(par[0]) - 1
                if 'logA' in par:
                    self.hparams[index,0] = hparams[par].value
                if 'logB' in par:
                    self.hparams[index,1] = hparams[par].value
                if 'logC' in par:
                    self.hparams[index,2] = hparams[par].value
                if 'logD' in par:
                    self.hparams[index,3] = hparams[par].value
        except AttributeError:
            raise AttributeError("CeleriteKernel requires dictionary of" \
                                 + " radvel.Parameter objects as input.")
        except IndexError:
            raise IndexError("CeleriteKernel hyperparameter indices (k in k_logA*)"
                             + " were named incorrectly. See CeleriteKernel documentation.")
        except ValueError:
            raise ValueError("CeleriteKernel hyperparameter indices (k in k_logA*)"
                             + " were named incorrectly. See CeleriteKernel documentation.")

    # get arrays of real and complex parameters
    def compute_real_and_complex_hparams(self):
        real_indices = []
        complex_indices = []
        for col in np.arange(self.num_terms):
            if np.exp(self.hparams[col,1])==0 \
            and np.exp(self.hparams[col,3])==0: 
                real_indices.append(col)
            else:
                complex_indices.append(col)
        self.real = self.hparams[real_indices]
        self.complex = self.hparams[complex_indices]

    def __repr__(self):
        msg = (
            "Celerite Kernel with log(a) = {}, log(b) = {}, log(c) = {}, log(d) = {}."
        ).format(
            self.hparams[:,0], self.hparams[:,1], 
            self.hparams[:,2], self.hparams[:,3]
          )
        return msg


    def compute_distances(self, x1, x2):
        """
        The celerite.solver.CholeskySolver object does 
        not require distances to be precomputed, so 
        this method has been co-opted to define some 
        unchanging variables.
        """
        self.x = x1

        # blank matrices (corresponding to Cholesky decomp of kernel) needed for celerite solver
        self.A = np.empty(0)
        self.U = np.empty((0,0))
        self.V = self.U

    def compute_covmatrix(self, errors):
        """ Compute the Cholesky decomposition of a celerite kernel

            Args:
                errors (array of float): observation errors and jitter added
                    in quadrature

            Returns:
                celerite.solver.CholeskySolver: the celerite solver object,
                with Cholesky decomposition computed.
        """
        # initialize celerite solver object
        solver = CholeskySolver()

        self.compute_real_and_complex_hparams()
        self.real = np.exp(self.real) # (celerite hyperparameters are fit in log-space)
        self.complex = np.exp(self.complex)
        solver.compute(
            0., self.real[:,0], self.real[:,2], 
            self.complex[:,0], self.complex[:,1], 
            self.complex[:,2], self.complex[:,3], 
            self.A, self.U, self.V,
            self.x, errors**2
        )

        return solver
