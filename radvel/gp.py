import sys
import radvel
import scipy
from scipy import spatial
import abc
import numpy as np
import celerite

# implemented kernels & list of their associated hyperparameters
KERNELS = {"SqExp":['gp_length','gp_amp'],
           "Per": ['gp_per','gp_length','gp_amp'],
           "QuasiPer": ['gp_per','gp_perlength','gp_explength','gp_amp'],
           "Celerite": []}

           # TODO: update kernel params for celerite

if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC

class Kernel(ABC):
    """
    Abstract base class to store kernel info and compute covariance matrix.
    All kernel objects inherit from this class

    """

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def compute_distances(self, X1, X2):
        pass

    @abc.abstractmethod
    def compute_covmatrix(self):
        pass

    def add_diagonal_errors(cls, errors):
        cls.covmatrix += (errors**2) * np.identity(cls.covmatrix.shape[0])
        return cls.covmatrix


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
        "SqExpKernel requires exactly 2 hyperparameters with names 'gp_length*' and 'gp_amp*'."

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
        except KeyError:
            raise KeyError("SqExpKernel requires hyperparameters 'gp_length*' and 'gp_amp*'.")
        except AttributeError:
            raise AttributeError("SqExpKernel requires dictionary of Parameter objects as input")



    def __repr__(self):
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        return "SqExp Kernel with length: {}, amp: {}".format(length, amp)

    def compute_distances(self, X1, X2):
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    def compute_covmatrix(self):
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value

        K = scipy.matrix(amp**2 * scipy.exp(-self.dist/(length**2)))
        self.covmatrix = K
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
        "PerKernel requires exactly 3 hyperparameters with names 'gp_length*', 'gp_amp*', and 'gp_per*'."

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
        except KeyError:
            raise KeyError("PerKernel requires hyperparameters 'gp_length*', 'gp_amp*', and 'gp_per*'.")
        except AttributeError:
            raise AttributeError("PerKernel requires dictionary of Parameter objects as input")

    def __repr__(self):
        length= self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        return "Per Kernel with length: {}, amp: {}, per: {}".format(length, amp, per)

    def compute_distances(self, X1, X2):
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')

    def compute_covmatrix(self):
        length= self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value

        K = scipy.matrix(amp**2 * scipy.exp(-np.sin(np.pi*self.dist/per)**2.
                                                 / (2.*length**2)))
        self.covmatrix = K
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
            if par.startswith('gp_per'):
                self.hparams['gp_per'] = hparams[par]
            if par.startswith('gp_explength'):
                self.hparams['gp_explength'] = hparams[par]

        assert len(hparams) == 4, \
        "QuasiPerKernel requires exactly 4 hyperparameters with names 'gp_perlength*', 'gp_amp*', 'gp_per*', and 'gp_explength*'."

        try:
            self.hparams['gp_perlength'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except KeyError:
            raise KeyError("QuasiPerKernel requires hyperparameters 'gp_perlength*', 'gp_amp*', 'gp_per*', and 'gp_explength*'.")
        except AttributeError:
            raise AttributeError("QuasiPerKernel requires dictionary of Parameter objects as input")

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

    def compute_distances(self, X1, X2):
        self.dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    def compute_covmatrix(self):
        perlength = self.hparams['gp_perlength'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        K = scipy.matrix(amp**2
                         * scipy.exp(-self.dist_se/(explength**2))
                         * scipy.exp((-np.sin(np.pi*self.dist_p/per)**2.)
                                      / (2.*perlength**2)))
        self.covmatrix = K
        return self.covmatrix

class CeleriteKernel(Kernel):
    """ A wrapper for the celerite.solver.CholeskySolver() methods and attributes.

    Class that computes and stores a kernel that can be modeled as the sum of 
    celerite terms, or kernels of the following form:

    .. math::

        C_{ij} = \\sum_\\limits_{n=1}^{J} 
                 \frac{1}{2}(a_n + ib_n)exp(-(c_n + id_n)|t_i - t_j|)
                 + \frac{1}{2}(a_n - ib_n)exp(-(c_n - id_n)|t_i - t_j|)

    where J is the numner of terms in the sum. The hyperparameters of 
    this kernel are :math:`a_j`,:math:`b_j`,:math:`c_j`, and :math:`d_j` 
    for each term :math:`C_{nm}` in the sum. 

    See celerite.readthedocs.io for more information about celerite kernels and
    computation.

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel.

        # TODO: better document string syntax for hyperparams.
    """

    @property
    def name(self):
        return "Celerite"

    def __init__(self, hparams):
        # initialize celerite solver object
        self.solver = celerite.solver.CholeskySolver()

        # define number of real and number of complex terms
        num_real_params = 0
        num_complex_params = 0
        print(hparams.keys())
        for par in hparams.keys():
            if 'Real' in par and int(par[0]) >= num_real_params:
                num_real_params = int(par[0])
            if 'Comp' in par and int(par[0]) >= num_complex_params:
                num_complex_params = int(par[0])

        self.aReal = np.zeros(num_real_params) + np.nan
        self.cReal = np.zeros(num_real_params) + np.nan
        self.aComp = np.zeros(num_complex_params) + np.nan
        self.bComp = np.zeros(num_complex_params) + np.nan
        self.cComp = np.zeros(num_complex_params) + np.nan
        self.dComp = np.zeros(num_complex_params) + np.nan

        for par in hparams:
            try:
                if 'aReal' in par:
                    self.aReal[int(par[0]) - 1] = hparams[par].value
                if 'cReal' in par:
                    self.cReal[int(par[0]) - 1] = hparams[par].value
                if 'aComp' in par:
                    self.aComp[int(par[0]) - 1] = hparams[par].value
                if 'bComp' in par:
                    self.bComp[int(par[0]) - 1] = hparams[par].value
                if 'cComp' in par:
                    self.cComp[int(par[0]) - 1] = hparams[par].value
                if 'dComp' in par:
                    self.dComp[int(par[0]) - 1] = hparams[par].value
            except IndexError:
                print("Celerite Kernel requires an equal number of *_aReal* and *_cReal* hyperparameters," 
                      + "and an equal number of *_aComp*, *_bComp*, *_cComp*, and *_dComp* hyperparameters.")

        print(self.bComp)
        print(self.aComp)
        print(self.cComp)
        print(self.dComp)
        print(self.aReal)
        print(self.cReal)
        # TODO: test error messages
        # TODO: modify test_api to test celerite kernel appropriately

        assert np.nan not in \
            np.array([self.aReal,self.cReal]).flatten(), \
            "Celerite Kernel requires an equal number of *_aReal* and *_cReal* hyperparameters."

        assert np.nan not in \
            np.array([self.aComp,self.bComp,self.cComp,self.dComp]).flatten(), \
            "Celerite Kernel requires an equal number of *_aComp*, *_bComp*, *_cComp*, and *_dComp* hyperparameters."

    def __repr__(self):
        msg = (
            "Celerite Kernel with real term coefficients: a = {}, c = {}"
            " and complex term coefficients: a = {}, b = {}, c = {}, d = {}."
        ).format(
            self.aReal, self.cReal, self.aComp, 
            self.bComp, self.cComp, self.dComp
          )
        return msg

    def compute_distances(self, X1, X2):
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')

        # blank matrices needed for celerite solver
        self.A = np.zeros(len(self.x))
        self.U = np.zeros((len(self.x),len(self.x)))
        self.V = self.U

    def compute_covmatrix(self):
        """ Compute the Cholesky decomposition of a celerite kernel

            Returns:
                celerite.solver.CholeskySolver: the celerite solver object,
                with Cholesky decomposition computed.
        """
        self.solver.compute(
            self.params[self.jit_param].value, 
            self.aReal, self.cReal, self.aComp, self.bComp, self.cComp, self.dComp,
            self.A,self.U,self.V,self.dist,self.yerr**2
        )
        return self.solver

    def add_diagonal_errors(self, errors):
        print("The celerite.solver.CholeskySolver() object adds errors along"
              + " the diagonal automatically. You should not need to use this"
              + " method with the CeleriteKernel.")


if __name__ == "__main__":
    params = {'gp_length_hires':  radvel.Parameter(value=1.), 'gp_length_harps':  radvel.Parameter(value=1.)}

    a = SqExpKernel(params)
