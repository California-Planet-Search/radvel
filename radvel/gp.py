import sys
import radvel
import scipy
from scipy import spatial
import abc
import numpy as np

# implemented kernels & list of their associated hyperparameters
KERNELS = {"SqExp":['gp_length','gp_amp'],
           "Per": ['gp_per','gp_length','gp_amp'],
           "QuasiPer": ['gp_per','gp_perlength','gp_explength','gp_amp']}

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
        "KERNEL ERROR: incorrect number of hyperparameters passed to SqExp Kernel "

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
        except:
            print("KERNEL ERROR: SqExp Kernel requires hyperparameters 'gp_length*' and 'gp_amp*'")


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
        "KERNEL ERROR: incorrect number of hyperparameters passed to Per Kernel "

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
        except:
            print("KERNEL ERROR: Per Kernel requires hyperparameters with string" +
                  "names: 'gp_length*', 'gp_amp*', and 'gp_per*'")

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
        "KERNEL ERROR: incorrect number of hyperparameters passed to QuasiPer Kernel "

        try:
            self.hparams['gp_perlength'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except:
            "KERNEL ERROR: QuasiPer Kernel requires hyperparameters 'gp_perlength*', 'gp_amp*', 'gp_per*', and 'gp_explength*'"

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
            of this kernel. Must contain a multiple of 6 Parameters objects
            with the following names:
                a_real_k: (k is an integer) the 
    """

    @property
    def name(self):
        return "Celerite"

    def __init__(self, hparams):
        # initialize celerite solver object
        self.solver = celerite.solver.CholeskySolver()

        num_hparams = len(hparams)

        # a_real should == c_real
        self.a_real = np.zeros(num_hparams)
        self.c_real = np.zeros(num_hparams)
        self.a_comp = np.zeros(num_hparams)
        self.b_comp = np.zeros(num_hparams)
        self.c_comp = np.zeros(num_hparams)
        self.d_comp = np.zeros(num_hparams)

        for par in hparams:
            if par.startswith('a_real'):
                self.a_real[int(par[-1])] = hparams[par].value
            if par.startswith('c_real'):
                self.c_real[int(par[-1])] = hparams[par].value
            if par.startswith('a_comp'):
                self.a_comp[int(par[-1])] = hparams[par].value
            if par.startswith('b_comp'):
                self.b_comp[int(par[-1])] = hparams[par].value
            if par.startswith('c_comp'):
                self.c_comp[int(par[-1])] = hparams[par].value
            if par.startswith('d_comp'):
                self.d_comp[int(par[-1])] = hparams[par].value
          
    def __repr__(self):
        msg = (
            "Celerite Kernel with real term coefficients: a = {}, c = {}"
            " and complex term coefficients: a = {}, b = {}, c = {}, d = {}."
        ).format(
            self.a_real, self.c_real, self.a_comp, 
            self.b_comp, self.c_comp, self.d_comp
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
            self.a_real, self.c_real, self.a_comp, self.b_comp, self.c_comp, self.d_comp,
            self.A,self.U,self.V,self.dist,self.yerr**2
        )
        return self.solver

    def add_diagonal_errors(cls, errors):
        print("The celerite.solver.CholeskySolver() object adds errors along"
              + " the diagonal automatically. You should not need to use this"
              + " method with the CeleriteKernel.")
