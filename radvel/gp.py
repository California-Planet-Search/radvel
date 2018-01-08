import sys
import radvel
import scipy
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


