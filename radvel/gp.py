import sys
import radvel
import scipy
import abc
import numpy as np

# implemented kernels
KERNELS = {"SqExp":"squared exponential",
           "Per": "periodic",
           "QuasiPer": "quasi periodic"}

if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC

class Kernel(ABC):
    """
    Abstract class to store kernel info and compute covariance matrix
    All kernel objects inherit from this class


    This class and all classes that inherit from it written by 
    Evan Sinukoff and Sarah Blunt, 2017
    """

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def compute_covmatrix(self, X1, X2):
        pass

    def add_diagonal_errors(cls, errors):
        cls.covmatrix += (errors**2) * np.identity(cls.covmatrix.shape[0])


class SqExpKernel(Kernel):

    @property
    def name(self):
        return "SqExp"

    def __init__(self, hparams, covmatrix=None):
        self.covmatrix = covmatrix

        assert len(hparams) == 2, \
        "KERNEL ERROR: incorrect number of hyperparameters passed to SqExp Kernel "

        for par in hparams:
            if par.startswith('gp_length'):
                self.length = hparams[par].value
            elif par.startswith('gp_amp'):
                self.amp = hparams[par].value

        try:
            self.length
            self.amp
        except:
            "KERNEL ERROR: SqExp Kernel requires hyperparameters 'length' and 'amp'"


    def __repr__(self):
        return "SqExp Kernel with length: {}, amp: {}".format(self.length, self.amp)

    def compute_covmatrix(self, X1, X2):
        dist = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
        K = scipy.matrix(self.amp**2 * scipy.exp(-dist/(self.length**2)))
        self.covmatrix = K


class PerKernel(Kernel):

    @property
    def name(self):
        return "Per"

    def __init__(self, hparams, covmatrix=None):
        self.covmatrix = covmatrix

        assert len(hparams) == 3, \
        "KERNEL ERROR: incorrect number of hyperparameters passed to Per Kernel "

        for par in hparams:
            if par.startswith('gp_length'):
                self.length = hparams[par].value
            if par.startswith('gp_amp'):
                self.amp = hparams[par].value
            if par.startswith('gp_per'):
                self.per = hparams[par].value

        try:
            self.length
            self.amp
            self.per
        except:
            "KERNEL ERROR: Per Kernel requires hyperparameters 'length', 'amp', and 'per'"

    def __repr__(self):
        return "Per Kernel with length: {}, amp: {}, per: {}".format(self.length, 
                                                                     self.amp, self.per)

    def compute_covmatrix(self, X1, X2):
        dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        K = scipy.matrix(self.amp**2 * scipy.exp(-np.sin(np.pi*dist/self.per)**2.
                                                 /(2.*self.length**2)))
        self.covmatrix = K

class QuasiPerKernel(Kernel):

    @property
    def name(self):
        return "QuasiPer"

    def __init__(self, hparams, covmatrix=None):
        self.covmatrix = covmatrix

        assert len(hparams) == 4, \
        "KERNEL ERROR: incorrect number of hyperparameters passed to QuasiPer Kernel "

        for par in hparams:
            if par.startswith('gp_perlength'):
                self.perlength = hparams[par].value
            if par.startswith('gp_amp'):
                self.amp = hparams[par].value
            if par.startswith('gp_per'):
                self.per = hparams[par].value
            if par.startswith('gp_explength'):
                self.explength = hparams[par].value
        try:
            self.perlength
            self.amp
            self.per
            self.explength
        except:
            "KERNEL ERROR: QuasiPer Kernel requires hyperparameters 'perlength', 'amp', 'per', and 'explength'"

    def __repr__(self):
        return "QuasiPer Kernel with amp: {}, per length: {}, per: {}, \
                exp length: {}".format(self.amp, self.perlength, self.per, self.explength)

    #Lopez-Moralez+ 2016 eq 2
    def compute_covmatrix(self, X1, X2):
        dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
        K = scipy.matrix(self.amp**2
                         * scipy.exp(-np.sin(np.pi*dist_p/self.per)**2.
                                     / (self.perlength**2))
                         * scipy.exp(-dist_se/(self.explength**2)))
        self.covmatrix = K


