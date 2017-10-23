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
        cls.covmatrix += (errors**2.) * np.identity(cls.covmatrix.shape[0])


class SqExpKernel(Kernel):

    @property
    def name(self):
        return "SqExp"

    def __init__(self, params, covmatrix=None):
        self.covmatrix = covmatrix
        for par in params:
            if par.startswith('gp_length'):
                self.length = params[par].value
            if par.startswith('gp_amp'):
                self.amp = params[par].value

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

    def __init__(self, params, covmatrix=None):
        self.covmatrix = covmatrix
        for par in params:
            if par.startswith('gp_length'):
                self.length = params[par].value
            if par.startswith('gp_amp'):
                self.amp = params[par].value
            if par.startswith('gp_per'):
                self.per = params[par].value

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

    def __init__(self, params, covmatrix=None):
        self.covmatrix = covmatrix
        for par in params:
            if par.startswith('gp_length'):
                self.length = params[par].value
            if par.startswith('gp_amp'):
                self.amp = params[par].value
            if par.startswith('gp_per'):
                self.per = params[par].value

    def __repr__(self):
        return "QuasiPer Kernel with length: {}, amp: {}, per: {}".format(self.length, 
                                                                          self.amp, self.per)

    def compute_covmatrix(self, X1, X2):
        dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
        K = scipy.matrix(self.amp**2
                         * scipy.exp(-np.sin(np.pi*dist_p/self.per)**2.
                                     / (2.*self.length**2))
                         * scipy.exp(-dist_se/(self.length**2)))
        self.covmatrix = K


