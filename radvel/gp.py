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
    def compute_distances(self, X1, X2):
        pass

    @abc.abstractmethod
    def compute_covmatrix(self):
        pass

    def add_diagonal_errors(cls, errors):
        cls.covmatrix += (errors**2) * np.identity(cls.covmatrix.shape[0])
        return cls.covmatrix


class SqExpKernel(Kernel):

    @property
    def name(self):
        return "SqExp"

    def __init__(self, hparams, covmatrix=None):
        self.hparams = hparams
        self.covmatrix = covmatrix

        assert len(hparams) == 2, \
        "KERNEL ERROR: incorrect number of hyperparameters passed to SqExp Kernel "

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
        except:
            "KERNEL ERROR: SqExp Kernel requires hyperparameters 'gp_length' and 'gp_amp'"


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

    @property
    def name(self):
        return "Per"

    def __init__(self, hparams, covmatrix=None):
        self.hparams = hparams
        self.covmatrix = covmatrix

        assert len(hparams) == 3, \
        "KERNEL ERROR: incorrect number of hyperparameters passed to Per Kernel "

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
        except:
            "KERNEL ERROR: Per Kernel requires hyperparameters 'gp_length', 'gp_amp', and 'gp_per'"

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

    @property
    def name(self):
        return "QuasiPer"

    def __init__(self, hparams, covmatrix=None):
        self.hparams = hparams
        self.covmatrix = covmatrix

        assert len(hparams) == 4, \
        "KERNEL ERROR: incorrect number of hyperparameters passed to QuasiPer Kernel "

        try:
            self.hparams['gp_perlength'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except:
            "KERNEL ERROR: QuasiPer Kernel requires hyperparameters 'gp_perlength', 'gp_amp', 'gp_per', and 'gp_explength'"

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

        # Grunblatt+ 2015 Table 2
        K = scipy.matrix(amp**2
                         * scipy.exp((-np.sin(np.pi*self.dist_p/per)**2.)
                                     / (2.*perlength**2))
                         * scipy.exp(-self.dist_se/(explength**2)))
        self.covmatrix = K
        return self.covmatrix



if __name__ == "__main__":
    hparams = {
            'gp_perlength': radvel.Parameter(value=1.),
            'gp_amp' : radvel.Parameter(value=1.),
            'gp_per' : radvel.Parameter(value=1.),
            'gp_explength' : radvel.Parameter(value=1.)
    }
    my_kern = QuasiPerKernel(hparams)
    x = np.array([np.array([1,2,3])]).T
    my_kern.compute_distances(x,x)
    my_kern.compute_covmatrix()
    print(my_kern)



