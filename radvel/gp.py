import sys
import radvel
import scipy
from scipy.linalg import cho_factor, cho_solve
from abc import ABCMeta, abstractproperty, abstractmethod
import numpy as np

# implemented kernels
KERNELS = {"SqExp":"squared exponential",
           "Per": "periodic",
           "QuasiPer": "quasi periodic"}

## TO PUT IN GP LIKELIHOOD
#    X1 = scipy.matrix([x1]).T
#    X2 = scipy.matrix([x2]).T 

if sys.version_info[0] < 3:
    ABC = ABCMeta('ABC', (), {})

class Kernel(ABC):
    """
    Abstract class to store kernel info and compute covariance matrix
    All kernel objects inherit from this class


    This class and all classes that inherit from it written by 
    Evan Sinukoff and Sarah Blunt, 2017
    """

    @abstractproperty
    def name(self):
    	pass

    @abstractmethod
    def compute_covmatrix(self, X1, X2):   
	"""Abstract method for computing a covariance matrix."""
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
			if par.startswith('length'):
				self.length = params[par].value
			if par.startswith('amp'):
				self.amp = params[par].value

	def compute_covmatrix(self, X1, X2):
		dist = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
		K = scipy.matrix(self.amp**2 * scipy.exp(-dist/(2.*self.length**2)))
		self.covmatrix = K


class PerKernel(Kernel):

	@property
	def name(self):
		return "Per"

	def __init__(self, params, covmatrix=None):
		self.covmatrix = covmatrix
		for par in params:
			if par.startswith('length'):
				self.length = params[par].value
			if par.startswith('amp'):
				self.amp = params[par].value
			if par.startswith('length'):
				self.length = params[par].value

	def compute_covmatrix(self, X1, X2):
		dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
		K = scipy.matrix(self.amp**2 * scipy.exp(-2.*np.sin(np.pi*dist/self.per)**2.
												 /(self.length**2)))

class QuasiPerKernel(Kernel):
	
	@property
	def name(self):
		return "QuasiPer"

	def __init__(self, params, covmatrix=None):
		self.covmatrix = covmatrix
		for par in params:
			if par.startswith('length'):
				self.length = params[par].value
			if par.startswith('amp'):
				self.amp = params[par].value
			if par.startswith('length'):
				self.length = params[par].value

	def compute_covmatrix(self, X1, X2):
		dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
		dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
		K = scipy.matrix(self.amp**2
						 * scipy.exp(-2.*np.sin(np.pi*dist_p/self.per)**2.
						 			 / (self.length**2))
						 * scipy.exp(-dist_se/(2.*self.length**2)))

if __name__ == "__main__":
	x = scipy.matrix([1,2]).T
	p = SqExpKernel(1,2)
	p.compute_covmatrix(x,x)
	print(p.covmatrix)
	p.add_diagonal_errors(1.)
	print(p.covmatrix)
	p.add_diagonal_errors(np.array([1.,2.]))
	print(p.covmatrix)

