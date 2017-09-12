from .likelihood  import Likelihood
from .likelihood  import RVLikelihood
from .likelihood  import CompositeLikelihood
from .model  import RVModel
import scipy
from scipy.linalg import cho_factor, cho_solve
import pdb
import numpy as np
import radvel

class RVModelGP(RVModel):
    """
    RVModel object with GP hyperparams added as params via the gp object. 
    __call__ method is identical to generic RVModel (for now).
    """
    def __init__(self, params, gp, time_base=0):
        super(RVModelGP, self).__init__(params, time_base)
        self.gp = gp
        for n in range(gp.Npars):
            for tel in gp.hpars.keys():
                if n in gp.shared:
                    suffix = ''
                else:
                    suffix = '_' + tel
                self.params[gp.parnames[n]+suffix] = gp.hpars[tel][n]
        

    def __call__(self, t, planet_num=None):
        temp = RVModel(self.params, time_base=self.time_base)
        vel = temp(t, planet_num=planet_num)
        return vel
        

    
class RVLikelihoodGP(RVLikelihood):
    """RV Likelihood GP

    The RVLikelihood object for GP modeling

    Args:
        model (radvel.model.RVModelGP): RV model object, which includes gp params
        t (array): time array
        vel (array): array of velocities
        errvel (array): array of velocity uncertainties
        suffix (string): suffix to identify this Likelihood object
           useful when constructing a `CompositeLikelihood` object.

    """
    
    def __init__(self, model, t, vel, errvel, suffix='', decorr_vars=[], decorr_vectors=[]):

        super(RVLikelihoodGP, self).__init__(model, t, vel, errvel, suffix=suffix, decorr_vars=decorr_vars, decorr_vectors= decorr_vectors)
        self.gp = model.gp
        self.kernel = self.gp.kernel
        self.xpred = self.gp.xpred
        self.extra_params = [self.gamma_param, self.jit_param]
       # for i in self.gp.parnames:
       #     self.extra_params.append(i)
        self.gp_params = [i + suffix for i in self.gp.parnames]


        #ENSURE THAT IF JITTER IS ADDED TO DIAGONAL OF COVARIANCE MATRIX AS PART OF GP, THEN AVOID FITTING FOR JITTER PARAMETER SEPARATELY (AVOID HAVING TWO SEPARATE JITTER PARAMS THAT REPRESENT THE SAME THING)

    def logprob(self):
        """
        Return log-likelihood given the data and model.
        log-likelihood computed using Cholesky decomposition as:
           lnL = -0.5*r.T*inverse(K)*r - 0.5*ln[det(K)] - N*ln(2pi)/2, 
           r = residuals vector, K = covariance matrix, N = number of datapoints.  
        Priors are not applied here.

        Returns:
            float: Natural log of likelihood
        """
        hpars = [self.params[self.gp.parnames[n]+'_'+self.suffix] for n in range(self.gp.Npars)]
        r = scipy.matrix([self.residuals()]).T
        K = self.kernel.cov(self.x, self.x, hpars, yerr = self.yerr, jitter = self.params[self.jit_param])
        #Solve b = inverse(K)*r, K = L*L.T
        L = cho_factor(K)
        b = cho_solve(L, r)

        #sign, logKdet = np.linalg.slogdet(K)
        logKdet = 2.0*np.sum(np.log(np.diag(L[0])))
        loglike = -(len(r)/2.)*np.log(2*np.pi) - 0.5 * (np.dot(r.T,b) + logKdet)

        return loglike[0,0]

    

class CompositeLikelihoodGP(CompositeLikelihood):

    def __init__(self, like_list):
         
        """Composite LikelihoodGP

        A thin wrapper to combine multiple GP `Likelihood`
        objects. One `Likelihood` applies to a dataset from
        a particular instrument.

        Args:
            like_list (list): list of `radvel.likelihood.RVLikelihoodGP` objects
        """
        super(CompositeLikelihoodGP, self).__init__(like_list)
        self.gp_params = like_list[0].gp_params
        for i in range(1,self.nlike):
            like = like_list[i]
            self.gp_params = np.append(self.gp_params,self.like.gp_params)
            

        
    def logprob(self):
        """
        See `RVLikelihoodGP.logprob`
        """
        
        _logprob = 0
        for like in self.like_list:
            _logprob += like.logprob()
        return _logprob

    

class GP(object):
    """
    Object to store GP parameters

    Args:
        kern (string): Name of GP kernel. 
                       Name and number of hyperparams must be defined in Kernel class.
        hyperparams (dict): Keys are instument names (str). 
                            Values are lists of hyperparams in order defined by Kernel class. 
                            Same kernel for each instrument, thus same number of hyperparams. 
        parnames (list of string(s)): Names assigned to hyperparameters. No suffixes here.

        shared:  List of hyperparams to be fit as single free param for all instruments 
                 (not yet functional)

        xpred (array, optional): X values at which to compute predictive mean and sigma for plotting purposes.
                                 Default is 10000 evenly spaced values.  

        plot_sigma (List, optional): Confidence interval (units of sigma) over which to shade predictive distribution. Will overplot multiple intervals if multiple values given.                            
    """
    
    def __init__(self, kern, hyperparams, parnames, shared=None, xpred=None, plot_sigma=[1]):
        self.kernel = Kernel(name=kern)
        self.hpars = hyperparams
        self.parnames = np.array(parnames)
        self.shared = np.array(shared)
        self.xpred = xpred
        self.plot_sigma = np.array(plot_sigma)
        self.kernel.check_pars(self.hpars, self.parnames, self.shared)
        self.Npars = len(self.hpars.values()[0])
        
    
    def predict(self, x, y, xpred, hpars, yerr=None, jitter=None, err = False):
        #mu = array of predictive means computed at x = xpred.
        #stdev = array of sqrt of predictive variance computed at x = xpred
        K = self.kernel.cov(x, x, hpars, yerr=yerr, jitter=jitter)
        Ks = self.kernel.cov(xpred, x, hpars)
        L = cho_factor(K)
        r = scipy.matrix(y).T
        b = cho_solve(L, r) 
        mu = np.array(Ks*scipy.matrix(b)).flatten()
        if err:
           Kss = self.kernel.cov(xpred, xpred, hpars)
           b = cho_solve(L, Ks.T) 
           stdev = scipy.array(scipy.sqrt(np.diag(Kss - Ks * scipy.matrix(b)))).flatten()
           return mu, stdev
        else:
           return mu

       
    def sample_conditional(self, x, xpred, hpars, yerr=None, jitter=None, n=1):
        
        K = self.kernel.cov(x, x, hpars, yerr=yerr, jitter=jitter) 
        Ks = self.kernel.cov(xpred, x, hpars)
        Kss = self.kernel.cov(xpred, xpred, hpars)
        L = cho_factor(K)
        
        b = cho_solve(L, Ks.T)
        cov = Kss - Ks * scipy.matrix(b) #predictive covariance matrix
        
        for i in range(n):
            samples = radvel.utils.multivariate_normal_sample(cov, mean=None, n=n)  

        return samples
       


class Kernel(object):
    """
    Object to store kernel info and compute covariance matrix

    Args:
        name (string): Name of GP kernel. 
                       New kernels can be added/modified by: 
                       1. updating the "kernels" attribute {name:N_hyperparams} AND
                       2. In method "cov", defining computation of covariance matrix if name = (new kernel name)   
    """
    
    
    def __init__(self, name):
        self.name=name
        #Continue to add new Kernels w/ respective number of hyperparameters
        self.kernels = {'SqExp':2,
                        'Periodic':3
                            }
        self.check_type()

    def check_type(self):
         assert self.name in self.kernels.keys(), \
            'GP Kernel not recognized: ' + self.name + '\n' + 'Available kernels: ' + self.kernels.keys()

    def check_pars(self, hpars, parnames, shared):
        tels = hpars.keys()
        Npars_kernel = self.kernels[self.name]
        for i in tels:
            assert len(hpars[i]) == Npars_kernel, \
              'GP kernel ' + "'" + self.name + "'" + \
              'requires exactly ' + str(Npars_kernel) + \
              ' hyperparameters (' + str(len(hpars[i])) + \
              ' given for instrument' + "'" + i + "')"
        assert len(parnames) == Npars_kernel, \
                'Length of list of names for GP hyperparameters (' + \
                + str(len(parnames)) + ') does not match number of' + \
                'hyperparameters (' + str(len(parnames)) + ')'
        if shared is not None:
            for i in shared:
                    vals = [hpars[tel][i] for tel in hpars.keys()]
                    assert vals.count(vals[0]) == len(vals),  \
                     "Initial values of 'shared' parameters" + \
                     ' must be set equal for all instruments'
        return 

    def cov(self, x1, x2, hpars, yerr = None, jitter = None):

        #Method can be used to compute K, Ks, or Kss depending on x1, x2.
        
        X1 = scipy.matrix([x1]).T
        X2 = scipy.matrix([x2]).T 

        #Continue to add new Kernels here
        if self.name == 'SqExp':
            dist = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
            K = scipy.matrix(hpars[0]**2 * scipy.exp(-dist/(2.*hpars[1]**2)))
        elif self.name == 'Periodic':
            dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
            K = scipy.matrix(hpars[0]**2 * scipy.exp(-np.sin(np.pi*dist/hpars[1])**2./(2.*hpars[2]**2)))

        if yerr is not None:
            K += yerr**2.*scipy.identity(X1.shape[0])

        if jitter is not None:
            K += jitter**2.*scipy.identity(X1.shape[0])

        return K
 
