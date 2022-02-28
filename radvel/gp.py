import scipy
import abc
import numpy as np
import warnings
import tinygp

warnings.simplefilter('once')

# implemented kernels & examples of their associated hyperparameters
KERNELS = {
    # "SqExp": ['gp_length', 'gp_amp'],
    # "Per": ['gp_per', 'gp_length', 'gp_amp'],
    "QuasiPer": ['gp_per', 'gp_perlength', 'gp_explength'],
}

ABC = abc.ABC


# celerite is an optional dependency
def _try_celerite():
    try:
        import celerite
        from celerite.solver import CholeskySolver
        return True
    except ImportError:
        warnings.warn("celerite not installed. GP kernals using celerite will not work. \
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


class TwoPerTiny(tinygp.kernels.Kernel):
    def __init__(
        self, hparams_dict, inst_indices, inst_X1=None, inst_x2=None
    ):
        #  hparams_dict: key(param name) -> Parameter object

        # self.perA = hparams_dict['perA']
        # self.perB = hparams_dict['per2']
        # self.perlenA = hparams_dict['perlen1']
        # self.perlenB = hparams_dict['perlen2'] #TODO: check that these automatically update when parameters array is updated

        self.amps_groupA = [hparams_dict[par] for par in hparams_dict.keys() if par.startswith('amp_groupA_')] # list of param objects [amp_groupA_sub, amp_groupA_keck,amp_groupA_neid]
        self.amps_groupB = [hparams_dict[par] for par in hparams_dict.keys() if par.startswith('amp_groupB_')] 

        self.inst_X1 = inst_X1 # str (either None or suffix)
        self.inst_X2 = inst_x2 # str
        self.inst_arr = inst_indices # {'keck':[0,1,2,7], 'sub':[3,4], 'neid':[5,6]


    def evaluate(self, X1, X2):

        amp1_groupA = jnp.empty(len(X1))
        amp2_groupA = jnp.empty(len(X2))
        amp1_groupB = jnp.empty(len(X1))
        amp2_groupB = jnp.empty(len(X2))


        if self.inst_X1 is None:

            # set amplitude array for first periodic component
            for instname in self.inst_arr.keys(): # ['keck', 'sub','neid']
                amp1_groupA[self.inst_arr[instname]] = self.hparams_dict[
                    'amp_groupA_{}'.format(instname)
                ].value

            # set amplitude array for second periodic component
            for instname in self.inst_arr.keys(): # ['keck', 'sub','neid']
                amp1_groupB[self.inst_arr[instname]] = self.hparams_dict[
                    'amp_groupB_{}'.format(instname)
                ].value
        else:
            # amp arrays all for single instrument
            amp1_groupA = self.hparams_dict['amp_groupA_{}'.format(self.inst_X1)].value
            amp1_groupB = self.hparams_dict['amp_groupB_{}'.format(self.inst_X1)].value


        if self.inst_X2 is None:

            # set amplitude array for first periodic component
            for instname in self.inst_arr.keys(): # ['keck', 'sub','neid']
                amp2_groupA[self.inst_arr[instname]] = self.hparams_dict[
                    'amp_groupA_{}'.format(instname)
                ].value

            # set amplitude array for second periodic component
            for instname in self.inst_arr.keys(): # ['keck', 'sub','neid']
                amp2_groupB[self.inst_arr[instname]] = self.hparams_dict[
                    'amp_groupB_{}'.format(instname)
                ].value
        else:
            # amp arrays all for single instrument
            amp2_groupA = self.hparams_dict['amp_groupA_{}'.format(self.inst_X1)].value
            amp2_groupB = self.hparams_dict['amp_groupB_{}'.format(self.inst_X1)].value


        tau = jnp.abs(X1 - X2)

        perA_term = jnp.sin(jnp.pi * tau / self.hparams_dict['perA'])**2
        perB_term = jnp.sin(jnp.pi * tau / self.hparams_dict['perB'])**2

        perA_kernel = jnp.exp( -perA_term / (self.hparams_dict['perlenA']**2))
        perB_kernel = jnp.exp( -perB_term / (self.hparams_dict['perlenB']**2))

        total_kernel = (amp1_groupA * amp2_groupA *  perA_kernel) + (amp1_groupB * amp2_groupB * perB_kernel)

        return total_kernel

