import abc
import numpy as np
import warnings
import tinygp
from jax import numpy as jnp

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


class TwoPer(tinygp.kernels.Kernel):
    def __init__(
        self, hparams_dict, insts, inst_X1=None, inst_x2=None
    ):
        #  hparams_dict: key(param name) -> Parameter object

        self.hparams_dict = hparams_dict

        self.inst_X1 = inst_X1 # str (either None or suffix)
        self.inst_X2 = inst_x2 # str
        self.insts = insts # ['Keck', 'NEID', 'KPF']


    def evaluate(self, X1, X2):


        ampparams_groupA = jnp.array([self.hparams_dict[par].value for par in self.hparams_dict.keys() if par.startswith('gp_amp_groupA_')]) # list of param objects [amp_groupA_sub, amp_groupA_keck,amp_groupA_neid]
        ampparams_groupB = jnp.array([self.hparams_dict[par].value for par in self.hparams_dict.keys() if par.startswith('gp_amp_groupB_')])

        amp1_groupA = ampparams_groupA[X1[1]]
        amp1_groupB = ampparams_groupB[X1[1]]

        amp2_groupA = ampparams_groupA[X2[1]]
        amp2_groupB = ampparams_groupB[X2[1]]

        tau = jnp.abs(X1[0] - X2[0])

        perA = self.hparams_dict['gp_per_avg'].value + 0.5 * self.hparams_dict['gp_per_diff'].value
        perB = self.hparams_dict['gp_per_avg'].value - 0.5 * self.hparams_dict['gp_per_diff'].value

        perA_term = jnp.sin(jnp.pi * tau / perA)**2
        perB_term = jnp.sin(jnp.pi * tau / perB)**2

        perA_kernel = jnp.exp( -perA_term / (self.hparams_dict['gp_perlenA'].value**2))
        perB_kernel = jnp.exp( -perB_term / (self.hparams_dict['gp_perlenB'].value**2))

        total_kernel = (amp1_groupA * amp2_groupA *  perA_kernel) + (amp1_groupB * amp2_groupB * perB_kernel)

        return total_kernel

