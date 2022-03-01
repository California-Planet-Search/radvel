import warnings
import tinygp
from jax import numpy as jnp

warnings.simplefilter('once')

# implemented kernels & examples of their associated hyperparameters
KERNELS = {
    "TwoPer": [
        'gp_per_avg', 'gp_per_diff', 'gp_perlenA', 'gp_perlenB', 
        'gp_amp_groupA_','gp_amp_groupB_'
    ],
    "SqExp": ['gp_length', 'gp_amp_'],
    "QuasiPer": ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp_']
}

class SqExp(tinygp.kernels.Kernel):
    """
    Squared-exponential kernel. An arbitrary element, :math:`C_{ij}`, of the 
    matrix is:
    
    .. math::
        C_{ij} = \\eta_1^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } )

    Args:
        hparams_dict (dict of radvel.Parameter objects): same as 
            Likelihood.params, passed into this object by radvel.Likelihood object
        insts (list of str): list of all instruments, e.g. ['HIRES', 'NEID', 'KPF']
    """
    def __init__(self, hparams_dict, insts):
        self.hparams_dict = hparams_dict
        self.insts = insts

    def evaluate(self, X1, X2):

        # build lists of amplitude values [gp_amp_j, gp_amp_k]
        ampparams = jnp.array([
            self.hparams_dict[par].value for par in self.hparams_dict.keys() if 
            par.startswith('gp_amp_')
        ])

        amp1 = ampparams[X1[1]]
        amp2 = ampparams[X2[1]]

        tau = jnp.abs(X1[0] - X2[0])

        exp_kernel = jnp.exp(
            -tau**2 / 
            self.hparams_dict['gp_explen'].value**2
        )

        total_kernel = amp1 * amp2 * exp_kernel
        return total_kernel


class QuasiPer(tinygp.kernels.Kernel):
    """
    Quasi-periodic kernel of Lopez-Moralez+ 2016. An arbitrary element, 
    :math:`C_{ij}`, of the matrix is:

    .. math::
        C_{ij} = \\eta_1^2 * exp( \\frac{ -\\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3^2 } ) }{\\eta_2^2 } )

    Args:
        hparams_dict (dict of radvel.Parameter objects): same as 
            Likelihood.params, passed into this object by radvel.Likelihood object
        insts (list of str): list of all instruments, e.g. ['HIRES', 'NEID', 'KPF']
    """
    def __init__(self, hparams_dict, insts):
        self.hparams_dict = hparams_dict
        self.insts = insts

    def evaluate(self, X1, X2):

        # build lists of amplitude values [gp_amp_j, gp_amp_k]
        ampparams = jnp.array([
            self.hparams_dict[par].value for par in self.hparams_dict.keys() if 
            par.startswith('gp_amp_')
        ])

        amp1 = ampparams[X1[1]]
        amp2 = ampparams[X2[1]]

        tau = jnp.abs(X1[0] - X2[0])

        per_term = jnp.sin(jnp.pi * tau / self.hparams_dict['gp_per'].value)**2
        per_kernel = jnp.exp( 
            -per_term / 
            (self.hparams_dict['gp_perlen'].value**2)
        )

        exp_kernel = jnp.exp(
            -tau**2 / 
            self.hparams_dict['gp_explen'].value**2
        )

        total_kernel = amp1 * amp2 * exp_kernel * per_kernel
        return total_kernel

class TwoPer(tinygp.kernels.Kernel):
    """
    Double periodic kernel. Blunt et al (in prep, est 2022).

    Args:
        hparams_dict (dict of radvel.Parameter objects): same as 
            Likelihood.params, passed into this object by radvel.Likelihood object
        insts (list of str): list of all instruments, e.g. ['HIRES', 'NEID', 'KPF']
    """
    def __init__(
        self, hparams_dict, insts
    ):

        self.hparams_dict = hparams_dict
        self.insts = insts

    def evaluate(self, X1, X2):

        # build lists of amplitude values 
        #   [amp_groupA_HIRES, amp_groupA_k_NEID, amp_groupA_k_KPF]
        ampparams_groupA = jnp.array([
            self.hparams_dict[par].value for par in self.hparams_dict.keys() if 
            par.startswith('gp_amp_groupA_')
        ])
        ampparams_groupB = jnp.array([
            self.hparams_dict[par].value for par in self.hparams_dict.keys() if 
            par.startswith('gp_amp_groupB_')
        ])

        amp1_groupA = ampparams_groupA[X1[1]]
        amp1_groupB = ampparams_groupB[X1[1]]

        amp2_groupA = ampparams_groupA[X2[1]]
        amp2_groupB = ampparams_groupB[X2[1]]

        tau = jnp.abs(X1[0] - X2[0])

        perA = (
            self.hparams_dict['gp_per_avg'].value + 
            0.5 * self.hparams_dict['gp_per_diff'].value
        )
        perB = (
            self.hparams_dict['gp_per_avg'].value - 
            0.5 * self.hparams_dict['gp_per_diff'].value
        )

        perA_term = jnp.sin(jnp.pi * tau / perA)**2
        perB_term = jnp.sin(jnp.pi * tau / perB)**2

        perA_kernel = jnp.exp(
            -perA_term / 
            (self.hparams_dict['gp_perlenA'].value**2)
        )
        perB_kernel = jnp.exp( 
            -perB_term / 
            (self.hparams_dict['gp_perlenB'].value**2)
        )

        total_kernel = (
            (amp1_groupA * amp2_groupA *  perA_kernel) + 
            (amp1_groupB * amp2_groupB * perB_kernel)
        )

        return total_kernel

