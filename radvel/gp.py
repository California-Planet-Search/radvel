import sys
from scipy import spatial
import abc
import numpy as np
import warnings

warnings.simplefilter("once")

# implemented kernels & examples of their associated hyperparameters
KERNELS = {
    "SqExp": ["gp_length", "gp_amp"],
    "Per": ["gp_per", "gp_length", "gp_amp"],
    "QuasiPer": ["gp_per", "gp_perlength", "gp_explength", "gp_amp"],
    "CeleriteQuasiPer": ["gp_B", "gp_C", "gp_L", "gp_Prot"],
    "CeleriteSHO": ["gp_S0", "gp_Q", "gp_w0"],
    "CeleriteMatern32": ["gp_sigma", "gp_rho"],
}

if sys.version_info[0] < 3:
    ABC = abc.ABCMeta("ABC", (), {})
else:
    ABC = abc.ABC


# celerite is an optional dependency
try:
    from celerite.solver import CholeskySolver, get_kernel_value

    HAS_CELERITE = True
except ImportError:
    warnings.warn(
        "celerite not installed. GP kernals using celerite will not work. \
Try installing celerite using 'pip install celerite'",
        ImportWarning,
    )
    HAS_CELERITE = False


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
            if par.startswith("gp_length"):
                self.hparams["gp_length"] = hparams[par]
            if par.startswith("gp_amp"):
                self.hparams["gp_amp"] = hparams[par]

        assert len(hparams) == 2, (
            "{}Kernel requires exactly 2 hyperparameters with names".format(self.name)
            + "'gp_length*' and 'gp_amp*'."
        )

        try:
            self.hparams["gp_length"].value
            self.hparams["gp_amp"].value
        except KeyError:
            raise KeyError(
                "{}Kernel requires hyperparameters 'gp_length*'".format(self.name)
                + " and 'gp_amp*'."
            )
        except AttributeError:
            raise AttributeError(
                "{}Kernel requires dictionary of".format(self.name)
                + " radvel.Parameter objects as input."
            )

    def __repr__(self):
        length = self.hparams["gp_length"].value
        amp = self.hparams["gp_amp"].value
        return "{} Kernel with length: {}, amp: {}".format(self.name, length, amp)

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist = spatial.distance.cdist(X1, X2, "sqeuclidean")

    def compute_covmatrix(self, errors):
        """Compute the covariance matrix, and optionally add errors along
        the diagonal.

        Args:
            errors (float or numpy array): If covariance matrix is non-square,
                this arg must be set to 0. If covariance matrix is square,
                this can be a numpy array of observational errors and jitter
                added in quadrature.
        """
        length = self.hparams["gp_length"].value
        amp = self.hparams["gp_amp"].value

        K = amp ** 2 * np.exp(-self.dist / (length ** 2))

        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors ** 2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

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
            if par.startswith("gp_length"):
                self.hparams["gp_length"] = hparams[par]
            if par.startswith("gp_amp"):
                self.hparams["gp_amp"] = hparams[par]
            if par.startswith("gp_per"):
                self.hparams["gp_per"] = hparams[par]

        assert len(hparams) == 3, (
            "{}Kernel requires exactly 3 hyperparameters with names 'gp_length*',".format(
                self.name
            )
            + " 'gp_amp*', and 'gp_per*'."
        )

        try:
            self.hparams["gp_length"].value
            self.hparams["gp_amp"].value
            self.hparams["gp_per"].value
        except KeyError:
            raise KeyError(
                "{}Kernel requires hyperparameters 'gp_length*',".format(self.name)
                + " 'gp_amp*', and 'gp_per*'."
            )
        except AttributeError:
            raise AttributeError(
                "{}Kernel requires dictionary of ".format(self.name)
                + "radvel.Parameter objects as input."
            )

    def __repr__(self):
        length = self.hparams["gp_length"].value
        amp = self.hparams["gp_amp"].value
        per = self.hparams["gp_per"].value
        return "{} Kernel with length: {}, amp: {}, per: {}".format(
            self.name, length, amp, per
        )

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist = spatial.distance.cdist(X1, X2, "euclidean")

    def compute_covmatrix(self, errors):
        """Compute the covariance matrix, and optionally add errors along
        the diagonal.

        Args:
            errors (float or numpy array): If covariance matrix is non-square,
                this arg must be set to 0. If covariance matrix is square,
                this can be a numpy array of observational errors and jitter
                added in quadrature.
        """
        length = self.hparams["gp_length"].value
        amp = self.hparams["gp_amp"].value
        per = self.hparams["gp_per"].value

        K = amp ** 2 * np.exp(
            -np.sin(np.pi * self.dist / per) ** 2.0 / (2.0 * length ** 2)
        )
        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors ** 2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix


class QuasiPerKernel(Kernel):
    """
    Class that computes and stores a quasi periodic kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } -
                 \\frac{ \\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3 } ) }{ 2\\eta_4^2 } )

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
            if par.startswith("gp_perlength"):
                self.hparams["gp_perlength"] = hparams[par]
            if par.startswith("gp_amp"):
                self.hparams["gp_amp"] = hparams[par]
            if par.startswith("gp_per") and "length" not in par:
                self.hparams["gp_per"] = hparams[par]
            if par.startswith("gp_explength"):
                self.hparams["gp_explength"] = hparams[par]

        assert len(hparams) == 4, (
            "{}Kernel requires exactly 4 hyperparameters with names".format(self.name)
            + " 'gp_perlength*', 'gp_amp*', 'gp_per*', and 'gp_explength*'."
        )

        try:
            self.hparams["gp_perlength"].value
            self.hparams["gp_amp"].value
            self.hparams["gp_per"].value
            self.hparams["gp_explength"].value
        except KeyError:
            raise KeyError(
                "{}Kernel requires hyperparameters".format(self.name)
                + " 'gp_perlength*', 'gp_amp*', 'gp_per*', "
                + "and 'gp_explength*'."
            )
        except AttributeError:
            raise AttributeError(
                "{}Kernel requires dictionary of".format(self.name)
                + " radvel.Parameter objects as input."
            )

    def __repr__(self):
        perlength = self.hparams["gp_perlength"].value
        amp = self.hparams["gp_amp"].value
        per = self.hparams["gp_per"].value
        explength = self.hparams["gp_explength"].value

        msg = (
            "{} Kernel with amp: {}, per length: {}, per: {}, " "exp length: {}"
        ).format(self.name, amp, perlength, per, explength)
        return msg

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist_p = spatial.distance.cdist(X1, X2, "euclidean")
        self.dist_se = spatial.distance.cdist(X1, X2, "sqeuclidean")

    def compute_covmatrix(self, errors):
        """Compute the covariance matrix, and optionally add errors along
        the diagonal.

        Args:
            errors (float or numpy array): If covariance matrix is non-square,
                this arg must be set to 0. If covariance matrix is square,
                this can be a numpy array of observational errors and jitter
                added in quadrature.
        """
        perlength = self.hparams["gp_perlength"].value
        amp = self.hparams["gp_amp"].value
        per = self.hparams["gp_per"].value
        explength = self.hparams["gp_explength"].value

        K = np.array(
            amp ** 2
            * np.exp(-self.dist_se / (explength ** 2))
            * np.exp(
                (-np.sin(np.pi * self.dist_p / per) ** 2.0) / (2.0 * perlength ** 2)
            )
        )

        self.covmatrix = K

        # add errors along the diagonal
        try:
            self.covmatrix += (errors ** 2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix


class CeleriteKernel(Kernel):
    """
    Abstract class for celerite kernels with methods to pre-compute values
    """

    def compute_distances(self, x1, x2):
        """
        The celerite.solver.CholeskySolver object does
        not require distances to be precomputed, so
        this method has been co-opted to define some
        unchanging variables.
        """
        self.x = x1

        # blank matrices (corresponding to Cholesky decomp of kernel) needed for celerite solver
        self.A = np.empty(0)
        self.U = np.empty((0, 0))
        self.V = self.U

    @property
    def coefficients(self):

        self.compute_real_and_complex_hparams()

        r = [self._get_coeff(self.real[:, i]) for i in range(0, 3, 2)]
        c = [self._get_coeff(self.complex[:, i]) for i in range(4)]

        return r + c

    def _get_coeff(self, c):
        return np.empty(0) if np.isnan(c) else c

    def compute_covmatrix(self, errors):
        """Compute the Cholesky decomposition of a celerite kernel

        Args:
            errors (array of float): observation errors and jitter added
                in quadrature

        Returns:
            celerite.solver.CholeskySolver: the celerite solver object,
            with Cholesky decomposition computed.
        """
        # initialize celerite solver object
        solver = CholeskySolver()

        self.compute_real_and_complex_hparams()
        solver.compute(
            0.0,
            self._get_coeff(self.real[:, 0]),
            self._get_coeff(self.real[:, 2]),
            self._get_coeff(self.complex[:, 0]),
            self._get_coeff(self.complex[:, 1]),
            self._get_coeff(self.complex[:, 2]),
            self._get_coeff(self.complex[:, 3]),
            self.A,
            self.U,
            self.V,
            self.x,
            errors ** 2,
        )

        self.covmatrix = self.get_matrix(errors=errors)

        return solver

    def get_value(self, tau):
        """Compute value of the kernel for an array of lag values

        Args:
            tau (np.array): Lags (ti-tj)
        Returns:
            k (np.array): Kernel computed for each tau value, same shape as tau
        """
        tau = np.asarray(tau)

        k = get_kernel_value(
            *self.coefficients,
            tau.flatten(),
        )

        return np.asarray(k).reshape(tau.shape)

    def get_matrix(
        self,
        x1=None,
        x2=None,
        errors=None,
        include_diagonal=None,
        include_general=None,
    ):
        """
        Get the covariance matrix at given independent coordinates
        Modified version of
        https://github.com/dfm/celerite/blob/main/celerite/celerite.py#L476

        Args:
            x1 (Optional[np.array]): The first set of independent coordinates.
                If this is omitted, ``x1`` will be assumed to be equal to ``x``
                from a previous call to
                :func:`CeleriteKernel.compute_distances`.
            x2 (Optional[np.array]): The second set of independent
                coordinates. If this is omitted, ``x2`` will be assumed to be
                ``x1``.
            include_diagonal (Optional[bool]): Should the white noise and
                ``yerr`` terms be included on the diagonal?
                (default: ``False``)
        """

        if errors is None:
            errors = np.sqrt(1e-10)

        if x1 is None and x2 is None:
            x1 = self.x
            if errors is None:
                raise TypeError("Error should not be None when x2 is None")
            K = self.get_value(x1[:, None] - x1[None, :])
            if include_diagonal is None or include_diagonal:
                K[np.diag_indices_from(K)] += errors ** 2
            if (include_general is None or include_general) and len(self.A):
                K[np.diag_indices_from(K)] += self.A
                K += np.tril(np.dot(self.U.T, self.V), -1)
                K += np.triu(np.dot(self.V.T, self.U), 1)
            return K

        incl = False
        x1 = np.ascontiguousarray(x1, dtype=float)
        if x2 is None:
            x2 = x1
            incl = include_diagonal is not None and include_diagonal
        K = self.get_value(x1[:, None] - x2[None, :])
        if incl:
            K[np.diag_indices_from(K)] += errors ** 2
        return K


class CeleriteQuasiPerKernel(CeleriteKernel):
    """
    Class that computes and stores a matrix approximating the quasi-periodic
    kernel.

    See `radvel/example_planets/k2-131_celerite.py` for an example of a setup
    file that uses this Kernel object.

    See celerite.readthedocs.io and Foreman-Mackey et al. 2017. AJ, 154, 220
    (equation 56) for more details.

    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = B/(2+C) * exp( -|t_i - t_j| / L) * (\\cos(\\frac{ 2\\pi|t_i-t_j| }{ P_{rot} }) + (1+C) )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly four objects, 'gp_B*',
            'gp_C*', 'gp_L*', and 'gp_Prot*', where * is a suffix
            identifying these hyperparameters with a likelihood object.
    """

    @property
    def name(self):
        return "CeleriteQuasiPer"

    def __init__(self, hparams):

        self.hparams = {}
        for par in hparams:
            if par.startswith("gp_B"):
                self.hparams["gp_B"] = hparams[par]
            if par.startswith("gp_C"):
                self.hparams["gp_C"] = hparams[par]
            if par.startswith("gp_L"):
                self.hparams["gp_L"] = hparams[par]
            if par.startswith("gp_Prot"):
                self.hparams["gp_Prot"] = hparams[par]

        assert (
            len(self.hparams) == 4
        ), """
{}Kernel requires exactly 4 hyperparameters with names 'gp_B', 'gp_C', 'gp_L', and 'gp_Prot'.
        """.format(
            self.name
        )

        try:
            self.hparams["gp_Prot"].value
            self.hparams["gp_C"].value
            self.hparams["gp_B"].value
            self.hparams["gp_L"].value
        except KeyError:
            raise KeyError(
                """
{}Kernel requires hyperparameters 'gp_B*', 'gp_C*', 'gp_L', and 'gp_Prot*'.
                """.format(
                    self.name
                )
            )
        except AttributeError:
            raise AttributeError(
                "{}Kernel requires dictionary of radvel.Parameter objects as input.".format(
                    self.name
                )
            )

    # get arrays of real and complex parameters
    def compute_real_and_complex_hparams(self):

        self.real = np.zeros((1, 4))
        self.complex = np.zeros((1, 4))

        B = self.hparams["gp_B"].value
        C = self.hparams["gp_C"].value
        L = self.hparams["gp_L"].value
        Prot = self.hparams["gp_Prot"].value

        # Foreman-Mackey et al. (2017) eq 56
        self.real[0, 0] = B * (1 + C) / (2 + C)
        self.real[0, 2] = 1 / L
        self.complex[0, 0] = B / (2 + C)
        self.complex[0, 1] = 0.0
        self.complex[0, 2] = 1 / L
        self.complex[0, 3] = 2 * np.pi / Prot

    def __repr__(self):

        B = self.hparams["gp_B"].value
        C = self.hparams["gp_C"].value
        L = self.hparams["gp_L"].value
        Prot = self.hparams["gp_Prot"].value

        msg = ("{} Kernel with B = {}, C = {}, L = {}, Prot = {}.").format(
            self.name, B, C, L, Prot
        )
        return msg


class CeleriteMatern32Kernel(CeleriteKernel):
    """
    Class that computes and stores a matrix approximating the Matern 3/2
    kernel.

    See `radvel/example_planets/k2-131_celerite.py` for an example of a setup
    file that uses this Kernel object.

    See celerite.readthedocs.io and Foreman-Mackey et al. 2017. AJ, 154, 220
    (equation 30) for more details.

    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = S_0 \\omega_0 \\exp{\\left(-\\omega_0 |t_i - t_j|\\right)}
                    \\left[1 + \\omega_0 |t_i-t_j|\\right]

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly four objects, 'gp_sigma*',
            'gp_rho*', where * is a suffix identifying these hyperparameters
             with a likelihood object.
    """

    @property
    def name(self):
        return "CeleriteMatern32"

    def __init__(self, hparams, eps=0.01):

        self.hparams = {}
        for par in hparams:
            if par.startswith("gp_sigma"):
                self.hparams["gp_sigma"] = hparams[par]
            if par.startswith("gp_rho"):
                self.hparams["gp_rho"] = hparams[par]

        self.eps = eps

        assert (
            len(self.hparams) == 2
        ), """
{}Kernel requires exactly 2 hyperparameters with names 'gp_sigma' and 'sigma_rho'.
        """.format(
            self.name
        )

        try:
            self.hparams["gp_sigma"].value
            self.hparams["gp_rho"].value
        except KeyError:
            raise KeyError(
                """
{}Kernel requires hyperparameters 'gp_sigma*' and 'gp_rho*'
                """.format(
                    self.name
                )
            )
        except AttributeError:
            raise AttributeError(
                "{}Kernel requires dictionary of radvel.Parameter objects as input.".format(
                    self.name
                )
            )

    # get arrays of real and complex parameters
    def compute_real_and_complex_hparams(self):

        self.real = np.zeros((1, 4))
        self.complex = np.zeros((1, 4))

        sigma = self.hparams["gp_sigma"].value
        rho = self.hparams["gp_rho"].value

        w0 = np.sqrt(3.0) / rho
        S0 = sigma ** 2 / w0

        # Foreman-Mackey et al. (2017) eq 29, 30
        self.real[0, 0] = np.nan
        self.real[0, 2] = np.nan
        self.complex[0, 0] = w0 * S0
        self.complex[0, 1] = w0 * w0 * S0 / self.eps
        self.complex[0, 2] = w0
        self.complex[0, 3] = self.eps

    def __repr__(self):

        sigma = self.hparams["gp_sigma"].value
        rho = self.hparams["gp_rho"].value

        msg = ("{} Kernel with sigma = {}, rho = {}.").format(
            self.name, sigma, rho
        )
        return msg


class CeleriteSHOKernel(CeleriteKernel):
    """
    Class that computes and stores a matrix approximating a
    simple harmonic oscillator kernel.

    See `radvel/example_planets/k2-131_celerite.py` for an example of a setup
    file that uses a celerite kernel object.

    See celerite.readthedocs.io and Foreman-Mackey et al. 2017. AJ, 154, 220
    (equations 20 and 23) for more details.

    The PSD of the term is

    .. math::
        S(\\omega) = \\sqrt{\\frac{2}{\\pi}} \\frac{S_0\\,\\omega_0^4}
        {(\\omega^2-{\\omega_0}^2)^2 + {\\omega_0}^2\\,\\omega^2/Q^2}.

    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::
        \\begin{array}{ll}
            C_{ij} &=
                S_0 \\omega_0 Q
                \\exp{\\left(-\\frac{\\omega_0 -|t_i - t_j|}{2 Q}\\right)} \\\\
                &\\times \\left\\{
                    \\begin{array}{ll}
                         \\cosh{(\\eta \\omega_0 -|t_i - t_j|)}
                            + \\frac{1}{2 \\eta Q}
                              \\sinh{(\\eta \\omega_0 |t_i - t_j|)}, & 0<Q<1/2 \\\\
                         2 (1 + \\omega_0 |t_i - t_j|), & Q=1/2\\\\
                        \\cos{(\\eta \\omega_0 |t_i - t_j|)}
                            + \\frac{1}{2 \\eta Q}
                              \\sin{(\\eta \\omega_0 |t_i - t_j|)}, & Q>1/2\\\\
                    \\end{array}
                \\right.
        \\end{array}


    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly four objects, 'gp_S0*',
            'gp_Q*', and 'gp_w0*', where * is a suffix
            identifying these hyperparameters with a likelihood object.
    """

    @property
    def name(self):
        return "CeleriteSHO"

    def __init__(self, hparams):

        self.hparams = {}
        for par in hparams:
            if par.startswith("gp_S0"):
                self.hparams["gp_S0"] = hparams[par]
            if par.startswith("gp_Q"):
                self.hparams["gp_Q"] = hparams[par]
            if par.startswith("gp_w0"):
                self.hparams["gp_w0"] = hparams[par]

        assert (
            len(self.hparams) == 3
        ), """
{}Kernel requires exactly 3 hyperparameters with names 'gp_S0', 'gp_Q', and 'gp_w0'.
        """.format(
            self.name
        )

        try:
            self.hparams["gp_S0"].value
            self.hparams["gp_Q"].value
            self.hparams["gp_w0"].value
        except KeyError:
            raise KeyError(
                """
{}Kernel requires hyperparameters 'gp_S0*', 'gp_Q*', and 'gp_w0*'.
                """.format(
                    self.name
                )
            )
        except AttributeError:
            raise AttributeError(
                "{}Kernel requires dictionary of radvel.Parameter objects as input.".format(
                    self.name
                )
            )

    # get arrays of real and complex parameters
    def compute_real_and_complex_hparams(self):

        self.real = np.zeros((1, 4))
        self.complex = np.zeros((1, 4))

        S0 = self.hparams["gp_S0"].value
        Q = self.hparams["gp_Q"].value
        w0 = self.hparams["gp_w0"].value

        # Foreman-Mackey et al. (2017) eq 23
        # Also celerite source:
        # https://github.com/dfm/celerite/blob/main/celerite/terms.py#L464
        if Q >= 0.5:
            f = np.sqrt(4.0 * Q ** 2 - 1)
            self.real[0, 0] = np.nan
            self.real[0, 2] = np.nan
            self.complex[0, 0] = S0 * w0 * Q
            self.complex[0, 1] = S0 * w0 * Q / f
            self.complex[0, 2] = 0.5 * w0 / Q
            self.complex[0, 3] = 0.5 * w0 / Q * f

        else:
            f = np.sqrt(1.0 - 4.0 * Q ** 2)
            self.real[0, 0] = (
                0.5 * S0 * w0 * Q * np.array([1.0 + 1.0 / f, 1.0 - 1.0 / f])
            )
            self.real[0, 2] = 0.5 * w0 / Q * np.array([1.0 - f, 1.0 + f])
            self.complex[0, 0] = np.nan
            self.complex[0, 1] = np.nan
            self.complex[0, 2] = np.nan
            self.complex[0, 3] = np.nan

    def __repr__(self):

        S0 = self.hparams["gp_S0"].value
        Q = self.hparams["gp_Q"].value
        w0 = self.hparams["gp_w0"].value

        msg = ("{} Kernel with S0 = {}, Q = {}, w0 = {}.").format(self.name, S0, Q, w0)
        return msg


