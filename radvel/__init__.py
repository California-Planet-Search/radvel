from __future__ import absolute_import

# pre-import the big packages to avoid some warnings
import emcee     # producing ABC warning
import nbsphinx  # producing ABC warning

from .model import *
from .likelihood import *
from . import posterior
from .mcmc import *
from .prior import *
from .utils import *
from .report import *
from .fitting import *
from .plot import *

import warnings
warnings.filterwarnings("ignore")


def _custom_warningfmt(msg, *a, **b):
    return "WARNING:", str(msg) + '\n'


__all__ = ['model', 'likelihood', 'posterior', 'mcmc', 'prior', 'utils',
         'fitting', 'report', 'cli', 'driver', 'gp']

__version__ = '1.2.7'
__spec__ = __name__
__package__ = __path__[0]

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(sys.prefix, 'radvel_example_data')
if not os.path.isdir(DATADIR):
    warnings.warn("Could not find radvel_example_data directory in {}".format(sys.prefix),
                  ImportWarning)
    trydir = os.path.join(os.environ['HOME'], '.local', 'radvel_example_data')
    if os.path.isdir(trydir):
        warnings.warn("Found radvel_example_data in ~/.local", ImportWarning)
        DATADIR = trydir
    else:
        warnings.warn("Failed to locate radvel_example_data directory. Example setup files will not work.",
                      ImportWarning)

# tell python how to pickle methods and fucntions; necessary for running MCMC in multi-
#  threaded mode.


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


if sys.version_info[0] < 3:
    import copy_reg
    import types
    copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
