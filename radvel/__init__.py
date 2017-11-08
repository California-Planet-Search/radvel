from .model import *
from .likelihood import *
from . import posterior
from .mcmc import *
from .prior import *
from .utils import *
from .report import *
from .plotting import *
from .fitting import *
import os 
import sys

__all__=['model', 'likelihood', 'posterior', 'mcmc', 'prior', 'utils',
         'fitting', 'report', 'cli', 'driver']

__version__ = '1.0.1'

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(sys.prefix, 'radvel_example_data')

# tell python how to pickle methods and fucntions; necessary for running MCMC in multi-
#   threaded mode.
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
    copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
