__all__=['model','likelihood','posterior','mcmc','prior', 'utils', 'fitting', 'report', 'cli', 'driver']
from .model import *
from .likelihood import *
from . import posterior
from mcmc import *
from .prior import *
from .utils import *
from .report import *
from .plotting import *
from .fitting import *
import os 
import sys

__version__ = '0.9.4'

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(sys.prefix,'radvel_example_data')
