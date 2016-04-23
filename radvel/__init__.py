__all__=['model','likelihood','posterior','mcmc','prior', 'utils']
from .model import *
from .likelihood import *
from .posterior import *
from .mcmc import *
from .prior import *
from .utils import *
import os 
import sys

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(MODULEDIR,'../example_data')
