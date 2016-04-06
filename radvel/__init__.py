__all__=['model','likelihood','posterior','mcmc','prior']
from .model import *
from .likelihood import *
from .posterior import *
from .mcmc import *
from .prior import *
import os 
import sys

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(MODULEDIR,'../data')
PLANETDIR = os.path.join(MODULEDIR,'../planets')

sys.path.append(PLANETDIR)
