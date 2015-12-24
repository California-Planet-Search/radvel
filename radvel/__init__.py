__all__=['model','likelihood','posterior','mcmc','prior']
from .model import *
from .likelihood import *
from .posterior import *
from .mcmc import *
from .prior import *
import os 

MODULEDIR, filename = os.path.split(__file__)
DATADIR = os.path.join(MODULEDIR,'data')
