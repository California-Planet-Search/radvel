import time
import curses
import sys
import os

import multiprocessing as mp

import pandas as pd
import numpy as np

import emcee
import pymc3 as pm
import h5py

from radvel import utils
import radvel

#where the model will be set up... will MAP and MCMC have to be run in here?
