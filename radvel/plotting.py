import string
import copy
import numpy as np
import pylab as pl
import matplotlib
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib import rcParams, gridspec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.time import Time
import corner

import radvel
from radvel.utils import t_to_phase, fastbin


   
# break into
# full timeseries
# residuals
# phasefold
# move into the report.py
""" 

   
    
plotter = Plotter(post)
plotter.plot_phasefold('b') # draw that plot in the current axis
plotter.plot_residuals() # draw that plot in the current axis
"""

def rv_multipanel_plot(post, saveplot=None, telfmts={}, nobin=False, 
                       yscale_auto=False, yscale_sigma=3.0, nophase=False, 
                       epoch=2450000, uparams=None, phase_ncols=None, 
                       phase_nrows=None, legend=True, legend_fontsize='x-small',
                       rv_phase_space=0.08, phase_limits=[],
                       subtract_gp_mean_model=False, 
                       plot_likelihoods_separately=True,
                       subtract_orbit_model=False):

    """Multi-panel RV plot to display model using post.params orbital
    parameters.

    Args:
        post (radvel.Posterior): Radvel posterior object. The model
            plotted will be generated from post.params
        saveplot (string, optional): Name of output file, will show as
             interactive matplotlib window if not defined.
        nobin (bool, optional): If True do not show binned data on
             phase plots. Will default to True if total number of
             measurements is less then 20.
        yscale_auto (bool, optional): Use matplotlib auto y-axis
             scaling (default: False)
        yscale_sigma (float, optional): Scale y-axis limits to be +/-
             yscale_sigma*(RMS of data plotted) if yscale_auto==False
        telfmts (dict, optional): dictionary of dictionaries mapping
             instrument code to plotting format code.
        nophase (bool, optional): Will omit phase-folded plots if true
        epoch (float, optional): Subtract this value from the time axis for
            more compact axis labels (default: 245000)
        uparams (dict, optional): parameter uncertainties, must
           contain 'per', 'k', and 'e' keys.
        phase_ncols (int, optional): number of columns in the phase
            folded plots. Default behavior is 1.
        phase_nrows (int, optional): number of columns in the phase
            folded plots. Default is nplanets.
        legend (bool, optional): include legend on plot? (default: True)
        legend_fontsize (str, optional): fontsize parameter to be passed
            to matplotlib.legend. Choose from {'xx-small', 'x-small', 
            'small', 'medium', 'large', 'x-large', 'xx-large'}. 
            (default: 'x-small')
        rv_phase_space (float, optional): verticle space between rv
            plot and phase-folded plots (in units of fraction of
            figure height)
        phase_limits (list, optional): two element list specifying 
            pyplot.xlim bounds for phase-folded array. Useful for
            partial orbits.
        subtract_gp_mean_model (bool, optional): if True, subtract the Gaussian
            process mean max likelihood model from the data and the
            model when plotting the results. 
        plot_likelihoods_separately (bool, optional): if True, plot a separate
            panel for each Likelihood object.
        subtract_orbit_model (bool, optional): if True, subtract the best-fit
            orbit model from the data and the model when plotting 
            the results. Useful for seeing the structure of correlated
            noise in the data.

    Returns:
        figure: current matplotlib figure object
        list: list of axis objects
    """
    
