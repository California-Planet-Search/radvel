import numpy as np
from matplotlib import rcParams, gridspec
from matplotlib import pyplot as pl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time

import radvel
from radvel import plot
from radvel.utils import t_to_phase, fastbin, sigfig
"""
- multipanel
"""

class MultipanelPlot(object):
    """
    Class to handle the creation of RV multipanel plots.

    Args:
        post (radvel.Posterior): radvel.Posterior object. The model
            plotted will be generated from ``post.params``
        epoch (int [optional]): epoch to subtract off of all time measurements
        yscale_auto (bool [optional]): Use matplotlib auto y-axis
             scaling (default: False)
        yscale_sigma (float [optional]): Scale y-axis limits for all panels to be +/-
             yscale_sigma*(RMS of data plotted) if yscale_auto==False
        uparams (dict [optional]): parameter uncertainties, must
           contain 'per', 'k', and 'e' keys.
    """
    def __init__(self, post, saveplot=None, epoch=2450000, yscale_auto=False, yscale_sigma=3.0,
                phase_nrows=None, phase_ncols=None, uparams=None, rv_phase_space=0.08, telfmts={},legend=True,
                phase_limits=[], nobin=False, phasetext_size='large'):

        self.post = post
        self.saveplot=saveplot
        self.epoch = epoch
        self.yscale_auto = yscale_auto
        self.yscale_sigma = yscale_sigma
        if phase_ncols is None:
            self.phase_ncols = 1
        if phase_nrows is None:
            self.phase_nrows = self.post.likelihood.model.num_planets
        self.uparams=uparams
        self.rv_phase_space=rv_phase_space
        self.telfmts=telfmts
        self.legend=legend
        self.phase_limits=phase_limits
        self.nobin=nobin
        self.phasetext_size=phasetext_size

        if isinstance(self.post.likelihood, radvel.likelihood.CompositeLikelihood):
            self.like_list = self.post.likelihood.like_list
        else:
            self.like_list = [ self.post.likelihood ] 

        is_gp=False
        for like in self.like_list:
            if isinstance(like, radvel.likelihood.GPLikelihood):
                is_gp=True
                break
            else:
                pass
        assert not is_gp, "This class is unable to plot results of GP fits. Use radvel.plot.GPMultipanelPlot instead."

    def set_configs(self):

        # numbers for figure provisioning
        self.figwidth = 7.5  # spans a page with 0.5in margins

        phasefac = 1.4
        self.ax_rv_height = self.figwidth * 0.6
        self.ax_phase_height = self.ax_rv_height / phasefac

        bin_fac = 1.75
        self.bin_markersize = bin_fac * rcParams['lines.markersize']
        self.bin_markeredgewidth = bin_fac * rcParams['lines.markeredgewidth']
        self.fit_linewidth = 2.0


        # convert params to synth basis
        synthparams = self.post.params.basis.to_synth(self.post.params)
        self.post.params.update(synthparams)

        self.model = self.post.likelihood.model
        self.rvtimes = self.post.likelihood.x
        self.rverr = self.post.likelihood.errorbars()
        self.num_planets = self.model.num_planets
   

        self.rawresid = self.post.likelihood.residuals()

        self.resid = (
            self.rawresid + self.post.params['dvdt'].value*(self.rvtimes-self.model.time_base)
            + self.post.params['curv'].value*(self.rvtimes-self.model.time_base)**2
        )



        if self.saveplot is not None:
            resolution = 10000
        else: 
            resolution = 2000

        periods = []
        for i in range(self.num_planets):
            periods.append(synthparams['per%d' % (i+1)].value)            
        longp = max(periods)


        self.dt = max(self.rvtimes) - min(self.rvtimes)
        self.rvmodt = np.linspace(
            min(self.rvtimes) - 0.05 * self.dt, max(self.rvtimes) + 0.05 * self.dt + longp,
            int(resolution)
        )
        
        self.orbit_model = self.model(self.rvmodt)
        self.rvmod = self.model(self.rvtimes)

        if ((self.rvtimes - self.epoch) < -2.4e6).any():
            self.plttimes = self.rvtimes
            self.mplttimes = self.rvmodt
        elif self.epoch == 0:
            self.epoch = 2450000
            self.plttimes = self.rvtimes - self.epoch
            self.mplttimes = self.rvmodt - self.epoch
        else:
           self.plttimes = self.rvtimes - self.epoch
           self.mplttimes = self.rvmodt - self.epoch


        self.slope = (
            self.post.params['dvdt'].value * (self.rvmodt-self.model.time_base)
            + self.post.params['curv'].value * (self.rvmodt-self.model.time_base)**2
        )
        self.slope_low = (
            self.post.params['dvdt'].value * (self.rvtimes-self.model.time_base)
            + self.post.params['curv'].value * (self.rvtimes-self.model.time_base)**2
        )

        # list for Axes objects
        self.ax_list = []


    def plot_timeseries(self):
        """
        Args:
            overplot (Bool): if True, axis 
        """

        ax = pl.gca()

        ax.axhline(0, color='0.5', linestyle='--')

        # plot orbit model
        ax.plot(self.mplttimes,self.orbit_model,'b-', rasterized=False, lw=0.1)

        # plot data
        plot.mtelplot(
            # data = residuals + model
            self.plttimes, self.rawresid+self.rvmod, self.rverr, self.post.likelihood.telvec, ax, self.telfmts
        )

        ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)    
        pl.setp(ax.get_xticklabels(), visible=False)

        # legend
        if self.legend:
            ax.legend(numpoints=1, loc='best')

        # years on upper axis
        axyrs = ax.twiny()
        xl = np.array(list(ax.get_xlim())) + self.epoch
        decimalyear = Time(xl, format='jd', scale='utc').decimalyear
        axyrs.plot(decimalyear, decimalyear)
        axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
        axyrs.set_xlim(*decimalyear)
        axyrs.set_xlabel('Year', fontweight='bold')

        if not self.yscale_auto: 
            scale = np.std(self.rawresid+self.rvmod)
            ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)


        ax.set_ylabel('RV [{ms:}]'.format(**plot.latex), weight='bold')
        ticks = ax.yaxis.get_majorticklocs()
        ax.yaxis.set_ticks(ticks[1:])




    def plot_residuals(self):

        
        ax = pl.gca()

        ax.plot(self.mplttimes, self.slope, 'b-', lw=self.fit_linewidth)

        plot.mtelplot(self.plttimes, self.resid, self.rverr, self.post.likelihood.telvec, ax, self.telfmts)
        if not self.yscale_auto: 
            scale = np.std(self.resid)
            ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)

        ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)
        ticks = ax.yaxis.get_majorticklocs()
        ax.yaxis.set_ticks([ticks[0], 0.0, ticks[-1]])
        pl.xlabel('JD - {:d}'.format(int(np.round(self.epoch))), weight='bold')
        ax.set_ylabel('Residuals', weight='bold')
        ax.yaxis.set_major_locator(MaxNLocator(5, prune='both'))
        

    def plot_phasefold(self, gs_phase, first_pltletter):

        if len(self.post.likelihood.x) < 20: 
            self.nobin = True

        pltletter = first_pltletter
        for i in range(self.num_planets):
            pnum = i+1

            rvmod2 = self.model(self.rvmodt, planet_num=pnum) - self.slope
            modph = t_to_phase(self.post.params, self.rvmodt, pnum, cat=True) - 1
            rvdat = self.rawresid + self.model(self.rvtimes, planet_num=pnum) - self.slope_low
            phase = t_to_phase(self.post.params, self.rvtimes, pnum, cat=True) - 1
            rvdatcat = np.concatenate((rvdat, rvdat))
            rverrcat = np.concatenate((self.rverr, self.rverr))
            rvmod2cat = np.concatenate((rvmod2, rvmod2))
            bint, bindat, binerr = fastbin(phase+1, rvdatcat, nbins=25)
            bint -= 1.0

            i_row = int(i / self.phase_ncols)
            i_col = int(i - i_row * self.phase_ncols)
            ax = pl.subplot(gs_phase[i_row, i_col])
            self.ax_list += [ax]

            ax.axhline(0, color='0.5', linestyle='--', )
            ax.plot(sorted(modph), rvmod2cat[np.argsort(modph)], 'b-', linewidth=self.fit_linewidth)
            plot.labelfig(pltletter)

            pltletter += 1
            telcat = np.concatenate((self.post.likelihood.telvec, self.post.likelihood.telvec))

            plot.mtelplot(phase, rvdatcat, rverrcat, telcat, ax, self.telfmts)
            if not self.nobin and len(rvdat) > 10: 
                ax.errorbar(
                    bint, bindat, yerr=binerr, fmt='ro', mec='w', ms=self.bin_markersize,
                    mew=self.bin_markeredgewidth
                )

            if self.phase_limits:
            	ax.set_xlim(self.phase_limits[0],self.phase_limits[1])
            else:
            	ax.set_xlim(-0.5, 0.5)
                

            if not self.yscale_auto: 
                scale = np.std(rvdatcat)
                ax.set_ylim(-self.yscale_sigma*scale, self.yscale_sigma*scale)
            
            keys = [p+str(pnum) for p in ['per', 'k', 'e']]
            labels = [self.post.params.tex_labels().get(k, k) for k in keys]
            if i < self.num_planets-1:
                ticks = ax.yaxis.get_majorticklocs()
                ax.yaxis.set_ticks(ticks[1:-1])

            ax.set_ylabel('RV [{ms:}]'.format(**plot.latex), weight='bold')
            ax.set_xlabel('Phase', weight='bold')

            print_params = ['per', 'k', 'e']
            units = {'per': 'days', 'k': plot.latex['ms'], 'e': ''}

            anotext = []
            for l, p in enumerate(print_params):
                val = self.post.params["%s%d" % (print_params[l], pnum)].value
                
                if self.uparams is None:
                    _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])
                else:
                    if hasattr(self.post, 'medparams'):
                        val = self.post.medparams["%s%d" % (print_params[l], pnum)]
                    else:
                        print("WARNING: medparams attribute not found in " +
                              "posterior object will annotate with " +
                              "max-likelihood values and reported uncertainties " +
                              "may not be appropriate.")
                    err = self.uparams["%s%d" % (print_params[l], pnum)]
                    if err > 0:
                        val, err, errlow = sigfig(val, err)
                        _anotext = '$\\mathregular{%s}$ = %s $\\mathregular{\\pm}$ %s %s' \
                                   % (labels[l].replace("$", ""), val, err, units[p])
                    else:
                        _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])

                anotext += [_anotext] 

            anotext = '\n'.join(anotext)
            plot.add_anchored(
                anotext, loc=1, frameon=True, prop=dict(size=self.phasetext_size, weight='bold'),
                bbox=dict(ec='none', fc='w', alpha=0.8)
            )
   
    
    def plot_multipanel(self, nophase=False):
        """
        Args:
            saveplot (str [optional]): save figure as this file name
        Returns:
            tuple containing:
                - current matplotlib figure object
                - list of axis objects
        """

        self.set_configs()

        if nophase:
        	scalefactor = 1
        else:
        	scalefactor = self.phase_nrows

        figheight = self.ax_rv_height + self.ax_phase_height * scalefactor


        # provision figure
        fig = pl.figure(figsize=(self.figwidth, figheight))
        
        fig.subplots_adjust(left=0.12, right=0.95)
        gs_rv = gridspec.GridSpec(1, 1)

        divide = 1 - self.ax_rv_height / figheight
        gs_rv.update(left=0.12, right=0.93, top=0.93,
                     bottom=divide+self.rv_phase_space*0.5)


        # orbit plot
        ax_rv = pl.subplot(gs_rv[0, 0])
        self.ax_list += [ax_rv]

        pl.sca(ax_rv)
        self.plot_timeseries()
        pltletter = ord('a')
        plot.labelfig(pltletter)
        pltletter += 1


        # residuals
        divider = make_axes_locatable(ax_rv)
        ax_resid = divider.append_axes(
            "bottom", size="50%", pad=0.0, sharex=ax_rv, sharey=None
        )
        self.ax_list += [ax_resid]


        pl.sca(ax_resid)
        self.plot_residuals()
        plot.labelfig(pltletter)
        pltletter += 1


        # phase-folded plots
        if not nophase:
	        gs_phase = gridspec.GridSpec(self.phase_nrows, self.phase_ncols)

	        if self.phase_ncols == 1:
	            gs_phase.update(left=0.12, right=0.93,
	                            top=divide - self.rv_phase_space * 0.5,
	                            bottom=0.07, hspace=0.003)
	        else:
	            gs_phase.update(left=0.12, right=0.93,
	                            top=divide - self.rv_phase_space * 0.5,
	                            bottom=0.07, hspace=0.25, wspace=0.25)

	        self.plot_phasefold(gs_phase, pltletter)


        if self.saveplot is not None:
            pl.savefig(self.saveplot, dpi=150)
            print("RV multi-panel plot saved to %s" % self.saveplot)

        return fig, self.ax_list



class GPMultipanelPlot(MultipanelPlot):
    """
    options for subtracting data, subtracting mean GP model, plotting each inst separately 

    subtract_gp_mean_model (bool, optional): if True, subtract the Gaussian
        process mean max likelihood model from the data and the
        model when plotting the results. 
    plot_likelihoods_separately (bool, optional): if True, plot a separate
        panel for each Likelihood object.
    subtract_orbit_model (bool, optional): if True, subtract the best-fit
        orbit model from the data and the model when plotting 
        the results. Useful for seeing the structure of correlated
        noise in the data.

    """
    def __init__(self, post, saveplot=None, epoch=2450000, yscale_auto=False, yscale_sigma=3.0,
                phase_nrows=None, phase_ncols=None, uparams=None, rv_phase_space=0.08, telfmts={},
                legend=True,
                phase_limits=[], nobin=False, phasetext_size='large', subtract_gp_mean_model=False,
                plot_likelihoods_separately=True, subtract_orbit_model=False):

        self.post = post
        self.saveplot=saveplot
        self.epoch = epoch
        self.yscale_auto = yscale_auto
        self.yscale_sigma = yscale_sigma
        if phase_ncols is None:
            self.phase_ncols = 1
        if phase_nrows is None:
            self.phase_nrows = self.post.likelihood.model.num_planets
        self.uparams=uparams
        self.rv_phase_space=rv_phase_space
        self.telfmts=telfmts
        self.legend=legend
        self.phase_limits=phase_limits
        self.nobin=nobin
        self.phasetext_size=phasetext_size
        self.subtract_gp_mean_model=subtract_gp_mean_model
        self.plot_likelihoods_separately=plot_likelihoods_separately
        self.subtract_orbit_model=subtract_orbit_model

        if isinstance(self.post.likelihood, radvel.likelihood.CompositeLikelihood):
            self.like_list = self.post.likelihood.like_list
        else:
            self.like_list = [ self.post.likelihood ] 

        is_gp=False
        for like in self.like_list:
            if isinstance(like, radvel.likelihood.GPLikelihood):
                is_gp=True
                break
            else:
                pass
        assert is_gp, "This class requires at least one GPLikelihood object in the Posterior."

    def plot_timeseries(self):

        ax = pl.gca()

        ax.axhline(0, color='0.5', linestyle='--')

        if self.subtract_orbit_model:
            orbit_model4data = np.zeros(self.rvmod.shape)
        else:
            orbit_model4data = self.rvmod

        ci = 0
        for like in self.like_list:
            if isinstance(like, radvel.likelihood.GPLikelihood):

                xpred = np.linspace(np.min(like.x),np.max(like.x),num=int(3e3))
                gpmu, stddev = like.predict(xpred)
                if self.subtract_orbit_model:
                    gp_orbit_model = np.zeros(xpred.shape)
                else:
                    gp_orbit_model = self.model(xpred)

                if ((xpred - self.epoch) < -2.4e6).any():
                    pass
                elif self.epoch == 0:
                    self.epoch = 2450000
                    xpred = xpred - self.epoch
                else:
                    xpred = xpred - self.epoch

                if self.subtract_gp_mean_model:
                    gpmu = 0.
                else:
                    gp_mean4data, _ = like.predict(like.x)
                    indx = np.where(self.post.likelihood.telvec == like.suffix)
                    orbit_model4data[indx] += gp_mean4data

                if like.suffix not in self.telfmts and like.suffix in plot.telfmts_default:
                    color = plot.telfmts_default[like.suffix]['color']
                if like.suffix in self.telfmts:
                    color = self.telfmts[like.suffix]['color']
                if like.suffix not in self.telfmts and like.suffix not in plot.telfmts_default:
                    color = plot.default_colors[ci]
                    ci += 1

                ax.fill_between(xpred, gpmu+gp_orbit_model-stddev, gpmu+gp_orbit_model+stddev, 
                                color=color, alpha=0.5, lw=0
                                )
                ax.plot(xpred, gpmu+gp_orbit_model, 'b-', rasterized=False, lw=0.1)

            else:
                # plot orbit model
                ax.plot(self.mplttimes,self.orbit_model,'b-', rasterized=False, lw=0.1)


        # plot data
        plot.mtelplot(
            # data = residuals + model
            self.plttimes, self.rawresid+orbit_model4data, self.rverr, self.post.likelihood.telvec, ax, self.telfmts
        )

        ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)    
        pl.setp(ax.get_xticklabels(), visible=False)

        # legend
        if self.legend:
            ax.legend(numpoints=1, loc='best')

        # years on upper axis
        axyrs = ax.twiny()
        xl = np.array(list(ax.get_xlim())) + self.epoch
        decimalyear = Time(xl, format='jd', scale='utc').decimalyear
        axyrs.plot(decimalyear, decimalyear)
        axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
        axyrs.set_xlim(*decimalyear)
        axyrs.set_xlabel('Year', fontweight='bold')

        if not self.yscale_auto: 
            scale = np.std(self.rawresid+self.rvmod)
            ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)


        ax.set_ylabel('RV [{ms:}]'.format(**plot.latex), weight='bold')
        ticks = ax.yaxis.get_majorticklocs()
        ax.yaxis.set_ticks(ticks[1:])


    def plot_multipanel(self, nophase=False):

        if not self.plot_likelihoods_separately:
            super(GPMultipanelPlot, self).plot_multipanel()



        else:
            self.set_configs()

            if nophase:
                scalefactor = 1
            else:
                scalefactor = self.phase_nrows

            figheight = self.ax_rv_height*(len(self.like_list)+0.5) + self.ax_phase_height * scalefactor


            # provision figure
            fig = pl.figure(figsize=(self.figwidth, figheight))
            
            fig.subplots_adjust(left=0.12, right=0.95)
            gs_rv = gridspec.GridSpec(len(self.like_list), 1)

            divide = 1 - self.ax_rv_height*len(self.like_list) / figheight
            gs_rv.update(left=0.12, right=0.93, top=0.93,
                         bottom=divide+self.rv_phase_space*0.5, hspace=0.0)

            # TODO: add gs_resids; set up figure correctly to account for residuals ahead of time rather than appending them
            # TODO: add functionality to plot likes separately


            # orbit plot for each likelihood
            pltletter = ord('a')

            i = 0
            for like in self.like_list:
                ax_rv = pl.subplot(gs_rv[i, 0])
                i += 1
                self.ax_list += [ax_rv]
                pl.sca(ax_rv)

                self.plot_timeseries()
                plot.labelfig(pltletter)
                pltletter += 1  

            # residuals
            divider = make_axes_locatable(ax_rv)
            ax_resid = divider.append_axes(
                "bottom", size="50%", pad=0.0, sharex=ax_rv, sharey=None
            )
            self.ax_list += [ax_resid]


            pl.sca(ax_resid)
            self.plot_residuals()
            plot.labelfig(pltletter)
            pltletter += 1


            # phase-folded plots
            if not nophase:
                gs_phase = gridspec.GridSpec(self.phase_nrows, self.phase_ncols)

                if self.phase_ncols == 1:
                    gs_phase.update(left=0.12, right=0.93,
                                    top=divide - self.rv_phase_space * 0.5,
                                    bottom=0.07, hspace=0.003)
                else:
                    gs_phase.update(left=0.12, right=0.93,
                                    top=divide - self.rv_phase_space * 0.5,
                                    bottom=0.07, hspace=0.25, wspace=0.25)

                self.plot_phasefold(gs_phase, pltletter)


            if self.saveplot is not None:
                pl.savefig(self.saveplot, dpi=150)
                print("RV multi-panel plot saved to %s" % self.saveplot)

            return fig, self.ax_list




