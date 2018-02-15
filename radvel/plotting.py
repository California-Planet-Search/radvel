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

latex = {
    'ms': 'm s$^{\\mathregular{-1}}$',
    'BJDTDB': 'BJD$_{\\mathregular{TDB}}$'
}

telfmts_default = {
    'j': dict(color='k', fmt='o', mfc='none', label='HIRES', mew=1),
    'k': dict(color='k', fmt='s', mfc='none', label='HIRES pre 2004', mew=1),
    'a': dict(color='g', fmt='d', label='APF'),
    'pfs': dict(color='magenta', fmt='p', label='PFS'),
    'h': dict(color='firebrick', fmt="s", label='HARPS'),
    'harps-n': dict(color='firebrick', fmt='^', label='HARPS-N'),
    'l': dict(color='g', fmt='*'),
}

telfmts_default['lick'] = telfmts_default['l']
telfmts_default['hires_rj'] = telfmts_default['j']
telfmts_default['hires'] = telfmts_default['j']
telfmts_default['hires_rk'] = telfmts_default['k']
telfmts_default['apf'] = telfmts_default['a']
telfmts_default['harps'] = telfmts_default['h']

cmap = matplotlib.cm.nipy_spectral
rcParams['font.size'] = 9
rcParams['lines.markersize'] = 5
rcParams['axes.grid'] = False


def _mtelplot(x, y, e, tel, ax, telfmts={}):
    """Plot data from from multiple telescopes

    x (array): Either time or phase
    y (array): RV
    e (array): RV error
    tel (array): telecsope string key
     telfmts (dict): dictionary of dictionaries corresponding to kwargs 
        passed to errorbar. Example:

        telfmts = {
             'hires': dict(fmt='o',label='HIRES',msize=),
             'harps-n' dict(fmt='s',)}
    """
    lw = 1.0

    default_colors = ['orange', 'purple', 'magenta', 'pink', 'blue', 'grey', 'red']
    ci = 0
    
    utel = np.unique(tel)
    for t in utel:
        xt = x[tel == t]
        yt = y[tel == t]
        et = e[tel == t]

        # Default formatting
        kw = dict(
            fmt='o', capsize=0, mew=0, 
            ecolor='0.6', lw=lw, color=default_colors[ci],
            label=t
        )

        # If not explicit format set, look among default formats
        telfmt = {}
        if t not in telfmts and t in telfmts_default:
            telfmt = telfmts_default[t]
        if t in telfmts:
            telfmt = telfmts[t]
            print(telfmt)
        if t not in telfmts and t not in telfmts_default:
            ci += 1
        for k in telfmt:
            kw[k] = telfmt[k]
            
        pl.errorbar(xt, yt, yerr=et, **kw)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )


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
    figwidth = 7.5  # spans a page with 0.5in margins
    phasefac = 1.4
    ax_rv_height = figwidth * 0.6
    ax_phase_height = ax_rv_height / phasefac
    bin_fac = 1.75
    bin_markersize = bin_fac * rcParams['lines.markersize']
    bin_markeredgewidth = bin_fac * rcParams['lines.markeredgewidth']
    fit_linewidth = 2.0

    synthpost = copy.deepcopy(post)
    model = synthpost.likelihood.model
    synthparams = post.params.basis.to_synth(post.params)
    synthpost.params.update(synthparams)
    rvtimes = synthpost.likelihood.x
    rverr = synthpost.likelihood.errorbars()
    num_planets = model.num_planets

    if nophase:
        num_planets = 1
        periods = [max(rvtimes) - min(rvtimes)]

    if phase_ncols is None:
        phase_ncols = 1
    if phase_nrows is None:
        phase_nrows = num_planets

    ax_phase_height /= phase_ncols
        
    e = epoch
    if len(post.likelihood.x) < 20: 
        nobin = True
    
    if saveplot is not None:
        resolution = 10000
    else: 
        resolution = 2000

    if isinstance(synthpost.likelihood, radvel.likelihood.CompositeLikelihood):
        like_list = synthpost.likelihood.like_list
    else:
        like_list = [ synthpost.likelihood ]

    if not nophase:
        periods = []
        for i in range(num_planets):
            periods.append(synthparams['per%d' % (i+1)].value)
            
    longp = max(periods)

    dt = max(rvtimes) - min(rvtimes)
    rvmodt = np.linspace(
        min(rvtimes) - 0.05 * dt, max(rvtimes) + 0.05 * dt + longp,
        int(resolution)
    )
    
    rvmod2 = model(rvmodt)
    rvmod = model(rvtimes)

    if ((rvtimes - e) < -2.4e6).any():
        plttimes = rvtimes
        mplttimes = rvmodt
    elif e == 0:
        e = 2450000
        plttimes = rvtimes - e
        mplttimes = rvmodt - e
    else:
        plttimes = rvtimes - e
        mplttimes = rvmodt - e

    rawresid = synthpost.likelihood.residuals()
    resid = (
        rawresid + synthparams['dvdt'].value*(rvtimes-model.time_base)
        + synthparams['curv'].value*(rvtimes-model.time_base)**2
    )
    slope = (
        synthparams['dvdt'].value * (rvmodt-model.time_base)
        + synthparams['curv'].value * (rvmodt-model.time_base)**2
    )
    slope_low = (
        synthparams['dvdt'].value * (rvtimes-model.time_base)
        + synthparams['curv'].value * (rvtimes-model.time_base)**2
    )

    # Provision figure
    figheight = ax_rv_height + ax_phase_height * phase_nrows
    divide = 1 - ax_rv_height / figheight
    fig = pl.figure(figsize=(figwidth, figheight))
    fig.subplots_adjust(left=0.12, right=0.95)
    gs_rv = gridspec.GridSpec(1, 1)
    gs_rv.update(left=0.12, right=0.93, top=0.93,
                 bottom=divide+rv_phase_space*0.5)
    gs_phase = gridspec.GridSpec(phase_nrows, phase_ncols)

    if phase_ncols == 1:
        gs_phase.update(left=0.12, right=0.93,
                        top=divide - rv_phase_space * 0.5,
                        bottom=0.07, hspace=0.003)
    else:
        gs_phase.update(left=0.12, right=0.93,
                        top=divide - rv_phase_space * 0.5,
                        bottom=0.07, hspace=0.25, wspace=0.25)

    ax_list = []
    ax_rv = pl.subplot(gs_rv[0, 0])
    pltletter = ord('a')
    ax = ax_rv

    ax_list += [ax_rv]
   

    ax.axhline(0, color='0.5', linestyle='--')

    # Default formatting
    lw = 0.01
    ci = 0
    default_colors = ['orange', 'purple', 'magenta' , 'pink']

    numdatapoints = 0
    for like in like_list:
        if isinstance(like, radvel.likelihood.GPLikelihood):
            gp_mean, _ = like.predict(like.x)
            if not subtract_gp_mean_model:
                rvmod[numdatapoints:numdatapoints+len(like.x)] += gp_mean
        numdatapoints += len(like.x)

    for like in like_list:
        if isinstance(like, radvel.likelihood.GPLikelihood): 
            
            t = like.suffix

            kw = dict(
            fmt='o', capsize=0, mew=0, 
            ecolor='0.6', lw = lw, color=default_colors[ci],
            label = t
            )

            # If not explicit format set, look among default formats
            telfmt = {}
            if t not in telfmts and t in telfmts_default:
                telfmt = telfmts_default[t]
            if t in telfmts:
                telfmt = telfmts[t]
                print(telfmt)
            if t not in telfmts and t not in telfmts_default:
                ci += 1
            for k in telfmt:
                kw[k] = telfmt[k]

            xpred = np.linspace(np.min(like.x),np.max(like.x),num=int(3e3))
            gpmu, stddev = like.predict(xpred)


            if ((xpred - e) < -2.4e6).any():
                pass
            elif e == 0:
                e = 2450000
                xpred = xpred - e
            else:
                xpred = xpred - e

            orbit_model = like.model(xpred)
            if subtract_gp_mean_model:
                gpmu = 0.
            if subtract_orbit_model:
                orbit_model = 0.
            ax.fill_between(xpred, gpmu+orbit_model-stddev, gpmu+orbit_model+stddev, 
                            color=kw['color'], alpha=0.5, lw=0
                            )
            ax.plot(xpred, gpmu+orbit_model, 'b-', rasterized=False, lw=0.1)

        else:
            # Unphased plot
            orbit_model = rvmod2
            if subtract_orbit_model:
                orbit_model = 0.
            ax.plot(mplttimes,orbit_model,'b-', rasterized=False, lw=0.1)

            
    def labelfig(letter):
        text = "{})".format(chr(letter))
        add_anchored(
            text, loc=2, prop=dict(fontweight='bold', size='large'),
            frameon=False
        )

    labelfig(pltletter)

    pltletter += 1

    if subtract_orbit_model:
        _mtelplot(
            # data = residuals (best fit model subtracted out)
            plttimes, rawresid, rverr, synthpost.likelihood.telvec, ax, telfmts
        )
    else:
        _mtelplot(
            # data = residuals + best fit model
            plttimes, rawresid+rvmod, rverr, synthpost.likelihood.telvec, ax, telfmts
        )
    ax.set_xlim(min(plttimes)-0.01*dt, max(plttimes)+0.01*dt)
    
    pl.setp(ax_rv.get_xticklabels(), visible=False)

    # Legend
    if legend:
        pl.legend(numpoints=1, fontsize=legend_fontsize, loc='best')


    # Years on upper axis
    axyrs = ax_rv.twiny()
    # axyrs.set_xlim(min(plttimes)-0.01*dt,max(plttimes)+0.01*dt)

    xl = np.array(list(ax.get_xlim())) + e
    decimalyear = Time(xl, format='jd', scale='utc').decimalyear
    axyrs.plot(decimalyear, decimalyear)
    axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
    axyrs.set_xlim(*decimalyear)
    axyrs.set_xlabel('Year', fontweight='bold')
    # axyrs.xaxis.set_major_locator(MaxNLocator(8))

    if not yscale_auto: 
        scale = np.std(rawresid+rvmod)
        ax.set_ylim(-yscale_sigma * scale, yscale_sigma * scale)

    ax.set_ylabel('RV [{ms:}]'.format(**latex), weight='bold')
    ticks = ax.yaxis.get_majorticklocs()
    ax.yaxis.set_ticks(ticks[1:])

    divider = make_axes_locatable(ax_rv)
    ax_resid = divider.append_axes(
        "bottom", size="50%", pad=0.0, sharex=ax_rv, sharey=None
    )
    ax = ax_resid
    ax_list += [ax_resid]

    # Residuals
    ax.plot(mplttimes, slope, 'b-', lw=fit_linewidth)

    labelfig(pltletter)

    pltletter += 1

    _mtelplot(plttimes, resid, rverr, synthpost.likelihood.telvec, ax, telfmts)
    if not yscale_auto: 
        scale = np.std(resid)
        ax.set_ylim(-yscale_sigma * scale, yscale_sigma * scale)

    ax.set_xlim(min(plttimes)-0.01*dt, max(plttimes)+0.01*dt)
    ticks = ax.yaxis.get_majorticklocs()
    ax.yaxis.set_ticks([ticks[0], 0.0, ticks[-1]])
    pl.xlabel('JD - {:d}'.format(int(np.round(e))), weight='bold')
    ax.set_ylabel('Residuals', weight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='both'))
    
    # Define the locations for the axes
    axbounds = ax.get_position().bounds
    bottom = axbounds[1]
    height = (bottom - 0.15) / num_planets
    bottom -= height + 0.05

    # Phase plots
    for i in range(num_planets):
        if nophase:
            break
        
        pnum = i+1

        rvmod2 = model(rvmodt, planet_num=pnum) - slope
        modph = t_to_phase(synthpost.params, rvmodt, pnum, cat=True) - 1
        rvdat = rawresid + model(rvtimes, planet_num=pnum) - slope_low
        phase = t_to_phase(synthpost.params, rvtimes, pnum, cat=True) - 1
        rvdatcat = np.concatenate((rvdat, rvdat))
        rverrcat = np.concatenate((rverr, rverr))
        rvmod2cat = np.concatenate((rvmod2, rvmod2))
        bint, bindat, binerr = fastbin(phase+1, rvdatcat, nbins=25)
        bint -= 1.0

        i_row = int(i / phase_ncols)
        i_col = int(i - i_row * phase_ncols)
        ax = pl.subplot(gs_phase[i_row, i_col])
        ax_list += [ax]

        ax.axhline(0, color='0.5', linestyle='--', )
        ax.plot(sorted(modph), rvmod2cat[np.argsort(modph)], 'b-', linewidth=fit_linewidth)
        labelfig(pltletter)

        pltletter += 1
        telcat = np.concatenate((synthpost.likelihood.telvec, synthpost.likelihood.telvec))

        _mtelplot(phase, rvdatcat, rverrcat, telcat, ax, telfmts)
        if not nobin and len(rvdat) > 10: 
            ax.errorbar(
                bint, bindat, yerr=binerr, fmt='ro', mec='w', ms=bin_markersize,
                mew=bin_markeredgewidth
            )

        if not phase_limits:
            pl.xlim(-0.5, 0.5)
        else:
            pl.xlim(phase_limits[0],phase_limits[1])

        if not yscale_auto: 
            scale = np.std(rvdatcat)
            pl.ylim(-yscale_sigma*scale, yscale_sigma*scale)
        
        keys = [p+str(pnum) for p in ['per', 'k', 'e']]
        labels = [synthpost.params.tex_labels().get(k, k) for k in keys]
        if i < num_planets-1:
            ticks = ax.yaxis.get_majorticklocs()
            ax.yaxis.set_ticks(ticks[1:-1])

        pl.ylabel('RV [{ms:}]'.format(**latex), weight='bold')
        pl.xlabel('Phase', weight='bold')

        print_params = ['per', 'k', 'e']
        units = {'per': 'days', 'k': latex['ms'], 'e': ''}

        anotext = []
        for l, p in enumerate(print_params):
            val = synthparams["%s%d" % (print_params[l], pnum)].value
            
            if uparams is None:
                _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])
            else:
                if hasattr(post, 'medparams'):
                    val = post.medparams["%s%d" % (print_params[l], pnum)]
                else:
                    print("WARNING: medparams attribute not found in " +
                          "posterior object will annotate with " +
                          "max-likelihood values and reported uncertainties " +
                          "may not be appropriate.")
                err = uparams["%s%d" % (print_params[l], pnum)]
                if err > 0:
                    val, err, errlow = radvel.utils.sigfig(val, err)
                    _anotext = '$\\mathregular{%s}$ = %s $\\mathregular{\\pm}$ %s %s' \
                               % (labels[l].replace("$", ""), val, err, units[p])
                else:
                    _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])

            anotext += [_anotext] 

        anotext = '\n'.join(anotext)
        add_anchored(
            anotext, loc=1, frameon=True, prop=dict(size='large', weight='bold'),
            bbox=dict(ec='none', fc='w', alpha=0.8)
        )

    if saveplot is not None:
        pl.savefig(saveplot, dpi=150)
        print("RV multi-panel plot saved to %s" % saveplot)
        
    return fig, ax_list

    
def corner_plot(post, chains, saveplot=None):
    """
    Make a corner plot from the output MCMC chains and a posterior object.

    Args:
        post (radvel.Posterior): Radvel posterior object
        chains (DataFrame): MCMC chains output by radvel.mcmc
        saveplot (str, optional):  Name of output file, will show as 
            interactive matplotlib window if not defined.

    Returns:
        None
    
    """
    labels = [k for k in post.params.keys() if post.params[k].vary]
    texlabels = [post.params.tex_labels().get(l, l) for l in labels]

    f = rcParams['font.size']
    rcParams['font.size'] = 12
    
    _ = corner.corner(
        chains[labels], labels=texlabels, label_kwargs={"fontsize": 14},
        plot_datapoints=False, bins=30, quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 14}, smooth=True
    )
    
    if saveplot is not None:
        pl.savefig(saveplot, dpi=150)
        print("Corner plot saved to %s" % saveplot)
    else:
        pl.show()

    rcParams['font.size'] = f


def texlabel(key, letter):
    if key.count('mpsini') == 1:
        return '$M_' + letter + '\\sin i$'
    if key.count('rhop') == 1:
        return '$\\rho_' + letter + '$'
    if key.count('a') == 1:
        return "$a_" + letter + "$"


def corner_plot_derived_pars(chains, planet, saveplot=None):
    """
    Make a corner plot from the output MCMC chains and a posterior object.

    Args:
        chains (DataFrame): MCMC chains output by radvel.mcmc
        planet (Planet object): Planet configuration object
        saveplot (Optional[string]: Name of output file, will show as 
            interactive matplotlib window if not defined.

    Returns:
        None
    
    """

    if 'planet_letters' in dir(planet):
        planet_letters = planet.planet_letters
    else:
        planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k'}
    
    # Determine which columns to include in corner plot
    labels = []
    texlabels = []
    for i in np.arange(1, planet.nplanets + 1, 1):
        letter = planet_letters[i]

        for key in 'mpsini rhop a'.split():
            label = '{}{}'.format(key, i)
            
            is_column = list(chains.columns).count(label) == 1
            if not is_column:
                break
            
            null_column = chains.isnull().any().loc[label]
            if null_column:
                break

            tl = texlabel(label, letter)

            # add units to label
            if key == 'mpsini':
                unit = "M$_{\\oplus}$"
                if np.median(chains[label]) > 100:
                    unit = "M$_{\\rm Jup}$"
                    chains[label] *= 0.00315
                if np.median(chains[label]) > 100:
                    unit = "M$_{\\odot}$"
                    chains[label] *= 0.000954265748

                tl += " [%s]" % unit
            elif key == 'rhop':
                tl += " [g cm$^{-3}$]"
            elif key == 'a':
                tl += " [AU]"
            else:
                tl += " "

            labels.append(label)
            texlabels.append(tl)

    f = rcParams['font.size']
    rcParams['font.size'] = 12
    _ = corner.corner(
        chains[labels], labels=texlabels, label_kwargs={"fontsize": 14}, 
        plot_datapoints=False, bins=30, quantiles=[0.16, 0.50, 0.84],
        show_titles=True, title_kwargs={"fontsize": 14}, smooth=True
    )
    
    if saveplot is not None:
        pl.savefig(saveplot, dpi=150)
        print("Corner plot saved to %s" % saveplot)
    else:
        pl.show()
    rcParams['font.size'] = f


def trend_plot(post, chains, nwalkers, outfile=None):
    """MCMC trend plot

    Make a trend plot to show the evolution of the MCMC as a function of step number.

    Args:
        post (radvel.Posterior): Radvel Posterior object
        chains (DataFrame): MCMC chains output by radvel.mcmc
        nwalkers (int): number of walkers used in this particular MCMC run
        outfile (string): name of output multi-page PDF file

    Returns:
        None
        
    """

    labels = sorted([k for k in post.params.keys() if post.params[k].vary])
    texlabels = [post.params.tex_labels().get(l, l) for l in labels]
    colors = [cmap(x) for x in np.linspace(0.05, 0.95, nwalkers)]

    with PdfPages(outfile) as pdf:
        for param, tex in zip(labels, texlabels):
            flatchain = chains[param].values
            wchain = flatchain.reshape((nwalkers, -1))

            _ = pl.figure(figsize=(18, 10))
            for w in range(nwalkers):
                pl.plot(
                    wchain[w, :], '.', rasterized=True, color=colors[w],
                    markersize=3
                )

            pl.xlim(0, wchain.shape[1])

            pl.xlabel('Step Number')
            pl.ylabel(tex)

            ax = pl.gca()
            ax.set_rasterized(True)

            pdf.savefig()
            pl.close()


def correlation_plot(post, outfile=None):
    """Correlation plot

    Plot parameter correlations.

    Args:
        post (radvel.Posterior): Radvel Posterior object
        outfile (string): name of output multi-page PDF file

    Returns:
        None
        
    """

    pltind = 1
    pl.subplot(431)
    pl.subplots_adjust(top=0.97, left=0.07, right=0.95,
                       bottom=0.10, hspace=0.22, wspace=0.22)
    
    for like in post.likelihood.like_list:
        resid = like.residuals()
        for parname in like.decorr_params:
            var = parname.split('_')[1]
            pars = []
            for par in like.decorr_params:
                if var in par:
                    pars.append(like.params[par])
            pars.append(0.0)
            if np.isfinite(like.decorr_vectors[var]).all():
                vec = like.decorr_vectors[var]
                vec -= np.mean(vec)
                p = np.poly1d(pars)
                print(var, pars)
                
                pl.subplot('33%d' % pltind)
                pl.plot(vec, p(vec), 'b-', lw=3)
                pl.plot(vec, resid + p(vec), 'ko')

                pl.xlabel("$\\Delta$ %s" % '_'.join(parname.split('_')[1:]))
                pl.ylabel('RV [m s$^{-1}$]')
                
                pltind += 1

    if outfile is None:
        pl.show()
    else:
        pl.savefig(outfile)


def add_anchored(*args, **kwargs):
    """
    Parameters
    ----------
    s : string
        Text.

    loc : str
        Location code.

    pad : float, optional
        Pad between the text and the frame as fraction of the font
        size.

    borderpad : float, optional
        Pad between the frame and the axes (or *bbox_to_anchor*).

    prop : `matplotlib.font_manager.FontProperties`
        Font properties.
    """

    bbox = {}
    if 'bbox' in kwargs:
        bbox = kwargs.pop('bbox')
    at = AnchoredText(*args, **kwargs)
    if len(bbox.keys()) > 0:
        pl.setp(at.patch, **bbox)

    ax = pl.gca()
    ax.add_artist(at)
